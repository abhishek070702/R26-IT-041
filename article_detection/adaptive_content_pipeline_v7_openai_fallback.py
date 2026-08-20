"""
Adaptive Local Content Summarization and Category Detection Pipeline
==================================================================

Research-oriented, fully local pipeline for English printed documents.
No cloud vision, OCR, speech, or summarization API is required.

Main flow
---------
1. Receive document type from the document-identification module.
2. Preprocess and deskew the captured page.
3. Run Tesseract once and retain OCR geometry.
4. For Newspaper/Magazine pages:
   - detect page separators locally with OpenCV,
   - find adaptive headline anchors,
   - combine multiline headlines,
   - form article regions from headline, separator, and whitespace geometry,
   - reconstruct body columns in reading order,
   - merge/remove duplicate article regions.
5. For single-category documents, process the full page.
6. Predict categories with the local TF-IDF/Logistic Regression model.
7. Ask the user for category and Summary/Full Text.
8. Use local Ollama llama3.2:3b only after selection.
9. Return a structured payload for the next module.

The detector is layout-adaptive. It is intended for diverse English newspaper
layouts, but no heuristic method can guarantee perfect segmentation for every
possible publication. It therefore exposes confidence/warnings and a safe
full-page fallback.
"""

from __future__ import annotations

import json
import math
import os
import queue
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytesseract
import requests
from pytesseract import Output


# ============================================================
# 1. CONFIGURATION AND DATA CLASSES
# ============================================================

PROJECT_CATEGORIES = [
    "Sports",
    "Politics",
    "Social",
    "Science",
    "Mathematics",
    "Story",
    "General Text",
]

MULTI_CATEGORY_TYPES = {"Newspaper", "Magazine"}


@dataclass
class PipelineConfig:
    # Files / local services
    category_model_path: str = "category_model(1).pkl"
    tesseract_cmd: Optional[str] = None
    ollama_url: str = "http://127.0.0.1:11434/api/generate"
    ollama_tags_url: str = "http://127.0.0.1:11434/api/tags"
    ollama_model: str = field(
        default_factory=lambda: os.getenv("LLAMA_MODEL", "llama3.2:3b")
    )
    ollama_timeout_seconds: int = 240

    # OCR
    target_width: int = 1800
    min_ocr_confidence: float = 18.0
    tesseract_psm: int = 3

    # Article construction
    min_article_words: int = 30
    max_article_words: int = 2200
    headline_min_height_ratio: float = 1.30
    headline_score_threshold: float = 2.4
    duplicate_title_similarity: float = 0.72

    # Category decision
    category_min_probability: float = 0.30
    category_min_margin: float = 0.08
    title_repeat_for_category: int = 3

    # Output / diagnostics
    summary_max_words: int = 100
    debug: bool = True


@dataclass(frozen=True)
class OCRLine:
    uid: str
    text: str
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    word_count: int
    block_num: int = 0
    par_num: int = 0
    line_num: int = 0

    @property
    def width(self) -> int:
        return max(1, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(1, self.y2 - self.y1)

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0


@dataclass
class Separator:
    orientation: str  # "vertical" or "horizontal"
    x1: int
    y1: int
    x2: int
    y2: int
    strength: float

    @property
    def x(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def y(self) -> float:
        return (self.y1 + self.y2) / 2.0


@dataclass
class HeadlineAnchor:
    lines: list[OCRLine]
    title: str
    bbox: tuple[int, int, int, int]
    score: float


@dataclass
class ArticleRegion:
    article_number: int
    title: str
    bbox: tuple[int, int, int, int]
    normalized_bbox: tuple[int, int, int, int]
    full_text: str
    category: str
    confidence: float
    probability_margin: float
    top_predictions: list[dict[str, Any]]
    ocr_quality: float
    layout_confidence: float
    warnings: list[str]
    source: str = "adaptive_headline_separator_geometry"


# ============================================================
# 2. SETUP AND DOCUMENT TYPE
# ============================================================


def configure_tesseract(config: PipelineConfig) -> None:
    if config.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = config.tesseract_cmd
        return

    env_cmd = os.getenv("TESSERACT_CMD")
    if env_cmd:
        pytesseract.pytesseract.tesseract_cmd = env_cmd
        return

    windows_default = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    if windows_default.exists():
        pytesseract.pytesseract.tesseract_cmd = str(windows_default)



def load_category_model(model_path: str):
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Category model not found: {path.resolve()}\n"
            "Set PipelineConfig(category_model_path='category_model(1).pkl')."
        )

    model = joblib.load(path)
    if not hasattr(model, "predict"):
        raise TypeError("The loaded category model does not provide predict().")
    return model



def normalize_document_type(document_type: Any) -> str:
    value = "" if document_type is None else str(document_type).strip().lower()

    if "news" in value:
        return "Newspaper"
    if "magazine" in value:
        return "Magazine"
    if any(token in value for token in ["novel", "storybook", "story book", "story"]):
        return "Novel/Story"
    if any(token in value for token in ["textbook", "text book", "school book"]):
        return "Textbook"
    if "letter" in value:
        return "Printed Letter"
    if "report" in value:
        return "Report"
    return "General Document"



def is_multi_category_document(document_type: Any) -> bool:
    return normalize_document_type(document_type) in MULTI_CATEGORY_TYPES


# ============================================================
# 3. TEXT HELPERS
# ============================================================


def clean_text_basic(text: Any) -> str:
    if text is None:
        return ""

    value = str(text).replace("\r", "\n")
    value = value.replace("“", '"').replace("”", '"')
    value = value.replace("‘", "'").replace("’", "'")
    value = value.replace("–", "-").replace("—", "-")
    value = re.sub(r"[^A-Za-z0-9.,!?;'\"\n:+\-*/()%=$& ]", " ", value)
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()



def alpha_ratio(text: str) -> float:
    chars = [character for character in text if not character.isspace()]
    if not chars:
        return 0.0
    return sum(character.isalpha() for character in chars) / len(chars)



def uppercase_ratio(text: str) -> float:
    letters = [character for character in text if character.isalpha()]
    if not letters:
        return 0.0
    return sum(character.isupper() for character in letters) / len(letters)



def title_case_ratio(text: str) -> float:
    words = re.findall(r"[A-Za-z]+", text)
    if not words:
        return 0.0
    titled = [word for word in words if len(word) > 1 and word[0].isupper()]
    return len(titled) / len(words)



def looks_like_byline(text: str) -> bool:
    clean = clean_text_basic(text)
    words = re.findall(r"[A-Za-z]+", clean)
    if not 1 <= len(words) <= 6:
        return False

    common_prefixes = (
        "by ",
        "reporting by ",
        "compiled by ",
        "pix by ",
        "photo by ",
    )
    if clean.lower().startswith(common_prefixes):
        return True
    if clean.lower().endswith(" and") and 2 <= len(words) <= 6:
        return True

    letters = [character for character in clean if character.isalpha()]
    return bool(letters) and uppercase_ratio(clean) >= 0.88 and len(words) <= 5



def is_masthead_or_metadata(text: str) -> bool:
    low = clean_text_basic(text).lower().strip()
    patterns = [
        r"\bregistered as a newspaper\b",
        r"\bnational newspaper\b.*\bsince\s+\d{4}\b",
        r"\bvolume\s*[-:]?\s*\d+\b",
        r"\bvol\.?\s*no\b",
        r"\blate city\b",
        r"\b\d+\s*pages?\b",
        r"\bwww\.",
        r"\.com\b",
        r"\.lk\b",
        r"\bfacebook\b",
        r"\binstagram\b",
        r"\btwitter\b",
        r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday),?\s+"
        r"(?:january|february|march|april|may|june|july|august|september|october|november|december)"
        r"\s+\d{1,2},?\s+\d{4}\b",
    ]
    return any(re.search(pattern, low) for pattern in patterns)


def is_noise_line(text: str, document_type: str = "General Document") -> bool:
    line = clean_text_basic(text)
    if not line:
        return True

    words = re.findall(r"[A-Za-z0-9]+", line)
    if not words:
        return True
    if len(words) == 1 and len(words[0]) <= 2:
        return True
    if alpha_ratio(line) < 0.24 and not re.search(r"[=+\-*/%$]", line):
        return True
    if any(len(word) >= 30 for word in words):
        return True
    if re.search(r"(.)\1{6,}", line):
        return True

    low = line.lower().strip()
    patterns = [
        r"^(?:to|ito|hito|mito|nto|bto)?\s*page\s*[o0-9]+$",
        r"^page\s*[o0-9]+$",
        r"^\d+\s*pages?$",
        r"^advertisement$",
        r"^classifieds?$",
        r"^figure$",
        r"^caption$",
        r"^photo$",
        r"^\d+[a-z]$",
        r"^[|;:=._-]+$",
    ]
    if any(re.search(pattern, low) for pattern in patterns):
        return True

    if document_type in MULTI_CATEGORY_TYPES and is_masthead_or_metadata(line):
        return True

    return False



def clean_ocr_lines(lines: Iterable[str], document_type: str) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()

    for raw in lines:
        line = re.sub(r"\s+", " ", clean_text_basic(raw)).strip()
        if is_noise_line(line, document_type):
            continue
        key = re.sub(r"\W+", "", line.lower())
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(line)

    return output



def join_hyphenated_lines(lines: Sequence[str]) -> list[str]:
    output: list[str] = []
    index = 0
    while index < len(lines):
        current = lines[index].strip()
        if (
            index + 1 < len(lines)
            and re.search(r"[A-Za-z]-$", current)
            and re.match(r"^[a-z]", lines[index + 1].strip())
        ):
            next_line = lines[index + 1].strip()
            current = current[:-1] + next_line
            index += 2
        else:
            index += 1
        output.append(current)
    return output



def ocr_quality_score(text: str) -> float:
    words = re.findall(r"[A-Za-z0-9]+", text)
    if not words:
        return 0.0

    readable = [word for word in words if re.fullmatch(r"[A-Za-z]{3,}", word)]
    strange = [
        word
        for word in words
        if len(word) >= 24 or re.search(r"(.)\1{4,}", word)
    ]
    score = len(readable) / len(words) - 0.45 * len(strange) / len(words)
    return round(max(0.0, min(1.0, score)), 3)


# ============================================================
# 4. IMAGE PREPROCESSING AND OCR
# ============================================================


def resize_for_ocr(image: np.ndarray, target_width: int) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    if width >= target_width:
        return image.copy(), 1.0

    scale = target_width / width
    resized = cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC,
    )
    return resized, scale



def deskew_image(image: np.ndarray, max_angle: float = 7.0) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )[1]
    coordinates = np.column_stack(np.where(binary > 0))
    if len(coordinates) < 150:
        return image

    angle = cv2.minAreaRect(coordinates)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    if abs(angle) < 0.15 or abs(angle) > max_angle:
        return image

    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width // 2, height // 2), angle, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )



def prepare_page_image(
    image: np.ndarray,
    config: PipelineConfig,
) -> tuple[np.ndarray, np.ndarray, float]:
    resized, scale = resize_for_ocr(image, config.target_width)
    deskewed = deskew_image(resized)
    gray = cv2.cvtColor(deskewed, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, 7, 7, 21)
    sharpened = cv2.addWeighted(
        denoised,
        1.40,
        cv2.GaussianBlur(denoised, (0, 0), 2.2),
        -0.40,
        0,
    )
    return deskewed, sharpened, scale



def extract_ocr_lines(
    prepared_gray: np.ndarray,
    document_type: str,
    config: PipelineConfig,
) -> list[OCRLine]:
    data = pytesseract.image_to_data(
        prepared_gray,
        config=(
            f"--oem 3 --psm {config.tesseract_psm} "
            "-c preserve_interword_spaces=1"
        ),
        output_type=Output.DATAFRAME,
    )

    if data is None or data.empty:
        return []

    data = data.dropna(subset=["text"]).copy()
    data["text"] = data["text"].astype(str).str.strip()
    data = data[data["text"] != ""]
    data["conf"] = pd.to_numeric(data["conf"], errors="coerce").fillna(-1)
    data = data[data["conf"] >= config.min_ocr_confidence]

    lines: list[OCRLine] = []
    grouping = ["block_num", "par_num", "line_num"]

    for (block_num, par_num, line_num), group in data.groupby(grouping, sort=False):
        group = group.sort_values("left")
        text = clean_text_basic(" ".join(group["text"].tolist()))
        if not text:
            continue

        x1 = int(group["left"].min())
        y1 = int(group["top"].min())
        x2 = int((group["left"] + group["width"]).max())
        y2 = int((group["top"] + group["height"]).max())
        if x2 <= x1 or y2 <= y1:
            continue

        uid = f"{int(block_num)}-{int(par_num)}-{int(line_num)}-{x1}-{y1}"
        lines.append(
            OCRLine(
                uid=uid,
                text=text,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                confidence=float(group["conf"].mean()),
                word_count=len(text.split()),
                block_num=int(block_num),
                par_num=int(par_num),
                line_num=int(line_num),
            )
        )

    return sorted(lines, key=lambda line: (line.y1, line.x1))



def best_full_page_ocr(prepared_gray: np.ndarray, document_type: str) -> str:
    best_text = ""
    best_score = -1.0

    for psm in (3, 4, 6, 11):
        raw = pytesseract.image_to_string(
            prepared_gray,
            config=f"--oem 3 --psm {psm}",
        )
        cleaned = clean_ocr_lines(raw.splitlines(), document_type)
        cleaned = join_hyphenated_lines(cleaned)
        text = "\n".join(cleaned).strip()
        score = len(text.split()) + 80.0 * ocr_quality_score(text)
        if score > best_score:
            best_score = score
            best_text = text

    return best_text


# ============================================================
# 5. GEOMETRY HELPERS
# ============================================================


def bbox_from_lines(lines: Sequence[OCRLine]) -> tuple[int, int, int, int]:
    if not lines:
        return (0, 0, 0, 0)
    return (
        min(line.x1 for line in lines),
        min(line.y1 for line in lines),
        max(line.x2 for line in lines),
        max(line.y2 for line in lines),
    )



def bbox_area(box: tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = box
    return max(1, x2 - x1) * max(1, y2 - y1)



def bbox_iou(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = bbox_area(first) + bbox_area(second) - intersection
    return intersection / max(1, union)



def bbox_containment(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    return intersection / max(1, min(bbox_area(first), bbox_area(second)))



def horizontal_overlap_boxes(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> float:
    overlap = max(0, min(first[2], second[2]) - max(first[0], second[0]))
    return overlap / max(1, min(first[2] - first[0], second[2] - second[0]))



def vertical_overlap_boxes(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> float:
    overlap = max(0, min(first[3], second[3]) - max(first[1], second[1]))
    return overlap / max(1, min(first[3] - first[1], second[3] - second[1]))



def normalize_bbox(
    bbox: tuple[int, int, int, int],
    image_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    height, width = image_shape
    x1, y1, x2, y2 = bbox
    return (
        int(1000 * x1 / max(1, width)),
        int(1000 * y1 / max(1, height)),
        int(1000 * x2 / max(1, width)),
        int(1000 * y2 / max(1, height)),
    )



def line_inside_box(
    line: OCRLine,
    box: tuple[int, int, int, int],
    margin: int = 0,
) -> bool:
    x1, y1, x2, y2 = box
    return (
        x1 - margin <= line.cx <= x2 + margin
        and y1 - margin <= line.cy <= y2 + margin
    )


# ============================================================
# 6. LOCAL SEPARATOR DETECTION
# ============================================================


def merge_nearby_separators(
    separators: list[Separator],
    orientation: str,
    coordinate_tolerance: int,
) -> list[Separator]:
    if not separators:
        return []

    key = (lambda separator: separator.x) if orientation == "vertical" else (lambda separator: separator.y)
    ordered = sorted(separators, key=key)
    groups: list[list[Separator]] = [[ordered[0]]]

    for separator in ordered[1:]:
        if abs(key(separator) - np.mean([key(item) for item in groups[-1]])) <= coordinate_tolerance:
            groups[-1].append(separator)
        else:
            groups.append([separator])

    merged: list[Separator] = []
    for group in groups:
        merged.append(
            Separator(
                orientation=orientation,
                x1=min(item.x1 for item in group),
                y1=min(item.y1 for item in group),
                x2=max(item.x2 for item in group),
                y2=max(item.y2 for item in group),
                strength=max(item.strength for item in group),
            )
        )
    return merged



def detect_page_separators(prepared_gray: np.ndarray) -> list[Separator]:
    page_h, page_w = prepared_gray.shape[:2]
    binary = cv2.threshold(
        prepared_gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )[1]

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(45, int(page_w * 0.055)), 1),
    )
    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, max(55, int(page_h * 0.045))),
    )

    horizontal_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
    vertical_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)

    separators: list[Separator] = []

    for orientation, mask in (("horizontal", horizontal_mask), ("vertical", vertical_mask)):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)

            if orientation == "horizontal":
                if width < 0.12 * page_w or height > 0.025 * page_h:
                    continue
                strength = width / page_w
            else:
                if height < 0.10 * page_h or width > 0.025 * page_w:
                    continue
                strength = height / page_h

            separators.append(
                Separator(
                    orientation=orientation,
                    x1=x,
                    y1=y,
                    x2=x + width,
                    y2=y + height,
                    strength=float(strength),
                )
            )

    vertical = merge_nearby_separators(
        [separator for separator in separators if separator.orientation == "vertical"],
        "vertical",
        max(8, int(page_w * 0.008)),
    )
    horizontal = merge_nearby_separators(
        [separator for separator in separators if separator.orientation == "horizontal"],
        "horizontal",
        max(8, int(page_h * 0.006)),
    )
    return vertical + horizontal


# ============================================================
# 7. ADAPTIVE CONTENT AREA AND HEADLINE ANCHORS
# ============================================================


def estimate_body_height(lines: Sequence[OCRLine], page_shape: tuple[int, int]) -> float:
    page_h, page_w = page_shape
    candidates = [
        line.height
        for line in lines
        if 3 <= line.word_count <= 18
        and line.confidence >= 25
        and line.width <= 0.55 * page_w
        and 0.05 * page_h <= line.cy <= 0.96 * page_h
    ]
    if not candidates:
        candidates = [line.height for line in lines]
    if not candidates:
        return 12.0

    values = np.asarray(candidates, dtype=float)
    lower, upper = np.percentile(values, [15, 65])
    trimmed = values[(values >= lower) & (values <= upper)]
    return float(np.median(trimmed if len(trimmed) else values))



def nearest_gap_above(line: OCRLine, lines: Sequence[OCRLine], body_height: float) -> float:
    previous_bottoms = [
        candidate.y2
        for candidate in lines
        if candidate.y2 <= line.y1
        and horizontal_overlap_boxes(
            (line.x1, line.y1, line.x2, line.y2),
            (candidate.x1, candidate.y1, candidate.x2, candidate.y2),
        ) >= 0.15
        and line.y1 - candidate.y2 <= 8 * body_height
    ]
    if not previous_bottoms:
        return 3.0 * body_height
    return max(0.0, line.y1 - max(previous_bottoms))



def generic_masthead_shape(
    line: OCRLine,
    body_height: float,
    page_shape: tuple[int, int],
) -> bool:
    page_h, page_w = page_shape
    return (
        line.y1 < 0.115 * page_h
        and line.height >= 2.8 * body_height
        and line.width >= 0.30 * page_w
        and line.word_count <= 7
        and not re.search(r"[:$%0-9]", line.text)
    )



def headline_line_score(
    line: OCRLine,
    all_lines: Sequence[OCRLine],
    body_height: float,
    page_shape: tuple[int, int],
    config: PipelineConfig,
) -> float:
    page_h, page_w = page_shape
    text = clean_text_basic(line.text)
    words = re.findall(r"[A-Za-z0-9]+", text)

    if not 1 <= len(words) <= 22:
        return -10.0
    if is_noise_line(text, "Newspaper"):
        return -10.0
    if looks_like_byline(text):
        return -10.0
    if generic_masthead_shape(line, body_height, page_shape):
        return -10.0
    if line.confidence < config.min_ocr_confidence:
        return -10.0

    ratio = line.height / max(1.0, body_height)
    if ratio < config.headline_min_height_ratio:
        return -10.0
    if len(words) == 1 and (len(words[0]) < 5 or ratio < 1.80):
        return -10.0

    gap_above = nearest_gap_above(line, all_lines, body_height)

    # Body lines can be slightly enlarged in newspaper sidebars. Reject long
    # moderate-size lines unless they have very strong headline typography.
    if ratio < 2.10 and len(words) > 9:
        return -10.0
    if ratio < 1.80 and len(words) > 7:
        return -10.0
    if ratio < 2.20 and title_case_ratio(text) < 0.40 and uppercase_ratio(text) < 0.50:
        return -10.0
    score = 0.0

    if ratio >= 2.4:
        score += 3.0
    elif ratio >= 1.8:
        score += 2.3
    elif ratio >= 1.5:
        score += 1.6
    else:
        score += 0.9

    title_ratio = title_case_ratio(text)
    upper_ratio = uppercase_ratio(text)
    if title_ratio >= 0.55:
        score += 0.9
    elif title_ratio >= 0.35:
        score += 0.45
    if upper_ratio >= 0.68:
        score += 0.65

    if gap_above >= 0.75 * body_height:
        score += 0.75
    elif gap_above >= 0.35 * body_height:
        score += 0.35

    if line.width >= 0.10 * page_w:
        score += 0.35
    if line.width >= 0.28 * page_w:
        score += 0.25

    if text[0].isupper() or text[0].isdigit() or text.startswith(("US$", "$")):
        score += 0.35
    elif ratio >= 1.80:
        # A wrapped second/third headline line may begin with a lowercase word.
        score -= 0.35
    else:
        score -= 1.2

    if text.endswith((".", ",", ";")):
        score -= 0.55
    else:
        score += 0.25

    # Short uppercase lines close to body size are usually author bylines.
    if upper_ratio >= 0.75 and len(words) <= 6 and ratio < 1.85:
        score -= 2.0

    # Very high lines are often publication metadata unless unusually news-like.
    if line.y1 < 0.08 * page_h and ratio < 2.0:
        score -= 0.8

    return score



def detect_adaptive_content_area(
    lines: Sequence[OCRLine],
    separators: Sequence[Separator],
    page_shape: tuple[int, int],
    body_height: float,
    config: PipelineConfig,
) -> tuple[int, int]:
    page_h, page_w = page_shape

    scored = []
    for line in lines:
        score = headline_line_score(line, lines, body_height, page_shape, config)
        if score >= config.headline_score_threshold:
            scored.append((line, score))

    content_start = 0
    early_candidates = [
        line
        for line, _ in scored
        if line.y1 <= 0.36 * page_h
        and not generic_masthead_shape(line, body_height, page_shape)
    ]
    if early_candidates:
        first = min(early_candidates, key=lambda line: line.y1)
        content_start = max(0, int(first.y1 - 0.7 * body_height))

    # A long horizontal separator above the first story is a useful generic header boundary.
    header_rules = [
        separator
        for separator in separators
        if separator.orientation == "horizontal"
        and 0.025 * page_h <= separator.y <= 0.26 * page_h
        and (separator.x2 - separator.x1) >= 0.45 * page_w
    ]
    if header_rules:
        candidates = [separator for separator in header_rules if separator.y <= content_start + 2 * body_height]
        if candidates:
            content_start = max(content_start, int(max(candidates, key=lambda item: item.y).y + body_height * 0.25))

    # Avoid deleting a large part of an inside page if no reliable heading was found.
    content_start = min(content_start, int(0.30 * page_h))

    content_end = page_h
    footer_rules = [
        separator
        for separator in separators
        if separator.orientation == "horizontal"
        and separator.y >= 0.92 * page_h
        and (separator.x2 - separator.x1) >= 0.45 * page_w
    ]
    if footer_rules:
        content_end = int(min(separator.y for separator in footer_rules))

    return content_start, content_end



def detect_headline_candidates(
    lines: Sequence[OCRLine],
    body_height: float,
    page_shape: tuple[int, int],
    content_area: tuple[int, int],
    config: PipelineConfig,
) -> list[tuple[OCRLine, float]]:
    """Return primary headline seeds. Lowercase wrapped lines are added later."""
    start_y, end_y = content_area
    candidates: list[tuple[OCRLine, float]] = []

    for line in lines:
        if not start_y <= line.cy <= end_y:
            continue
        score = headline_line_score(line, lines, body_height, page_shape, config)
        if score < config.headline_score_threshold:
            continue

        text = clean_text_basic(line.text)
        ratio = line.height / max(1.0, body_height)
        starts_like_headline = bool(text) and (
            text[0].isupper()
            or text[0].isdigit()
            or text.startswith(("US$", "$"))
            or ratio >= 2.80
        )
        if not starts_like_headline:
            continue

        candidates.append((line, score))

    return sorted(candidates, key=lambda item: (item[0].y1, item[0].x1))


def combine_multiline_headlines(
    candidates: Sequence[tuple[OCRLine, float]],
    all_lines: Sequence[OCRLine],
    body_height: float,
    page_shape: tuple[int, int],
) -> list[HeadlineAnchor]:
    """Build complete multiline titles from primary seeds and nearby large lines."""
    _, page_w = page_shape
    remaining_seeds = list(candidates)
    used_ids: set[str] = set()
    anchors: list[HeadlineAnchor] = []

    while remaining_seeds:
        seed_line, seed_score = remaining_seeds.pop(0)
        if seed_line.uid in used_ids:
            continue

        group = [seed_line]
        scores = [seed_score]
        used_ids.add(seed_line.uid)
        changed = True

        while changed:
            changed = False
            group_box = bbox_from_lines(group)
            group_bottom = group_box[3]
            group_center = (group_box[0] + group_box[2]) / 2

            possible = []
            for line in all_lines:
                if line.uid in used_ids:
                    continue
                if line.y1 < group_box[1] - 0.5 * body_height:
                    continue

                vertical_gap = line.y1 - group_bottom
                words = re.findall(r"[A-Za-z0-9]+", line.text)
                ratio = line.height / max(1.0, body_height)
                is_primary_line = any(candidate.uid == line.uid for candidate, _ in candidates)
                positive_gap_limit = 4.0 * body_height if (is_primary_line and ratio >= 2.20) else 1.35 * body_height
                if not (-1.75 * body_height <= vertical_gap <= positive_gap_limit):
                    continue

                group_max_ratio = max(item.height / max(1.0, body_height) for item in group)
                if group_max_ratio >= 4.0:
                    span_limit = 22.0 * body_height
                elif group_max_ratio >= 2.8:
                    span_limit = 15.0 * body_height
                else:
                    span_limit = 11.5 * body_height
                if line.y2 - group_box[1] > span_limit:
                    continue

                if not words or ratio < 1.45:
                    continue
                if len(words) > 10:
                    continue
                if len(words) > 8 and ratio < 2.40:
                    continue
                if line.text and line.text[0].islower() and ratio < 2.20 and len(words) >= 4:
                    continue
                if looks_like_byline(line.text) or is_noise_line(line.text, "Newspaper"):
                    continue
                if line.text.endswith((".", ",", ";")) and ratio < 2.30:
                    continue

                line_box = (line.x1, line.y1, line.x2, line.y2)
                overlap = horizontal_overlap_boxes(group_box, line_box)
                center_distance = abs(line.cx - group_center)
                if overlap < 0.30 and center_distance > 0.085 * page_w:
                    continue

                possible.append((abs(vertical_gap), -ratio, line))

            if possible:
                _, _, chosen = min(possible, key=lambda item: (item[0], item[1]))
                group.append(chosen)
                used_ids.add(chosen.uid)
                chosen_score = next(
                    (score for candidate, score in candidates if candidate.uid == chosen.uid),
                    0.75 * seed_score,
                )
                scores.append(chosen_score)
                changed = True

        group = sorted(group, key=lambda line: (line.y1, line.x1))
        title = clean_text_basic(" ".join(line.text for line in group))
        title = re.sub(r"\s+", " ", title).strip()
        words = title.split()
        if len(words) > 30:
            title = " ".join(words[:30])

        anchors.append(
            HeadlineAnchor(
                lines=group,
                title=title or "Untitled article",
                bbox=bbox_from_lines(group),
                score=float(np.mean(scores)),
            )
        )

    unique: list[HeadlineAnchor] = []
    for anchor in sorted(anchors, key=lambda item: (item.bbox[1], item.bbox[0])):
        duplicate = False
        for kept in unique:
            if (
                title_similarity(anchor.title, kept.title) >= 0.86
                and bbox_containment(anchor.bbox, kept.bbox) >= 0.70
            ):
                duplicate = True
                break
        if not duplicate:
            unique.append(anchor)

    return unique


def merge_short_headline_continuations(
    anchors: Sequence[HeadlineAnchor],
    body_height: float,
) -> list[HeadlineAnchor]:
    """Merge a short orphan line into the nearest earlier headline in the same column."""
    ordered = sorted(anchors, key=lambda item: (item.bbox[1], item.bbox[0]))
    output: list[HeadlineAnchor] = []

    for anchor in ordered:
        merged = False
        if len(anchor.title.split()) <= 4:
            for index in range(len(output) - 1, -1, -1):
                previous = output[index]
                gap = anchor.bbox[1] - previous.bbox[3]
                if gap < 0:
                    continue
                if gap > 4.5 * body_height:
                    # Earlier output items are even farther away vertically.
                    break
                overlap = horizontal_overlap_boxes(previous.bbox, anchor.bbox)
                previous_open = previous.title.rstrip().endswith((",", ":", "-"))
                if previous_open and overlap >= 0.45:
                    combined_lines = sorted(
                        previous.lines + anchor.lines,
                        key=lambda line: (line.y1, line.x1),
                    )
                    output[index] = HeadlineAnchor(
                        lines=combined_lines,
                        title=clean_text_basic(previous.title + " " + anchor.title),
                        bbox=bbox_from_lines(combined_lines),
                        score=(previous.score + anchor.score) / 2.0,
                    )
                    merged = True
                    break
        if not merged:
            output.append(anchor)

    return sorted(output, key=lambda item: (item.bbox[1], item.bbox[0]))


# ============================================================
# 8. ARTICLE REGION CONSTRUCTION
# ============================================================


def title_tokens(text: str) -> set[str]:
    stop = {"the", "a", "an", "of", "to", "and", "in", "for", "on", "with", "our"}
    return {
        word
        for word in re.findall(r"[a-z0-9]+", text.lower())
        if len(word) >= 3 and word not in stop
    }



def title_similarity(first: str, second: str) -> float:
    first_tokens = title_tokens(first)
    second_tokens = title_tokens(second)
    if not first_tokens or not second_tokens:
        return 0.0
    return len(first_tokens & second_tokens) / len(first_tokens | second_tokens)



def separator_overlaps_vertical_band(
    separator: Separator,
    top: int,
    bottom: int,
    minimum_ratio: float = 0.20,
) -> bool:
    overlap = max(0, min(separator.y2, bottom) - max(separator.y1, top))
    return overlap / max(1, bottom - top) >= minimum_ratio



def separator_overlaps_horizontal_band(
    separator: Separator,
    left: int,
    right: int,
    minimum_ratio: float = 0.25,
) -> bool:
    overlap = max(0, min(separator.x2, right) - max(separator.x1, left))
    return overlap / max(1, right - left) >= minimum_ratio



def find_article_horizontal_bounds(
    anchor: HeadlineAnchor,
    lines: Sequence[OCRLine],
    separators: Sequence[Separator],
    page_shape: tuple[int, int],
    provisional_bottom: int,
) -> tuple[int, int]:
    page_h, page_w = page_shape
    title_left, title_top, title_right, title_bottom = anchor.bbox
    title_center = (title_left + title_right) / 2
    title_width = title_right - title_left

    vertical_separators = [
        separator
        for separator in separators
        if separator.orientation == "vertical"
        and separator_overlaps_vertical_band(
            separator,
            title_top,
            provisional_bottom,
            minimum_ratio=0.18,
        )
        and not (title_left + 0.05 * title_width < separator.x < title_right - 0.05 * title_width)
    ]

    left_candidates = [separator.x for separator in vertical_separators if separator.x < title_center]
    right_candidates = [separator.x for separator in vertical_separators if separator.x > title_center]

    left = int(max(left_candidates)) if left_candidates else max(0, int(title_left - 0.035 * page_w))
    right = int(min(right_candidates)) if right_candidates else min(page_w, int(title_right + 0.035 * page_w))

    # Wide headlines normally define a multi-column article span themselves.
    if title_width >= 0.30 * page_w:
        left = max(0, int(title_left - 0.018 * page_w))
        right = min(page_w, int(title_right + 0.018 * page_w))

    # Iteratively expand to nearby body lines, but do not cross strong separators.
    for _ in range(2):
        nearby = [
            line
            for line in lines
            if title_bottom <= line.cy <= provisional_bottom
            and left - 0.015 * page_w <= line.cx <= right + 0.015 * page_w
            and not is_noise_line(line.text, "Newspaper")
        ]
        if nearby:
            candidate_left = min([left] + [line.x1 for line in nearby])
            candidate_right = max([right] + [line.x2 for line in nearby])

            crossing_left = [
                separator.x
                for separator in vertical_separators
                if candidate_left < separator.x < title_center
            ]
            crossing_right = [
                separator.x
                for separator in vertical_separators
                if title_center < separator.x < candidate_right
            ]
            left = int(max(crossing_left)) if crossing_left else max(0, candidate_left)
            right = int(min(crossing_right)) if crossing_right else min(page_w, candidate_right)

    if right - left < max(80, 0.07 * page_w):
        padding = int(0.04 * page_w)
        left = max(0, title_left - padding)
        right = min(page_w, title_right + padding)

    return left, right



def find_next_article_boundary(
    current: HeadlineAnchor,
    anchors: Sequence[HeadlineAnchor],
    horizontal_bounds: tuple[int, int],
    separators: Sequence[Separator],
    content_end: int,
    body_height: float,
) -> int:
    left, right = horizontal_bounds
    current_bottom = current.bbox[3]
    current_box = (left, current.bbox[1], right, current.bbox[3])

    next_headlines = []
    for anchor in anchors:
        if anchor is current or anchor.bbox[1] <= current_bottom + 0.25 * body_height:
            continue
        anchor_center = (anchor.bbox[0] + anchor.bbox[2]) / 2
        overlap = horizontal_overlap_boxes(current_box, anchor.bbox)
        if left <= anchor_center <= right or overlap >= 0.55:
            next_headlines.append(anchor.bbox[1])

    headline_boundary = min(next_headlines) if next_headlines else content_end

    # Headline anchors are more reliable than decorative/internal rules.
    # Many newspaper image articles contain horizontal borders above captions,
    # so using every rule as an article end would cut the body prematurely.
    boundary = min(headline_boundary, content_end)
    return max(current_bottom + int(3 * body_height), int(boundary - 0.35 * body_height))



def detect_body_columns(
    lines: Sequence[OCRLine],
    region_box: tuple[int, int, int, int],
    headline_line_ids: set[str],
    body_height: float,
) -> list[list[OCRLine]]:
    body_lines = [
        line
        for line in lines
        if line.uid not in headline_line_ids
        and not looks_like_byline(line.text)
        and line.height <= 1.65 * body_height
    ]
    bylines_and_subheads = [
        line
        for line in lines
        if line.uid not in headline_line_ids and line not in body_lines
    ]

    if not body_lines:
        return [sorted(bylines_and_subheads, key=lambda line: (line.y1, line.x1))]

    region_width = max(1, region_box[2] - region_box[0])
    tolerance = max(18.0, 0.045 * region_width)
    ordered = sorted(body_lines, key=lambda line: line.x1)

    groups: list[list[OCRLine]] = []
    for line in ordered:
        if not groups:
            groups.append([line])
            continue

        current_center = float(np.median([item.x1 for item in groups[-1]]))
        if abs(line.x1 - current_center) <= tolerance:
            groups[-1].append(line)
        else:
            groups.append([line])

    # Merge tiny x-start clusters into the nearest substantial cluster.
    changed = True
    while changed and len(groups) > 1:
        changed = False
        for index, group in enumerate(groups):
            if len(group) >= 3:
                continue
            group_center = float(np.median([item.x1 for item in group]))
            others = [candidate for candidate in range(len(groups)) if candidate != index]
            nearest = min(
                others,
                key=lambda candidate: abs(
                    group_center - float(np.median([item.x1 for item in groups[candidate]]))
                ),
            )
            groups[nearest].extend(group)
            groups.pop(index)
            changed = True
            break

    groups = sorted(groups, key=lambda group: np.median([line.x1 for line in group]))
    groups = [sorted(group, key=lambda line: (line.y1, line.x1)) for group in groups]

    # Add bylines/subheads before the body columns in top-to-bottom order.
    prefix = sorted(bylines_and_subheads, key=lambda line: (line.y1, line.x1))
    return ([prefix] if prefix else []) + groups



def article_lines_to_text(
    lines: Sequence[OCRLine],
    anchor: HeadlineAnchor,
    document_type: str,
    body_height: float,
) -> str:
    headline_ids = {line.uid for line in anchor.lines}
    region_box = bbox_from_lines(lines)
    columns = detect_body_columns(lines, region_box, headline_ids, body_height)

    ordered_body: list[OCRLine] = []
    for column in columns:
        ordered_body.extend(column)

    raw_lines = [line.text for line in ordered_body]
    cleaned_lines = clean_ocr_lines(raw_lines, document_type)
    cleaned_lines = join_hyphenated_lines(cleaned_lines)

    # Remove a byline only when it was accidentally retained as the first body line.
    while cleaned_lines and looks_like_byline(cleaned_lines[0]):
        cleaned_lines.pop(0)

    body = " ".join(cleaned_lines)
    body = re.sub(r"([A-Za-z])-\s+([a-z])", r"\1\2", body)
    body = re.sub(r"\s+", " ", body).strip()

    title = clean_text_basic(anchor.title)
    if body.lower().startswith(title.lower()):
        body = body[len(title):].lstrip(" .:-")

    return f"{title}. {body}".strip() if body else title



def build_provisional_article_regions(
    lines: Sequence[OCRLine],
    anchors: Sequence[HeadlineAnchor],
    separators: Sequence[Separator],
    content_area: tuple[int, int],
    page_shape: tuple[int, int],
    body_height: float,
    config: PipelineConfig,
) -> list[dict[str, Any]]:
    content_start, content_end = content_area
    page_h, page_w = page_shape
    regions: list[dict[str, Any]] = []

    for anchor in anchors:
        provisional_bottom = content_end
        initial_left = max(0, int(anchor.bbox[0] - 0.04 * page_w))
        initial_right = min(page_w, int(anchor.bbox[2] + 0.04 * page_w))

        preliminary_next = [
            candidate.bbox[1]
            for candidate in anchors
            if candidate is not anchor
            and candidate.bbox[1] > anchor.bbox[3] + 0.25 * body_height
            and (
                initial_left <= (candidate.bbox[0] + candidate.bbox[2]) / 2 <= initial_right
                or horizontal_overlap_boxes(
                    (initial_left, anchor.bbox[1], initial_right, anchor.bbox[3]),
                    candidate.bbox,
                ) >= 0.55
            )
        ]
        if preliminary_next:
            provisional_bottom = min(preliminary_next)

        left, right = find_article_horizontal_bounds(
            anchor,
            lines,
            separators,
            page_shape,
            provisional_bottom,
        )
        bottom = find_next_article_boundary(
            anchor,
            anchors,
            (left, right),
            separators,
            content_end,
            body_height,
        )

        box = (
            max(0, left),
            max(content_start, anchor.bbox[1] - int(0.3 * body_height)),
            min(page_w, right),
            min(content_end, bottom),
        )
        region_lines = [
            line
            for line in lines
            if line_inside_box(line, box, margin=int(0.15 * body_height))
            and not is_noise_line(line.text, "Newspaper")
        ]

        # Do not include headline lines belonging to a different article.
        other_headline_ids = {
            line.uid
            for other in anchors
            if other is not anchor and other.bbox[1] >= anchor.bbox[3]
            for line in other.lines
        }
        region_lines = [
            line
            for line in region_lines
            if line.uid not in other_headline_ids or line.uid in {item.uid for item in anchor.lines}
        ]
        region_lines.extend([line for line in anchor.lines if line not in region_lines])
        region_lines = list({line.uid: line for line in region_lines}.values())

        text = article_lines_to_text(
            region_lines,
            anchor,
            "Newspaper",
            body_height,
        )
        words = len(text.split())

        if words < config.min_article_words:
            continue
        if words > config.max_article_words:
            text = " ".join(text.split()[: config.max_article_words])
            words = config.max_article_words

        article_bbox = bbox_from_lines(region_lines)
        layout_confidence = min(
            1.0,
            0.30
            + min(0.25, anchor.score / 16.0)
            + min(0.25, words / 450.0)
            + (0.12 if any(
                separator.orientation == "vertical"
                and separator_overlaps_vertical_band(separator, box[1], box[3], 0.18)
                for separator in separators
            ) else 0.0)
            + (0.08 if any(
                separator.orientation == "horizontal"
                and separator_overlaps_horizontal_band(separator, box[0], box[2], 0.30)
                for separator in separators
            ) else 0.0),
        )

        warnings: list[str] = []
        if words < 45:
            warnings.append("Article contains limited text; summary reliability may be lower.")
        if ocr_quality_score(text) < 0.45:
            warnings.append("OCR quality is low.")

        regions.append(
            {
                "anchor": anchor,
                "lines": region_lines,
                "bbox": article_bbox,
                "text": text,
                "layout_confidence": round(layout_confidence, 3),
                "warnings": warnings,
            }
        )

    return regions



def deduplicate_and_merge_regions(
    regions: list[dict[str, Any]],
    config: PipelineConfig,
) -> list[dict[str, Any]]:
    if not regions:
        return []

    ordered = sorted(
        regions,
        key=lambda region: (region["bbox"][1], region["bbox"][0]),
    )
    kept: list[dict[str, Any]] = []

    for region in ordered:
        merged_or_duplicate = False
        for index, existing in enumerate(kept):
            title_score = title_similarity(
                region["anchor"].title,
                existing["anchor"].title,
            )
            containment = bbox_containment(region["bbox"], existing["bbox"])
            iou = bbox_iou(region["bbox"], existing["bbox"])
            shared_ids = {
                line.uid for line in region["lines"]
            } & {
                line.uid for line in existing["lines"]
            }
            shared_ratio = len(shared_ids) / max(
                1,
                min(len(region["lines"]), len(existing["lines"])),
            )

            if (
                title_score >= config.duplicate_title_similarity
                and (containment >= 0.55 or iou >= 0.35 or shared_ratio >= 0.45)
            ):
                # Keep the more complete version.
                if len(region["text"].split()) > len(existing["text"].split()):
                    kept[index] = region
                merged_or_duplicate = True
                break

        if not merged_or_duplicate:
            kept.append(region)

    return sorted(kept, key=lambda region: (region["bbox"][1], region["bbox"][0]))



def detect_adaptive_article_regions(
    lines: Sequence[OCRLine],
    prepared_gray: np.ndarray,
    document_type: str,
    config: PipelineConfig,
) -> dict[str, Any]:
    page_shape = prepared_gray.shape[:2]
    separators = detect_page_separators(prepared_gray)
    body_height = estimate_body_height(lines, page_shape)
    content_area = detect_adaptive_content_area(
        lines,
        separators,
        page_shape,
        body_height,
        config,
    )

    usable_lines = [
        line
        for line in lines
        if content_area[0] <= line.cy <= content_area[1]
        and not is_noise_line(line.text, document_type)
    ]

    candidates = detect_headline_candidates(
        usable_lines,
        body_height,
        page_shape,
        content_area,
        config,
    )
    anchors = combine_multiline_headlines(
        candidates,
        usable_lines,
        body_height,
        page_shape,
    )
    anchors = merge_short_headline_continuations(anchors, body_height)

    provisional = build_provisional_article_regions(
        usable_lines,
        anchors,
        separators,
        content_area,
        page_shape,
        body_height,
        config,
    )
    final_regions = deduplicate_and_merge_regions(provisional, config)

    warnings: list[str] = []
    if not anchors:
        warnings.append("No reliable headline anchors were detected.")
    if not final_regions:
        warnings.append("Article segmentation failed; the full-page fallback will be used.")
    if len(final_regions) > 24:
        warnings.append("High region count suggests possible over-segmentation.")

    return {
        "regions": final_regions,
        "anchors": anchors,
        "separators": separators,
        "body_height": body_height,
        "content_area": content_area,
        "warnings": warnings,
    }


# ============================================================
# 9. CATEGORY MODEL
# ============================================================

CATEGORY_KEYWORDS: dict[str, set[str]] = {
    "Sports": {
        "cricket", "slc", "icc", "football", "rugby", "tennis", "match",
        "team", "player", "tournament", "wicket", "goal", "coach", "captain",
    },
    "Politics": {
        "parliament", "minister", "president", "government", "opposition",
        "cabinet", "election", "treasury", "speaker", "copf", "ciaboc",
        "court", "secretary", "committee", "official", "policy", "rally", "rallies",
    },
    "Social": {
        "workers", "labour", "community", "rights", "trade union", "welfare",
        "families", "society", "dignity", "working people", "employment",
        "working", "workplace", "workforce", "social",
    },
    "Science": {
        "science", "scientific", "research", "biology", "chemistry", "physics",
        "experiment", "environment", "energy", "medical", "technology",
    },
    "Mathematics": {
        "mathematics", "maths", "equation", "formula", "algebra", "geometry",
        "percentage", "ratio", "fraction", "calculate", "number",
    },
    "Story": {
        "story", "chapter", "novel", "character", "once", "forest", "king",
        "queen", "boy", "girl", "narrative",
    },
    "General Text": {
        "oil", "price", "market", "business", "bank", "trade", "industry",
    },
}



def get_model_classes(model) -> list[str]:
    if hasattr(model, "classes_"):
        return [str(item) for item in model.classes_]
    if hasattr(model, "named_steps"):
        final_step = list(model.named_steps.values())[-1]
        if hasattr(final_step, "classes_"):
            return [str(item) for item in final_step.classes_]
    return PROJECT_CATEGORIES



def strong_title_keyword_category(title: str) -> Optional[tuple[str, int]]:
    tokens = set(re.findall(r"[a-z]+", title.lower()))
    ranking = []
    for category, keywords in CATEGORY_KEYWORDS.items():
        count = len(tokens & keywords)
        if count:
            ranking.append((category, count))
    if not ranking:
        return None
    ranking.sort(key=lambda item: item[1], reverse=True)
    if ranking[0][1] >= 2 or (ranking[0][1] == 1 and len(title.split()) <= 12):
        return ranking[0]
    return None



def predict_category(
    model,
    text: str,
    document_type: str,
    title: str,
    config: PipelineConfig,
) -> dict[str, Any]:
    normalized_type = normalize_document_type(document_type)
    cleaned = clean_text_basic(text)

    if normalized_type == "Novel/Story":
        return {
            "category": "Story",
            "confidence": 1.0,
            "margin": 1.0,
            "top_predictions": [{"category": "Story", "probability": 1.0}],
            "decision": "document_type_rule",
        }

    if len(cleaned.split()) < 8:
        return {
            "category": "General Text",
            "confidence": 0.0,
            "margin": 0.0,
            "top_predictions": [],
            "decision": "too_little_text",
        }

    repeated_title = " ".join([title] * max(1, config.title_repeat_for_category))
    classification_text = clean_text_basic(f"{repeated_title} {cleaned}")

    predicted = str(model.predict([classification_text])[0])
    category = predicted if predicted in PROJECT_CATEGORIES else "General Text"
    confidence = 1.0
    margin = 1.0
    top_predictions: list[dict[str, Any]] = []

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([classification_text])[0]
        classes = get_model_classes(model)
        ranking = np.argsort(probabilities)[::-1]
        top_predictions = [
            {
                "category": classes[int(index)],
                "probability": round(float(probabilities[int(index)]), 4),
            }
            for index in ranking[:3]
        ]
        confidence = float(probabilities[int(ranking[0])])
        margin = confidence - float(probabilities[int(ranking[1])]) if len(ranking) > 1 else confidence
        category = classes[int(ranking[0])]

    decision = "model_probability_and_margin"
    keyword = strong_title_keyword_category(title)
    if keyword is not None:
        keyword_category, keyword_count = keyword
        if keyword_count >= 2 or confidence < 0.55:
            category = keyword_category
            decision = "strong_title_keyword_support"

    if confidence < config.category_min_probability or margin < config.category_min_margin:
        if keyword is None:
            category = "General Text"
            decision = "low_confidence_or_margin_fallback"

    # Prevent weak textbook predictions from becoming newspaper categories.
    if normalized_type == "Textbook" and category in {"Politics", "Social", "Sports", "Story"}:
        if confidence < 0.70:
            category = "General Text"
            decision = "textbook_type_gate"

    return {
        "category": category if category in PROJECT_CATEGORIES else "General Text",
        "confidence": round(confidence, 4),
        "margin": round(margin, 4),
        "top_predictions": top_predictions,
        "decision": decision,
    }


# ============================================================
# 10. LOCAL OLLAMA / LLAMA 3.2:3B
# ============================================================


def check_ollama(config: PipelineConfig) -> dict[str, Any]:
    try:
        response = requests.get(config.ollama_tags_url, timeout=10)
        response.raise_for_status()
        models = [item.get("name", "") for item in response.json().get("models", [])]
        exact_or_base = any(
            model == config.ollama_model
            or model.startswith(config.ollama_model + ":")
            or config.ollama_model.startswith(model + ":")
            for model in models
        )
        return {
            "available": True,
            "configured_model": config.ollama_model,
            "installed_models": models,
            "configured_model_found": exact_or_base or config.ollama_model in models,
        }
    except Exception as error:
        return {
            "available": False,
            "configured_model": config.ollama_model,
            "installed_models": [],
            "configured_model_found": False,
            "error": str(error),
        }



def ollama_generate(
    prompt: str,
    config: PipelineConfig,
    temperature: float = 0.1,
    num_predict: int = 260,
) -> str:
    try:
        response = requests.post(
            config.ollama_url,
            json={
                "model": config.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "top_p": 0.9,
                    "num_predict": num_predict,
                },
            },
            timeout=config.ollama_timeout_seconds,
        )
        if response.status_code != 200:
            if config.debug:
                print("Ollama status:", response.status_code)
                print("Ollama response:", response.text[:500])
            return ""
        return clean_text_basic(response.json().get("response", ""))
    except Exception as error:
        if config.debug:
            print(f"Ollama unavailable; using local extractive fallback: {error}")
        return ""



def extractive_summary(text: str, max_sentences: int = 3) -> str:
    cleaned = clean_text_basic(text)
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    sentences = [sentence.strip() for sentence in sentences if len(sentence.split()) >= 6]
    if not sentences:
        return " ".join(cleaned.split()[:100])

    words = re.findall(r"[A-Za-z]{3,}", cleaned.lower())
    stop = {
        "the", "and", "for", "that", "with", "this", "from", "were", "have",
        "has", "had", "but", "not", "are", "was", "will", "their", "they",
        "into", "about", "after", "before", "which", "when", "where", "who",
    }
    frequencies = Counter(word for word in words if word not in stop)
    scored = []
    for index, sentence in enumerate(sentences):
        sentence_words = re.findall(r"[A-Za-z]{3,}", sentence.lower())
        score = sum(frequencies[word] for word in sentence_words) / max(1, len(sentence_words))
        scored.append((score, index, sentence))

    selected = sorted(scored, reverse=True)[:max_sentences]
    return " ".join(item[2] for item in sorted(selected, key=lambda item: item[1]))



def reconstruct_selected_text_with_llama(
    text: str,
    title: str,
    document_type: str,
    config: PipelineConfig,
) -> str:
    cleaned = clean_text_basic(text)
    if len(cleaned.split()) < 30:
        return cleaned

    prompt = f"""
You clean OCR article text for a blind reader.
Document type: {document_type}
Article title: {title}

Rules:
- Fix only obvious OCR spacing, broken line endings, and broken words.
- Preserve all names, facts, dates, places, numbers, and quotations.
- Do not summarize.
- Do not add information.
- Remove page numbers, masthead fragments, captions that are unrelated to the article,
  and clearly unrelated neighboring fragments.
- If uncertain, preserve the original wording rather than guessing.
- Return only the cleaned full article text.

OCR article text:
{cleaned[:12000]}

Clean full text:
""".strip()

    result = ollama_generate(prompt, config, temperature=0.05, num_predict=1200)
    return result if len(result.split()) >= 15 else cleaned



def summarize_with_local_llama(
    text: str,
    title: str,
    document_type: str,
    category: str,
    config: PipelineConfig,
) -> str:
    cleaned = clean_text_basic(text)
    if not cleaned:
        return "Readable text was not found."
    if len(cleaned.split()) < 35:
        return extractive_summary(cleaned, max_sentences=2)

    words = cleaned.split()
    chunks = [" ".join(words[index:index + 750]) for index in range(0, len(words), 750)]
    partials: list[str] = []

    for chunk in chunks[:4]:
        prompt = f"""
Summarize this locally extracted printed article for a blind reader.
Document type: {document_type}
Category: {category}
Article title: {title}

Rules:
- Use only facts present in the article.
- Keep important names, places, dates, numbers, and the main point.
- Use clear simple English.
- Do not mention OCR, the prompt, or system processing.
- Do not add missing facts or explanations.
- Maximum {config.summary_max_words} words.
- Return only the summary.

Article:
{chunk}

Summary:
""".strip()
        partial = ollama_generate(prompt, config, temperature=0.1, num_predict=240)
        partials.append(partial or extractive_summary(chunk, max_sentences=3))

    if len(partials) == 1:
        return partials[0]

    combined = " ".join(partials)
    final_prompt = f"""
Combine these partial summaries into one accurate summary for a blind reader.
Use only the supplied information. Maximum {config.summary_max_words} words.
Return only the final summary.

Partial summaries:
{combined}

Final summary:
""".strip()
    final = ollama_generate(final_prompt, config, temperature=0.05, num_predict=240)
    return final or extractive_summary(combined, max_sentences=3)


# ============================================================
# 11. OPTIONAL LOCAL TTS AND OFFLINE VOICE INPUT
# ============================================================


def speak_local(text: str, enabled: bool = False) -> None:
    if not enabled:
        return
    try:
        import pyttsx3

        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except Exception as error:
        print(f"Local TTS unavailable: {error}")



def normalize_choice(value: str, allowed: Sequence[str]) -> Optional[str]:
    spoken = re.sub(r"[^a-z0-9 ]", " ", value.lower())
    spoken = re.sub(r"\s+", " ", spoken).strip()
    aliases = {
        "full text": "full",
        "complete text": "full",
        "summarize": "summary",
        "general": "General Text",
        "math": "Mathematics",
        "maths": "Mathematics",
    }
    if spoken in aliases and aliases[spoken] in allowed:
        return aliases[spoken]
    for option in allowed:
        if spoken == option.lower() or option.lower() in spoken:
            return option
    return None



def get_user_choice(
    prompt: str,
    allowed: list[str],
    input_mode: str = "keyboard",
    vosk_model_path: Optional[str] = None,
    tts_enabled: bool = False,
    timeout_seconds: int = 10,
) -> str:
    speak_local(f"{prompt}. Options are {', '.join(allowed)}", enabled=tts_enabled)

    if input_mode.lower() != "voice":
        while True:
            print(prompt)
            print("Options:", ", ".join(allowed))
            value = input("Your choice: ").strip()
            normalized = normalize_choice(value, allowed)
            if normalized:
                return normalized
            print("Invalid choice. Please try again.\n")

    if not vosk_model_path:
        print("Vosk model path is missing. Falling back to keyboard.")
        return get_user_choice(prompt, allowed, input_mode="keyboard", tts_enabled=tts_enabled)

    try:
        import sounddevice as sd
        from vosk import KaldiRecognizer, Model

        audio_queue: queue.Queue[bytes] = queue.Queue()
        model = Model(vosk_model_path)
        grammar = json.dumps([option.lower() for option in allowed] + ["unknown"])
        recognizer = KaldiRecognizer(model, 16000, grammar)

        def callback(indata, frames, callback_time, status):
            del frames, callback_time
            if status:
                print(status)
            audio_queue.put(bytes(indata))

        print(prompt)
        print("Say one of:", ", ".join(allowed))
        start_time = time.time()

        with sd.RawInputStream(
            samplerate=16000,
            blocksize=8000,
            dtype="int16",
            channels=1,
            callback=callback,
        ):
            while time.time() - start_time < timeout_seconds:
                data = audio_queue.get(timeout=timeout_seconds)
                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result()).get("text", "")
                    normalized = normalize_choice(result, allowed)
                    if normalized:
                        print("Recognized:", normalized)
                        return normalized

        final_text = json.loads(recognizer.FinalResult()).get("text", "")
        normalized = normalize_choice(final_text, allowed)
        if normalized:
            return normalized

        print("Voice was not recognized. Falling back to keyboard.")
        return get_user_choice(prompt, allowed, input_mode="keyboard", tts_enabled=tts_enabled)

    except Exception as error:
        print(f"Offline voice input unavailable: {error}")
        return get_user_choice(prompt, allowed, input_mode="keyboard", tts_enabled=tts_enabled)


# ============================================================
# 12. MAIN ANALYSIS
# ============================================================


def analyze_content(
    image_path: str,
    document_type: str,
    model,
    config: PipelineConfig,
) -> dict[str, Any]:
    normalized_type = normalize_document_type(document_type)
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Image not found or unreadable: {image_path}")

    prepared_color, prepared_gray, scale = prepare_page_image(image, config)
    page_shape = prepared_gray.shape[:2]
    warnings: list[str] = []
    layout_debug: dict[str, Any] = {}

    if normalized_type in MULTI_CATEGORY_TYPES:
        ocr_lines = extract_ocr_lines(prepared_gray, normalized_type, config)
        layout = detect_adaptive_article_regions(
            ocr_lines,
            prepared_gray,
            normalized_type,
            config,
        )
        warnings.extend(layout["warnings"])
        layout_debug = {
            "ocr_line_count": len(ocr_lines),
            "body_height": round(float(layout["body_height"]), 2),
            "content_area": layout["content_area"],
            "headline_anchor_count": len(layout["anchors"]),
            "separator_count": len(layout["separators"]),
            "region_count": len(layout["regions"]),
            "anchors": [
                {"title": anchor.title, "bbox": anchor.bbox, "score": round(anchor.score, 3)}
                for anchor in layout["anchors"]
            ],
            "separators": [asdict(separator) for separator in layout["separators"]],
        }

        results: list[ArticleRegion] = []
        for number, region in enumerate(layout["regions"], start=1):
            anchor = region["anchor"]
            prediction = predict_category(
                model,
                region["text"],
                normalized_type,
                anchor.title,
                config,
            )
            results.append(
                ArticleRegion(
                    article_number=number,
                    title=anchor.title,
                    bbox=region["bbox"],
                    normalized_bbox=normalize_bbox(region["bbox"], page_shape),
                    full_text=region["text"],
                    category=prediction["category"],
                    confidence=prediction["confidence"],
                    probability_margin=prediction["margin"],
                    top_predictions=prediction["top_predictions"],
                    ocr_quality=ocr_quality_score(region["text"]),
                    layout_confidence=region["layout_confidence"],
                    warnings=region["warnings"],
                )
            )

        if not results:
            full_text = best_full_page_ocr(prepared_gray, normalized_type)
            prediction = predict_category(
                model,
                full_text,
                normalized_type,
                "Full page",
                config,
            )
            results = [
                ArticleRegion(
                    article_number=1,
                    title="Full page",
                    bbox=(0, 0, page_shape[1], page_shape[0]),
                    normalized_bbox=(0, 0, 1000, 1000),
                    full_text=full_text,
                    category=prediction["category"],
                    confidence=prediction["confidence"],
                    probability_margin=prediction["margin"],
                    top_predictions=prediction["top_predictions"],
                    ocr_quality=ocr_quality_score(full_text),
                    layout_confidence=0.25,
                    warnings=["Article segmentation fallback: the full page was processed as one region."],
                    source="full_page_fallback",
                )
            ]
    else:
        full_text = best_full_page_ocr(prepared_gray, normalized_type)
        title = "Full page"
        prediction = predict_category(
            model,
            full_text,
            normalized_type,
            title,
            config,
        )
        results = [
            ArticleRegion(
                article_number=1,
                title=title,
                bbox=(0, 0, page_shape[1], page_shape[0]),
                normalized_bbox=(0, 0, 1000, 1000),
                full_text=full_text,
                category=prediction["category"],
                confidence=prediction["confidence"],
                probability_margin=prediction["margin"],
                top_predictions=prediction["top_predictions"],
                ocr_quality=ocr_quality_score(full_text),
                layout_confidence=1.0,
                warnings=[],
                source="full_page_tesseract",
            )
        ]

    category_counts = Counter(result.category for result in results)
    categories = sorted(category_counts)
    document_mode = "Multi Category" if normalized_type in MULTI_CATEGORY_TYPES else "Single Category"

    return {
        "status": "ANALYZED",
        "document_type_received": document_type,
        "document_type": normalized_type,
        "mode": document_mode,
        "detected_categories": categories,
        "category_counts": dict(category_counts),
        "needs_category_selection": normalized_type in MULTI_CATEGORY_TYPES and len(categories) > 1,
        "image_shape": {"height": page_shape[0], "width": page_shape[1]},
        "resize_scale": scale,
        "warnings": warnings,
        "layout_debug": layout_debug,
        "results": [asdict(result) for result in results],
        "processing": {
            "cloud_api_used": False,
            "ocr": "local_tesseract",
            "article_detection": "adaptive_headline_separator_geometry",
            "category_model": "local_tfidf_logistic_regression",
            "summary_model": f"local_ollama_{config.ollama_model}",
        },
    }


# ============================================================
# 13. USER SELECTION AND STRUCTURED OUTPUT
# ============================================================


def generate_selected_output(
    analysis: dict[str, Any],
    selected_category: Optional[str],
    depth: str,
    config: PipelineConfig,
    clean_full_text_with_llama: bool = False,
) -> dict[str, Any]:
    categories = analysis.get("detected_categories", [])
    if not categories:
        return {"status": "ERROR", "message": "No category was detected."}

    if selected_category is None:
        selected_category = categories[0] if len(categories) == 1 else None

    if selected_category not in categories:
        return {
            "status": "ERROR",
            "message": "Selected category was not detected.",
            "available_categories": categories,
        }

    depth = depth.strip().lower()
    if depth not in {"summary", "full"}:
        return {"status": "ERROR", "message": "Depth must be 'summary' or 'full'."}

    selected_items = [
        item for item in analysis["results"] if item["category"] == selected_category
    ]

    output_items = []
    for item in selected_items:
        if depth == "summary":
            content = summarize_with_local_llama(
                text=item["full_text"],
                title=item["title"],
                document_type=analysis["document_type"],
                category=selected_category,
                config=config,
            )
        elif clean_full_text_with_llama:
            content = reconstruct_selected_text_with_llama(
                text=item["full_text"],
                title=item["title"],
                document_type=analysis["document_type"],
                config=config,
            )
        else:
            content = item["full_text"]

        output_items.append(
            {
                "article_number": item["article_number"],
                "title": item["title"],
                "category": item["category"],
                "category_confidence": item["confidence"],
                "probability_margin": item["probability_margin"],
                "ocr_quality": item["ocr_quality"],
                "layout_confidence": item["layout_confidence"],
                "warnings": item["warnings"],
                "text": content,
                "bbox": item["normalized_bbox"],
            }
        )

    if analysis["document_type"] in MULTI_CATEGORY_TYPES:
        intro = (
            f"{selected_category} section. I found {len(output_items)} "
            f"article{'s' if len(output_items) != 1 else ''}."
        )
        spoken_parts = [intro]
        for index, item in enumerate(output_items, start=1):
            spoken_parts.append(f"Article {index}. {item['title']}. {item['text']}")
        final_text = "\n\n".join(spoken_parts)
    else:
        final_text = "\n\n".join(item["text"] for item in output_items)

    return {
        "status": "READY",
        "document_type": analysis["document_type"],
        "mode": analysis["mode"],
        "selected_category": selected_category,
        "content_depth": depth,
        "article_count": len(output_items),
        "items": output_items,
        "final_output_text": final_text,
        "next_module_payload": {
            "text": final_text,
            "document_type": analysis["document_type"],
            "category": selected_category,
            "content_depth": depth,
            "source_article_count": len(output_items),
            "articles": output_items,
        },
    }



def run_interactive_flow(
    image_path: str,
    document_type: str,
    model,
    config: PipelineConfig,
    input_mode: str = "keyboard",
    vosk_model_path: Optional[str] = None,
    tts_enabled: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    analysis = analyze_content(image_path, document_type, model, config)
    categories = analysis["detected_categories"]

    if len(categories) == 1:
        selected_category = categories[0]
        print("Single detected category:", selected_category)
        speak_local(f"The detected category is {selected_category}", enabled=tts_enabled)
    else:
        selected_category = get_user_choice(
            "Which category do you want to hear?",
            categories,
            input_mode=input_mode,
            vosk_model_path=vosk_model_path,
            tts_enabled=tts_enabled,
        )

    depth = get_user_choice(
        "Do you want a summary or full text?",
        ["summary", "full"],
        input_mode=input_mode,
        vosk_model_path=vosk_model_path,
        tts_enabled=tts_enabled,
    )

    output = generate_selected_output(
        analysis,
        selected_category,
        depth,
        config,
    )
    return analysis, output


# ============================================================
# 14. JUPYTER TEST AND VISUALIZATION HELPERS
# ============================================================


def analysis_table(analysis: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "article": item["article_number"],
                "title": item["title"],
                "category": item["category"],
                "confidence": item["confidence"],
                "margin": item["probability_margin"],
                "layout_confidence": item["layout_confidence"],
                "ocr_quality": item["ocr_quality"],
                "words": len(item["full_text"].split()),
                "source": item["source"],
            }
            for item in analysis.get("results", [])
        ]
    )



def print_analysis_details(analysis: dict[str, Any], max_text_chars: int = 1000) -> None:
    print("Document type:", analysis["document_type"])
    print("Mode:", analysis["mode"])
    print("Categories:", analysis["detected_categories"])
    print("Category counts:", analysis["category_counts"])
    print("Warnings:", analysis.get("warnings", []))

    for item in analysis.get("results", []):
        print("\n" + "=" * 90)
        print("Article:", item["article_number"])
        print("Title:", item["title"])
        print("Category:", item["category"])
        print("Confidence:", item["confidence"])
        print("Margin:", item["probability_margin"])
        print("Layout confidence:", item["layout_confidence"])
        print("OCR quality:", item["ocr_quality"])
        print("Words:", len(item["full_text"].split()))
        print("Bounding box:", item["bbox"])
        if item["warnings"]:
            print("Article warnings:", item["warnings"])
        print("\nText preview:\n", item["full_text"][:max_text_chars])



def show_analysis_regions(
    image_path: str,
    analysis: dict[str, Any],
    show_titles: bool = True,
) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)

    target_w = analysis["image_shape"]["width"]
    target_h = analysis["image_shape"]["height"]
    display = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

    for item in analysis.get("results", []):
        x1, y1, x2, y2 = item["bbox"]
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 3)
        label = f"{item['article_number']}: {item['category']}"
        if show_titles:
            label += " - " + " ".join(item["title"].split()[:5])
        cv2.putText(
            display,
            label,
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    plt.figure(figsize=(14, 18))
    plt.imshow(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Adaptive Local Article Regions and Categories")
    plt.show()



def show_layout_debug(
    image_path: str,
    analysis: dict[str, Any],
) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)

    target_w = analysis["image_shape"]["width"]
    target_h = analysis["image_shape"]["height"]
    display = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    debug = analysis.get("layout_debug", {})

    for separator in debug.get("separators", []):
        cv2.line(
            display,
            (int(separator["x1"]), int(separator["y1"])),
            (int(separator["x2"]), int(separator["y2"])),
            (255, 0, 0),
            2,
        )

    for index, anchor in enumerate(debug.get("anchors", []), start=1):
        x1, y1, x2, y2 = anchor["bbox"]
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 165, 255), 2)
        cv2.putText(
            display,
            f"H{index}",
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
        )

    content_area = debug.get("content_area")
    if content_area:
        cv2.line(display, (0, content_area[0]), (target_w, content_area[0]), (255, 0, 255), 3)
        cv2.line(display, (0, content_area[1]), (target_w, content_area[1]), (255, 0, 255), 3)

    plt.figure(figsize=(14, 18))
    plt.imshow(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Blue: separators | Orange: headline anchors | Magenta: content area")
    plt.show()



def test_category_model(model, config: PipelineConfig) -> pd.DataFrame:
    samples = [
        ("Cricket team wins championship", "The cricket team won the match by five wickets."),
        ("Minister addresses Parliament", "The minister addressed parliament before the election."),
        ("New scientific discovery", "Scientists studied cells, energy and the environment."),
        ("Mathematics lesson", "Calculate the percentage and solve the equation."),
        ("Workers discuss labour rights", "Workers discussed rights and community welfare."),
        ("A child enters the forest", "Once upon a time a child walked into the forest."),
        ("General public notice", "This notice contains ordinary administrative information."),
    ]
    rows = []
    for title, text in samples:
        result = predict_category(model, text, "General Document", title, config)
        rows.append(
            {
                "title": title,
                "prediction": result["category"],
                "confidence": result["confidence"],
                "margin": result["margin"],
                "decision": result["decision"],
                "top_predictions": result["top_predictions"],
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("Import this module into Jupyter and use the accompanying notebook cells.")


# ============================================================
# 15. V2 ROBUST OFFLINE MICROPHONE SUPPORT
# ============================================================

def list_microphone_devices() -> list[dict[str, Any]]:
    """
    Return available input devices from sounddevice.
    Useful in Jupyter before voice testing.
    """
    try:
        import sounddevice as sd
    except Exception as error:
        print(f"sounddevice is unavailable: {error}")
        return []

    devices = sd.query_devices()
    output = []
    for index, device in enumerate(devices):
        max_inputs = int(device.get("max_input_channels", 0))
        if max_inputs <= 0:
            continue
        item = {
            "id": index,
            "name": str(device.get("name", "")),
            "input_channels": max_inputs,
            "default_samplerate": float(device.get("default_samplerate", 16000)),
        }
        output.append(item)
        print(
            f"[{item['id']}] {item['name']} | "
            f"inputs={item['input_channels']} | "
            f"default_rate={item['default_samplerate']:.0f}"
        )
    return output


def _voice_alias_map(allowed: Sequence[str]) -> dict[str, str]:
    aliases: dict[str, str] = {
        # Depth
        "summary": "summary",
        "summarize": "summary",
        "summarise": "summary",
        "short summary": "summary",
        "brief summary": "summary",
        "give summary": "summary",
        "read summary": "summary",

        "full": "full",
        "full text": "full",
        "complete text": "full",
        "complete": "full",
        "read full": "full",
        "read full text": "full",
        "read everything": "full",

        # Categories
        "sport": "Sports",
        "sports": "Sports",
        "sports news": "Sports",

        "politics": "Politics",
        "political": "Politics",
        "political news": "Politics",

        "social": "Social",
        "social news": "Social",

        "science": "Science",
        "scientific": "Science",

        "mathematics": "Mathematics",
        "math": "Mathematics",
        "maths": "Mathematics",

        "story": "Story",
        "stories": "Story",

        "general": "General Text",
        "general text": "General Text",
        "general news": "General Text",
    }

    # Keep only aliases whose mapped value is currently allowed.
    allowed_set = set(allowed)
    return {
        phrase: target
        for phrase, target in aliases.items()
        if target in allowed_set
    }


def normalize_voice_choice(value: str, allowed: Sequence[str]) -> Optional[str]:
    spoken = re.sub(r"[^a-z0-9 ]", " ", str(value).lower())
    spoken = re.sub(r"\s+", " ", spoken).strip()

    if not spoken:
        return None

    alias_map = _voice_alias_map(allowed)

    if spoken in alias_map:
        return alias_map[spoken]

    # Exact option
    for option in allowed:
        if spoken == option.lower():
            return option

    # Phrase contains alias or option.
    # Longer phrases first so "full text" is preferred over "full".
    for phrase in sorted(alias_map, key=len, reverse=True):
        if phrase in spoken:
            return alias_map[phrase]

    for option in sorted(allowed, key=len, reverse=True):
        if option.lower() in spoken:
            return option

    return None


def build_vosk_grammar(allowed: Sequence[str]) -> list[str]:
    """
    Build a small command grammar. Constraining Vosk to this vocabulary
    improves reliability for category/depth selection.
    """
    grammar = {str(option).lower() for option in allowed}
    grammar.update(_voice_alias_map(allowed).keys())
    grammar.add("[unk]")
    return sorted(grammar)


class OfflineVoiceSelector:
    """
    Cached offline Vosk recognizer for short command choices.

    Why cache the model?
    Loading the Vosk model for category selection and then loading it again
    for Summary/Full wastes time and RAM, especially on Raspberry Pi.
    """

    def __init__(
        self,
        vosk_model_path: str,
        device_id: Optional[int] = None,
        sample_rate: Optional[int] = None,
        blocksize: int = 4000,
    ):
        try:
            import sounddevice as sd
            from vosk import Model
        except Exception as error:
            raise RuntimeError(
                "Install vosk and sounddevice before using voice input."
            ) from error

        model_path = Path(vosk_model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Vosk model folder not found: {model_path.resolve()}"
            )

        self.sd = sd
        self.device_id = device_id
        self.blocksize = int(blocksize)
        self.model = Model(str(model_path))

        device_info = sd.query_devices(device_id, "input")
        detected_rate = int(float(device_info.get("default_samplerate", 16000)))
        self.sample_rate = int(sample_rate or detected_rate)

        print("Offline Vosk model loaded.")
        print("Microphone:", device_info.get("name", "Unknown input device"))
        print("Sample rate:", self.sample_rate)

    def listen_choice(
        self,
        prompt: str,
        allowed: Sequence[str],
        timeout_seconds: float = 12.0,
        retries: int = 2,
        tts_enabled: bool = False,
    ) -> str:
        from vosk import KaldiRecognizer

        allowed = list(allowed)
        if not allowed:
            raise ValueError("No allowed choices were provided.")

        speak_local(
            f"{prompt}. Options are {', '.join(allowed)}.",
            enabled=tts_enabled,
        )

        grammar = build_vosk_grammar(allowed)

        for attempt in range(1, retries + 1):
            audio_queue: queue.Queue[bytes] = queue.Queue()
            recognizer = KaldiRecognizer(
                self.model,
                float(self.sample_rate),
                json.dumps(grammar),
            )
            recognizer.SetWords(False)

            def callback(indata, frames, callback_time, status):
                del frames, callback_time
                if status:
                    print("Audio status:", status)
                audio_queue.put(bytes(indata))

            print("\n" + prompt)
            print("Allowed:", ", ".join(allowed))
            print(f"Listening... attempt {attempt}/{retries}")

            start = time.monotonic()
            last_partial = ""

            try:
                with self.sd.RawInputStream(
                    samplerate=self.sample_rate,
                    blocksize=self.blocksize,
                    device=self.device_id,
                    dtype="int16",
                    channels=1,
                    callback=callback,
                ):
                    while time.monotonic() - start < timeout_seconds:
                        remaining = max(
                            0.1,
                            timeout_seconds - (time.monotonic() - start)
                        )
                        try:
                            data = audio_queue.get(timeout=min(0.75, remaining))
                        except queue.Empty:
                            continue

                        if recognizer.AcceptWaveform(data):
                            result = json.loads(recognizer.Result())
                            transcript = result.get("text", "").strip()
                            choice = normalize_voice_choice(transcript, allowed)

                            if transcript:
                                print("Recognized speech:", transcript)

                            if choice:
                                print("Selected:", choice)
                                return choice
                        else:
                            partial = json.loads(
                                recognizer.PartialResult()
                            ).get("partial", "").strip()

                            if partial and partial != last_partial:
                                last_partial = partial
                                print("Partial:", partial)

                final_result = json.loads(recognizer.FinalResult())
                final_text = final_result.get("text", "").strip()

                if final_text:
                    print("Final recognized speech:", final_text)

                choice = normalize_voice_choice(final_text, allowed)
                if choice:
                    print("Selected:", choice)
                    return choice

            except Exception as error:
                print(f"Microphone attempt failed: {error}")

            if attempt < retries:
                print("I did not recognize an allowed choice. Please try again.")
                speak_local(
                    "I did not recognize the choice. Please say one option again.",
                    enabled=tts_enabled,
                )

        raise RuntimeError(
            "Voice input was not recognized after the configured retries."
        )


def test_microphone_choice(
    voice_selector: OfflineVoiceSelector,
    allowed: Sequence[str] = ("summary", "full"),
    tts_enabled: bool = False,
) -> str:
    """Small Jupyter test before running OCR or Llama."""
    return voice_selector.listen_choice(
        "Microphone test. Please say one option",
        allowed,
        timeout_seconds=10,
        retries=2,
        tts_enabled=tts_enabled,
    )


def collect_user_preferences_voice(
    analysis: dict[str, Any],
    voice_selector: OfflineVoiceSelector,
    tts_enabled: bool = False,
) -> tuple[str, str]:
    """
    Ask only for choices that are actually needed.

    Newspaper/Magazine:
      - If >1 detected category: ask category.
      - If only 1 category: skip category selection.

    All documents:
      - Ask Summary or Full Text.
    """
    categories = list(analysis.get("detected_categories", []))
    if not categories:
        raise ValueError("No categories were detected.")

    if len(categories) == 1:
        selected_category = categories[0]
        print("Single category detected:", selected_category)
        speak_local(
            f"The detected category is {selected_category}.",
            enabled=tts_enabled,
        )
    else:
        selected_category = voice_selector.listen_choice(
            "Which category do you want to hear",
            categories,
            timeout_seconds=12,
            retries=3,
            tts_enabled=tts_enabled,
        )

    depth = voice_selector.listen_choice(
        "Do you want a summary or full text",
        ["summary", "full"],
        timeout_seconds=10,
        retries=3,
        tts_enabled=tts_enabled,
    )

    return selected_category, depth


def run_voice_flow_from_existing_analysis(
    analysis: dict[str, Any],
    config: PipelineConfig,
    voice_selector: OfflineVoiceSelector,
    tts_enabled: bool = False,
    clean_full_text_with_llama: bool = False,
) -> dict[str, Any]:
    """
    Recommended Jupyter voice test:
    analyze the page ONCE, then use microphone choices without re-running OCR.
    """
    selected_category, depth = collect_user_preferences_voice(
        analysis=analysis,
        voice_selector=voice_selector,
        tts_enabled=tts_enabled,
    )

    output = generate_selected_output(
        analysis=analysis,
        selected_category=selected_category,
        depth=depth,
        config=config,
        clean_full_text_with_llama=clean_full_text_with_llama,
    )

    print("\nSelected category:", selected_category)
    print("Selected depth:", depth)
    return output


# ============================================================
# 16. V2 ARTICLE-REGION QUALITY DIAGNOSTICS
# ============================================================

def article_region_quality_table(analysis: dict[str, Any]) -> pd.DataFrame:
    """
    Helps identify which detections should be inspected/fine-tuned.
    It does not pretend to be ground-truth accuracy.
    """
    rows = []

    for item in analysis.get("results", []):
        words = len(item.get("full_text", "").split())
        layout = float(item.get("layout_confidence", 0.0))
        ocr_q = float(item.get("ocr_quality", 0.0))
        cat_conf = float(item.get("confidence", 0.0) or 0.0)
        margin = float(item.get("probability_margin", 0.0) or 0.0)

        score = (
            0.40 * layout
            + 0.25 * ocr_q
            + 0.20 * min(1.0, words / 120.0)
            + 0.10 * cat_conf
            + 0.05 * min(1.0, max(0.0, margin) / 0.30)
        )

        flags = []
        if words < 35:
            flags.append("short_region")
        if layout < 0.55:
            flags.append("low_layout_confidence")
        if ocr_q < 0.45:
            flags.append("low_ocr_quality")
        if item.get("title", "").strip().lower() in {"full page", "untitled article"}:
            flags.append("weak_title")

        rows.append(
            {
                "article": item.get("article_number"),
                "title": item.get("title"),
                "words": words,
                "layout_confidence": round(layout, 3),
                "ocr_quality": round(ocr_q, 3),
                "category": item.get("category"),
                "category_confidence": round(cat_conf, 3),
                "quality_score": round(score, 3),
                "inspect": ", ".join(flags) if flags else "OK",
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["quality_score", "article"],
        ascending=[True, True],
    ).reset_index(drop=True)


# ============================================================
# 17. HYBRID DOCLAYOUT-YOLO + LOCAL GEOMETRY DETECTOR
# ============================================================
#
# DocLayout-YOLO is used only as a LOCAL pretrained layout detector.
# It supplies structural hints such as:
#   title, plain_text, figure, figure_caption, ...
#
# It does NOT directly decide which complete newspaper article each
# element belongs to. The grouping below is our own local association
# logic, fused with the existing adaptive headline/separator geometry.
# ============================================================


@dataclass
class HybridPipelineConfig(PipelineConfig):
    # DocLayout-YOLO model
    use_doclayout_yolo: bool = True
    doclayout_repo_id: str = "juliozhao/DocLayout-YOLO-DocStructBench"
    doclayout_filename: str = "doclayout_yolo_docstructbench_imgsz1024.pt"
    doclayout_checkpoint_path: Optional[str] = None

    # "auto", "cpu", "cuda", "cuda:0", ...
    doclayout_device: str = "auto"
    doclayout_imgsz: int = 1024
    doclayout_confidence: float = 0.20
    doclayout_iou: float = 0.45

    # Which pretrained layout elements help our newspaper grouping.
    doclayout_title_classes: tuple[str, ...] = ("title",)
    doclayout_text_classes: tuple[str, ...] = ("plain_text",)
    doclayout_caption_classes: tuple[str, ...] = ("figure_caption",)
    doclayout_figure_classes: tuple[str, ...] = ("figure",)

    # Fusion / article association
    doclayout_min_title_confidence: float = 0.22
    doclayout_min_text_confidence: float = 0.18
    doclayout_title_line_overlap: float = 0.22
    doclayout_text_line_overlap: float = 0.18
    doclayout_max_body_distance_ratio: float = 0.42
    doclayout_min_horizontal_relation: float = 0.08
    hybrid_keep_geometry_fallback: bool = True


@dataclass(frozen=True)
class LayoutElement:
    class_id: int
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def width(self) -> int:
        return max(1, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(1, self.y2 - self.y1)

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0


def resolve_doclayout_device(config: HybridPipelineConfig) -> str:
    requested = str(config.doclayout_device).strip().lower()

    if requested != "auto":
        return config.doclayout_device

    try:
        import torch
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def load_doclayout_model(config: HybridPipelineConfig):
    """
    Download the official DocStructBench checkpoint once (if needed)
    and load it locally.

    After the first successful Hugging Face download, inference is local.
    """
    if not config.use_doclayout_yolo:
        return None, None

    try:
        from doclayout_yolo import YOLOv10
    except Exception as error:
        raise RuntimeError(
            "DocLayout-YOLO is not installed. Run: pip install doclayout-yolo"
        ) from error

    checkpoint = config.doclayout_checkpoint_path

    if checkpoint:
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"DocLayout checkpoint not found: {checkpoint_path.resolve()}"
            )
        checkpoint = str(checkpoint_path)
    else:
        try:
            from huggingface_hub import hf_hub_download
        except Exception as error:
            raise RuntimeError(
                "huggingface_hub is not installed. Run: pip install huggingface_hub"
            ) from error

        checkpoint = hf_hub_download(
            repo_id=config.doclayout_repo_id,
            filename=config.doclayout_filename,
        )

    model = YOLOv10(checkpoint)

    print("DocLayout-YOLO loaded.")
    print("Checkpoint:", checkpoint)
    print("Device:", resolve_doclayout_device(config))

    return model, checkpoint


def _safe_numpy(value):
    if value is None:
        return np.array([])

    try:
        return value.detach().cpu().numpy()
    except Exception:
        try:
            return value.cpu().numpy()
        except Exception:
            return np.asarray(value)


def detect_doclayout_elements(
    model,
    prepared_color: np.ndarray,
    config: HybridPipelineConfig,
) -> list[LayoutElement]:
    """
    Run the pretrained model on the SAME prepared page used by OCR.
    Therefore model boxes and OCR-line boxes share one coordinate system.
    """
    if model is None:
        return []

    kwargs = {
        "imgsz": config.doclayout_imgsz,
        "conf": config.doclayout_confidence,
        "device": resolve_doclayout_device(config),
    }

    # Some DocLayout-YOLO/Ultralytics versions accept these, some older ones
    # may not. Try the richer call first, then a compatible fallback.
    try:
        prediction = model.predict(
            prepared_color,
            iou=config.doclayout_iou,
            verbose=False,
            **kwargs,
        )
    except TypeError:
        prediction = model.predict(
            prepared_color,
            **kwargs,
        )

    if not prediction:
        return []

    result = prediction[0]
    boxes = getattr(result, "boxes", None)

    if boxes is None:
        return []

    xyxy = _safe_numpy(getattr(boxes, "xyxy", None))
    confs = _safe_numpy(getattr(boxes, "conf", None))
    classes = _safe_numpy(getattr(boxes, "cls", None))

    names = getattr(result, "names", None)
    if names is None:
        names = getattr(model, "names", {})

    elements: list[LayoutElement] = []

    for index in range(len(xyxy)):
        coordinates = xyxy[index].tolist()
        if len(coordinates) < 4:
            continue

        class_id = int(classes[index]) if len(classes) > index else -1

        if isinstance(names, dict):
            class_name = str(names.get(class_id, class_id))
        elif isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
            class_name = str(names[class_id])
        else:
            class_name = str(class_id)

        confidence = float(confs[index]) if len(confs) > index else 0.0

        x1, y1, x2, y2 = [int(round(v)) for v in coordinates[:4]]

        if x2 <= x1 or y2 <= y1:
            continue

        elements.append(
            LayoutElement(
                class_id=class_id,
                class_name=class_name,
                confidence=confidence,
                x1=max(0, x1),
                y1=max(0, y1),
                x2=max(1, x2),
                y2=max(1, y2),
            )
        )

    return sorted(elements, key=lambda item: (item.y1, item.x1))


def rectangle_intersection_area(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> int:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])

    return max(0, x2 - x1) * max(0, y2 - y1)


def line_overlap_with_box(
    line: OCRLine,
    box: tuple[int, int, int, int],
) -> float:
    line_box = (line.x1, line.y1, line.x2, line.y2)
    intersection = rectangle_intersection_area(line_box, box)
    return intersection / max(1, bbox_area(line_box))


def lines_overlapping_element(
    lines: Sequence[OCRLine],
    element: LayoutElement,
    min_overlap: float,
) -> list[OCRLine]:
    selected = []

    for line in lines:
        overlap = line_overlap_with_box(line, element.bbox)

        center_inside = (
            element.x1 <= line.cx <= element.x2
            and element.y1 <= line.cy <= element.y2
        )

        if overlap >= min_overlap or center_inside:
            selected.append(line)

    return sorted(selected, key=lambda line: (line.y1, line.x1))


def _title_from_model_box(
    element: LayoutElement,
    lines: Sequence[OCRLine],
    document_type: str,
    config: HybridPipelineConfig,
) -> Optional[HeadlineAnchor]:
    if element.confidence < config.doclayout_min_title_confidence:
        return None

    title_lines = lines_overlapping_element(
        lines,
        element,
        config.doclayout_title_line_overlap,
    )

    title_lines = [
        line
        for line in title_lines
        if not is_noise_line(line.text, document_type)
    ]

    if not title_lines:
        return None

    # Model title boxes can contain multiple OCR rows.
    title_lines = sorted(title_lines, key=lambda line: (line.y1, line.x1))
    title_text = " ".join(line.text.strip() for line in title_lines)
    title_text = clean_text_basic(title_text)

    if len(title_text.split()) < 2:
        return None

    if is_masthead_or_metadata(title_text):
        return None

    # Avoid treating a long paragraph accidentally detected as title.
    if len(title_text.split()) > 28:
        return None

    return HeadlineAnchor(
        lines=title_lines,
        title=title_text,
        bbox=bbox_from_lines(title_lines),
        score=round(5.0 + 3.0 * float(element.confidence), 3),
    )


def anchors_from_doclayout(
    elements: Sequence[LayoutElement],
    lines: Sequence[OCRLine],
    document_type: str,
    content_area: tuple[int, int],
    config: HybridPipelineConfig,
) -> list[HeadlineAnchor]:
    content_start, content_end = content_area
    title_classes = {name.lower() for name in config.doclayout_title_classes}

    anchors: list[HeadlineAnchor] = []

    for element in elements:
        if element.class_name.lower() not in title_classes:
            continue

        if element.cy < content_start or element.cy > content_end:
            continue

        anchor = _title_from_model_box(
            element,
            lines,
            document_type,
            config,
        )

        if anchor is not None:
            anchors.append(anchor)

    return anchors


def fuse_headline_anchors(
    heuristic_anchors: Sequence[HeadlineAnchor],
    model_anchors: Sequence[HeadlineAnchor],
) -> list[HeadlineAnchor]:
    """
    Fuse OCR-geometry headline candidates and pretrained-model title boxes.
    A model title can recover headlines missed by geometry, while the
    geometry detector remains available when the pretrained model misses one.
    """
    combined = list(heuristic_anchors)

    for model_anchor in model_anchors:
        duplicate_index = None

        for index, existing in enumerate(combined):
            sim = title_similarity(model_anchor.title, existing.title)
            containment = bbox_containment(model_anchor.bbox, existing.bbox)
            iou = bbox_iou(model_anchor.bbox, existing.bbox)

            if sim >= 0.58 and (containment >= 0.35 or iou >= 0.20):
                duplicate_index = index
                break

        if duplicate_index is None:
            combined.append(model_anchor)
        else:
            existing = combined[duplicate_index]

            # Prefer the model-assisted title if it contains more headline words
            # without becoming paragraph-like.
            model_words = len(model_anchor.title.split())
            existing_words = len(existing.title.split())

            if 2 <= model_words <= 22 and model_words > existing_words:
                combined[duplicate_index] = model_anchor
            elif model_anchor.score > existing.score:
                combined[duplicate_index] = HeadlineAnchor(
                    lines=existing.lines,
                    title=existing.title,
                    bbox=existing.bbox,
                    score=max(existing.score, model_anchor.score),
                )

    return sorted(combined, key=lambda item: (item.bbox[1], item.bbox[0]))


def element_horizontal_relation(
    anchor_box: tuple[int, int, int, int],
    element_box: tuple[int, int, int, int],
) -> float:
    ax1, _, ax2, _ = anchor_box
    ex1, _, ex2, _ = element_box

    overlap = max(0, min(ax2, ex2) - max(ax1, ex1))
    min_width = max(1, min(ax2 - ax1, ex2 - ex1))

    overlap_ratio = overlap / min_width

    acx = (ax1 + ax2) / 2.0
    ecx = (ex1 + ex2) / 2.0
    center_distance = abs(acx - ecx)

    combined_width = max(1.0, (ax2 - ax1 + ex2 - ex1) / 2.0)
    center_score = max(0.0, 1.0 - center_distance / combined_width)

    return max(overlap_ratio, 0.65 * center_score)


def choose_anchor_for_layout_element(
    element: LayoutElement,
    anchors: Sequence[HeadlineAnchor],
    page_shape: tuple[int, int],
    body_height: float,
    config: HybridPipelineConfig,
) -> Optional[int]:
    """
    Assign a text/image/caption element to the most plausible headline.

    Preference:
    - headline is above or slightly overlapping the element,
    - strong horizontal relationship,
    - smaller vertical distance,
    - don't jump very far down the page.
    """
    page_h, _ = page_shape
    max_distance = config.doclayout_max_body_distance_ratio * page_h

    candidates: list[tuple[float, int]] = []

    for index, anchor in enumerate(anchors):
        # Layout block should normally be below the headline.
        if element.y2 < anchor.bbox[1]:
            continue

        vertical_gap = max(0.0, element.y1 - anchor.bbox[3])

        if vertical_gap > max_distance:
            continue

        horizontal_relation = element_horizontal_relation(
            anchor.bbox,
            element.bbox,
        )

        if horizontal_relation < config.doclayout_min_horizontal_relation:
            continue

        distance_score = 1.0 / (
            1.0 + vertical_gap / max(1.0, 4.0 * body_height)
        )

        # The nearer valid headline should normally win.
        score = (
            1.65 * horizontal_relation
            + 1.25 * distance_score
            + 0.10 * min(1.0, anchor.score / 8.0)
        )

        # If another headline lies vertically between this anchor and element
        # with similar horizontal relationship, penalize the older anchor.
        for other_index, other in enumerate(anchors):
            if other_index == index:
                continue

            if (
                anchor.bbox[3] < other.bbox[1] < element.y1
                and element_horizontal_relation(other.bbox, element.bbox)
                >= horizontal_relation * 0.75
            ):
                score -= 1.10

        candidates.append((score, index))

    if not candidates:
        return None

    candidates.sort(reverse=True)
    best_score, best_index = candidates[0]

    if best_score < 0.60:
        return None

    return best_index


def build_doclayout_assisted_regions(
    lines: Sequence[OCRLine],
    anchors: Sequence[HeadlineAnchor],
    elements: Sequence[LayoutElement],
    page_shape: tuple[int, int],
    body_height: float,
    document_type: str,
    config: HybridPipelineConfig,
) -> list[dict[str, Any]]:
    if not anchors or not elements:
        return []

    text_classes = {name.lower() for name in config.doclayout_text_classes}
    caption_classes = {name.lower() for name in config.doclayout_caption_classes}
    figure_classes = {name.lower() for name in config.doclayout_figure_classes}

    useful_elements = [
        element
        for element in elements
        if (
            element.class_name.lower() in text_classes
            or element.class_name.lower() in caption_classes
            or element.class_name.lower() in figure_classes
        )
    ]

    assigned: dict[int, list[LayoutElement]] = {
        index: [] for index in range(len(anchors))
    }

    for element in useful_elements:
        min_conf = (
            config.doclayout_min_text_confidence
            if element.class_name.lower() in text_classes
            else config.doclayout_confidence
        )

        if element.confidence < min_conf:
            continue

        index = choose_anchor_for_layout_element(
            element,
            anchors,
            page_shape,
            body_height,
            config,
        )

        if index is not None:
            assigned[index].append(element)

    regions: list[dict[str, Any]] = []

    for index, anchor in enumerate(anchors):
        article_elements = assigned.get(index, [])
        region_lines = list(anchor.lines)

        for element in article_elements:
            if (
                element.class_name.lower() in text_classes
                or element.class_name.lower() in caption_classes
            ):
                overlap_threshold = (
                    config.doclayout_text_line_overlap
                    if element.class_name.lower() in text_classes
                    else 0.14
                )

                region_lines.extend(
                    lines_overlapping_element(
                        lines,
                        element,
                        overlap_threshold,
                    )
                )

        # Deduplicate OCR lines.
        region_lines = list(
            {line.uid: line for line in region_lines}.values()
        )

        if len(region_lines) <= len(anchor.lines):
            continue

        text = article_lines_to_text(
            region_lines,
            anchor,
            document_type,
            body_height,
        )

        word_count = len(text.split())

        if word_count < config.min_article_words:
            continue

        if word_count > config.max_article_words:
            text = " ".join(text.split()[: config.max_article_words])

        boxes = [anchor.bbox]
        boxes.extend(element.bbox for element in article_elements)

        x1 = min(box[0] for box in boxes)
        y1 = min(box[1] for box in boxes)
        x2 = max(box[2] for box in boxes)
        y2 = max(box[3] for box in boxes)

        # OCR lines are the authoritative text bounds.
        ocr_bbox = bbox_from_lines(region_lines)
        region_bbox = (
            min(x1, ocr_bbox[0]),
            min(y1, ocr_bbox[1]),
            max(x2, ocr_bbox[2]),
            max(y2, ocr_bbox[3]),
        )

        title_support = any(
            element.class_name.lower() in {
                name.lower() for name in config.doclayout_title_classes
            }
            and bbox_iou(element.bbox, anchor.bbox) > 0.05
            for element in elements
        )

        layout_confidence = min(
            1.0,
            0.42
            + min(0.22, 0.025 * len(article_elements))
            + min(0.20, word_count / 500.0)
            + (0.10 if title_support else 0.0),
        )

        warnings = []

        if word_count < 45:
            warnings.append(
                "DocLayout-assisted article contains limited text."
            )

        if ocr_quality_score(text) < 0.45:
            warnings.append("OCR quality is low.")

        regions.append(
            {
                "anchor": anchor,
                "lines": region_lines,
                "bbox": region_bbox,
                "text": text,
                "layout_confidence": round(layout_confidence, 3),
                "warnings": warnings,
                "source": "doclayout_yolo_custom_article_grouping",
                "layout_elements": [
                    {
                        "class_name": element.class_name,
                        "confidence": round(element.confidence, 4),
                        "bbox": element.bbox,
                    }
                    for element in article_elements
                ],
            }
        )

    return regions


def _region_quality_for_fusion(region: dict[str, Any]) -> float:
    words = len(region.get("text", "").split())
    layout = float(region.get("layout_confidence", 0.0))

    text_bonus = min(0.35, words / 600.0)

    # Reward complete-enough regions but avoid rewarding huge mixed boxes.
    if words > 1200:
        text_bonus -= 0.12

    source_bonus = (
        0.06
        if region.get("source") == "doclayout_yolo_custom_article_grouping"
        else 0.0
    )

    return layout + text_bonus + source_bonus


def fuse_article_regions(
    geometry_regions: Sequence[dict[str, Any]],
    doclayout_regions: Sequence[dict[str, Any]],
    config: HybridPipelineConfig,
) -> list[dict[str, Any]]:
    """
    Fuse model-assisted and geometry-assisted candidate articles.

    The pretrained detector is not trusted blindly. For overlapping candidates
    with the same/similar headline, keep the candidate with better balance of
    layout confidence and useful article length.
    """
    combined = list(doclayout_regions)

    if config.hybrid_keep_geometry_fallback:
        combined.extend(geometry_regions)

    if not combined:
        return []

    ordered = sorted(
        combined,
        key=lambda item: (item["bbox"][1], item["bbox"][0]),
    )

    kept: list[dict[str, Any]] = []

    for region in ordered:
        duplicate_index = None

        for index, existing in enumerate(kept):
            sim = title_similarity(
                region["anchor"].title,
                existing["anchor"].title,
            )
            containment = bbox_containment(
                region["bbox"],
                existing["bbox"],
            )
            iou = bbox_iou(
                region["bbox"],
                existing["bbox"],
            )

            ids_a = {line.uid for line in region.get("lines", [])}
            ids_b = {line.uid for line in existing.get("lines", [])}
            shared = len(ids_a & ids_b) / max(
                1,
                min(len(ids_a), len(ids_b)),
            )

            if (
                sim >= 0.56
                and (
                    containment >= 0.45
                    or iou >= 0.28
                    or shared >= 0.38
                )
            ):
                duplicate_index = index
                break

        if duplicate_index is None:
            kept.append(region)
        else:
            existing = kept[duplicate_index]

            if _region_quality_for_fusion(region) > _region_quality_for_fusion(existing):
                kept[duplicate_index] = region

    # Second pass using the existing robust duplicate logic.
    kept = deduplicate_and_merge_regions(kept, config)

    return sorted(
        kept,
        key=lambda item: (item["bbox"][1], item["bbox"][0]),
    )


def detect_hybrid_article_regions(
    lines: Sequence[OCRLine],
    prepared_gray: np.ndarray,
    prepared_color: np.ndarray,
    document_type: str,
    config: HybridPipelineConfig,
    doclayout_model=None,
) -> dict[str, Any]:
    """
    Final local article detector:
      A) OpenCV/Tesseract adaptive geometry
      B) DocLayout-YOLO structural hints
      C) custom fusion/grouping
    """
    page_shape = prepared_gray.shape[:2]
    separators = detect_page_separators(prepared_gray)
    body_height = estimate_body_height(lines, page_shape)

    content_area = detect_adaptive_content_area(
        lines,
        separators,
        page_shape,
        body_height,
        config,
    )

    usable_lines = [
        line
        for line in lines
        if content_area[0] <= line.cy <= content_area[1]
        and not is_noise_line(line.text, document_type)
    ]

    # A) Existing geometry headline detector.
    candidates = detect_headline_candidates(
        usable_lines,
        body_height,
        page_shape,
        content_area,
        config,
    )

    geometry_anchors = combine_multiline_headlines(
        candidates,
        usable_lines,
        body_height,
        page_shape,
    )

    geometry_anchors = merge_short_headline_continuations(
        geometry_anchors,
        body_height,
    )

    # B) Local pretrained model.
    layout_elements: list[LayoutElement] = []
    model_anchors: list[HeadlineAnchor] = []
    model_warning = None

    if config.use_doclayout_yolo and doclayout_model is not None:
        try:
            layout_elements = detect_doclayout_elements(
                doclayout_model,
                prepared_color,
                config,
            )

            model_anchors = anchors_from_doclayout(
                layout_elements,
                usable_lines,
                document_type,
                content_area,
                config,
            )
        except Exception as error:
            model_warning = (
                "DocLayout-YOLO inference failed. "
                f"Geometry fallback used. Reason: {error}"
            )

    anchors = fuse_headline_anchors(
        geometry_anchors,
        model_anchors,
    )

    # Geometry candidates using the fused headline list.
    geometry_regions = build_provisional_article_regions(
        usable_lines,
        anchors,
        separators,
        content_area,
        page_shape,
        body_height,
        config,
    )

    for region in geometry_regions:
        region["source"] = "adaptive_headline_separator_geometry"

    geometry_regions = deduplicate_and_merge_regions(
        geometry_regions,
        config,
    )

    # Model-assisted candidates use title/text/figure/caption elements.
    model_regions = build_doclayout_assisted_regions(
        usable_lines,
        anchors,
        layout_elements,
        page_shape,
        body_height,
        document_type,
        config,
    )

    final_regions = fuse_article_regions(
        geometry_regions,
        model_regions,
        config,
    )

    warnings: list[str] = []

    if model_warning:
        warnings.append(model_warning)

    if not geometry_anchors and not model_anchors:
        warnings.append(
            "No reliable headline anchors were detected."
        )

    if not final_regions:
        warnings.append(
            "Hybrid article segmentation failed; "
            "the full-page fallback will be used."
        )

    if len(final_regions) > 24:
        warnings.append(
            "High article count suggests possible over-segmentation."
        )

    return {
        "regions": final_regions,
        "anchors": anchors,
        "geometry_anchors": geometry_anchors,
        "doclayout_anchors": model_anchors,
        "layout_elements": layout_elements,
        "separators": separators,
        "body_height": body_height,
        "content_area": content_area,
        "warnings": warnings,
        "debug_counts": {
            "geometry_anchor_count": len(geometry_anchors),
            "doclayout_anchor_count": len(model_anchors),
            "fused_anchor_count": len(anchors),
            "doclayout_element_count": len(layout_elements),
            "geometry_region_count": len(geometry_regions),
            "doclayout_region_count": len(model_regions),
            "final_region_count": len(final_regions),
        },
    }


# ============================================================
# 18. HYBRID FULL ANALYSIS
# ============================================================


def analyze_content_hybrid(
    image_path: str,
    document_type: str,
    model,
    config: HybridPipelineConfig,
    doclayout_model=None,
) -> dict[str, Any]:
    """
    Full research pipeline analysis.

    For Newspaper/Magazine:
      DocLayout-YOLO + Tesseract/OpenCV geometry + custom grouping.

    For Novel/Story/Textbook/Report/Letter/General Document:
      full-page local OCR + category model.
    """
    normalized_type = normalize_document_type(document_type)

    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(
            f"Image not found or unreadable: {image_path}"
        )

    prepared_color, prepared_gray, scale = prepare_page_image(
        image,
        config,
    )

    page_shape = prepared_gray.shape[:2]
    warnings: list[str] = []
    layout_debug: dict[str, Any] = {}

    if normalized_type in MULTI_CATEGORY_TYPES:
        ocr_lines = extract_ocr_lines(
            prepared_gray,
            normalized_type,
            config,
        )

        layout = detect_hybrid_article_regions(
            ocr_lines,
            prepared_gray,
            prepared_color,
            normalized_type,
            config,
            doclayout_model=doclayout_model,
        )

        warnings.extend(layout["warnings"])

        layout_debug = {
            "ocr_line_count": len(ocr_lines),
            "body_height": round(
                float(layout["body_height"]),
                2,
            ),
            "content_area": layout["content_area"],
            **layout["debug_counts"],
            "headline_anchors": [
                {
                    "title": anchor.title,
                    "bbox": anchor.bbox,
                    "score": round(
                        float(anchor.score),
                        3,
                    ),
                }
                for anchor in layout["anchors"]
            ],
            "doclayout_elements": [
                {
                    "class_name": element.class_name,
                    "confidence": round(
                        float(element.confidence),
                        4,
                    ),
                    "bbox": element.bbox,
                }
                for element in layout["layout_elements"]
            ],
            "separators": [
                asdict(separator)
                for separator in layout["separators"]
            ],
        }

        results: list[ArticleRegion] = []

        for number, region in enumerate(
            layout["regions"],
            start=1,
        ):
            anchor = region["anchor"]

            prediction = predict_category(
                model,
                region["text"],
                normalized_type,
                anchor.title,
                config,
            )

            results.append(
                ArticleRegion(
                    article_number=number,
                    title=anchor.title,
                    bbox=region["bbox"],
                    normalized_bbox=normalize_bbox(
                        region["bbox"],
                        page_shape,
                    ),
                    full_text=region["text"],
                    category=prediction["category"],
                    confidence=prediction["confidence"],
                    probability_margin=prediction["margin"],
                    top_predictions=prediction["top_predictions"],
                    ocr_quality=ocr_quality_score(
                        region["text"]
                    ),
                    layout_confidence=float(
                        region["layout_confidence"]
                    ),
                    warnings=list(
                        region.get("warnings", [])
                    ),
                    source=region.get(
                        "source",
                        "hybrid_doclayout_geometry",
                    ),
                )
            )

        if not results:
            full_text = best_full_page_ocr(
                prepared_gray,
                normalized_type,
            )

            prediction = predict_category(
                model,
                full_text,
                normalized_type,
                "Full page",
                config,
            )

            results = [
                ArticleRegion(
                    article_number=1,
                    title="Full page",
                    bbox=(
                        0,
                        0,
                        page_shape[1],
                        page_shape[0],
                    ),
                    normalized_bbox=(
                        0,
                        0,
                        1000,
                        1000,
                    ),
                    full_text=full_text,
                    category=prediction["category"],
                    confidence=prediction["confidence"],
                    probability_margin=prediction["margin"],
                    top_predictions=prediction["top_predictions"],
                    ocr_quality=ocr_quality_score(
                        full_text
                    ),
                    layout_confidence=0.20,
                    warnings=[
                        "Hybrid article segmentation fallback: "
                        "the full page was processed as one region."
                    ],
                    source="full_page_fallback",
                )
            ]

    else:
        full_text = best_full_page_ocr(
            prepared_gray,
            normalized_type,
        )

        title = "Full page"

        prediction = predict_category(
            model,
            full_text,
            normalized_type,
            title,
            config,
        )

        results = [
            ArticleRegion(
                article_number=1,
                title=title,
                bbox=(
                    0,
                    0,
                    page_shape[1],
                    page_shape[0],
                ),
                normalized_bbox=(
                    0,
                    0,
                    1000,
                    1000,
                ),
                full_text=full_text,
                category=prediction["category"],
                confidence=prediction["confidence"],
                probability_margin=prediction["margin"],
                top_predictions=prediction["top_predictions"],
                ocr_quality=ocr_quality_score(full_text),
                layout_confidence=1.0,
                warnings=[],
                source="full_page_tesseract",
            )
        ]

    category_counts = Counter(
        result.category
        for result in results
    )

    categories = sorted(category_counts)

    document_mode = (
        "Multi Category"
        if normalized_type in MULTI_CATEGORY_TYPES
        else "Single Category"
    )

    return {
        "status": "ANALYZED",
        "document_type_received": document_type,
        "document_type": normalized_type,
        "mode": document_mode,
        "detected_categories": categories,
        "category_counts": dict(category_counts),
        "needs_category_selection": (
            normalized_type in MULTI_CATEGORY_TYPES
            and len(categories) > 1
        ),
        "image_shape": {
            "height": page_shape[0],
            "width": page_shape[1],
        },
        "resize_scale": scale,
        "warnings": warnings,
        "layout_debug": layout_debug,
        "results": [
            asdict(result)
            for result in results
        ],
        "processing": {
            "cloud_api_used": False,
            "ocr": "local_tesseract",
            "pretrained_layout_model": (
                "local_DocLayout_YOLO_DocStructBench"
                if (
                    config.use_doclayout_yolo
                    and doclayout_model is not None
                )
                else "disabled"
            ),
            "article_detection": (
                "hybrid_doclayout_yolo_plus_custom_geometry_grouping"
            ),
            "category_model": (
                "local_tfidf_logistic_regression"
            ),
            "speech_input": "local_vosk",
            "summary_model": (
                f"local_ollama_{config.ollama_model}"
            ),
        },
    }


# ============================================================
# 19. HYBRID VISUALIZATION
# ============================================================


def show_doclayout_elements(
    image_path: str,
    analysis: dict[str, Any],
) -> None:
    """
    Visualize the pretrained model structural boxes stored in layout_debug.
    """
    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(image_path)

    target_shape = analysis.get(
        "image_shape",
        {
            "height": image.shape[0],
            "width": image.shape[1],
        },
    )

    target_w = int(target_shape["width"])
    target_h = int(target_shape["height"])

    display = cv2.resize(
        image,
        (target_w, target_h),
        interpolation=cv2.INTER_CUBIC,
    )

    elements = analysis.get(
        "layout_debug",
        {},
    ).get(
        "doclayout_elements",
        [],
    )

    for index, element in enumerate(
        elements,
        start=1,
    ):
        x1, y1, x2, y2 = [
            int(value)
            for value in element["bbox"]
        ]

        cv2.rectangle(
            display,
            (x1, y1),
            (x2, y2),
            (0, 180, 255),
            2,
        )

        label = (
            f"{index} "
            f"{element['class_name']} "
            f"{element['confidence']:.2f}"
        )

        cv2.putText(
            display,
            label[:45],
            (
                x1,
                max(18, y1 - 5),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            display,
            label[:45],
            (
                x1,
                max(18, y1 - 5),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    plt.figure(figsize=(14, 18))
    plt.imshow(
        cv2.cvtColor(
            display,
            cv2.COLOR_BGR2RGB,
        )
    )
    plt.axis("off")
    plt.title(
        "DocLayout-YOLO Structural Elements"
    )
    plt.show()


def hybrid_debug_summary(
    analysis: dict[str, Any],
) -> pd.DataFrame:
    debug = analysis.get(
        "layout_debug",
        {},
    )

    rows = [
        (
            "OCR lines",
            debug.get("ocr_line_count", 0),
        ),
        (
            "Geometry headlines",
            debug.get(
                "geometry_anchor_count",
                0,
            ),
        ),
        (
            "DocLayout headlines",
            debug.get(
                "doclayout_anchor_count",
                0,
            ),
        ),
        (
            "Fused headlines",
            debug.get(
                "fused_anchor_count",
                0,
            ),
        ),
        (
            "DocLayout elements",
            debug.get(
                "doclayout_element_count",
                0,
            ),
        ),
        (
            "Geometry regions",
            debug.get(
                "geometry_region_count",
                0,
            ),
        ),
        (
            "DocLayout-assisted regions",
            debug.get(
                "doclayout_region_count",
                0,
            ),
        ),
        (
            "Final fused articles",
            debug.get(
                "final_region_count",
                len(analysis.get("results", [])),
            ),
        ),
    ]

    return pd.DataFrame(
        rows,
        columns=[
            "Stage",
            "Count",
        ],
    )


# ============================================================
# 20. ONE-CALL FULL HYBRID + MICROPHONE FLOW
# ============================================================


def run_hybrid_voice_pipeline(
    image_path: str,
    document_type: str,
    category_model,
    config: HybridPipelineConfig,
    doclayout_model,
    voice_selector: OfflineVoiceSelector,
    tts_enabled: bool = False,
    clean_full_text_with_llama: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Complete notebook/device flow:
      document type ->
      hybrid article detection ->
      category detection ->
      microphone category choice ->
      microphone summary/full choice ->
      local Llama ->
      structured payload.
    """
    analysis = analyze_content_hybrid(
        image_path=image_path,
        document_type=document_type,
        model=category_model,
        config=config,
        doclayout_model=doclayout_model,
    )

    selected_category, depth = collect_user_preferences_voice(
        analysis=analysis,
        voice_selector=voice_selector,
        tts_enabled=tts_enabled,
    )

    output = generate_selected_output(
        analysis=analysis,
        selected_category=selected_category,
        depth=depth,
        config=config,
        clean_full_text_with_llama=clean_full_text_with_llama,
    )

    return analysis, output


# =====================================================================
# 21. V3 FIXES — DOCLAYOUT ARTICLE GROUPING, CLEANING, VOICE, SUMMARY
# =====================================================================
#
# These definitions intentionally appear at the END of this module.
# Python resolves function names at runtime, so the V3 definitions below
# replace the earlier implementations without duplicating the entire base
# pipeline architecture.
#
# Main fixes:
# - normalize DocLayout class labels ("plain text" == "plain_text")
# - reject bylines/body fragments from headline anchors
# - remove section-only headings when child article headlines are present
# - actually use DocLayout plain-text / caption / figure blocks
# - bridge headline -> figure -> article body
# - stronger hybrid fusion and fragment cleanup
# - stronger OCR page-marker cleanup
# - strict, grounded local Llama summaries
# - summary quality gate
# - Vosk aliases: read/read it/read all => full
# =====================================================================


@dataclass
class HybridPipelineConfigV3(HybridPipelineConfig):
    # Headline filtering
    reject_lowercase_body_as_headline: bool = True
    section_heading_child_count: int = 2
    section_heading_search_ratio: float = 0.24

    # DocLayout association
    doclayout_min_text_confidence: float = 0.15
    doclayout_text_line_overlap: float = 0.12
    doclayout_min_horizontal_relation: float = 0.06
    doclayout_max_body_distance_ratio: float = 0.48
    doclayout_figure_bridge_bonus: float = 0.90
    doclayout_intervening_headline_penalty: float = 1.50

    # Final region quality
    min_summary_words_for_llama: int = 55
    min_summary_ocr_quality: float = 0.45
    min_summary_layout_confidence: float = 0.50

    # Grounding / factual safety
    summary_grounding_min_sentence_overlap: float = 0.14
    summary_max_unsupported_number_count: int = 0

    # Debug
    print_doclayout_class_counts: bool = True


# ---------------------------------------------------------------------
# 21.1 DOCLAYOUT CLASS NORMALIZATION
# ---------------------------------------------------------------------


def normalize_layout_class_name(name: Any) -> str:
    """
    Normalize different DocLayout label spellings:
        'plain text' -> 'plain_text'
        'plain-text' -> 'plain_text'
        'Plain_Text' -> 'plain_text'
    """
    value = str(name).strip().lower()
    value = value.replace("-", "_").replace(" ", "_")
    value = re.sub(r"_+", "_", value)

    aliases = {
        "plaintext": "plain_text",
        "plain_text": "plain_text",
        "figurecaption": "figure_caption",
        "figure_caption": "figure_caption",
        "tablecaption": "table_caption",
        "table_caption": "table_caption",
        "tablefootnote": "table_footnote",
        "table_footnote": "table_footnote",
        "isolateformula": "isolate_formula",
        "isolate_formula": "isolate_formula",
        "formulacaption": "formula_caption",
        "formula_caption": "formula_caption",
    }
    return aliases.get(value, value)


def normalized_layout_class_set(values: Sequence[str]) -> set[str]:
    return {normalize_layout_class_name(value) for value in values}


def doclayout_class_counts(elements: Sequence[LayoutElement]) -> dict[str, int]:
    counts = Counter(
        normalize_layout_class_name(element.class_name)
        for element in elements
    )
    return dict(sorted(counts.items()))


# ---------------------------------------------------------------------
# 21.2 OCR / TEXT CLEANING
# ---------------------------------------------------------------------


def remove_newspaper_artifacts(text: str) -> str:
    value = clean_text_basic(text)

    # Inline continuation/page markers.
    patterns = [
        r"\b(?:I?TO|T0)\s+PAGE\s*[0-9O]*\b",
        r"\bPAGE\s*[0-9O]{1,3}\b",
        r"\bCONTINUED\s+ON\s+PAGE\s*[0-9O]{1,3}\b",
        r"\bFROM\s+PAGE\s*[0-9O]{1,3}\b",
    ]
    for pattern in patterns:
        value = re.sub(pattern, " ", value, flags=re.IGNORECASE)

    # Common isolated print/figure markers.
    value = re.sub(r"(?<![A-Za-z0-9])\d+[A-Z](?![A-Za-z0-9])", " ", value)

    # Reconnect OCR line-break hyphenation after geometry reading order.
    value = re.sub(r"([A-Za-z])-\s+([a-z])", r"\1\2", value)

    # Clean punctuation/spacing without destroying meaningful punctuation.
    value = re.sub(r"\s+([,.;:!?])", r"\1", value)
    value = re.sub(r"\s+", " ", value).strip()

    return value


def clean_ocr_lines(lines: Iterable[str], document_type: str) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()

    for raw in lines:
        line = re.sub(r"\s+", " ", clean_text_basic(raw)).strip()

        # Remove inline page markers before deciding whether line is useful.
        line = re.sub(
            r"\b(?:I?TO|T0)\s+PAGE\s*[0-9O]*\b",
            " ",
            line,
            flags=re.IGNORECASE,
        )
        line = re.sub(
            r"\bPAGE\s*[0-9O]{1,3}\b",
            " ",
            line,
            flags=re.IGNORECASE,
        )
        line = re.sub(r"\s+", " ", line).strip(" -:;|")

        if not line:
            continue
        if is_noise_line(line, document_type):
            continue

        key = re.sub(r"\W+", "", line.lower())
        if not key or key in seen:
            continue

        seen.add(key)
        output.append(line)

    return output


# ---------------------------------------------------------------------
# 21.3 HEADLINE / BYLINE FILTERING
# ---------------------------------------------------------------------


BYLINE_NON_NAME_WORDS = {
    "treasury", "secretary", "court", "president", "minister",
    "cricket", "team", "may", "day", "oil", "price", "country",
    "funds", "speaker", "rallies", "programme", "judgement",
    "economic", "working", "environment", "official", "cyber",
}


def looks_like_byline(text: str) -> bool:
    clean = clean_text_basic(text)
    low = clean.lower()
    words = re.findall(r"[A-Za-z]+", clean)

    if not 1 <= len(words) <= 7:
        return False

    if low.startswith(
        ("by ", "reporting by ", "compiled by ", "pix by ", "photo by ")
    ):
        return True

    # Author lists sometimes end in "and" because OCR splits the next name.
    if low.endswith(" and") and 2 <= len(words) <= 7:
        return True

    lowered_words = {word.lower() for word in words}
    if lowered_words & BYLINE_NON_NAME_WORDS:
        return False

    # Mostly/all-capital short personal-name line.
    if uppercase_ratio(clean) >= 0.84 and len(words) <= 6:
        # Require name-like words instead of a one-word section label.
        name_like = sum(
            1
            for word in words
            if len(word) >= 2 and word.isalpha()
        )
        return name_like >= 2

    return False


def headline_text_is_valid(text: str, config: PipelineConfig) -> bool:
    title = clean_text_basic(text).strip()
    words = re.findall(r"[A-Za-z0-9$%]+", title)

    if len(words) < 2 or len(words) > 28:
        return False

    if is_masthead_or_metadata(title):
        return False

    if looks_like_byline(title):
        return False

    if alpha_ratio(title) < 0.38:
        return False

    # Body fragment accidentally classified as a title.
    first_alpha = next((character for character in title if character.isalpha()), "")
    if (
        getattr(config, "reject_lowercase_body_as_headline", True)
        and first_alpha
        and first_alpha.islower()
        and len(words) >= 4
    ):
        return False

    # Common sentence/body ending.
    if (
        len(words) >= 8
        and title.endswith((".", ",", ";"))
    ):
        return False

    return True


def _anchor_horizontal_overlap(
    first: HeadlineAnchor,
    second: HeadlineAnchor,
) -> float:
    return horizontal_overlap_boxes(first.bbox, second.bbox)


def probable_section_heading(
    anchor: HeadlineAnchor,
    anchors: Sequence[HeadlineAnchor],
    page_shape: tuple[int, int],
    config: PipelineConfig,
) -> bool:
    """
    Suppress a section label such as "May Day Messages" only when it has
    multiple smaller child headlines directly below it in the same zone.
    """
    words = anchor.title.split()

    if not 2 <= len(words) <= 5:
        return False

    page_h, _ = page_shape
    search_bottom = anchor.bbox[3] + int(
        getattr(config, "section_heading_search_ratio", 0.24) * page_h
    )

    children = []
    for other in anchors:
        if other is anchor:
            continue

        if not (
            anchor.bbox[3] < other.bbox[1] <= search_bottom
        ):
            continue

        overlap = _anchor_horizontal_overlap(anchor, other)
        center_inside = (
            anchor.bbox[0] - 20
            <= (other.bbox[0] + other.bbox[2]) / 2
            <= anchor.bbox[2] + 20
        )

        if overlap >= 0.30 or center_inside:
            # Child should look like a real article headline.
            if len(other.title.split()) >= 4:
                children.append(other)

    return len(children) >= getattr(
        config,
        "section_heading_child_count",
        2,
    )


def filter_headline_anchors_v3(
    anchors: Sequence[HeadlineAnchor],
    page_shape: tuple[int, int],
    config: PipelineConfig,
) -> list[HeadlineAnchor]:
    valid = [
        anchor
        for anchor in anchors
        if headline_text_is_valid(anchor.title, config)
    ]

    # Deduplicate near-identical model/geometry titles.
    valid = sorted(valid, key=lambda a: (a.bbox[1], a.bbox[0]))
    deduplicated: list[HeadlineAnchor] = []

    for anchor in valid:
        duplicate_index = None

        for index, existing in enumerate(deduplicated):
            similarity = title_similarity(anchor.title, existing.title)
            iou = bbox_iou(anchor.bbox, existing.bbox)
            containment = bbox_containment(anchor.bbox, existing.bbox)

            if (
                similarity >= 0.62
                and (iou >= 0.10 or containment >= 0.28)
            ):
                duplicate_index = index
                break

        if duplicate_index is None:
            deduplicated.append(anchor)
        else:
            existing = deduplicated[duplicate_index]

            # Prefer the more complete title, unless it becomes paragraph-like.
            current_words = len(anchor.title.split())
            existing_words = len(existing.title.split())

            if (
                2 <= current_words <= 22
                and current_words > existing_words
            ):
                deduplicated[duplicate_index] = anchor
            elif anchor.score > existing.score:
                deduplicated[duplicate_index] = anchor

    # Section heading suppression must run AFTER deduplication.
    result = [
        anchor
        for anchor in deduplicated
        if not probable_section_heading(
            anchor,
            deduplicated,
            page_shape,
            config,
        )
    ]

    return sorted(result, key=lambda a: (a.bbox[1], a.bbox[0]))


# ---------------------------------------------------------------------
# 21.4 DOCLAYOUT ANCHORS WITH NORMALIZED CLASSES
# ---------------------------------------------------------------------


def _title_from_model_box(
    element: LayoutElement,
    lines: Sequence[OCRLine],
    document_type: str,
    config: HybridPipelineConfig,
) -> Optional[HeadlineAnchor]:
    if element.confidence < config.doclayout_min_title_confidence:
        return None

    title_lines = lines_overlapping_element(
        lines,
        element,
        config.doclayout_title_line_overlap,
    )

    title_lines = [
        line
        for line in title_lines
        if not is_noise_line(line.text, document_type)
    ]

    if not title_lines:
        return None

    title_lines = sorted(
        {line.uid: line for line in title_lines}.values(),
        key=lambda line: (line.y1, line.x1),
    )

    title_text = remove_newspaper_artifacts(
        " ".join(line.text.strip() for line in title_lines)
    )

    if not headline_text_is_valid(title_text, config):
        return None

    return HeadlineAnchor(
        lines=title_lines,
        title=title_text,
        bbox=bbox_from_lines(title_lines),
        score=round(
            5.0 + 3.0 * float(element.confidence),
            3,
        ),
    )


def anchors_from_doclayout(
    elements: Sequence[LayoutElement],
    lines: Sequence[OCRLine],
    document_type: str,
    content_area: tuple[int, int],
    config: HybridPipelineConfig,
) -> list[HeadlineAnchor]:
    content_start, content_end = content_area
    title_classes = normalized_layout_class_set(
        config.doclayout_title_classes
    )

    anchors: list[HeadlineAnchor] = []

    for element in elements:
        element_class = normalize_layout_class_name(
            element.class_name
        )

        if element_class not in title_classes:
            continue

        if element.cy < content_start or element.cy > content_end:
            continue

        anchor = _title_from_model_box(
            element,
            lines,
            document_type,
            config,
        )

        if anchor is not None:
            anchors.append(anchor)

    return anchors


# ---------------------------------------------------------------------
# 21.5 DOCLAYOUT BODY ASSOCIATION
# ---------------------------------------------------------------------


def separator_blocks_relationship(
    anchor: HeadlineAnchor,
    element: LayoutElement,
    separators: Sequence[Separator],
) -> bool:
    """
    True only for a strong separator that clearly lies between the
    headline and candidate element.
    """
    anchor_cx = (anchor.bbox[0] + anchor.bbox[2]) / 2.0
    element_cx = element.cx

    left_x = min(anchor_cx, element_cx)
    right_x = max(anchor_cx, element_cx)

    corridor_top = min(anchor.bbox[1], element.y1)
    corridor_bottom = max(anchor.bbox[3], element.y2)
    corridor_height = max(1, corridor_bottom - corridor_top)

    for separator in separators:
        if separator.orientation != "vertical":
            continue

        if not (left_x < separator.x < right_x):
            continue

        overlap_top = max(corridor_top, separator.y1)
        overlap_bottom = min(corridor_bottom, separator.y2)
        overlap = max(0, overlap_bottom - overlap_top)

        if overlap / corridor_height >= 0.55 and separator.strength >= 0.35:
            return True

    return False


def _intervening_headline_exists(
    anchor_index: int,
    element: LayoutElement,
    anchors: Sequence[HeadlineAnchor],
    relation: float,
) -> bool:
    anchor = anchors[anchor_index]

    for other_index, other in enumerate(anchors):
        if other_index == anchor_index:
            continue

        if not (
            anchor.bbox[3] + 2
            < other.bbox[1]
            < element.y1
        ):
            continue

        other_relation = element_horizontal_relation(
            other.bbox,
            element.bbox,
        )

        # A newer headline between the old headline and text block is a
        # strong signal that the text belongs to the newer article.
        if other_relation >= max(0.12, relation * 0.62):
            return True

    return False


def _figure_bridges_anchor_to_text(
    anchor: HeadlineAnchor,
    text_element: LayoutElement,
    figure_elements: Sequence[LayoutElement],
) -> bool:
    for figure in figure_elements:
        if not (
            anchor.bbox[3] <= figure.y1
            and figure.y2 <= text_element.y2 + 30
        ):
            continue

        if not (
            figure.y1 <= text_element.y1
            or figure.y2 <= text_element.y1 + 40
        ):
            continue

        anchor_to_figure = element_horizontal_relation(
            anchor.bbox,
            figure.bbox,
        )
        figure_to_text = element_horizontal_relation(
            figure.bbox,
            text_element.bbox,
        )

        if anchor_to_figure >= 0.14 and figure_to_text >= 0.14:
            return True

    return False


def choose_anchor_for_layout_element_v3(
    element: LayoutElement,
    anchors: Sequence[HeadlineAnchor],
    page_shape: tuple[int, int],
    body_height: float,
    config: HybridPipelineConfig,
    separators: Sequence[Separator],
    figures_by_anchor: Optional[dict[int, list[LayoutElement]]] = None,
) -> Optional[int]:
    page_h, _ = page_shape
    max_distance = (
        config.doclayout_max_body_distance_ratio
        * page_h
    )

    candidates: list[tuple[float, int]] = []

    for index, anchor in enumerate(anchors):
        # Body/caption/figure should not sit clearly above the headline.
        if element.y2 < anchor.bbox[1] - body_height:
            continue

        vertical_gap = max(
            0.0,
            element.y1 - anchor.bbox[3],
        )

        if vertical_gap > max_distance:
            continue

        horizontal_relation = element_horizontal_relation(
            anchor.bbox,
            element.bbox,
        )

        if (
            horizontal_relation
            < config.doclayout_min_horizontal_relation
        ):
            continue

        if separator_blocks_relationship(
            anchor,
            element,
            separators,
        ):
            continue

        # Reject old headline when a closer competing headline lies between.
        if _intervening_headline_exists(
            index,
            element,
            anchors,
            horizontal_relation,
        ):
            continue

        distance_score = 1.0 / (
            1.0
            + vertical_gap
            / max(1.0, 5.0 * body_height)
        )

        score = (
            1.70 * horizontal_relation
            + 1.35 * distance_score
            + 0.12 * min(1.0, anchor.score / 8.0)
        )

        if figures_by_anchor:
            figures = figures_by_anchor.get(index, [])
            if _figure_bridges_anchor_to_text(
                anchor,
                element,
                figures,
            ):
                score += getattr(
                    config,
                    "doclayout_figure_bridge_bonus",
                    0.90,
                )

        candidates.append((score, index))

    if not candidates:
        return None

    candidates.sort(reverse=True)
    best_score, best_index = candidates[0]

    return best_index if best_score >= 0.55 else None


def build_doclayout_assisted_regions(
    lines: Sequence[OCRLine],
    anchors: Sequence[HeadlineAnchor],
    elements: Sequence[LayoutElement],
    page_shape: tuple[int, int],
    body_height: float,
    document_type: str,
    config: HybridPipelineConfig,
    separators: Optional[Sequence[Separator]] = None,
) -> list[dict[str, Any]]:
    """
    V3 primary DocLayout grouping.

    Important:
    DocLayout detects STRUCTURAL ELEMENTS. This custom function associates
    title + plain_text + figure + figure_caption into article candidates.
    """
    if not anchors or not elements:
        return []

    separators = list(separators or [])

    text_classes = normalized_layout_class_set(
        config.doclayout_text_classes
    )
    caption_classes = normalized_layout_class_set(
        config.doclayout_caption_classes
    )
    figure_classes = normalized_layout_class_set(
        config.doclayout_figure_classes
    )

    normalized_elements = [
        (
            element,
            normalize_layout_class_name(
                element.class_name
            ),
        )
        for element in elements
    ]

    text_elements = [
        element
        for element, class_name in normalized_elements
        if (
            class_name in text_classes
            and element.confidence
            >= config.doclayout_min_text_confidence
        )
    ]

    caption_elements = [
        element
        for element, class_name in normalized_elements
        if (
            class_name in caption_classes
            and element.confidence
            >= max(0.10, config.doclayout_confidence * 0.60)
        )
    ]

    figure_elements = [
        element
        for element, class_name in normalized_elements
        if (
            class_name in figure_classes
            and element.confidence
            >= max(0.10, config.doclayout_confidence * 0.60)
        )
    ]

    # ----------------------------------------------------------
    # Stage A: assign figures to headlines first.
    # Figures provide a bridge for layouts:
    # headline -> photograph -> caption/body.
    # ----------------------------------------------------------

    figures_by_anchor: dict[int, list[LayoutElement]] = {
        index: []
        for index in range(len(anchors))
    }

    for figure in figure_elements:
        index = choose_anchor_for_layout_element_v3(
            figure,
            anchors,
            page_shape,
            body_height,
            config,
            separators,
            figures_by_anchor=None,
        )

        if index is not None:
            figures_by_anchor[index].append(figure)

    # ----------------------------------------------------------
    # Stage B: assign text/captions.
    # ----------------------------------------------------------

    elements_by_anchor: dict[int, list[LayoutElement]] = {
        index: list(figures_by_anchor[index])
        for index in range(len(anchors))
    }

    for element in text_elements + caption_elements:
        index = choose_anchor_for_layout_element_v3(
            element,
            anchors,
            page_shape,
            body_height,
            config,
            separators,
            figures_by_anchor=figures_by_anchor,
        )

        if index is not None:
            elements_by_anchor[index].append(element)

    # ----------------------------------------------------------
    # Stage C: construct article text from OCR lines belonging to
    # the associated structural boxes.
    # ----------------------------------------------------------

    regions: list[dict[str, Any]] = []

    for index, anchor in enumerate(anchors):
        article_elements = elements_by_anchor.get(
            index,
            [],
        )

        text_or_caption = [
            element
            for element in article_elements
            if normalize_layout_class_name(
                element.class_name
            )
            in (text_classes | caption_classes)
        ]

        if not text_or_caption:
            continue

        region_lines: list[OCRLine] = list(
            anchor.lines
        )

        for element in text_or_caption:
            class_name = normalize_layout_class_name(
                element.class_name
            )

            threshold = (
                config.doclayout_text_line_overlap
                if class_name in text_classes
                else 0.08
            )

            region_lines.extend(
                lines_overlapping_element(
                    lines,
                    element,
                    threshold,
                )
            )

        region_lines = sorted(
            {
                line.uid: line
                for line in region_lines
            }.values(),
            key=lambda line: (
                line.y1,
                line.x1,
            ),
        )

        if len(region_lines) <= len(anchor.lines):
            continue

        text = article_lines_to_text(
            region_lines,
            anchor,
            document_type,
            body_height,
        )

        text = remove_newspaper_artifacts(text)
        word_count = len(text.split())

        if word_count < config.min_article_words:
            continue

        if word_count > config.max_article_words:
            text = " ".join(
                text.split()[
                    : config.max_article_words
                ]
            )

        all_boxes = [anchor.bbox]
        all_boxes.extend(
            element.bbox
            for element in article_elements
        )

        ocr_bbox = bbox_from_lines(
            region_lines
        )
        all_boxes.append(ocr_bbox)

        region_bbox = (
            min(box[0] for box in all_boxes),
            min(box[1] for box in all_boxes),
            max(box[2] for box in all_boxes),
            max(box[3] for box in all_boxes),
        )

        text_count = sum(
            normalize_layout_class_name(
                element.class_name
            ) in text_classes
            for element in article_elements
        )
        figure_count = sum(
            normalize_layout_class_name(
                element.class_name
            ) in figure_classes
            for element in article_elements
        )
        caption_count = sum(
            normalize_layout_class_name(
                element.class_name
            ) in caption_classes
            for element in article_elements
        )

        layout_confidence = min(
            1.0,
            0.55
            + min(0.22, text_count * 0.045)
            + min(0.08, figure_count * 0.04)
            + min(0.06, caption_count * 0.03)
            + min(0.09, word_count / 900.0),
        )

        warnings: list[str] = []

        if word_count < 50:
            warnings.append(
                "Article contains limited text; summary reliability may be lower."
            )

        if ocr_quality_score(text) < 0.45:
            warnings.append(
                "OCR quality is low."
            )

        regions.append(
            {
                "anchor": anchor,
                "lines": region_lines,
                "bbox": region_bbox,
                "text": text,
                "layout_confidence": round(
                    layout_confidence,
                    3,
                ),
                "warnings": warnings,
                "source": (
                    "doclayout_yolo_custom_article_grouping_v3"
                ),
                "layout_elements": [
                    {
                        "class_name": (
                            normalize_layout_class_name(
                                element.class_name
                            )
                        ),
                        "confidence": round(
                            element.confidence,
                            4,
                        ),
                        "bbox": element.bbox,
                    }
                    for element in article_elements
                ],
            }
        )

    return regions


# ---------------------------------------------------------------------
# 21.6 ARTICLE TEXT / TITLE RECONSTRUCTION
# ---------------------------------------------------------------------


def article_lines_to_text(
    lines: Sequence[OCRLine],
    anchor: HeadlineAnchor,
    document_type: str,
    body_height: float,
) -> str:
    headline_ids = {
        line.uid
        for line in anchor.lines
    }

    region_box = bbox_from_lines(lines)

    columns = detect_body_columns(
        lines,
        region_box,
        headline_ids,
        body_height,
    )

    ordered_body: list[OCRLine] = []

    for column in columns:
        ordered_body.extend(column)

    raw_lines = [
        line.text
        for line in ordered_body
    ]

    cleaned_lines = clean_ocr_lines(
        raw_lines,
        document_type,
    )

    cleaned_lines = join_hyphenated_lines(
        cleaned_lines
    )

    # Bylines belong to the article but should not become title/body lead.
    while (
        cleaned_lines
        and looks_like_byline(
            cleaned_lines[0]
        )
    ):
        cleaned_lines.pop(0)

    body = " ".join(cleaned_lines)
    body = remove_newspaper_artifacts(body)

    title = remove_newspaper_artifacts(
        anchor.title
    )

    # If OCR repeated title inside the first body block, remove duplicate.
    if (
        body.lower().startswith(
            title.lower()
        )
    ):
        body = body[
            len(title):
        ].lstrip(" .:-")

    if body:
        return f"{title}. {body}".strip()

    return title


# ---------------------------------------------------------------------
# 21.7 FINAL HYBRID DETECTOR
# ---------------------------------------------------------------------


def fuse_headline_anchors(
    heuristic_anchors: Sequence[HeadlineAnchor],
    model_anchors: Sequence[HeadlineAnchor],
) -> list[HeadlineAnchor]:
    """
    Initial duplicate fusion. Final semantic filtering is performed in
    detect_hybrid_article_regions(), where page shape is available.
    """
    combined = list(heuristic_anchors)

    for model_anchor in model_anchors:
        duplicate_index = None

        for index, existing in enumerate(
            combined
        ):
            similarity = title_similarity(
                model_anchor.title,
                existing.title,
            )
            containment = bbox_containment(
                model_anchor.bbox,
                existing.bbox,
            )
            iou = bbox_iou(
                model_anchor.bbox,
                existing.bbox,
            )

            if (
                similarity >= 0.58
                and (
                    containment >= 0.28
                    or iou >= 0.12
                )
            ):
                duplicate_index = index
                break

        if duplicate_index is None:
            combined.append(model_anchor)
            continue

        existing = combined[duplicate_index]
        model_words = len(
            model_anchor.title.split()
        )
        existing_words = len(
            existing.title.split()
        )

        # Prefer a valid and more complete model headline.
        if (
            headline_text_is_valid(
                model_anchor.title,
                PipelineConfig(),
            )
            and model_words > existing_words
            and model_words <= 22
        ):
            combined[
                duplicate_index
            ] = model_anchor

    return sorted(
        combined,
        key=lambda item: (
            item.bbox[1],
            item.bbox[0],
        ),
    )


def _region_quality_for_fusion(
    region: dict[str, Any],
) -> float:
    words = len(
        region.get("text", "").split()
    )
    layout = float(
        region.get(
            "layout_confidence",
            0.0,
        )
    )

    source = region.get(
        "source",
        "",
    )

    text_bonus = min(
        0.30,
        words / 700.0,
    )

    if words > 1300:
        text_bonus -= 0.16

    source_bonus = (
        0.14
        if source.startswith(
            "doclayout_yolo"
        )
        else 0.0
    )

    return (
        layout
        + text_bonus
        + source_bonus
    )


def fuse_article_regions(
    geometry_regions: Sequence[dict[str, Any]],
    doclayout_regions: Sequence[dict[str, Any]],
    config: HybridPipelineConfig,
) -> list[dict[str, Any]]:
    """
    Prefer complete DocLayout-assisted candidates while retaining geometry
    candidates when the pretrained model misses an article.
    """
    candidates = list(
        doclayout_regions
    )

    if config.hybrid_keep_geometry_fallback:
        candidates.extend(
            geometry_regions
        )

    if not candidates:
        return []

    candidates = sorted(
        candidates,
        key=lambda region: (
            region["bbox"][1],
            region["bbox"][0],
        ),
    )

    kept: list[dict[str, Any]] = []

    for candidate in candidates:
        if not headline_text_is_valid(
            candidate["anchor"].title,
            config,
        ):
            continue

        duplicate_index = None

        for index, existing in enumerate(
            kept
        ):
            title_sim = title_similarity(
                candidate["anchor"].title,
                existing["anchor"].title,
            )
            iou = bbox_iou(
                candidate["bbox"],
                existing["bbox"],
            )
            containment = bbox_containment(
                candidate["bbox"],
                existing["bbox"],
            )

            ids_a = {
                line.uid
                for line in candidate.get(
                    "lines",
                    [],
                )
            }
            ids_b = {
                line.uid
                for line in existing.get(
                    "lines",
                    [],
                )
            }

            shared = (
                len(ids_a & ids_b)
                / max(
                    1,
                    min(
                        len(ids_a),
                        len(ids_b),
                    ),
                )
            )

            same_article = (
                (
                    title_sim >= 0.50
                    and (
                        iou >= 0.14
                        or containment >= 0.30
                        or shared >= 0.22
                    )
                )
                or (
                    title_sim >= 0.72
                    and shared >= 0.10
                )
            )

            if same_article:
                duplicate_index = index
                break

        if duplicate_index is None:
            kept.append(candidate)
        else:
            existing = kept[
                duplicate_index
            ]

            if (
                _region_quality_for_fusion(
                    candidate
                )
                > _region_quality_for_fusion(
                    existing
                )
            ):
                kept[
                    duplicate_index
                ] = candidate

    kept = deduplicate_and_merge_regions(
        kept,
        config,
    )

    # Last cleanup: reject any region that somehow starts as a byline/body.
    cleaned_regions = []

    for region in kept:
        title = region["anchor"].title

        if looks_like_byline(title):
            continue

        first_alpha = next(
            (
                character
                for character in title
                if character.isalpha()
            ),
            "",
        )

        if (
            first_alpha
            and first_alpha.islower()
            and len(title.split()) >= 4
        ):
            continue

        region["text"] = remove_newspaper_artifacts(
            region["text"]
        )

        cleaned_regions.append(region)

    return sorted(
        cleaned_regions,
        key=lambda region: (
            region["bbox"][1],
            region["bbox"][0],
        ),
    )


def detect_hybrid_article_regions(
    lines: Sequence[OCRLine],
    prepared_gray: np.ndarray,
    prepared_color: np.ndarray,
    document_type: str,
    config: HybridPipelineConfig,
    doclayout_model=None,
) -> dict[str, Any]:
    page_shape = prepared_gray.shape[:2]

    separators = detect_page_separators(
        prepared_gray
    )

    body_height = estimate_body_height(
        lines,
        page_shape,
    )

    content_area = detect_adaptive_content_area(
        lines,
        separators,
        page_shape,
        body_height,
        config,
    )

    usable_lines = [
        line
        for line in lines
        if (
            content_area[0]
            <= line.cy
            <= content_area[1]
        )
        and not is_noise_line(
            line.text,
            document_type,
        )
    ]

    # ----------------------------------------------------------
    # A. Geometry headlines
    # ----------------------------------------------------------

    candidates = detect_headline_candidates(
        usable_lines,
        body_height,
        page_shape,
        content_area,
        config,
    )

    geometry_anchors = combine_multiline_headlines(
        candidates,
        usable_lines,
        body_height,
        page_shape,
    )

    geometry_anchors = merge_short_headline_continuations(
        geometry_anchors,
        body_height,
    )

    # ----------------------------------------------------------
    # B. DocLayout elements + model headlines
    # ----------------------------------------------------------

    layout_elements: list[LayoutElement] = []
    model_anchors: list[HeadlineAnchor] = []
    model_warning = None

    if (
        config.use_doclayout_yolo
        and doclayout_model is not None
    ):
        try:
            layout_elements = detect_doclayout_elements(
                doclayout_model,
                prepared_color,
                config,
            )

            model_anchors = anchors_from_doclayout(
                layout_elements,
                usable_lines,
                document_type,
                content_area,
                config,
            )

        except Exception as error:
            model_warning = (
                "DocLayout-YOLO inference failed. "
                "Geometry fallback used. "
                f"Reason: {error}"
            )

    # ----------------------------------------------------------
    # C. Fuse and FILTER anchors.
    # This is where bylines / lowercase body fragments / section
    # labels are removed before region construction.
    # ----------------------------------------------------------

    raw_fused_anchors = fuse_headline_anchors(
        geometry_anchors,
        model_anchors,
    )

    anchors = filter_headline_anchors_v3(
        raw_fused_anchors,
        page_shape,
        config,
    )

    geometry_anchors_filtered = filter_headline_anchors_v3(
        geometry_anchors,
        page_shape,
        config,
    )

    model_anchors_filtered = filter_headline_anchors_v3(
        model_anchors,
        page_shape,
        config,
    )

    # ----------------------------------------------------------
    # D. Geometry candidates
    # ----------------------------------------------------------

    geometry_regions = build_provisional_article_regions(
        usable_lines,
        anchors,
        separators,
        content_area,
        page_shape,
        body_height,
        config,
    )

    for region in geometry_regions:
        region[
            "source"
        ] = (
            "adaptive_headline_separator_geometry_v3"
        )
        region["text"] = remove_newspaper_artifacts(
            region["text"]
        )

    geometry_regions = deduplicate_and_merge_regions(
        geometry_regions,
        config,
    )

    # ----------------------------------------------------------
    # E. DocLayout-assisted candidates
    # ----------------------------------------------------------

    model_regions = build_doclayout_assisted_regions(
        usable_lines,
        anchors,
        layout_elements,
        page_shape,
        body_height,
        document_type,
        config,
        separators=separators,
    )

    # ----------------------------------------------------------
    # F. Final hybrid fusion
    # ----------------------------------------------------------

    final_regions = fuse_article_regions(
        geometry_regions,
        model_regions,
        config,
    )

    warnings: list[str] = []

    if model_warning:
        warnings.append(
            model_warning
        )

    if not anchors:
        warnings.append(
            "No reliable headline anchors were detected."
        )

    if not final_regions:
        warnings.append(
            "Hybrid article segmentation failed; "
            "the full-page fallback will be used."
        )

    if len(final_regions) > 22:
        warnings.append(
            "High article count suggests possible over-segmentation."
        )

    class_counts = doclayout_class_counts(
        layout_elements
    )

    if (
        getattr(
            config,
            "print_doclayout_class_counts",
            False,
        )
        and config.debug
    ):
        print(
            "DocLayout normalized class counts:",
            class_counts,
        )

    return {
        "regions": final_regions,
        "anchors": anchors,
        "geometry_anchors": geometry_anchors_filtered,
        "doclayout_anchors": model_anchors_filtered,
        "layout_elements": layout_elements,
        "separators": separators,
        "body_height": body_height,
        "content_area": content_area,
        "warnings": warnings,
        "doclayout_class_counts": class_counts,
        "debug_counts": {
            "geometry_anchor_count": len(
                geometry_anchors_filtered
            ),
            "doclayout_anchor_count": len(
                model_anchors_filtered
            ),
            "raw_fused_anchor_count": len(
                raw_fused_anchors
            ),
            "fused_anchor_count": len(
                anchors
            ),
            "doclayout_element_count": len(
                layout_elements
            ),
            "geometry_region_count": len(
                geometry_regions
            ),
            "doclayout_region_count": len(
                model_regions
            ),
            "final_region_count": len(
                final_regions
            ),
        },
    }


# ---------------------------------------------------------------------
# 21.8 CATEGORY IMPROVEMENTS
# ---------------------------------------------------------------------


def strong_article_keyword_category(
    title: str,
    text: str,
) -> Optional[tuple[str, int]]:
    """
    Strong support from headline + a small article-text window.
    The trained classifier remains primary; these rules only recover
    obvious domain cases such as SLC/cricket.
    """
    title_tokens = set(
        re.findall(
            r"[a-z]+",
            title.lower(),
        )
    )

    body_tokens = set(
        re.findall(
            r"[a-z]+",
            " ".join(text.split()[:160]).lower(),
        )
    )

    ranking = []

    for category, keywords in CATEGORY_KEYWORDS.items():
        normalized_keywords = {
            token.lower()
            for phrase in keywords
            for token in phrase.split()
        }

        title_hits = len(
            title_tokens
            & normalized_keywords
        )
        body_hits = len(
            body_tokens
            & normalized_keywords
        )

        weighted = (
            3 * title_hits
            + body_hits
        )

        if weighted:
            ranking.append(
                (
                    category,
                    weighted,
                    title_hits,
                    body_hits,
                )
            )

    if not ranking:
        return None

    ranking.sort(
        key=lambda item: (
            item[1],
            item[2],
        ),
        reverse=True,
    )

    category, weighted, title_hits, body_hits = ranking[0]

    # Require clear evidence.
    if (
        title_hits >= 1
        or weighted >= 3
        or body_hits >= 3
    ):
        return category, weighted

    return None


def predict_category(
    model,
    text: str,
    document_type: str,
    title: str,
    config: PipelineConfig,
) -> dict[str, Any]:
    normalized_type = normalize_document_type(
        document_type
    )

    cleaned = remove_newspaper_artifacts(
        text
    )

    if normalized_type == "Novel/Story":
        return {
            "category": "Story",
            "confidence": 1.0,
            "margin": 1.0,
            "top_predictions": [
                {
                    "category": "Story",
                    "probability": 1.0,
                }
            ],
            "decision": "document_type_rule",
        }

    if len(cleaned.split()) < 8:
        return {
            "category": "General Text",
            "confidence": 0.0,
            "margin": 0.0,
            "top_predictions": [],
            "decision": "too_little_text",
        }

    repeated_title = " ".join(
        [title]
        * max(
            1,
            config.title_repeat_for_category,
        )
    )

    classification_text = clean_text_basic(
        f"{repeated_title} {cleaned}"
    )

    predicted = str(
        model.predict(
            [classification_text]
        )[0]
    )

    category = (
        predicted
        if predicted in PROJECT_CATEGORIES
        else "General Text"
    )

    confidence = 1.0
    margin = 1.0
    top_predictions: list[
        dict[str, Any]
    ] = []

    if hasattr(
        model,
        "predict_proba",
    ):
        probabilities = model.predict_proba(
            [classification_text]
        )[0]

        classes = get_model_classes(
            model
        )

        ranking = np.argsort(
            probabilities
        )[::-1]

        top_predictions = [
            {
                "category": classes[
                    int(index)
                ],
                "probability": round(
                    float(
                        probabilities[
                            int(index)
                        ]
                    ),
                    4,
                ),
            }
            for index in ranking[:3]
        ]

        confidence = float(
            probabilities[
                int(ranking[0])
            ]
        )

        margin = (
            confidence
            - float(
                probabilities[
                    int(ranking[1])
                ]
            )
            if len(ranking) > 1
            else confidence
        )

        category = classes[
            int(ranking[0])
        ]

    decision = (
        "model_probability_and_margin"
    )

    support = strong_article_keyword_category(
        title,
        cleaned,
    )

    if support is not None:
        supported_category, strength = support

        # Very strong article/title support can override an uncertain
        # classifier, but does not blindly override a very confident model.
        if (
            confidence < 0.66
            or margin < 0.22
            or strength >= 6
        ):
            category = supported_category
            decision = (
                "model_with_strong_article_keyword_support"
            )

    if (
        confidence
        < config.category_min_probability
        or margin
        < config.category_min_margin
    ):
        if support is None:
            category = "General Text"
            decision = (
                "low_confidence_or_margin_fallback"
            )

    if (
        normalized_type == "Textbook"
        and category
        in {
            "Politics",
            "Social",
            "Sports",
            "Story",
        }
        and confidence < 0.70
    ):
        category = "General Text"
        decision = "textbook_type_gate"

    return {
        "category": (
            category
            if category in PROJECT_CATEGORIES
            else "General Text"
        ),
        "confidence": round(
            confidence,
            4,
        ),
        "margin": round(
            margin,
            4,
        ),
        "top_predictions": (
            top_predictions
        ),
        "decision": decision,
    }


# ---------------------------------------------------------------------
# 21.9 STRICT / GROUNDED LLAMA SUMMARY
# ---------------------------------------------------------------------


SUMMARY_BAD_PREFACES = (
    "i couldn't find",
    "i could not find",
    "however, i can",
    "the article does not",
    "the text does not",
    "as an ai",
    "based on the prompt",
)


def _important_summary_tokens(
    text: str,
) -> set[str]:
    stop = {
        "the", "and", "that", "with", "from", "this", "were",
        "have", "has", "had", "for", "but", "not", "are", "was",
        "will", "their", "they", "into", "about", "after", "before",
        "which", "when", "where", "who", "his", "her", "its", "said",
        "says", "also", "been", "being", "than", "then", "them", "there",
    }

    return {
        token
        for token in re.findall(
            r"[a-z0-9$%]+",
            text.lower(),
        )
        if (
            len(token) >= 3
            and token not in stop
        )
    }


def summary_is_grounded(
    summary: str,
    source: str,
    config: PipelineConfig,
) -> bool:
    clean_summary = clean_text_basic(
        summary
    )
    clean_source = clean_text_basic(
        source
    )

    if not clean_summary:
        return False

    low = clean_summary.lower()

    if any(
        phrase in low
        for phrase in SUMMARY_BAD_PREFACES
    ):
        return False

    # Numbers/dates/money are especially dangerous to hallucinate.
    source_numbers = set(
        re.findall(
            r"\b\d+(?:[.,]\d+)*\b",
            clean_source,
        )
    )
    summary_numbers = set(
        re.findall(
            r"\b\d+(?:[.,]\d+)*\b",
            clean_summary,
        )
    )

    unsupported_numbers = (
        summary_numbers
        - source_numbers
    )

    if len(unsupported_numbers) > getattr(
        config,
        "summary_max_unsupported_number_count",
        0,
    ):
        return False

    source_tokens = _important_summary_tokens(
        clean_source
    )

    sentences = [
        sentence.strip()
        for sentence in re.split(
            r"(?<=[.!?])\s+",
            clean_summary,
        )
        if len(sentence.split()) >= 4
    ]

    if not sentences:
        return False

    min_overlap = getattr(
        config,
        "summary_grounding_min_sentence_overlap",
        0.14,
    )

    for sentence in sentences:
        sentence_tokens = _important_summary_tokens(
            sentence
        )

        if not sentence_tokens:
            continue

        overlap = (
            len(
                sentence_tokens
                & source_tokens
            )
            / max(
                1,
                len(sentence_tokens),
            )
        )

        if overlap < min_overlap:
            return False

    return True


def summarize_with_local_llama(
    text: str,
    title: str,
    document_type: str,
    category: str,
    config: PipelineConfig,
) -> str:
    cleaned = remove_newspaper_artifacts(
        text
    )

    if not cleaned:
        return (
            "Readable text was not found."
        )

    min_words = getattr(
        config,
        "min_summary_words_for_llama",
        55,
    )

    if len(cleaned.split()) < min_words:
        return extractive_summary(
            cleaned,
            max_sentences=2,
        )

    words = cleaned.split()

    chunks = [
        " ".join(
            words[
                index:
                index + 700
            ]
        )
        for index in range(
            0,
            len(words),
            700,
        )
    ]

    partials: list[str] = []

    for chunk in chunks[:4]:
        prompt = f"""
You summarize a printed article for a blind reader.

Document type: {document_type}
Category: {category}
Article title: {title}

STRICT FACTUAL RULES:
- Use ONLY information explicitly present in the supplied article text.
- Never infer missing facts.
- Never guess causes, motives, people responsible, charges, or outcomes.
- Never invent names, dates, numbers, places, events, explanations, or quotations.
- If OCR text is incomplete, summarize only facts that are clearly present.
- If a sentence is damaged or uncertain, omit that sentence.
- Do not say "I couldn't find", "the article does not say", or discuss the task.
- Do not combine information from another article.
- Keep important names, dates, places, numbers and the main point exactly grounded in the text.
- Use clear simple English.
- Maximum {config.summary_max_words} words.
- Return ONLY the summary.

ARTICLE TEXT:
{chunk}

SUMMARY:
""".strip()

        candidate = ollama_generate(
            prompt,
            config,
            temperature=0.05,
            num_predict=220,
        )

        if (
            candidate
            and summary_is_grounded(
                candidate,
                chunk,
                config,
            )
        ):
            partials.append(
                candidate
            )
        else:
            partials.append(
                extractive_summary(
                    chunk,
                    max_sentences=3,
                )
            )

    if len(partials) == 1:
        return partials[0]

    combined_source = " ".join(
        partials
    )

    final_prompt = f"""
Combine the partial summaries below into one short factual summary.

STRICT RULES:
- Use only facts already present in the partial summaries.
- Add no new names, numbers, causes, motives, events, or explanations.
- Remove repetition.
- Do not mention the summarization process.
- Maximum {config.summary_max_words} words.
- Return only the final summary.

PARTIAL SUMMARIES:
{combined_source}

FINAL SUMMARY:
""".strip()

    final = ollama_generate(
        final_prompt,
        config,
        temperature=0.03,
        num_predict=220,
    )

    # Validate final against the ORIGINAL article, not only partials.
    if (
        final
        and summary_is_grounded(
            final,
            cleaned,
            config,
        )
    ):
        return final

    return extractive_summary(
        cleaned,
        max_sentences=3,
    )


def article_safe_for_llama_summary(
    item: dict[str, Any],
    config: PipelineConfig,
) -> bool:
    word_count = len(
        item.get(
            "full_text",
            "",
        ).split()
    )

    ocr_quality = float(
        item.get(
            "ocr_quality",
            0.0,
        )
    )

    layout_confidence = float(
        item.get(
            "layout_confidence",
            0.0,
        )
    )

    return (
        word_count
        >= getattr(
            config,
            "min_summary_words_for_llama",
            55,
        )
        and ocr_quality
        >= getattr(
            config,
            "min_summary_ocr_quality",
            0.45,
        )
        and layout_confidence
        >= getattr(
            config,
            "min_summary_layout_confidence",
            0.50,
        )
    )


# ---------------------------------------------------------------------
# 21.10 OUTPUT WITH SUMMARY QUALITY GATE
# ---------------------------------------------------------------------


def generate_selected_output(
    analysis: dict[str, Any],
    selected_category: Optional[str],
    depth: str,
    config: PipelineConfig,
    clean_full_text_with_llama: bool = False,
) -> dict[str, Any]:
    categories = analysis.get(
        "detected_categories",
        [],
    )

    if not categories:
        return {
            "status": "ERROR",
            "message": "No category was detected.",
        }

    if selected_category is None:
        selected_category = (
            categories[0]
            if len(categories) == 1
            else None
        )

    if selected_category not in categories:
        return {
            "status": "ERROR",
            "message": (
                "Selected category was not detected."
            ),
            "available_categories": categories,
        }

    depth = depth.strip().lower()

    if depth not in {
        "summary",
        "full",
    }:
        return {
            "status": "ERROR",
            "message": (
                "Depth must be 'summary' or 'full'."
            ),
        }

    selected_items = [
        item
        for item in analysis[
            "results"
        ]
        if item["category"]
        == selected_category
    ]

    output_items = []

    for item in selected_items:
        summary_method = None
        warnings = list(
            item.get(
                "warnings",
                [],
            )
        )

        source_text = remove_newspaper_artifacts(
            item["full_text"]
        )

        if depth == "summary":
            if article_safe_for_llama_summary(
                item,
                config,
            ):
                content = summarize_with_local_llama(
                    text=source_text,
                    title=item["title"],
                    document_type=analysis[
                        "document_type"
                    ],
                    category=selected_category,
                    config=config,
                )
                summary_method = (
                    "local_llama_grounded"
                )
            else:
                # Extractive summary cannot invent facts.
                content = extractive_summary(
                    source_text,
                    max_sentences=2,
                )
                summary_method = (
                    "extractive_quality_fallback"
                )
                warnings.append(
                    "Llama summary skipped because the detected article "
                    "did not pass the article quality gate."
                )

        elif clean_full_text_with_llama:
            content = reconstruct_selected_text_with_llama(
                text=source_text,
                title=item["title"],
                document_type=analysis[
                    "document_type"
                ],
                config=config,
            )
            summary_method = (
                "llama_full_text_cleanup"
            )

        else:
            content = source_text
            summary_method = (
                "local_reconstructed_ocr"
            )

        output_items.append(
            {
                "article_number": item[
                    "article_number"
                ],
                "title": item["title"],
                "category": item["category"],
                "category_confidence": item[
                    "confidence"
                ],
                "probability_margin": item[
                    "probability_margin"
                ],
                "ocr_quality": item[
                    "ocr_quality"
                ],
                "layout_confidence": item[
                    "layout_confidence"
                ],
                "source": item.get(
                    "source",
                    "",
                ),
                "processing_method": (
                    summary_method
                ),
                "warnings": warnings,
                "text": content,
                "bbox": item[
                    "normalized_bbox"
                ],
            }
        )

    if (
        analysis["document_type"]
        in MULTI_CATEGORY_TYPES
    ):
        intro = (
            f"{selected_category} section. "
            f"I found {len(output_items)} "
            f"article"
            f"{'s' if len(output_items) != 1 else ''}."
        )

        spoken_parts = [intro]

        for index, item in enumerate(
            output_items,
            start=1,
        ):
            spoken_parts.append(
                f"Article {index}. "
                f"{item['title']}. "
                f"{item['text']}"
            )

        final_text = "\n\n".join(
            spoken_parts
        )

    else:
        final_text = "\n\n".join(
            item["text"]
            for item in output_items
        )

    return {
        "status": "READY",
        "document_type": analysis[
            "document_type"
        ],
        "mode": analysis["mode"],
        "selected_category": (
            selected_category
        ),
        "content_depth": depth,
        "article_count": len(
            output_items
        ),
        "items": output_items,
        "final_output_text": (
            final_text
        ),
        "next_module_payload": {
            "text": final_text,
            "document_type": analysis[
                "document_type"
            ],
            "category": selected_category,
            "content_depth": depth,
            "source_article_count": len(
                output_items
            ),
            "articles": [
                {
                    "article_number": item[
                        "article_number"
                    ],
                    "title": item[
                        "title"
                    ],
                    "category": item[
                        "category"
                    ],
                    "processing_method": item[
                        "processing_method"
                    ],
                    "ocr_quality": item[
                        "ocr_quality"
                    ],
                    "layout_confidence": item[
                        "layout_confidence"
                    ],
                    "text": item[
                        "text"
                    ],
                }
                for item in output_items
            ],
        },
    }


# ---------------------------------------------------------------------
# 21.11 VOICE ALIASES
# ---------------------------------------------------------------------


def _voice_alias_map(
    allowed: Sequence[str],
) -> dict[str, str]:
    aliases: dict[str, str] = {
        # Depth
        "summary": "summary",
        "summarize": "summary",
        "summarise": "summary",
        "short summary": "summary",
        "brief summary": "summary",
        "give summary": "summary",
        "read summary": "summary",

        "full": "full",
        "full text": "full",
        "complete text": "full",
        "complete": "full",
        "read": "full",
        "read it": "full",
        "read all": "full",
        "read everything": "full",
        "read full": "full",
        "read full text": "full",

        # Categories
        "sport": "Sports",
        "sports": "Sports",
        "sports news": "Sports",

        "politics": "Politics",
        "political": "Politics",
        "political news": "Politics",

        "social": "Social",
        "social news": "Social",

        "science": "Science",
        "scientific": "Science",

        "mathematics": "Mathematics",
        "math": "Mathematics",
        "maths": "Mathematics",

        "story": "Story",
        "stories": "Story",

        "general": "General Text",
        "general text": "General Text",
        "general news": "General Text",
    }

    allowed_set = set(
        allowed
    )

    return {
        phrase: target
        for phrase, target in aliases.items()
        if target in allowed_set
    }


# ---------------------------------------------------------------------
# 21.12 UPDATED HYBRID ANALYSIS WITH V3 DEBUG INFO
# ---------------------------------------------------------------------


def analyze_content_hybrid(
    image_path: str,
    document_type: str,
    model,
    config: HybridPipelineConfig,
    doclayout_model=None,
) -> dict[str, Any]:
    normalized_type = normalize_document_type(
        document_type
    )

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        raise FileNotFoundError(
            f"Image not found or unreadable: {image_path}"
        )

    prepared_color, prepared_gray, scale = prepare_page_image(
        image,
        config,
    )

    page_shape = prepared_gray.shape[:2]
    warnings: list[str] = []
    layout_debug: dict[str, Any] = {}

    if normalized_type in MULTI_CATEGORY_TYPES:
        ocr_lines = extract_ocr_lines(
            prepared_gray,
            normalized_type,
            config,
        )

        layout = detect_hybrid_article_regions(
            ocr_lines,
            prepared_gray,
            prepared_color,
            normalized_type,
            config,
            doclayout_model=doclayout_model,
        )

        warnings.extend(
            layout["warnings"]
        )

        layout_debug = {
            "pipeline_version": "V3_FIXED",
            "ocr_line_count": len(
                ocr_lines
            ),
            "body_height": round(
                float(
                    layout[
                        "body_height"
                    ]
                ),
                2,
            ),
            "content_area": layout[
                "content_area"
            ],
            **layout["debug_counts"],
            "doclayout_class_counts": (
                layout.get(
                    "doclayout_class_counts",
                    {},
                )
            ),
            "headline_anchors": [
                {
                    "title": anchor.title,
                    "bbox": anchor.bbox,
                    "score": round(
                        float(
                            anchor.score
                        ),
                        3,
                    ),
                }
                for anchor in layout[
                    "anchors"
                ]
            ],
            "doclayout_elements": [
                {
                    "class_name": (
                        normalize_layout_class_name(
                            element.class_name
                        )
                    ),
                    "raw_class_name": (
                        element.class_name
                    ),
                    "confidence": round(
                        float(
                            element.confidence
                        ),
                        4,
                    ),
                    "bbox": (
                        element.bbox
                    ),
                }
                for element in layout[
                    "layout_elements"
                ]
            ],
            "separators": [
                asdict(
                    separator
                )
                for separator in layout[
                    "separators"
                ]
            ],
        }

        results: list[ArticleRegion] = []

        for number, region in enumerate(
            layout["regions"],
            start=1,
        ):
            anchor = region[
                "anchor"
            ]

            clean_region_text = remove_newspaper_artifacts(
                region["text"]
            )

            prediction = predict_category(
                model,
                clean_region_text,
                normalized_type,
                anchor.title,
                config,
            )

            results.append(
                ArticleRegion(
                    article_number=number,
                    title=remove_newspaper_artifacts(
                        anchor.title
                    ),
                    bbox=region[
                        "bbox"
                    ],
                    normalized_bbox=normalize_bbox(
                        region[
                            "bbox"
                        ],
                        page_shape,
                    ),
                    full_text=clean_region_text,
                    category=prediction[
                        "category"
                    ],
                    confidence=prediction[
                        "confidence"
                    ],
                    probability_margin=prediction[
                        "margin"
                    ],
                    top_predictions=prediction[
                        "top_predictions"
                    ],
                    ocr_quality=ocr_quality_score(
                        clean_region_text
                    ),
                    layout_confidence=float(
                        region[
                            "layout_confidence"
                        ]
                    ),
                    warnings=list(
                        region.get(
                            "warnings",
                            [],
                        )
                    ),
                    source=region.get(
                        "source",
                        "hybrid_v3",
                    ),
                )
            )

        if not results:
            full_text = best_full_page_ocr(
                prepared_gray,
                normalized_type,
            )

            full_text = remove_newspaper_artifacts(
                full_text
            )

            prediction = predict_category(
                model,
                full_text,
                normalized_type,
                "Full page",
                config,
            )

            results = [
                ArticleRegion(
                    article_number=1,
                    title="Full page",
                    bbox=(
                        0,
                        0,
                        page_shape[1],
                        page_shape[0],
                    ),
                    normalized_bbox=(
                        0,
                        0,
                        1000,
                        1000,
                    ),
                    full_text=full_text,
                    category=prediction[
                        "category"
                    ],
                    confidence=prediction[
                        "confidence"
                    ],
                    probability_margin=prediction[
                        "margin"
                    ],
                    top_predictions=prediction[
                        "top_predictions"
                    ],
                    ocr_quality=ocr_quality_score(
                        full_text
                    ),
                    layout_confidence=0.20,
                    warnings=[
                        "Hybrid V3 article segmentation fallback: "
                        "the full page was processed as one region."
                    ],
                    source=(
                        "full_page_fallback"
                    ),
                )
            ]

    else:
        full_text = best_full_page_ocr(
            prepared_gray,
            normalized_type,
        )

        full_text = remove_newspaper_artifacts(
            full_text
        )

        title = "Full page"

        prediction = predict_category(
            model,
            full_text,
            normalized_type,
            title,
            config,
        )

        results = [
            ArticleRegion(
                article_number=1,
                title=title,
                bbox=(
                    0,
                    0,
                    page_shape[1],
                    page_shape[0],
                ),
                normalized_bbox=(
                    0,
                    0,
                    1000,
                    1000,
                ),
                full_text=full_text,
                category=prediction[
                    "category"
                ],
                confidence=prediction[
                    "confidence"
                ],
                probability_margin=prediction[
                    "margin"
                ],
                top_predictions=prediction[
                    "top_predictions"
                ],
                ocr_quality=ocr_quality_score(
                    full_text
                ),
                layout_confidence=1.0,
                warnings=[],
                source=(
                    "full_page_tesseract"
                ),
            )
        ]

    category_counts = Counter(
        result.category
        for result in results
    )

    categories = sorted(
        category_counts
    )

    mode = (
        "Multi Category"
        if normalized_type
        in MULTI_CATEGORY_TYPES
        else "Single Category"
    )

    return {
        "status": "ANALYZED",
        "pipeline_version": "V3_FIXED",
        "document_type_received": (
            document_type
        ),
        "document_type": (
            normalized_type
        ),
        "mode": mode,
        "detected_categories": (
            categories
        ),
        "category_counts": dict(
            category_counts
        ),
        "needs_category_selection": (
            normalized_type
            in MULTI_CATEGORY_TYPES
            and len(categories) > 1
        ),
        "image_shape": {
            "height": page_shape[0],
            "width": page_shape[1],
        },
        "resize_scale": scale,
        "warnings": warnings,
        "layout_debug": layout_debug,
        "results": [
            asdict(result)
            for result in results
        ],
        "processing": {
            "cloud_api_used": False,
            "ocr": "local_tesseract",
            "pretrained_layout_model": (
                "local_DocLayout_YOLO_DocStructBench"
                if (
                    config.use_doclayout_yolo
                    and doclayout_model
                    is not None
                )
                else "disabled"
            ),
            "article_detection": (
                "V3_hybrid_doclayout_plus_custom_grouping_plus_geometry_fallback"
            ),
            "category_model": (
                "local_tfidf_logistic_regression"
            ),
            "speech_input": (
                "local_vosk"
            ),
            "summary_model": (
                f"local_ollama_{config.ollama_model}"
            ),
        },
    }


# ---------------------------------------------------------------------
# 21.13 V3 DEBUG TABLE
# ---------------------------------------------------------------------


def hybrid_debug_summary(
    analysis: dict[str, Any],
) -> pd.DataFrame:
    debug = analysis.get(
        "layout_debug",
        {},
    )

    rows = [
        (
            "Pipeline version",
            debug.get(
                "pipeline_version",
                analysis.get(
                    "pipeline_version",
                    "",
                ),
            ),
        ),
        (
            "OCR lines",
            debug.get(
                "ocr_line_count",
                0,
            ),
        ),
        (
            "Geometry headlines",
            debug.get(
                "geometry_anchor_count",
                0,
            ),
        ),
        (
            "DocLayout headlines",
            debug.get(
                "doclayout_anchor_count",
                0,
            ),
        ),
        (
            "Raw fused headlines",
            debug.get(
                "raw_fused_anchor_count",
                0,
            ),
        ),
        (
            "Filtered fused headlines",
            debug.get(
                "fused_anchor_count",
                0,
            ),
        ),
        (
            "DocLayout elements",
            debug.get(
                "doclayout_element_count",
                0,
            ),
        ),
        (
            "Geometry regions",
            debug.get(
                "geometry_region_count",
                0,
            ),
        ),
        (
            "DocLayout-assisted regions",
            debug.get(
                "doclayout_region_count",
                0,
            ),
        ),
        (
            "Final fused articles",
            debug.get(
                "final_region_count",
                len(
                    analysis.get(
                        "results",
                        [],
                    )
                ),
            ),
        ),
    ]

    return pd.DataFrame(
        rows,
        columns=[
            "Stage",
            "Value",
        ],
    )


def doclayout_class_table(
    analysis: dict[str, Any],
) -> pd.DataFrame:
    counts = analysis.get(
        "layout_debug",
        {},
    ).get(
        "doclayout_class_counts",
        {},
    )

    return pd.DataFrame(
        [
            {
                "class": name,
                "count": count,
            }
            for name, count in counts.items()
        ]
    ).sort_values(
        "count",
        ascending=False,
    ).reset_index(
        drop=True
    )


# ---------------------------------------------------------------------
# 21.14 FULL ONE-CALL V3 VOICE PIPELINE
# ---------------------------------------------------------------------


def run_hybrid_voice_pipeline_v3(
    image_path: str,
    document_type: str,
    category_model,
    config: HybridPipelineConfigV3,
    doclayout_model,
    voice_selector: OfflineVoiceSelector,
    tts_enabled: bool = False,
    clean_full_text_with_llama: bool = False,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
]:
    analysis = analyze_content_hybrid(
        image_path=image_path,
        document_type=document_type,
        model=category_model,
        config=config,
        doclayout_model=doclayout_model,
    )

    selected_category, depth = collect_user_preferences_voice(
        analysis=analysis,
        voice_selector=voice_selector,
        tts_enabled=tts_enabled,
    )

    output = generate_selected_output(
        analysis=analysis,
        selected_category=selected_category,
        depth=depth,
        config=config,
        clean_full_text_with_llama=clean_full_text_with_llama,
    )

    return analysis, output


# =====================================================================
# 22. V4 — ADAPTIVE HEADLINE HIERARCHY + DUPLICATE MERGING
# =====================================================================
#
# V4 is a GENERAL newspaper-layout improvement based on diverse-layout
# testing. It does NOT use fixed page coordinates or publication-specific
# headline strings.
#
# New capabilities:
#   1. dominant headline rescue from OCR geometry
#   2. kicker -> main headline -> deck/subheadline hierarchy grouping
#   3. body-fragment rejection using typography + sentence shape
#   4. prefix/contained-title duplicate article removal
#   5. better article-title selection inside a title hierarchy
#   6. deck/subheadline preservation in article text
#   7. standalone figure/caption blocks do not become articles
#
# Existing V3 features remain:
#   - DocLayout-YOLO structural elements
#   - local OpenCV/Tesseract geometry fallback
#   - custom article grouping
#   - local category model
#   - offline Vosk microphone
#   - local Ollama llama3.2:3b with summary grounding
# =====================================================================


@dataclass
class HybridPipelineConfigV4(HybridPipelineConfigV3):
    # Dominant headline rescue
    dominant_headline_height_ratio: float = 1.75
    dominant_headline_max_words_per_line: int = 18
    dominant_headline_group_gap_ratio: float = 1.55

    # Headline hierarchy
    hierarchy_max_vertical_gap_body_heights: float = 5.5
    hierarchy_max_page_gap_ratio: float = 0.055
    hierarchy_min_horizontal_relation: float = 0.22
    hierarchy_strong_title_ratio: float = 1.75
    hierarchy_two_strong_title_gap_body_heights: float = 2.0
    hierarchy_body_lines_to_block_merge: int = 2

    # Generic body-fragment rejection
    body_fragment_max_height_ratio: float = 1.55
    body_fragment_max_words: int = 20

    # Prefix duplicate merging
    prefix_duplicate_min_tokens: int = 2
    prefix_duplicate_min_box_relation: float = 0.28
    prefix_duplicate_max_center_distance_ratio: float = 0.08

    # Article cleanup
    min_final_article_words: int = 25


# ---------------------------------------------------------------------
# 22.1 HEADLINE TEXT NORMALIZATION
# ---------------------------------------------------------------------


_COMMON_SHORT_HEADLINE_TOKENS = {
    "US", "UK", "UN", "EU", "AI", "IT", "PM", "MP", "MPs",
    "SLC", "ICC", "NPP", "SJB", "UNP", "NCPA", "NMRA", "DGHS",
    "COVID", "WHO", "IMF", "GDP", "VAT",
}


def clean_headline_ocr_artifacts(text: str) -> str:
    """
    Conservative cleanup for headline OCR.

    Removes a likely decorative/icon token only when:
    - it is 1-2 alphabetic uppercase characters,
    - it is not a common acronym,
    - it is followed by a long uppercase-looking headline.

    This handles arrow/bullet OCR without hard-coding a newspaper.
    """
    value = remove_newspaper_artifacts(text)
    value = re.sub(r"\s+", " ", value).strip(" |:;-")

    tokens = value.split()

    if len(tokens) >= 5:
        first = re.sub(r"[^A-Za-z]", "", tokens[0])

        rest = " ".join(tokens[1:])

        if (
            1 <= len(first) <= 2
            and first.isupper()
            and first not in _COMMON_SHORT_HEADLINE_TOKENS
            and uppercase_ratio(rest) >= 0.55
        ):
            value = rest.strip()

    return value


def _headline_tokens(text: str) -> list[str]:
    return re.findall(
        r"[a-z0-9]+",
        clean_headline_ocr_artifacts(text).lower(),
    )


def title_prefix_relation(first: str, second: str, min_tokens: int = 2) -> bool:
    """
    True when the shorter title is a meaningful prefix of the longer title.

    Examples:
      "Death of domestic"
      "Death of domestic Rishad's wife two others arrested"

    Also supports OCR where one title has one extra short/noisy token.
    """
    a = _headline_tokens(first)
    b = _headline_tokens(second)

    if len(a) < min_tokens or len(b) < min_tokens:
        return False

    short, long = (a, b) if len(a) <= len(b) else (b, a)

    if long[: len(short)] == short:
        return True

    # Soft prefix: allow one missing/noisy token in the first part.
    compare_len = min(len(short), 6)
    if compare_len >= min_tokens:
        matches = sum(
            short[index] == long[index]
            for index in range(compare_len)
        )
        if matches / compare_len >= 0.80:
            return True

    return False


# ---------------------------------------------------------------------
# 22.2 TYPOGRAPHIC HEADLINE SCORING
# ---------------------------------------------------------------------


def anchor_typography(
    anchor: HeadlineAnchor,
    body_height: float,
    page_shape: tuple[int, int],
) -> dict[str, float]:
    page_h, page_w = page_shape

    heights = [
        max(1, line.height)
        for line in anchor.lines
    ]

    max_height = max(heights) if heights else max(
        1,
        anchor.bbox[3] - anchor.bbox[1],
    )

    mean_height = (
        float(np.mean(heights))
        if heights
        else float(max_height)
    )

    width_ratio = (
        max(1, anchor.bbox[2] - anchor.bbox[0])
        / max(1, page_w)
    )

    height_ratio = max_height / max(1.0, body_height)
    mean_height_ratio = mean_height / max(1.0, body_height)

    visual_strength = (
        1.80 * min(4.0, height_ratio)
        + 0.65 * min(1.0, width_ratio / 0.65)
        + 0.15 * min(1.0, anchor.score / 8.0)
    )

    return {
        "max_height": float(max_height),
        "mean_height": float(mean_height),
        "height_ratio": float(height_ratio),
        "mean_height_ratio": float(mean_height_ratio),
        "width_ratio": float(width_ratio),
        "visual_strength": float(visual_strength),
    }


def sentence_like_body_fragment(
    anchor: HeadlineAnchor,
    body_height: float,
    config: HybridPipelineConfigV4,
) -> bool:
    title = clean_headline_ocr_artifacts(anchor.title)
    words = title.split()

    if not 4 <= len(words) <= config.body_fragment_max_words:
        return False

    max_height = max(
        [line.height for line in anchor.lines]
        or [anchor.bbox[3] - anchor.bbox[1]]
    )

    height_ratio = max_height / max(1.0, body_height)

    if height_ratio > config.body_fragment_max_height_ratio:
        return False

    low_uppercase = uppercase_ratio(title) < 0.25
    low_titlecase = title_case_ratio(title) < 0.45

    punctuation_sentence = (
        title.endswith(".")
        or (
            "," in title
            and len(words) >= 5
        )
    )

    common_body_starters = (
        "under ", "the ", "this ", "that ", "these ", "those ",
        "he ", "she ", "they ", "it ", "there ", "after ", "before ",
        "during ", "according ", "meanwhile ", "however ", "while ",
    )

    starts_like_body = title.lower().startswith(common_body_starters)

    return (
        low_uppercase
        and low_titlecase
        and (punctuation_sentence or starts_like_body)
    )


def filter_headline_anchors_v4(
    anchors: Sequence[HeadlineAnchor],
    page_shape: tuple[int, int],
    body_height: float,
    config: HybridPipelineConfigV4,
) -> list[HeadlineAnchor]:
    cleaned: list[HeadlineAnchor] = []

    for anchor in anchors:
        cleaned_title = clean_headline_ocr_artifacts(anchor.title)

        candidate = HeadlineAnchor(
            lines=list(anchor.lines),
            title=cleaned_title,
            bbox=anchor.bbox,
            score=anchor.score,
        )

        if not headline_text_is_valid(cleaned_title, config):
            continue

        if sentence_like_body_fragment(
            candidate,
            body_height,
            config,
        ):
            continue

        cleaned.append(candidate)

    # Deduplicate nearly identical headline anchors.
    cleaned = sorted(
        cleaned,
        key=lambda item: (item.bbox[1], item.bbox[0]),
    )

    kept: list[HeadlineAnchor] = []

    for anchor in cleaned:
        duplicate_index = None

        for index, existing in enumerate(kept):
            similarity = title_similarity(anchor.title, existing.title)
            iou = bbox_iou(anchor.bbox, existing.bbox)
            containment = bbox_containment(anchor.bbox, existing.bbox)

            if (
                similarity >= 0.64
                and (
                    iou >= 0.10
                    or containment >= 0.25
                )
            ):
                duplicate_index = index
                break

        if duplicate_index is None:
            kept.append(anchor)
            continue

        existing = kept[duplicate_index]

        existing_strength = anchor_typography(
            existing,
            body_height,
            page_shape,
        )["visual_strength"]

        new_strength = anchor_typography(
            anchor,
            body_height,
            page_shape,
        )["visual_strength"]

        if (
            len(anchor.title.split())
            > len(existing.title.split())
            and new_strength >= existing_strength * 0.70
        ):
            kept[duplicate_index] = anchor
        elif new_strength > existing_strength:
            kept[duplicate_index] = anchor

    return sorted(
        kept,
        key=lambda item: (item.bbox[1], item.bbox[0]),
    )


# ---------------------------------------------------------------------
# 22.3 DOMINANT HEADLINE RESCUE
# ---------------------------------------------------------------------


def rescue_dominant_headlines_v4(
    lines: Sequence[OCRLine],
    body_height: float,
    page_shape: tuple[int, int],
    content_area: tuple[int, int],
    config: HybridPipelineConfigV4,
) -> list[HeadlineAnchor]:
    """
    Recover very large headlines even when the generic title detector/model
    fails to create a usable anchor.
    """
    content_start, content_end = content_area

    candidates = []

    for line in lines:
        if not (
            content_start <= line.cy <= content_end
        ):
            continue

        if line.height < (
            config.dominant_headline_height_ratio
            * body_height
        ):
            continue

        if not (
            2 <= line.word_count
            <= config.dominant_headline_max_words_per_line
        ):
            continue

        text = clean_headline_ocr_artifacts(line.text)

        if not headline_text_is_valid(text, config):
            continue

        if is_masthead_or_metadata(text):
            continue

        candidates.append(line)

    if not candidates:
        return []

    candidates = sorted(
        candidates,
        key=lambda line: (line.y1, line.x1),
    )

    groups: list[list[OCRLine]] = []

    for line in candidates:
        if not groups:
            groups.append([line])
            continue

        previous_group = groups[-1]
        previous = previous_group[-1]

        vertical_gap = line.y1 - previous.y2

        previous_box = bbox_from_lines(previous_group)
        line_box = (
            line.x1,
            line.y1,
            line.x2,
            line.y2,
        )

        relation = max(
            horizontal_overlap_boxes(
                previous_box,
                line_box,
            ),
            0.65 * (
                1.0
                - abs(
                    (
                        (previous_box[0] + previous_box[2]) / 2
                    )
                    - line.cx
                )
                / max(
                    1.0,
                    (
                        previous_box[2] - previous_box[0]
                        + line.width
                    )
                    / 2.0,
                )
            ),
        )

        max_gap = (
            config.dominant_headline_group_gap_ratio
            * max(previous.height, line.height)
        )

        if (
            vertical_gap <= max_gap
            and relation >= 0.20
        ):
            previous_group.append(line)
        else:
            groups.append([line])

    anchors = []

    for group in groups:
        title = clean_headline_ocr_artifacts(
            " ".join(
                line.text.strip()
                for line in group
            )
        )

        if not headline_text_is_valid(title, config):
            continue

        max_ratio = max(
            line.height
            / max(1.0, body_height)
            for line in group
        )

        anchors.append(
            HeadlineAnchor(
                lines=group,
                title=title,
                bbox=bbox_from_lines(group),
                score=round(
                    7.5
                    + min(4.0, max_ratio),
                    3,
                ),
            )
        )

    return anchors


# ---------------------------------------------------------------------
# 22.4 HEADLINE HIERARCHY
# ---------------------------------------------------------------------


def _headline_horizontal_relation(
    first: HeadlineAnchor,
    second: HeadlineAnchor,
) -> float:
    overlap = horizontal_overlap_boxes(
        first.bbox,
        second.bbox,
    )

    first_cx = (
        first.bbox[0] + first.bbox[2]
    ) / 2.0

    second_cx = (
        second.bbox[0] + second.bbox[2]
    ) / 2.0

    mean_width = max(
        1.0,
        (
            first.bbox[2] - first.bbox[0]
            + second.bbox[2] - second.bbox[0]
        )
        / 2.0,
    )

    center_relation = max(
        0.0,
        1.0
        - abs(first_cx - second_cx)
        / mean_width,
    )

    return max(
        overlap,
        0.70 * center_relation,
    )


def body_evidence_between_headlines(
    upper: HeadlineAnchor,
    lower: HeadlineAnchor,
    lines: Sequence[OCRLine],
    body_height: float,
    config: HybridPipelineConfigV4,
) -> bool:
    y1 = upper.bbox[3]
    y2 = lower.bbox[1]

    if y2 <= y1:
        return False

    left = min(
        upper.bbox[0],
        lower.bbox[0],
    )
    right = max(
        upper.bbox[2],
        lower.bbox[2],
    )

    anchor_ids = {
        line.uid
        for line in upper.lines + lower.lines
    }

    body_lines = []

    for line in lines:
        if line.uid in anchor_ids:
            continue

        if not (
            y1 <= line.cy <= y2
        ):
            continue

        if not (
            left - 20
            <= line.cx
            <= right + 20
        ):
            continue

        if (
            0.55 * body_height
            <= line.height
            <= 1.35 * body_height
            and line.word_count >= 3
        ):
            body_lines.append(line)

    return (
        len(body_lines)
        >= config.hierarchy_body_lines_to_block_merge
    )


def strong_horizontal_separator_between(
    upper: HeadlineAnchor,
    lower: HeadlineAnchor,
    separators: Sequence[Separator],
) -> bool:
    y1 = upper.bbox[3]
    y2 = lower.bbox[1]

    overlap_left = max(
        upper.bbox[0],
        lower.bbox[0],
    )

    overlap_right = min(
        upper.bbox[2],
        lower.bbox[2],
    )

    if overlap_right <= overlap_left:
        overlap_left = min(
            upper.bbox[0],
            lower.bbox[0],
        )
        overlap_right = max(
            upper.bbox[2],
            lower.bbox[2],
        )

    span = max(
        1,
        overlap_right - overlap_left,
    )

    for separator in separators:
        if separator.orientation != "horizontal":
            continue

        if not (
            y1 <= separator.y <= y2
        ):
            continue

        horizontal_overlap = max(
            0,
            min(overlap_right, separator.x2)
            - max(overlap_left, separator.x1),
        )

        if (
            horizontal_overlap / span >= 0.55
            and separator.strength >= 0.30
        ):
            return True

    return False


def headline_hierarchy_edge(
    upper: HeadlineAnchor,
    lower: HeadlineAnchor,
    lines: Sequence[OCRLine],
    separators: Sequence[Separator],
    body_height: float,
    page_shape: tuple[int, int],
    config: HybridPipelineConfigV4,
) -> bool:
    if lower.bbox[1] < upper.bbox[1]:
        upper, lower = lower, upper

    page_h, _ = page_shape

    vertical_gap = max(
        0,
        lower.bbox[1] - upper.bbox[3],
    )

    max_gap = min(
        config.hierarchy_max_vertical_gap_body_heights
        * body_height,
        config.hierarchy_max_page_gap_ratio
        * page_h,
    )

    if vertical_gap > max_gap:
        return False

    relation = _headline_horizontal_relation(
        upper,
        lower,
    )

    if relation < config.hierarchy_min_horizontal_relation:
        return False

    if strong_horizontal_separator_between(
        upper,
        lower,
        separators,
    ):
        return False

    if body_evidence_between_headlines(
        upper,
        lower,
        lines,
        body_height,
        config,
    ):
        return False

    upper_type = anchor_typography(
        upper,
        body_height,
        page_shape,
    )

    lower_type = anchor_typography(
        lower,
        body_height,
        page_shape,
    )

    # Two similarly strong, independent headlines should not be merged
    # unless they are very close.
    both_strong = (
        upper_type["height_ratio"]
        >= config.hierarchy_strong_title_ratio
        and lower_type["height_ratio"]
        >= config.hierarchy_strong_title_ratio
    )

    if both_strong and (
        vertical_gap
        > config.hierarchy_two_strong_title_gap_body_heights
        * body_height
    ):
        return False

    return True


def select_main_headline_from_group(
    group: Sequence[HeadlineAnchor],
    body_height: float,
    page_shape: tuple[int, int],
) -> HeadlineAnchor:
    """
    Main headline is selected by typography, not simply the first title.
    This lets a small kicker above a huge headline become context rather
    than the article title.
    """
    scored = []

    for anchor in group:
        metrics = anchor_typography(
            anchor,
            body_height,
            page_shape,
        )

        # Slight preference for informative multiword titles.
        word_bonus = min(
            0.30,
            len(anchor.title.split()) / 45.0,
        )

        scored.append(
            (
                metrics["visual_strength"] + word_bonus,
                anchor,
            )
        )

    scored.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    return scored[0][1]


def merge_headline_hierarchies_v4(
    anchors: Sequence[HeadlineAnchor],
    lines: Sequence[OCRLine],
    separators: Sequence[Separator],
    body_height: float,
    page_shape: tuple[int, int],
    config: HybridPipelineConfigV4,
) -> list[HeadlineAnchor]:
    """
    Merge stacked title-like blocks that form one newspaper article hierarchy:
        kicker
        MAIN HEADLINE
        deck / subheadline
        second deck

    Main title is chosen by typography.
    """
    anchors = sorted(
        anchors,
        key=lambda item: (item.bbox[1], item.bbox[0]),
    )

    if not anchors:
        return []

    parent = list(range(len(anchors)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first, second):
        root_a = find(first)
        root_b = find(second)

        if root_a != root_b:
            parent[root_b] = root_a

    # Only nearby anchors can form a hierarchy.
    for first_index, first in enumerate(anchors):
        for second_index in range(
            first_index + 1,
            min(len(anchors), first_index + 5),
        ):
            second = anchors[second_index]

            if headline_hierarchy_edge(
                first,
                second,
                lines,
                separators,
                body_height,
                page_shape,
                config,
            ):
                union(
                    first_index,
                    second_index,
                )

    groups: dict[int, list[HeadlineAnchor]] = {}

    for index, anchor in enumerate(anchors):
        groups.setdefault(
            find(index),
            [],
        ).append(anchor)

    merged = []

    for group in groups.values():
        group = sorted(
            group,
            key=lambda item: (
                item.bbox[1],
                item.bbox[0],
            ),
        )

        if len(group) == 1:
            merged.append(group[0])
            continue

        main = select_main_headline_from_group(
            group,
            body_height,
            page_shape,
        )

        all_lines = sorted(
            {
                line.uid: line
                for anchor in group
                for line in anchor.lines
            }.values(),
            key=lambda line: (
                line.y1,
                line.x1,
            ),
        )

        union_bbox = (
            min(anchor.bbox[0] for anchor in group),
            min(anchor.bbox[1] for anchor in group),
            max(anchor.bbox[2] for anchor in group),
            max(anchor.bbox[3] for anchor in group),
        )

        merged.append(
            HeadlineAnchor(
                lines=all_lines,
                title=clean_headline_ocr_artifacts(
                    main.title
                ),
                bbox=union_bbox,
                score=max(
                    anchor.score
                    for anchor in group
                )
                + min(
                    1.2,
                    0.20 * (len(group) - 1),
                ),
            )
        )

    merged = sorted(
        merged,
        key=lambda item: (item.bbox[1], item.bbox[0]),
    )

    # Suppress section-only labels after hierarchy grouping.
    output = []

    for anchor in merged:
        if probable_section_heading(
            anchor,
            merged,
            page_shape,
            config,
        ):
            continue

        output.append(anchor)

    return output


# ---------------------------------------------------------------------
# 22.5 ARTICLE TEXT WITH DECK/SUBHEAD PRESERVATION
# ---------------------------------------------------------------------


def _normalized_word_set(text: str) -> set[str]:
    return set(
        re.findall(
            r"[a-z0-9]+",
            text.lower(),
        )
    )


def hierarchy_context_lines(
    anchor: HeadlineAnchor,
) -> list[str]:
    """
    Return kicker/deck lines that are part of the headline hierarchy but
    are not simply lines of the selected main headline.
    """
    title_tokens = _normalized_word_set(
        anchor.title
    )

    context = []

    for line in sorted(
        anchor.lines,
        key=lambda item: (item.y1, item.x1),
    ):
        text = clean_headline_ocr_artifacts(
            line.text
        )

        if not text:
            continue

        line_tokens = _normalized_word_set(
            text
        )

        if not line_tokens:
            continue

        overlap = (
            len(line_tokens & title_tokens)
            / max(1, len(line_tokens))
        )

        # Main headline lines strongly overlap the chosen title.
        if overlap >= 0.72:
            continue

        if looks_like_byline(text):
            continue

        context.append(text)

    # Deduplicate while keeping order.
    seen = set()
    clean_context = []

    for item in context:
        key = re.sub(
            r"\W+",
            "",
            item.lower(),
        )

        if not key or key in seen:
            continue

        seen.add(key)
        clean_context.append(item)

    return clean_context


def article_lines_to_text(
    lines: Sequence[OCRLine],
    anchor: HeadlineAnchor,
    document_type: str,
    body_height: float,
) -> str:
    headline_ids = {
        line.uid
        for line in anchor.lines
    }

    region_box = bbox_from_lines(lines)

    columns = detect_body_columns(
        lines,
        region_box,
        headline_ids,
        body_height,
    )

    ordered_body: list[OCRLine] = []

    for column in columns:
        ordered_body.extend(column)

    raw_lines = [
        line.text
        for line in ordered_body
    ]

    cleaned_lines = clean_ocr_lines(
        raw_lines,
        document_type,
    )

    cleaned_lines = join_hyphenated_lines(
        cleaned_lines
    )

    while (
        cleaned_lines
        and looks_like_byline(cleaned_lines[0])
    ):
        cleaned_lines.pop(0)

    body = remove_newspaper_artifacts(
        " ".join(cleaned_lines)
    )

    title = clean_headline_ocr_artifacts(
        anchor.title
    )

    context = hierarchy_context_lines(
        anchor
    )

    # Keep hierarchy context before the body; this helps category detection
    # without turning each deck/subheadline into a separate article.
    context_text = ". ".join(
        context
    ).strip()

    parts = [title]

    if context_text:
        parts.append(context_text)

    if body:
        # Avoid body repeating the selected headline/context.
        body_low = body.lower()

        for prefix in [title] + context:
            prefix = prefix.strip()
            if (
                prefix
                and body_low.startswith(
                    prefix.lower()
                )
            ):
                body = body[
                    len(prefix):
                ].lstrip(" .:-")
                body_low = body.lower()

        if body:
            parts.append(body)

    return ". ".join(
        part.strip(" .")
        for part in parts
        if part.strip(" .")
    ).strip() + "."


# ---------------------------------------------------------------------
# 22.6 FINAL DUPLICATE / PREFIX ARTICLE MERGING
# ---------------------------------------------------------------------


def _region_center_distance_ratio(
    first_box: tuple[int, int, int, int],
    second_box: tuple[int, int, int, int],
) -> float:
    first_cx = (
        first_box[0] + first_box[2]
    ) / 2.0
    first_cy = (
        first_box[1] + first_box[3]
    ) / 2.0

    second_cx = (
        second_box[0] + second_box[2]
    ) / 2.0
    second_cy = (
        second_box[1] + second_box[3]
    ) / 2.0

    union_left = min(
        first_box[0],
        second_box[0],
    )
    union_top = min(
        first_box[1],
        second_box[1],
    )
    union_right = max(
        first_box[2],
        second_box[2],
    )
    union_bottom = max(
        first_box[3],
        second_box[3],
    )

    diagonal = math.hypot(
        max(1, union_right - union_left),
        max(1, union_bottom - union_top),
    )

    return math.hypot(
        first_cx - second_cx,
        first_cy - second_cy,
    ) / max(1.0, diagonal)


def region_line_shared_ratio(
    first: dict[str, Any],
    second: dict[str, Any],
) -> float:
    first_ids = {
        line.uid
        for line in first.get("lines", [])
    }

    second_ids = {
        line.uid
        for line in second.get("lines", [])
    }

    if not first_ids or not second_ids:
        return 0.0

    return (
        len(first_ids & second_ids)
        / max(
            1,
            min(
                len(first_ids),
                len(second_ids),
            ),
        )
    )


def region_box_relation(
    first: dict[str, Any],
    second: dict[str, Any],
) -> float:
    return max(
        bbox_iou(
            first["bbox"],
            second["bbox"],
        ),
        bbox_containment(
            first["bbox"],
            second["bbox"],
        ),
        region_line_shared_ratio(
            first,
            second,
        ),
    )


def duplicate_region_pair_v4(
    first: dict[str, Any],
    second: dict[str, Any],
    config: HybridPipelineConfigV4,
) -> bool:
    first_title = first["anchor"].title
    second_title = second["anchor"].title

    similarity = title_similarity(
        first_title,
        second_title,
    )

    prefix = title_prefix_relation(
        first_title,
        second_title,
        config.prefix_duplicate_min_tokens,
    )

    relation = region_box_relation(
        first,
        second,
    )

    center_distance = _region_center_distance_ratio(
        first["bbox"],
        second["bbox"],
    )

    if (
        similarity >= 0.60
        and relation >= 0.18
    ):
        return True

    if (
        prefix
        and (
            relation
            >= config.prefix_duplicate_min_box_relation
            or center_distance
            <= config.prefix_duplicate_max_center_distance_ratio
        )
    ):
        return True

    return False


def choose_better_duplicate_region_v4(
    first: dict[str, Any],
    second: dict[str, Any],
) -> dict[str, Any]:
    first_title_words = len(
        _headline_tokens(
            first["anchor"].title
        )
    )
    second_title_words = len(
        _headline_tokens(
            second["anchor"].title
        )
    )

    first_words = len(
        first.get("text", "").split()
    )
    second_words = len(
        second.get("text", "").split()
    )

    # Prefix duplicates: prefer the more complete headline unless its body
    # is dramatically less complete.
    if title_prefix_relation(
        first["anchor"].title,
        second["anchor"].title,
        2,
    ):
        if (
            first_title_words > second_title_words
            and first_words >= 0.60 * max(1, second_words)
        ):
            return first

        if (
            second_title_words > first_title_words
            and second_words >= 0.60 * max(1, first_words)
        ):
            return second

    first_quality = (
        _region_quality_for_fusion(first)
        + min(0.18, first_title_words / 80.0)
    )

    second_quality = (
        _region_quality_for_fusion(second)
        + min(0.18, second_title_words / 80.0)
    )

    if second_quality > first_quality:
        return second

    return first


def fuse_article_regions(
    geometry_regions: Sequence[dict[str, Any]],
    doclayout_regions: Sequence[dict[str, Any]],
    config: HybridPipelineConfig,
) -> list[dict[str, Any]]:
    """
    V4 hybrid fusion:
    - prefer model-assisted complete article candidates
    - retain geometry-only articles missed by the model
    - merge prefix/contained duplicates
    """
    candidates = list(
        doclayout_regions
    )

    if config.hybrid_keep_geometry_fallback:
        candidates.extend(
            geometry_regions
        )

    candidates = [
        region
        for region in candidates
        if (
            headline_text_is_valid(
                region["anchor"].title,
                config,
            )
            and not looks_like_byline(
                region["anchor"].title
            )
        )
    ]

    candidates = sorted(
        candidates,
        key=lambda region: (
            region["bbox"][1],
            region["bbox"][0],
        ),
    )

    kept: list[dict[str, Any]] = []

    for candidate in candidates:
        duplicate_index = None

        for index, existing in enumerate(kept):
            if duplicate_region_pair_v4(
                candidate,
                existing,
                config,
            ):
                duplicate_index = index
                break

        if duplicate_index is None:
            kept.append(candidate)
            continue

        kept[duplicate_index] = choose_better_duplicate_region_v4(
            candidate,
            kept[duplicate_index],
        )

    # Existing duplicate logic remains useful as a second pass.
    kept = deduplicate_and_merge_regions(
        kept,
        config,
    )

    final = []

    for region in kept:
        region["text"] = remove_newspaper_artifacts(
            region.get("text", "")
        )

        word_count = len(
            region["text"].split()
        )

        min_words = getattr(
            config,
            "min_final_article_words",
            25,
        )

        if word_count < min_words:
            continue

        if sentence_like_body_fragment(
            region["anchor"],
            # estimate from headline lines when exact body height isn't available
            max(
                1.0,
                np.median(
                    [
                        line.height
                        for line in region.get("lines", [])
                        if line.uid not in {
                            item.uid
                            for item in region["anchor"].lines
                        }
                    ]
                    or [12.0]
                ),
            ),
            config,
        ):
            continue

        final.append(region)

    return sorted(
        final,
        key=lambda region: (
            region["bbox"][1],
            region["bbox"][0],
        ),
    )


# ---------------------------------------------------------------------
# 22.7 V4 HYBRID DETECTOR
# ---------------------------------------------------------------------


def detect_hybrid_article_regions(
    lines: Sequence[OCRLine],
    prepared_gray: np.ndarray,
    prepared_color: np.ndarray,
    document_type: str,
    config: HybridPipelineConfig,
    doclayout_model=None,
) -> dict[str, Any]:
    page_shape = prepared_gray.shape[:2]

    separators = detect_page_separators(
        prepared_gray
    )

    body_height = estimate_body_height(
        lines,
        page_shape,
    )

    content_area = detect_adaptive_content_area(
        lines,
        separators,
        page_shape,
        body_height,
        config,
    )

    usable_lines = [
        line
        for line in lines
        if (
            content_area[0]
            <= line.cy
            <= content_area[1]
        )
        and not is_noise_line(
            line.text,
            document_type,
        )
    ]

    # ----------------------------------------------------------
    # A. Geometry headline detector
    # ----------------------------------------------------------

    candidates = detect_headline_candidates(
        usable_lines,
        body_height,
        page_shape,
        content_area,
        config,
    )

    geometry_anchors = combine_multiline_headlines(
        candidates,
        usable_lines,
        body_height,
        page_shape,
    )

    geometry_anchors = merge_short_headline_continuations(
        geometry_anchors,
        body_height,
    )

    # Very large headline rescue independent of the normal detector.
    dominant_anchors = rescue_dominant_headlines_v4(
        usable_lines,
        body_height,
        page_shape,
        content_area,
        config,
    )

    geometry_anchors = fuse_headline_anchors(
        geometry_anchors,
        dominant_anchors,
    )

    # ----------------------------------------------------------
    # B. DocLayout-YOLO structural elements
    # ----------------------------------------------------------

    layout_elements: list[LayoutElement] = []
    model_anchors: list[HeadlineAnchor] = []
    model_warning = None

    if (
        config.use_doclayout_yolo
        and doclayout_model is not None
    ):
        try:
            layout_elements = detect_doclayout_elements(
                doclayout_model,
                prepared_color,
                config,
            )

            model_anchors = anchors_from_doclayout(
                layout_elements,
                usable_lines,
                document_type,
                content_area,
                config,
            )

        except Exception as error:
            model_warning = (
                "DocLayout-YOLO inference failed. "
                "Geometry fallback used. "
                f"Reason: {error}"
            )

    # ----------------------------------------------------------
    # C. Pre-filter -> headline hierarchy -> section filtering
    # ----------------------------------------------------------

    raw_fused = fuse_headline_anchors(
        geometry_anchors,
        model_anchors,
    )

    basic_filtered = filter_headline_anchors_v4(
        raw_fused,
        page_shape,
        body_height,
        config,
    )

    anchors = merge_headline_hierarchies_v4(
        basic_filtered,
        usable_lines,
        separators,
        body_height,
        page_shape,
        config,
    )

    geometry_debug = merge_headline_hierarchies_v4(
        filter_headline_anchors_v4(
            geometry_anchors,
            page_shape,
            body_height,
            config,
        ),
        usable_lines,
        separators,
        body_height,
        page_shape,
        config,
    )

    model_debug = merge_headline_hierarchies_v4(
        filter_headline_anchors_v4(
            model_anchors,
            page_shape,
            body_height,
            config,
        ),
        usable_lines,
        separators,
        body_height,
        page_shape,
        config,
    )

    # ----------------------------------------------------------
    # D. Geometry article candidates
    # ----------------------------------------------------------

    geometry_regions = build_provisional_article_regions(
        usable_lines,
        anchors,
        separators,
        content_area,
        page_shape,
        body_height,
        config,
    )

    for region in geometry_regions:
        region["source"] = (
            "adaptive_headline_hierarchy_geometry_v4"
        )
        region["text"] = remove_newspaper_artifacts(
            region["text"]
        )

    geometry_regions = deduplicate_and_merge_regions(
        geometry_regions,
        config,
    )

    # ----------------------------------------------------------
    # E. DocLayout-assisted article candidates
    # ----------------------------------------------------------

    model_regions = build_doclayout_assisted_regions(
        usable_lines,
        anchors,
        layout_elements,
        page_shape,
        body_height,
        document_type,
        config,
        separators=separators,
    )

    # The V3 function labels these V3; update source for diagnostics.
    for region in model_regions:
        region["source"] = (
            "doclayout_yolo_headline_hierarchy_grouping_v4"
        )

    # ----------------------------------------------------------
    # F. Hybrid fusion + prefix duplicate merging
    # ----------------------------------------------------------

    final_regions = fuse_article_regions(
        geometry_regions,
        model_regions,
        config,
    )

    warnings: list[str] = []

    if model_warning:
        warnings.append(model_warning)

    if not anchors:
        warnings.append(
            "No reliable headline hierarchy was detected."
        )

    if not final_regions:
        warnings.append(
            "Hybrid V4 article segmentation failed; "
            "the full-page fallback will be used."
        )

    if len(final_regions) > 22:
        warnings.append(
            "High article count suggests possible over-segmentation."
        )

    class_counts = doclayout_class_counts(
        layout_elements
    )

    if (
        getattr(
            config,
            "print_doclayout_class_counts",
            False,
        )
        and config.debug
    ):
        print(
            "DocLayout normalized class counts:",
            class_counts,
        )

    return {
        "regions": final_regions,
        "anchors": anchors,
        "geometry_anchors": geometry_debug,
        "doclayout_anchors": model_debug,
        "layout_elements": layout_elements,
        "separators": separators,
        "body_height": body_height,
        "content_area": content_area,
        "warnings": warnings,
        "doclayout_class_counts": class_counts,
        "debug_counts": {
            "geometry_anchor_count": len(
                geometry_debug
            ),
            "doclayout_anchor_count": len(
                model_debug
            ),
            "dominant_headline_rescue_count": len(
                dominant_anchors
            ),
            "raw_fused_anchor_count": len(
                raw_fused
            ),
            "basic_filtered_anchor_count": len(
                basic_filtered
            ),
            "fused_anchor_count": len(
                anchors
            ),
            "doclayout_element_count": len(
                layout_elements
            ),
            "geometry_region_count": len(
                geometry_regions
            ),
            "doclayout_region_count": len(
                model_regions
            ),
            "final_region_count": len(
                final_regions
            ),
        },
    }


# ---------------------------------------------------------------------
# 22.8 V4 ANALYSIS WRAPPER
# ---------------------------------------------------------------------


def analyze_content_hybrid(
    image_path: str,
    document_type: str,
    model,
    config: HybridPipelineConfig,
    doclayout_model=None,
) -> dict[str, Any]:
    normalized_type = normalize_document_type(
        document_type
    )

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        raise FileNotFoundError(
            f"Image not found or unreadable: {image_path}"
        )

    prepared_color, prepared_gray, scale = prepare_page_image(
        image,
        config,
    )

    page_shape = prepared_gray.shape[:2]
    warnings: list[str] = []
    layout_debug: dict[str, Any] = {}

    if normalized_type in MULTI_CATEGORY_TYPES:
        ocr_lines = extract_ocr_lines(
            prepared_gray,
            normalized_type,
            config,
        )

        layout = detect_hybrid_article_regions(
            ocr_lines,
            prepared_gray,
            prepared_color,
            normalized_type,
            config,
            doclayout_model=doclayout_model,
        )

        warnings.extend(
            layout["warnings"]
        )

        layout_debug = {
            "pipeline_version": "V4_HEADLINE_HIERARCHY",
            "ocr_line_count": len(
                ocr_lines
            ),
            "body_height": round(
                float(layout["body_height"]),
                2,
            ),
            "content_area": layout[
                "content_area"
            ],
            **layout["debug_counts"],
            "doclayout_class_counts": (
                layout.get(
                    "doclayout_class_counts",
                    {},
                )
            ),
            "headline_anchors": [
                {
                    "title": anchor.title,
                    "bbox": anchor.bbox,
                    "score": round(
                        float(anchor.score),
                        3,
                    ),
                    "hierarchy_line_count": len(
                        anchor.lines
                    ),
                }
                for anchor in layout[
                    "anchors"
                ]
            ],
            "doclayout_elements": [
                {
                    "class_name": (
                        normalize_layout_class_name(
                            element.class_name
                        )
                    ),
                    "raw_class_name": (
                        element.class_name
                    ),
                    "confidence": round(
                        float(
                            element.confidence
                        ),
                        4,
                    ),
                    "bbox": element.bbox,
                }
                for element in layout[
                    "layout_elements"
                ]
            ],
            "separators": [
                asdict(separator)
                for separator in layout[
                    "separators"
                ]
            ],
        }

        results: list[ArticleRegion] = []

        for number, region in enumerate(
            layout["regions"],
            start=1,
        ):
            anchor = region["anchor"]

            clean_region_text = remove_newspaper_artifacts(
                region["text"]
            )

            prediction = predict_category(
                model,
                clean_region_text,
                normalized_type,
                anchor.title,
                config,
            )

            results.append(
                ArticleRegion(
                    article_number=number,
                    title=clean_headline_ocr_artifacts(
                        anchor.title
                    ),
                    bbox=region[
                        "bbox"
                    ],
                    normalized_bbox=normalize_bbox(
                        region["bbox"],
                        page_shape,
                    ),
                    full_text=clean_region_text,
                    category=prediction[
                        "category"
                    ],
                    confidence=prediction[
                        "confidence"
                    ],
                    probability_margin=prediction[
                        "margin"
                    ],
                    top_predictions=prediction[
                        "top_predictions"
                    ],
                    ocr_quality=ocr_quality_score(
                        clean_region_text
                    ),
                    layout_confidence=float(
                        region[
                            "layout_confidence"
                        ]
                    ),
                    warnings=list(
                        region.get(
                            "warnings",
                            [],
                        )
                    ),
                    source=region.get(
                        "source",
                        "hybrid_v4",
                    ),
                )
            )

        if not results:
            full_text = remove_newspaper_artifacts(
                best_full_page_ocr(
                    prepared_gray,
                    normalized_type,
                )
            )

            prediction = predict_category(
                model,
                full_text,
                normalized_type,
                "Full page",
                config,
            )

            results = [
                ArticleRegion(
                    article_number=1,
                    title="Full page",
                    bbox=(
                        0,
                        0,
                        page_shape[1],
                        page_shape[0],
                    ),
                    normalized_bbox=(
                        0,
                        0,
                        1000,
                        1000,
                    ),
                    full_text=full_text,
                    category=prediction[
                        "category"
                    ],
                    confidence=prediction[
                        "confidence"
                    ],
                    probability_margin=prediction[
                        "margin"
                    ],
                    top_predictions=prediction[
                        "top_predictions"
                    ],
                    ocr_quality=ocr_quality_score(
                        full_text
                    ),
                    layout_confidence=0.20,
                    warnings=[
                        "Hybrid V4 article segmentation fallback: "
                        "the full page was processed as one region."
                    ],
                    source="full_page_fallback",
                )
            ]

    else:
        full_text = remove_newspaper_artifacts(
            best_full_page_ocr(
                prepared_gray,
                normalized_type,
            )
        )

        prediction = predict_category(
            model,
            full_text,
            normalized_type,
            "Full page",
            config,
        )

        results = [
            ArticleRegion(
                article_number=1,
                title="Full page",
                bbox=(
                    0,
                    0,
                    page_shape[1],
                    page_shape[0],
                ),
                normalized_bbox=(
                    0,
                    0,
                    1000,
                    1000,
                ),
                full_text=full_text,
                category=prediction[
                    "category"
                ],
                confidence=prediction[
                    "confidence"
                ],
                probability_margin=prediction[
                    "margin"
                ],
                top_predictions=prediction[
                    "top_predictions"
                ],
                ocr_quality=ocr_quality_score(
                    full_text
                ),
                layout_confidence=1.0,
                warnings=[],
                source="full_page_tesseract",
            )
        ]

    category_counts = Counter(
        result.category
        for result in results
    )

    categories = sorted(
        category_counts
    )

    mode = (
        "Multi Category"
        if normalized_type in MULTI_CATEGORY_TYPES
        else "Single Category"
    )

    return {
        "status": "ANALYZED",
        "pipeline_version": "V4_HEADLINE_HIERARCHY",
        "document_type_received": document_type,
        "document_type": normalized_type,
        "mode": mode,
        "detected_categories": categories,
        "category_counts": dict(
            category_counts
        ),
        "needs_category_selection": (
            normalized_type in MULTI_CATEGORY_TYPES
            and len(categories) > 1
        ),
        "image_shape": {
            "height": page_shape[0],
            "width": page_shape[1],
        },
        "resize_scale": scale,
        "warnings": warnings,
        "layout_debug": layout_debug,
        "results": [
            asdict(result)
            for result in results
        ],
        "processing": {
            "cloud_api_used": False,
            "ocr": "local_tesseract",
            "pretrained_layout_model": (
                "local_DocLayout_YOLO_DocStructBench"
                if (
                    config.use_doclayout_yolo
                    and doclayout_model is not None
                )
                else "disabled"
            ),
            "article_detection": (
                "V4_headline_hierarchy_doclayout_custom_grouping_geometry_fallback"
            ),
            "category_model": (
                "local_tfidf_logistic_regression"
            ),
            "speech_input": "local_vosk",
            "summary_model": (
                f"local_ollama_{config.ollama_model}"
            ),
        },
    }


# ---------------------------------------------------------------------
# 22.9 V4 DEBUG TABLES
# ---------------------------------------------------------------------


def hybrid_debug_summary(
    analysis: dict[str, Any],
) -> pd.DataFrame:
    debug = analysis.get(
        "layout_debug",
        {},
    )

    rows = [
        (
            "Pipeline version",
            debug.get(
                "pipeline_version",
                analysis.get(
                    "pipeline_version",
                    "",
                ),
            ),
        ),
        (
            "OCR lines",
            debug.get(
                "ocr_line_count",
                0,
            ),
        ),
        (
            "Geometry headline hierarchies",
            debug.get(
                "geometry_anchor_count",
                0,
            ),
        ),
        (
            "DocLayout headline hierarchies",
            debug.get(
                "doclayout_anchor_count",
                0,
            ),
        ),
        (
            "Dominant headlines rescued",
            debug.get(
                "dominant_headline_rescue_count",
                0,
            ),
        ),
        (
            "Raw fused headline anchors",
            debug.get(
                "raw_fused_anchor_count",
                0,
            ),
        ),
        (
            "Basic filtered anchors",
            debug.get(
                "basic_filtered_anchor_count",
                0,
            ),
        ),
        (
            "Final headline hierarchies",
            debug.get(
                "fused_anchor_count",
                0,
            ),
        ),
        (
            "DocLayout elements",
            debug.get(
                "doclayout_element_count",
                0,
            ),
        ),
        (
            "Geometry regions",
            debug.get(
                "geometry_region_count",
                0,
            ),
        ),
        (
            "DocLayout-assisted regions",
            debug.get(
                "doclayout_region_count",
                0,
            ),
        ),
        (
            "Final fused articles",
            debug.get(
                "final_region_count",
                len(
                    analysis.get(
                        "results",
                        [],
                    )
                ),
            ),
        ),
    ]

    return pd.DataFrame(
        rows,
        columns=["Stage", "Value"],
    )


def headline_hierarchy_table(
    analysis: dict[str, Any],
) -> pd.DataFrame:
    anchors = analysis.get(
        "layout_debug",
        {},
    ).get(
        "headline_anchors",
        [],
    )

    return pd.DataFrame(
        anchors
    )


# ---------------------------------------------------------------------
# 22.10 ONE-CALL V4 VOICE PIPELINE
# ---------------------------------------------------------------------


def run_hybrid_voice_pipeline_v4(
    image_path: str,
    document_type: str,
    category_model,
    config: HybridPipelineConfigV4,
    doclayout_model,
    voice_selector: OfflineVoiceSelector,
    tts_enabled: bool = False,
    clean_full_text_with_llama: bool = False,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
]:
    analysis = analyze_content_hybrid(
        image_path=image_path,
        document_type=document_type,
        model=category_model,
        config=config,
        doclayout_model=doclayout_model,
    )

    selected_category, depth = collect_user_preferences_voice(
        analysis=analysis,
        voice_selector=voice_selector,
        tts_enabled=tts_enabled,
    )

    output = generate_selected_output(
        analysis=analysis,
        selected_category=selected_category,
        depth=depth,
        config=config,
        clean_full_text_with_llama=clean_full_text_with_llama,
    )

    return analysis, output


# =====================================================================
# 23. V5 — ADAPTIVE LOCAL GRAPH-BASED NEWSPAPER PIPELINE
# =====================================================================
#
# IMPORTANT
# ---------
# This V5 branch is intentionally FULLY LOCAL.
#
# It DOES NOT call OpenAI or any cloud vision/OCR API.
#
# Main newspaper strategy:
#   1. Tesseract OCR -> local text + geometry
#   2. DocLayout-YOLO -> local structural elements
#   3. Geometry/OCR headline candidates -> secondary evidence
#   4. Generic headline hierarchy -> title seeds
#   5. Graph-style relationship scoring:
#        title <-> plain_text / figure / caption
#   6. Connected visible article groups
#   7. Handle article continuations at capture edges
#   8. Evaluate grouping quality
#   9. If graph grouping is weak:
#        use V4 local hybrid grouping
#  10. If both are weak:
#        use local full-visible-capture OCR fallback
#  11. Local category model
#  12. Offline Vosk user choices
#  13. Local Ollama llama3.2:3b for summaries
#
# No special code exists for:
#   - Daily News
#   - Daily Mirror
#   - Sunday Observer
#   - top half
#   - bottom half
#
# Measurements are relative to body-text height and image dimensions.
# =====================================================================


# Keep references to the previous local functions before defining V5 wrappers.
_V4_LOCAL_DETECTOR = detect_hybrid_article_regions
_V4_LOCAL_ANALYZER = analyze_content_hybrid
_V4_LOCAL_OUTPUT = generate_selected_output
_BASE_LOCAL_ANALYZER = analyze_content


@dataclass
class AdaptiveGraphPipelineConfig(HybridPipelineConfigV4):
    # ---------------------------------------------------------
    # Graph edge scoring
    # ---------------------------------------------------------
    graph_min_text_edge_score: float = 0.40
    graph_min_figure_edge_score: float = 0.34
    graph_assignment_margin: float = 0.035

    # Relative distance limits (body-height units, never fixed page pixels)
    graph_max_text_distance_body_heights: float = 24.0
    graph_max_figure_distance_body_heights: float = 20.0
    graph_max_upward_distance_body_heights: float = 3.5

    # Edge-score weights
    graph_horizontal_weight: float = 0.36
    graph_distance_weight: float = 0.30
    graph_direction_weight: float = 0.12
    graph_alignment_weight: float = 0.12
    graph_model_conf_weight: float = 0.10

    # Penalties
    graph_separator_penalty: float = 0.46
    graph_intervening_headline_penalty: float = 0.52

    # A nearby figure can bridge headline -> lower body text
    graph_figure_bridge_bonus: float = 0.16

    # ---------------------------------------------------------
    # Visible continuation handling
    # ---------------------------------------------------------
    continuation_min_words: int = 24
    continuation_min_ocr_lines: int = 3
    continuation_group_gap_body_heights: float = 11.0
    continuation_min_horizontal_relation: float = 0.16

    # Edge contact is measured as a percentage of the current capture.
    capture_edge_margin_ratio: float = 0.025

    # ---------------------------------------------------------
    # Final article filtering
    # ---------------------------------------------------------
    graph_min_article_words: int = 24
    graph_min_article_ocr_quality: float = 0.16

    # ---------------------------------------------------------
    # Local quality controller
    # ---------------------------------------------------------
    graph_quality_threshold: float = 0.53
    v4_fallback_quality_threshold: float = 0.46
    prefer_graph_if_within: float = 0.035

    # If True, V4 remains the second fully-local method.
    use_v4_local_fallback: bool = True

    # Last fallback is local OCR of the entire visible capture.
    allow_full_visible_ocr_fallback: bool = True

    # ---------------------------------------------------------
    # Debug
    # ---------------------------------------------------------
    graph_debug_edges: bool = True


# ---------------------------------------------------------------------
# 23.1 BASIC NORMALIZED GEOMETRY
# ---------------------------------------------------------------------


def _box_center(
    box: tuple[int, int, int, int],
) -> tuple[float, float]:
    return (
        (box[0] + box[2]) / 2.0,
        (box[1] + box[3]) / 2.0,
    )


def _box_union(
    boxes: Sequence[tuple[int, int, int, int]],
) -> tuple[int, int, int, int]:
    valid = [
        box
        for box in boxes
        if box is not None
        and box[2] > box[0]
        and box[3] > box[1]
    ]

    if not valid:
        return (0, 0, 1, 1)

    return (
        min(box[0] for box in valid),
        min(box[1] for box in valid),
        max(box[2] for box in valid),
        max(box[3] for box in valid),
    )


def _horizontal_relation_boxes(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> float:
    return element_horizontal_relation(
        first,
        second,
    )


def _vertical_gap(
    upper_box: tuple[int, int, int, int],
    lower_box: tuple[int, int, int, int],
) -> float:
    if lower_box[1] >= upper_box[3]:
        return float(lower_box[1] - upper_box[3])

    if upper_box[1] >= lower_box[3]:
        return float(upper_box[1] - lower_box[3])

    return 0.0


def _center_alignment_score(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> float:
    first_cx, _ = _box_center(first)
    second_cx, _ = _box_center(second)

    mean_width = max(
        1.0,
        (
            (first[2] - first[0])
            + (second[2] - second[0])
        )
        / 2.0,
    )

    return max(
        0.0,
        min(
            1.0,
            1.0 - abs(first_cx - second_cx) / mean_width,
        ),
    )


def _clamp01(value: float) -> float:
    return float(
        max(
            0.0,
            min(1.0, value),
        )
    )


# ---------------------------------------------------------------------
# 23.2 PAGE / CAPTURE EDGE FLAGS
# ---------------------------------------------------------------------


def visible_capture_edge_flags(
    bbox: tuple[int, int, int, int],
    page_shape: tuple[int, int],
    config: AdaptiveGraphPipelineConfig,
) -> dict[str, bool]:
    page_h, page_w = page_shape

    margin_y = max(
        4,
        int(
            page_h
            * config.capture_edge_margin_ratio
        ),
    )

    margin_x = max(
        4,
        int(
            page_w
            * config.capture_edge_margin_ratio
        ),
    )

    return {
        "partial_top": bbox[1] <= margin_y,
        "partial_bottom": bbox[3] >= page_h - margin_y,
        "touches_left_edge": bbox[0] <= margin_x,
        "touches_right_edge": bbox[2] >= page_w - margin_x,
    }


# ---------------------------------------------------------------------
# 23.3 CREATE GENERIC HEADLINE SEEDS
# ---------------------------------------------------------------------


def build_adaptive_headline_seeds_v5(
    usable_lines: Sequence[OCRLine],
    layout_elements: Sequence[LayoutElement],
    separators: Sequence[Separator],
    body_height: float,
    content_area: tuple[int, int],
    page_shape: tuple[int, int],
    document_type: str,
    config: AdaptiveGraphPipelineConfig,
) -> dict[str, Any]:
    # Geometry evidence
    candidates = detect_headline_candidates(
        usable_lines,
        body_height,
        page_shape,
        content_area,
        config,
    )

    geometry = combine_multiline_headlines(
        candidates,
        usable_lines,
        body_height,
        page_shape,
    )

    geometry = merge_short_headline_continuations(
        geometry,
        body_height,
    )

    # Recover unusually large title text using a relative typography rule.
    dominant = rescue_dominant_headlines_v4(
        usable_lines,
        body_height,
        page_shape,
        content_area,
        config,
    )

    geometry = fuse_headline_anchors(
        geometry,
        dominant,
    )

    # DocLayout title evidence
    model_anchors = anchors_from_doclayout(
        layout_elements,
        usable_lines,
        document_type,
        content_area,
        config,
    )

    fused = fuse_headline_anchors(
        geometry,
        model_anchors,
    )

    filtered = filter_headline_anchors_v4(
        fused,
        page_shape,
        body_height,
        config,
    )

    # Generic hierarchy:
    # kicker + main headline + deck are represented as one seed.
    hierarchical = merge_headline_hierarchies_v4(
        filtered,
        usable_lines,
        separators,
        body_height,
        page_shape,
        config,
    )

    return {
        "anchors": hierarchical,
        "geometry_anchors": geometry,
        "model_anchors": model_anchors,
        "dominant_anchors": dominant,
        "raw_fused_anchors": fused,
        "filtered_anchors": filtered,
    }


# ---------------------------------------------------------------------
# 23.4 GRAPH EDGE SCORE
# ---------------------------------------------------------------------


def graph_anchor_element_score_v5(
    anchor_index: int,
    anchor: HeadlineAnchor,
    element: LayoutElement,
    anchors: Sequence[HeadlineAnchor],
    figure_elements: Sequence[LayoutElement],
    separators: Sequence[Separator],
    body_height: float,
    page_shape: tuple[int, int],
    config: AdaptiveGraphPipelineConfig,
) -> dict[str, Any]:
    cls = normalize_layout_class_name(
        element.class_name
    )

    horizontal = _horizontal_relation_boxes(
        anchor.bbox,
        element.bbox,
    )

    alignment = _center_alignment_score(
        anchor.bbox,
        element.bbox,
    )

    # Body normally follows its headline.
    if element.y1 >= anchor.bbox[3]:
        raw_gap = float(
            element.y1 - anchor.bbox[3]
        )

        max_distance = (
            config.graph_max_text_distance_body_heights
            if cls
            in normalized_layout_class_set(
                config.doclayout_text_classes
            )
            else config.graph_max_figure_distance_body_heights
        ) * max(1.0, body_height)

        distance_score = math.exp(
            -raw_gap
            / max(
                1.0,
                0.42 * max_distance,
            )
        )

        direction_score = 1.0

    elif element.y2 <= anchor.bbox[1]:
        raw_gap = float(
            anchor.bbox[1] - element.y2
        )

        max_up = (
            config.graph_max_upward_distance_body_heights
            * max(1.0, body_height)
        )

        if raw_gap > max_up:
            return {
                "score": 0.0,
                "horizontal": horizontal,
                "alignment": alignment,
                "distance": 0.0,
                "direction": 0.0,
                "separator_block": False,
                "intervening_headline": False,
                "figure_bridge": False,
                "reason": "too_far_above_headline",
            }

        distance_score = math.exp(
            -raw_gap
            / max(
                1.0,
                0.60 * max_up,
            )
        )

        # Text slightly above can be a kicker/deck/layout artifact,
        # but is intentionally weaker than text below.
        direction_score = 0.30

    else:
        # Element vertically overlaps title seed.
        raw_gap = 0.0
        distance_score = 1.0
        direction_score = 0.72

    model_confidence = _clamp01(
        float(element.confidence)
    )

    score = (
        config.graph_horizontal_weight
        * horizontal
        + config.graph_distance_weight
        * distance_score
        + config.graph_direction_weight
        * direction_score
        + config.graph_alignment_weight
        * alignment
        + config.graph_model_conf_weight
        * model_confidence
    )

    separator_block = separator_blocks_relationship(
        anchor,
        element,
        separators,
    )

    if separator_block:
        score -= (
            config.graph_separator_penalty
        )

    intervening = False

    if element.y1 > anchor.bbox[3]:
        intervening = _intervening_headline_exists(
            anchor_index,
            element,
            anchors,
            horizontal,
        )

        if intervening:
            score -= (
                config.graph_intervening_headline_penalty
            )

    figure_bridge = False

    if cls in normalized_layout_class_set(
        config.doclayout_text_classes
    ):
        figure_bridge = _figure_bridges_anchor_to_text(
            anchor,
            element,
            figure_elements,
        )

        if figure_bridge:
            score += (
                config.graph_figure_bridge_bonus
            )

    return {
        "score": _clamp01(score),
        "horizontal": round(
            float(horizontal),
            4,
        ),
        "alignment": round(
            float(alignment),
            4,
        ),
        "distance": round(
            float(distance_score),
            4,
        ),
        "direction": round(
            float(direction_score),
            4,
        ),
        "model_confidence": round(
            float(model_confidence),
            4,
        ),
        "separator_block": bool(
            separator_block
        ),
        "intervening_headline": bool(
            intervening
        ),
        "figure_bridge": bool(
            figure_bridge
        ),
        "reason": "scored",
    }


# ---------------------------------------------------------------------
# 23.5 ASSIGN DOCLAYOUT BLOCKS TO HEADLINE SEEDS
# ---------------------------------------------------------------------


def assign_layout_elements_to_articles_v5(
    anchors: Sequence[HeadlineAnchor],
    layout_elements: Sequence[LayoutElement],
    separators: Sequence[Separator],
    body_height: float,
    page_shape: tuple[int, int],
    config: AdaptiveGraphPipelineConfig,
) -> dict[str, Any]:
    text_classes = normalized_layout_class_set(
        config.doclayout_text_classes
    )

    figure_classes = normalized_layout_class_set(
        config.doclayout_figure_classes
    )

    caption_classes = normalized_layout_class_set(
        config.doclayout_caption_classes
    )

    candidate_elements = [
        element
        for element in layout_elements
        if normalize_layout_class_name(
            element.class_name
        )
        in (
            text_classes
            | figure_classes
            | caption_classes
        )
    ]

    figures = [
        element
        for element in candidate_elements
        if normalize_layout_class_name(
            element.class_name
        )
        in figure_classes
    ]

    assignments: dict[
        int,
        list[
            tuple[
                LayoutElement,
                float,
                dict[str, Any],
            ]
        ],
    ] = {
        index: []
        for index in range(
            len(anchors)
        )
    }

    unassigned: list[LayoutElement] = []
    edge_debug: list[dict[str, Any]] = []

    for element_index, element in enumerate(
        candidate_elements
    ):
        cls = normalize_layout_class_name(
            element.class_name
        )

        scored = []

        for anchor_index, anchor in enumerate(
            anchors
        ):
            detail = graph_anchor_element_score_v5(
                anchor_index=anchor_index,
                anchor=anchor,
                element=element,
                anchors=anchors,
                figure_elements=figures,
                separators=separators,
                body_height=body_height,
                page_shape=page_shape,
                config=config,
            )

            scored.append(
                (
                    detail["score"],
                    anchor_index,
                    detail,
                )
            )

        if not scored:
            unassigned.append(element)
            continue

        scored.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        best_score, best_anchor, best_detail = scored[0]
        second_score = (
            scored[1][0]
            if len(scored) > 1
            else 0.0
        )

        min_score = (
            config.graph_min_text_edge_score
            if cls in text_classes
            else config.graph_min_figure_edge_score
        )

        # Strong enough AND either clearly better than the alternative
        # or itself very strong.
        accepted = (
            best_score >= min_score
            and (
                best_score - second_score
                >= config.graph_assignment_margin
                or best_score >= min_score + 0.16
            )
        )

        if accepted:
            assignments[
                best_anchor
            ].append(
                (
                    element,
                    best_score,
                    best_detail,
                )
            )
        else:
            unassigned.append(
                element
            )

        if config.graph_debug_edges:
            edge_debug.append(
                {
                    "element_index": (
                        element_index
                    ),
                    "class_name": cls,
                    "element_bbox": (
                        element.bbox
                    ),
                    "best_anchor_index": (
                        int(best_anchor)
                    ),
                    "best_anchor_title": (
                        anchors[
                            best_anchor
                        ].title
                    ),
                    "best_score": round(
                        float(
                            best_score
                        ),
                        4,
                    ),
                    "second_score": round(
                        float(
                            second_score
                        ),
                        4,
                    ),
                    "accepted": bool(
                        accepted
                    ),
                    **best_detail,
                }
            )

    return {
        "assignments": assignments,
        "unassigned_elements": unassigned,
        "edge_debug": edge_debug,
    }


# ---------------------------------------------------------------------
# 23.6 BUILD ARTICLE FROM EACH GRAPH COMPONENT
# ---------------------------------------------------------------------


def _unique_ocr_lines(
    lines: Iterable[OCRLine],
) -> list[OCRLine]:
    unique = {
        line.uid: line
        for line in lines
    }

    return sorted(
        unique.values(),
        key=lambda line: (
            line.y1,
            line.x1,
        ),
    )


def _collect_lines_from_elements(
    usable_lines: Sequence[OCRLine],
    elements: Sequence[LayoutElement],
    config: AdaptiveGraphPipelineConfig,
) -> list[OCRLine]:
    collected: list[OCRLine] = []

    text_classes = normalized_layout_class_set(
        config.doclayout_text_classes
    )

    caption_classes = normalized_layout_class_set(
        config.doclayout_caption_classes
    )

    for element in elements:
        cls = normalize_layout_class_name(
            element.class_name
        )

        if cls in text_classes:
            min_overlap = (
                config.doclayout_text_line_overlap
            )
        elif cls in caption_classes:
            min_overlap = 0.10
        else:
            continue

        collected.extend(
            lines_overlapping_element(
                usable_lines,
                element,
                min_overlap=min_overlap,
            )
        )

    return _unique_ocr_lines(
        collected
    )


def build_graph_seed_regions_v5(
    usable_lines: Sequence[OCRLine],
    anchors: Sequence[HeadlineAnchor],
    assignments: dict[
        int,
        list[
            tuple[
                LayoutElement,
                float,
                dict[str, Any],
            ]
        ],
    ],
    document_type: str,
    body_height: float,
    page_shape: tuple[int, int],
    config: AdaptiveGraphPipelineConfig,
) -> list[dict[str, Any]]:
    regions: list[
        dict[str, Any]
    ] = []

    for anchor_index, anchor in enumerate(
        anchors
    ):
        assigned = assignments.get(
            anchor_index,
            [],
        )

        elements = [
            item[0]
            for item in assigned
        ]

        edge_scores = [
            float(item[1])
            for item in assigned
        ]

        article_lines = _collect_lines_from_elements(
            usable_lines,
            elements,
            config,
        )

        # Headline OCR is always retained.
        article_lines = _unique_ocr_lines(
            list(anchor.lines)
            + article_lines
        )

        if not article_lines:
            continue

        boxes = [
            anchor.bbox,
            bbox_from_lines(
                article_lines
            ),
        ] + [
            element.bbox
            for element in elements
        ]

        region_bbox = _box_union(
            boxes
        )

        text = article_lines_to_text(
            article_lines,
            anchor,
            document_type,
            body_height,
        )

        text = remove_newspaper_artifacts(
            text
        )

        word_count = len(
            text.split()
        )

        if (
            word_count
            < config.graph_min_article_words
        ):
            continue

        quality = ocr_quality_score(
            text
        )

        if (
            quality
            < config.graph_min_article_ocr_quality
        ):
            continue

        mean_edge = (
            float(
                np.mean(
                    edge_scores
                )
            )
            if edge_scores
            else 0.32
        )

        # Anchor score is not a probability, so squash it to [0,1].
        anchor_strength = _clamp01(
            float(anchor.score)
            / 10.0
        )

        layout_confidence = _clamp01(
            0.68
            * mean_edge
            + 0.32
            * anchor_strength
        )

        edge_flags = visible_capture_edge_flags(
            region_bbox,
            page_shape,
            config,
        )

        region = {
            "anchor": anchor,
            "lines": article_lines,
            "bbox": region_bbox,
            "text": text,
            "layout_confidence": (
                layout_confidence
            ),
            "warnings": [],
            "source": (
                "local_doclayout_graph_grouping_v5"
            ),
            "headline_visible": True,
            "partial_top": (
                edge_flags[
                    "partial_top"
                ]
            ),
            "partial_bottom": (
                edge_flags[
                    "partial_bottom"
                ]
            ),
            "touches_left_edge": (
                edge_flags[
                    "touches_left_edge"
                ]
            ),
            "touches_right_edge": (
                edge_flags[
                    "touches_right_edge"
                ]
            ),
            "assigned_layout_element_count": len(
                elements
            ),
            "mean_graph_edge_score": round(
                mean_edge,
                4,
            ),
            "assigned_line_uids": [
                line.uid
                for line in article_lines
            ],
        }

        if (
            region[
                "partial_top"
            ]
            or region[
                "partial_bottom"
            ]
        ):
            region[
                "warnings"
            ].append(
                "Visible article region touches the capture edge; "
                "the physical newspaper article may continue outside this image."
            )

        regions.append(
            region
        )

    return regions


# ---------------------------------------------------------------------
# 23.7 CONTINUATION GROUPS WITHOUT A VISIBLE HEADLINE
# ---------------------------------------------------------------------


def _element_pair_connected_for_continuation(
    first: LayoutElement,
    second: LayoutElement,
    separators: Sequence[Separator],
    body_height: float,
    config: AdaptiveGraphPipelineConfig,
) -> bool:
    first_box = first.bbox
    second_box = second.bbox

    relation = _horizontal_relation_boxes(
        first_box,
        second_box,
    )

    if (
        relation
        < config.continuation_min_horizontal_relation
    ):
        return False

    gap = _vertical_gap(
        first_box,
        second_box,
    )

    if gap > (
        config.continuation_group_gap_body_heights
        * max(1.0, body_height)
    ):
        return False

    # Generic separator check between two text elements.
    first_cx, first_cy = _box_center(
        first_box
    )
    second_cx, second_cy = _box_center(
        second_box
    )

    corridor_left = min(
        first_cx,
        second_cx,
    )

    corridor_right = max(
        first_cx,
        second_cx,
    )

    corridor_top = min(
        first_cy,
        second_cy,
    )

    corridor_bottom = max(
        first_cy,
        second_cy,
    )

    for sep in separators:
        if (
            sep.orientation
            == "vertical"
            and corridor_left
            < sep.x
            < corridor_right
        ):
            overlap_top = max(
                corridor_top,
                sep.y1,
            )
            overlap_bottom = min(
                corridor_bottom,
                sep.y2,
            )

            if (
                overlap_bottom
                > overlap_top
                and sep.strength >= 0.35
            ):
                return False

    return True


def group_unassigned_text_elements_v5(
    unassigned_elements: Sequence[LayoutElement],
    usable_lines: Sequence[OCRLine],
    separators: Sequence[Separator],
    document_type: str,
    body_height: float,
    page_shape: tuple[int, int],
    config: AdaptiveGraphPipelineConfig,
) -> list[dict[str, Any]]:
    text_classes = normalized_layout_class_set(
        config.doclayout_text_classes
    )

    text_elements = [
        element
        for element in unassigned_elements
        if normalize_layout_class_name(
            element.class_name
        )
        in text_classes
    ]

    if not text_elements:
        return []

    parent = list(
        range(
            len(
                text_elements
            )
        )
    )

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[
                parent[index]
            ]
            index = parent[index]
        return index

    def union(
        first: int,
        second: int,
    ) -> None:
        root_a = find(first)
        root_b = find(second)

        if root_a != root_b:
            parent[root_b] = root_a

    ordered_indices = sorted(
        range(
            len(
                text_elements
            )
        ),
        key=lambda index: (
            text_elements[
                index
            ].y1,
            text_elements[
                index
            ].x1,
        ),
    )

    for position, first_index in enumerate(
        ordered_indices
    ):
        first = text_elements[
            first_index
        ]

        for second_index in ordered_indices[
            position + 1:
            position + 7
        ]:
            second = text_elements[
                second_index
            ]

            # Stop considering far-lower blocks.
            if (
                second.y1
                - first.y2
                > config.continuation_group_gap_body_heights
                * max(1.0, body_height)
            ):
                break

            if _element_pair_connected_for_continuation(
                first,
                second,
                separators,
                body_height,
                config,
            ):
                union(
                    first_index,
                    second_index,
                )

    groups: dict[
        int,
        list[LayoutElement],
    ] = {}

    for index, element in enumerate(
        text_elements
    ):
        groups.setdefault(
            find(index),
            [],
        ).append(
            element
        )

    results: list[
        dict[str, Any]
    ] = []

    page_h, _ = page_shape
    edge_margin = max(
        4,
        int(
            page_h
            * config.capture_edge_margin_ratio
        ),
    )

    for elements in groups.values():
        lines = _collect_lines_from_elements(
            usable_lines,
            elements,
            config,
        )

        if (
            len(lines)
            < config.continuation_min_ocr_lines
        ):
            continue

        group_bbox = _box_union(
            [
                element.bbox
                for element in elements
            ]
            + [
                bbox_from_lines(
                    lines
                )
            ]
        )

        # Main reason for a no-headline region:
        # the current capture starts/ends in the middle of an article.
        touches_top = (
            group_bbox[1]
            <= edge_margin
        )

        touches_bottom = (
            group_bbox[3]
            >= page_h - edge_margin
        )

        # Build body text without pretending a headline exists.
        synthetic_anchor = HeadlineAnchor(
            lines=[],
            title="",
            bbox=(
                group_bbox[0],
                group_bbox[1],
                group_bbox[2],
                min(
                    group_bbox[3],
                    group_bbox[1]
                    + max(
                        1,
                        int(body_height)
                    ),
                ),
            ),
            score=0.0,
        )

        text = article_lines_to_text(
            lines,
            synthetic_anchor,
            document_type,
            body_height,
        )

        text = remove_newspaper_artifacts(
            text
        )

        word_count = len(
            text.split()
        )

        if (
            word_count
            < config.continuation_min_words
        ):
            continue

        # If not touching an edge, require substantially more text.
        # This avoids turning every unassigned interior paragraph into
        # a fake article.
        if (
            not touches_top
            and not touches_bottom
            and word_count
            < 2
            * config.continuation_min_words
        ):
            continue

        ocr_q = ocr_quality_score(
            text
        )

        if (
            ocr_q
            < config.graph_min_article_ocr_quality
        ):
            continue

        mean_conf = float(
            np.mean(
                [
                    element.confidence
                    for element in elements
                ]
            )
        )

        layout_confidence = _clamp01(
            0.30
            + 0.35
            * mean_conf
            + (
                0.18
                if touches_top
                or touches_bottom
                else 0.0
            )
        )

        results.append(
            {
                "anchor": synthetic_anchor,
                "lines": lines,
                "bbox": group_bbox,
                "text": text,
                "layout_confidence": (
                    layout_confidence
                ),
                "warnings": [
                    (
                        "Headline is not visible in this capture. "
                        "This region is treated as visible article continuation text."
                    )
                ],
                "source": (
                    "local_doclayout_continuation_group_v5"
                ),
                "headline_visible": False,
                "partial_top": (
                    bool(
                        touches_top
                    )
                ),
                "partial_bottom": (
                    bool(
                        touches_bottom
                    )
                ),
                "touches_left_edge": False,
                "touches_right_edge": False,
                "assigned_layout_element_count": len(
                    elements
                ),
                "mean_graph_edge_score": 0.0,
                "assigned_line_uids": [
                    line.uid
                    for line in lines
                ],
            }
        )

    return results


# ---------------------------------------------------------------------
# 23.8 DEDUPLICATE GRAPH REGIONS
# ---------------------------------------------------------------------


def graph_region_overlap_quality(
    first: dict[str, Any],
    second: dict[str, Any],
) -> dict[str, float]:
    first_ids = set(
        first.get(
            "assigned_line_uids",
            [
                line.uid
                for line in first.get(
                    "lines",
                    [],
                )
            ],
        )
    )

    second_ids = set(
        second.get(
            "assigned_line_uids",
            [
                line.uid
                for line in second.get(
                    "lines",
                    [],
                )
            ],
        )
    )

    shared = (
        len(
            first_ids
            & second_ids
        )
        / max(
            1,
            min(
                len(first_ids),
                len(second_ids),
            ),
        )
    )

    return {
        "iou": bbox_iou(
            first["bbox"],
            second["bbox"],
        ),
        "containment": bbox_containment(
            first["bbox"],
            second["bbox"],
        ),
        "shared_lines": shared,
    }


def choose_better_graph_region_v5(
    first: dict[str, Any],
    second: dict[str, Any],
) -> dict[str, Any]:
    def quality(
        region: dict[str, Any],
    ) -> float:
        word_score = min(
            1.0,
            len(
                region.get(
                    "text",
                    "",
                ).split()
            )
            / 120.0,
        )

        headline_bonus = (
            0.12
            if region.get(
                "headline_visible",
                False,
            )
            else 0.0
        )

        return (
            0.48
            * float(
                region.get(
                    "layout_confidence",
                    0.0,
                )
            )
            + 0.40
            * word_score
            + headline_bonus
        )

    return (
        first
        if quality(first)
        >= quality(second)
        else second
    )


def deduplicate_graph_regions_v5(
    regions: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    ordered = sorted(
        regions,
        key=lambda region: (
            region["bbox"][1],
            region["bbox"][0],
        ),
    )

    kept: list[
        dict[str, Any]
    ] = []

    for region in ordered:
        duplicate_index = None

        for index, existing in enumerate(
            kept
        ):
            overlap = graph_region_overlap_quality(
                region,
                existing,
            )

            region_title = (
                region["anchor"].title
                if region.get(
                    "anchor"
                )
                is not None
                else ""
            )

            existing_title = (
                existing["anchor"].title
                if existing.get(
                    "anchor"
                )
                is not None
                else ""
            )

            title_score = (
                title_similarity(
                    region_title,
                    existing_title,
                )
                if region_title
                and existing_title
                else 0.0
            )

            if (
                overlap[
                    "shared_lines"
                ]
                >= 0.62
                or overlap[
                    "iou"
                ]
                >= 0.52
                or (
                    title_score
                    >= 0.67
                    and overlap[
                        "containment"
                    ]
                    >= 0.35
                )
            ):
                duplicate_index = index
                break

        if (
            duplicate_index
            is None
        ):
            kept.append(
                region
            )
        else:
            kept[
                duplicate_index
            ] = choose_better_graph_region_v5(
                kept[
                    duplicate_index
                ],
                region,
            )

    return sorted(
        kept,
        key=lambda region: (
            region["bbox"][1],
            region["bbox"][0],
        ),
    )


# ---------------------------------------------------------------------
# 23.9 LOCAL GROUPING QUALITY
# ---------------------------------------------------------------------


def local_article_grouping_quality_v5(
    regions: Sequence[dict[str, Any]],
    usable_lines: Sequence[OCRLine],
    anchors: Sequence[HeadlineAnchor],
) -> dict[str, Any]:
    total_word_weight = sum(
        max(
            1,
            line.word_count,
        )
        for line in usable_lines
    )

    assigned_ids: set[str] = set()

    for region in regions:
        assigned_ids.update(
            region.get(
                "assigned_line_uids",
                [
                    line.uid
                    for line in region.get(
                        "lines",
                        [],
                    )
                ],
            )
        )

    assigned_word_weight = sum(
        max(
            1,
            line.word_count,
        )
        for line in usable_lines
        if line.uid in assigned_ids
    )

    line_word_coverage = (
        assigned_word_weight
        / max(
            1,
            total_word_weight,
        )
    )

    valid_regions = [
        region
        for region in regions
        if len(
            region.get(
                "text",
                "",
            ).split()
        )
        >= 20
    ]

    mean_ocr = (
        float(
            np.mean(
                [
                    ocr_quality_score(
                        region.get(
                            "text",
                            "",
                        )
                    )
                    for region in valid_regions
                ]
            )
        )
        if valid_regions
        else 0.0
    )

    mean_layout = (
        float(
            np.mean(
                [
                    float(
                        region.get(
                            "layout_confidence",
                            0.0,
                        )
                    )
                    for region in valid_regions
                ]
            )
        )
        if valid_regions
        else 0.0
    )

    headline_regions = sum(
        1
        for region in valid_regions
        if region.get(
            "headline_visible",
            bool(
                region.get(
                    "anchor"
                )
                and region[
                    "anchor"
                ].title
            ),
        )
    )

    headline_use = (
        min(
            1.0,
            headline_regions
            / max(
                1,
                len(anchors),
            ),
        )
        if anchors
        else (
            0.65
            if valid_regions
            else 0.0
        )
    )

    # Article-count term only checks that useful regions exist.
    # It deliberately does NOT expect a fixed number of articles.
    region_presence = (
        1.0
        if len(
            valid_regions
        )
        >= 2
        else (
            0.72
            if len(
                valid_regions
            )
            == 1
            else 0.0
        )
    )

    local_quality = _clamp01(
        0.38
        * line_word_coverage
        + 0.23
        * mean_ocr
        + 0.19
        * mean_layout
        + 0.11
        * headline_use
        + 0.09
        * region_presence
    )

    return {
        "quality": round(
            local_quality,
            4,
        ),
        "line_word_coverage": round(
            line_word_coverage,
            4,
        ),
        "mean_ocr_quality": round(
            mean_ocr,
            4,
        ),
        "mean_layout_confidence": round(
            mean_layout,
            4,
        ),
        "headline_use_ratio": round(
            headline_use,
            4,
        ),
        "article_count": len(
            valid_regions
        ),
        "assigned_line_count": len(
            assigned_ids
        ),
        "usable_line_count": len(
            usable_lines
        ),
    }


# ---------------------------------------------------------------------
# 23.10 CONVERT V4 REGIONS INTO V5-COMPATIBLE REGIONS
# ---------------------------------------------------------------------


def normalize_v4_regions_for_quality_v5(
    regions: Sequence[dict[str, Any]],
    page_shape: tuple[int, int],
    config: AdaptiveGraphPipelineConfig,
) -> list[dict[str, Any]]:
    normalized = []

    for region in regions:
        item = dict(
            region
        )

        flags = visible_capture_edge_flags(
            item["bbox"],
            page_shape,
            config,
        )

        anchor = item.get(
            "anchor"
        )

        item[
            "headline_visible"
        ] = bool(
            anchor
            and anchor.title.strip()
        )

        item[
            "partial_top"
        ] = flags[
            "partial_top"
        ]

        item[
            "partial_bottom"
        ] = flags[
            "partial_bottom"
        ]

        item[
            "touches_left_edge"
        ] = flags[
            "touches_left_edge"
        ]

        item[
            "touches_right_edge"
        ] = flags[
            "touches_right_edge"
        ]

        item[
            "assigned_line_uids"
        ] = [
            line.uid
            for line in item.get(
                "lines",
                [],
            )
        ]

        item[
            "source"
        ] = (
            "local_v4_hybrid_fallback"
        )

        normalized.append(
            item
        )

    return normalized


# ---------------------------------------------------------------------
# 23.11 LEVEL-3 FULL VISIBLE CAPTURE OCR FALLBACK
# ---------------------------------------------------------------------


def build_full_visible_capture_fallback_v5(
    prepared_gray: np.ndarray,
    usable_lines: Sequence[OCRLine],
    document_type: str,
    page_shape: tuple[int, int],
) -> list[dict[str, Any]]:
    full_text = remove_newspaper_artifacts(
        best_full_page_ocr(
            prepared_gray,
            document_type,
        )
    )

    if not full_text:
        full_text = remove_newspaper_artifacts(
            " ".join(
                line.text
                for line in sorted(
                    usable_lines,
                    key=lambda line: (
                        line.y1,
                        line.x1,
                    ),
                )
            )
        )

    synthetic_anchor = HeadlineAnchor(
        lines=[],
        title="",
        bbox=(
            0,
            0,
            page_shape[1],
            max(
                1,
                int(
                    0.03
                    * page_shape[0]
                ),
            ),
        ),
        score=0.0,
    )

    return [
        {
            "anchor": synthetic_anchor,
            "lines": list(
                usable_lines
            ),
            "bbox": (
                0,
                0,
                page_shape[1],
                page_shape[0],
            ),
            "text": full_text,
            "layout_confidence": 0.22,
            "warnings": [
                (
                    "Article grouping confidence was low. "
                    "The full visible camera capture was processed locally as one OCR region."
                )
            ],
            "source": (
                "local_full_visible_capture_ocr_fallback"
            ),
            "headline_visible": False,
            "partial_top": False,
            "partial_bottom": False,
            "touches_left_edge": True,
            "touches_right_edge": True,
            "assigned_layout_element_count": 0,
            "mean_graph_edge_score": 0.0,
            "assigned_line_uids": [
                line.uid
                for line in usable_lines
            ],
        }
    ]


# ---------------------------------------------------------------------
# 23.12 MAIN V5 LOCAL ARTICLE DETECTOR
# ---------------------------------------------------------------------


def detect_adaptive_graph_articles_v5(
    lines: Sequence[OCRLine],
    prepared_gray: np.ndarray,
    prepared_color: np.ndarray,
    document_type: str,
    config: AdaptiveGraphPipelineConfig,
    doclayout_model=None,
) -> dict[str, Any]:
    page_shape = prepared_gray.shape[:2]

    separators = detect_page_separators(
        prepared_gray
    )

    body_height = estimate_body_height(
        lines,
        page_shape,
    )

    content_area = detect_adaptive_content_area(
        lines,
        separators,
        page_shape,
        body_height,
        config,
    )

    usable_lines = [
        line
        for line in lines
        if (
            content_area[0]
            <= line.cy
            <= content_area[1]
        )
        and not is_noise_line(
            line.text,
            document_type,
        )
    ]

    warnings: list[str] = []

    # ----------------------------------------------------------
    # A. Local DocLayout
    # ----------------------------------------------------------

    layout_elements: list[
        LayoutElement
    ] = []

    doclayout_error = None

    if (
        config.use_doclayout_yolo
        and doclayout_model is not None
    ):
        try:
            layout_elements = detect_doclayout_elements(
                doclayout_model,
                prepared_color,
                config,
            )
        except Exception as error:
            doclayout_error = str(
                error
            )

            warnings.append(
                "DocLayout-YOLO inference failed; "
                "the fully local V4/OCR fallback will be used."
            )

    # ----------------------------------------------------------
    # B. Headline seeds
    # ----------------------------------------------------------

    seed_info = build_adaptive_headline_seeds_v5(
        usable_lines=usable_lines,
        layout_elements=layout_elements,
        separators=separators,
        body_height=body_height,
        content_area=content_area,
        page_shape=page_shape,
        document_type=document_type,
        config=config,
    )

    anchors = seed_info[
        "anchors"
    ]

    # ----------------------------------------------------------
    # C. Graph association
    # ----------------------------------------------------------

    graph_assignment = assign_layout_elements_to_articles_v5(
        anchors=anchors,
        layout_elements=layout_elements,
        separators=separators,
        body_height=body_height,
        page_shape=page_shape,
        config=config,
    )

    graph_regions = build_graph_seed_regions_v5(
        usable_lines=usable_lines,
        anchors=anchors,
        assignments=graph_assignment[
            "assignments"
        ],
        document_type=document_type,
        body_height=body_height,
        page_shape=page_shape,
        config=config,
    )

    continuation_regions = group_unassigned_text_elements_v5(
        unassigned_elements=graph_assignment[
            "unassigned_elements"
        ],
        usable_lines=usable_lines,
        separators=separators,
        document_type=document_type,
        body_height=body_height,
        page_shape=page_shape,
        config=config,
    )

    graph_regions = deduplicate_graph_regions_v5(
        list(
            graph_regions
        )
        + list(
            continuation_regions
        )
    )

    graph_quality = local_article_grouping_quality_v5(
        graph_regions,
        usable_lines,
        anchors,
    )

    # ----------------------------------------------------------
    # D. Fully local V4 fallback
    # ----------------------------------------------------------

    v4_regions: list[
        dict[str, Any]
    ] = []

    v4_quality = {
        "quality": 0.0,
        "article_count": 0,
    }

    v4_layout = None

    if config.use_v4_local_fallback:
        try:
            v4_layout = _V4_LOCAL_DETECTOR(
                usable_lines,
                prepared_gray,
                prepared_color,
                document_type,
                config,
                doclayout_model=doclayout_model,
            )

            v4_regions = normalize_v4_regions_for_quality_v5(
                v4_layout.get(
                    "regions",
                    [],
                ),
                page_shape,
                config,
            )

            v4_quality = local_article_grouping_quality_v5(
                v4_regions,
                usable_lines,
                v4_layout.get(
                    "anchors",
                    anchors,
                ),
            )

        except Exception as error:
            warnings.append(
                "V4 local fallback failed: "
                f"{error}"
            )

    # ----------------------------------------------------------
    # E. Select LOCAL method by measured quality
    # ----------------------------------------------------------

    selected_method = (
        "local_doclayout_graph_v5"
    )

    selected_regions = graph_regions
    selected_quality = graph_quality

    graph_score = float(
        graph_quality.get(
            "quality",
            0.0,
        )
    )

    v4_score = float(
        v4_quality.get(
            "quality",
            0.0,
        )
    )

    graph_good = (
        graph_score
        >= config.graph_quality_threshold
    )

    v4_good = (
        v4_score
        >= config.v4_fallback_quality_threshold
    )

    if graph_good:
        # Prefer graph unless V4 is clearly better.
        if (
            v4_good
            and v4_score
            > graph_score
            + config.prefer_graph_if_within
        ):
            selected_method = (
                "local_v4_hybrid_fallback"
            )
            selected_regions = (
                v4_regions
            )
            selected_quality = (
                v4_quality
            )

    elif v4_good:
        selected_method = (
            "local_v4_hybrid_fallback"
        )
        selected_regions = (
            v4_regions
        )
        selected_quality = (
            v4_quality
        )

    elif (
        config.allow_full_visible_ocr_fallback
    ):
        selected_method = (
            "local_full_visible_capture_ocr_fallback"
        )

        selected_regions = build_full_visible_capture_fallback_v5(
            prepared_gray=prepared_gray,
            usable_lines=usable_lines,
            document_type=document_type,
            page_shape=page_shape,
        )

        selected_quality = local_article_grouping_quality_v5(
            selected_regions,
            usable_lines,
            [],
        )

        warnings.append(
            "Graph grouping and V4 local grouping were both below "
            "the configured confidence threshold; full visible OCR fallback was used."
        )

    # ----------------------------------------------------------
    # F. Diagnostics
    # ----------------------------------------------------------

    return {
        "regions": selected_regions,
        "selected_method": selected_method,
        "selected_quality": selected_quality,
        "graph_regions": graph_regions,
        "graph_quality": graph_quality,
        "v4_regions": v4_regions,
        "v4_quality": v4_quality,
        "anchors": anchors,
        "layout_elements": layout_elements,
        "separators": separators,
        "usable_lines": usable_lines,
        "body_height": body_height,
        "content_area": content_area,
        "warnings": warnings,
        "doclayout_error": (
            doclayout_error
        ),
        "edge_debug": (
            graph_assignment[
                "edge_debug"
            ]
        ),
        "unassigned_layout_elements": (
            graph_assignment[
                "unassigned_elements"
            ]
        ),
        "doclayout_class_counts": (
            doclayout_class_counts(
                layout_elements
            )
        ),
        "debug_counts": {
            "ocr_line_count": len(
                lines
            ),
            "usable_ocr_line_count": len(
                usable_lines
            ),
            "doclayout_element_count": len(
                layout_elements
            ),
            "headline_seed_count": len(
                anchors
            ),
            "dominant_headline_count": len(
                seed_info[
                    "dominant_anchors"
                ]
            ),
            "graph_region_count": len(
                graph_regions
            ),
            "continuation_region_count": len(
                continuation_regions
            ),
            "v4_region_count": len(
                v4_regions
            ),
            "selected_region_count": len(
                selected_regions
            ),
        },
    }


# ---------------------------------------------------------------------
# 23.13 MAIN DOCUMENT ANALYZER
# ---------------------------------------------------------------------


def analyze_content_graph_local(
    image_path: str,
    document_type: str,
    model,
    config: AdaptiveGraphPipelineConfig,
    doclayout_model=None,
) -> dict[str, Any]:
    normalized_type = normalize_document_type(
        document_type
    )

    # Novel and other single-content documents keep the existing
    # local full-page branch.
    if (
        normalized_type
        not in MULTI_CATEGORY_TYPES
    ):
        analysis = _BASE_LOCAL_ANALYZER(
            image_path=image_path,
            document_type=normalized_type,
            model=model,
            config=config,
        )

        analysis[
            "pipeline_version"
        ] = (
            "V5_ADAPTIVE_LOCAL_GRAPH_SINGLE_DOCUMENT"
        )

        analysis[
            "processing"
        ] = {
            "cloud_api_used": False,
            "ocr": "local_tesseract",
            "layout": (
                "full_visible_capture_reading_order"
            ),
            "category_model": (
                "local_tfidf_logistic_regression"
            ),
            "speech_input": (
                "local_vosk"
            ),
            "summary_model": (
                f"local_ollama_{config.ollama_model}"
            ),
        }

        return analysis

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        raise FileNotFoundError(
            f"Image not found or unreadable: {image_path}"
        )

    prepared_color, prepared_gray, scale = prepare_page_image(
        image,
        config,
    )

    page_shape = prepared_gray.shape[:2]

    ocr_lines = extract_ocr_lines(
        prepared_gray,
        normalized_type,
        config,
    )

    local_detection = detect_adaptive_graph_articles_v5(
        lines=ocr_lines,
        prepared_gray=prepared_gray,
        prepared_color=prepared_color,
        document_type=normalized_type,
        config=config,
        doclayout_model=doclayout_model,
    )

    results = []

    for number, region in enumerate(
        local_detection[
            "regions"
        ],
        start=1,
    ):
        anchor = region.get(
            "anchor"
        )

        raw_title = (
            anchor.title.strip()
            if anchor is not None
            and anchor.title
            else ""
        )

        title_for_category = (
            raw_title
        )

        display_title = (
            raw_title
            if raw_title
            else "Visible article continuation"
        )

        text = remove_newspaper_artifacts(
            region.get(
                "text",
                "",
            )
        )

        if not text:
            continue

        prediction = predict_category(
            model=model,
            text=text,
            document_type=normalized_type,
            title=title_for_category,
            config=config,
        )

        warnings = list(
            region.get(
                "warnings",
                [],
            )
        )

        partial_top = bool(
            region.get(
                "partial_top",
                False,
            )
        )

        partial_bottom = bool(
            region.get(
                "partial_bottom",
                False,
            )
        )

        if (
            partial_top
            or partial_bottom
        ):
            warnings.append(
                "Summary/full-text output must use only visible extracted text "
                "because the article may continue outside the current camera capture."
            )

        results.append(
            {
                "article_number": (
                    number
                ),
                "title": (
                    display_title
                ),
                "raw_visible_title": (
                    raw_title
                ),
                "headline_visible": bool(
                    raw_title
                ),
                "bbox": region[
                    "bbox"
                ],
                "normalized_bbox": normalize_bbox(
                    region[
                        "bbox"
                    ],
                    page_shape,
                ),
                "full_text": text,
                "category": prediction[
                    "category"
                ],
                "confidence": prediction[
                    "confidence"
                ],
                "probability_margin": prediction[
                    "margin"
                ],
                "top_predictions": prediction[
                    "top_predictions"
                ],
                "category_decision": prediction.get(
                    "decision",
                    "",
                ),
                "ocr_quality": ocr_quality_score(
                    text
                ),
                "layout_confidence": round(
                    float(
                        region.get(
                            "layout_confidence",
                            0.0,
                        )
                    ),
                    4,
                ),
                "partial_top": (
                    partial_top
                ),
                "partial_bottom": (
                    partial_bottom
                ),
                "source": region.get(
                    "source",
                    local_detection[
                        "selected_method"
                    ],
                ),
                "warnings": warnings,
            }
        )

    if not results:
        # Extremely defensive local fallback.
        fallback = build_full_visible_capture_fallback_v5(
            prepared_gray=prepared_gray,
            usable_lines=local_detection[
                "usable_lines"
            ],
            document_type=normalized_type,
            page_shape=page_shape,
        )[0]

        text = fallback[
            "text"
        ]

        prediction = predict_category(
            model=model,
            text=text,
            document_type=normalized_type,
            title="",
            config=config,
        )

        results = [
            {
                "article_number": 1,
                "title": (
                    "Visible newspaper content"
                ),
                "raw_visible_title": "",
                "headline_visible": False,
                "bbox": fallback[
                    "bbox"
                ],
                "normalized_bbox": (
                    0,
                    0,
                    1000,
                    1000,
                ),
                "full_text": text,
                "category": prediction[
                    "category"
                ],
                "confidence": prediction[
                    "confidence"
                ],
                "probability_margin": prediction[
                    "margin"
                ],
                "top_predictions": prediction[
                    "top_predictions"
                ],
                "category_decision": prediction.get(
                    "decision",
                    "",
                ),
                "ocr_quality": ocr_quality_score(
                    text
                ),
                "layout_confidence": 0.20,
                "partial_top": False,
                "partial_bottom": False,
                "source": (
                    "local_full_visible_capture_ocr_fallback"
                ),
                "warnings": fallback[
                    "warnings"
                ],
            }
        ]

    category_counts = Counter(
        item[
            "category"
        ]
        for item in results
    )

    categories = sorted(
        category_counts
    )

    return {
        "status": "ANALYZED",
        "pipeline_version": (
            "V5_ADAPTIVE_LOCAL_GRAPH"
        ),
        "document_type_received": (
            document_type
        ),
        "document_type": (
            normalized_type
        ),
        "mode": (
            "Multi Category"
        ),
        "detected_categories": (
            categories
        ),
        "category_counts": dict(
            category_counts
        ),
        "needs_category_selection": (
            len(categories)
            > 1
        ),
        "image_shape": {
            "height": (
                page_shape[0]
            ),
            "width": (
                page_shape[1]
            ),
        },
        "resize_scale": scale,
        "warnings": local_detection[
            "warnings"
        ],
        "local_grouping": {
            "selected_method": (
                local_detection[
                    "selected_method"
                ]
            ),
            "selected_quality": (
                local_detection[
                    "selected_quality"
                ]
            ),
            "graph_quality": (
                local_detection[
                    "graph_quality"
                ]
            ),
            "v4_quality": (
                local_detection[
                    "v4_quality"
                ]
            ),
        },
        "layout_debug": {
            "pipeline_version": (
                "V5_ADAPTIVE_LOCAL_GRAPH"
            ),
            **local_detection[
                "debug_counts"
            ],
            "body_height": round(
                float(
                    local_detection[
                        "body_height"
                    ]
                ),
                2,
            ),
            "content_area": (
                local_detection[
                    "content_area"
                ]
            ),
            "doclayout_class_counts": (
                local_detection[
                    "doclayout_class_counts"
                ]
            ),
            "headline_anchors": [
                {
                    "title": (
                        anchor.title
                    ),
                    "bbox": (
                        anchor.bbox
                    ),
                    "score": round(
                        float(
                            anchor.score
                        ),
                        3,
                    ),
                    "line_count": len(
                        anchor.lines
                    ),
                }
                for anchor in local_detection[
                    "anchors"
                ]
            ],
            "edge_debug": (
                local_detection[
                    "edge_debug"
                ]
            ),
            "doclayout_elements": [
                {
                    "class_name": normalize_layout_class_name(
                        element.class_name
                    ),
                    "raw_class_name": (
                        element.class_name
                    ),
                    "confidence": round(
                        float(
                            element.confidence
                        ),
                        4,
                    ),
                    "bbox": (
                        element.bbox
                    ),
                }
                for element in local_detection[
                    "layout_elements"
                ]
            ],
        },
        "results": results,
        "processing": {
            "cloud_api_used": False,
            "openai_used": False,
            "ocr": (
                "local_tesseract"
            ),
            "layout_model": (
                "local_DocLayout_YOLO_DocStructBench"
            ),
            "article_grouping": (
                local_detection[
                    "selected_method"
                ]
            ),
            "local_fallback_chain": [
                "local_doclayout_graph_v5",
                "local_v4_hybrid_fallback",
                "local_full_visible_capture_ocr_fallback",
            ],
            "category_model": (
                "local_tfidf_logistic_regression"
            ),
            "speech_input": (
                "local_vosk"
            ),
            "summary_model": (
                f"local_ollama_{config.ollama_model}"
            ),
        },
    }


# ---------------------------------------------------------------------
# 23.14 OUTPUT GENERATION WITH PARTIAL-CAPTURE METADATA
# ---------------------------------------------------------------------


def generate_selected_output_graph_local(
    analysis: dict[str, Any],
    selected_category: Optional[str],
    depth: str,
    config: AdaptiveGraphPipelineConfig,
    clean_full_text_with_llama: bool = False,
) -> dict[str, Any]:
    # Reuse the existing V3/V4 grounded local summarization logic.
    output = _V4_LOCAL_OUTPUT(
        analysis=analysis,
        selected_category=selected_category,
        depth=depth,
        config=config,
        clean_full_text_with_llama=clean_full_text_with_llama,
    )

    if output.get(
        "status"
    ) != "READY":
        return output

    source_lookup = {
        item[
            "article_number"
        ]: item
        for item in analysis.get(
            "results",
            []
        )
    }

    for output_item in output.get(
        "items",
        []
    ):
        original_item = source_lookup.get(
            output_item[
                "article_number"
            ],
            {},
        )

        output_item[
            "headline_visible"
        ] = original_item.get(
            "headline_visible",
            True,
        )

        output_item[
            "partial_top"
        ] = original_item.get(
            "partial_top",
            False,
        )

        output_item[
            "partial_bottom"
        ] = original_item.get(
            "partial_bottom",
            False,
        )

        output_item[
            "source"
        ] = original_item.get(
            "source",
            output_item.get(
                "source",
                "",
            ),
        )

    # Replace next-module article payload with the enriched items.
    if (
        "next_module_payload"
        in output
    ):
        output[
            "next_module_payload"
        ][
            "articles"
        ] = output[
            "items"
        ]

        output[
            "next_module_payload"
        ][
            "local_grouping"
        ] = analysis.get(
            "local_grouping",
            {},
        )

        output[
            "next_module_payload"
        ][
            "cloud_api_used"
        ] = False

    return output


# ---------------------------------------------------------------------
# 23.15 VOICE END-TO-END WRAPPER
# ---------------------------------------------------------------------


def run_adaptive_local_voice_pipeline_v5(
    image_path: str,
    document_type: str,
    category_model,
    config: AdaptiveGraphPipelineConfig,
    doclayout_model,
    voice_selector: OfflineVoiceSelector,
    tts_enabled: bool = False,
    clean_full_text_with_llama: bool = False,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
]:
    analysis = analyze_content_graph_local(
        image_path=image_path,
        document_type=document_type,
        model=category_model,
        config=config,
        doclayout_model=doclayout_model,
    )

    selected_category, depth = collect_user_preferences_voice(
        analysis=analysis,
        voice_selector=voice_selector,
        tts_enabled=tts_enabled,
    )

    output = generate_selected_output_graph_local(
        analysis=analysis,
        selected_category=selected_category,
        depth=depth,
        config=config,
        clean_full_text_with_llama=clean_full_text_with_llama,
    )

    return analysis, output


# ---------------------------------------------------------------------
# 23.16 DEBUG TABLES
# ---------------------------------------------------------------------


def adaptive_local_debug_summary_v5(
    analysis: dict[str, Any],
) -> pd.DataFrame:
    local = analysis.get(
        "local_grouping",
        {},
    )

    debug = analysis.get(
        "layout_debug",
        {},
    )

    selected = local.get(
        "selected_quality",
        {},
    )

    graph = local.get(
        "graph_quality",
        {},
    )

    v4 = local.get(
        "v4_quality",
        {},
    )

    rows = [
        (
            "Pipeline",
            analysis.get(
                "pipeline_version",
                "",
            ),
        ),
        (
            "Cloud API used",
            analysis.get(
                "processing",
                {},
            ).get(
                "cloud_api_used",
                False,
            ),
        ),
        (
            "Selected local method",
            local.get(
                "selected_method",
                "",
            ),
        ),
        (
            "Selected quality",
            selected.get(
                "quality",
                0.0,
            ),
        ),
        (
            "Graph quality",
            graph.get(
                "quality",
                0.0,
            ),
        ),
        (
            "Graph OCR-line coverage",
            graph.get(
                "line_word_coverage",
                0.0,
            ),
        ),
        (
            "V4 fallback quality",
            v4.get(
                "quality",
                0.0,
            ),
        ),
        (
            "OCR lines",
            debug.get(
                "ocr_line_count",
                0,
            ),
        ),
        (
            "Usable OCR lines",
            debug.get(
                "usable_ocr_line_count",
                0,
            ),
        ),
        (
            "DocLayout elements",
            debug.get(
                "doclayout_element_count",
                0,
            ),
        ),
        (
            "Headline seeds",
            debug.get(
                "headline_seed_count",
                0,
            ),
        ),
        (
            "Graph regions",
            debug.get(
                "graph_region_count",
                0,
            ),
        ),
        (
            "Continuation regions",
            debug.get(
                "continuation_region_count",
                0,
            ),
        ),
        (
            "V4 regions",
            debug.get(
                "v4_region_count",
                0,
            ),
        ),
        (
            "Final regions",
            debug.get(
                "selected_region_count",
                len(
                    analysis.get(
                        "results",
                        [],
                    )
                ),
            ),
        ),
    ]

    return pd.DataFrame(
        rows,
        columns=[
            "Stage",
            "Value",
        ],
    )


def graph_edge_table_v5(
    analysis: dict[str, Any],
    accepted_only: bool = False,
) -> pd.DataFrame:
    rows = analysis.get(
        "layout_debug",
        {},
    ).get(
        "edge_debug",
        [],
    )

    table = pd.DataFrame(
        rows
    )

    if (
        accepted_only
        and not table.empty
        and "accepted"
        in table.columns
    ):
        table = table[
            table[
                "accepted"
            ]
            == True
        ].copy()

    return table


def adaptive_article_table_v5(
    analysis: dict[str, Any],
) -> pd.DataFrame:
    rows = []

    for item in analysis.get(
        "results",
        []
    ):
        rows.append(
            {
                "article": (
                    item[
                        "article_number"
                    ]
                ),
                "title": (
                    item[
                        "title"
                    ]
                ),
                "category": (
                    item[
                        "category"
                    ]
                ),
                "category_confidence": (
                    item[
                        "confidence"
                    ]
                ),
                "ocr_quality": (
                    item[
                        "ocr_quality"
                    ]
                ),
                "layout_confidence": (
                    item[
                        "layout_confidence"
                    ]
                ),
                "headline_visible": (
                    item.get(
                        "headline_visible",
                        True,
                    )
                ),
                "partial_top": (
                    item.get(
                        "partial_top",
                        False,
                    )
                ),
                "partial_bottom": (
                    item.get(
                        "partial_bottom",
                        False,
                    )
                ),
                "words": len(
                    item.get(
                        "full_text",
                        "",
                    ).split()
                ),
                "source": (
                    item.get(
                        "source",
                        "",
                    )
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ---------------------------------------------------------------------
# 23.17 OPTIONAL VISUALIZATION OF FINAL V5 REGIONS
# ---------------------------------------------------------------------


def show_adaptive_regions_v5(
    image_path: str,
    analysis: dict[str, Any],
) -> None:
    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        raise FileNotFoundError(
            image_path
        )

    # The analysis boxes are based on the prepared target-width image.
    prepared_color, _, _ = prepare_page_image(
        image,
        AdaptiveGraphPipelineConfig(
            target_width=analysis.get(
                "image_shape",
                {},
            ).get(
                "width",
                1800,
            )
        ),
    )

    canvas = prepared_color.copy()

    for item in analysis.get(
        "results",
        []
    ):
        x1, y1, x2, y2 = item[
            "bbox"
        ]

        cv2.rectangle(
            canvas,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            3,
        )

        label = (
            f"{item['article_number']}: "
            f"{item['title'][:45]}"
        )

        cv2.putText(
            canvas,
            label,
            (
                x1,
                max(
                    20,
                    y1 - 8,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    plt.figure(
        figsize=(14, 18)
    )

    plt.imshow(
        cv2.cvtColor(
            canvas,
            cv2.COLOR_BGR2RGB,
        )
    )

    plt.axis(
        "off"
    )

    plt.title(
        "V5 Adaptive Local Article Regions"
    )

    plt.show()


# =====================================================================
# 24. V6 — LOCAL READING-ORDER + GRAPH ARTICLE RECONSTRUCTION
# =====================================================================
#
# V6 is a fully-local improvement over V5.
#
# NO OpenAI / cloud vision / cloud OCR API is used.
#
# The central improvement is an explicit READING-ORDER layer:
#
#   DocLayout-YOLO structural blocks
#               +
#        Tesseract OCR geometry
#               ↓
#      adaptive column/lane inference
#               ↓
#   local reading-order relationships
#               ↓
#   headline -> body direct edges
#               +
#   body -> body continuation-chain edges
#               ↓
#      article graph components
#
# This is designed for:
#   - full newspaper pages
#   - top-half captures
#   - bottom-half captures
#   - shifted partial captures
#   - pages whose first visible body text has no visible headline
#
# There is NO publication-specific code.
#
# Fallback chain:
#   1. V6 reading-order graph
#   2. V5 adaptive graph (fully local)
#   3. V4 hybrid (inside V5 fallback)
#   4. full visible OCR (inside V5 fallback)
# =====================================================================


@dataclass
class AdaptiveReadingGraphConfig(AdaptiveGraphPipelineConfig):
    # ---------------------------------------------------------
    # Reading-lane inference
    # ---------------------------------------------------------
    reading_lane_width_ratio_max: float = 0.58
    reading_lane_center_distance_ratio: float = 0.48
    reading_lane_min_overlap: float = 0.16

    # ---------------------------------------------------------
    # Reading-order contribution to title -> block association
    # ---------------------------------------------------------
    reading_order_weight: float = 0.22
    reading_same_lane_bonus: float = 0.13
    reading_before_next_headline_bonus: float = 0.10
    reading_after_next_headline_penalty: float = 0.35
    reading_immediate_successor_bonus: float = 0.08

    # ---------------------------------------------------------
    # Body -> body continuation graph
    # ---------------------------------------------------------
    body_chain_max_gap_body_heights: float = 8.5
    body_chain_min_horizontal_relation: float = 0.24
    body_chain_max_center_shift_ratio: float = 0.72

    # A component can inherit an article label when enough evidence exists.
    body_chain_min_seed_score: float = 0.43
    body_chain_min_label_margin: float = 0.06

    # ---------------------------------------------------------
    # Quality controller
    # ---------------------------------------------------------
    reading_graph_quality_threshold: float = 0.55
    fallback_to_v5_when_below: bool = True

    # Prefer V6 when the two methods are very close.
    prefer_v6_within: float = 0.04


@dataclass
class ReadingOrderNode:
    node_id: int
    class_name: str
    bbox: tuple[int, int, int, int]
    confidence: float
    element_index: int

    primary_lane: int = -1
    lane_memberships: tuple[int, ...] = field(default_factory=tuple)

    lane_rank: int = -1
    reading_rank: int = -1

    text: str = ""
    line_uids: tuple[str, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------
# 24.1 GENERIC HORIZONTAL / SEPARATOR HELPERS
# ---------------------------------------------------------------------


def _interval_overlap_ratio_v6(
    a1: float,
    a2: float,
    b1: float,
    b2: float,
) -> float:
    overlap = max(
        0.0,
        min(a2, b2) - max(a1, b1),
    )

    denominator = max(
        1.0,
        min(a2 - a1, b2 - b1),
    )

    return float(
        overlap / denominator
    )


def _strong_separator_between_boxes_v6(
    first_box: tuple[int, int, int, int],
    second_box: tuple[int, int, int, int],
    separators: Sequence[Separator],
) -> bool:
    first_cx, first_cy = _box_center(
        first_box
    )

    second_cx, second_cy = _box_center(
        second_box
    )

    min_x = min(
        first_cx,
        second_cx,
    )
    max_x = max(
        first_cx,
        second_cx,
    )

    min_y = min(
        first_cy,
        second_cy,
    )
    max_y = max(
        first_cy,
        second_cy,
    )

    for separator in separators:
        if separator.strength < 0.34:
            continue

        if separator.orientation == "vertical":
            if not (
                min_x
                < separator.x
                < max_x
            ):
                continue

            vertical_overlap = max(
                0.0,
                min(
                    max_y,
                    separator.y2,
                )
                - max(
                    min_y,
                    separator.y1,
                ),
            )

            required = max(
                1.0,
                0.25
                * max(
                    1.0,
                    max_y - min_y,
                ),
            )

            if vertical_overlap >= required:
                return True

        else:
            sep_y = separator.y

            if not (
                min_y
                < sep_y
                < max_y
            ):
                continue

            min_left = min(
                first_box[0],
                second_box[0],
            )

            max_right = max(
                first_box[2],
                second_box[2],
            )

            overlap = max(
                0.0,
                min(
                    max_right,
                    separator.x2,
                )
                - max(
                    min_left,
                    separator.x1,
                ),
            )

            width = max(
                1.0,
                max_right - min_left,
            )

            if (
                overlap / width
                >= 0.42
            ):
                return True

    return False


# ---------------------------------------------------------------------
# 24.2 BUILD READING-ORDER NODES
# ---------------------------------------------------------------------


def build_reading_order_nodes_v6(
    layout_elements: Sequence[LayoutElement],
    usable_lines: Sequence[OCRLine],
    config: AdaptiveReadingGraphConfig,
) -> list[ReadingOrderNode]:
    text_classes = normalized_layout_class_set(
        config.doclayout_text_classes
    )

    caption_classes = normalized_layout_class_set(
        config.doclayout_caption_classes
    )

    title_classes = {
        "title",
    }

    useful_classes = (
        text_classes
        | caption_classes
        | title_classes
    )

    nodes: list[
        ReadingOrderNode
    ] = []

    for element_index, element in enumerate(
        layout_elements
    ):
        cls = normalize_layout_class_name(
            element.class_name
        )

        if cls not in useful_classes:
            continue

        if cls in text_classes:
            min_overlap = (
                config.doclayout_text_line_overlap
            )
        else:
            min_overlap = 0.08

        overlapping_lines = lines_overlapping_element(
            usable_lines,
            element,
            min_overlap=min_overlap,
        )

        text = remove_newspaper_artifacts(
            " ".join(
                line.text
                for line in sorted(
                    overlapping_lines,
                    key=lambda line: (
                        line.y1,
                        line.x1,
                    ),
                )
            )
        )

        nodes.append(
            ReadingOrderNode(
                node_id=len(nodes),
                class_name=cls,
                bbox=element.bbox,
                confidence=float(
                    element.confidence
                ),
                element_index=element_index,
                text=text,
                line_uids=tuple(
                    line.uid
                    for line in overlapping_lines
                ),
            )
        )

    return nodes


# ---------------------------------------------------------------------
# 24.3 INFER GENERIC READING LANES / COLUMNS
# ---------------------------------------------------------------------


def infer_reading_lanes_v6(
    nodes: Sequence[ReadingOrderNode],
    page_shape: tuple[int, int],
    config: AdaptiveReadingGraphConfig,
) -> dict[str, Any]:
    page_h, page_w = page_shape

    body_nodes = [
        node
        for node in nodes
        if (
            node.class_name
            == "plain_text"
            and (
                node.bbox[2]
                - node.bbox[0]
            )
            <= config.reading_lane_width_ratio_max
            * page_w
        )
    ]

    candidates = (
        body_nodes
        if body_nodes
        else [
            node
            for node in nodes
            if node.class_name
            != "title"
        ]
    )

    widths = [
        max(
            1,
            node.bbox[2]
            - node.bbox[0],
        )
        for node in candidates
    ]

    median_width = (
        float(
            np.median(
                widths
            )
        )
        if widths
        else max(
            1.0,
            0.22 * page_w,
        )
    )

    center_threshold = (
        config.reading_lane_center_distance_ratio
        * median_width
    )

    # Each lane stores horizontal evidence from narrow text blocks.
    lanes: list[
        dict[str, Any]
    ] = []

    for node in sorted(
        candidates,
        key=lambda item: (
            (
                item.bbox[0]
                + item.bbox[2]
            )
            / 2.0,
            item.bbox[1],
        ),
    ):
        x1, _, x2, _ = node.bbox
        cx = (
            x1 + x2
        ) / 2.0

        best_index = None
        best_score = -1.0

        for lane_index, lane in enumerate(
            lanes
        ):
            lane_x1 = lane[
                "x1"
            ]
            lane_x2 = lane[
                "x2"
            ]
            lane_cx = lane[
                "cx"
            ]

            overlap = _interval_overlap_ratio_v6(
                x1,
                x2,
                lane_x1,
                lane_x2,
            )

            center_score = max(
                0.0,
                1.0
                - abs(
                    cx - lane_cx
                )
                / max(
                    1.0,
                    center_threshold,
                ),
            )

            score = max(
                overlap,
                0.72
                * center_score,
            )

            if score > best_score:
                best_score = score
                best_index = lane_index

        should_join = (
            best_index
            is not None
            and (
                best_score
                >= config.reading_lane_min_overlap
                or abs(
                    cx
                    - lanes[
                        best_index
                    ][
                        "cx"
                    ]
                )
                <= center_threshold
            )
        )

        if should_join:
            lane = lanes[
                best_index
            ]

            lane[
                "members"
            ].append(
                node.node_id
            )

            member_boxes = [
                nodes[
                    member_id
                ].bbox
                for member_id in lane[
                    "members"
                ]
            ]

            # Robust x span: median of left/right edges keeps a wide
            # outlier from merging neighboring newspaper columns.
            lefts = [
                box[0]
                for box in member_boxes
            ]

            rights = [
                box[2]
                for box in member_boxes
            ]

            centers = [
                (
                    box[0]
                    + box[2]
                )
                / 2.0
                for box in member_boxes
            ]

            lane[
                "x1"
            ] = int(
                np.median(
                    lefts
                )
            )

            lane[
                "x2"
            ] = int(
                np.median(
                    rights
                )
            )

            lane[
                "cx"
            ] = float(
                np.median(
                    centers
                )
            )

        else:
            lanes.append(
                {
                    "x1": int(x1),
                    "x2": int(x2),
                    "cx": float(cx),
                    "members": [
                        node.node_id
                    ],
                }
            )

    # Sort from left to right.
    lanes = sorted(
        lanes,
        key=lambda lane: lane[
            "cx"
        ],
    )

    for lane_index, lane in enumerate(
        lanes
    ):
        lane[
            "lane_id"
        ] = lane_index

    # Assign every node to one or more inferred lanes.
    node_lane_map: dict[
        int,
        dict[str, Any]
    ] = {}

    for node in nodes:
        x1, _, x2, _ = node.bbox
        cx = (
            x1 + x2
        ) / 2.0

        memberships = []

        for lane in lanes:
            overlap = _interval_overlap_ratio_v6(
                x1,
                x2,
                lane[
                    "x1"
                ],
                lane[
                    "x2"
                ],
            )

            if (
                overlap
                >= config.reading_lane_min_overlap
            ):
                memberships.append(
                    lane[
                        "lane_id"
                    ]
                )

        if not memberships and lanes:
            primary_lane = min(
                lanes,
                key=lambda lane: abs(
                    cx
                    - lane[
                        "cx"
                    ]
                ),
            )[
                "lane_id"
            ]

            memberships = [
                primary_lane
            ]

        if memberships:
            primary_lane = min(
                memberships,
                key=lambda lane_id: abs(
                    cx
                    - lanes[
                        lane_id
                    ][
                        "cx"
                    ]
                ),
            )
        else:
            primary_lane = -1

        node_lane_map[
            node.node_id
        ] = {
            "primary_lane": (
                primary_lane
            ),
            "memberships": tuple(
                memberships
            ),
        }

        node.primary_lane = (
            primary_lane
        )

        node.lane_memberships = tuple(
            memberships
        )

    # Within each lane: top -> bottom.
    lane_orders: dict[
        int,
        list[int],
    ] = {}

    for lane in lanes:
        lane_id = lane[
            "lane_id"
        ]

        members = [
            node.node_id
            for node in nodes
            if lane_id
            in node.lane_memberships
        ]

        ordered = sorted(
            members,
            key=lambda node_id: (
                nodes[
                    node_id
                ].bbox[1],
                nodes[
                    node_id
                ].bbox[0],
            ),
        )

        lane_orders[
            lane_id
        ] = ordered

        for rank, node_id in enumerate(
            ordered
        ):
            # For spanning nodes, keep the smallest rank obtained.
            previous = nodes[
                node_id
            ].lane_rank

            if (
                previous < 0
                or rank < previous
            ):
                nodes[
                    node_id
                ].lane_rank = rank

    # Stable global diagnostic order:
    # primarily page Y, then inferred lane.
    global_order = sorted(
        [
            node.node_id
            for node in nodes
        ],
        key=lambda node_id: (
            nodes[
                node_id
            ].bbox[1],
            nodes[
                node_id
            ].primary_lane,
            nodes[
                node_id
            ].bbox[0],
        ),
    )

    for reading_rank, node_id in enumerate(
        global_order
    ):
        nodes[
            node_id
        ].reading_rank = (
            reading_rank
        )

    return {
        "lanes": lanes,
        "lane_orders": lane_orders,
        "node_lane_map": node_lane_map,
        "global_order": global_order,
        "median_narrow_text_width": (
            median_width
        ),
    }


# ---------------------------------------------------------------------
# 24.4 HEADLINE -> LANE RELATIONSHIP
# ---------------------------------------------------------------------


def anchor_lane_memberships_v6(
    anchor: HeadlineAnchor,
    lane_info: dict[str, Any],
    config: AdaptiveReadingGraphConfig,
) -> tuple[int, ...]:
    lanes = lane_info.get(
        "lanes",
        [],
    )

    memberships = []

    for lane in lanes:
        overlap = _interval_overlap_ratio_v6(
            anchor.bbox[0],
            anchor.bbox[2],
            lane["x1"],
            lane["x2"],
        )

        if (
            overlap
            >= config.reading_lane_min_overlap
        ):
            memberships.append(
                lane[
                    "lane_id"
                ]
            )

    if memberships:
        return tuple(
            memberships
        )

    if not lanes:
        return tuple()

    anchor_cx, _ = _box_center(
        anchor.bbox
    )

    nearest = min(
        lanes,
        key=lambda lane: abs(
            anchor_cx
            - lane[
                "cx"
            ]
        ),
    )

    return (
        int(
            nearest[
                "lane_id"
            ]
        ),
    )


def next_competing_headline_y_v6(
    anchor_index: int,
    anchors: Sequence[HeadlineAnchor],
    anchor_lane_ids: Sequence[int],
    lane_info: dict[str, Any],
    config: AdaptiveReadingGraphConfig,
) -> Optional[int]:
    current = anchors[
        anchor_index
    ]

    candidates = []

    for other_index, other in enumerate(
        anchors
    ):
        if other_index == anchor_index:
            continue

        if (
            other.bbox[1]
            <= current.bbox[1]
        ):
            continue

        other_lanes = anchor_lane_memberships_v6(
            other,
            lane_info,
            config,
        )

        if (
            anchor_lane_ids
            and other_lanes
            and not (
                set(
                    anchor_lane_ids
                )
                & set(
                    other_lanes
                )
            )
        ):
            continue

        horizontal = _horizontal_relation_boxes(
            current.bbox,
            other.bbox,
        )

        if horizontal < 0.12:
            continue

        candidates.append(
            other.bbox[1]
        )

    return (
        min(candidates)
        if candidates
        else None
    )


# ---------------------------------------------------------------------
# 24.5 READING-ORDER AFFINITY
# ---------------------------------------------------------------------


def reading_order_affinity_v6(
    anchor_index: int,
    anchor: HeadlineAnchor,
    node: ReadingOrderNode,
    anchors: Sequence[HeadlineAnchor],
    lane_info: dict[str, Any],
    body_height: float,
    config: AdaptiveReadingGraphConfig,
) -> dict[str, Any]:
    anchor_lanes = anchor_lane_memberships_v6(
        anchor,
        lane_info,
        config,
    )

    node_lanes = node.lane_memberships

    same_lane = bool(
        set(
            anchor_lanes
        )
        & set(
            node_lanes
        )
    )

    node_below = (
        node.bbox[1]
        >= anchor.bbox[3]
    )

    if node_below:
        gap = float(
            node.bbox[1]
            - anchor.bbox[3]
        )
    else:
        gap = float(
            anchor.bbox[1]
            - node.bbox[3]
        )

    distance_body_heights = (
        gap
        / max(
            1.0,
            body_height,
        )
    )

    next_title_y = next_competing_headline_y_v6(
        anchor_index,
        anchors,
        anchor_lanes,
        lane_info,
        config,
    )

    before_next = (
        next_title_y is None
        or node.bbox[1]
        < next_title_y
    )

    score = 0.0

    if node_below:
        score += 0.42
    else:
        score += 0.08

    if same_lane:
        score += (
            config.reading_same_lane_bonus
        )

    # Closer visible blocks are more plausible reading successors.
    score += (
        0.27
        * math.exp(
            -distance_body_heights
            / 7.0
        )
    )

    if before_next:
        score += (
            config.reading_before_next_headline_bonus
        )
    else:
        score -= (
            config.reading_after_next_headline_penalty
        )

    if (
        same_lane
        and node_below
        and distance_body_heights
        <= 2.8
    ):
        score += (
            config.reading_immediate_successor_bonus
        )

    return {
        "score": _clamp01(
            score
        ),
        "same_lane": (
            same_lane
        ),
        "node_below": (
            node_below
        ),
        "distance_body_heights": round(
            float(
                distance_body_heights
            ),
            3,
        ),
        "before_next_headline": (
            bool(
                before_next
            )
        ),
        "anchor_lanes": tuple(
            anchor_lanes
        ),
        "node_lanes": tuple(
            node_lanes
        ),
    }


# ---------------------------------------------------------------------
# 24.6 DIRECT TITLE -> STRUCTURE ASSIGNMENT WITH READING ORDER
# ---------------------------------------------------------------------


def assign_layout_elements_with_reading_order_v6(
    anchors: Sequence[HeadlineAnchor],
    layout_elements: Sequence[LayoutElement],
    reading_nodes: Sequence[ReadingOrderNode],
    lane_info: dict[str, Any],
    separators: Sequence[Separator],
    body_height: float,
    page_shape: tuple[int, int],
    config: AdaptiveReadingGraphConfig,
) -> dict[str, Any]:
    text_classes = normalized_layout_class_set(
        config.doclayout_text_classes
    )

    figure_classes = normalized_layout_class_set(
        config.doclayout_figure_classes
    )

    caption_classes = normalized_layout_class_set(
        config.doclayout_caption_classes
    )

    candidate_classes = (
        text_classes
        | figure_classes
        | caption_classes
    )

    candidate_elements = [
        element
        for element in layout_elements
        if normalize_layout_class_name(
            element.class_name
        )
        in candidate_classes
    ]

    figures = [
        element
        for element in candidate_elements
        if normalize_layout_class_name(
            element.class_name
        )
        in figure_classes
    ]

    # Map original layout element index -> reading node.
    reading_by_element_index = {
        node.element_index: node
        for node in reading_nodes
    }

    assignments: dict[
        int,
        list[
            tuple[
                LayoutElement,
                float,
                dict[str, Any],
            ]
        ],
    ] = {
        index: []
        for index in range(
            len(anchors)
        )
    }

    element_assignment: dict[
        int,
        dict[str, Any],
    ] = {}

    unassigned_elements: list[
        LayoutElement
    ] = []

    debug_edges = []

    # Need original indexes because ReadingOrderNode.element_index
    # references layout_elements.
    candidate_with_indexes = [
        (
            original_index,
            element,
        )
        for original_index, element in enumerate(
            layout_elements
        )
        if normalize_layout_class_name(
            element.class_name
        )
        in candidate_classes
    ]

    for original_index, element in candidate_with_indexes:
        cls = normalize_layout_class_name(
            element.class_name
        )

        node = reading_by_element_index.get(
            original_index
        )

        scored = []

        for anchor_index, anchor in enumerate(
            anchors
        ):
            base_detail = graph_anchor_element_score_v5(
                anchor_index=anchor_index,
                anchor=anchor,
                element=element,
                anchors=anchors,
                figure_elements=figures,
                separators=separators,
                body_height=body_height,
                page_shape=page_shape,
                config=config,
            )

            reading_detail = {
                "score": 0.0,
                "same_lane": False,
                "node_below": False,
                "distance_body_heights": 999.0,
                "before_next_headline": True,
                "anchor_lanes": tuple(),
                "node_lanes": tuple(),
            }

            if node is not None:
                reading_detail = reading_order_affinity_v6(
                    anchor_index=anchor_index,
                    anchor=anchor,
                    node=node,
                    anchors=anchors,
                    lane_info=lane_info,
                    body_height=body_height,
                    config=config,
                )

            # Blend instead of replacing the proven V5 spatial score.
            final_score = _clamp01(
                (
                    1.0
                    - config.reading_order_weight
                )
                * float(
                    base_detail[
                        "score"
                    ]
                )
                + config.reading_order_weight
                * float(
                    reading_detail[
                        "score"
                    ]
                )
            )

            detail = {
                **base_detail,
                "spatial_score": round(
                    float(
                        base_detail[
                            "score"
                        ]
                    ),
                    4,
                ),
                "reading_score": round(
                    float(
                        reading_detail[
                            "score"
                        ]
                    ),
                    4,
                ),
                "same_reading_lane": (
                    reading_detail[
                        "same_lane"
                    ]
                ),
                "before_next_headline": (
                    reading_detail[
                        "before_next_headline"
                    ]
                ),
                "reading_distance_body_heights": (
                    reading_detail[
                        "distance_body_heights"
                    ]
                ),
                "final_score": round(
                    float(
                        final_score
                    ),
                    4,
                ),
            }

            scored.append(
                (
                    final_score,
                    anchor_index,
                    detail,
                )
            )

        if not scored:
            unassigned_elements.append(
                element
            )
            continue

        scored.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        best_score, best_anchor, best_detail = scored[0]

        second_score = (
            scored[1][0]
            if len(scored) > 1
            else 0.0
        )

        min_score = (
            config.graph_min_text_edge_score
            if cls in text_classes
            else config.graph_min_figure_edge_score
        )

        accepted = (
            best_score
            >= min_score
            and (
                (
                    best_score
                    - second_score
                )
                >= config.graph_assignment_margin
                or best_score
                >= min_score
                + 0.14
            )
        )

        element_assignment[
            original_index
        ] = {
            "accepted": (
                bool(
                    accepted
                )
            ),
            "anchor_index": (
                int(
                    best_anchor
                )
            ),
            "score": float(
                best_score
            ),
            "second_score": float(
                second_score
            ),
        }

        if accepted:
            assignments[
                best_anchor
            ].append(
                (
                    element,
                    best_score,
                    best_detail,
                )
            )
        else:
            unassigned_elements.append(
                element
            )

        if config.graph_debug_edges:
            debug_edges.append(
                {
                    "element_index": (
                        original_index
                    ),
                    "class_name": (
                        cls
                    ),
                    "element_bbox": (
                        element.bbox
                    ),
                    "best_anchor_index": (
                        int(
                            best_anchor
                        )
                    ),
                    "best_anchor_title": (
                        anchors[
                            best_anchor
                        ].title
                    ),
                    "best_score": round(
                        float(
                            best_score
                        ),
                        4,
                    ),
                    "second_score": round(
                        float(
                            second_score
                        ),
                        4,
                    ),
                    "accepted": (
                        bool(
                            accepted
                        )
                    ),
                    **best_detail,
                }
            )

    return {
        "assignments": assignments,
        "element_assignment": (
            element_assignment
        ),
        "unassigned_elements": (
            unassigned_elements
        ),
        "debug_edges": (
            debug_edges
        ),
    }


# ---------------------------------------------------------------------
# 24.7 BUILD BODY-CONTINUATION CHAINS
# ---------------------------------------------------------------------


def body_chain_edge_v6(
    first_node: ReadingOrderNode,
    second_node: ReadingOrderNode,
    anchors: Sequence[HeadlineAnchor],
    separators: Sequence[Separator],
    body_height: float,
    page_shape: tuple[int, int],
    config: AdaptiveReadingGraphConfig,
) -> bool:
    text_classes = normalized_layout_class_set(
        config.doclayout_text_classes
    )

    caption_classes = normalized_layout_class_set(
        config.doclayout_caption_classes
    )

    allowed = (
        text_classes
        | caption_classes
    )

    if (
        first_node.class_name
        not in allowed
        or second_node.class_name
        not in allowed
    ):
        return False

    # Only chain within a common inferred reading lane.
    common_lanes = (
        set(
            first_node.lane_memberships
        )
        & set(
            second_node.lane_memberships
        )
    )

    if not common_lanes:
        return False

    # Direction: first should be above second.
    if (
        second_node.bbox[1]
        < first_node.bbox[1]
    ):
        first_node, second_node = (
            second_node,
            first_node,
        )

    gap = max(
        0.0,
        float(
            second_node.bbox[1]
            - first_node.bbox[3]
        ),
    )

    if gap > (
        config.body_chain_max_gap_body_heights
        * max(
            1.0,
            body_height,
        )
    ):
        return False

    horizontal = _horizontal_relation_boxes(
        first_node.bbox,
        second_node.bbox,
    )

    if (
        horizontal
        < config.body_chain_min_horizontal_relation
    ):
        return False

    first_cx, _ = _box_center(
        first_node.bbox
    )

    second_cx, _ = _box_center(
        second_node.bbox
    )

    mean_width = max(
        1.0,
        (
            (
                first_node.bbox[2]
                - first_node.bbox[0]
            )
            + (
                second_node.bbox[2]
                - second_node.bbox[0]
            )
        )
        / 2.0,
    )

    center_shift = (
        abs(
            first_cx
            - second_cx
        )
        / mean_width
    )

    if (
        center_shift
        > config.body_chain_max_center_shift_ratio
    ):
        return False

    if _strong_separator_between_boxes_v6(
        first_node.bbox,
        second_node.bbox,
        separators,
    ):
        return False

    # Do not continue through another visible article headline.
    corridor_y1 = (
        first_node.bbox[3]
    )

    corridor_y2 = (
        second_node.bbox[1]
    )

    if corridor_y2 > corridor_y1:
        for anchor in anchors:
            if not (
                corridor_y1
                < anchor.bbox[1]
                < corridor_y2
            ):
                continue

            if (
                _horizontal_relation_boxes(
                    first_node.bbox,
                    anchor.bbox,
                )
                >= 0.16
                or _horizontal_relation_boxes(
                    second_node.bbox,
                    anchor.bbox,
                )
                >= 0.16
            ):
                return False

    return True


def build_body_chain_components_v6(
    reading_nodes: Sequence[ReadingOrderNode],
    anchors: Sequence[HeadlineAnchor],
    separators: Sequence[Separator],
    body_height: float,
    page_shape: tuple[int, int],
    config: AdaptiveReadingGraphConfig,
) -> list[list[int]]:
    text_classes = normalized_layout_class_set(
        config.doclayout_text_classes
    )

    caption_classes = normalized_layout_class_set(
        config.doclayout_caption_classes
    )

    allowed = (
        text_classes
        | caption_classes
    )

    eligible_ids = [
        node.node_id
        for node in reading_nodes
        if node.class_name
        in allowed
    ]

    if not eligible_ids:
        return []

    parent = {
        node_id: node_id
        for node_id in eligible_ids
    }

    def find(
        node_id: int,
    ) -> int:
        while (
            parent[
                node_id
            ]
            != node_id
        ):
            parent[
                node_id
            ] = parent[
                parent[
                    node_id
                ]
            ]
            node_id = parent[
                node_id
            ]

        return node_id

    def union(
        first_id: int,
        second_id: int,
    ) -> None:
        root_a = find(
            first_id
        )

        root_b = find(
            second_id
        )

        if root_a != root_b:
            parent[
                root_b
            ] = root_a

    # Compare consecutive/nearby blocks in each inferred lane.
    by_lane: dict[
        int,
        list[int],
    ] = {}

    for node_id in eligible_ids:
        node = reading_nodes[
            node_id
        ]

        for lane_id in node.lane_memberships:
            by_lane.setdefault(
                lane_id,
                [],
            ).append(
                node_id
            )

    for lane_id, node_ids in by_lane.items():
        ordered = sorted(
            set(
                node_ids
            ),
            key=lambda node_id: (
                reading_nodes[
                    node_id
                ].bbox[1],
                reading_nodes[
                    node_id
                ].bbox[0],
            ),
        )

        for position, first_id in enumerate(
            ordered
        ):
            # Nearby 3 nodes are enough; farther blocks should not
            # jump over unrelated newspaper content.
            for second_id in ordered[
                position + 1:
                position + 4
            ]:
                if body_chain_edge_v6(
                    reading_nodes[
                        first_id
                    ],
                    reading_nodes[
                        second_id
                    ],
                    anchors,
                    separators,
                    body_height,
                    page_shape,
                    config,
                ):
                    union(
                        first_id,
                        second_id,
                    )

    components: dict[
        int,
        list[int],
    ] = {}

    for node_id in eligible_ids:
        components.setdefault(
            find(
                node_id
            ),
            [],
        ).append(
            node_id
        )

    return [
        sorted(
            members,
            key=lambda node_id: (
                reading_nodes[
                    node_id
                ].bbox[1],
                reading_nodes[
                    node_id
                ].bbox[0],
            ),
        )
        for members in components.values()
    ]


# ---------------------------------------------------------------------
# 24.8 PROPAGATE ARTICLE LABELS THROUGH BODY CHAINS
# ---------------------------------------------------------------------


def propagate_assignments_through_body_chains_v6(
    reading_nodes: Sequence[ReadingOrderNode],
    components: Sequence[Sequence[int]],
    direct_assignment: dict[int, dict[str, Any]],
    layout_elements: Sequence[LayoutElement],
    assignments: dict[
        int,
        list[
            tuple[
                LayoutElement,
                float,
                dict[str, Any],
            ]
        ],
    ],
    config: AdaptiveReadingGraphConfig,
) -> dict[str, Any]:
    node_by_element_index = {
        node.element_index: node
        for node in reading_nodes
    }

    propagated = 0
    unresolved_components = []

    for component in components:
        evidence: dict[
            int,
            list[float],
        ] = {}

        component_element_indexes = [
            reading_nodes[
                node_id
            ].element_index
            for node_id in component
        ]

        for element_index in component_element_indexes:
            direct = direct_assignment.get(
                element_index
            )

            if not direct:
                continue

            if not direct.get(
                "accepted",
                False,
            ):
                continue

            anchor_index = int(
                direct[
                    "anchor_index"
                ]
            )

            evidence.setdefault(
                anchor_index,
                [],
            ).append(
                float(
                    direct[
                        "score"
                    ]
                )
            )

        if not evidence:
            unresolved_components.append(
                list(
                    component
                )
            )
            continue

        ranked = sorted(
            [
                (
                    float(
                        np.mean(
                            scores
                        )
                    ),
                    anchor_index,
                    len(
                        scores
                    ),
                )
                for anchor_index, scores in evidence.items()
            ],
            reverse=True,
        )

        best_score, best_anchor, best_count = ranked[0]

        second_score = (
            ranked[1][0]
            if len(
                ranked
            )
            > 1
            else 0.0
        )

        if (
            best_score
            < config.body_chain_min_seed_score
            or (
                len(
                    ranked
                )
                > 1
                and (
                    best_score
                    - second_score
                )
                < config.body_chain_min_label_margin
            )
        ):
            unresolved_components.append(
                list(
                    component
                )
            )
            continue

        # Existing element indexes already assigned to this article.
        assigned_indexes = {
            element_index
            for element_index, direct in direct_assignment.items()
            if direct.get(
                "accepted",
                False,
            )
            and int(
                direct[
                    "anchor_index"
                ]
            )
            == best_anchor
        }

        for element_index in component_element_indexes:
            if element_index in assigned_indexes:
                continue

            element = layout_elements[
                element_index
            ]

            # Avoid duplicates if already appended by propagation.
            existing_boxes = {
                item[
                    0
                ].bbox
                for item in assignments[
                    best_anchor
                ]
            }

            if element.bbox in existing_boxes:
                continue

            assignments[
                best_anchor
            ].append(
                (
                    element,
                    max(
                        0.34,
                        min(
                            0.75,
                            best_score
                            * 0.88,
                        ),
                    ),
                    {
                        "reason": (
                            "reading_order_body_chain_propagation"
                        ),
                        "chain_seed_score": round(
                            float(
                                best_score
                            ),
                            4,
                        ),
                        "chain_seed_count": (
                            int(
                                best_count
                            )
                        ),
                    },
                )
            )

            propagated += 1

    return {
        "assignments": (
            assignments
        ),
        "propagated_element_count": (
            propagated
        ),
        "unresolved_components": (
            unresolved_components
        ),
    }


# ---------------------------------------------------------------------
# 24.9 CREATE CONTINUATION REGIONS FROM UNRESOLVED CHAINS
# ---------------------------------------------------------------------


def continuation_regions_from_unresolved_components_v6(
    unresolved_components: Sequence[Sequence[int]],
    reading_nodes: Sequence[ReadingOrderNode],
    layout_elements: Sequence[LayoutElement],
    usable_lines: Sequence[OCRLine],
    document_type: str,
    body_height: float,
    page_shape: tuple[int, int],
    config: AdaptiveReadingGraphConfig,
) -> list[dict[str, Any]]:
    text_classes = normalized_layout_class_set(
        config.doclayout_text_classes
    )

    caption_classes = normalized_layout_class_set(
        config.doclayout_caption_classes
    )

    allowed = (
        text_classes
        | caption_classes
    )

    page_h, _ = page_shape

    edge_margin = max(
        4,
        int(
            page_h
            * config.capture_edge_margin_ratio
        ),
    )

    output = []

    for component in unresolved_components:
        element_indexes = [
            reading_nodes[
                node_id
            ].element_index
            for node_id in component
            if reading_nodes[
                node_id
            ].class_name
            in allowed
        ]

        if not element_indexes:
            continue

        elements = [
            layout_elements[
                element_index
            ]
            for element_index in element_indexes
        ]

        lines = _collect_lines_from_elements(
            usable_lines,
            elements,
            config,
        )

        if (
            len(
                lines
            )
            < config.continuation_min_ocr_lines
        ):
            continue

        bbox = _box_union(
            [
                element.bbox
                for element in elements
            ]
            + [
                bbox_from_lines(
                    lines
                )
            ]
        )

        partial_top = (
            bbox[1]
            <= edge_margin
        )

        partial_bottom = (
            bbox[3]
            >= page_h
            - edge_margin
        )

        # A no-headline interior component is suspicious.
        # Only accept it if it is large; edge components need less text
        # because they naturally represent article continuation.
        synthetic_anchor = HeadlineAnchor(
            lines=[],
            title="",
            bbox=(
                bbox[0],
                bbox[1],
                bbox[2],
                min(
                    bbox[3],
                    bbox[1]
                    + max(
                        1,
                        int(
                            body_height
                        ),
                    ),
                ),
            ),
            score=0.0,
        )

        text = article_lines_to_text(
            lines,
            synthetic_anchor,
            document_type,
            body_height,
        )

        text = remove_newspaper_artifacts(
            text
        )

        word_count = len(
            text.split()
        )

        minimum = (
            config.continuation_min_words
            if (
                partial_top
                or partial_bottom
            )
            else (
                2
                * config.continuation_min_words
            )
        )

        if word_count < minimum:
            continue

        if (
            ocr_quality_score(
                text
            )
            < config.graph_min_article_ocr_quality
        ):
            continue

        mean_conf = float(
            np.mean(
                [
                    element.confidence
                    for element in elements
                ]
            )
        )

        output.append(
            {
                "anchor": (
                    synthetic_anchor
                ),
                "lines": (
                    lines
                ),
                "bbox": (
                    bbox
                ),
                "text": (
                    text
                ),
                "layout_confidence": _clamp01(
                    0.31
                    + 0.36
                    * mean_conf
                    + (
                        0.16
                        if (
                            partial_top
                            or partial_bottom
                        )
                        else 0.0
                    )
                ),
                "warnings": [
                    (
                        "No reliable visible headline was linked to this text chain. "
                        "The region is treated as visible continuation content."
                    )
                ],
                "source": (
                    "local_reading_order_continuation_v6"
                ),
                "headline_visible": False,
                "partial_top": (
                    partial_top
                ),
                "partial_bottom": (
                    partial_bottom
                ),
                "touches_left_edge": False,
                "touches_right_edge": False,
                "assigned_layout_element_count": len(
                    elements
                ),
                "mean_graph_edge_score": 0.0,
                "assigned_line_uids": [
                    line.uid
                    for line in lines
                ],
            }
        )

    return output


# ---------------------------------------------------------------------
# 24.10 V6 DETECTOR
# ---------------------------------------------------------------------


def _title_anchor_duplicate(
    first: HeadlineAnchor,
    second: HeadlineAnchor,
) -> bool:
    iou = bbox_iou(first.bbox, second.bbox)
    if iou >= 0.42:
        return True
    if bbox_containment(first.bbox, second.bbox) >= 0.72:
        return True
    if bbox_containment(second.bbox, first.bbox) >= 0.72:
        return True
    return title_similarity(first.title, second.title) >= 0.82


def _body_lines_between_titles(
    upper: HeadlineAnchor,
    lower: HeadlineAnchor,
    lines: Sequence[OCRLine],
    body_height: float,
) -> list[OCRLine]:
    band = (
        min(upper.bbox[0], lower.bbox[0]),
        upper.bbox[3],
        max(upper.bbox[2], lower.bbox[2]),
        lower.bbox[1],
    )
    if band[3] <= band[1]:
        return []
    selected = []
    for line in lines:
        if line.height > 1.55 * body_height:
            continue
        if looks_like_byline(line.text) or is_noise_line(line.text, "Newspaper"):
            continue
        if line_inside_box(line, band, margin=2):
            selected.append(line)
    return selected


def select_ai_style_title_anchors(
    seed_info: dict[str, Any],
    lines: Sequence[OCRLine],
    body_height: float,
    page_shape: tuple[int, int] | None = None,
    config: PipelineConfig | None = None,
) -> list[HeadlineAnchor]:
    """
    One editorial title should become one article.

    Prefer DocLayout titles when at least two exist. Mixing in every OCR
    headline candidate (decks, subheads, section labels) inflates the
    local article count and duplicates body text across regions.
    """
    model_anchors = [
        anchor
        for anchor in (seed_info.get("model_anchors") or [])
        if anchor and str(anchor.title).strip()
    ]
    ocr_anchors = [
        anchor
        for anchor in (
            seed_info.get("filtered_anchors")
            or seed_info.get("geometry_anchors")
            or []
        )
        if anchor and str(anchor.title).strip()
    ]

    source = model_anchors if len(model_anchors) >= 2 else (model_anchors + ocr_anchors)

    combined: list[HeadlineAnchor] = []
    for anchor in source:
        title = str(anchor.title).strip()
        if not title or is_masthead_or_metadata(title):
            continue
        replaced = False
        for index, existing in enumerate(combined):
            if not _title_anchor_duplicate(anchor, existing):
                continue
            if float(anchor.score) > float(existing.score):
                combined[index] = anchor
            replaced = True
            break
        if not replaced:
            combined.append(anchor)

    combined.sort(key=lambda item: (item.bbox[1], item.bbox[0]))

    # Merge kicker + main headline only (stacked titles, no real body between).
    merged: list[HeadlineAnchor] = []
    skip: set[int] = set()
    for index, anchor in enumerate(combined):
        if index in skip:
            continue
        chosen = anchor
        for other_index in range(index + 1, len(combined)):
            other = combined[other_index]
            overlap = horizontal_overlap_boxes(anchor.bbox, other.bbox)
            gap = other.bbox[1] - chosen.bbox[3]
            if overlap < 0.48 or gap > 1.7 * body_height:
                continue
            between = _body_lines_between_titles(chosen, other, lines, body_height)
            between_words = sum(line.word_count for line in between)
            if len(between) >= 3 or between_words >= 18:
                continue
            skip.add(other_index)
            title = chosen.title
            other_title = other.title.strip()
            if other_title and other_title.lower() not in title.lower():
                title = f"{title} {other_title}".strip()
            chosen = HeadlineAnchor(
                lines=list(
                    {
                        line.uid: line
                        for line in list(chosen.lines) + list(other.lines)
                    }.values()
                ),
                title=title,
                bbox=(
                    min(chosen.bbox[0], other.bbox[0]),
                    min(chosen.bbox[1], other.bbox[1]),
                    max(chosen.bbox[2], other.bbox[2]),
                    max(chosen.bbox[3], other.bbox[3]),
                ),
                score=max(float(chosen.score), float(other.score)),
            )
        merged.append(chosen)

    if page_shape is not None and config is not None and len(merged) >= 3:
        kept = [
            anchor
            for anchor in merged
            if not probable_section_heading(anchor, merged, page_shape, config)
        ]
        if len(kept) >= 2:
            merged = kept

    return merged


def build_one_title_one_article_regions(
    *,
    lines: Sequence[OCRLine],
    seed_info: dict[str, Any],
    separators: Sequence[Separator],
    content_area: tuple[int, int],
    page_shape: tuple[int, int],
    body_height: float,
    document_type: str,
    config: PipelineConfig,
) -> dict[str, Any]:
    """
    AI-style local split: each remaining title starts a new article.
    Body text is collected only until the next title in the same column.
    """
    content_start, content_end = content_area
    page_h, page_w = page_shape
    anchors = select_ai_style_title_anchors(
        seed_info,
        lines,
        body_height,
        page_shape=page_shape,
        config=config,
    )
    min_words = max(18, min(int(getattr(config, "min_article_words", 30)), 24))

    regions: list[dict[str, Any]] = []
    warnings: list[str] = []

    if not anchors:
        warnings.append("No DocLayout/OCR titles were available for one-title splitting.")
        return {"regions": [], "anchors": [], "warnings": warnings}

    boxed: list[tuple[HeadlineAnchor, tuple[int, int, int, int]]] = []
    for anchor in anchors:
        initial_left = max(0, int(anchor.bbox[0] - 0.04 * page_w))
        initial_right = min(page_w, int(anchor.bbox[2] + 0.04 * page_w))
        provisional_bottom = content_end
        next_tops = [
            other.bbox[1]
            for other in anchors
            if other is not anchor
            and other.bbox[1] > anchor.bbox[3] + 0.2 * body_height
            and (
                initial_left <= (other.bbox[0] + other.bbox[2]) / 2 <= initial_right
                or horizontal_overlap_boxes(
                    (initial_left, anchor.bbox[1], initial_right, anchor.bbox[3]),
                    other.bbox,
                ) >= 0.45
            )
        ]
        if next_tops:
            provisional_bottom = min(next_tops)

        left, right = find_article_horizontal_bounds(
            anchor,
            lines,
            separators,
            page_shape,
            provisional_bottom,
        )
        bottom = find_next_article_boundary(
            anchor,
            anchors,
            (left, right),
            separators,
            content_end,
            body_height,
        )
        boxed.append(
            (
                anchor,
                (
                    max(0, left),
                    max(content_start, anchor.bbox[1] - int(0.25 * body_height)),
                    min(page_w, right),
                    min(content_end, bottom),
                ),
            )
        )

    assigned: dict[int, list[OCRLine]] = {index: [] for index in range(len(boxed))}
    title_line_ids = {
        line.uid: index
        for index, (anchor, _box) in enumerate(boxed)
        for line in anchor.lines
    }
    margin = int(0.12 * body_height)

    for line in lines:
        if is_noise_line(line.text, document_type):
            continue
        if line.uid in title_line_ids:
            assigned[title_line_ids[line.uid]].append(line)
            continue

        best_index = None
        best_key = None
        line_box = (line.x1, line.y1, line.x2, line.y2)
        for index, (anchor, box) in enumerate(boxed):
            if not line_inside_box(line, box, margin=margin):
                continue
            if line.cy < anchor.bbox[1] - margin:
                continue
            vertical = max(0.0, line.cy - anchor.bbox[3])
            x_overlap = horizontal_overlap_boxes(line_box, box)
            key = (vertical, -x_overlap)
            if best_key is None or key < best_key:
                best_key = key
                best_index = index
        if best_index is not None:
            assigned[best_index].append(line)

    for index, (anchor, box) in enumerate(boxed):
        region_lines = list({line.uid: line for line in assigned[index]}.values())
        for line in anchor.lines:
            if line.uid not in {item.uid for item in region_lines}:
                region_lines.append(line)

        text = article_lines_to_text(
            region_lines,
            anchor,
            document_type,
            body_height,
        )
        words = len(text.split())
        if words < min_words:
            warnings.append(
                f"Title {anchor.title!r} produced only {words} words; dropped as a fragment."
            )
            continue

        article_bbox = bbox_from_lines(region_lines) if region_lines else box
        regions.append(
            {
                "anchor": anchor,
                "lines": region_lines,
                "bbox": article_bbox,
                "text": text,
                "layout_confidence": round(min(1.0, 0.42 + min(0.4, words / 280.0)), 3),
                "warnings": [],
                "source": "local_one_title_one_article",
                "partial_top": article_bbox[1] <= content_start + int(0.8 * body_height),
                "partial_bottom": article_bbox[3] >= content_end - int(0.8 * body_height),
            }
        )

    regions.sort(key=lambda region: (region["bbox"][1], region["bbox"][0]))
    return {
        "regions": regions,
        "anchors": anchors,
        "warnings": warnings,
    }


def detect_reading_order_graph_articles_v6(
    lines: Sequence[OCRLine],
    prepared_gray: np.ndarray,
    prepared_color: np.ndarray,
    document_type: str,
    config: AdaptiveReadingGraphConfig,
    doclayout_model=None,
) -> dict[str, Any]:
    page_shape = prepared_gray.shape[
        :2
    ]

    separators = detect_page_separators(
        prepared_gray
    )

    body_height = estimate_body_height(
        lines,
        page_shape,
    )

    content_area = detect_adaptive_content_area(
        lines,
        separators,
        page_shape,
        body_height,
        config,
    )

    usable_lines = [
        line
        for line in lines
        if (
            content_area[
                0
            ]
            <= line.cy
            <= content_area[
                1
            ]
        )
        and not is_noise_line(
            line.text,
            document_type,
        )
    ]

    warnings = []

    # ----------------------------------------------------------
    # A. Local DocLayout inference
    # ----------------------------------------------------------

    layout_elements = []

    if (
        config.use_doclayout_yolo
        and doclayout_model
        is not None
    ):
        try:
            layout_elements = detect_doclayout_elements(
                doclayout_model,
                prepared_color,
                config,
            )
        except Exception as error:
            warnings.append(
                "DocLayout-YOLO inference failed in V6: "
                f"{error}"
            )

    # ----------------------------------------------------------
    # B. Build generic headline seeds
    # ----------------------------------------------------------

    seed_info = build_adaptive_headline_seeds_v5(
        usable_lines=usable_lines,
        layout_elements=layout_elements,
        separators=separators,
        body_height=body_height,
        content_area=content_area,
        page_shape=page_shape,
        document_type=document_type,
        config=config,
    )

    anchors = seed_info[
        "anchors"
    ]

    title_split = build_one_title_one_article_regions(
        lines=usable_lines,
        seed_info=seed_info,
        separators=separators,
        content_area=content_area,
        page_shape=page_shape,
        body_height=body_height,
        document_type=document_type,
        config=config,
    )
    warnings.extend(title_split.get("warnings") or [])

    # ----------------------------------------------------------
    # C. Explicit local reading-order layer
    # ----------------------------------------------------------

    reading_nodes = build_reading_order_nodes_v6(
        layout_elements,
        usable_lines,
        config,
    )

    lane_info = infer_reading_lanes_v6(
        reading_nodes,
        page_shape,
        config,
    )

    # ----------------------------------------------------------
    # D. Direct headline -> layout association
    # ----------------------------------------------------------

    direct = assign_layout_elements_with_reading_order_v6(
        anchors=anchors,
        layout_elements=layout_elements,
        reading_nodes=reading_nodes,
        lane_info=lane_info,
        separators=separators,
        body_height=body_height,
        page_shape=page_shape,
        config=config,
    )

    # ----------------------------------------------------------
    # E. Body -> body continuation graph
    # ----------------------------------------------------------

    components = build_body_chain_components_v6(
        reading_nodes=reading_nodes,
        anchors=anchors,
        separators=separators,
        body_height=body_height,
        page_shape=page_shape,
        config=config,
    )

    propagation = propagate_assignments_through_body_chains_v6(
        reading_nodes=reading_nodes,
        components=components,
        direct_assignment=direct[
            "element_assignment"
        ],
        layout_elements=layout_elements,
        assignments=direct[
            "assignments"
        ],
        config=config,
    )

    assignments = propagation[
        "assignments"
    ]

    # ----------------------------------------------------------
    # F. Build titled article regions
    # ----------------------------------------------------------

    titled_regions = build_graph_seed_regions_v5(
        usable_lines=usable_lines,
        anchors=anchors,
        assignments=assignments,
        document_type=document_type,
        body_height=body_height,
        page_shape=page_shape,
        config=config,
    )

    for region in titled_regions:
        region[
            "source"
        ] = (
            "local_doclayout_reading_order_graph_v6"
        )

    # ----------------------------------------------------------
    # G. Build no-headline continuation regions
    # ----------------------------------------------------------

    continuation_regions = (
        continuation_regions_from_unresolved_components_v6(
            unresolved_components=propagation[
                "unresolved_components"
            ],
            reading_nodes=reading_nodes,
            layout_elements=layout_elements,
            usable_lines=usable_lines,
            document_type=document_type,
            body_height=body_height,
            page_shape=page_shape,
            config=config,
        )
    )

    v6_regions = deduplicate_graph_regions_v5(
        list(
            titled_regions
        )
        + list(
            continuation_regions
        )
    )

    v6_quality = local_article_grouping_quality_v5(
        v6_regions,
        usable_lines,
        anchors,
    )

    # ----------------------------------------------------------
    # H. V5 fallback only when V6 is weak
    # ----------------------------------------------------------

    fallback_detection = None

    selected_method = (
        "local_doclayout_reading_order_graph_v6"
    )

    selected_regions = (
        v6_regions
    )

    selected_quality = (
        v6_quality
    )

    title_split_regions = title_split.get("regions") or []
    if title_split_regions:
        selected_method = "local_one_title_one_article"
        selected_regions = title_split_regions
        selected_quality = {
            "quality": min(1.0, 0.55 + 0.05 * min(8, len(title_split_regions))),
            "article_count": len(title_split_regions),
            "method": "local_one_title_one_article",
        }
        warnings.append(
            "Newspaper/Magazine used AI-style splitting: "
            "each DocLayout/OCR title is one article."
        )

    if (
        not title_split_regions
        and config.fallback_to_v5_when_below
        and float(
            v6_quality.get(
                "quality",
                0.0,
            )
        )
        < config.reading_graph_quality_threshold
    ):
        fallback_detection = detect_adaptive_graph_articles_v5(
            lines=lines,
            prepared_gray=prepared_gray,
            prepared_color=prepared_color,
            document_type=document_type,
            config=config,
            doclayout_model=doclayout_model,
        )

        fallback_score = float(
            fallback_detection.get(
                "selected_quality",
                {},
            ).get(
                "quality",
                0.0,
            )
        )

        v6_score = float(
            v6_quality.get(
                "quality",
                0.0,
            )
        )

        # Prefer V6 when approximately tied because it uses the explicit
        # reading-order evidence we want to evaluate as the proposed method.
        if (
            fallback_score
            > v6_score
            + config.prefer_v6_within
        ):
            selected_method = (
                fallback_detection[
                    "selected_method"
                ]
            )

            selected_regions = (
                fallback_detection[
                    "regions"
                ]
            )

            selected_quality = (
                fallback_detection[
                    "selected_quality"
                ]
            )

            warnings.extend(
                fallback_detection.get(
                    "warnings",
                    [],
                )
            )
        else:
            warnings.append(
                "V6 quality was below the preferred threshold, "
                "but the fully-local V5 fallback did not score clearly better; "
                "V6 reading-order result was retained."
            )

    return {
        "regions": (
            selected_regions
        ),
        "selected_method": (
            selected_method
        ),
        "selected_quality": (
            selected_quality
        ),
        "v6_regions": (
            v6_regions
        ),
        "v6_quality": (
            v6_quality
        ),
        "fallback_detection": (
            fallback_detection
        ),
        "anchors": (
            anchors
        ),
        "layout_elements": (
            layout_elements
        ),
        "reading_nodes": (
            reading_nodes
        ),
        "lane_info": (
            lane_info
        ),
        "body_chain_components": (
            components
        ),
        "separators": (
            separators
        ),
        "usable_lines": (
            usable_lines
        ),
        "body_height": (
            body_height
        ),
        "content_area": (
            content_area
        ),
        "warnings": (
            warnings
        ),
        "edge_debug": (
            direct[
                "debug_edges"
            ]
        ),
        "doclayout_class_counts": (
            doclayout_class_counts(
                layout_elements
            )
        ),
        "debug_counts": {
            "ocr_line_count": (
                len(
                    lines
                )
            ),
            "usable_ocr_line_count": (
                len(
                    usable_lines
                )
            ),
            "doclayout_element_count": (
                len(
                    layout_elements
                )
            ),
            "headline_seed_count": (
                len(
                    anchors
                )
            ),
            "reading_node_count": (
                len(
                    reading_nodes
                )
            ),
            "reading_lane_count": (
                len(
                    lane_info.get(
                        "lanes",
                        [],
                    )
                )
            ),
            "body_chain_component_count": (
                len(
                    components
                )
            ),
            "propagated_element_count": (
                propagation[
                    "propagated_element_count"
                ]
            ),
            "v6_titled_region_count": (
                len(
                    titled_regions
                )
            ),
            "v6_continuation_region_count": (
                len(
                    continuation_regions
                )
            ),
            "v6_region_count": (
                len(
                    v6_regions
                )
            ),
            "selected_region_count": (
                len(
                    selected_regions
                )
            ),
        },
    }


# ---------------------------------------------------------------------
# 24.11 MAIN V6 ANALYZER
# ---------------------------------------------------------------------


def analyze_content_reading_graph_local(
    image_path: str,
    document_type: str,
    model,
    config: AdaptiveReadingGraphConfig,
    doclayout_model=None,
) -> dict[str, Any]:
    normalized_type = normalize_document_type(
        document_type
    )

    # Single-content path (especially Novel/Story):
    # keep the established local OCR/category pipeline.
    if (
        normalized_type
        not in MULTI_CATEGORY_TYPES
    ):
        analysis = _BASE_LOCAL_ANALYZER(
            image_path=image_path,
            document_type=normalized_type,
            model=model,
            config=config,
        )

        analysis[
            "pipeline_version"
        ] = (
            "V6_LOCAL_SINGLE_DOCUMENT"
        )

        analysis[
            "processing"
        ] = {
            "cloud_api_used": False,
            "openai_used": False,
            "ocr": (
                "local_tesseract"
            ),
            "article_detection": (
                "not_required_for_single_content_document"
            ),
            "category_model": (
                "local_tfidf_logistic_regression"
            ),
            "speech_input": (
                "local_vosk"
            ),
            "summary_model": (
                f"local_ollama_{config.ollama_model}"
            ),
        }

        return analysis

    image = cv2.imread(
        str(
            image_path
        )
    )

    if image is None:
        raise FileNotFoundError(
            f"Image not found or unreadable: {image_path}"
        )

    prepared_color, prepared_gray, scale = prepare_page_image(
        image,
        config,
    )

    page_shape = prepared_gray.shape[
        :2
    ]

    ocr_lines = extract_ocr_lines(
        prepared_gray,
        normalized_type,
        config,
    )

    detection = detect_reading_order_graph_articles_v6(
        lines=ocr_lines,
        prepared_gray=prepared_gray,
        prepared_color=prepared_color,
        document_type=normalized_type,
        config=config,
        doclayout_model=doclayout_model,
    )

    results = []

    for number, region in enumerate(
        detection[
            "regions"
        ],
        start=1,
    ):
        anchor = region.get(
            "anchor"
        )

        raw_title = (
            anchor.title.strip()
            if (
                anchor is not None
                and anchor.title
            )
            else ""
        )

        display_title = (
            raw_title
            if raw_title
            else "Visible article continuation"
        )

        text = remove_newspaper_artifacts(
            region.get(
                "text",
                "",
            )
        )

        if not text.strip():
            continue

        prediction = predict_category(
            model=model,
            text=text,
            document_type=normalized_type,
            title=raw_title,
            config=config,
        )

        partial_top = bool(
            region.get(
                "partial_top",
                False,
            )
        )

        partial_bottom = bool(
            region.get(
                "partial_bottom",
                False,
            )
        )

        warnings = list(
            region.get(
                "warnings",
                [],
            )
        )

        if (
            partial_top
            or partial_bottom
        ):
            warnings.append(
                "This is a partial camera-visible article region. "
                "Only visible extracted text may be summarized or read."
            )

        results.append(
            {
                "article_number": (
                    number
                ),
                "title": (
                    display_title
                ),
                "raw_visible_title": (
                    raw_title
                ),
                "headline_visible": (
                    bool(
                        raw_title
                    )
                ),
                "bbox": (
                    region[
                        "bbox"
                    ]
                ),
                "normalized_bbox": normalize_bbox(
                    region[
                        "bbox"
                    ],
                    page_shape,
                ),
                "full_text": (
                    text
                ),
                "category": (
                    prediction[
                        "category"
                    ]
                ),
                "confidence": (
                    prediction[
                        "confidence"
                    ]
                ),
                "probability_margin": (
                    prediction[
                        "margin"
                    ]
                ),
                "top_predictions": (
                    prediction[
                        "top_predictions"
                    ]
                ),
                "category_decision": prediction.get(
                    "decision",
                    "",
                ),
                "ocr_quality": (
                    ocr_quality_score(
                        text
                    )
                ),
                "layout_confidence": round(
                    float(
                        region.get(
                            "layout_confidence",
                            0.0,
                        )
                    ),
                    4,
                ),
                "partial_top": (
                    partial_top
                ),
                "partial_bottom": (
                    partial_bottom
                ),
                "source": region.get(
                    "source",
                    detection[
                        "selected_method"
                    ],
                ),
                "warnings": (
                    warnings
                ),
            }
        )

    # Defensive local fallback if selected method somehow produced nothing.
    if not results:
        fallback = build_full_visible_capture_fallback_v5(
            prepared_gray=prepared_gray,
            usable_lines=detection[
                "usable_lines"
            ],
            document_type=normalized_type,
            page_shape=page_shape,
        )[0]

        text = fallback[
            "text"
        ]

        prediction = predict_category(
            model=model,
            text=text,
            document_type=normalized_type,
            title="",
            config=config,
        )

        results = [
            {
                "article_number": 1,
                "title": (
                    "Visible newspaper content"
                ),
                "raw_visible_title": "",
                "headline_visible": False,
                "bbox": (
                    fallback[
                        "bbox"
                    ]
                ),
                "normalized_bbox": (
                    0,
                    0,
                    1000,
                    1000,
                ),
                "full_text": (
                    text
                ),
                "category": (
                    prediction[
                        "category"
                    ]
                ),
                "confidence": (
                    prediction[
                        "confidence"
                    ]
                ),
                "probability_margin": (
                    prediction[
                        "margin"
                    ]
                ),
                "top_predictions": (
                    prediction[
                        "top_predictions"
                    ]
                ),
                "category_decision": prediction.get(
                    "decision",
                    "",
                ),
                "ocr_quality": (
                    ocr_quality_score(
                        text
                    )
                ),
                "layout_confidence": 0.20,
                "partial_top": False,
                "partial_bottom": False,
                "source": (
                    "local_full_visible_capture_ocr_fallback"
                ),
                "warnings": (
                    fallback[
                        "warnings"
                    ]
                ),
            }
        ]

    category_counts = Counter(
        item[
            "category"
        ]
        for item in results
    )

    categories = sorted(
        category_counts
    )

    lane_rows = []

    for lane in detection[
        "lane_info"
    ].get(
        "lanes",
        [],
    ):
        lane_rows.append(
            {
                "lane_id": lane[
                    "lane_id"
                ],
                "x1": lane[
                    "x1"
                ],
                "x2": lane[
                    "x2"
                ],
                "cx": round(
                    float(
                        lane[
                            "cx"
                        ]
                    ),
                    2,
                ),
                "member_count": len(
                    lane.get(
                        "members",
                        [],
                    )
                ),
            }
        )

    return {
        "status": (
            "ANALYZED"
        ),
        "pipeline_version": (
            "V6_LOCAL_READING_ORDER_GRAPH"
        ),
        "document_type_received": (
            document_type
        ),
        "document_type": (
            normalized_type
        ),
        "mode": (
            "Multi Category"
        ),
        "detected_categories": (
            categories
        ),
        "category_counts": dict(
            category_counts
        ),
        "needs_category_selection": (
            len(
                categories
            )
            > 1
        ),
        "image_shape": {
            "height": (
                page_shape[
                    0
                ]
            ),
            "width": (
                page_shape[
                    1
                ]
            ),
        },
        "resize_scale": (
            scale
        ),
        "warnings": (
            detection[
                "warnings"
            ]
        ),
        "local_grouping": {
            "selected_method": (
                detection[
                    "selected_method"
                ]
            ),
            "selected_quality": (
                detection[
                    "selected_quality"
                ]
            ),
            "v6_quality": (
                detection[
                    "v6_quality"
                ]
            ),
            "fallback_method": (
                (
                    detection[
                        "fallback_detection"
                    ].get(
                        "selected_method"
                    )
                )
                if detection[
                    "fallback_detection"
                ]
                else None
            ),
        },
        "layout_debug": {
            "pipeline_version": (
                "V6_LOCAL_READING_ORDER_GRAPH"
            ),
            **detection[
                "debug_counts"
            ],
            "body_height": round(
                float(
                    detection[
                        "body_height"
                    ]
                ),
                2,
            ),
            "content_area": (
                detection[
                    "content_area"
                ]
            ),
            "doclayout_class_counts": (
                detection[
                    "doclayout_class_counts"
                ]
            ),
            "reading_lanes": (
                lane_rows
            ),
            "headline_anchors": [
                {
                    "title": (
                        anchor.title
                    ),
                    "bbox": (
                        anchor.bbox
                    ),
                    "score": round(
                        float(
                            anchor.score
                        ),
                        3,
                    ),
                    "line_count": len(
                        anchor.lines
                    ),
                }
                for anchor in detection[
                    "anchors"
                ]
            ],
            "edge_debug": (
                detection[
                    "edge_debug"
                ]
            ),
            "reading_nodes": [
                {
                    "node_id": (
                        node.node_id
                    ),
                    "class_name": (
                        node.class_name
                    ),
                    "bbox": (
                        node.bbox
                    ),
                    "primary_lane": (
                        node.primary_lane
                    ),
                    "lane_memberships": (
                        node.lane_memberships
                    ),
                    "lane_rank": (
                        node.lane_rank
                    ),
                    "reading_rank": (
                        node.reading_rank
                    ),
                    "text_preview": (
                        node.text[
                            :180
                        ]
                    ),
                }
                for node in detection[
                    "reading_nodes"
                ]
            ],
        },
        "results": (
            results
        ),
        "processing": {
            "cloud_api_used": False,
            "openai_used": False,
            "ocr": (
                "local_tesseract"
            ),
            "layout_model": (
                "local_DocLayout_YOLO_DocStructBench"
            ),
            "reading_order": (
                "local_adaptive_lane_and_body_chain_graph"
            ),
            "article_grouping": (
                detection[
                    "selected_method"
                ]
            ),
            "local_fallback_chain": [
                "local_doclayout_reading_order_graph_v6",
                "local_doclayout_graph_v5",
                "local_v4_hybrid_fallback",
                "local_full_visible_capture_ocr_fallback",
            ],
            "category_model": (
                "local_tfidf_logistic_regression"
            ),
            "speech_input": (
                "local_vosk"
            ),
            "summary_model": (
                f"local_ollama_{config.ollama_model}"
            ),
        },
    }


# ---------------------------------------------------------------------
# 24.12 OUTPUT / VOICE WRAPPERS
# ---------------------------------------------------------------------


def generate_selected_output_reading_graph_local(
    analysis: dict[str, Any],
    selected_category: Optional[str],
    depth: str,
    config: AdaptiveReadingGraphConfig,
    clean_full_text_with_llama: bool = False,
) -> dict[str, Any]:
    return generate_selected_output_graph_local(
        analysis=analysis,
        selected_category=selected_category,
        depth=depth,
        config=config,
        clean_full_text_with_llama=clean_full_text_with_llama,
    )


def run_adaptive_local_voice_pipeline_v6(
    image_path: str,
    document_type: str,
    category_model,
    config: AdaptiveReadingGraphConfig,
    doclayout_model,
    voice_selector: OfflineVoiceSelector,
    tts_enabled: bool = False,
    clean_full_text_with_llama: bool = False,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
]:
    analysis = analyze_content_reading_graph_local(
        image_path=image_path,
        document_type=document_type,
        model=category_model,
        config=config,
        doclayout_model=doclayout_model,
    )

    selected_category, depth = collect_user_preferences_voice(
        analysis=analysis,
        voice_selector=voice_selector,
        tts_enabled=tts_enabled,
    )

    output = generate_selected_output_reading_graph_local(
        analysis=analysis,
        selected_category=selected_category,
        depth=depth,
        config=config,
        clean_full_text_with_llama=clean_full_text_with_llama,
    )

    return analysis, output


# ---------------------------------------------------------------------
# 24.13 V6 DEBUG TABLES
# ---------------------------------------------------------------------


def reading_order_debug_summary_v6(
    analysis: dict[str, Any],
) -> pd.DataFrame:
    local = analysis.get(
        "local_grouping",
        {},
    )

    selected = local.get(
        "selected_quality",
        {},
    )

    v6_quality = local.get(
        "v6_quality",
        {},
    )

    debug = analysis.get(
        "layout_debug",
        {},
    )

    rows = [
        (
            "Pipeline",
            analysis.get(
                "pipeline_version",
                "",
            ),
        ),
        (
            "Cloud API used",
            analysis.get(
                "processing",
                {},
            ).get(
                "cloud_api_used",
                False,
            ),
        ),
        (
            "Selected local method",
            local.get(
                "selected_method",
                "",
            ),
        ),
        (
            "Selected quality",
            selected.get(
                "quality",
                0.0,
            ),
        ),
        (
            "V6 reading-graph quality",
            v6_quality.get(
                "quality",
                0.0,
            ),
        ),
        (
            "V6 OCR-line coverage",
            v6_quality.get(
                "line_word_coverage",
                0.0,
            ),
        ),
        (
            "OCR lines",
            debug.get(
                "ocr_line_count",
                0,
            ),
        ),
        (
            "Usable OCR lines",
            debug.get(
                "usable_ocr_line_count",
                0,
            ),
        ),
        (
            "DocLayout elements",
            debug.get(
                "doclayout_element_count",
                0,
            ),
        ),
        (
            "Headline seeds",
            debug.get(
                "headline_seed_count",
                0,
            ),
        ),
        (
            "Reading nodes",
            debug.get(
                "reading_node_count",
                0,
            ),
        ),
        (
            "Reading lanes",
            debug.get(
                "reading_lane_count",
                0,
            ),
        ),
        (
            "Body-chain components",
            debug.get(
                "body_chain_component_count",
                0,
            ),
        ),
        (
            "Propagated layout elements",
            debug.get(
                "propagated_element_count",
                0,
            ),
        ),
        (
            "V6 titled regions",
            debug.get(
                "v6_titled_region_count",
                0,
            ),
        ),
        (
            "V6 continuation regions",
            debug.get(
                "v6_continuation_region_count",
                0,
            ),
        ),
        (
            "V6 total regions",
            debug.get(
                "v6_region_count",
                0,
            ),
        ),
        (
            "Final selected regions",
            debug.get(
                "selected_region_count",
                len(
                    analysis.get(
                        "results",
                        [],
                    )
                ),
            ),
        ),
    ]

    return pd.DataFrame(
        rows,
        columns=[
            "Stage",
            "Value",
        ],
    )


def reading_lane_table_v6(
    analysis: dict[str, Any],
) -> pd.DataFrame:
    return pd.DataFrame(
        analysis.get(
            "layout_debug",
            {},
        ).get(
            "reading_lanes",
            [],
        )
    )


def reading_node_table_v6(
    analysis: dict[str, Any],
) -> pd.DataFrame:
    return pd.DataFrame(
        analysis.get(
            "layout_debug",
            {},
        ).get(
            "reading_nodes",
            [],
        )
    )


def reading_edge_table_v6(
    analysis: dict[str, Any],
    accepted_only: bool = False,
) -> pd.DataFrame:
    table = pd.DataFrame(
        analysis.get(
            "layout_debug",
            {},
        ).get(
            "edge_debug",
            [],
        )
    )

    if (
        accepted_only
        and not table.empty
        and "accepted"
        in table.columns
    ):
        table = table[
            table[
                "accepted"
            ]
            == True
        ].copy()

    return table


# ================================================================
# V6.1 GENERIC FIXES
# ================================================================
#
# This cell patches the OUTPUT of V6 using the structural information
# already detected locally by DocLayout-YOLO.
#
# No publication-specific coordinates or newspaper names are used.
# No external/cloud API is used.
# ================================================================

import re
import math
from difflib import SequenceMatcher

import cv2
import numpy as np
import pandas as pd
import pytesseract


def _intersection_area_v61(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    return float((x2 - x1) * (y2 - y1))


def _bbox_area_v61(box):
    return float(
        max(0, box[2] - box[0])
        * max(0, box[3] - box[1])
    )


def _containment_v61(inner_candidate, outer):
    """
    Fraction of the candidate box covered by outer.
    Useful for checking if a DocLayout title is inside an article/table.
    """
    area = _bbox_area_v61(inner_candidate)

    if area <= 0:
        return 0.0

    return _intersection_area_v61(
        inner_candidate,
        outer
    ) / area


def _normalize_short_acronyms_v61(text):
    """
    Generic OCR cleanup:
    Converts short mixed-case tokens such as FtA -> FTA
    only when the token already contains at least two uppercase letters.
    """
    def replace(match):
        token = match.group(0)

        if (
            2 <= len(token) <= 5
            and token.isalpha()
            and sum(ch.isupper() for ch in token) >= 2
        ):
            return token.upper()

        return token

    return re.sub(
        r"\b[A-Za-z]{2,5}\b",
        replace,
        text
    )


def clean_title_text_v61(text):
    if text is None:
        return ""

    text = str(text)

    text = (
        text.replace("\r", " ")
            .replace("\n", " ")
            .replace("|", " ")
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip(" -:;|")

    text = _normalize_short_acronyms_v61(
        text
    )

    # Conservative joining repair for common OCR spacing artifacts.
    text = re.sub(
        r"\s+([,.;:!?])",
        r"\1",
        text
    )

    return text.strip()


def looks_like_byline_v61(text):
    t = clean_title_text_v61(text).lower()

    patterns = [
        r"^by\s+[a-z]",
        r"^written\s+by\s+",
        r"^reported\s+by\s+",
        r"^reporting\s+by\s+",
        r"^staff\s+reporter\b",
        r"^staff\s+writer\b",
        r"^special\s+correspondent\b",
        r"^our\s+correspondent\b",
        r"^correspondent\b",
    ]

    return any(
        re.search(pattern, t)
        for pattern in patterns
    )


def looks_like_navigation_or_page_teaser_v61(text):
    """
    Rejects generic newspaper navigation/header labels rather than
    real article headlines.
    """
    t = clean_title_text_v61(text).lower()

    if not t:
        return True

    # Common page-navigation language.
    if re.search(
        r"\bpages?\s+\d+(?:\s*[-–]\s*\d+)?\b",
        t
    ):
        return True

    if re.search(
        r"\b(?:sport|sports|business|political commentary|education)"
        r"\s+pages?\s+\d+\b",
        t
    ):
        return True

    # Contents-box language. This is still combined with table-overlap
    # rejection, so it is deliberately conservative.
    if (
        "this week" in t
        and len(t.split()) <= 10
    ):
        return True

    return False


def title_text_quality_v61(text):
    text = clean_title_text_v61(text)

    if not text:
        return 0.0

    words = text.split()

    if len(words) < 2:
        return 0.0

    if len(words) > 24:
        return 0.10

    chars = [
        ch
        for ch in text
        if not ch.isspace()
    ]

    if not chars:
        return 0.0

    alpha_ratio_local = (
        sum(ch.isalpha() for ch in chars)
        / len(chars)
    )

    word_score = min(
        1.0,
        len(words) / 7.0
    )

    length_score = min(
        1.0,
        len(text) / 45.0
    )

    score = (
        0.46 * alpha_ratio_local
        + 0.31 * word_score
        + 0.23 * length_score
    )

    if looks_like_byline_v61(text):
        score *= 0.10

    if looks_like_navigation_or_page_teaser_v61(text):
        score *= 0.20

    return float(
        max(
            0.0,
            min(
                1.0,
                score
            )
        )
    )


def ocr_title_crop_v61(
    prepared_gray,
    bbox,
):
    """
    Re-OCR a DocLayout title crop locally.
    Multi-line PSM 6 is usually strongest for newspaper headlines.
    PSM 11 is also tested as a fallback.
    """
    h_img, w_img = prepared_gray.shape[:2]

    x1, y1, x2, y2 = map(
        int,
        bbox
    )

    pad_x = max(
        4,
        int(
            0.015
            * max(
                1,
                x2 - x1
            )
        )
    )

    pad_y = max(
        3,
        int(
            0.05
            * max(
                1,
                y2 - y1
            )
        )
    )

    x1 = max(
        0,
        x1 - pad_x
    )

    y1 = max(
        0,
        y1 - pad_y
    )

    x2 = min(
        w_img,
        x2 + pad_x
    )

    y2 = min(
        h_img,
        y2 + pad_y
    )

    crop = prepared_gray[
        y1:y2,
        x1:x2
    ]

    if crop.size == 0:
        return "", []

    # Upscale smaller title crops before Tesseract.
    crop_h = crop.shape[0]

    if crop_h < 180:
        scale = max(
            2.0,
            180.0
            / max(
                1,
                crop_h
            )
        )

        crop = cv2.resize(
            crop,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC
        )

    crop = cv2.fastNlMeansDenoising(
        crop,
        None,
        8,
        7,
        21
    )

    crop = cv2.addWeighted(
        crop,
        1.45,
        cv2.GaussianBlur(
            crop,
            (0, 0),
            2
        ),
        -0.45,
        0
    )

    candidates = []

    for psm in [6, 11]:
        try:
            raw = pytesseract.image_to_string(
                crop,
                config=(
                    f"--oem 3 --psm {psm}"
                )
            )

            text = clean_title_text_v61(
                raw
            )

            if text:
                candidates.append(
                    {
                        "text": text,
                        "source": (
                            f"title_crop_psm_{psm}"
                        ),
                        "quality": (
                            title_text_quality_v61(
                                text
                            )
                        ),
                    }
                )

        except Exception:
            pass

    if not candidates:
        return "", []

    candidates = sorted(
        candidates,
        key=lambda item: (
            item["quality"],
            len(item["text"])
        ),
        reverse=True
    )

    return (
        candidates[0]["text"],
        candidates
    )


def ocr_lines_for_title_bbox_v61(
    ocr_lines,
    bbox,
):
    """
    Uses the original full-capture OCR lines as another independent
    local source for the title.
    """
    selected = []

    for line in ocr_lines:
        line_box = (
            line.x1,
            line.y1,
            line.x2,
            line.y2,
        )

        overlap = (
            _intersection_area_v61(
                line_box,
                bbox
            )
            / max(
                1.0,
                _bbox_area_v61(
                    line_box
                )
            )
        )

        cx = (
            line.x1
            + line.x2
        ) / 2.0

        cy = (
            line.y1
            + line.y2
        ) / 2.0

        center_inside = (
            bbox[0]
            <= cx
            <= bbox[2]
            and bbox[1]
            <= cy
            <= bbox[3]
        )

        if (
            overlap >= 0.25
            or center_inside
        ):
            selected.append(
                line
            )

    selected = sorted(
        selected,
        key=lambda line: (
            line.y1,
            line.x1
        )
    )

    # Deduplicate repeated OCR lines.
    texts = []

    for line in selected:
        value = clean_title_text_v61(
            line.text
        )

        if (
            value
            and value.lower()
            not in {
                item.lower()
                for item in texts
            }
        ):
            texts.append(
                value
            )

    text = clean_title_text_v61(
        " ".join(
            texts
        )
    )

    return (
        text,
        selected
    )


def title_overlaps_table_v61(
    title_bbox,
    table_bboxes,
):
    for table_bbox in table_bboxes:
        coverage = _containment_v61(
            title_bbox,
            table_bbox
        )

        if coverage >= 0.25:
            return True

    return False


def build_doclayout_title_candidates_v61(
    analysis,
    image_path,
    config,
):
    """
    Build strong title candidates from the DocLayout title boxes that
    already exist in V6 diagnostics.

    DocLayout is NOT executed again here.
    Only Tesseract title re-OCR is repeated locally.
    """
    image = cv2.imread(
        str(
            image_path
        )
    )

    if image is None:
        raise FileNotFoundError(
            image_path
        )

    prepared_color, prepared_gray, _ = prepare_page_image(
        image,
        config
    )

    ocr_lines = extract_ocr_lines(
        prepared_gray,
        analysis.get(
            "document_type",
            "Newspaper"
        ),
        config
    )

    elements = (
        analysis
        .get(
            "layout_debug",
            {}
        )
        .get(
            "doclayout_elements",
            []
        )
    )

    title_elements = [
        element
        for element in elements
        if normalize_layout_class_name(
            element.get(
                "class_name",
                ""
            )
        )
        == "title"
    ]

    table_bboxes = [
        tuple(
            element[
                "bbox"
            ]
        )
        for element in elements
        if normalize_layout_class_name(
            element.get(
                "class_name",
                ""
            )
        )
        == "table"
    ]

    body_height = float(
        analysis
        .get(
            "layout_debug",
            {}
        )
        .get(
            "body_height",
            20.0
        )
    )

    page_h, page_w = (
        prepared_gray.shape[:2]
    )

    candidates = []

    for element in title_elements:
        bbox = tuple(
            map(
                int,
                element[
                    "bbox"
                ]
            )
        )

        confidence = float(
            element.get(
                "confidence",
                0.0
            )
        )

        line_text, title_lines = (
            ocr_lines_for_title_bbox_v61(
                ocr_lines,
                bbox
            )
        )

        crop_text, crop_candidates = (
            ocr_title_crop_v61(
                prepared_gray,
                bbox
            )
        )

        text_options = []

        if line_text:
            text_options.append(
                {
                    "text": (
                        line_text
                    ),
                    "source": (
                        "full_page_ocr_lines"
                    ),
                    "quality": (
                        title_text_quality_v61(
                            line_text
                        )
                    ),
                }
            )

        text_options.extend(
            crop_candidates
        )

        if not text_options:
            continue

        text_options = sorted(
            text_options,
            key=lambda item: (
                item[
                    "quality"
                ],
                len(
                    item[
                        "text"
                    ]
                ),
            ),
            reverse=True
        )

        best = text_options[0]

        text = clean_title_text_v61(
            best[
                "text"
            ]
        )

        table_overlap = (
            title_overlaps_table_v61(
                bbox,
                table_bboxes
            )
        )

        byline = looks_like_byline_v61(
            text
        )

        navigation = (
            looks_like_navigation_or_page_teaser_v61(
                text
            )
        )

        # Relative typography strength.
        title_height = max(
            1,
            bbox[3]
            - bbox[1]
        )

        typography_strength = min(
            1.0,
            title_height
            / max(
                1.0,
                4.0
                * body_height
            )
        )

        title_quality = float(
            best[
                "quality"
            ]
        )

        structural_score = (
            0.34
            * title_quality
            + 0.32
            * min(
                1.0,
                confidence
            )
            + 0.34
            * typography_strength
        )

        if table_overlap:
            structural_score *= 0.05

        if byline:
            structural_score *= 0.05

        if navigation:
            structural_score *= 0.15

        candidates.append(
            {
                "text": (
                    text
                ),
                "bbox": (
                    bbox
                ),
                "doclayout_confidence": (
                    confidence
                ),
                "ocr_source": (
                    best[
                        "source"
                    ]
                ),
                "ocr_quality": round(
                    title_quality,
                    4
                ),
                "typography_strength": round(
                    typography_strength,
                    4
                ),
                "structural_score": round(
                    float(
                        structural_score
                    ),
                    4
                ),
                "table_overlap": (
                    bool(
                        table_overlap
                    )
                ),
                "is_byline": (
                    bool(
                        byline
                    )
                ),
                "is_navigation": (
                    bool(
                        navigation
                    )
                ),
                "accepted_as_candidate": (
                    not table_overlap
                    and not byline
                    and not navigation
                    and title_quality >= 0.35
                    and len(
                        text.split()
                    )
                    >= 2
                ),
                "all_ocr_options": (
                    text_options
                ),
            }
        )

    return {
        "candidates": (
            candidates
        ),
        "prepared_gray": (
            prepared_gray
        ),
        "ocr_lines": (
            ocr_lines
        ),
        "table_bboxes": (
            table_bboxes
        ),
        "page_shape": (
            (
                page_h,
                page_w
            )
        ),
    }


def score_title_for_article_v61(
    article,
    candidate,
    body_height,
):
    if not candidate.get(
        "accepted_as_candidate",
        False
    ):
        return 0.0

    article_bbox = tuple(
        article[
            "bbox"
        ]
    )

    title_bbox = tuple(
        candidate[
            "bbox"
        ]
    )

    inside = _containment_v61(
        title_bbox,
        article_bbox
    )

    if inside < 0.20:
        return 0.0

    article_height = max(
        1,
        article_bbox[3]
        - article_bbox[1]
    )

    # Prefer titles in the upper part of an article, but allow a lower
    # title when a contents box/header incorrectly extended the old region.
    relative_y = (
        title_bbox[1]
        - article_bbox[1]
    ) / article_height

    vertical_score = math.exp(
        -max(
            0.0,
            relative_y
        )
        / 0.55
    )

    title_height = max(
        1,
        title_bbox[3]
        - title_bbox[1]
    )

    typography = min(
        1.0,
        title_height
        / max(
            1.0,
            4.0
            * body_height
        )
    )

    word_count = len(
        candidate[
            "text"
        ].split()
    )

    word_score = min(
        1.0,
        word_count / 7.0
    )

    score = (
        0.30
        * float(
            candidate[
                "structural_score"
            ]
        )
        + 0.25
        * inside
        + 0.19
        * typography
        + 0.14
        * vertical_score
        + 0.12
        * word_score
    )

    return float(
        max(
            0.0,
            min(
                1.0,
                score
            )
        )
    )


def normalize_article_continuation_artifacts_v61(
    text
):
    """
    Removes only the continuation instruction itself.

    Example:
      "visit to Please turn Colombo"
      -> "visit to Colombo"

    Example:
      "Please turn to Page 2"
      -> ""
    """
    if text is None:
        return ""

    text = str(
        text
    )

    # Remove full explicit page instruction.
    text = re.sub(
        r"(?i)\bplease\s+turn\s+to\s+page\s+\d+\b",
        " ",
        text
    )

    text = re.sub(
        r"(?i)\bturn\s+to\s+page\s+\d+\b",
        " ",
        text
    )

    text = re.sub(
        r"(?i)\bcontinued\s+on\s+page\s+\d+\b",
        " ",
        text
    )

    # OCR sometimes loses "to Page 2" but keeps "Please turn".
    # Remove only those two words so surrounding article text survives.
    text = re.sub(
        r"(?i)\bplease\s+turn\b",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    text = re.sub(
        r"\s+([,.;:!?])",
        r"\1",
        text
    )

    return text.strip()


def _normalized_title_tokens_v61(text):
    return {
        token
        for token in re.findall(
            r"[a-z0-9]+",
            clean_title_text_v61(
                text
            ).lower()
        )
        if len(token) >= 2
    }


def remove_old_title_prefix_v61(
    text,
    old_title,
):
    text = str(
        text or ""
    ).strip()

    old_title = clean_title_text_v61(
        old_title
    )

    if not (
        text
        and old_title
    ):
        return text

    # Exact old title at the beginning.
    if (
        text.lower()
        .startswith(
            old_title.lower()
        )
    ):
        text = text[
            len(
                old_title
            ):
        ].lstrip(
            " .,:;-"
        )

    return text.strip()


def remove_duplicate_title_fragments_v61(
    body,
    corrected_title,
):
    """
    Newspaper OCR can split one headline into multiple short sentences.
    If the first 1-4 short sentences mostly repeat the corrected headline,
    remove them before prepending the reconstructed full title.
    """
    body = str(
        body or ""
    ).strip()

    if not body:
        return body

    title_tokens = _normalized_title_tokens_v61(
        corrected_title
    )

    if not title_tokens:
        return body

    sentence_matches = list(
        re.finditer(
            r"[^.!?]+[.!?]?",
            body
        )
    )

    if not sentence_matches:
        return body

    combined = ""

    best_end = 0
    best_overlap = 0.0

    for match in sentence_matches[
        :4
    ]:
        combined += " " + match.group(
            0
        )

        combined_tokens = (
            _normalized_title_tokens_v61(
                combined
            )
        )

        if not combined_tokens:
            continue

        overlap = (
            len(
                title_tokens
                & combined_tokens
            )
            / max(
                1,
                len(
                    title_tokens
                )
            )
        )

        # Avoid consuming a real body paragraph.
        if (
            len(
                combined_tokens
            )
            <= len(
                title_tokens
            )
            + 7
            and overlap
            > best_overlap
        ):
            best_overlap = overlap
            best_end = match.end()

    if (
        best_overlap
        >= 0.48
        and best_end
        > 0
    ):
        body = body[
            best_end:
        ].lstrip(
            " .,:;-"
        )

    return body.strip()


def choose_best_title_for_article_v61(
    article,
    candidates,
    body_height,
):
    ranked = []

    for candidate in candidates:
        score = score_title_for_article_v61(
            article,
            candidate,
            body_height
        )

        if score <= 0:
            continue

        ranked.append(
            {
                **candidate,
                "article_match_score": round(
                    score,
                    4
                ),
            }
        )

    if not ranked:
        return None, []

    ranked = sorted(
        ranked,
        key=lambda item: (
            item[
                "article_match_score"
            ],
            item[
                "typography_strength"
            ],
            item[
                "doclayout_confidence"
            ],
        ),
        reverse=True
    )

    return (
        ranked[0],
        ranked
    )


def postprocess_analysis_v61(
    analysis,
    image_path,
    category_model,
    config,
):
    """
    Correct headline reconstruction and article text AFTER V6 has formed
    article regions.

    This deliberately does NOT use a fixed expected article count.
    """
    title_data = (
        build_doclayout_title_candidates_v61(
            analysis,
            image_path,
            config
        )
    )

    candidates = title_data[
        "candidates"
    ]

    body_height = float(
        analysis
        .get(
            "layout_debug",
            {}
        )
        .get(
            "body_height",
            20.0
        )
    )

    article_debug = []

    for article in analysis.get(
        "results",
        []
    ):
        old_title = article.get(
            "title",
            ""
        )

        best, ranked = (
            choose_best_title_for_article_v61(
                article,
                candidates,
                body_height
            )
        )

        corrected_title = (
            best[
                "text"
            ]
            if best
            else old_title
        )

        corrected_title = (
            clean_title_text_v61(
                corrected_title
            )
        )

        text = article.get(
            "full_text",
            ""
        )

        text = (
            normalize_article_continuation_artifacts_v61(
                text
            )
        )

        # Remove the old short/wrong seed title.
        if (
            corrected_title
            and corrected_title.lower()
            != clean_title_text_v61(
                old_title
            ).lower()
        ):
            text = remove_old_title_prefix_v61(
                text,
                old_title
            )

        # Remove OCR fragments that are already represented by the
        # reconstructed complete title.
        text = (
            remove_duplicate_title_fragments_v61(
                text,
                corrected_title
            )
        )

        if corrected_title:
            body_text = text.strip()

            # Avoid duplicating the title when it already starts correctly.
            if not (
                body_text.lower()
                .startswith(
                    corrected_title.lower()
                )
            ):
                text = (
                    corrected_title.rstrip(
                        "."
                    )
                    + ". "
                    + body_text
                ).strip()

        article[
            "title"
        ] = (
            corrected_title
            if corrected_title
            else article.get(
                "title",
                "Visible article continuation"
            )
        )

        article[
            "raw_visible_title"
        ] = (
            corrected_title
        )

        article[
            "headline_visible"
        ] = bool(
            corrected_title
        )

        article[
            "full_text"
        ] = text

        article[
            "headline_reconstruction_source"
        ] = (
            best[
                "ocr_source"
            ]
            if best
            else "original_v6_title"
        )

        article[
            "headline_reconstruction_score"
        ] = (
            round(
                float(
                    best[
                        "article_match_score"
                    ]
                ),
                4
            )
            if best
            else 0.0
        )

        # Re-run category using corrected title + cleaned article text.
        prediction = predict_category(
            model=category_model,
            text=text,
            document_type=analysis.get(
                "document_type",
                "Newspaper"
            ),
            title=corrected_title,
            config=config,
        )

        article[
            "category"
        ] = prediction[
            "category"
        ]

        article[
            "confidence"
        ] = prediction[
            "confidence"
        ]

        article[
            "probability_margin"
        ] = prediction[
            "margin"
        ]

        article[
            "top_predictions"
        ] = prediction[
            "top_predictions"
        ]

        article[
            "category_decision"
        ] = prediction.get(
            "decision",
            ""
        )

        article[
            "ocr_quality"
        ] = ocr_quality_score(
            text
        )

        # V6.1 ignores arbitrary figure boxes when extending article
        # boundaries. The config below already prevents figure assignment,
        # so partial_bottom / partial_top now come from text/layout content.
        article_debug.append(
            {
                "article": article[
                    "article_number"
                ],
                "old_title": (
                    old_title
                ),
                "corrected_title": (
                    corrected_title
                ),
                "headline_score": (
                    article[
                        "headline_reconstruction_score"
                    ]
                ),
                "headline_source": (
                    article[
                        "headline_reconstruction_source"
                    ]
                ),
                "candidate_count": len(
                    ranked
                ),
            }
        )

    # Recalculate category lists because corrected titles can change
    # category prediction.
    category_counts = {}

    for article in analysis.get(
        "results",
        []
    ):
        category = article[
            "category"
        ]

        category_counts[
            category
        ] = (
            category_counts.get(
                category,
                0
            )
            + 1
        )

    analysis[
        "category_counts"
    ] = (
        category_counts
    )

    analysis[
        "detected_categories"
    ] = sorted(
        category_counts
    )

    analysis[
        "needs_category_selection"
    ] = (
        len(
            category_counts
        )
        > 1
    )

    valid_headline_scores = [
        float(
            row[
                "headline_score"
            ]
        )
        for row in article_debug
        if row[
            "corrected_title"
        ]
    ]

    headline_quality = (
        float(
            np.mean(
                valid_headline_scores
            )
        )
        if valid_headline_scores
        else 0.0
    )

    original_layout_quality = float(
        analysis.get(
            "local_grouping",
            {}
        )
        .get(
            "selected_quality",
            {}
        )
        .get(
            "quality",
            0.0
        )
    )

    # Diagnostic score only — NOT accuracy.
    combined_internal_quality = (
        0.75
        * original_layout_quality
        + 0.25
        * headline_quality
    )

    analysis.setdefault(
        "local_grouping",
        {}
    )[
        "v61_headline_quality"
    ] = round(
        headline_quality,
        4
    )

    analysis[
        "local_grouping"
    ][
        "v61_combined_internal_quality"
    ] = round(
        combined_internal_quality,
        4
    )

    analysis[
        "pipeline_version"
    ] = (
        "V6.1_LOCAL_READING_GRAPH_FIXED_HEADLINES"
    )

    analysis.setdefault(
        "processing",
        {}
    )[
        "headline_reconstruction"
    ] = (
        "local_doclayout_title_bbox_plus_tesseract_reocr"
    )

    analysis[
        "processing"
    ][
        "figure_boxes_extend_article_region"
    ] = False

    analysis[
        "processing"
    ][
        "cloud_api_used"
    ] = False

    analysis[
        "processing"
    ][
        "openai_used"
    ] = False

    analysis[
        "v61_title_candidates"
    ] = candidates

    analysis[
        "v61_article_title_debug"
    ] = article_debug

    return analysis


def title_candidate_table_v61(
    analysis
):
    rows = []

    for candidate in analysis.get(
        "v61_title_candidates",
        []
    ):
        rows.append(
            {
                "text": candidate[
                    "text"
                ],
                "bbox": candidate[
                    "bbox"
                ],
                "doclayout_conf": round(
                    candidate[
                        "doclayout_confidence"
                    ],
                    4
                ),
                "ocr_source": candidate[
                    "ocr_source"
                ],
                "ocr_quality": candidate[
                    "ocr_quality"
                ],
                "typography": candidate[
                    "typography_strength"
                ],
                "structural_score": candidate[
                    "structural_score"
                ],
                "table_overlap": candidate[
                    "table_overlap"
                ],
                "byline": candidate[
                    "is_byline"
                ],
                "navigation": candidate[
                    "is_navigation"
                ],
                "accepted": candidate[
                    "accepted_as_candidate"
                ],
            }
        )

    return pd.DataFrame(
        rows
    )


def headline_fix_table_v61(
    analysis
):
    return pd.DataFrame(
        analysis.get(
            "v61_article_title_debug",
            []
        )
    )


def final_article_table_v61(
    analysis
):
    rows = []

    for item in analysis.get(
        "results",
        []
    ):
        rows.append(
            {
                "article": item[
                    "article_number"
                ],
                "title": item[
                    "title"
                ],
                "category": item[
                    "category"
                ],
                "category_confidence": item[
                    "confidence"
                ],
                "headline_score": item.get(
                    "headline_reconstruction_score",
                    0.0
                ),
                "headline_source": item.get(
                    "headline_reconstruction_source",
                    ""
                ),
                "ocr_quality": item[
                    "ocr_quality"
                ],
                "layout_confidence": item[
                    "layout_confidence"
                ],
                "partial_top": item[
                    "partial_top"
                ],
                "partial_bottom": item[
                    "partial_bottom"
                ],
                "words": len(
                    item[
                        "full_text"
                    ].split()
                ),
                "source": item[
                    "source"
                ],
            }
        )

    return pd.DataFrame(
        rows
    )


print("V6.1 headline/article fixes loaded.")
print("Cloud API functions added: NONE")


# ================================================================
# V6.2.1 COMPATIBILITY FIXES
# ================================================================
# Fixes missing helper names found in V6.2.
# ================================================================

def v621_box_union(boxes):
    valid = [
        tuple(map(int, box))
        for box in boxes
        if (
            box is not None
            and len(box) == 4
            and box[2] > box[0]
            and box[3] > box[1]
        )
    ]

    if not valid:
        return (0, 0, 1, 1)

    return (
        min(box[0] for box in valid),
        min(box[1] for box in valid),
        max(box[2] for box in valid),
        max(box[3] for box in valid),
    )


def v621_looks_like_byline(text):
    # Prefer V6.1 helper if present.
    if "looks_like_byline_v61" in globals():
        return looks_like_byline_v61(text)

    t = clean_title_text_v61(text).lower()

    return bool(
        re.search(
            r"^(?:by|written by|reported by|reporting by|staff reporter|staff writer|special correspondent|our correspondent|correspondent)\b",
            t
        )
    )


print("V6.2.1 compatibility helpers loaded.")


# ================================================================
# V6.2 PATCH
# Requires the V6.1 helper functions already defined above.
# ================================================================

from collections import Counter
import math
import re
import numpy as np
import pandas as pd


def v62_box_area(box):
    return float(
        max(0, box[2] - box[0])
        * max(0, box[3] - box[1])
    )


def v62_intersection(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    return float(
        (x2 - x1)
        * (y2 - y1)
    )


def v62_containment(inner, outer):
    return (
        v62_intersection(
            inner,
            outer
        )
        / max(
            1.0,
            v62_box_area(
                inner
            )
        )
    )


def v62_horizontal_relation(a, b):
    overlap = max(
        0.0,
        min(a[2], b[2])
        - max(a[0], b[0])
    )

    return (
        overlap
        / max(
            1.0,
            min(
                a[2] - a[0],
                b[2] - b[0]
            )
        )
    )


def v62_is_address_or_footer_title(text):
    t = clean_title_text_v61(
        text
    ).lower()

    if not t:
        return False

    address_terms = (
        "road",
        "street",
        "avenue",
        "mawatha",
        "lane",
        "drive",
        "junction",
    )

    has_address_term = any(
        term in t
        for term in address_terms
    )

    has_number = bool(
        re.search(
            r"\b\d+[a-z]?\b",
            t
        )
    )

    # Address-like title: short text + location/address term + number.
    return (
        len(t.split()) <= 12
        and has_number
        and (
            has_address_term
            or t.count(",") >= 2
        )
    )


def v62_collect_structure(
    image_path,
    document_type,
    config,
    doclayout_model,
):
    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        raise FileNotFoundError(
            image_path
        )

    prepared_color, prepared_gray, scale = prepare_page_image(
        image,
        config
    )

    ocr_lines = extract_ocr_lines(
        prepared_gray,
        document_type,
        config
    )

    # Key V6.1 fix:
    # locally re-run DocLayout because V6 did not store raw element boxes
    # in analysis["layout_debug"].
    elements = detect_doclayout_elements(
        doclayout_model,
        prepared_color,
        config
    )

    return {
        "prepared_color": prepared_color,
        "prepared_gray": prepared_gray,
        "ocr_lines": ocr_lines,
        "elements": elements,
        "page_shape": prepared_gray.shape[:2],
        "scale": scale,
    }


def v62_make_title_candidates(
    structure,
    body_height,
):
    ocr_lines = structure[
        "ocr_lines"
    ]

    prepared_gray = structure[
        "prepared_gray"
    ]

    elements = structure[
        "elements"
    ]

    table_boxes = [
        element.bbox
        for element in elements
        if normalize_layout_class_name(
            element.class_name
        )
        == "table"
    ]

    output = []

    for element in elements:
        if normalize_layout_class_name(
            element.class_name
        ) != "title":
            continue

        bbox = element.bbox

        line_text, title_lines = ocr_lines_for_title_bbox_v61(
            ocr_lines,
            bbox
        )

        crop_text, crop_options = ocr_title_crop_v61(
            prepared_gray,
            bbox
        )

        options = []

        if line_text:
            options.append(
                {
                    "text": line_text,
                    "quality": title_text_quality_v61(
                        line_text
                    ),
                    "source": "full_page_ocr_lines",
                }
            )

        options.extend(
            crop_options
        )

        if not options:
            continue

        options = sorted(
            options,
            key=lambda item: (
                item["quality"],
                len(item["text"])
            ),
            reverse=True
        )

        best = options[0]

        text = clean_title_text_v61(
            best["text"]
        )

        table_overlap = max(
            [
                v62_containment(
                    bbox,
                    table_box
                )
                for table_box in table_boxes
            ]
            or [0.0]
        )

        title_height = max(
            1,
            bbox[3] - bbox[1]
        )

        typography = min(
            1.0,
            title_height
            / max(
                1.0,
                4.0 * body_height
            )
        )

        quality = float(
            best["quality"]
        )

        structural_score = (
            0.38 * quality
            + 0.34 * min(
                1.0,
                float(
                    element.confidence
                )
            )
            + 0.28 * typography
        )

        rejected_reason = ""

        if table_overlap >= 0.25:
            rejected_reason = "overlaps_table"

        elif looks_like_byline_v61(
            text
        ):
            rejected_reason = "byline"

        elif looks_like_navigation_or_page_teaser_v61(
            text
        ):
            rejected_reason = "navigation_or_contents"

        elif v62_is_address_or_footer_title(
            text
        ):
            rejected_reason = "address_or_footer"

        elif quality < 0.34:
            rejected_reason = "weak_title"

        output.append(
            {
                "text": text,
                "bbox": bbox,
                "confidence": float(
                    element.confidence
                ),
                "quality": round(
                    quality,
                    4
                ),
                "typography": round(
                    typography,
                    4
                ),
                "structural_score": round(
                    structural_score,
                    4
                ),
                "ocr_source": best[
                    "source"
                ],
                "title_lines": title_lines,
                "accepted": (
                    rejected_reason == ""
                ),
                "rejected_reason": rejected_reason,
            }
        )

    return output


def v62_ad_signal_count(text):
    t = str(
        text or ""
    ).lower()

    signals = [
        bool(
            re.search(
                r"\bwww\.",
                t
            )
        ),
        bool(
            re.search(
                r"\b(?:hotline|call|scan|qr)\b",
                t
            )
        ),
        bool(
            re.search(
                r"\b\d{3,4}[\s-]?\d{3}[\s-]?\d{3,4}\b",
                t
            )
        ),
        t.count("%") >= 2,
        bool(
            re.search(
                r"\b(?:terms and conditions|conditions apply)\b",
                t
            )
        ),
        bool(
            re.search(
                r"\b(?:fixed deposits?|maturity|aer|monthly rate)\b",
                t
            )
        ),
        bool(
            re.search(
                r"\b(?:apply now|contact us|visit our website|offer)\b",
                t
            )
        ),
    ]

    return int(
        sum(signals)
    )


def v62_region_overlap_with_class(
    bbox,
    elements,
    class_name,
):
    values = []

    for element in elements:
        if normalize_layout_class_name(
            element.class_name
        ) != class_name:
            continue

        values.append(
            v62_intersection(
                bbox,
                element.bbox
            )
            / max(
                1.0,
                v62_box_area(
                    bbox
                )
            )
        )

    return max(
        values
        or [0.0]
    )


def v62_probable_ad(
    bbox,
    text,
    title,
    elements,
):
    """
    Text keywords alone NEVER reject an article.
    Structural evidence + multiple ad signals are required.
    """
    figure_overlap = v62_region_overlap_with_class(
        bbox,
        elements,
        "figure"
    )

    table_overlap = v62_region_overlap_with_class(
        bbox,
        elements,
        "table"
    )

    structural = (
        figure_overlap >= 0.28
        or table_overlap >= 0.20
    )

    signals = v62_ad_signal_count(
        text
    )

    weak_or_address_title = (
        title_text_quality_v61(
            title
        ) < 0.38
        or v62_is_address_or_footer_title(
            title
        )
    )

    digit_chars = sum(
        ch.isdigit()
        for ch in str(text)
    )

    visible_chars = max(
        1,
        sum(
            not ch.isspace()
            for ch in str(text)
        )
    )

    digit_ratio = (
        digit_chars
        / visible_chars
    )

    return bool(
        structural
        and (
            signals >= 2
            or (
                signals >= 1
                and digit_ratio >= 0.10
                and weak_or_address_title
            )
        )
    )


def v62_candidate_in_region(
    candidate,
    region_bbox,
):
    bbox = candidate[
        "bbox"
    ]

    cx = (
        bbox[0]
        + bbox[2]
    ) / 2.0

    cy = (
        bbox[1]
        + bbox[3]
    ) / 2.0

    center_inside = (
        region_bbox[0]
        <= cx
        <= region_bbox[2]
        and region_bbox[1]
        <= cy
        <= region_bbox[3]
    )

    return (
        center_inside
        or v62_containment(
            bbox,
            region_bbox
        ) >= 0.28
    )


def v62_intervening_title(
    candidate,
    line,
    candidates,
):
    title_box = candidate[
        "bbox"
    ]

    line_box = (
        line.x1,
        line.y1,
        line.x2,
        line.y2
    )

    for other in candidates:
        if other is candidate:
            continue

        other_box = other[
            "bbox"
        ]

        if not (
            title_box[3]
            < other_box[1]
            < line_box[1]
        ):
            continue

        if (
            v62_horizontal_relation(
                title_box,
                other_box
            ) >= 0.15
            or v62_horizontal_relation(
                line_box,
                other_box
            ) >= 0.15
        ):
            return True

    return False


def v62_line_to_title_score(
    line,
    candidate,
    candidates,
    body_height,
):
    title_box = candidate[
        "bbox"
    ]

    line_box = (
        line.x1,
        line.y1,
        line.x2,
        line.y2
    )

    # Body cannot belong to a title far below it.
    if (
        line.cy
        < title_box[1]
        - 0.30 * body_height
    ):
        return 0.0

    if v62_intervening_title(
        candidate,
        line,
        candidates
    ):
        return 0.01

    horizontal = v62_horizontal_relation(
        line_box,
        title_box
    )

    gap = max(
        0.0,
        line.y1
        - title_box[3]
    )

    distance = math.exp(
        -(
            gap
            / max(
                1.0,
                body_height
            )
        )
        / 8.0
    )

    title_cx = (
        title_box[0]
        + title_box[2]
    ) / 2.0

    line_cx = (
        line_box[0]
        + line_box[2]
    ) / 2.0

    mean_width = max(
        1.0,
        (
            title_box[2]
            - title_box[0]
            + line_box[2]
            - line_box[0]
        )
        / 2.0
    )

    center_alignment = max(
        0.0,
        1.0
        - abs(
            title_cx
            - line_cx
        )
        / mean_width
    )

    return (
        0.52 * horizontal
        + 0.34 * distance
        + 0.14 * center_alignment
    )


def v62_make_region_from_candidate(
    candidate,
    body_lines,
    base_article,
    document_type,
    body_height,
    page_shape,
    config,
):
    title_lines = list(
        candidate.get(
            "title_lines",
            []
        )
    )

    line_map = {
        line.uid: line
        for line in (
            title_lines
            + list(
                body_lines
            )
        )
    }

    lines = sorted(
        line_map.values(),
        key=lambda line: (
            line.y1,
            line.x1
        )
    )

    if not lines:
        return None

    anchor = HeadlineAnchor(
        lines=title_lines,
        title=candidate[
            "text"
        ],
        bbox=candidate[
            "bbox"
        ],
        score=(
            5.0
            + 3.0
            * candidate[
                "confidence"
            ]
        ),
    )

    text = article_lines_to_text(
        lines,
        anchor,
        document_type,
        body_height
    )

    text = normalize_article_continuation_artifacts_v61(
        text
    )

    if (
        not text
        or len(
            text.split()
        ) < 18
    ):
        return None

    article_bbox = v621_box_union(
        [
            candidate[
                "bbox"
            ],
            bbox_from_lines(
                lines
            ),
        ]
    )

    edge_flags = visible_capture_edge_flags(
        article_bbox,
        page_shape,
        config
    )

    layout_confidence = min(
        1.0,
        0.58
        * float(
            base_article.get(
                "layout_confidence",
                0.5
            )
        )
        + 0.42
        * float(
            candidate[
                "structural_score"
            ]
        )
    )

    return {
        "anchor": anchor,
        "lines": lines,
        "bbox": article_bbox,
        "text": text,
        "layout_confidence": (
            layout_confidence
        ),
        "warnings": [],
        "source": (
            "local_v62_split_reconstruction"
        ),
        "headline_visible": True,
        "partial_top": edge_flags[
            "partial_top"
        ],
        "partial_bottom": edge_flags[
            "partial_bottom"
        ],
        "headline_reconstruction_source": candidate[
            "ocr_source"
        ],
        "headline_reconstruction_score": candidate[
            "structural_score"
        ],
    }


def v62_process_base_article(
    base_article,
    accepted_candidates,
    structure,
    body_height,
    document_type,
    config,
):
    region_bbox = tuple(
        base_article[
            "bbox"
        ]
    )

    candidates = [
        candidate
        for candidate in accepted_candidates
        if v62_candidate_in_region(
            candidate,
            region_bbox
        )
    ]

    candidates = sorted(
        candidates,
        key=lambda c: (
            c[
                "bbox"
            ][1],
            c[
                "bbox"
            ][0],
        )
    )

    elements = structure[
        "elements"
    ]

    # No reliable title inside region: keep continuation unless it is an ad.
    if not candidates:
        text = normalize_article_continuation_artifacts_v61(
            base_article.get(
                "full_text",
                ""
            )
        )

        if v62_probable_ad(
            region_bbox,
            text,
            base_article.get(
                "title",
                ""
            ),
            elements
        ):
            return []

        retained = dict(
            base_article
        )

        retained[
            "full_text"
        ] = text

        retained[
            "source"
        ] = (
            retained.get(
                "source",
                "local_v6"
            )
            + "_v62_retained"
        )

        return [
            retained
        ]

    region_lines = [
        line
        for line in structure[
            "ocr_lines"
        ]
        if (
            region_bbox[0]
            <= line.cx
            <= region_bbox[2]
            and region_bbox[1]
            <= line.cy
            <= region_bbox[3]
        )
    ]

    title_ids = {
        line.uid
        for candidate in candidates
        for line in candidate.get(
            "title_lines",
            []
        )
    }

    assigned = {
        index: []
        for index in range(
            len(candidates)
        )
    }

    for line in region_lines:
        if line.uid in title_ids:
            continue

        if v621_looks_like_byline(
            line.text
        ):
            continue

        scores = [
            (
                v62_line_to_title_score(
                    line,
                    candidate,
                    candidates,
                    body_height
                ),
                index,
            )
            for index, candidate in enumerate(
                candidates
            )
        ]

        scores.sort(
            reverse=True
        )

        if not scores:
            continue

        best_score, best_index = scores[0]

        second_score = (
            scores[1][0]
            if len(scores) > 1
            else 0.0
        )

        if (
            best_score >= 0.22
            and (
                best_score
                - second_score
                >= 0.035
                or best_score >= 0.58
            )
        ):
            assigned[
                best_index
            ].append(
                line
            )

    output = []

    for index, candidate in enumerate(
        candidates
    ):
        built = v62_make_region_from_candidate(
            candidate=candidate,
            body_lines=assigned[
                index
            ],
            base_article=base_article,
            document_type=document_type,
            body_height=body_height,
            page_shape=structure[
                "page_shape"
            ],
            config=config,
        )

        if built is None:
            continue

        if v62_probable_ad(
            built[
                "bbox"
            ],
            built[
                "text"
            ],
            candidate[
                "text"
            ],
            elements
        ):
            continue

        output.append(
            built
        )

    # Preserve original only when it is not a likely ad.
    if not output:
        text = normalize_article_continuation_artifacts_v61(
            base_article.get(
                "full_text",
                ""
            )
        )

        if not v62_probable_ad(
            region_bbox,
            text,
            base_article.get(
                "title",
                ""
            ),
            elements
        ):
            retained = dict(
                base_article
            )

            retained[
                "full_text"
            ] = text

            output.append(
                retained
            )

    return output


def postprocess_analysis_v62(
    analysis,
    image_path,
    category_model,
    config,
    doclayout_model,
):
    document_type = analysis.get(
        "document_type",
        "Newspaper"
    )

    structure = v62_collect_structure(
        image_path=image_path,
        document_type=document_type,
        config=config,
        doclayout_model=doclayout_model,
    )

    body_height = float(
        analysis.get(
            "layout_debug",
            {}
        ).get(
            "body_height",
            20.0
        )
    )

    all_candidates = v62_make_title_candidates(
        structure,
        body_height
    )

    accepted_candidates = [
        candidate
        for candidate in all_candidates
        if candidate[
            "accepted"
        ]
    ]

    rebuilt_regions = []

    for base_article in analysis.get(
        "results",
        []
    ):
        rebuilt_regions.extend(
            v62_process_base_article(
                base_article=base_article,
                accepted_candidates=accepted_candidates,
                structure=structure,
                body_height=body_height,
                document_type=document_type,
                config=config,
            )
        )

    # Deduplicate by title + box overlap.
    deduped = []

    for region in sorted(
        rebuilt_regions,
        key=lambda r: (
            r[
                "bbox"
            ][1],
            r[
                "bbox"
            ][0],
        )
    ):
        title = (
            region[
                "anchor"
            ].title
            if region.get(
                "anchor"
            )
            else region.get(
                "title",
                ""
            )
        )

        duplicate_index = None

        for index, existing in enumerate(
            deduped
        ):
            existing_title = (
                existing[
                    "anchor"
                ].title
                if existing.get(
                    "anchor"
                )
                else existing.get(
                    "title",
                    ""
                )
            )

            if (
                title_similarity(
                    title,
                    existing_title
                )
                >= 0.78
                and bbox_iou(
                    region[
                        "bbox"
                    ],
                    existing[
                        "bbox"
                    ]
                )
                >= 0.28
            ):
                duplicate_index = index
                break

        if duplicate_index is None:
            deduped.append(
                region
            )

        else:
            current_words = len(
                region.get(
                    "text",
                    region.get(
                        "full_text",
                        ""
                    )
                ).split()
            )

            existing_words = len(
                deduped[
                    duplicate_index
                ].get(
                    "text",
                    deduped[
                        duplicate_index
                    ].get(
                        "full_text",
                        ""
                    )
                ).split()
            )

            if current_words > existing_words:
                deduped[
                    duplicate_index
                ] = region

    results = []

    for region in deduped:
        anchor = region.get(
            "anchor"
        )

        title = (
            anchor.title.strip()
            if (
                anchor is not None
                and anchor.title
            )
            else region.get(
                "title",
                "Visible article continuation"
            )
        )

        text = region.get(
            "text",
            region.get(
                "full_text",
                ""
            )
        )

        text = normalize_article_continuation_artifacts_v61(
            text
        )

        if not text:
            continue

        prediction = predict_category(
            model=category_model,
            text=text,
            document_type=document_type,
            title=title,
            config=config,
        )

        bbox = tuple(
            region[
                "bbox"
            ]
        )

        results.append(
            {
                "article_number": 0,
                "title": title,
                "raw_visible_title": title,
                "headline_visible": bool(
                    title
                ),
                "bbox": bbox,
                "normalized_bbox": normalize_bbox(
                    bbox,
                    structure[
                        "page_shape"
                    ],
                ),
                "full_text": text,
                "category": prediction[
                    "category"
                ],
                "confidence": prediction[
                    "confidence"
                ],
                "probability_margin": prediction[
                    "margin"
                ],
                "top_predictions": prediction[
                    "top_predictions"
                ],
                "category_decision": prediction.get(
                    "decision",
                    ""
                ),
                "ocr_quality": ocr_quality_score(
                    text
                ),
                "layout_confidence": round(
                    float(
                        region.get(
                            "layout_confidence",
                            0.0
                        )
                    ),
                    4
                ),
                "partial_top": bool(
                    region.get(
                        "partial_top",
                        False
                    )
                ),
                "partial_bottom": bool(
                    region.get(
                        "partial_bottom",
                        False
                    )
                ),
                "source": region.get(
                    "source",
                    "local_v62"
                ),
                "headline_reconstruction_source": region.get(
                    "headline_reconstruction_source",
                    "original_v6"
                ),
                "headline_reconstruction_score": round(
                    float(
                        region.get(
                            "headline_reconstruction_score",
                            0.0
                        )
                    ),
                    4
                ),
                "warnings": list(
                    region.get(
                        "warnings",
                        []
                    )
                ),
            }
        )

    # Reading order for final articles.
    results = sorted(
        results,
        key=lambda item: (
            item[
                "bbox"
            ][1],
            item[
                "bbox"
            ][0],
        )
    )

    for index, item in enumerate(
        results,
        start=1
    ):
        item[
            "article_number"
        ] = index

    category_counts = Counter(
        item[
            "category"
        ]
        for item in results
    )

    analysis[
        "results"
    ] = results

    analysis[
        "detected_categories"
    ] = sorted(
        category_counts
    )

    analysis[
        "category_counts"
    ] = dict(
        category_counts
    )

    analysis[
        "needs_category_selection"
    ] = (
        len(
            category_counts
        )
        > 1
    )

    headline_scores = [
        item[
            "headline_reconstruction_score"
        ]
        for item in results
        if item[
            "headline_visible"
        ]
    ]

    analysis[
        "pipeline_version"
    ] = (
        "V6.2_LOCAL_READING_GRAPH_SPLIT_AD_FILTER"
    )

    analysis.setdefault(
        "local_grouping",
        {}
    )[
        "v62_headline_quality"
    ] = round(
        float(
            np.mean(
                headline_scores
            )
        )
        if headline_scores
        else 0.0,
        4
    )

    analysis[
        "local_grouping"
    ][
        "v62_final_article_count"
    ] = len(
        results
    )

    analysis[
        "v62_title_candidates"
    ] = [
        {
            key: value
            for key, value in candidate.items()
            if key != "title_lines"
        }
        for candidate in all_candidates
    ]

    analysis[
        "v62_doclayout_classes"
    ] = dict(
        Counter(
            normalize_layout_class_name(
                element.class_name
            )
            for element in structure[
                "elements"
            ]
        )
    )

    analysis.setdefault(
        "processing",
        {}
    )[
        "headline_reconstruction"
    ] = (
        "local_doclayout_rerun_plus_tesseract"
    )

    analysis[
        "processing"
    ][
        "article_split"
    ] = (
        "local_competing_headline_body_assignment"
    )

    analysis[
        "processing"
    ][
        "advertisement_filter"
    ] = (
        "local_structural_plus_text_evidence"
    )

    analysis[
        "processing"
    ][
        "cloud_api_used"
    ] = False

    analysis[
        "processing"
    ][
        "openai_used"
    ] = False

    return analysis


def v62_title_candidate_table(
    analysis
):
    return pd.DataFrame(
        analysis.get(
            "v62_title_candidates",
            []
        )
    )


def v62_final_article_table(
    analysis
):
    return pd.DataFrame(
        [
            {
                "article": item[
                    "article_number"
                ],
                "title": item[
                    "title"
                ],
                "category": item[
                    "category"
                ],
                "category_conf": item[
                    "confidence"
                ],
                "headline_score": item.get(
                    "headline_reconstruction_score",
                    0.0
                ),
                "ocr_quality": item[
                    "ocr_quality"
                ],
                "layout_conf": item[
                    "layout_confidence"
                ],
                "partial_top": item[
                    "partial_top"
                ],
                "partial_bottom": item[
                    "partial_bottom"
                ],
                "words": len(
                    item[
                        "full_text"
                    ].split()
                ),
                "source": item[
                    "source"
                ],
            }
            for item in analysis.get(
                "results",
                []
            )
        ]
    )


print("V6.2 local fixes loaded.")
print("Cloud/API article detection: NONE")




# =====================================================================
# 25. V7 — LOCAL-FIRST PIPELINE WITH OPENAI APPLICATION FALLBACK
# =====================================================================
#
# IMPORTANT ACADEMIC SEPARATION
# -----------------------------
# research_mode=True
#     -> ALL OpenAI calls are disabled.
#     -> Use this for research evaluation / local accuracy testing.
#
# research_mode=False
#     -> Local research pipeline still runs FIRST.
#     -> OpenAI is used ONLY when a configured local quality gate fails.
#
# No API key is stored in this file.
# The OpenAI SDK reads OPENAI_API_KEY from the environment.
#
# LOCAL PRIMARY
# -------------
# Newspaper/Magazine:
#   V6 reading-order graph
#       -> V6.2 article splitting/title recovery/ad filtering
#
# Novel/other single content:
#   existing local OCR/reading-order branch
#
# FALLBACKS
# ---------
# 1. Article/content fallback:
#      OpenAI image input -> visible editorial articles/text
#
# 2. Category fallback:
#      local category classifier FIRST
#      -> OpenAI text classification only if local confidence/margin is low
#
# 3. Summary fallback:
#      local Llama FIRST when the article passes the quality gate
#      -> OpenAI text summary only if local summarization is unavailable/
#         unsuitable
#      -> extractive summary as final no-cloud fallback
#
# Voice selection remains local Vosk.
# =====================================================================

import os
import base64
import mimetypes
import json as _json_v7


@dataclass
class ApplicationFallbackConfig(AdaptiveReadingGraphConfig):
    # ---------------------------------------------------------
    # Academic switch
    # ---------------------------------------------------------
    research_mode: bool = False

    # ---------------------------------------------------------
    # OpenAI fallback
    # ---------------------------------------------------------
    enable_openai_fallback: bool = True

    openai_model: str = field(
        default_factory=lambda: os.getenv(
            "OPENAI_FALLBACK_MODEL",
            "gpt-5-mini",
        )
    )

    openai_image_detail: str = "high"

    # Useful only for deliberate API testing.
    # Leave False in normal operation.
    force_openai_article_fallback: bool = False

    # Run OpenAI article detection first (Newspaper/Magazine) so the
    # application article count comes from AI, then keep local results
    # for comparison. Local is still the research component.
    prefer_openai_article_detection_first: bool = True

    # ---------------------------------------------------------
    # Local article/content quality gate
    # ---------------------------------------------------------
    local_min_selected_quality: float = 0.44
    local_min_average_ocr_quality: float = 0.28
    local_hard_min_average_ocr_quality: float = 0.20
    local_min_total_visible_words: int = 30
    local_max_short_article_ratio: float = 0.65
    local_short_article_word_threshold: int = 18

    # ---------------------------------------------------------
    # Category fallback
    # ---------------------------------------------------------
    enable_openai_category_fallback: bool = True

    category_fallback_probability_threshold: float = 0.34
    category_fallback_margin_threshold: float = 0.055

    fallback_categories: tuple[str, ...] = (
        "General Text",
        "Mathematics",
        "Politics",
        "Science",
        "Social",
        "Sports",
        "Story",
    )

    # ---------------------------------------------------------
    # Summary fallback
    # ---------------------------------------------------------
    enable_openai_summary_fallback: bool = True
    openai_summary_max_words: int = 90

    # ---------------------------------------------------------
    # API limits
    # ---------------------------------------------------------
    openai_max_output_tokens_articles: int = 12000
    openai_max_output_tokens_text: int = 1000


# ---------------------------------------------------------------------
# 25.1 OPENAI CLIENT — LAZY, ENVIRONMENT KEY ONLY
# ---------------------------------------------------------------------


def openai_fallback_available_v7(
    config: ApplicationFallbackConfig,
) -> tuple[bool, str]:
    if config.research_mode:
        return (
            False,
            "research_mode=True",
        )

    if not config.enable_openai_fallback:
        return (
            False,
            "OpenAI fallback disabled in config",
        )

    if not os.getenv(
        "OPENAI_API_KEY"
    ):
        return (
            False,
            "OPENAI_API_KEY environment variable is not set",
        )

    try:
        import openai  # noqa: F401
    except Exception:
        return (
            False,
            "openai Python package is not installed",
        )

    return (
        True,
        "available",
    )


def get_openai_client_v7(
    config: ApplicationFallbackConfig,
):
    available, reason = openai_fallback_available_v7(
        config
    )

    if not available:
        raise RuntimeError(
            f"OpenAI fallback is unavailable: {reason}"
        )

    from openai import OpenAI

    # The SDK automatically reads OPENAI_API_KEY.
    return OpenAI()


# ---------------------------------------------------------------------
# 25.2 IMAGE -> DATA URL
# ---------------------------------------------------------------------


def image_to_data_url_v7(
    image_path: str,
) -> str:
    mime_type, _ = mimetypes.guess_type(
        str(
            image_path
        )
    )

    if mime_type not in {
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/gif",
    }:
        mime_type = "image/jpeg"

    with open(
        image_path,
        "rb",
    ) as file:
        encoded = base64.b64encode(
            file.read()
        ).decode(
            "ascii"
        )

    return (
        f"data:{mime_type};base64,{encoded}"
    )


# ---------------------------------------------------------------------
# 25.3 STRUCTURED RESPONSES HELPER
# ---------------------------------------------------------------------


def openai_structured_response_v7(
    *,
    client,
    model: str,
    input_items: list[dict[str, Any]],
    schema_name: str,
    schema: dict[str, Any],
    max_output_tokens: int,
) -> dict[str, Any]:
    response = client.responses.create(
        model=model,
        input=input_items,
        text={
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "strict": True,
                "schema": schema,
            }
        },
        max_output_tokens=max_output_tokens,
    )

    raw = response.output_text

    if not raw:
        raise RuntimeError(
            "OpenAI returned an empty structured response."
        )

    return _json_v7.loads(
        raw
    )


# ---------------------------------------------------------------------
# 25.4 LOCAL QUALITY GATE
# ---------------------------------------------------------------------


def evaluate_local_analysis_v7(
    analysis: dict[str, Any],
    config: ApplicationFallbackConfig,
) -> dict[str, Any]:
    results = analysis.get(
        "results",
        [],
    )

    reasons = []
    warnings = []

    if not results:
        reasons.append(
            "no_local_results"
        )

        return {
            "reliable": False,
            "reasons": reasons,
            "warnings": warnings,
            "metrics": {
                "article_count": 0,
            },
        }

    word_counts = [
        len(
            str(
                item.get(
                    "full_text",
                    "",
                )
            ).split()
        )
        for item in results
    ]

    total_words = sum(
        word_counts
    )

    ocr_qualities = [
        float(
            item.get(
                "ocr_quality",
                0.0,
            )
            or 0.0
        )
        for item in results
    ]

    average_ocr_quality = (
        float(
            np.mean(
                ocr_qualities
            )
        )
        if ocr_qualities
        else 0.0
    )

    short_count = sum(
        count
        < config.local_short_article_word_threshold
        for count in word_counts
    )

    short_ratio = (
        short_count
        / max(
            1,
            len(
                word_counts
            ),
        )
    )

    local_grouping = analysis.get(
        "local_grouping",
        {},
    )

    selected_quality = float(
        local_grouping.get(
            "selected_quality",
            {},
        ).get(
            "quality",
            0.0,
        )
        or 0.0
    )

    selected_method = str(
        local_grouping.get(
            "selected_method",
            ""
        )
    )

    # ----------------------------------------------------------
    # Hard failure conditions
    # ----------------------------------------------------------

    if (
        total_words
        < config.local_min_total_visible_words
    ):
        reasons.append(
            "too_little_local_text"
        )

    if (
        average_ocr_quality
        < config.local_hard_min_average_ocr_quality
    ):
        reasons.append(
            "very_low_local_text_quality"
        )

    if (
        "full_visible_capture_ocr_fallback"
        in selected_method
    ):
        reasons.append(
            "local_article_grouping_reached_final_ocr_fallback"
        )

    # ----------------------------------------------------------
    # Combined weak-signal conditions
    # ----------------------------------------------------------

    if (
        selected_quality
        and selected_quality
        < config.local_min_selected_quality
        and average_ocr_quality
        < config.local_min_average_ocr_quality
    ):
        reasons.append(
            "low_layout_and_ocr_quality"
        )

    if (
        len(
            results
        )
        >= 2
        and short_ratio
        > config.local_max_short_article_ratio
        and average_ocr_quality
        < config.local_min_average_ocr_quality
    ):
        reasons.append(
            "most_local_articles_are_too_short"
        )

    # Headline quality alone does NOT cause failure because a bottom-half
    # capture can legitimately begin with article continuation text.
    headline_quality = float(
        local_grouping.get(
            "v62_headline_quality",
            local_grouping.get(
                "v61_headline_quality",
                0.0,
            ),
        )
        or 0.0
    )

    if headline_quality < 0.20:
        warnings.append(
            "Headline reconstruction confidence is low, but this alone "
            "does not trigger API fallback because partial captures may "
            "have no visible headline."
        )

    reliable = (
        len(
            reasons
        )
        == 0
    )

    if config.force_openai_article_fallback:
        reliable = False
        reasons.append(
            "forced_openai_fallback_for_testing"
        )

    return {
        "reliable": reliable,
        "reasons": reasons,
        "warnings": warnings,
        "metrics": {
            "article_count": len(
                results
            ),
            "total_words": int(
                total_words
            ),
            "average_ocr_quality": round(
                average_ocr_quality,
                4,
            ),
            "short_article_ratio": round(
                short_ratio,
                4,
            ),
            "selected_local_quality": round(
                selected_quality,
                4,
            ),
            "selected_local_method": (
                selected_method
            ),
            "headline_quality": round(
                headline_quality,
                4,
            ),
        },
    }


# ---------------------------------------------------------------------
# 25.5 OPENAI IMAGE FALLBACK — VISIBLE ARTICLES
# ---------------------------------------------------------------------


def openai_visible_articles_v7(
    image_path: str,
    document_type: str,
    config: ApplicationFallbackConfig,
) -> list[dict[str, Any]]:
    client = get_openai_client_v7(
        config
    )

    image_data = image_to_data_url_v7(
        image_path
    )

    schema = {
        "type": "object",
        "properties": {
            "articles": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string"
                        },
                        "text": {
                            "type": "string"
                        },
                        "partial_top": {
                            "type": "boolean"
                        },
                        "partial_bottom": {
                            "type": "boolean"
                        },
                        "bbox_norm": {
                            "type": "object",
                            "properties": {
                                "x1": {
                                    "type": "integer"
                                },
                                "y1": {
                                    "type": "integer"
                                },
                                "x2": {
                                    "type": "integer"
                                },
                                "y2": {
                                    "type": "integer"
                                },
                            },
                            "required": [
                                "x1",
                                "y1",
                                "x2",
                                "y2",
                            ],
                            "additionalProperties": False,
                        },
                    },
                    "required": [
                        "title",
                        "text",
                        "partial_top",
                        "partial_bottom",
                        "bbox_norm",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": [
            "articles"
        ],
        "additionalProperties": False,
    }

    prompt = f"""
You are the EMERGENCY APPLICATION FALLBACK for a wearable reading
assistant.

Document type: {document_type}

Analyze ONLY the content visibly present in this camera image.

For Newspaper or Magazine:
- First count every distinct visible EDITORIAL article on this page.
- Return each of those articles separately. Do not drop a story.
- Do not merge neighboring stories or columns into one article.
- Prefer more smaller article boxes over one large mixed box.
- Do not return advertisements, rate tables, product promotions,
  navigation boxes, page headers, section labels, page numbers,
  QR-code promotions, or contact/address blocks as articles.
- A short photo-story/caption can be an article when it has a clear
  editorial headline and meaningful editorial text.
- If two different headlines have separate body text, keep them as
  separate articles.
- Do not merge neighboring stories.
- If the article started before the image, title may be empty and
  partial_top must be true.
- If it continues below the image, partial_bottom must be true.

For Novel/Story or another single reading document:
- Return the visible reading content as one article/section.

Text rules:
- Transcribe/reconstruct only what is visible.
- Keep names, numbers, dates, places, scores and facts.
- Do NOT invent text hidden outside the image.
- Do NOT invent missing beginning/endings.
- Remove obvious advertisements and unrelated surrounding text.
- If the headline is visible, return its complete headline.
- If no headline is visible, return an empty title string.

bbox_norm:
- Approximate the visible editorial region in normalized coordinates
  from 0 to 1000, where (0,0) is top-left and (1000,1000) is bottom-right.
""".strip()

    data = openai_structured_response_v7(
        client=client,
        model=config.openai_model,
        input_items=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": prompt,
                    },
                    {
                        "type": "input_image",
                        "image_url": image_data,
                        "detail": config.openai_image_detail,
                    },
                ],
            }
        ],
        schema_name=(
            "visible_editorial_articles"
        ),
        schema=schema,
        max_output_tokens=(
            config.openai_max_output_tokens_articles
        ),
    )

    articles = data.get(
        "articles",
        [],
    )

    cleaned = []

    for article in articles:
        title = remove_newspaper_artifacts(
            str(
                article.get(
                    "title",
                    ""
                )
            )
        )

        text = remove_newspaper_artifacts(
            str(
                article.get(
                    "text",
                    ""
                )
            )
        )

        if len(
            text.split()
        ) < 8:
            continue

        bbox = article.get(
            "bbox_norm",
            {},
        )

        def clamp(
            value,
        ):
            try:
                return max(
                    0,
                    min(
                        1000,
                        int(
                            value
                        ),
                    ),
                )
            except Exception:
                return 0

        x1 = clamp(
            bbox.get(
                "x1",
                0,
            )
        )
        y1 = clamp(
            bbox.get(
                "y1",
                0,
            )
        )
        x2 = clamp(
            bbox.get(
                "x2",
                1000,
            )
        )
        y2 = clamp(
            bbox.get(
                "y2",
                1000,
            )
        )

        if x2 <= x1:
            x1, x2 = (
                0,
                1000,
            )

        if y2 <= y1:
            y1, y2 = (
                0,
                1000,
            )

        cleaned.append(
            {
                "title": title,
                "text": text,
                "partial_top": bool(
                    article.get(
                        "partial_top",
                        False,
                    )
                ),
                "partial_bottom": bool(
                    article.get(
                        "partial_bottom",
                        False,
                    )
                ),
                "bbox_norm": (
                    x1,
                    y1,
                    x2,
                    y2,
                ),
            }
        )

    return cleaned


# ---------------------------------------------------------------------
# 25.6 OPENAI CATEGORY FALLBACK
# ---------------------------------------------------------------------


def openai_category_v7(
    title: str,
    text: str,
    config: ApplicationFallbackConfig,
) -> str:
    client = get_openai_client_v7(
        config
    )

    allowed = list(
        config.fallback_categories
    )

    schema = {
        "type": "object",
        "properties": {
            "category": {
                "type": "string"
            }
        },
        "required": [
            "category"
        ],
        "additionalProperties": False,
    }

    sample = " ".join(
        str(
            text
        ).split()[
            :700
        ]
    )

    prompt = f"""
Choose exactly ONE category for this reading content.

Allowed categories:
{", ".join(allowed)}

Guidance:
- Sports: sports matches, athletes, teams, scores, tournaments.
- Politics: government, parliament, political institutions, elections,
  public policy, state officials.
- Social: community, labour, society, rights, social issues.
- Science: science, medicine, environment, scientific/technical topics.
- Mathematics: mathematical teaching/calculation/formulas.
- Story: novels, fiction, narrative stories.
- General Text: content that does not reliably fit another allowed class.

Do not create a new category.

Title:
{title}

Text:
{sample}
""".strip()

    data = openai_structured_response_v7(
        client=client,
        model=config.openai_model,
        input_items=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": prompt,
                    }
                ],
            }
        ],
        schema_name="category_choice",
        schema=schema,
        max_output_tokens=(
            config.openai_max_output_tokens_text
        ),
    )

    category = str(
        data.get(
            "category",
            ""
        )
    ).strip()

    if category not in allowed:
        raise RuntimeError(
            f"OpenAI returned unsupported category: {category}"
        )

    return category


def maybe_apply_category_fallback_v7(
    *,
    item: dict[str, Any],
    document_type: str,
    local_prediction: dict[str, Any],
    config: ApplicationFallbackConfig,
) -> dict[str, Any]:
    # Novel/Story remains deterministic.
    if normalize_document_type(
        document_type
    ) == "Novel/Story":
        return {
            **local_prediction,
            "category": "Story",
            "category_source": (
                "local_document_type_rule"
            ),
            "openai_category_used": False,
        }

    confidence = float(
        local_prediction.get(
            "confidence",
            0.0,
        )
        or 0.0
    )

    margin = float(
        local_prediction.get(
            "margin",
            0.0,
        )
        or 0.0
    )

    low_confidence = (
        confidence
        < config.category_fallback_probability_threshold
        or margin
        < config.category_fallback_margin_threshold
    )

    available, _ = openai_fallback_available_v7(
        config
    )

    if (
        not low_confidence
        or not config.enable_openai_category_fallback
        or not available
    ):
        return {
            **local_prediction,
            "category_source": (
                "local_category_model"
            ),
            "openai_category_used": False,
        }

    try:
        category = openai_category_v7(
            title=item.get(
                "title",
                ""
            ),
            text=item.get(
                "full_text",
                item.get(
                    "text",
                    "",
                ),
            ),
            config=config,
        )

        return {
            **local_prediction,
            "category": category,
            "category_source": (
                "openai_category_fallback"
            ),
            "openai_category_used": True,
        }

    except Exception as error:
        result = {
            **local_prediction,
            "category_source": (
                "local_category_model_after_openai_failure"
            ),
            "openai_category_used": False,
            "category_fallback_error": str(
                error
            ),
        }

        return result


# ---------------------------------------------------------------------
# 25.7 CONVERT OPENAI ARTICLES TO NORMAL PIPELINE RESULTS
# ---------------------------------------------------------------------


def build_openai_analysis_results_v7(
    *,
    articles: Sequence[dict[str, Any]],
    document_type: str,
    category_model,
    config: ApplicationFallbackConfig,
) -> list[dict[str, Any]]:
    results = []

    for index, article in enumerate(
        articles,
        start=1,
    ):
        title = article.get(
            "title",
            "",
        ).strip()

        display_title = (
            title
            if title
            else "Visible article continuation"
        )

        text = remove_newspaper_artifacts(
            article.get(
                "text",
                "",
            )
        )

        if not text:
            continue

        local_prediction = predict_category(
            model=category_model,
            text=text,
            document_type=document_type,
            title=title,
            config=config,
        )

        item_for_category = {
            "title": title,
            "full_text": text,
        }

        prediction = maybe_apply_category_fallback_v7(
            item=item_for_category,
            document_type=document_type,
            local_prediction=local_prediction,
            config=config,
        )

        results.append(
            {
                "article_number": index,
                "title": display_title,
                "raw_visible_title": title,
                "headline_visible": bool(
                    title
                ),
                "bbox": (
                    article.get(
                        "bbox_norm",
                        (
                            0,
                            0,
                            1000,
                            1000,
                        ),
                    )
                ),
                "normalized_bbox": (
                    article.get(
                        "bbox_norm",
                        (
                            0,
                            0,
                            1000,
                            1000,
                        ),
                    )
                ),
                "full_text": text,
                "category": prediction[
                    "category"
                ],
                "confidence": prediction.get(
                    "confidence",
                    0.0,
                ),
                "probability_margin": prediction.get(
                    "margin",
                    0.0,
                ),
                "top_predictions": prediction.get(
                    "top_predictions",
                    [],
                ),
                "category_decision": prediction.get(
                    "decision",
                    "",
                ),
                "category_source": prediction.get(
                    "category_source",
                    "local_category_model",
                ),
                "openai_category_used": prediction.get(
                    "openai_category_used",
                    False,
                ),
                # This is a text-quality heuristic, not OCR accuracy.
                "ocr_quality": ocr_quality_score(
                    text
                ),
                # Do not pretend API output is local layout confidence.
                "layout_confidence": 0.0,
                "partial_top": bool(
                    article.get(
                        "partial_top",
                        False,
                    )
                ),
                "partial_bottom": bool(
                    article.get(
                        "partial_bottom",
                        False,
                    )
                ),
                "source": (
                    "openai_visible_article_fallback"
                ),
                "fallback_used": True,
                "warnings": [
                    (
                        "The local article/content quality gate failed. "
                        "This result was generated by the application-level "
                        "OpenAI fallback and must not be included in local "
                        "research accuracy measurements."
                    )
                ],
            }
        )

    return results


# ---------------------------------------------------------------------
# 25.8 MAIN ANALYZER — LOCAL FIRST, OPENAI ONLY IF NEEDED
# ---------------------------------------------------------------------


def _adopt_ai_article_results_v7(
    *,
    analysis: dict[str, Any],
    ai_results: list[dict[str, Any]],
    local_results: list[dict[str, Any]],
    local_quality: dict[str, Any],
    fallback_record: dict[str, Any],
    pipeline_version: str,
    article_source: str,
) -> dict[str, Any]:
    category_counts = Counter(
        item.get("category", "General Text") for item in ai_results
    )

    analysis["results"] = ai_results
    analysis["local_results"] = local_results
    analysis["ai_results"] = ai_results
    analysis["detected_categories"] = sorted(category_counts)
    analysis["categories"] = sorted(category_counts)
    analysis["category_counts"] = dict(category_counts)
    analysis["needs_category_selection"] = len(category_counts) > 1
    analysis["mode"] = (
        "Multi Category" if len(category_counts) > 1 else "Single Category"
    )
    analysis["pipeline_version"] = pipeline_version

    fallback_record["article_fallback_used"] = True
    fallback_record["api_article_count"] = len(ai_results)
    fallback_record["local_primary_snapshot"] = {
        "result_count": len(local_results),
        "quality": local_quality,
        "detected_categories": sorted(
            {
                item.get("category", "General Text")
                for item in local_results
            }
        ),
        "results": local_results,
    }
    analysis["fallback"] = fallback_record
    analysis.setdefault("processing", {})
    analysis["processing"]["openai_used"] = True
    analysis["processing"]["article_source"] = article_source
    analysis["processing"]["ai_article_count"] = len(ai_results)
    analysis["processing"]["local_article_count"] = len(local_results)

    analysis.setdefault("warnings", []).append(
        "AI article detection ran first and is used for the application "
        "article count. Local detections are stored separately for comparison."
    )
    return analysis


def analyze_content_v7(
    *,
    image_path: str,
    document_type: str,
    category_model,
    config: ApplicationFallbackConfig,
    doclayout_model=None,
) -> dict[str, Any]:
    normalized_type = normalize_document_type(
        document_type
    )

    fallback_available, fallback_reason = (
        openai_fallback_available_v7(config)
    )

    run_ai_first = (
        bool(getattr(config, "prefer_openai_article_detection_first", True))
        and normalized_type in MULTI_CATEGORY_TYPES
        and fallback_available
    )

    ai_results_first: list[dict[str, Any]] = []
    ai_first_error = None

    # ----------------------------------------------------------
    # A0. AI article detection FIRST (Newspaper/Magazine)
    # ----------------------------------------------------------
    if run_ai_first:
        try:
            api_articles = openai_visible_articles_v7(
                image_path=image_path,
                document_type=normalized_type,
                config=config,
            )
            ai_results_first = build_openai_analysis_results_v7(
                articles=api_articles,
                document_type=normalized_type,
                category_model=category_model,
                config=config,
            )
        except Exception as error:
            ai_first_error = str(error)

    # ----------------------------------------------------------
    # A. Always run LOCAL (research component / comparison)
    # ----------------------------------------------------------

    local_analysis = analyze_content_reading_graph_local(
        image_path=image_path,
        document_type=normalized_type,
        model=category_model,
        config=config,
        doclayout_model=doclayout_model,
    )

    # V6.2 can merge/split regions and would undo one-title-one-article.
    local_method = str(
        (local_analysis.get("local_grouping") or {}).get("selected_method")
        or ""
    )
    if (
        normalized_type in MULTI_CATEGORY_TYPES
        and "one_title" not in local_method
    ):
        try:
            local_analysis = postprocess_analysis_v62(
                analysis=local_analysis,
                image_path=image_path,
                category_model=category_model,
                config=config,
                doclayout_model=doclayout_model,
            )

        except Exception as error:
            local_analysis.setdefault(
                "warnings",
                []
            ).append(
                "V6.2 local post-processing failed: "
                f"{error}"
            )

            local_analysis.setdefault(
                "processing",
                {}
            )[
                "v62_postprocess_error"
            ] = str(
                error
            )

    # ----------------------------------------------------------
    # B. Re-evaluate uncertain LOCAL categories with API only
    #    if application mode permits it.
    # ----------------------------------------------------------

    local_openai_category_count = 0

    # Preserve unmodified local articles so the UI can compare local vs AI.
    local_results_snapshot = [
        dict(item)
        for item in local_analysis.get("results", [])
        if isinstance(item, dict)
    ]
    local_analysis["local_results"] = local_results_snapshot
    local_analysis["ai_results"] = []

    for item in local_analysis.get(
        "results",
        [],
    ):
        local_prediction = {
            "category": item.get(
                "category",
                "General Text",
            ),
            "confidence": item.get(
                "confidence",
                0.0,
            ),
            "margin": item.get(
                "probability_margin",
                0.0,
            ),
            "top_predictions": item.get(
                "top_predictions",
                [],
            ),
            "decision": item.get(
                "category_decision",
                "",
            ),
        }

        prediction = maybe_apply_category_fallback_v7(
            item=item,
            document_type=normalized_type,
            local_prediction=local_prediction,
            config=config,
        )

        item[
            "category"
        ] = prediction[
            "category"
        ]

        item[
            "category_source"
        ] = prediction.get(
            "category_source",
            "local_category_model",
        )

        item[
            "openai_category_used"
        ] = prediction.get(
            "openai_category_used",
            False,
        )

        if item[
            "openai_category_used"
        ]:
            local_openai_category_count += 1

    # Rebuild category list after optional category fallback.
    category_counts = Counter(
        item[
            "category"
        ]
        for item in local_analysis.get(
            "results",
            [],
        )
    )

    local_analysis[
        "detected_categories"
    ] = sorted(
        category_counts
    )

    local_analysis[
        "category_counts"
    ] = dict(
        category_counts
    )

    local_analysis[
        "needs_category_selection"
    ] = (
        len(
            category_counts
        )
        > 1
    )

    # ----------------------------------------------------------
    # C. Evaluate LOCAL article/content result
    # ----------------------------------------------------------

    local_quality = evaluate_local_analysis_v7(
        local_analysis,
        config,
    )

    fallback_available, fallback_reason = (
        openai_fallback_available_v7(
            config
        )
    )

    fallback_record = {
        "research_mode": bool(
            config.research_mode
        ),
        "openai_enabled": bool(
            config.enable_openai_fallback
        ),
        "openai_available": bool(
            fallback_available
        ),
        "openai_availability_reason": (
            fallback_reason
        ),
        "local_quality": local_quality,
        "article_fallback_used": False,
        "category_fallback_count": (
            local_openai_category_count
        ),
        "summary_fallback_used": False,
        "fallback_errors": [],
    }

    local_analysis[
        "fallback"
    ] = fallback_record

    local_analysis.setdefault(
        "processing",
        {}
    )[
        "primary_research_pipeline"
    ] = (
        "local_v6_2_1_reconstruction"
        if normalized_type
        in MULTI_CATEGORY_TYPES
        else "local_single_document_pipeline"
    )

    if ai_first_error:
        local_analysis.setdefault("warnings", []).append(
            "AI-first article detection failed: " + ai_first_error
        )
        fallback_record.setdefault("fallback_errors", []).append(ai_first_error)

    if ai_results_first:
        return _adopt_ai_article_results_v7(
            analysis=local_analysis,
            ai_results=ai_results_first,
            local_results=local_results_snapshot,
            local_quality=local_quality,
            fallback_record=fallback_record,
            pipeline_version="V7_AI_ARTICLE_DETECTION_FIRST",
            article_source="openai_article_detection_first",
        )

    # ----------------------------------------------------------
    # D. Keep local if reliable
    # ----------------------------------------------------------

    if local_quality[
        "reliable"
    ]:
        local_analysis[
            "pipeline_version"
        ] = (
            "V7_LOCAL_PRIMARY"
        )

        local_analysis[
            "processing"
        ][
            "openai_used"
        ] = (
            local_openai_category_count
            > 0
        )

        return local_analysis

    # ----------------------------------------------------------
    # E. OpenAI article/content fallback
    # ----------------------------------------------------------

    if run_ai_first:
        local_analysis[
            "pipeline_version"
        ] = (
            "V7_LOCAL_AFTER_AI_FIRST_EMPTY"
        )
        local_analysis[
            "processing"
        ][
            "openai_used"
        ] = False
        local_analysis.setdefault("warnings", []).append(
            "AI-first article detection returned no usable articles; "
            "local detections are used instead."
        )
        return local_analysis

    if not fallback_available:
        local_analysis[
            "pipeline_version"
        ] = (
            "V7_LOCAL_RESULT_OPENAI_UNAVAILABLE"
        )

        local_analysis[
            "processing"
        ][
            "openai_used"
        ] = False

        local_analysis.setdefault(
            "warnings",
            []
        ).append(
            "Local quality gate was weak, but OpenAI application fallback "
            f"was unavailable: {fallback_reason}"
        )

        return local_analysis

    try:
        api_articles = openai_visible_articles_v7(
            image_path=image_path,
            document_type=normalized_type,
            config=config,
        )

        api_results = build_openai_analysis_results_v7(
            articles=api_articles,
            document_type=normalized_type,
            category_model=category_model,
            config=config,
        )

        if not api_results:
            raise RuntimeError(
                "OpenAI fallback did not return usable visible reading content."
            )

        # Keep local analysis in debug for comparison, but use fallback
        # results for the application output.
        local_snapshot = {
            "pipeline_version": local_analysis.get(
                "pipeline_version",
                "",
            ),
            "detected_categories": local_analysis.get(
                "detected_categories",
                [],
            ),
            "category_counts": local_analysis.get(
                "category_counts",
                {},
            ),
            "result_count": len(
                local_results_snapshot
            ),
            "quality": local_quality,
            "results": local_results_snapshot,
        }

        category_counts = Counter(
            item[
                "category"
            ]
            for item in api_results
        )

        local_analysis[
            "results"
        ] = api_results

        local_analysis[
            "local_results"
        ] = local_results_snapshot

        local_analysis[
            "ai_results"
        ] = api_results

        local_analysis[
            "detected_categories"
        ] = sorted(
            category_counts
        )

        local_analysis[
            "category_counts"
        ] = dict(
            category_counts
        )

        local_analysis[
            "needs_category_selection"
        ] = (
            len(
                category_counts
            )
            > 1
        )

        local_analysis[
            "mode"
        ] = (
            "Multi Category"
            if len(
                category_counts
            )
            > 1
            else "Single Category"
        )

        local_analysis[
            "pipeline_version"
        ] = (
            "V7_OPENAI_APPLICATION_FALLBACK"
        )

        fallback_record[
            "article_fallback_used"
        ] = True

        fallback_record[
            "local_primary_snapshot"
        ] = local_snapshot

        fallback_record[
            "api_article_count"
        ] = len(
            api_results
        )

        local_analysis[
            "processing"
        ][
            "openai_used"
        ] = True

        local_analysis[
            "processing"
        ][
            "article_source"
        ] = (
            "openai_application_fallback_after_local_failure"
        )

        local_analysis.setdefault(
            "warnings",
            []
        ).append(
            "Application fallback was used because the local research "
            "pipeline did not pass the configured quality gate. "
            "Do not use this API-assisted result when calculating the "
            "local research component's accuracy."
        )

        return local_analysis

    except Exception as error:
        fallback_record[
            "fallback_errors"
        ].append(
            str(
                error
            )
        )

        local_analysis[
            "pipeline_version"
        ] = (
            "V7_LOCAL_RESULT_AFTER_OPENAI_FALLBACK_FAILURE"
        )

        local_analysis[
            "processing"
        ][
            "openai_used"
        ] = (
            local_openai_category_count
            > 0
        )

        local_analysis.setdefault(
            "warnings",
            []
        ).append(
            "OpenAI article/content fallback failed, so the best local "
            f"result was retained. Error: {error}"
        )

        return local_analysis


# ---------------------------------------------------------------------
# 25.9 OPENAI SUMMARY FALLBACK
# ---------------------------------------------------------------------


def openai_summary_v7(
    *,
    text: str,
    title: str,
    document_type: str,
    category: str,
    config: ApplicationFallbackConfig,
) -> str:
    client = get_openai_client_v7(
        config
    )

    source = " ".join(
        str(
            text
        ).split()[
            :1200
        ]
    )

    schema = {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string"
            }
        },
        "required": [
            "summary"
        ],
        "additionalProperties": False,
    }

    prompt = f"""
Summarize ONLY the source text below for a visually impaired reader.

Document type: {document_type}
Category: {category}
Title: {title}

Rules:
- Maximum {config.openai_summary_max_words} words.
- Preserve important names, numbers, scores, dates, places and main facts.
- Do not add facts that are not in the source text.
- Do not infer missing beginning/endings.
- If the source is partial, summarize only the visible source.
- Do not mention OCR, API, model, or summarization process.
- Return only the summary field required by the schema.

SOURCE TEXT:
{source}
""".strip()

    data = openai_structured_response_v7(
        client=client,
        model=config.openai_model,
        input_items=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": prompt,
                    }
                ],
            }
        ],
        schema_name="grounded_summary",
        schema=schema,
        max_output_tokens=(
            config.openai_max_output_tokens_text
        ),
    )

    return remove_newspaper_artifacts(
        str(
            data.get(
                "summary",
                ""
            )
        )
    )


# ---------------------------------------------------------------------
# 25.10 OUTPUT — LOCAL LLAMA FIRST, OPENAI SUMMARY SECOND
# ---------------------------------------------------------------------


def generate_selected_output_v7(
    *,
    analysis: dict[str, Any],
    selected_category: Optional[str],
    depth: str,
    config: ApplicationFallbackConfig,
    clean_full_text_with_llama: bool = False,
) -> dict[str, Any]:
    categories = analysis.get(
        "detected_categories",
        [],
    )

    if not categories:
        return {
            "status": "ERROR",
            "message": (
                "No category was detected."
            ),
        }

    if selected_category is None:
        selected_category = (
            categories[
                0
            ]
            if len(
                categories
            )
            == 1
            else None
        )

    if selected_category not in categories:
        return {
            "status": "ERROR",
            "message": (
                "Selected category was not detected."
            ),
            "available_categories": (
                categories
            ),
        }

    depth = str(
        depth
    ).strip().lower()

    if depth not in {
        "summary",
        "full",
    }:
        return {
            "status": "ERROR",
            "message": (
                "Depth must be 'summary' or 'full'."
            ),
        }

    selected_items = [
        item
        for item in analysis.get(
            "results",
            [],
        )
        if item.get(
            "category"
        )
        == selected_category
    ]

    output_items = []

    available, _ = openai_fallback_available_v7(
        config
    )

    openai_summary_count = 0

    # Check Ollama once.
    try:
        ollama_status = check_ollama(
            config
        )

        ollama_ready = bool(
            ollama_status.get(
                "available",
                ollama_status.get(
                    "ok",
                    False,
                ),
            )
        )

    except Exception:
        ollama_ready = False

    for item in selected_items:
        source_text = remove_newspaper_artifacts(
            item.get(
                "full_text",
                "",
            )
        )

        warnings = list(
            item.get(
                "warnings",
                [],
            )
        )

        method = ""

        if depth == "full":
            content = source_text
            method = (
                "visible_reconstructed_text"
            )

        else:
            content = ""

            # --------------------------------------------------
            # A. Local Llama first
            # --------------------------------------------------

            if (
                ollama_ready
                and article_safe_for_llama_summary(
                    item,
                    config,
                )
            ):
                try:
                    local_summary = summarize_with_local_llama(
                        text=source_text,
                        title=item.get(
                            "title",
                            "",
                        ),
                        document_type=analysis[
                            "document_type"
                        ],
                        category=selected_category,
                        config=config,
                    )

                    if (
                        local_summary
                        and summary_is_grounded(
                            local_summary,
                            source_text,
                            config,
                        )
                    ):
                        content = local_summary
                        method = (
                            "local_llama_grounded"
                        )

                except Exception as error:
                    warnings.append(
                        "Local Llama summary failed: "
                        f"{error}"
                    )

            # --------------------------------------------------
            # B. OpenAI summary only if local summary did not
            #    produce an accepted result.
            # --------------------------------------------------

            if (
                not content
                and config.enable_openai_summary_fallback
                and available
            ):
                try:
                    api_summary = openai_summary_v7(
                        text=source_text,
                        title=item.get(
                            "title",
                            "",
                        ),
                        document_type=analysis[
                            "document_type"
                        ],
                        category=selected_category,
                        config=config,
                    )

                    # Use the existing local grounding check as a guard.
                    if (
                        api_summary
                        and summary_is_grounded(
                            api_summary,
                            source_text,
                            config,
                        )
                    ):
                        content = api_summary
                        method = (
                            "openai_summary_fallback"
                        )
                        openai_summary_count += 1

                    else:
                        warnings.append(
                            "OpenAI summary did not pass the local grounding "
                            "check; extractive fallback was used."
                        )

                except Exception as error:
                    warnings.append(
                        "OpenAI summary fallback failed: "
                        f"{error}"
                    )

            # --------------------------------------------------
            # C. Final no-cloud safe fallback
            # --------------------------------------------------

            if not content:
                content = extractive_summary(
                    source_text,
                    max_sentences=2,
                )

                method = (
                    "extractive_final_fallback"
                )

        output_items.append(
            {
                "article_number": item.get(
                    "article_number",
                    len(
                        output_items
                    )
                    + 1,
                ),
                "title": item.get(
                    "title",
                    "",
                ),
                "category": item.get(
                    "category",
                    selected_category,
                ),
                "category_source": item.get(
                    "category_source",
                    "local_category_model",
                ),
                "category_confidence": item.get(
                    "confidence",
                    0.0,
                ),
                "probability_margin": item.get(
                    "probability_margin",
                    0.0,
                ),
                "ocr_quality": item.get(
                    "ocr_quality",
                    0.0,
                ),
                "layout_confidence": item.get(
                    "layout_confidence",
                    0.0,
                ),
                "source": item.get(
                    "source",
                    "",
                ),
                "fallback_used": bool(
                    item.get(
                        "fallback_used",
                        False,
                    )
                ),
                "partial_top": item.get(
                    "partial_top",
                    False,
                ),
                "partial_bottom": item.get(
                    "partial_bottom",
                    False,
                ),
                "processing_method": method,
                "warnings": warnings,
                "text": content,
                "bbox": item.get(
                    "normalized_bbox",
                    (
                        0,
                        0,
                        1000,
                        1000,
                    ),
                ),
            }
        )

    if (
        analysis.get(
            "document_type"
        )
        in MULTI_CATEGORY_TYPES
    ):
        intro = (
            f"{selected_category} section. "
            f"I found {len(output_items)} "
            f"article"
            f"{'s' if len(output_items) != 1 else ''}."
        )

        spoken_parts = [
            intro
        ]

        for index, item in enumerate(
            output_items,
            start=1,
        ):
            title = item[
                "title"
            ].strip()

            if title:
                spoken_parts.append(
                    f"Article {index}. "
                    f"{title}. "
                    f"{item['text']}"
                )
            else:
                spoken_parts.append(
                    f"Article {index}. "
                    f"{item['text']}"
                )

        final_text = "\n\n".join(
            spoken_parts
        )

    else:
        final_text = "\n\n".join(
            item[
                "text"
            ]
            for item in output_items
        )

    analysis.setdefault(
        "fallback",
        {}
    )[
        "summary_fallback_used"
    ] = (
        openai_summary_count
        > 0
    )

    analysis[
        "fallback"
    ][
        "openai_summary_count"
    ] = openai_summary_count

    return {
        "status": "READY",
        "document_type": analysis.get(
            "document_type",
            "",
        ),
        "mode": analysis.get(
            "mode",
            "",
        ),
        "selected_category": (
            selected_category
        ),
        "content_depth": depth,
        "article_count": len(
            output_items
        ),
        "items": output_items,
        "final_output_text": final_text,
        "fallback": analysis.get(
            "fallback",
            {},
        ),
        "next_module_payload": {
            "text": final_text,
            "document_type": analysis.get(
                "document_type",
                "",
            ),
            "category": selected_category,
            "content_depth": depth,
            "source_article_count": len(
                output_items
            ),
            "openai_used": bool(
                analysis.get(
                    "processing",
                    {},
                ).get(
                    "openai_used",
                    False,
                )
                or openai_summary_count
                > 0
            ),
            "research_mode": bool(
                config.research_mode
            ),
            "articles": output_items,
        },
    }


# ---------------------------------------------------------------------
# 25.11 VOICE END-TO-END
# ---------------------------------------------------------------------


def run_v7_voice_pipeline(
    *,
    image_path: str,
    document_type: str,
    category_model,
    config: ApplicationFallbackConfig,
    doclayout_model,
    voice_selector: OfflineVoiceSelector,
    tts_enabled: bool = False,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
]:
    analysis = analyze_content_v7(
        image_path=image_path,
        document_type=document_type,
        category_model=category_model,
        config=config,
        doclayout_model=doclayout_model,
    )

    selected_category, depth = collect_user_preferences_voice(
        analysis=analysis,
        voice_selector=voice_selector,
        tts_enabled=tts_enabled,
    )

    output = generate_selected_output_v7(
        analysis=analysis,
        selected_category=selected_category,
        depth=depth,
        config=config,
    )

    return (
        analysis,
        output,
    )


# ---------------------------------------------------------------------
# 25.12 DEBUG TABLES
# ---------------------------------------------------------------------


def fallback_status_table_v7(
    analysis: dict[str, Any],
) -> pd.DataFrame:
    fallback = analysis.get(
        "fallback",
        {},
    )

    quality = fallback.get(
        "local_quality",
        {},
    )

    metrics = quality.get(
        "metrics",
        {},
    )

    rows = [
        (
            "Pipeline version",
            analysis.get(
                "pipeline_version",
                "",
            ),
        ),
        (
            "Research mode",
            fallback.get(
                "research_mode",
                False,
            ),
        ),
        (
            "OpenAI available",
            fallback.get(
                "openai_available",
                False,
            ),
        ),
        (
            "Local result reliable",
            quality.get(
                "reliable",
                False,
            ),
        ),
        (
            "Local failure reasons",
            ", ".join(
                quality.get(
                    "reasons",
                    [],
                )
            ),
        ),
        (
            "Local article count",
            metrics.get(
                "article_count",
                0,
            ),
        ),
        (
            "Local total words",
            metrics.get(
                "total_words",
                0,
            ),
        ),
        (
            "Average local text quality",
            metrics.get(
                "average_ocr_quality",
                0.0,
            ),
        ),
        (
            "Selected local quality",
            metrics.get(
                "selected_local_quality",
                0.0,
            ),
        ),
        (
            "Selected local method",
            metrics.get(
                "selected_local_method",
                "",
            ),
        ),
        (
            "OpenAI article fallback used",
            fallback.get(
                "article_fallback_used",
                False,
            ),
        ),
        (
            "OpenAI category fallback count",
            fallback.get(
                "category_fallback_count",
                0,
            ),
        ),
        (
            "OpenAI summary fallback used",
            fallback.get(
                "summary_fallback_used",
                False,
            ),
        ),
    ]

    return pd.DataFrame(
        rows,
        columns=[
            "Stage",
            "Value",
        ],
    )


def result_table_v7(
    analysis: dict[str, Any],
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "article": item.get(
                    "article_number"
                ),
                "title": item.get(
                    "title"
                ),
                "category": item.get(
                    "category"
                ),
                "category_source": item.get(
                    "category_source",
                    "local_category_model",
                ),
                "category_confidence": item.get(
                    "confidence"
                ),
                "ocr_quality": item.get(
                    "ocr_quality"
                ),
                "layout_confidence": item.get(
                    "layout_confidence"
                ),
                "partial_top": item.get(
                    "partial_top"
                ),
                "partial_bottom": item.get(
                    "partial_bottom"
                ),
                "source": item.get(
                    "source"
                ),
                "fallback_used": item.get(
                    "fallback_used",
                    False,
                ),
                "words": len(
                    str(
                        item.get(
                            "full_text",
                            "",
                        )
                    ).split()
                ),
            }
            for item in analysis.get(
                "results",
                []
            )
        ]
    )

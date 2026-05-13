# ============================================================
# STABLE HYBRID CONTENT SUMMARIZATION + CATEGORY DETECTION
#
# Fixed issue:
# - Article region detection sometimes returns different counts
# - Added image-hash-based region cache
# - Added deterministic region post-processing
#
# Workflow:
# - Newspaper/Magazine:
#     API detects article regions only
#     Detected regions are cached using image hash
#     Tesseract extracts full-page OCR lines once
#     OCR lines are assigned to article regions
#     Category model detects category
#
# - Output:
#     User selects category
#     User selects summary/full
#     Full text also passes through Llama reconstruction
# ============================================================

import os
import re
import cv2
import json
import base64
import hashlib
import mimetypes
import joblib
import requests
import numpy as np
import pandas as pd
import pytesseract
import matplotlib.pyplot as plt

from collections import Counter
from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# 1. SETUP
# ============================================================

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_PACKAGE_DIR, "..", ".."))
load_dotenv(os.path.join(_PROJECT_ROOT, "backend", ".env"))
load_dotenv()

# Run before this code:
# import os
# os.environ["OPENAI_API_KEY"] = "YOUR_KEY"
# os.environ["TESSERACT_CMD"] = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# os.environ["LLAMA_MODEL"] = "llama3.2:3b"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

VISION_MODEL = "gpt-4.1-mini"

OLLAMA_BASE_URL = "http://localhost:11434"
LLAMA_MODEL = os.getenv("LLAMA_MODEL", "llama3.2:3b")

_DEFAULT_CATEGORY_MODEL = os.path.join(_PROJECT_ROOT, "model", "category_model.pkl")
MODEL_PATH = os.getenv("CATEGORY_MODEL_PATH", _DEFAULT_CATEGORY_MODEL)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "category_model.pkl not found. Place it in model/category_model.pkl "
        "or set CATEGORY_MODEL_PATH."
    )

category_model = joblib.load(MODEL_PATH)
print("Category model loaded successfully.")

TESSERACT_CMD = os.getenv("TESSERACT_CMD")

if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
else:
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

PROJECT_CATEGORIES = [
    "Sports",
    "Politics",
    "Social",
    "Science",
    "Mathematics",
    "Story",
    "General Text"
]

SHOW_EXTRACTED_TEXT = True
SHOW_REGION_IMAGE = False

ALLOW_REGION_RETRY = True
MIN_NEWSPAPER_REGIONS = 6

REGION_CACHE_DIR = os.path.join(_PACKAGE_DIR, "region_cache")
os.makedirs(REGION_CACHE_DIR, exist_ok=True)


# ============================================================
# 2. DOCUMENT TYPE
# ============================================================

def normalize_document_type(document_type):
    if document_type is None:
        return "General Document"

    dt = str(document_type).strip().lower()

    if "news" in dt or "newspaper" in dt:
        return "Newspaper"

    if "magazine" in dt:
        return "Magazine"

    if "textbook" in dt or "text book" in dt:
        return "Textbook"

    if "novel" in dt or "story" in dt or "storybook" in dt or "story book" in dt:
        return "Novel/Story"

    if "general" in dt or "document" in dt or "text" in dt:
        return "General Document"

    return "General Document"


def is_multi_category_document(document_type):
    return normalize_document_type(document_type) in ["Newspaper", "Magazine"]


# ============================================================
# 3. CACHE HELPERS FOR STABLE ARTICLE COUNT
# ============================================================

def get_image_hash(image_path):
    with open(image_path, "rb") as f:
        data = f.read()

    return hashlib.sha256(data).hexdigest()[:24]


def get_cache_path(image_path, document_type):
    image_hash = get_image_hash(image_path)
    doc_type = normalize_document_type(document_type).replace("/", "_").replace(" ", "_")

    filename = f"{doc_type}_{image_hash}.json"

    return os.path.join(REGION_CACHE_DIR, filename)


def load_cached_regions(image_path, document_type):
    cache_path = get_cache_path(image_path, document_type)

    if not os.path.exists(cache_path):
        return None

    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_cached_regions(image_path, document_type, region_result):
    cache_path = get_cache_path(image_path, document_type)

    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(region_result, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


def clear_region_cache_for_image(image_path, document_type):
    cache_path = get_cache_path(image_path, document_type)

    if os.path.exists(cache_path):
        os.remove(cache_path)


# ============================================================
# 4. TEXT CLEANING
# ============================================================

def clean_text_basic(text):
    if text is None:
        return ""

    text = str(text)
    text = text.replace("\r", "\n")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")

    text = re.sub(r"([A-Za-z])-\s*\n\s*([a-z])", r"\1\2", text)
    text = re.sub(r"[^a-zA-Z0-9.,!?'\n =+\-*/()%:;$]", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def alpha_ratio(text):
    chars = [c for c in str(text) if not c.isspace()]
    if not chars:
        return 0.0
    return sum(c.isalpha() for c in chars) / len(chars)


def symbol_ratio(text):
    chars = [c for c in str(text) if not c.isspace()]
    if not chars:
        return 0.0

    allowed = set(".,!?':;()-/%+=*$")
    symbols = sum(1 for c in chars if not c.isalnum() and c not in allowed)

    return symbols / len(chars)


def uppercase_ratio(text):
    letters = [c for c in str(text) if c.isalpha()]
    if not letters:
        return 0.0
    return sum(c.isupper() for c in letters) / len(letters)


def title_case_ratio(text):
    words = re.findall(r"[A-Za-z]+", str(text))
    if not words:
        return 0.0

    title_words = [
        w for w in words
        if len(w) > 1 and w[0].isupper()
    ]

    return len(title_words) / len(words)


def has_math_content(line):
    return bool(re.search(r"\d", str(line))) and bool(re.search(r"[=+\-*/()%]", str(line)))


def looks_like_heading_line(line):
    line = str(line).strip()
    words = re.findall(r"[A-Za-z0-9]+", line)

    if len(words) < 2 or len(words) > 28:
        return False

    if alpha_ratio(line) < 0.45:
        return False

    if symbol_ratio(line) > 0.35:
        return False

    if title_case_ratio(line) >= 0.35:
        return True

    if uppercase_ratio(line) >= 0.40:
        return True

    return False


def is_newspaper_masthead_or_header(line):
    lower = str(line).lower().strip()

    patterns = [
        "sri lanka", "newspaper", "friday", "monday", "tuesday",
        "wednesday", "thursday", "saturday", "sunday",
        "volume", "pages", "daily news", "dailynews",
        "late city", "weather", "registered",
        "www.", ".lk", ".com", "lake house"
    ]

    return any(p in lower for p in patterns)


def is_known_noise_line(line):
    lower = str(line).lower().strip()

    noise_patterns = [
        r"^page\s*\d+$",
        r"^\d+\s*pages?$",
        r"\bwww\.",
        r"\.com\b",
        r"\.lk\b",
        r"\bdaily\s*news\b",
        r"\bdailynews\b",
        r"\bweather\b",
        r"\bvol\.?\s*no\b",
        r"\bregistered\b",
        r"\blate\s*city\b",
        r"\blake\s*house\b",
        r"\bsince\s*1918\b",
        r"^advertisement$",
        r"^classifieds?$",
        r"^photo$",
        r"^caption$",
        r"^to page",
        r"^ito page",
        r"^mito page",
        r"^hito page",
        r"^nto page",
        r"^bto page"
    ]

    return any(re.search(pattern, lower) for pattern in noise_patterns)


def is_garbage_text_line(line):
    line = str(line).strip()

    if not line:
        return True

    words = re.findall(r"[A-Za-z0-9]+", line)

    if not words:
        return True

    if any(len(w) >= 24 for w in words):
        return True

    short_words = [w for w in words if len(w) <= 2]

    if len(words) >= 4 and len(short_words) / len(words) > 0.65:
        return True

    if re.search(r"(.)\1{6,}", line):
        return True

    if alpha_ratio(line) < 0.35:
        return True

    return False


def remove_long_garbage_lines(text):
    lines = str(text).split("\n")
    clean_lines = []

    for line in lines:
        line = line.strip()

        if not line:
            continue

        words = re.findall(r"[A-Za-z0-9]+", line)

        if any(len(w) >= 24 for w in words):
            continue

        clean_lines.append(line)

    return "\n".join(clean_lines).strip()


def is_bad_ocr_line(line, document_type="General Document"):
    line = str(line).strip()

    if not line:
        return True

    doc_type = normalize_document_type(document_type)

    if doc_type in ["Newspaper", "Magazine"]:
        if is_newspaper_masthead_or_header(line):
            return True

    if doc_type == "Textbook" and has_math_content(line):
        return False

    if is_known_noise_line(line):
        return True

    if is_garbage_text_line(line):
        return True

    words = re.findall(r"[A-Za-z0-9]+", line)

    if len(words) <= 1:
        return True

    if len(words) <= 2 and len(line) < 12:
        return True

    return False


def clean_ocr_text(text, document_type="General Document", preserve_line_breaks=False):
    text = clean_text_basic(text)
    text = remove_long_garbage_lines(text)

    raw_lines = text.split("\n")
    cleaned_lines = []
    seen = set()

    for line in raw_lines:
        line = line.strip()
        line = re.sub(r"\s+", " ", line)

        if is_bad_ocr_line(line, document_type=document_type):
            continue

        key = line.lower()

        if key in seen:
            continue

        seen.add(key)
        cleaned_lines.append(line)

    if not cleaned_lines:
        return ""

    if preserve_line_breaks:
        return "\n".join(cleaned_lines).strip()

    paragraphs = []
    current = ""

    for line in cleaned_lines:
        if looks_like_heading_line(line):
            if current:
                paragraphs.append(current.strip())
                current = ""

            paragraphs.append(line)
            continue

        if not current:
            current = line
        else:
            if current.endswith((".", "!", "?", ":")):
                paragraphs.append(current.strip())
                current = line
            else:
                current += " " + line

    if current:
        paragraphs.append(current.strip())

    final_paragraphs = [
        p for p in paragraphs
        if len(p.split()) >= 3 or looks_like_heading_line(p)
    ]

    return "\n\n".join(final_paragraphs).strip()


def clean_newspaper_ocr_text(text):
    text = clean_text_basic(text)
    lines = text.split("\n")

    cleaned_lines = []

    for line in lines:
        line = line.strip()

        if not line:
            continue

        if is_newspaper_masthead_or_header(line):
            continue

        if is_known_noise_line(line):
            continue

        if is_garbage_text_line(line):
            continue

        cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines)

    cleaned = clean_ocr_text(
        cleaned,
        document_type="Newspaper",
        preserve_line_breaks=False
    )

    return cleaned.strip()


def stronger_clean_text(text, document_type="General Document"):
    doc_type = normalize_document_type(document_type)

    if doc_type in ["Newspaper", "Magazine"]:
        return clean_newspaper_ocr_text(text)

    text = clean_text_basic(text)
    text = remove_long_garbage_lines(text)

    text = clean_ocr_text(
        text,
        document_type=doc_type,
        preserve_line_breaks=False
    )

    return text.strip()


# ============================================================
# 5. OCR QUALITY
# ============================================================

def ocr_quality_score(text):
    words = str(text).split()

    if not words:
        return 0.0

    readable_words = [
        w for w in words
        if re.match(r"^[A-Za-z]{3,}$", w)
    ]

    return round(len(readable_words) / len(words), 3)


def is_ocr_good(text, min_words=25, min_score=0.25):
    if len(str(text).split()) < min_words:
        return False

    return ocr_quality_score(text) >= min_score


# ============================================================
# 6. FULL PAGE OCR WITH LINE POSITIONS
# ============================================================

def upscale_if_needed(img, target_width=2200):
    h, w = img.shape[:2]

    if w >= target_width:
        return img, 1.0

    scale = target_width / w

    resized = cv2.resize(
        img,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC
    )

    return resized, scale


def prepare_ocr_image(img):
    working, scale = upscale_if_needed(img, target_width=2200)

    gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

    sharpen = cv2.addWeighted(
        gray,
        1.5,
        cv2.GaussianBlur(gray, (0, 0), 3),
        -0.5,
        0
    )

    return sharpen, scale


def extract_full_page_ocr_lines(img, document_type="Newspaper", debug=False):
    ocr_img, scale = prepare_ocr_image(img)

    try:
        data = pytesseract.image_to_data(
            ocr_img,
            config="--oem 3 --psm 11",
            output_type=pytesseract.Output.DATAFRAME
        )
    except Exception:
        return []

    data = data.dropna(subset=["text"])
    data["text"] = data["text"].astype(str).str.strip()
    data = data[data["text"] != ""]

    try:
        data = data[data["conf"].astype(float) > 5]
    except Exception:
        pass

    if data.empty:
        return []

    group_cols = ["block_num", "par_num", "line_num"]
    lines = []

    for _, group in data.groupby(group_cols):
        group = group.sort_values("left")

        words = []

        for _, row in group.iterrows():
            word = str(row["text"]).strip()

            if word:
                words.append(word)

        text = " ".join(words)
        text = clean_text_basic(text)

        if not text:
            continue

        x1 = int(group["left"].min() / scale)
        y1 = int(group["top"].min() / scale)
        x2 = int((group["left"] + group["width"]).max() / scale)
        y2 = int((group["top"] + group["height"]).max() / scale)

        if x2 <= x1 or y2 <= y1:
            continue

        if is_bad_ocr_line(text, document_type=document_type):
            continue

        line = {
            "text": text,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "cx": (x1 + x2) / 2,
            "cy": (y1 + y2) / 2,
            "width": x2 - x1,
            "height": y2 - y1,
            "word_count": len(text.split())
        }

        lines.append(line)

    lines = sorted(lines, key=lambda l: (l["y1"], l["x1"]))

    if debug:
        print("Full-page OCR lines:", len(lines))

    return lines


def line_inside_bbox(line, bbox, margin=6):
    x1, y1, x2, y2 = bbox

    return (
        x1 - margin <= line["cx"] <= x2 + margin
        and y1 - margin <= line["cy"] <= y2 + margin
    )


def line_overlap_ratio_with_bbox(line, bbox):
    lx1, ly1, lx2, ly2 = line["x1"], line["y1"], line["x2"], line["y2"]
    bx1, by1, bx2, by2 = bbox

    ix1 = max(lx1, bx1)
    iy1 = max(ly1, by1)
    ix2 = min(lx2, bx2)
    iy2 = min(ly2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)

    inter = iw * ih
    line_area = max(1, (lx2 - lx1) * (ly2 - ly1))

    return inter / line_area


def get_region_lines(all_lines, bbox):
    selected = []

    for line in all_lines:
        if line_inside_bbox(line, bbox) or line_overlap_ratio_with_bbox(line, bbox) > 0.35:
            selected.append(line)

    return selected


def split_lines_into_columns(lines, bbox):
    if not lines:
        return []

    x1, y1, x2, y2 = bbox
    region_width = x2 - x1

    sorted_lines = sorted(lines, key=lambda l: l["x1"])

    column_groups = []
    current = []

    for line in sorted_lines:
        if not current:
            current.append(line)
            continue

        prev_x = np.median([l["x1"] for l in current])

        if abs(line["x1"] - prev_x) > region_width * 0.22:
            column_groups.append(current)
            current = [line]
        else:
            current.append(line)

    if current:
        column_groups.append(current)

    if len(column_groups) <= 1 or len(column_groups) > 4:
        return sorted(lines, key=lambda l: (l["y1"], l["x1"]))

    ordered = []

    column_groups = sorted(
        column_groups,
        key=lambda group: np.median([l["x1"] for l in group])
    )

    for group in column_groups:
        group_sorted = sorted(group, key=lambda l: (l["y1"], l["x1"]))
        ordered.extend(group_sorted)

    return ordered


def lines_to_article_text(lines, bbox, document_type):
    ordered_lines = split_lines_into_columns(lines, bbox)

    raw_text = "\n".join(line["text"] for line in ordered_lines)

    cleaned = stronger_clean_text(
        raw_text,
        document_type=document_type
    )

    return cleaned


def best_full_page_text(img, document_type="General Document", debug=False):
    ocr_img, scale = prepare_ocr_image(img)

    best_text = ""
    best_score = -999

    for psm in [3, 4, 6, 11]:
        try:
            raw = pytesseract.image_to_string(
                ocr_img,
                config=f"--oem 3 --psm {psm}"
            )
        except Exception:
            continue

        cleaned = stronger_clean_text(
            raw,
            document_type=document_type
        )

        score = len(cleaned.split()) + (ocr_quality_score(cleaned) * 100)

        if score > best_score:
            best_score = score
            best_text = cleaned

    return best_text


# ============================================================
# 7. CATEGORY DETECTION
# ============================================================

def safe_category(category):
    category = str(category).strip()

    if category in PROJECT_CATEGORIES:
        return category

    return "General Text"


def get_model_classes():
    if hasattr(category_model, "classes_"):
        return list(category_model.classes_)

    if hasattr(category_model, "named_steps"):
        last_step = list(category_model.named_steps.values())[-1]

        if hasattr(last_step, "classes_"):
            return list(last_step.classes_)

    return None


def count_keywords(text, keywords):
    text = str(text).lower()
    return sum(1 for word in keywords if word.lower() in text)


def keyword_hints(text):
    t = str(text).lower()

    keyword_map = {
        "Sports": [
            "cricket", "slc", "football", "tennis", "match",
            "team", "player", "tournament", "score", "wicket",
            "goal", "sports", "coach", "captain", "icc"
        ],
        "Politics": [
            "treasury", "minister", "president", "speaker",
            "court", "judgement", "government", "opposition",
            "parliament", "ciaboc", "debt", "funds", "official",
            "police", "election", "secretary", "committee",
            "cabinet", "rallies", "finance", "copf"
        ],
        "Social": [
            "workers", "labour", "working environment", "community",
            "society", "social", "families", "rights", "dignity",
            "trade union", "festival", "may day", "working people",
            "safe working", "international labour day"
        ],
        "Science": [
            "science", "scientific", "anatomy", "medical", "biology",
            "chemistry", "physics", "experiment", "research",
            "population", "environment", "energy"
        ],
        "Mathematics": [
            "math", "maths", "mathematics", "equation", "formula",
            "algebra", "geometry", "percentage", "interest",
            "compound", "growth rate", "rule of 72", "calculate",
            "number", "fraction", "ratio"
        ],
        "Story": [
            "story", "boy", "girl", "king", "queen", "river",
            "forest", "moral", "once", "asked", "said",
            "jumping competition", "baker"
        ]
    }

    hints = []

    for category, words in keyword_map.items():
        count = sum(1 for w in words if w in t)

        if count > 0:
            hints.append((category, count))

    return sorted(hints, key=lambda x: x[1], reverse=True)


def local_model_predict_category(text):
    cleaned = clean_text_basic(text)

    if len(cleaned.split()) < 8:
        return {
            "category": "General Text",
            "confidence": 0.0
        }

    hints = keyword_hints(cleaned)

    try:
        if hasattr(category_model, "predict_proba"):
            probs = category_model.predict_proba([cleaned])[0]
            best_index = int(np.argmax(probs))

            classes = get_model_classes()

            if classes is not None:
                category = str(classes[best_index])
            else:
                category = str(category_model.predict([cleaned])[0])

            confidence = float(probs[best_index])
        else:
            category = str(category_model.predict([cleaned])[0])
            confidence = None

    except Exception:
        category = "General Text"
        confidence = 0.0

    category = safe_category(category)

    if hints:
        top_hint_category, top_hint_count = hints[0]

        if category == "General Text" and top_hint_count >= 2:
            category = top_hint_category

        elif confidence is not None and confidence < 0.45 and top_hint_count >= 3:
            category = top_hint_category

    return {
        "category": category,
        "confidence": None if confidence is None else round(confidence, 3)
    }


def document_type_category_hint(text, document_type, title=""):
    doc_type = normalize_document_type(document_type)
    combined = f"{title} {text}".lower()
    title_lower = str(title).lower()

    sports_keywords = [
        "cricket", "slc", "football", "tennis", "match",
        "tournament", "player", "team", "wicket", "goal",
        "sports", "coach", "captain", "icc"
    ]

    politics_keywords = [
        "president", "minister", "parliament", "government",
        "opposition", "speaker", "treasury", "court", "election",
        "cabinet", "finance", "ciaboc", "police", "rallies", "copf"
    ]

    social_keywords = [
        "labour", "workers", "working people", "community",
        "society", "rights", "dignity", "trade union",
        "may day", "families", "safe working", "international labour day"
    ]

    science_keywords = [
        "science", "scientific", "anatomy", "medical", "biology",
        "chemistry", "physics", "experiment", "research",
        "population", "environment", "energy"
    ]

    mathematics_keywords = [
        "math", "maths", "mathematics", "equation", "formula",
        "algebra", "geometry", "percentage", "interest",
        "compound", "growth rate", "rule of 72", "calculate",
        "number", "fraction", "ratio"
    ]

    if doc_type == "Novel/Story":
        return "Story"

    if doc_type == "Textbook":
        math_score = count_keywords(combined, mathematics_keywords)
        science_score = count_keywords(combined, science_keywords)

        if math_score >= 2:
            return "Mathematics"

        if science_score >= 2:
            return "Science"

        return "General Text"

    if count_keywords(title_lower, sports_keywords) >= 1:
        return "Sports"

    if count_keywords(title_lower, social_keywords) >= 1:
        return "Social"

    if count_keywords(title_lower, politics_keywords) >= 1:
        return "Politics"

    return None


def fix_category_for_document_type(category, document_type, text="", title=""):
    doc_type = normalize_document_type(document_type)
    category = safe_category(category)

    combined = f"{title} {text}".lower()

    if doc_type in ["Newspaper", "Magazine"]:
        if category == "Story":
            if any(w in combined for w in ["cricket", "slc", "match", "team", "sports", "icc"]):
                return "Sports"

            if any(w in combined for w in ["president", "minister", "parliament", "treasury", "court", "government", "copf"]):
                return "Politics"

            if any(w in combined for w in ["workers", "labour", "may day", "community", "rights"]):
                return "Social"

            return "General Text"

    return category


def predict_category_final(text, document_type, title=""):
    doc_type = normalize_document_type(document_type)

    cleaned_text = stronger_clean_text(
        text,
        document_type=doc_type
    )

    model_prediction = local_model_predict_category(cleaned_text)
    model_category = model_prediction.get("category", "General Text")
    model_confidence = model_prediction.get("confidence", 0.0)

    rule_category = document_type_category_hint(
        text=cleaned_text,
        document_type=doc_type,
        title=title
    )

    if rule_category is not None:
        final_category = rule_category
    else:
        final_category = model_category

    final_category = fix_category_for_document_type(
        category=final_category,
        document_type=doc_type,
        text=cleaned_text,
        title=title
    )

    return {
        "category": final_category,
        "confidence": model_confidence
    }


# ============================================================
# 8. LOCAL LLAMA THROUGH OLLAMA
# ============================================================

def ollama_generate(prompt, temperature=0.2, timeout=160):
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": LLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "top_p": 0.9
                }
            },
            timeout=timeout
        )

        if response.status_code != 200:
            return ""

        data = response.json()
        return data.get("response", "").strip()

    except Exception:
        return ""


def fallback_summary(text, max_sentences=2):
    text = clean_text_basic(text)

    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [
        s.strip()
        for s in sentences
        if len(s.strip().split()) >= 6
    ]

    if not sentences:
        return text[:400]

    return " ".join(sentences[:max_sentences])


def llama_reconstruct_text(ocr_text, title="", document_type="Newspaper"):
    text = clean_text_basic(ocr_text)

    if not text:
        return ""

    words = text.split()

    if len(words) < 25:
        return text

    if len(words) > 900:
        text = " ".join(words[:900])

    prompt = f"""
You are cleaning OCR text from a printed {document_type} article for a blind reading assistant.

Article title:
{title}

Task:
- Reconstruct the OCR text into clear readable article text.
- Fix broken words and line breaks when possible.
- Remove obvious OCR garbage, masthead text, page footer text, and unrelated fragments.
- Do not summarize.
- Do not add new information.
- Keep names, numbers, dates, places, and facts as much as possible.
- If a sentence is too damaged, omit it instead of guessing.
- Return only the cleaned article text.

OCR text:
{text}

Cleaned article text:
"""

    result = ollama_generate(prompt, temperature=0.1, timeout=180)
    result = clean_text_basic(result)

    if not result:
        return text

    if len(result.split()) < 10:
        return text

    return result


def llama_summary(text, document_type="General Document", category="General Text", max_words=90):
    text = clean_text_basic(text)

    if not text:
        return "Readable text was not found."

    words = text.split()

    if len(words) > 900:
        text = " ".join(words[:900])

    prompt = f"""
You are summarizing content for a blind user using a wearable reading assistant.

Document type: {document_type}
Category: {category}

Task:
- Give a clear and meaningful summary.
- Use simple language.
- Keep important names, places, numbers, and main idea.
- Do not add new information.
- Do not mention OCR or system processing.
- Do not include phrases like "Here is a summary".
- Maximum {max_words} words.

Text:
{text}

Summary:
"""

    summary = ollama_generate(prompt, temperature=0.2, timeout=180)
    summary = clean_text_basic(summary)

    if not summary:
        return fallback_summary(text, max_sentences=2)

    summary = re.sub(r"^(here is|here's).*?:\s*", "", summary, flags=re.IGNORECASE)

    return summary


def get_clean_final_text(item, doc_type):
    original_text = item.get("full_text", "")
    title = item.get("title", "Untitled article")

    cleaned_text = stronger_clean_text(
        original_text,
        document_type=doc_type
    )

    reconstructed_text = llama_reconstruct_text(
        ocr_text=cleaned_text,
        title=title,
        document_type=doc_type
    )

    reconstructed_text = stronger_clean_text(
        reconstructed_text,
        document_type=doc_type
    )

    if len(reconstructed_text.split()) < 8:
        return cleaned_text

    return reconstructed_text


# ============================================================
# 9. API REGION DETECTION ONLY
# ============================================================

def image_to_data_url(image_path):
    mime_type, _ = mimetypes.guess_type(image_path)

    if mime_type is None:
        mime_type = "image/jpeg"

    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode("utf-8")

    return f"data:{mime_type};base64,{encoded}"


def detect_reading_regions_once(image_path, document_type, retry=False):
    if client is None:
        raise ValueError(
            "OPENAI_API_KEY not found. Set your key before newspaper/magazine region detection."
        )

    doc_type = normalize_document_type(document_type)
    image_data_url = image_to_data_url(image_path)

    retry_instruction = ""

    if retry:
        retry_instruction = """
Extra instruction:
Your previous detection may have missed smaller articles.
This page likely contains many separate visible news articles.
Detect every visible headline and its article body separately, including small bottom/right articles.
Return more smaller article boxes instead of few large boxes.
"""

    prompt = f"""
You are detecting article reading regions in a printed document image.

Document type: {doc_type}

Very important:
- Return ONLY article/section bounding boxes.
- Do NOT extract article body text.
- Do NOT summarize.
- Do NOT classify category.
- Use tight bounding boxes around ONE article only.
- Ignore newspaper masthead/header/logo/date/page number/footer/advertisements/decorative areas.
- Do not include unrelated neighboring columns.
- Do not merge separate articles.
- If an article continues to another page, include only the visible part on this page.
- For each region, include only a short visible headline as title_hint.
- title_hint must be the article headline only, not body text.

For Newspaper or Magazine:
- Detect each visible article separately.
- Each bbox should include headline + related body text only.
- Avoid large boxes that contain multiple articles.
- Prefer more smaller article boxes over one large mixed box.

For Novel/Story, Textbook, or General Document:
- Return one full-page reading region.

Bounding boxes must be normalized from 0 to 1000:
x1, y1, x2, y2.

{retry_instruction}

Return JSON only.
"""

    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "document_type": {
                "type": "string"
            },
            "regions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "region_number": {
                            "type": "integer"
                        },
                        "region_type": {
                            "type": "string",
                            "enum": ["article", "section", "full_page"]
                        },
                        "title_hint": {
                            "type": "string"
                        },
                        "bbox": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "x1": {"type": "integer"},
                                "y1": {"type": "integer"},
                                "x2": {"type": "integer"},
                                "y2": {"type": "integer"}
                            },
                            "required": ["x1", "y1", "x2", "y2"]
                        },
                        "confidence_note": {
                            "type": "string"
                        }
                    },
                    "required": [
                        "region_number",
                        "region_type",
                        "title_hint",
                        "bbox",
                        "confidence_note"
                    ]
                }
            },
            "warnings": {
                "type": "array",
                "items": {
                    "type": "string"
                }
            }
        },
        "required": [
            "document_type",
            "regions",
            "warnings"
        ]
    }

    response = client.responses.create(
        model=VISION_MODEL,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": prompt
                    },
                    {
                        "type": "input_image",
                        "image_url": image_data_url,
                        "detail": "high"
                    }
                ]
            }
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "article_regions_only",
                "strict": True,
                "schema": schema
            }
        },
        max_output_tokens=5000,
        temperature=0
    )

    return json.loads(response.output_text)


def detect_reading_regions_uncached(image_path, document_type):
    result = detect_reading_regions_once(
        image_path=image_path,
        document_type=document_type,
        retry=False
    )

    doc_type = normalize_document_type(document_type)
    regions = result.get("regions", [])

    if (
        ALLOW_REGION_RETRY
        and doc_type in ["Newspaper", "Magazine"]
        and len(regions) < MIN_NEWSPAPER_REGIONS
    ):
        retry_result = detect_reading_regions_once(
            image_path=image_path,
            document_type=document_type,
            retry=True
        )

        retry_regions = retry_result.get("regions", [])

        if len(retry_regions) > len(regions):
            return retry_result

    return result


def detect_reading_regions_cached(image_path, document_type, force_refresh=False):
    if not force_refresh:
        cached = load_cached_regions(image_path, document_type)

        if cached is not None:
            cached["from_cache"] = True
            return cached

    result = detect_reading_regions_uncached(
        image_path=image_path,
        document_type=document_type
    )

    result["from_cache"] = False

    save_cached_regions(
        image_path=image_path,
        document_type=document_type,
        region_result=result
    )

    return result


def normalized_bbox_to_pixel(bbox, image_width, image_height, padding=4):
    x1 = int((bbox["x1"] / 1000) * image_width)
    y1 = int((bbox["y1"] / 1000) * image_height)
    x2 = int((bbox["x2"] / 1000) * image_width)
    y2 = int((bbox["y2"] / 1000) * image_height)

    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(image_width, x2 + padding)
    y2 = min(image_height, y2 + padding)

    if x2 <= x1:
        x2 = min(image_width, x1 + 20)

    if y2 <= y1:
        y2 = min(image_height, y1 + 20)

    return x1, y1, x2, y2


def bbox_area(box):
    x1, y1, x2, y2 = box
    return max(1, x2 - x1) * max(1, y2 - y1)


def bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter_w = max(0, ix2 - ix1)
    inter_h = max(0, iy2 - iy1)
    inter_area = inter_w * inter_h

    union_area = bbox_area(a) + bbox_area(b) - inter_area

    return inter_area / max(1, union_area)


def filter_duplicate_regions(regions_pixel):
    if not regions_pixel:
        return []

    regions_pixel = sorted(
        regions_pixel,
        key=lambda r: bbox_area(r["pixel_bbox"]),
        reverse=True
    )

    kept = []

    for region in regions_pixel:
        box = region["pixel_bbox"]
        duplicate = False

        for kept_region in kept:
            if bbox_iou(box, kept_region["pixel_bbox"]) > 0.85:
                duplicate = True
                break

        if not duplicate:
            kept.append(region)

    kept = sorted(
        kept,
        key=lambda r: (r["pixel_bbox"][1], r["pixel_bbox"][0])
    )

    for i, region in enumerate(kept, start=1):
        region["region_number"] = i

    return kept


def show_detected_regions(image_path, regions_pixel):
    img = cv2.imread(image_path)

    if img is None:
        return

    display = img.copy()

    for i, region in enumerate(regions_pixel, start=1):
        x1, y1, x2, y2 = region["pixel_bbox"]

        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 3)

        cv2.putText(
            display,
            str(i),
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    plt.figure(figsize=(12, 16))
    plt.imshow(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Detected Reading Regions")
    plt.show()


# ============================================================
# 10. TITLE + ARTICLE VALIDATION
# ============================================================

def clean_region_title(title):
    title = clean_text_basic(title)

    if not title:
        return ""

    if is_newspaper_masthead_or_header(title):
        return ""

    if is_garbage_text_line(title):
        return ""

    words = title.split()

    if len(words) > 18:
        title = " ".join(words[:18])

    return title.strip()


def get_title_from_text(text, document_type="Newspaper"):
    cleaned = clean_ocr_text(
        text,
        document_type=document_type,
        preserve_line_breaks=True
    )

    lines = [
        line.strip()
        for line in cleaned.split("\n")
        if line.strip()
    ]

    for line in lines[:5]:
        if looks_like_heading_line(line) and len(line.split()) >= 2:
            return " ".join(line.split()[:14])

    for line in lines:
        if len(line.split()) >= 3:
            return " ".join(line.split()[:14])

    return "Untitled article"


def get_title_from_text_or_hint(text, title_hint, document_type="Newspaper"):
    title_hint = clean_region_title(title_hint)

    if title_hint and len(title_hint.split()) >= 2:
        return title_hint

    return get_title_from_text(
        text=text,
        document_type=document_type
    )


def article_is_usable(text, title, document_type):
    doc_type = normalize_document_type(document_type)
    text = str(text).strip()
    title = str(title).strip()

    word_count = len(text.split())

    if word_count < 10:
        return False

    if doc_type in ["Newspaper", "Magazine"]:
        if is_garbage_text_line(title):
            return False

        if is_newspaper_masthead_or_header(title):
            return False

        if ocr_quality_score(text) < 0.20:
            return False

    return True


# ============================================================
# 11. BUILD ANALYSIS OBJECT
# ============================================================

def build_analysis_object(document_type, results, warnings=None):
    if warnings is None:
        warnings = []

    categories = sorted(set(r["category"] for r in results))
    category_counts = dict(Counter(r["category"] for r in results))

    return {
        "status": "ANALYZED",
        "document_type": document_type,
        "mode": "Multi Category" if len(categories) > 1 else "Single Category",
        "categories": categories,
        "category_counts": category_counts,
        "needs_category_selection": len(categories) > 1,
        "results": results,
        "warnings": warnings
    }


# ============================================================
# 12. PROCESSING FUNCTIONS
# ============================================================

def process_full_page_local(img, document_type="General Document", warnings=None, debug=False):
    if warnings is None:
        warnings = []

    cleaned_text = best_full_page_text(
        img,
        document_type=document_type,
        debug=debug
    )

    title = "Full page"

    prediction = predict_category_final(
        text=cleaned_text,
        document_type=document_type,
        title=title
    )

    result = {
        "block_id": "PAGE_FULL",
        "article_number": 1,
        "title": title,
        "bbox": None,
        "full_text": cleaned_text,
        "summary": "",
        "category": prediction["category"],
        "confidence": prediction["confidence"],
        "ocr_quality": ocr_quality_score(cleaned_text),
        "ocr_good": is_ocr_good(cleaned_text),
        "source": "full_page_tesseract_ocr"
    }

    return build_analysis_object(
        document_type=document_type,
        results=[result],
        warnings=warnings
    )


def process_newspaper_or_magazine(
    image_path,
    document_type="Newspaper",
    debug=False,
    force_refresh_regions=False
):
    doc_type = normalize_document_type(document_type)

    img = cv2.imread(image_path)

    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    image_height, image_width = img.shape[:2]

    region_result = detect_reading_regions_cached(
        image_path=image_path,
        document_type=doc_type,
        force_refresh=force_refresh_regions
    )

    raw_regions = region_result.get("regions", [])
    warnings = region_result.get("warnings", [])

    if region_result.get("from_cache"):
        warnings.append("Article regions loaded from cache for stable repeated results.")

    regions_pixel = []

    for region in raw_regions:
        pixel_bbox = normalized_bbox_to_pixel(
            bbox=region["bbox"],
            image_width=image_width,
            image_height=image_height,
            padding=4
        )

        if bbox_area(pixel_bbox) < (image_width * image_height * 0.003):
            continue

        regions_pixel.append({
            "region_number": region["region_number"],
            "region_type": region.get("region_type", "article"),
            "title_hint": region.get("title_hint", ""),
            "pixel_bbox": pixel_bbox,
            "confidence_note": region.get("confidence_note", "")
        })

    regions_pixel = filter_duplicate_regions(regions_pixel)

    all_lines = extract_full_page_ocr_lines(
        img,
        document_type=doc_type,
        debug=debug
    )

    if debug:
        print("\n========== DETECTED READING REGIONS ==========")
        print("Region count:", len(regions_pixel))
        print("Loaded from cache:", region_result.get("from_cache", False))

        for r in regions_pixel:
            print(
                r["region_number"],
                "|",
                r["region_type"],
                "|",
                r.get("title_hint", ""),
                "|",
                r["pixel_bbox"]
            )

        print("Full-page OCR line count:", len(all_lines))

        if regions_pixel and SHOW_REGION_IMAGE:
            show_detected_regions(image_path, regions_pixel)

    if not regions_pixel or not all_lines:
        return process_full_page_local(
            img=img,
            document_type=doc_type,
            warnings=warnings,
            debug=debug
        )

    results = []
    used_titles = set()

    for region in regions_pixel:
        bbox = region["pixel_bbox"]

        region_lines = get_region_lines(
            all_lines=all_lines,
            bbox=bbox
        )

        article_text = lines_to_article_text(
            lines=region_lines,
            bbox=bbox,
            document_type=doc_type
        )

        if len(article_text.split()) < 8:
            continue

        title = get_title_from_text_or_hint(
            text=article_text,
            title_hint=region.get("title_hint", ""),
            document_type=doc_type
        )

        title_key = title.lower().strip()

        if title_key in used_titles:
            continue

        used_titles.add(title_key)

        if not article_is_usable(article_text, title, doc_type):
            continue

        prediction = predict_category_final(
            text=article_text,
            document_type=doc_type,
            title=title
        )

        prediction["category"] = fix_category_for_document_type(
            category=prediction["category"],
            document_type=doc_type,
            text=article_text,
            title=title
        )

        results.append({
            "block_id": f"ARTICLE_{len(results) + 1}",
            "article_number": len(results) + 1,
            "title": title,
            "bbox": bbox,
            "full_text": article_text,
            "summary": "",
            "category": prediction["category"],
            "confidence": prediction["confidence"],
            "ocr_quality": ocr_quality_score(article_text),
            "ocr_good": is_ocr_good(article_text, min_words=10, min_score=0.20),
            "source": "cached_api_region_fullpage_tesseract_line_assignment"
        })

    if not results:
        return process_full_page_local(
            img=img,
            document_type=doc_type,
            warnings=warnings,
            debug=debug
        )

    return build_analysis_object(
        document_type=doc_type,
        results=results,
        warnings=warnings
    )


# ============================================================
# 13. MAIN ANALYSIS FUNCTION
# ============================================================

def analyze_content(image_path, document_type, debug=False, force_refresh_regions=False):
    doc_type = normalize_document_type(document_type)

    img = cv2.imread(image_path)

    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    if debug:
        print("Document type received:", document_type)
        print("Normalized document type:", doc_type)

    if is_multi_category_document(doc_type):
        analysis = process_newspaper_or_magazine(
            image_path=image_path,
            document_type=doc_type,
            debug=debug,
            force_refresh_regions=force_refresh_regions
        )
    else:
        analysis = process_full_page_local(
            img=img,
            document_type=doc_type,
            warnings=[],
            debug=debug
        )

    if debug:
        print("\n========== EXTRACTED FULL TEXT ==========")

        if SHOW_EXTRACTED_TEXT:
            if is_multi_category_document(doc_type):
                for r in analysis["results"]:
                    print("\n" + "=" * 70)
                    print(f"Article {r['article_number']}: {r['title']}")
                    print(r["full_text"])
            else:
                if analysis["results"]:
                    print(analysis["results"][0]["full_text"])

        print("\n========== DETECTED CATEGORIES ==========")
        print("Document Type:", analysis["document_type"])
        print("Mode:", analysis["mode"])
        print("Categories:", analysis["categories"])
        print("Category Counts:", analysis["category_counts"])

        if analysis["warnings"]:
            print("\nWarnings:")
            for w in analysis["warnings"]:
                print("-", w)

        print("\n========== ITEMS ==========")

        for r in analysis["results"]:
            print("\n" + "-" * 70)
            print("Item:", r["block_id"])
            print("Title:", r["title"])
            print("Final Category:", r["category"])
            print("OCR Quality:", r["ocr_quality"])

    return analysis


# ============================================================
# 14. OUTPUT GENERATION
# ============================================================

def generate_selected_output(analysis, selected_category, depth):
    selected_category = str(selected_category).strip()
    depth = str(depth).strip().lower()

    if depth not in ["summary", "full"]:
        return {
            "status": "ERROR",
            "message": "Depth must be summary or full."
        }

    if selected_category not in analysis["categories"]:
        return {
            "status": "ERROR",
            "message": f"Selected category '{selected_category}' not found.",
            "available_categories": analysis["categories"]
        }

    selected_items = [
        r for r in analysis["results"]
        if r["category"] == selected_category
    ]

    if not selected_items:
        return {
            "status": "ERROR",
            "message": "No content found for selected category.",
            "available_categories": analysis["categories"]
        }

    doc_type = analysis["document_type"]

    if doc_type in ["Newspaper", "Magazine"]:
        parts = []

        for index, item in enumerate(selected_items, start=1):

            clean_article_text = get_clean_final_text(
                item=item,
                doc_type=doc_type
            )

            if depth == "summary":
                content = llama_summary(
                    text=clean_article_text,
                    document_type=doc_type,
                    category=selected_category,
                    max_words=80
                )
            else:
                content = clean_article_text

            parts.append(
                f"Article {index}. {item['title']}. {content}"
            )

        intro = (
            f"{selected_category} section. "
            f"I found {len(selected_items)} article"
            f"{'s' if len(selected_items) != 1 else ''}."
        )

        final_text = intro + "\n\n" + "\n\n".join(parts)

    else:
        final_parts = []

        for item in selected_items:
            clean_page_text = get_clean_final_text(
                item=item,
                doc_type=doc_type
            )

            if depth == "summary":
                content = llama_summary(
                    text=clean_page_text,
                    document_type=doc_type,
                    category=selected_category,
                    max_words=100
                )
            else:
                content = clean_page_text

            final_parts.append(content)

        final_text = "\n\n".join(final_parts)

    return {
        "status": "READY",
        "document_type": doc_type,
        "selected_category": selected_category,
        "depth": depth,
        "final_output_text": final_text,
        "next_module_payload": {
            "text": final_text
        }
    }


def print_final_output(output):
    print("\n========== FINAL OUTPUT ==========")
    print("Status:", output["status"])

    if output["status"] != "READY":
        print("Message:", output.get("message", ""))

        if "available_categories" in output:
            print("Available Categories:", output["available_categories"])

        return

    print("Document Type:", output["document_type"])
    print("Selected Category:", output["selected_category"])
    print("Depth:", output["depth"])

    print("\nMeaningful Output Text:\n")
    print(output["final_output_text"])

    print("\n========== NEXT MODULE PAYLOAD ==========")
    print(output["next_module_payload"])


# ============================================================
# 15. INTERACTIVE TEST FLOW
# ============================================================

def run_full_test_flow(force_refresh_regions=False):
    image_path = input("Enter image path: ").strip()

    print("\nDocument type examples:")
    print("Newspaper, Magazine, Novel/Story, Textbook, General Document")

    document_type = input("Enter document type from friend's module: ").strip()

    analysis = analyze_content(
        image_path=image_path,
        document_type=document_type,
        debug=True,
        force_refresh_regions=force_refresh_regions
    )

    if not analysis["categories"]:
        print("No categories detected.")
        return analysis, None

    if len(analysis["categories"]) == 1:
        selected_category = analysis["categories"][0]
        print("\nSingle category detected:", selected_category)
        print("Category selection skipped.")
    else:
        print("\nAvailable categories:")
        for c in analysis["categories"]:
            print("-", c)

        while True:
            selected_category = input("\nSelect one category: ").strip()

            if selected_category in analysis["categories"]:
                break

            print("Invalid category. Please select from:")
            for c in analysis["categories"]:
                print("-", c)

    while True:
        depth = input("Do you want summary or full text? Type 'summary' or 'full': ").strip().lower()

        if depth in ["summary", "full"]:
            break

        print("Invalid input. Type only 'summary' or 'full'.")

    output = generate_selected_output(
        analysis=analysis,
        selected_category=selected_category,
        depth=depth
    )

    print_final_output(output)

    return analysis, output
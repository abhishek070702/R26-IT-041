"""
Optional PaddleOCR title reader for the Abhishek PP2 pipeline.

Used when TITLE_READER_MODEL=paddleocr.
EasyOCR remains the default and fallback (title_reader.py is unchanged).
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
from PIL import Image

from backend.document_identification.title_reader import (
    NEWSPAPER_NAME_FALLBACK,
    NEWSPAPER_PATTERNS,
    apply_known_title_correction,
    clean_easyocr_title_text,
    collapse_repeated_words,
    compose_known_title_from_tokens,
    detect_known_easyocr_title,
    finalize_cover_title,
    is_bad_newspaper_masthead,
    is_barcode_or_sticker_text,
    is_long_author_or_subtitle_fragment,
    is_publisher_only_line,
    looks_like_person_name,
    match_newspaper_from_tokens,
    nice_title_case,
    split_glued_masthead_text,
    strip_cover_noise_words,
    _masthead_lexicon_words,
)

logger = logging.getLogger(__name__)

_paddle_ocr = None
_paddle_load_error: str | None = None

MAGAZINE_MASTHEAD_TOKENS = {
    "TIME",
    "FORBES",
    "SHAPE",
    "VOGUE",
    "ELLE",
    "WIRED",
    "GQ",
    "ECONOMIST",
    "FORTUNE",
    "PEOPLE",
    "LMD",
    "PULSE",
    "ALLURE",
    "GLAMOUR",
    "INSTYLE",
    "ESQUIRE",
}


def _paddleocr_version() -> str:
    try:
        import paddleocr

        version = getattr(paddleocr, "__version__", None)
        if version:
            return str(version)
    except Exception:
        pass
    try:
        from importlib.metadata import version as pkg_version

        return str(pkg_version("paddleocr"))
    except Exception:
        return "unknown"


def _load_paddleocr():
    """
    Lazy-load PaddleOCR once using PaddleOCR 3.x-compatible args.

    Prefer CPU for pipeline stability. Raises RuntimeError on failure so the
    pipeline can fall back to EasyOCR.
    """
    global _paddle_ocr, _paddle_load_error

    if _paddle_ocr is not None:
        return _paddle_ocr

    try:
        from paddleocr import PaddleOCR

        version = _paddleocr_version()
        print(f"[title_reader] PaddleOCR version={version}")
        logger.info("PaddleOCR version=%s", version)

        init_attempts = (
            ("v3_device_cpu", {"lang": "en", "device": "cpu"}),
            ("v3_device_cpu_minimal", {"device": "cpu"}),
        )

        last_error: Exception | None = None
        for mode, kwargs in init_attempts:
            try:
                print(f"[title_reader] PaddleOCR init mode={mode}")
                logger.info("PaddleOCR init mode=%s kwargs=%s", mode, kwargs)
                _paddle_ocr = PaddleOCR(**kwargs)
                print("[title_reader] PaddleOCR loaded successfully")
                logger.info("PaddleOCR loaded successfully mode=%s", mode)
                _paddle_load_error = None
                return _paddle_ocr
            except Exception as exc:
                last_error = exc
                print(
                    f"[title_reader] PaddleOCR init mode={mode} failed: {exc}"
                )
                logger.warning(
                    "PaddleOCR init mode=%s failed: %s",
                    mode,
                    exc,
                )

        raise RuntimeError(
            f"Failed to load PaddleOCR with compatible CPU settings: {last_error}"
        )
    except Exception as exc:
        _paddle_ocr = None
        _paddle_load_error = f"Failed to load PaddleOCR: {exc}"
        print(f"[title_reader] PaddleOCR load failed: {_paddle_load_error}")
        logger.exception("PaddleOCR load failed")
        raise RuntimeError(_paddle_load_error) from exc


def _run_paddle_ocr(image_np: np.ndarray) -> list[Any]:
    """
    Run PaddleOCR and normalize to a list of detection rows.
    Supports both classic ocr() and newer predict() APIs.
    """
    ocr = _load_paddleocr()
    result = None

    if hasattr(ocr, "ocr"):
        try:
            result = ocr.ocr(image_np, cls=True)
        except TypeError:
            result = ocr.ocr(image_np)
    elif hasattr(ocr, "predict"):
        result = ocr.predict(image_np)
    else:
        raise RuntimeError("Unsupported PaddleOCR API (no ocr/predict)")

    if result is None:
        return []

    # Classic: [ [ [box], (text, conf) ], ... ] or [[...]] wrapped
    if isinstance(result, list) and result and isinstance(result[0], list):
        first = result[0]
        if first and isinstance(first, list) and first and isinstance(first[0], (list, tuple)):
            # Could be page wrapper [[detections]] or detections themselves.
            sample = first[0]
            if (
                isinstance(sample, (list, tuple))
                and len(sample) >= 2
                and isinstance(sample[0], (list, tuple))
            ):
                return first
            return result

    # predict() may return list of dict-like pages.
    if isinstance(result, list) and result and isinstance(result[0], dict):
        page = result[0]
        texts = page.get("rec_texts") or page.get("texts") or []
        scores = page.get("rec_scores") or page.get("scores") or []
        boxes = page.get("dt_polys") or page.get("rec_polys") or page.get("boxes") or []
        rows = []
        for idx, text in enumerate(texts):
            conf = float(scores[idx]) if idx < len(scores) else 0.0
            box = boxes[idx] if idx < len(boxes) else [[0, 0], [1, 0], [1, 1], [0, 1]]
            rows.append([box, (str(text), conf)])
        return rows

    if isinstance(result, list):
        return result
    return []


def _parse_paddle_lines(
    raw_rows: list[Any],
    page_width: float,
    page_height: float,
    y_offset: float = 0.0,
) -> list[dict]:
    lines: list[dict] = []
    for row in raw_rows or []:
        try:
            if not row or len(row) < 2:
                continue
            bbox = row[0]
            payload = row[1]
            if isinstance(payload, (list, tuple)) and len(payload) >= 2:
                raw_text = str(payload[0]).strip()
                conf = float(payload[1])
            elif isinstance(payload, str):
                raw_text = payload.strip()
                conf = 0.0
            else:
                continue
            if not raw_text:
                continue

            points = np.array(bbox, dtype=float)
            x_min = float(points[:, 0].min())
            x_max = float(points[:, 0].max())
            y_min = float(points[:, 1].min()) + float(y_offset)
            y_max = float(points[:, 1].max()) + float(y_offset)
            cleaned = clean_easyocr_title_text(raw_text)
            if not cleaned:
                continue

            lines.append(
                {
                    "raw": raw_text,
                    "text": cleaned,
                    "confidence": conf,
                    "height": max(1.0, y_max - y_min),
                    "width": max(1.0, x_max - x_min),
                    "x_min": x_min,
                    "x_max": x_max,
                    "y_min": y_min,
                    "y_max": y_max,
                    "y_center_ratio": ((y_min + y_max) / 2.0) / max(page_height, 1.0),
                    "x_center_ratio": ((x_min + x_max) / 2.0) / max(page_width, 1.0),
                    "word_count": len(cleaned.split()),
                }
            )
        except Exception:
            logger.debug("skip malformed PaddleOCR row=%r", row, exc_info=True)
            continue
    return lines


def _crop_region(image: Image.Image, document_type: str) -> tuple[Image.Image, float]:
    """Return (crop, y_offset_in_full_page)."""
    width, height = image.size
    if document_type == "Newspaper":
        top = 0
        bottom = max(1, int(height * 0.25))
        return image.crop((0, top, width, bottom)), float(top)
    if document_type == "Magazine":
        top = 0
        bottom = max(1, int(height * 0.55))
        return image.crop((0, top, width, bottom)), float(top)
    if document_type in {"Report", "Printed_Letter"}:
        top = 0
        bottom = max(1, int(height * 0.35))
        return image.crop((0, top, width, bottom)), float(top)
    # Novel: upper/center cover band
    top = int(height * 0.08)
    bottom = int(height * 0.78)
    left = int(width * 0.04)
    right = int(width * 0.96)
    return image.crop((left, top, right, max(top + 1, bottom))), float(top)


def _score_novel_line(item: dict) -> float:
    y_ratio = float(item.get("y_center_ratio", 0.5))
    x_ratio = float(item.get("x_center_ratio", 0.5))
    size_score = float(item.get("height", 0.0)) * 5.0
    if 0.18 <= y_ratio <= 0.65:
        band_score = 40.0
    elif 0.12 <= y_ratio < 0.18:
        band_score = 8.0
    elif y_ratio < 0.12:
        band_score = -20.0
    elif y_ratio > 0.78:
        band_score = -50.0
    else:
        band_score = 5.0
    center_score = (1.0 - abs(x_ratio - 0.5) * 2.0) * 18.0
    conf_score = float(item.get("confidence", 0.0)) * 12.0
    name_penalty = -35.0 if looks_like_person_name(item.get("text", "")) else 0.0
    return size_score + band_score + center_score + conf_score + name_penalty


def _reject_novel_line(text: str) -> bool:
    if is_barcode_or_sticker_text(text):
        return True
    if is_publisher_only_line(text):
        return True
    if is_long_author_or_subtitle_fragment(text):
        return True
    if looks_like_person_name(text):
        return True
    return False


def _select_novel_title(lines: list[dict]) -> str | None:
    candidates = []
    for item in lines:
        text = item["text"]
        if _reject_novel_line(text):
            continue
        cleaned = strip_cover_noise_words(text)
        if not cleaned or _reject_novel_line(cleaned):
            continue
        scored = dict(item)
        scored["text"] = cleaned
        scored["cover_score"] = _score_novel_line(item)
        candidates.append(scored)

    if not candidates:
        return None

    ranked = sorted(
        candidates,
        key=lambda x: (float(x.get("height", 0.0)), float(x.get("cover_score", 0.0))),
        reverse=True,
    )
    # Prefer known titles from any strong candidate.
    for item in ranked:
        known = apply_known_title_correction(item["text"], "Novel")
        if known:
            return known
        finalized = finalize_cover_title(item["text"], "Novel")
        if finalized:
            return finalized

    blob = " ".join(item["text"] for item in ranked[:4])
    known = apply_known_title_correction(blob, "Novel")
    if known:
        return known
    composed = compose_known_title_from_tokens(blob.split(), "Novel")
    if composed:
        return composed

    best = ranked[0]["text"]
    finalized = finalize_cover_title(best, "Novel")
    return finalized


def _is_magazine_story_line(text: str) -> bool:
    """Reject long cover-story / article teaser lines."""
    cleaned = clean_easyocr_title_text(text)
    words = cleaned.split()
    if len(words) >= 5:
        return True
    if any(tok in cleaned for tok in ("EXCLUSIVE", "SPECIAL REPORT", "HOW TO", "PLUS")):
        return True
    return False


def _select_magazine_title(lines: list[dict]) -> str | None:
    # Prefer short masthead / brand lines in the upper band.
    upper = [item for item in lines if float(item.get("y_center_ratio", 1.0)) <= 0.40]
    pool = upper if upper else lines

    ranked = sorted(
        pool,
        key=lambda x: (
            float(x.get("height", 0.0)),
            1.0 - float(x.get("y_center_ratio", 1.0)),
            float(x.get("confidence", 0.0)),
        ),
        reverse=True,
    )

    for item in ranked:
        text = strip_cover_noise_words(item["text"])
        if not text:
            continue
        if is_barcode_or_sticker_text(text) or is_publisher_only_line(text):
            continue
        if _is_magazine_story_line(text):
            continue
        known = detect_known_easyocr_title(text, "Magazine")
        if known:
            return known
        words = set(text.split())
        for token in MAGAZINE_MASTHEAD_TOKENS:
            if token in words or token in text.replace(" ", ""):
                known = detect_known_easyocr_title(token, "Magazine")
                if known:
                    return known
                return nice_title_case(token)

    # Token bag across short upper lines.
    token_bag: list[str] = []
    for item in ranked:
        if item.get("word_count", 99) > 3:
            continue
        token_bag.extend(item["text"].split())
    known = compose_known_title_from_tokens(token_bag, "Magazine")
    if known:
        return known

    for item in ranked:
        text = strip_cover_noise_words(item["text"])
        if not text or _is_magazine_story_line(text):
            continue
        finalized = finalize_cover_title(text, "Magazine")
        if finalized:
            return finalized
    return None


def _select_newspaper_title(lines: list[dict]) -> str | None:
    top_band = [
        item for item in lines if float(item.get("y_center_ratio", 1.0)) <= 0.28
    ]
    if not top_band:
        top_band = lines
    strict_top = [
        item for item in top_band if float(item.get("y_center_ratio", 1.0)) <= 0.16
    ]
    search_pool = strict_top if strict_top else top_band

    candidates = []
    for item in search_pool:
        text = item["text"]
        if item.get("word_count", 99) > 4:
            continue
        if is_bad_newspaper_masthead(text):
            continue
        top_score = 1.0 - float(item.get("y_center_ratio", 1.0))
        left_score = 1.0 - float(item.get("x_center_ratio", 0.5))
        score = (
            float(item.get("height", 0.0)) * 2.5
            + top_score * 40
            + left_score * 12
            + float(item.get("confidence", 0.0)) * 20
        )
        scored = dict(item)
        scored["masthead_score"] = score
        candidates.append(scored)

    for item in sorted(candidates, key=lambda x: x["masthead_score"], reverse=True):
        known = detect_known_easyocr_title(item["text"], "Newspaper")
        if known:
            return known
        # Direct NEWSPAPER_PATTERNS check via clean text.
        cleaned = clean_easyocr_title_text(item["text"])
        if cleaned in NEWSPAPER_PATTERNS:
            return NEWSPAPER_PATTERNS[cleaned]

    token_bag: list[str] = []
    for item in sorted(search_pool, key=lambda x: x.get("y_min", 0)):
        if item.get("word_count", 99) > 4:
            continue
        token_bag.extend(split_glued_masthead_text(item["text"]))
        token_bag.extend(item["text"].split())
    known = match_newspaper_from_tokens(token_bag)
    if known:
        return known

    short_top = sorted(
        [c for c in candidates if c.get("word_count", 99) <= 3],
        key=lambda x: (x.get("y_min", 0), x.get("x_min", 0)),
    )[:2]
    if short_top:
        joined = " ".join(item["text"] for item in short_top)
        known = detect_known_easyocr_title(joined, "Newspaper")
        if known:
            return known
        known = match_newspaper_from_tokens(
            split_glued_masthead_text(joined) + joined.split()
        )
        if known:
            return known

    if candidates:
        best = sorted(candidates, key=lambda x: x["masthead_score"], reverse=True)[0]
        pieces = split_glued_masthead_text(best["text"])
        known = match_newspaper_from_tokens(pieces)
        if known:
            return known
        lexicon = set(_masthead_lexicon_words())
        cleaned_words = [w for w in pieces if w in lexicon][:4]
        if cleaned_words:
            return nice_title_case(" ".join(cleaned_words))

    return None


def _select_strict_heading(lines: list[dict]) -> str | None:
    """Report / Printed_Letter: only clear short heading near the top."""
    upper = [
        item
        for item in lines
        if float(item.get("y_center_ratio", 1.0)) <= 0.22
        and int(item.get("word_count", 99)) <= 8
    ]
    if not upper:
        return None
    ranked = sorted(
        upper,
        key=lambda x: (
            float(x.get("height", 0.0)),
            float(x.get("confidence", 0.0)),
        ),
        reverse=True,
    )
    for item in ranked[:3]:
        text = collapse_repeated_words(strip_cover_noise_words(item["text"]))
        if not text:
            continue
        if is_barcode_or_sticker_text(text) or is_publisher_only_line(text):
            continue
        if looks_like_person_name(text):
            continue
        words = text.split()
        if len(words) < 2:
            continue
        # Require stronger confidence / size for forcing a report/letter title.
        if float(item.get("confidence", 0.0)) < 0.75 and float(item.get("height", 0.0)) < 28:
            continue
        return nice_title_case(text)
    return None


def paddleocr_extract_title(image_path: str, document_type: str) -> str | None:
    """
    Extract title using PaddleOCR with document-type-aware selection.

    Returns:
        title string, or None when extraction fails / no clear title.
    Raises:
        RuntimeError if PaddleOCR cannot be loaded or OCR crashes hard.
    """
    if document_type not in {
        "Novel",
        "Magazine",
        "Newspaper",
        "Report",
        "Printed_Letter",
    }:
        return None

    if not image_path or not os.path.isfile(image_path):
        print(f"[title_reader] paddleocr skipped: missing image path={image_path!r}")
        return None

    image = Image.open(image_path).convert("RGB")
    page_width, page_height = image.size
    crop, y_offset = _crop_region(image, document_type)
    crop_np = np.array(crop)

    raw_rows = _run_paddle_ocr(crop_np)
    lines = _parse_paddle_lines(
        raw_rows,
        page_width=float(page_width),
        page_height=float(page_height),
        y_offset=y_offset,
    )
    raw_preview = [item.get("raw", item.get("text", "")) for item in lines[:20]]
    print(f"[title_reader] paddleocr raw lines={raw_preview!r}")
    logger.info("paddleocr raw lines=%r", raw_preview)

    if not lines:
        print("[title_reader] paddleocr selected title=None")
        return None

    selected: str | None = None
    if document_type == "Novel":
        selected = _select_novel_title(lines)
    elif document_type == "Magazine":
        selected = _select_magazine_title(lines)
    elif document_type == "Newspaper":
        selected = _select_newspaper_title(lines)
        if not selected:
            selected = NEWSPAPER_NAME_FALLBACK
    elif document_type in {"Report", "Printed_Letter"}:
        selected = _select_strict_heading(lines)

    print(f"[title_reader] paddleocr selected title={selected!r}")
    logger.info("paddleocr selected title=%r", selected)
    return selected


print("PaddleOCR title reader module loaded (lazy model load on first use)")

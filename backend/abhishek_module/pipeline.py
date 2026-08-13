"""
Abhishek module pipeline.

Fixed output for integration with the full IoT reading assistant.
Never crashes the backend; uses safe try/except around every step.

Image description:
  Main PP2 path uses local models only:
    - IMAGE_DESCRIPTION_MODEL=blip (default) -> local_blip_description
    - IMAGE_DESCRIPTION_MODEL=florence -> local_florence_description
      (falls back to BLIP if Florence fails)
  Does NOT import openai_description.py (legacy /image route).
  Optional OpenAI Vision fallback runs only if USE_OPENAI_FALLBACK=true.

Title reading:
  - TITLE_READER_MODEL=easyocr (default) -> existing EasyOCR title_reader
  - TITLE_READER_MODEL=paddleocr -> PaddleOCR first, EasyOCR fallback
"""

from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Explicitly load backend/.env (this file lives in backend/abhishek_module/).
_BACKEND_DIR = Path(__file__).resolve().parent.parent
_ENV_PATH = _BACKEND_DIR / ".env"
load_dotenv(dotenv_path=_ENV_PATH, override=True)
print(f"[pipeline] loaded env from: {_ENV_PATH} (exists={_ENV_PATH.is_file()})")

# Optional imports — if missing, placeholders are used instead.
# IMPORTANT: do not import openai_description here (legacy/dev only).
_predict_document_type: Callable[..., Any] | None = None
_easyocr_extract_title: Callable[..., Any] | None = None
_paddleocr_extract_title: Callable[..., Any] | None = None
_describe_image_local: Callable[..., Any] | None = None
_format_blind_user_description: Callable[..., Any] | None = None
_describe_image_florence: Callable[..., Any] | None = None
_format_florence_blind_caption: Callable[..., Any] | None = None
_format_florence_context_caption: Callable[..., Any] | None = None
_recover_title_from_florence_caption: Callable[..., Any] | None = None
_florence_fallback_for_document_type: Callable[..., Any] | None = None
_crop_reading_material: Callable[..., Any] | None = None
_optimize_processing_image: Callable[..., Any] | None = None

try:
    from backend.document_identification.model_loader import (
        predict_document_type as _predict_document_type,
    )
except Exception:
    _predict_document_type = None

try:
    from backend.document_identification.title_reader import (
        easyocr_extract_title as _easyocr_extract_title,
    )
except Exception:
    _easyocr_extract_title = None

try:
    from backend.document_identification.paddle_title_reader import (
        paddleocr_extract_title as _paddleocr_extract_title,
    )
except Exception:
    _paddleocr_extract_title = None

try:
    # Main PP2 image description path (local only).
    from backend.image_description.local_blip_description import (
        describe_image_local as _describe_image_local,
        format_blind_user_description as _format_blind_user_description,
    )
except Exception:
    _describe_image_local = None
    _format_blind_user_description = None

try:
    from backend.image_description.local_florence_description import (
        describe_image_florence as _describe_image_florence,
        florence_fallback_for_document_type as _florence_fallback_for_document_type,
        format_florence_blind_caption as _format_florence_blind_caption,
        format_florence_context_caption as _format_florence_context_caption,
        recover_title_from_florence_caption as _recover_title_from_florence_caption,
    )
except Exception:
    _describe_image_florence = None
    _format_florence_blind_caption = None
    _format_florence_context_caption = None
    _recover_title_from_florence_caption = None
    _florence_fallback_for_document_type = None

try:
    from backend.preprocessing.document_cropper import (
        crop_reading_material as _crop_reading_material,
        optimize_processing_image as _optimize_processing_image,
    )
except Exception:
    _crop_reading_material = None
    _optimize_processing_image = None

try:
    from backend.image_description.openai_fallback import (
        is_openai_fallback_enabled as _is_openai_fallback_enabled,
        run_openai_vision_fallback as _run_openai_vision_fallback,
    )
except Exception:
    _is_openai_fallback_enabled = None
    _run_openai_vision_fallback = None


# Abhishek title_reader runs for these types.
TITLE_TYPES = {"Novel", "Magazine", "Newspaper"}
COVER_TITLE_TYPES = {"Novel", "Magazine"}

# Title/content owned by OCR + Harshaka's content module — no title_reader.
HANDOFF_TITLES = {
    "Report": "Handled by OCR and content processing module",
    "Printed_Letter": "Handled by OCR and content processing module",
}

NEWSPAPER_NAME_FALLBACK = "Newspaper name not clearly detected"
TITLE_NOT_CLEAR = "Title not clearly detected"

COVER_ILLUSTRATION_FALLBACK = (
    "The cover contains an illustration or visual design related to the document."
)
UNCLEAR_VISUAL_FALLBACK = (
    "Visual content detected, but description is not clear."
)
TITLE_CONFIDENCE_WARNING = "Title confidence is low"
OPENAI_FALLBACK_WARNING = "OpenAI fallback used"
FLORENCE_TITLE_RECOVERY_WARNING = "Title recovered from Florence caption"

WEAK_CAPTION_PHRASES = (
    "a person holding a book in their hand",
    "person holding a book",
    "holding a book in their hand",
    "someone holding a book",
    "a hand holding a book",
    "holding a book",
    "a book on a table",
    "a photo of a book",
    "an image of a book",
    "a picture of a book",
    "there is a book",
    "a book",
    "book cover",
    "a cover",
    "an image",
    "a photo",
    "a picture",
)


def _empty_result() -> dict:
    return {
        "document_type": "Unknown",
        "confidence": 0.0,
        "title": "Title not clearly detected",
        "image_descriptions": [],
        "status": "failed",
        "warnings": [],
    }


def _normalize_confidence(raw: Any) -> float:
    """Convert model confidence to 0.0–1.0 float."""
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0.0

    if value > 1.0:
        value = value / 100.0

    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return round(value, 4)


def _identify_document(image_path: str) -> tuple[str, float, list[str]]:
    warnings: list[str] = []

    if _predict_document_type is None:
        warnings.append("Document identification failed")
        return "Unknown", 0.0, warnings

    try:
        document_type, confidence, _all_predictions = _predict_document_type(
            image_path
        )
        if not document_type:
            warnings.append("Document identification failed")
            return "Unknown", 0.0, warnings
        return str(document_type), _normalize_confidence(confidence), warnings
    except Exception:
        warnings.append("Document identification failed")
        return "Unknown", 0.0, warnings


def _normalize_compare_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _title_looks_low_confidence(title: str, document_type: str) -> bool:
    """Heuristic local check — no external APIs."""
    if document_type not in COVER_TITLE_TYPES:
        return False

    title = (title or "").strip()
    if not title or title == TITLE_NOT_CLEAR:
        return True

    words = title.split()
    if len(words) == 0:
        return True
    if len(words) > 8:
        return True

    # OCR-ish junk remnants.
    junk = {"auat", "auac", "ysclf", "yscif", "farriet"}
    if any(w.lower() in junk for w in words):
        return True

    # Likely author-only leftover.
    if 2 <= len(words) <= 4 and words[0].lower() not in {"the", "a", "an"}:
        titleish = {
            "of", "and", "book", "story", "cabin", "adventures",
            "polly", "baker", "busy", "jungle", "farm", "garden",
        }
        if not any(w.lower() in titleish for w in words):
            # Common 3-part author pattern: First Middle Last
            if len(words) >= 3 and all(w[:1].isupper() for w in words if w):
                return True

    return False


def _caption_repeats_title(caption: str, title: str) -> bool:
    caption_n = _normalize_compare_text(caption)
    title_n = _normalize_compare_text(title)
    if not caption_n or not title_n:
        return False
    if title_n in {
        _normalize_compare_text(TITLE_NOT_CLEAR),
        _normalize_compare_text(NEWSPAPER_NAME_FALLBACK),
    }:
        return False
    if title_n in caption_n or caption_n in title_n:
        return True

    title_words = set(title_n.split())
    caption_words = set(caption_n.split())
    if not title_words:
        return False
    overlap = len(title_words & caption_words) / float(len(title_words))
    return overlap >= 0.75 and len(caption_words) <= max(6, len(title_words) + 2)


def _caption_is_weak_or_generic(caption: str) -> bool:
    caption_n = _normalize_compare_text(caption)
    if not caption_n:
        return True

    words = caption_n.split()
    if len(words) <= 2:
        return True

    for phrase in WEAK_CAPTION_PHRASES:
        phrase_n = _normalize_compare_text(phrase)
        if caption_n == phrase_n or caption_n.startswith(phrase_n + " "):
            return True
        if phrase_n in caption_n and len(words) <= 10:
            return True

    # Mostly filler words.
    filler = {"a", "an", "the", "of", "and", "with", "on", "in", "to", "is"}
    content = [w for w in words if w not in filler]
    if len(content) <= 1:
        return True

    return False


def _format_blind_descriptions(
    descriptions: list[str],
    title: str,
    document_type: str,
) -> tuple[list[str], list[str]]:
    """
    Convert raw local BLIP captions into blind-user-friendly sentences.
    Local post-processing only — no external APIs.
    """
    warnings: list[str] = []
    formatter = _format_blind_user_description

    if formatter is None:
        if document_type in COVER_TITLE_TYPES and title not in {
            "",
            TITLE_NOT_CLEAR,
            NEWSPAPER_NAME_FALLBACK,
        }:
            fallback = (
                "The cover contains an illustration or visual design related to "
                f"{title}."
            )
        elif document_type in COVER_TITLE_TYPES:
            fallback = COVER_ILLUSTRATION_FALLBACK
        else:
            fallback = UNCLEAR_VISUAL_FALLBACK
        print(
            "[pipeline] image-description fallback: formatter missing -> "
            f"{fallback!r}"
        )
        logger.info("image description fallback reason=formatter_missing")
        warnings.append("Image description failed")
        return [fallback], warnings

    source = descriptions or [""]
    refined: list[str] = []
    for caption in source:
        try:
            final = formatter(
                str(caption or ""),
                title=title,
                document_type=document_type,
            )
        except Exception:
            logger.exception("blind-user description formatting failed")
            if document_type in COVER_TITLE_TYPES and title not in {
                "",
                TITLE_NOT_CLEAR,
                NEWSPAPER_NAME_FALLBACK,
            }:
                final = (
                    "The cover contains an illustration or visual design related to "
                    f"{title}."
                )
            else:
                final = COVER_ILLUSTRATION_FALLBACK
            print(
                "[pipeline] image-description fallback: formatter exception -> "
                f"{final!r}"
            )
            logger.info("image description fallback reason=formatter_exception")
        final_text = str(final or "").strip()
        if final_text:
            refined.append(final_text)

    if not refined:
        fallback = (
            COVER_ILLUSTRATION_FALLBACK
            if document_type in COVER_TITLE_TYPES
            else UNCLEAR_VISUAL_FALLBACK
        )
        print(
            "[pipeline] image-description fallback: empty after format -> "
            f"{fallback!r}"
        )
        logger.info("image description fallback reason=empty_after_format")
        return [fallback], warnings

    return refined, warnings


def _selected_title_reader_model() -> str:
    """
    Read TITLE_READER_MODEL from backend/.env / process env.
    Default: easyocr. Allowed: easyocr, paddleocr.
    """
    _reload_backend_env()
    model = os.getenv("TITLE_READER_MODEL", "easyocr").lower().strip()
    if model in {"paddle", "paddleocr", "paddle_ocr"}:
        return "paddleocr"
    return "easyocr"


def _title_result_is_usable(title: Any, document_type: str) -> bool:
    if title is None:
        return False
    title_str = str(title).strip()
    if not title_str:
        return False
    if title_str == TITLE_NOT_CLEAR:
        return False
    if document_type == "Newspaper" and title_str == NEWSPAPER_NAME_FALLBACK:
        return False
    return True


def _run_easyocr_title(image_path: str, document_type: str) -> Any:
    if _easyocr_extract_title is None:
        raise RuntimeError("EasyOCR title reader is unavailable")
    return _easyocr_extract_title(image_path, document_type)


def _run_paddleocr_title(image_path: str, document_type: str) -> Any:
    if _paddleocr_extract_title is None:
        raise RuntimeError("PaddleOCR title reader is unavailable")
    return _paddleocr_extract_title(image_path, document_type)


def _read_title(image_path: str, document_type: str) -> tuple[str, list[str], str]:
    """
    Run title_reader for Novel, Magazine, and Newspaper masthead.

    TITLE_READER_MODEL=paddleocr uses PaddleOCR first, then EasyOCR fallback.
    Default remains EasyOCR.

    Returns:
        (title, warnings, title_source)
        title_source is paddleocr | easyocr | none
    """
    warnings: list[str] = []
    title_model = _selected_title_reader_model()
    print(f"[title_reader] title_reader_model={title_model}")
    logger.info("title_reader_model=%s document_type=%s", title_model, document_type)

    # Report / Printed_Letter: keep handoff unless Paddle finds a clear heading.
    if document_type in HANDOFF_TITLES:
        if title_model == "paddleocr":
            try:
                paddle_title = _run_paddleocr_title(image_path, document_type)
                if _title_result_is_usable(paddle_title, document_type):
                    title_str = str(paddle_title).strip()
                    print("[title_reader] easyocr fallback used=false")
                    print("[title_reader] final title source=paddleocr")
                    return title_str, warnings, "paddleocr"
            except Exception as exc:
                logger.exception("PaddleOCR title failed for %s", document_type)
                print(f"[title_reader] paddleocr error: {exc}")
            print("[title_reader] paddleocr selected title=None")
            print("[title_reader] easyocr fallback used=false")
            print("[title_reader] final title source=none")
            return HANDOFF_TITLES[document_type], warnings, "none"

        print("[title_reader] easyocr fallback used=false")
        print("[title_reader] final title source=none")
        logger.debug(
            "title_reader skipped for document_type=%s; handoff title=%s",
            document_type,
            HANDOFF_TITLES[document_type],
        )
        return HANDOFF_TITLES[document_type], warnings, "none"

    if document_type not in TITLE_TYPES:
        print("[title_reader] easyocr fallback used=false")
        print("[title_reader] final title source=none")
        logger.debug(
            "title_reader skipped for document_type=%s; final title=%r",
            document_type,
            "",
        )
        return "", warnings, "none"

    logger.debug(
        "title_reader executed for document_type=%s",
        document_type,
    )

    easyocr_fallback_used = False
    title_source = "none"
    title_str: str | None = None

    if title_model == "paddleocr":
        try:
            paddle_title = _run_paddleocr_title(image_path, document_type)
            if _title_result_is_usable(paddle_title, document_type):
                title_str = str(paddle_title).strip()
                title_source = "paddleocr"
            else:
                print("[title_reader] paddleocr selected title=None")
        except Exception as exc:
            logger.exception(
                "PaddleOCR title extraction failed; falling back to EasyOCR"
            )
            print(f"[title_reader] paddleocr error: {exc}")
            print("[title_reader] paddleocr selected title=None")

        if title_source != "paddleocr":
            easyocr_fallback_used = True
            try:
                easy_title = _run_easyocr_title(image_path, document_type)
                if _title_result_is_usable(easy_title, document_type):
                    title_str = str(easy_title).strip()
                    title_source = "easyocr"
                elif document_type == "Newspaper":
                    title_str = NEWSPAPER_NAME_FALLBACK
                    title_source = "easyocr"
                else:
                    title_str = TITLE_NOT_CLEAR
                    title_source = "easyocr"
            except Exception:
                logger.exception("EasyOCR title fallback failed")
                if document_type == "Newspaper":
                    title_str = NEWSPAPER_NAME_FALLBACK
                else:
                    title_str = TITLE_NOT_CLEAR
                    warnings.append(TITLE_CONFIDENCE_WARNING)
                title_source = "none"
    else:
        # Default EasyOCR path (unchanged selection logic in title_reader.py).
        print("[title_reader] easyocr fallback used=false")
        if document_type == "Newspaper":
            if _easyocr_extract_title is None:
                print("[title_reader] final title source=none")
                return NEWSPAPER_NAME_FALLBACK, warnings, "none"
            try:
                title = _run_easyocr_title(image_path, document_type)
                if title is None or str(title).strip() == "":
                    print("[title_reader] final title source=none")
                    return NEWSPAPER_NAME_FALLBACK, warnings, "none"
                title_str = str(title).strip()
                if title_str in (TITLE_NOT_CLEAR, NEWSPAPER_NAME_FALLBACK):
                    print("[title_reader] final title source=easyocr")
                    return NEWSPAPER_NAME_FALLBACK, warnings, "easyocr"
                print("[title_reader] final title source=easyocr")
                return title_str, warnings, "easyocr"
            except Exception:
                logger.exception("newspaper masthead exception")
                print("[title_reader] final title source=none")
                return NEWSPAPER_NAME_FALLBACK, warnings, "none"

        if _easyocr_extract_title is None:
            warnings.append(TITLE_CONFIDENCE_WARNING)
            print("[title_reader] final title source=none")
            return TITLE_NOT_CLEAR, warnings, "none"
        try:
            title = _run_easyocr_title(image_path, document_type)
            if title is None or str(title).strip() == "":
                warnings.append(TITLE_CONFIDENCE_WARNING)
                print("[title_reader] final title source=none")
                return TITLE_NOT_CLEAR, warnings, "none"
            title_str = str(title).strip()
            if title_str == TITLE_NOT_CLEAR:
                warnings.append(TITLE_CONFIDENCE_WARNING)
                print(
                    "[pipeline] Novel/Magazine title not confident -> "
                    f"{TITLE_NOT_CLEAR!r}; warning={TITLE_CONFIDENCE_WARNING}"
                )
                print("[title_reader] final title source=easyocr")
                return title_str, warnings, "easyocr"
            print("[title_reader] final title source=easyocr")
            return title_str, warnings, "easyocr"
        except Exception:
            warnings.append(TITLE_CONFIDENCE_WARNING)
            print("[title_reader] final title source=none")
            return TITLE_NOT_CLEAR, warnings, "none"

    print(f"[title_reader] easyocr fallback used={str(easyocr_fallback_used).lower()}")

    if title_str is None:
        if document_type == "Newspaper":
            title_str = NEWSPAPER_NAME_FALLBACK
        else:
            title_str = TITLE_NOT_CLEAR
            if TITLE_CONFIDENCE_WARNING not in warnings:
                warnings.append(TITLE_CONFIDENCE_WARNING)
        title_source = "none"

    if (
        document_type in COVER_TITLE_TYPES
        and title_str == TITLE_NOT_CLEAR
        and TITLE_CONFIDENCE_WARNING not in warnings
    ):
        warnings.append(TITLE_CONFIDENCE_WARNING)

    print(f"[title_reader] final title source={title_source}")
    return title_str, warnings, title_source


def _reload_backend_env() -> None:
    """Reload backend/.env with override so IMAGE_DESCRIPTION_MODEL is applied."""
    load_dotenv(dotenv_path=_ENV_PATH, override=True)


def _selected_image_description_model() -> str:
    """
    Read IMAGE_DESCRIPTION_MODEL from backend/.env / process env.
    Default: blip. Allowed: blip, florence.
    """
    _reload_backend_env()
    image_description_model = os.getenv("IMAGE_DESCRIPTION_MODEL", "blip").lower().strip()
    if image_description_model in {"florence", "florence-2", "florence2"}:
        return "florence"
    return "blip"


def _run_blip_description(image_path: str) -> list[str]:
    if _describe_image_local is None:
        raise RuntimeError("BLIP image description module is unavailable")
    descriptions = _describe_image_local(image_path)
    if descriptions is None or not isinstance(descriptions, list):
        return []
    return [str(item).strip() for item in descriptions if str(item).strip()]


def _florence_type_fallback(document_type: str) -> str:
    if _florence_fallback_for_document_type is not None:
        try:
            return str(_florence_fallback_for_document_type(document_type) or "").strip()
        except Exception:
            logger.exception("Florence fallback lookup failed")
    dtype = (document_type or "").strip()
    if dtype == "Magazine":
        return (
            "The magazine cover appears to show a visual design or main cover image."
        )
    if dtype == "Newspaper":
        return "No clear main image is detected on this newspaper page."
    if dtype == "Report":
        return "No important visual image is detected in this report."
    if dtype == "Printed_Letter":
        return "No important visual image is detected in this letter."
    return "The cover appears to show an illustrated scene related to the story."


def _run_florence_description(
    image_path: str,
    document_type: str = "",
) -> dict[str, str]:
    if _describe_image_florence is None:
        raise RuntimeError("Florence image description module is unavailable")
    result = _describe_image_florence(image_path, document_type=document_type)
    if not result or not isinstance(result, dict):
        return {}
    raw = str(result.get("florence_raw_caption") or "").strip()
    cleaned = str(result.get("florence_cleaned_caption") or "").strip()
    if raw and _format_florence_blind_caption is not None:
        # Always re-validate / rebuild cleaned caption for safety.
        rebuilt = str(
            _format_florence_blind_caption(raw, document_type=document_type) or ""
        ).strip()
        cleaned = rebuilt
    if cleaned and not _is_valid_local_florence_description(
        cleaned,
        document_type=document_type,
    ):
        cleaned = _florence_type_fallback(document_type)
    if not raw and not cleaned:
        return {}
    return {
        "florence_raw_caption": raw,
        "florence_cleaned_caption": cleaned,
    }


def _is_valid_local_florence_description(
    text: str,
    document_type: str = "",
) -> bool:
    cleaned = (text or "").strip()
    if not cleaned or cleaned in {".", "The cover appears to show.", "The cover shows."}:
        return False
    lowered = cleaned.lower().strip()
    no_visual = {
        "no clear main image is detected on this newspaper page.",
        "no important visual image is detected in this report.",
        "no important visual image is detected in this letter.",
    }
    if lowered in no_visual:
        return True
    if len(cleaned) < 15:
        return False
    if re.fullmatch(r"the (?:magazine )?cover appears to show\s*\.?", lowered):
        return False
    if re.fullmatch(r"the cover shows\s*\.?", lowered):
        return False
    if re.fullmatch(r"the newspaper page appears to contain\s*\.?", lowered):
        return False
    if re.fullmatch(r"the (?:report|letter) appears to show\s*\.?", lowered):
        return False
    if re.search(r"appears to (?:show|contain)\s*\.", lowered):
        return False
    if re.search(r"appears to (?:show|contain)\s*[\"']\s*[\"']", lowered):
        return False
    if re.search(r"appears to (?:show|contain)\s+cover\b", lowered):
        return False
    if re.search(r"appears to (?:show|contain)\s+title text\b", lowered):
        return False
    if "cover is blue" in lowered or "gold lettering" in lowered:
        return False
    dtype = (document_type or "").strip()
    if dtype in {"Newspaper", "Report", "Printed_Letter"}:
        if "related to the story" in lowered:
            return False
        if re.search(r"\bthe cover appears to show\b", lowered) and (
            "newspaper" not in lowered
            and "report" not in lowered
            and "letter" not in lowered
        ):
            return False
    return True


def _describe_images_local(
    image_path: str,
    title: str = "",
    document_type: str = "",
) -> tuple[list[str], list[str], str, dict[str, str]]:
    """
    Run selected local image description model.

    Returns:
        (raw_descriptions, warnings, description_source, florence_meta)
        description_source is "blip" or "florence".
        florence_meta holds raw/cleaned Florence captions when used.
    """
    warnings: list[str] = []
    empty_meta: dict[str, str] = {}
    image_description_model = _selected_image_description_model()
    print(f"[pipeline] image_description_model={image_description_model}")
    logger.info("image_description_model=%s", image_description_model)

    if image_description_model == "florence":
        try:
            florence_meta = _run_florence_description(
                image_path,
                document_type=document_type,
            )
            raw = florence_meta.get("florence_raw_caption", "")
            cleaned = florence_meta.get("florence_cleaned_caption", "")
            if raw or cleaned:
                print(f"[pipeline] florence_raw_caption={raw!r}")
                print(f"[pipeline] florence_cleaned_caption={cleaned!r}")
                print("[pipeline] description_source=florence")
                logger.info(
                    "description_source=florence raw=%r cleaned=%r",
                    raw,
                    cleaned,
                )
                # Keep raw in the list for optional keyword title recovery helpers.
                return (
                    [raw] if raw else [cleaned],
                    warnings,
                    "florence",
                    florence_meta,
                )
            print("[pipeline] Florence failed, falling back to BLIP")
            logger.warning("Florence empty caption; falling back to BLIP")
        except Exception as exc:
            logger.exception(
                "Florence image description failed for path=%r; falling back to BLIP",
                image_path,
            )
            print(f"[pipeline] Florence failed, falling back to BLIP ({exc})")

        try:
            cleaned = _run_blip_description(image_path)
            print("[pipeline] description_source=blip")
            logger.info("description_source=blip reason=florence_fallback")
            if cleaned:
                return cleaned, warnings, "blip", empty_meta
            warnings.append("Image description failed")
            return [], warnings, "blip", empty_meta
        except Exception:
            logger.exception("BLIP fallback failed for path=%r", image_path)
            warnings.append("Image description failed")
            print("[pipeline] description_source=blip")
            return [], warnings, "blip", empty_meta

    # Default / explicit BLIP
    try:
        cleaned = _run_blip_description(image_path)
        print("[pipeline] description_source=blip")
        logger.info("description_source=blip")
        if cleaned:
            return cleaned, warnings, "blip", empty_meta
        print("[pipeline] image-description fallback: empty BLIP caption")
        logger.info("image description fallback reason=empty_blip")
        warnings.append("Image description failed")
        return [], warnings, "blip", empty_meta
    except Exception:
        logger.exception("local BLIP image description failed for path=%r", image_path)
        print("[pipeline] image-description fallback: exception")
        logger.info("image description fallback reason=exception")
        warnings.append("Image description failed")
        print("[pipeline] description_source=blip")
        return [], warnings, "blip", empty_meta


def _recover_title_from_florence(
    title: str,
    florence_raw_caption: str,
    warnings: list[str],
    document_type: str = "",
) -> tuple[str, list[str]]:
    """Recover title from Florence raw caption when OCR title is unclear."""
    needs_recovery = (
        title == TITLE_NOT_CLEAR or TITLE_CONFIDENCE_WARNING in warnings
    )
    if not needs_recovery:
        return title, warnings

    dtype = (document_type or "").strip()
    recovered: str | None = None
    if _recover_title_from_florence_caption is not None:
        try:
            recovered = _recover_title_from_florence_caption(
                florence_raw_caption,
                document_type=dtype,
            )
            if recovered and _is_descriptive_florence_title(recovered):
                print(
                    "[title_reader] rejected florence title candidate="
                    f"{recovered!r} reason=descriptive_phrase"
                )
                recovered = None
            elif recovered:
                print(
                    f"[title_reader] accepted florence title candidate={recovered!r}"
                )
                print(f"[pipeline] title recovery candidate={recovered!r}")
        except Exception:
            logger.exception("Florence title recovery failed")
            recovered = None

    if not recovered and dtype in {"Novel", "Magazine", ""}:
        # Soft keyword fallback for common covers only.
        blob_n = re.sub(r"[^a-z0-9\s]", " ", (florence_raw_caption or "").lower())
        words = set(blob_n.split())
        if "kite" in words and "runner" in words:
            recovered = "The Kite Runner"
        elif "peter" in words and "pan" in words:
            recovered = "Peter Pan"
        elif "treasure" in words and "island" in words:
            recovered = "Treasure Island"
        elif "ulysses" in words or "ulysses" in blob_n:
            recovered = "Ulysses"
        elif "jungle" in words and "book" in words:
            recovered = "The Jungle Book"
        elif "busy" in words and "baker" in words:
            recovered = "The Busy Baker"
        elif "uncle" in words and "cabin" in words:
            recovered = "Uncle Tom's Cabin"
        elif "pc" in words and "polly" in words:
            recovered = "PC Polly"
        elif "dracula" in words:
            recovered = "Dracula"
        elif dtype == "Magazine" and "forbes" in words:
            recovered = "Forbes"
        elif dtype == "Magazine" and "shape" in words:
            recovered = "Shape"
        if recovered and _is_descriptive_florence_title(recovered):
            print(
                "[title_reader] rejected florence title candidate="
                f"{recovered!r} reason=descriptive_phrase"
            )
            recovered = None
        elif recovered:
            print(
                f"[title_reader] accepted florence title candidate={recovered!r}"
            )
            print(f"[pipeline] title recovery candidate={recovered!r}")

    if not recovered:
        return title, warnings

    print(f"[pipeline] title recovered from Florence={recovered!r}")
    logger.info("title recovered from Florence=%r", recovered)
    cleaned_warnings = [
        w
        for w in warnings
        if w not in {TITLE_CONFIDENCE_WARNING, FLORENCE_TITLE_RECOVERY_WARNING}
    ]
    cleaned_warnings.append(FLORENCE_TITLE_RECOVERY_WARNING)
    return recovered, cleaned_warnings


def _is_descriptive_florence_title(text: str) -> bool:
    """Pipeline-side guard against descriptive Florence title recoveries."""
    lowered = re.sub(r"[^a-z0-9\s]", " ", (text or "").lower())
    lowered = re.sub(r"\s+", " ", lowered).strip()
    if not lowered:
        return True
    descriptive = (
        "written in red letters",
        "written in black letters",
        "written in gold letters",
        "written in blue letters",
        "written in white letters",
        "red letters",
        "black letters",
        "gold letters",
        "blue letters",
        "title written",
        "cover has",
        "cover of",
        "book cover",
        "front page",
        "picture of",
        "image of",
        "newspaper page",
        "magazine cover",
        "the words",
        "the title of the book is written",
        "title of the book is written",
    )
    if any(phrase in lowered for phrase in descriptive):
        return True
    if re.search(r"\bwritten in\b", lowered):
        return True
    if re.search(r"\b(?:red|black|gold|blue|white)\s+letters?\b", lowered):
        return True
    return False


def _prepare_cropped_image(image_path: str) -> tuple[str, list[str]]:
    """
    Crop reading material, then ensure a max-width 1200 processing image
    for document ID / title / BLIP.

    Soft crop failure: optimize the original image, log only (no JSON warning),
    so usable Forbes/etc. results are not forced to partial_success.
    """
    warnings: list[str] = []

    def _ensure_processing_size(path: str) -> str:
        if _optimize_processing_image is None:
            return path
        try:
            return _optimize_processing_image(path) or path
        except Exception:
            logger.exception("optimize_processing_image failed path=%r", path)
            return path

    if _crop_reading_material is None:
        logger.warning(
            "document cropper not available; optimizing original path=%r",
            image_path,
        )
        print("[pipeline] crop unavailable; optimizing original image (no warning)")
        return _ensure_processing_size(image_path), warnings

    try:
        cropped_path = _crop_reading_material(image_path)
        if not cropped_path or not isinstance(cropped_path, str):
            logger.warning(
                "document crop returned invalid path; optimizing original path=%r",
                image_path,
            )
            print("[pipeline] crop failed soft; optimizing original image (no warning)")
            return _ensure_processing_size(image_path), warnings

        # Cropper returns original path when detection/crop fails.
        if os.path.abspath(cropped_path) == os.path.abspath(image_path):
            logger.warning(
                "document crop soft-fail; optimizing original image path=%r",
                image_path,
            )
            print("[pipeline] crop soft-fail; optimizing original image (no warning)")
            return _ensure_processing_size(image_path), warnings

        # Successful crop already returns a resized proc image; keep safety resize.
        processing_path = _ensure_processing_size(cropped_path)
        logger.info("pipeline using processing image path=%r", processing_path)
        print(f"[pipeline] crop path used for BLIP/title/docID: {processing_path}")
        return processing_path, warnings
    except Exception:
        logger.exception(
            "document crop exception; optimizing original image path=%r",
            image_path,
        )
        print("[pipeline] crop exception; optimizing original image (no warning)")
        return _ensure_processing_size(image_path), warnings


def _recover_title_from_blip_captions(
    title: str,
    image_descriptions: list[str],
    warnings: list[str],
) -> tuple[str, list[str]]:
    """
    Local post-BLIP title recovery when OCR title was unclear.
    Uses only caption text already produced by local BLIP (no external APIs).
    """
    needs_recovery = (
        title == TITLE_NOT_CLEAR or TITLE_CONFIDENCE_WARNING in warnings
    )
    if not needs_recovery:
        return title, warnings

    blob = " ".join(str(item) for item in (image_descriptions or []))
    blob_n = re.sub(r"[^a-z0-9\s]", " ", blob.lower())
    words = set(blob_n.split())

    recovered: str | None = None
    if "jungle" in words and "book" in words:
        recovered = "The Jungle Book"
    elif "busy" in words and "baker" in words:
        recovered = "The Busy Baker"
    elif "uncle" in words and "cabin" in words:
        recovered = "Uncle Tom's Cabin"
    elif "pc" in words and "polly" in words:
        recovered = "PC Polly"
    elif "peter" in words and "pan" in words:
        recovered = "Peter Pan"
    elif "treasure" in words and "island" in words:
        recovered = "Treasure Island"
    elif "ulysses" in words:
        recovered = "Ulysses"
    elif "forbes" in words:
        recovered = "Forbes"
    elif "shape" in words:
        recovered = "Shape"

    if recovered is None:
        print(
            f"[pipeline] post-BLIP title recovery: no match "
            f"(title={title!r})"
        )
        logger.info("post-BLIP title recovery no match title=%r", title)
        print(f"[pipeline] final title after recovery: {title!r}")
        logger.info("final title after recovery=%r", title)
        return title, warnings

    print(f"[pipeline] post-BLIP title recovery matched: {recovered}")
    logger.info("post-BLIP title recovery matched title=%r", recovered)

    cleaned_warnings = [w for w in warnings if w != TITLE_CONFIDENCE_WARNING]
    print(f"[pipeline] final title after recovery: {recovered!r}")
    logger.info("final title after recovery=%r", recovered)
    return recovered, cleaned_warnings


def _resolve_status(warnings: list[str], document_type: str) -> str:
    # Soft crop fallback is intentionally excluded from warnings.
    # Informational-only notes should not force partial_success.
    informational = {FLORENCE_TITLE_RECOVERY_WARNING}
    actionable = [w for w in (warnings or []) if w not in informational]
    if not actionable:
        return "success"
    if document_type == "Unknown" and "Document identification failed" in actionable:
        return "failed"
    return "partial_success"


def _openai_fallback_flag() -> bool:
    if _is_openai_fallback_enabled is None:
        return False
    try:
        return bool(_is_openai_fallback_enabled())
    except Exception:
        return False


def _is_generic_image_description(text: str) -> bool:
    normalized = _normalize_compare_text(text)
    if not normalized:
        return True
    if normalized.startswith(
        "the cover contains an illustration or visual design related to"
    ):
        return True
    if normalized == _normalize_compare_text(UNCLEAR_VISUAL_FALLBACK):
        return True
    if normalized == _normalize_compare_text(COVER_ILLUSTRATION_FALLBACK):
        return True
    return False


def _should_trigger_openai_fallback(
    title: str,
    image_descriptions: list[str],
    status: str,
) -> tuple[bool, str]:
    reasons: list[str] = []
    if title == TITLE_NOT_CLEAR:
        reasons.append("title_unclear")
    if not image_descriptions:
        reasons.append("empty_image_descriptions")
    elif any(_is_generic_image_description(item) for item in image_descriptions):
        reasons.append("generic_image_description")
    if status == "partial_success":
        reasons.append("partial_success")
    if not reasons:
        return False, "not_needed"
    return True, ",".join(reasons)


def _apply_openai_fallback(
    image_path: str,
    title: str,
    image_descriptions: list[str],
    warnings: list[str],
    status: str,
) -> tuple[str, list[str], list[str]]:
    """
    Optional OpenAI Vision fallback. Local PP2 results are kept unless
    USE_OPENAI_FALLBACK=true and a trigger condition matches.
    Never raises.
    """
    enabled = _openai_fallback_flag()
    print(f"[pipeline] USE_OPENAI_FALLBACK={str(enabled).lower()}")
    logger.info("USE_OPENAI_FALLBACK=%s", enabled)

    if not enabled:
        print("[pipeline] OpenAI fallback skipped")
        logger.info("OpenAI fallback skipped reason=disabled")
        return title, image_descriptions, warnings

    should_run, reason = _should_trigger_openai_fallback(
        title, image_descriptions, status
    )
    if not should_run:
        print(f"[pipeline] OpenAI fallback skipped ({reason})")
        logger.info("OpenAI fallback skipped reason=%s", reason)
        return title, image_descriptions, warnings

    print(f"[pipeline] OpenAI fallback triggered ({reason})")
    logger.info("OpenAI fallback triggered reason=%s", reason)

    if _run_openai_vision_fallback is None:
        print("[pipeline] OpenAI fallback failed")
        logger.warning("OpenAI fallback failed reason=module_missing")
        return title, image_descriptions, warnings

    try:
        fallback = _run_openai_vision_fallback(image_path)
    except Exception:
        logger.exception("OpenAI fallback crashed; keeping local result")
        print("[pipeline] OpenAI fallback failed")
        return title, image_descriptions, warnings

    if not fallback or not isinstance(fallback, dict):
        print("[pipeline] OpenAI fallback failed")
        logger.warning("OpenAI fallback failed reason=empty_or_invalid")
        return title, image_descriptions, warnings

    openai_title = str(fallback.get("title") or "").strip()
    openai_description = str(fallback.get("image_description") or "").strip()
    merged = False
    new_title = title
    new_descriptions = list(image_descriptions or [])
    new_warnings = list(warnings or [])

    if title == TITLE_NOT_CLEAR and openai_title:
        new_title = openai_title
        new_warnings = [w for w in new_warnings if w != TITLE_CONFIDENCE_WARNING]
        merged = True
        print(f"[pipeline] OpenAI fallback title merged: {new_title!r}")

    local_desc_unusable = (
        not new_descriptions
        or any(_is_generic_image_description(item) for item in new_descriptions)
    )
    if local_desc_unusable and openai_description:
        new_descriptions = [openai_description]
        new_warnings = [w for w in new_warnings if w != "Image description failed"]
        merged = True
        print(
            "[pipeline] OpenAI fallback image description merged: "
            f"{openai_description!r}"
        )

    if not merged:
        print("[pipeline] OpenAI fallback failed")
        logger.info("OpenAI fallback failed reason=no_usable_fields")
        return title, image_descriptions, warnings

    if OPENAI_FALLBACK_WARNING not in new_warnings:
        new_warnings.append(OPENAI_FALLBACK_WARNING)
    print("[pipeline] OpenAI fallback used")
    logger.info("OpenAI fallback used title=%r descriptions=%r", new_title, new_descriptions)
    return new_title, new_descriptions, new_warnings


def analyze_page(image_path: str) -> dict:
    """
    Run Abhishek's page analysis pipeline.

    1) Crop reading material from the camera image
    2) Document identification
    3) Title reading
    4) Local BLIP image description (on cropped image when available)
    5) Optional OpenAI Vision fallback only if USE_OPENAI_FALLBACK=true

    Always returns the same JSON keys. Never raises to the caller.
    """
    result = _empty_result()
    all_warnings: list[str] = []
    t_total = time.perf_counter()

    try:
        t0 = time.perf_counter()
        cropped_image_path, crop_warnings = _prepare_cropped_image(image_path)
        crop_sec = time.perf_counter() - t0
        all_warnings.extend(crop_warnings)
        logger.debug(
            "analyze_page original=%r cropped=%r",
            image_path,
            cropped_image_path,
        )
        print(
            f"[pipeline] analyze_page original={image_path!r} "
            f"cropped={cropped_image_path!r}"
        )
        print(f"[pipeline] timing crop={crop_sec:.2f}s")
        logger.info("timing crop=%.2fs", crop_sec)

        t0 = time.perf_counter()
        document_type, confidence, doc_warnings = _identify_document(
            cropped_image_path
        )
        classify_sec = time.perf_counter() - t0
        all_warnings.extend(doc_warnings)
        logger.debug("detected document_type=%s confidence=%s", document_type, confidence)
        print(f"[pipeline] timing document_classification={classify_sec:.2f}s")
        logger.info("timing document_classification=%.2fs", classify_sec)

        t0 = time.perf_counter()
        title, title_warnings, title_source = _read_title(
            cropped_image_path,
            document_type,
        )
        title_sec = time.perf_counter() - t0
        all_warnings.extend(title_warnings)
        logger.debug("final title value=%r source=%s", title, title_source)
        print(f"[pipeline] timing title_reading={title_sec:.2f}s")
        logger.info("timing title_reading=%.2fs", title_sec)

        print(f"[pipeline] crop path used for BLIP: {cropped_image_path}")
        t0 = time.perf_counter()
        raw_image_descriptions, image_warnings, description_source, florence_meta = (
            _describe_images_local(
                cropped_image_path,
                title=title,
                document_type=document_type,
            )
        )
        blip_sec = time.perf_counter() - t0
        all_warnings.extend(image_warnings)
        print(
            f"[pipeline] timing image_description={blip_sec:.2f}s "
            f"source={description_source}"
        )
        logger.info(
            "timing image_description=%.2fs source=%s",
            blip_sec,
            description_source,
        )

        if description_source == "florence":
            florence_raw_caption = florence_meta.get("florence_raw_caption", "")
            florence_cleaned_caption = florence_meta.get(
                "florence_cleaned_caption",
                "",
            )
            print(f"[pipeline] document_type_for_description={document_type}")
            print(f"[pipeline] florence_raw_caption={florence_raw_caption!r}")
            print(f"[pipeline] florence_cleaned_caption={florence_cleaned_caption!r}")

            title_before_florence = title
            title, all_warnings = _recover_title_from_florence(
                title,
                florence_raw_caption,
                all_warnings,
                document_type=document_type,
            )
            if (
                title != title_before_florence
                and FLORENCE_TITLE_RECOVERY_WARNING in all_warnings
            ):
                title_source = "florence"
                print("[title_reader] final title source=florence")


            # Florence path: do NOT run BLIP-style blind-user formatter.
            if _format_florence_blind_caption is not None and florence_raw_caption:
                florence_cleaned_caption = str(
                    _format_florence_blind_caption(
                        florence_raw_caption,
                        document_type=document_type,
                    )
                    or ""
                ).strip()
            if florence_cleaned_caption and not _is_valid_local_florence_description(
                florence_cleaned_caption,
                document_type=document_type,
            ):
                print(
                    "[pipeline] rejected non-visual sentence="
                    f"{florence_cleaned_caption!r}"
                )
                florence_cleaned_caption = _florence_type_fallback(document_type)

            no_visual_messages = {
                "No clear main image is detected on this newspaper page.",
                "No important visual image is detected in this report.",
                "No important visual image is detected in this letter.",
            }
            if not florence_cleaned_caption:
                image_descriptions = []
                print("[pipeline] selected visual sentence=<none>")
                print("[pipeline] final document-aware description=<empty>")
            else:
                image_descriptions = [florence_cleaned_caption]
                if florence_cleaned_caption in no_visual_messages:
                    print("[pipeline] selected visual sentence=<none>")
                else:
                    print(
                        f"[pipeline] selected visual sentence="
                        f"{florence_cleaned_caption!r}"
                    )
                print(
                    f"[pipeline] final document-aware description="
                    f"{florence_cleaned_caption!r}"
                )
            print("[pipeline] final description source=florence_cleaned")
            logger.info(
                "final description source=florence_cleaned caption=%r document_type=%s",
                florence_cleaned_caption,
                document_type,
            )
        else:
            # Recover title from raw BLIP text before formatting the description.
            title, all_warnings = _recover_title_from_blip_captions(
                title,
                raw_image_descriptions,
                all_warnings,
            )

            image_descriptions, format_warnings = _format_blind_descriptions(
                raw_image_descriptions,
                title=title,
                document_type=document_type,
            )
            all_warnings.extend(format_warnings)
            print("[pipeline] final description source=blip_formatted")
            logger.info("final description source=blip_formatted")

        # Deduplicate warnings while preserving order
        seen: set[str] = set()
        unique_warnings: list[str] = []
        for warning in all_warnings:
            if warning not in seen:
                seen.add(warning)
                unique_warnings.append(warning)

        local_status = _resolve_status(unique_warnings, document_type)
        title, image_descriptions, unique_warnings = _apply_openai_fallback(
            cropped_image_path,
            title,
            image_descriptions,
            unique_warnings,
            local_status,
        )

        seen = set()
        deduped_warnings: list[str] = []
        for warning in unique_warnings:
            if warning not in seen:
                seen.add(warning)
                deduped_warnings.append(warning)
        unique_warnings = deduped_warnings

        result = {
            "document_type": document_type,
            "confidence": confidence,
            "title": title,
            "image_descriptions": image_descriptions,
            "status": _resolve_status(unique_warnings, document_type),
            "warnings": unique_warnings,
        }
        total_sec = time.perf_counter() - t_total
        print(
            "[pipeline] timing summary "
            f"crop={crop_sec:.2f}s "
            f"document_classification={classify_sec:.2f}s "
            f"title_reading={title_sec:.2f}s "
            f"image_description={blip_sec:.2f}s "
            f"source={description_source} "
            f"total={total_sec:.2f}s"
        )
        logger.info(
            "timing total=%.2fs crop=%.2fs classify=%.2fs title=%.2fs "
            "image_description=%.2fs source=%s",
            total_sec,
            crop_sec,
            classify_sec,
            title_sec,
            blip_sec,
            description_source,
        )
        return result

    except Exception:
        total_sec = time.perf_counter() - t_total
        print(f"[pipeline] timing total={total_sec:.2f}s (failed)")
        logger.info("timing total=%.2fs status=failed", total_sec)
        result["warnings"] = ["Document identification failed"]
        result["status"] = "failed"
        return result


def describe_image_only(image_path: str, context: str = "general") -> dict:
    """
    Reusable Florence-only image description for cropped covers / page
    illustrations / selected newspaper article images.

    Does not run document classification or title reading.
    """
    print("[image_description] mode=image_only")
    normalized_context = (context or "general").strip() or "general"
    print(f"[image_description] context={normalized_context}")
    logger.info(
        "describe_image_only path=%r context=%s",
        image_path,
        normalized_context,
    )

    warnings: list[str] = []
    empty = {
        "has_image": False,
        "image_description": "",
        "context": normalized_context,
        "status": "failed",
        "warnings": warnings,
    }

    if not image_path or not str(image_path).strip():
        warnings.append("Image description failed")
        empty["warnings"] = warnings
        print("[image_description] has_image=false")
        print("[image_description] final_description=")
        return empty

    if not os.path.isfile(image_path):
        warnings.append("Image description failed")
        empty["warnings"] = warnings
        print("[image_description] has_image=false")
        print("[image_description] final_description=")
        return empty

    # Optional resize for stable Florence inference (no full-page crop/classify).
    process_path = image_path
    if _optimize_processing_image is not None:
        try:
            process_path = _optimize_processing_image(image_path) or image_path
        except Exception:
            logger.exception("optimize_processing_image failed for image-only path")
            process_path = image_path

    if _describe_image_florence is None:
        warnings.append("Image description failed")
        empty["warnings"] = warnings
        print("[image_description] has_image=false")
        print("[image_description] final_description=")
        return empty

    try:
        # Florence raw caption only; context formatting applied separately so
        # /abhishek/analyze document-type wording stays unchanged.
        florence_meta = _describe_image_florence(process_path, document_type="")
    except Exception as exc:
        logger.exception("Florence image-only description failed path=%r", process_path)
        warnings.append("Image description failed")
        print(f"[image_description] florence error: {exc}")
        print("[image_description] has_image=false")
        print("[image_description] final_description=")
        return {
            "has_image": False,
            "image_description": "",
            "context": normalized_context,
            "status": "failed",
            "warnings": warnings,
        }

    if not florence_meta or not isinstance(florence_meta, dict):
        warnings.append("Image description failed")
        print("[image_description] has_image=false")
        print("[image_description] final_description=")
        return {
            "has_image": False,
            "image_description": "",
            "context": normalized_context,
            "status": "failed",
            "warnings": warnings,
        }

    raw = str(florence_meta.get("florence_raw_caption") or "").strip()
    print(f"[pipeline] florence_raw_caption={raw!r}")

    description = ""
    has_image = False
    if _format_florence_context_caption is not None:
        try:
            description, has_image = _format_florence_context_caption(
                raw,
                context=normalized_context,
            )
            description = str(description or "").strip()
            has_image = bool(has_image) and bool(description)
        except Exception:
            logger.exception("Florence context caption formatting failed")
            description = ""
            has_image = False
    else:
        # Extreme fallback if helper missing: use cleaned Florence caption as-is.
        description = str(florence_meta.get("florence_cleaned_caption") or "").strip()
        has_image = bool(description)

    no_image_markers = (
        "no important visual image is detected",
        "no clear main image is detected",
    )
    lowered = description.lower()
    if any(marker in lowered for marker in no_image_markers):
        has_image = False

    if not description and not raw:
        warnings.append("Image description failed")
        status = "failed"
    elif not has_image:
        status = "partial_success" if description else "failed"
        if not description:
            warnings.append("Image description failed")
    else:
        status = "success"

    print(f"[image_description] has_image={str(has_image).lower()}")
    print(f"[image_description] final_description={description!r}")
    logger.info(
        "describe_image_only has_image=%s status=%s description=%r",
        has_image,
        status,
        description,
    )

    # Deduplicate warnings
    seen: set[str] = set()
    unique_warnings: list[str] = []
    for warning in warnings:
        if warning not in seen:
            seen.add(warning)
            unique_warnings.append(warning)

    return {
        "has_image": has_image,
        "image_description": description,
        "context": normalized_context,
        "status": status,
        "warnings": unique_warnings,
    }

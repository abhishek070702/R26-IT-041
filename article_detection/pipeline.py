"""
Thin adapter so FastAPI and the UI can call the latest V7 pipeline.

Latest implementation:
  adaptive_content_pipeline_v7_openai_fallback.py
    analyze_content_v7(...)
    generate_selected_output_v7(...)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv()

from adaptive_content_pipeline_v7_openai_fallback import (
    ApplicationFallbackConfig,
    analyze_content_v7,
    configure_tesseract,
    generate_selected_output_v7,
    load_category_model,
    load_doclayout_model,
)

logger = logging.getLogger(__name__)

_BACKEND_DIR = Path(__file__).resolve().parent


def _resolve_category_model_path() -> str:
    env_path = os.getenv("CATEGORY_MODEL_PATH")
    if env_path and Path(env_path).is_file():
        return env_path

    for name in ("category_model.pkl", "category_model(1).pkl"):
        candidate = _BACKEND_DIR / name
        if candidate.is_file():
            return str(candidate)

    return str(_BACKEND_DIR / "category_model.pkl")


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


CONFIG = ApplicationFallbackConfig(
    category_model_path=_resolve_category_model_path(),
    tesseract_cmd=os.getenv("TESSERACT_CMD") or None,
    research_mode=_bool_env("RESEARCH_MODE", False),
    enable_openai_fallback=_bool_env("ENABLE_OPENAI_FALLBACK", True),
    enable_openai_category_fallback=_bool_env(
        "ENABLE_OPENAI_CATEGORY_FALLBACK", True
    ),
    enable_openai_summary_fallback=_bool_env(
        "ENABLE_OPENAI_SUMMARY_FALLBACK", True
    ),
    prefer_openai_article_detection_first=_bool_env(
        "PREFER_OPENAI_ARTICLE_DETECTION_FIRST", True
    ),
    prefer_openai_full_page_text_first=_bool_env(
        "PREFER_OPENAI_FULL_PAGE_TEXT_FIRST", True
    ),
    clean_full_text_with_llama=_bool_env(
        "CLEAN_FULL_TEXT_WITH_LLAMA", True
    ),
    debug=False,
)

configure_tesseract(CONFIG)
CATEGORY_MODEL = load_category_model(CONFIG.category_model_path)

DOCLAYOUT_MODEL = None
try:
    loaded = load_doclayout_model(CONFIG)
    if isinstance(loaded, tuple):
        DOCLAYOUT_MODEL = loaded[0]
    else:
        DOCLAYOUT_MODEL = loaded
except Exception as error:
    logger.warning("DocLayout-YOLO not loaded (%s). Using local geometry only.", error)


def _join_article_text(results: Any) -> str:
    if not isinstance(results, list):
        return ""
    return "\n\n".join(
        (item.get("full_text") or "").strip()
        for item in results
        if isinstance(item, dict) and (item.get("full_text") or "").strip()
    )


def _normalize_analysis(analysis: dict[str, Any]) -> dict[str, Any]:
    categories = (
        analysis.get("categories")
        or analysis.get("detected_categories")
        or []
    )
    analysis["categories"] = list(categories)
    analysis["detected_categories"] = list(
        analysis.get("detected_categories") or categories
    )

    local_results = analysis.get("local_results")
    if not isinstance(local_results, list) or not local_results:
        local_results = list(analysis.get("results") or [])
        analysis["local_results"] = local_results

    ai_results = analysis.get("ai_results")
    if not isinstance(ai_results, list):
        ai_results = []
        analysis["ai_results"] = ai_results

    analysis["local_extracted_text"] = _join_article_text(local_results)
    analysis["ai_extracted_text"] = _join_article_text(ai_results)

    if not (analysis.get("full_page_text") or "").strip():
        analysis["full_page_text"] = (
            analysis["ai_extracted_text"]
            or analysis["local_extracted_text"]
            or _join_article_text(analysis.get("results") or [])
        )

    return analysis


def analyze_content(
    image_path: str,
    document_type: str,
    debug: bool = False,
) -> dict[str, Any]:
    """API-compatible wrapper around analyze_content_v7."""
    previous_debug = getattr(CONFIG, "debug", False)
    if debug:
        CONFIG.debug = True
    try:
        analysis = analyze_content_v7(
            image_path=image_path,
            document_type=document_type,
            category_model=CATEGORY_MODEL,
            config=CONFIG,
            doclayout_model=DOCLAYOUT_MODEL,
        )
    finally:
        CONFIG.debug = previous_debug

    return _normalize_analysis(analysis)


def generate_selected_output(
    analysis: dict[str, Any],
    selected_category: Optional[str],
    depth: str,
) -> dict[str, Any]:
    """API-compatible wrapper around generate_selected_output_v7."""
    analysis = _normalize_analysis(dict(analysis))
    return generate_selected_output_v7(
        analysis=analysis,
        selected_category=selected_category,
        depth=depth,
        config=CONFIG,
    )

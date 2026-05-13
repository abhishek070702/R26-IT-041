"""
API routes for content summarization and category detection.
Mounted under prefix /content (see backend.main).
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from backend.content_summarization_category_detection.pipeline import (
    analyze_content,
    generate_selected_output,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/content",
    tags=["Content Summarization & Category Detection"],
)


class GenerateOutputRequest(BaseModel):
    """Full `analysis` object as returned by analyze_content (JSON-serializable)."""

    analysis: dict[str, Any]
    selected_category: str
    depth: str = Field(
        ...,
        description="Either 'summary' or 'full'.",
    )


@router.post("/analyze")
async def analyze(
    document_type: Annotated[str, Form()],
    file: Annotated[UploadFile, File()],
):
    """
    Accept an image and document type, run OCR + category pipeline,
    return extracted text, categories, and full analysis payload.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file.")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff", ".tif"}:
        suffix = ".jpg"

    tmp_path: str | None = None
    try:
        fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix="content_upload_")
        with os.fdopen(fd, "wb") as f:
            f.write(raw)

        analysis = analyze_content(tmp_path, document_type or "", debug=False)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("analyze_content failed")
        raise HTTPException(
            status_code=500,
            detail="Analysis failed. Check server logs, Tesseract, OpenCV, and model file.",
        ) from e
    finally:
        if tmp_path and os.path.isfile(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                logger.warning("Could not remove temp file %s", tmp_path)

    extracted = (analysis.get("full_page_text") or "").strip()
    if not extracted and analysis.get("results"):
        extracted = "\n\n".join(
            (r.get("full_text") or "").strip()
            for r in analysis["results"]
            if isinstance(r, dict) and (r.get("full_text") or "").strip()
        )

    return {
        "extracted_text": extracted,
        "categories": analysis.get("categories", []),
        "category_counts": analysis.get("category_counts", {}),
        "results": analysis.get("results", []),
        "analysis": analysis,
    }


@router.post("/generate-output")
def generate_output(body: GenerateOutputRequest):
    """
    Build final text and next-module payload from analysis JSON,
    user-selected category, and depth (summary vs full).
    """
    out = generate_selected_output(
        body.analysis,
        body.selected_category,
        body.depth,
    )

    if out.get("status") == "ERROR":
        raise HTTPException(status_code=400, detail=out)

    payload = out.get("next_module_payload") or {}
    next_text = payload.get("text", "") if isinstance(payload, dict) else str(payload)

    return {
        "final_output_text": out.get("final_output_text", ""),
        "next_module_payload": next_text,
    }

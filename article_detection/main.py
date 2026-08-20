"""
FastAPI entrypoint: image analysis and selected output generation.
Loads pipeline helpers from pipeline.py. Keep API keys in environment only.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Annotated, Any

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Load .env before pipeline import (pipeline reads OPENAI_API_KEY, paths at import time).
load_dotenv()

from pipeline import analyze_content, generate_selected_output

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ALLOW_ORIGINS = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173",
).split(",")

app = FastAPI(title="Content Analysis API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOW_ORIGINS if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateOutputRequest(BaseModel):
    """Full `analysis` object as returned by analyze_content (serializable JSON)."""

    analysis: dict[str, Any]
    selected_category: str
    depth: str = Field(
        ...,
        description="Either 'summary' or 'full' (same as pipeline.generate_selected_output).",
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(
    document_type: Annotated[str, Form()],
    file: Annotated[UploadFile, File()],
):
    """
    Accept an image and document type, save to a temp file, run analyze_content,
    return extracted text, categories, category_counts, and per-item results.
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
        fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix="upload_")
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
            detail="Analysis failed. Check server logs and dependencies (Tesseract, OpenCV, model file).",
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

    categories = (
        analysis.get("categories")
        or analysis.get("detected_categories")
        or []
    )

    fallback = analysis.get("fallback") if isinstance(analysis.get("fallback"), dict) else {}

    return {
        "extracted_text": extracted,
        "local_extracted_text": analysis.get("local_extracted_text") or extracted,
        "ai_extracted_text": analysis.get("ai_extracted_text") or "",
        "categories": categories,
        "category_counts": analysis.get("category_counts", {}),
        "results": analysis.get("results", []),
        "local_results": analysis.get("local_results") or analysis.get("results") or [],
        "ai_results": analysis.get("ai_results") or [],
        "article_fallback_used": bool(fallback.get("article_fallback_used")),
        "local_quality": fallback.get("local_quality") or {},
        "pipeline_version": analysis.get("pipeline_version"),
        "analysis": analysis,
    }


@app.post("/generate-output")
def generate_output(body: GenerateOutputRequest):
    """
    Build final text and next-module payload from a stored analysis JSON,
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=True,
    )

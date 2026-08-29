import os
import uuid

from fastapi import APIRouter, File, Form, UploadFile

from backend.abhishek_module.pipeline import analyze_page, describe_image_only


router = APIRouter(
    prefix="/abhishek",
    tags=["Abhishek Module"],
)

PROJECT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

UPLOAD_DIR = os.path.join(PROJECT_DIR, "backend", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.get("/status")
def abhishek_module_status():
    from backend.abhishek_module.pipeline import _selected_image_description_model

    return {
        "status": "ok",
        "module": "abhishek_module",
        "endpoint": "/abhishek/analyze",
        "describe_image_endpoint": "/abhishek/describe-image",
        "image_description_model": _selected_image_description_model(),
        "message": "Abhishek module is ready.",
    }


@router.post("/analyze")
async def analyze_uploaded_page(file: UploadFile = File(...)):
    """
    Accept a page image, run analyze_page, return fixed JSON.
    """
    original_name = file.filename or "captured_page.jpg"
    ext = os.path.splitext(original_name)[1] or ".jpg"
    unique_name = f"{uuid.uuid4().hex}{ext}"
    save_path = os.path.join(UPLOAD_DIR, unique_name)

    try:
        with open(save_path, "wb") as buffer:
            buffer.write(await file.read())
    except Exception:
        return {
            "document_type": "Unknown",
            "confidence": 0.0,
            "title": "Title not clearly detected",
            "image_descriptions": [],
            "status": "failed",
            "warnings": ["Document identification failed"],
        }

    return analyze_page(save_path)


@router.post("/describe-image")
async def describe_uploaded_image(
    file: UploadFile = File(...),
    context: str = Form("general"),
):
    """
    Image-description-only endpoint for reusable Florence captions.

    Intended for:
      - novel_cover
      - novel_page illustrations
      - magazine_page
      - newspaper_article_image (cropped article photo from Harshaka)
      - newspaper_page
      - general
    """
    original_name = file.filename or "selected_image.jpg"
    ext = os.path.splitext(original_name)[1] or ".jpg"
    unique_name = f"{uuid.uuid4().hex}{ext}"
    save_path = os.path.join(UPLOAD_DIR, unique_name)

    try:
        with open(save_path, "wb") as buffer:
            buffer.write(await file.read())
    except Exception:
        return {
            "has_image": False,
            "image_description": "",
            "context": (context or "general").strip() or "general",
            "status": "failed",
            "warnings": ["Image description failed"],
        }

    return describe_image_only(save_path, context=context or "general")

"""
LEGACY / DEV ONLY image description routes (OpenAI).

These endpoints are NOT part of the main PP2 research pipeline.
Abhishek's main path is:
  POST /abhishek/analyze  →  local_blip_description.describe_image_local

This file is not used in the main research pipeline.
Main PP2 pipeline uses local_blip_description.py.
"""

import os
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException
from backend.image_description.openai_description import (
    describe_uploaded_image,
    image_description_available,
)

router = APIRouter(prefix="/image", tags=["Image Description (Legacy OpenAI)"])

UPLOAD_DIR = "backend/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.get("/status")
def image_description_status():
    return {
        "module": "Image Description",
        "mode": "legacy_openai_dev_only",
        "pp2_pipeline": "use POST /abhishek/analyze with local_blip_description",
        "openai_available": image_description_available(),
        "message": (
            "Legacy OpenAI image description status checked successfully. "
            "Main research pipeline uses local_blip_description.py."
        ),
    }


@router.post("/describe-image")
async def describe_image(file: UploadFile = File(...)):
    """Legacy/dev OpenAI endpoint. Prefer /abhishek/analyze for PP2."""
    if not image_description_available():
        raise HTTPException(
            status_code=500,
            detail="OpenAI image description module is not available."
        )

    try:
        ext = os.path.splitext(file.filename)[1]
        unique_name = f"{uuid.uuid4().hex}{ext}"
        save_path = os.path.join(UPLOAD_DIR, unique_name)

        with open(save_path, "wb") as f:
            f.write(await file.read())

        result = describe_uploaded_image(save_path)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

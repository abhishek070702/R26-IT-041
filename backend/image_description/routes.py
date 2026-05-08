import os
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException
from backend.image_description.openai_description import (
    describe_uploaded_image,
    image_description_available,
)

router = APIRouter(prefix="/image", tags=["Image Description"])

UPLOAD_DIR = "backend/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.get("/status")
def image_description_status():
    return {
        "module": "Image Description",
        "openai_available": image_description_available(),
        "message": "Image description module status checked successfully."
    }


@router.post("/describe-image")
async def describe_image(file: UploadFile = File(...)):
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
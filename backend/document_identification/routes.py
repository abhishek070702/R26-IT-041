import os
import shutil

from fastapi import APIRouter, UploadFile, File

from backend.document_identification.model_loader import (
    predict_document_type,
    get_model_status
)

from backend.document_identification.title_reader import (
    easyocr_extract_title,
    create_device_message
)


router = APIRouter(
    prefix="/document",
    tags=["Document Identification"]
)


PROJECT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

UPLOAD_DIR = os.path.join(
    PROJECT_DIR,
    "backend",
    "uploads"
)

os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.get("/status")
def document_model_status():
    return {
        "status": "ok",
        "module": "document_identification",
        **get_model_status()
    }


@router.post("/predict-document-type")
async def predict_document_type_api(file: UploadFile = File(...)):
    file_path = os.path.join(
        UPLOAD_DIR,
        file.filename
    )

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    document_type, confidence, all_predictions = predict_document_type(file_path)

    title = None

    if document_type in ["Novel", "Magazine"]:
        title = easyocr_extract_title(
            file_path,
            document_type
        )

    final_message = create_device_message(
        document_type,
        title
    )

    return {
        "document_type": document_type,
        "confidence": confidence,
        "all_predictions": all_predictions,
        "title": title,
        "final_message": final_message
    }
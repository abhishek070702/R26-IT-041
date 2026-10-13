from __future__ import annotations

import logging
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.document_identification.routes import router as document_router
from backend.image_description.routes import router as image_router

from backend.abhishek_module.routes import router as abhishek_router

logger = logging.getLogger(__name__)


def _preload_configured_image_description_model() -> None:
    """Warm frequently used local models without blocking API startup."""
    started = time.perf_counter()
    try:
        from backend.abhishek_module.pipeline import _selected_image_description_model

        selected = _selected_image_description_model()
        if selected == "florence":
            from backend.image_description.local_florence_description import (
                preload_florence_model,
            )

            preload_florence_model()
            elapsed = time.perf_counter() - started
            print(f"[startup] Florence preload completed in {elapsed:.2f}s")
            logger.info("Florence preload completed in %.2fs", elapsed)
        else:
            from backend.image_description.local_blip_description import (
                preload_blip_model,
            )

            preload_blip_model()
            elapsed = time.perf_counter() - started
            print(f"[startup] BLIP preload completed in {elapsed:.2f}s")
            logger.info("BLIP preload completed in %.2fs", elapsed)
    except Exception as exc:
        elapsed = time.perf_counter() - started
        print(f"[startup] image-description preload failed after {elapsed:.2f}s: {exc}")
        logger.exception("image-description preload failed after %.2fs", elapsed)

    ocr_started = time.perf_counter()
    try:
        from backend.abhishek_module.pipeline import _selected_title_reader_model

        title_model = _selected_title_reader_model()
        if title_model == "paddleocr":
            from backend.document_identification.paddle_title_reader import (
                preload_paddleocr,
            )

            preload_paddleocr()
            print(
                f"[startup] PaddleOCR preload completed in "
                f"{time.perf_counter() - ocr_started:.2f}s"
            )
            logger.info(
                "PaddleOCR preload completed in %.2fs",
                time.perf_counter() - ocr_started,
            )
        else:
            from backend.document_identification.title_reader import preload_easyocr

            preload_easyocr()
            print(
                f"[startup] EasyOCR preload completed in "
                f"{time.perf_counter() - ocr_started:.2f}s"
            )
            logger.info(
                "EasyOCR preload completed in %.2fs",
                time.perf_counter() - ocr_started,
            )
    except Exception as exc:
        elapsed = time.perf_counter() - ocr_started
        print(f"[startup] title-OCR preload failed after {elapsed:.2f}s: {exc}")
        logger.exception("title-OCR preload failed after %.2fs", elapsed)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    thread = threading.Thread(
        target=_preload_configured_image_description_model,
        name="model-preload",
        daemon=True,
    )
    thread.start()
    yield


app = FastAPI(
    title="Smart Wearable Reading Assistant API",
    description=(
        "Backend API for Document Identification, Image Description, "
        "and Content Summarization & Category Detection. "
        "Main Abhishek PP2 research path: POST /abhishek/analyze "
        "and reusable POST /abhishek/describe-image "
        "(local Florence/BLIP image description)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://172.27.206.233:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(document_router)
app.include_router(image_router)
app.include_router(abhishek_router)


@app.get("/")
def home():
    return {
        "message": "Smart Wearable Reading Assistant Backend is running",
        "modules": [
            "Document Identification",
            "Image Description (legacy OpenAI /image routes)",
            "Content Summarization & Category Detection",
            "Abhishek Module (PP2: /abhishek/analyze, /abhishek/describe-image)",
        ],
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "backend": "running"
    }


print("Main backend loaded successfully")
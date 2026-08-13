from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.document_identification.routes import router as document_router
from backend.image_description.routes import router as image_router

from backend.abhishek_module.routes import router as abhishek_router


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
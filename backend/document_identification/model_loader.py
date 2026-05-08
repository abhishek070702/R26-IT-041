import os
import json

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


# =========================
# Project paths
# =========================

# This automatically finds your main project folder.
# Works on Windows and Raspberry Pi.
PROJECT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "model",
    "best_document_model_finetuned.pth"
)

CLASS_NAMES_PATH = os.path.join(
    PROJECT_DIR,
    "model",
    "class_names.json"
)


# =========================
# Device
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# Load class names
# =========================

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

num_classes = len(class_names)


# =========================
# Image transforms
# =========================

val_test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# =========================
# Load MobileNetV2 model
# =========================

model = models.mobilenet_v2(weights=None)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    num_classes
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()


# =========================
# Prediction function
# =========================

def predict_document_type(image_path):
    image = Image.open(image_path).convert("RGB")

    input_tensor = val_test_transforms(image)
    input_tensor = input_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)

        top_probs, top_classes = torch.topk(
            probabilities,
            len(class_names)
        )

    best_index = top_classes[0][0].item()
    best_class = class_names[best_index]
    best_confidence = top_probs[0][0].item() * 100

    all_predictions = []

    for i in range(len(class_names)):
        class_index = top_classes[0][i].item()
        class_name = class_names[class_index]
        confidence = top_probs[0][i].item() * 100

        all_predictions.append({
            "class_name": class_name,
            "confidence": round(confidence, 2)
        })

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return best_class, round(best_confidence, 2), all_predictions


def get_model_status():
    return {
        "device": str(device),
        "document_model_gpu": torch.cuda.is_available(),
        "classes": class_names,
        "project_dir": PROJECT_DIR,
        "model_path": MODEL_PATH
    }


print("Document Identification model loaded successfully")
print("Project folder:", PROJECT_DIR)
print("Model path:", MODEL_PATH)
print("Using device:", device)
print("Classes:", class_names)
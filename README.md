# Smart Wearable Reading Assistant - Document Identification and Image Description

This project is part of the Smart Wearable Reading Assistant research project.

The main goal of this module is to help visually impaired users understand printed materials by identifying the document type and describing important images inside printed pages.

---

## Main Components

### 1. Document Identification

This module detects the type of printed document.

Supported document classes:

- Magazine
- Newspaper
- Novel
- Printed Letter
- Report

If the document is a Novel or Magazine, the system also tries to extract the title and gives a simple instruction to the user.

Example output:

```text
This is a Novel. The title is Little Frog. Please turn the next page.

2. Image Description

This module describes visual content inside printed pages.

It is mainly used for:

Storybook inside-page images
Newspaper photos
Magazine images
Textbook diagrams
Figures and illustrations

This module is not mainly for novel/book covers. If a novel cover is detected, the system can skip image description.

Example output:

The first picture shows a boy falling into water, and a man trying to help him. The second picture shows a princess, a king, and a kangaroo near a castle.
Technologies Used
Backend
Python
FastAPI
PyTorch
TorchVision
EasyOCR
OpenAI Vision API
Python Dotenv
Frontend
React
Vite
TypeScript
CSS
Backend APIs
Home
GET /
Health Check
GET /health
Document Identification Status
GET /document/status
Predict Document Type
POST /document/predict-document-type
Image Description Status
GET /image/status
Describe Image
POST /image/describe-image
Project Structure
backend/
  document_identification/
    __init__.py
    model_loader.py
    title_reader.py
    routes.py

  image_description/
    __init__.py
    openai_description.py
    routes.py

  __init__.py
  main.py
  requirements.txt
  uploads/

frontend/
  src/
    App.tsx
    App.css
    main.tsx

model/
  best_document_model_finetuned.pth
  class_names.json

.gitignore
README.md
Important Security Note

The .env file is not uploaded to GitHub for security reasons.

To run the image description module, create a .env file inside the backend folder:

backend/.env

Add your OpenAI API key:

OPENAI_API_KEY=your_api_key_here

Do not share or upload your real API key.

Required Model Files

The document identification module requires these files inside the model folder:

model/
  best_document_model_finetuned.pth
  class_names.json

The .pth file contains the trained PyTorch model.

The class_names.json file contains the class labels used by the model.

How the System Works
Document Identification Flow
User uploads or captures an image
        ↓
Frontend sends image to backend
        ↓
Backend loads trained PyTorch model
        ↓
Model predicts document type
        ↓
If Novel/Magazine, EasyOCR extracts title
        ↓
Backend returns final message
Image Description Flow
User uploads or captures an inside-page image
        ↓
Frontend sends image to backend
        ↓
Backend sends image to OpenAI Vision API
        ↓
OpenAI returns simple image description
        ↓
Backend returns final message
Run Backend

First, open terminal in the project root folder.

Activate the virtual environment:

venv_easyocr\Scripts\activate

Run the FastAPI backend:

uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

Backend Swagger documentation:

http://127.0.0.1:8000/docs
Run Frontend

Open another terminal and go to the frontend folder:

cd frontend

Run the frontend:

npx vite --host 0.0.0.0 --port 5173

Open in browser:

http://localhost:5173

For phone testing on the same Wi-Fi network, use the network URL shown by Vite.

Example:

http://YOUR_LAPTOP_IP:5173
Raspberry Pi Integration Idea

For Raspberry Pi integration, notebooks are not required.

Only the following are needed:

Final trained model files
Python inference code
Backend modules
Required libraries
Camera input
OpenAI API key for image description
Internet connection for OpenAI image description

Basic Raspberry Pi flow:

Camera Guidance
        ↓
Capture clear page image
        ↓
Document Identification
        ↓
OCR and Summarization
        ↓
Image Description if images are available
        ↓
Text-to-Speech output

The document identification model can run locally on the Raspberry Pi.
The image description module uses OpenAI API, so it requires internet access.

Research Purpose

This project supports visually impaired users by helping them understand printed materials.

The system can identify the type of document, read important title information, and describe images inside pages. This module can be integrated with other group modules such as camera guidance, OCR, content summarization, category detection, personalized reading, and text-to-speech.

Author

Smart Wearable Reading Assistant Research Project
Document Identification and Image Description Module


After pasting, save:

```text
Ctrl + S
# Smart Wearable Reading Assistant for the Visually Impaired

## Project ID

**R26-IT-041**

---

## Project Overview

The **Smart Wearable Reading Assistant for the Visually Impaired** is an assistive technology project designed to help blind and visually impaired users access printed materials more independently.

Many visually impaired users face difficulties when reading printed documents such as **newspapers, magazines, novels, printed letters, reports, and educational materials**. They may depend on another person to identify the document, read the text, understand images, and continue reading page by page.

This system aims to reduce that dependency by using a smart wearable device that can capture printed pages, identify the document type, extract and process text, summarize content, describe images, and deliver the final output through personalized Text-to-Speech.

---

## Main Objective

The main objective of this project is to develop a smart wearable reading assistant that supports visually impaired users by converting printed materials into meaningful, personalized, and easy-to-understand audio output.

---

## Key Features

- Wearable camera-based printed page capture
- Automatic camera position guidance
- RFID-based user identification
- User preference registration and loading
- Document type identification
- Title reading for novels and magazines
- OCR-based text extraction
- Content summarization
- Category detection for newspaper content
- User-selected category reading
- Image description for inside-page visuals
- Personalized Text-to-Speech output
- Page completion and turn-page guidance

---

## Target Users

This system is designed for:

- Visually impaired students
- Office workers with visual impairment
- Elderly readers with low vision
- Blind or low-vision general readers
- Schools, universities, libraries, NGOs, and rehabilitation centers

---

## Overall System Workflow

```text
RFID User Identification
        ↓
New User Registration / Existing User Preference Loading
        ↓
Automatic Camera Position Guidance
        ↓
Page Capture
        ↓
Document Type Identification
        ↓
OCR Text Extraction
        ↓
Category Detection / Summarization
        ↓
Image Description if Required
        ↓
Personalized Reading Preparation
        ↓
Text-to-Speech Output
        ↓
Page Completion Detection
        ↓
Prompt User to Turn Next Page


How the System Works

First, the device identifies the user using an RFID card. If the user is new, the system registers their reading preferences such as reading level, preferred voice, reading speed, and tone. If the user is already registered, the system suggests their previously saved preferences.

After that, the wearable camera captures the printed page with the help of automatic camera position guidance. The system then identifies the document type, such as a novel, magazine, newspaper, printed letter, or report.

If the document is a novel or magazine cover, the system reads the title using OCR and asks the user to turn the next page. For inside pages, the system extracts the text, detects categories where needed, summarizes or reads full text based on user preference, describes important images, and finally sends the processed content to the personalized Text-to-Speech module.

Main System Functions
1. Automatic Camera Position Guidance

This function helps visually impaired users capture a clear image of the printed page. Since the user may not know whether the camera is correctly positioned, this module gives voice guidance to help align the page properly.

Main Responsibilities
Detect whether the page is visible
Guide the user to adjust the camera position
Provide voice instructions such as move up, move down, move left, move right, or ready
Automatically capture the image when the page is clear and stable
Send the captured image to other modules
Importance

A clear page image is important because all other modules depend on it. If the captured image is tilted, cropped, too far, or unclear, document identification, OCR, summarization, and image description may give inaccurate results.

2. Document Identification and Image Description

This function identifies the type of printed material captured by the camera. It supports printed materials such as novels, magazines, newspapers, printed letters, and reports.

Main Responsibilities
Identify the printed document type
Inform the user about the detected document type
Read the title for novels and magazines using OCR
Skip unnecessary image description for novel and magazine covers
Describe important images, figures, or illustrations inside printed pages
Send simple and meaningful output to the Text-to-Speech module
Supported Document Classes
Novel
Magazine
Newspaper
Printed Letter
Report
How It Works

The captured image is passed to a trained MobileNetV2 document identification model. The model predicts the document type. If the detected document is a novel or magazine, EasyOCR is used to extract and read the title. For inside pages, if there are images or illustrations, the image description module explains them in simple language.

Importance

This module acts as the first document understanding layer of the system. It helps the device decide how the captured printed material should be processed next.

3. Content Summarization and Category Detection

This function processes the text extracted from the captured page. It helps users listen only to the content they prefer instead of reading the full page automatically.

Main Responsibilities
Extract text using OCR
Identify content categories
Support multi-category detection for newspapers
Support single-category detection for novels, textbooks, reports, and letters
Allow users to choose a preferred category
Allow users to choose summary or full-text reading
Send processed content to the personalized reading module
How It Works

For newspapers, one page may contain multiple categories such as politics, sports, business, education, or entertainment. The system identifies these categories and asks the user which category they want to hear first.

For single-category documents such as novels or reports, the system confirms the content type and asks whether the user wants a summary or the full text.

Importance

This function makes reading more user-focused and saves time by allowing users to access only the content they need.

4. Personalized-Based Reading

This function personalizes the final reading output according to the user’s needs and preferences.

Main Responsibilities
Identify the user using RFID
Register preferences for new users
Load previous preferences for registered users
Support reading level selection
Support preferred voice selection
Support reading speed selection
Support tone selection
Prepare final content for Text-to-Speech
Detect page completion and ask the user to turn the page
User Preferences
Reading level: Very simple, moderate, or light
Preferred voice: Male or female
Reading speed: Slow, normal, or fast
Tone: Playful, formal, friendly, or other suitable tone


Full Project Integration

Automatic Camera Position Guidance
        ↓
Captures a clear page image

Document Identification and Image Description
        ↓
Identifies document type, reads titles, and describes images

Content Summarization and Category Detection
        ↓
Extracts text, detects categories, and prepares summary or full text

Personalized-Based Reading
        ↓
Adjusts output based on user preferences and sends it to TTS




Technologies Used

Backend

Python
FastAPI
Uvicorn
PyTorch
MobileNetV2
EasyOCR
OpenAI Vision API
Pillow
Python-dotenv


Frontend

React
TypeScript
Vite
CSS


AI / Machine Learning

MobileNetV2
PyTorch
OCR
Image classification
Image description support
Text processing


Hardware / Integration

Raspberry Pi 5
Camera module
RFID module
Microphone
Speaker / earphones
Wearable device concept
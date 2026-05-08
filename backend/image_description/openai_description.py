import os
import base64
from io import BytesIO

from PIL import Image
from dotenv import load_dotenv
from openai import OpenAI
from fastapi import HTTPException


PROJECT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

ENV_PATH = os.path.join(
    PROJECT_DIR,
    "backend",
    ".env"
)

load_dotenv(ENV_PATH)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

openai_client = None

if OPENAI_API_KEY:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)


def image_description_available():
    return openai_client is not None


def image_to_base64_for_openai(image_path):
    image = Image.open(image_path).convert("RGB")

    # Smaller image = faster API response
    max_size = 650
    width, height = image.size

    if width > height:
        if width > max_size:
            new_height = int(height * max_size / width)
            image = image.resize((max_size, new_height))
    else:
        if height > max_size:
            new_width = int(width * max_size / height)
            image = image.resize((new_width, max_size))

    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=65)

    image_bytes = buffer.getvalue()
    base64_image = base64.b64encode(image_bytes).decode("utf-8")

    return f"data:image/jpeg;base64,{base64_image}"


def describe_uploaded_image(image_path):
    if openai_client is None:
        raise HTTPException(
            status_code=500,
            detail="OpenAI API key is missing. Please check backend/.env file."
        )

    image_data_url = image_to_base64_for_openai(image_path)

    prompt = """
Describe the important pictures on this printed page in very simple English.

Rules:
- Use 2 short sentences only.
- Maximum 35 words.
- If there are multiple pictures, describe them briefly.
- Do not read paragraphs.
- If this is a novel/book cover, say exactly: NOVEL_COVER_SKIP
"""

    try:
        response = openai_client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": prompt
                        },
                        {
                            "type": "input_image",
                            "image_url": image_data_url
                        }
                    ]
                }
            ],
            max_output_tokens=70
        )

        description = response.output_text.strip()

        if description == "NOVEL_COVER_SKIP":
            return {
                "description": "Novel cover detected. Description skipped.",
                "final_message": "This is a novel book cover. Image description is skipped."
            }

        if description == "":
            description = "There is no clear picture to describe."

        return {
            "description": description,
            "final_message": f"Image description: {description}"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"OpenAI image description failed: {str(e)}"
        )


def get_image_description_status():
    return {
        "openai_image_description": OPENAI_API_KEY is not None,
        "env_path": ENV_PATH
    }


print("OpenAI image description module loaded successfully")
print("Project folder:", PROJECT_DIR)
print("ENV path:", ENV_PATH)
print("OpenAI Image Description:", OPENAI_API_KEY is not None)
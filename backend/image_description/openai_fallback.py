"""
Optional OpenAI Vision fallback for difficult title / image-description cases.

Main PP2 research mode is local-only.
This module is used only when USE_OPENAI_FALLBACK=true.
It is separate from openai_description.py (legacy /image route).
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
from io import BytesIO

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Keep USE_OPENAI_FALLBACK reading in sync with backend/.env.
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ENV_PATH = os.path.join(PROJECT_DIR, "backend", ".env")
load_dotenv(ENV_PATH, override=True)

FALLBACK_PROMPT = """
You are helping a visually impaired reader.

Look only at the reading material / document / book / magazine / newspaper page in this image.
Ignore any real-world background such as a hand, table, bottle, wall, desk, or room.

Return JSON only, with exactly these keys:
{
  "title": "...",
  "image_description": "..."
}

Rules:
- title: extract the main document, book, magazine, or newspaper title if it is visible.
  If no title is clearly visible, return an empty string.
  Do not invent a title.
- image_description: one clear sentence a blind user can use to imagine the cover/page.
  Describe the main subject, people/animals/objects, action or scene, and style/mood if useful.
  Use "appears to show" when uncertain.
  Do not invent too much.
- Do not mention hand, table, bottle, wall, or other real background objects.
- Do not include markdown or extra text. JSON only.
""".strip()


def is_openai_fallback_enabled() -> bool:
    raw = os.getenv("USE_OPENAI_FALLBACK", "false")
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _image_to_data_url(image_path: str) -> str:
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    max_size = 768
    width, height = image.size
    longest = max(width, height)
    if longest > max_size:
        scale = max_size / float(longest)
        image = image.resize(
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            Image.Resampling.LANCZOS,
        )

    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=80)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def _parse_json_payload(text: str) -> dict | None:
    if not text or not str(text).strip():
        return None

    cleaned = str(text).strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        return None

    try:
        data = json.loads(match.group(0))
    except Exception:
        return None

    if not isinstance(data, dict):
        return None
    return data


def run_openai_vision_fallback(image_path: str) -> dict | None:
    """
    Ask OpenAI Vision for title + image_description JSON.

    Returns {"title": str, "image_description": str} or None on failure.
    Never raises to the caller.
    """
    if not is_openai_fallback_enabled():
        return None

    if not image_path or not os.path.isfile(image_path):
        logger.warning("OpenAI fallback skipped: missing image path=%r", image_path)
        print("[openai_fallback] failed: missing image path")
        return None

    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        logger.warning("OpenAI fallback failed: OPENAI_API_KEY missing")
        print("[openai_fallback] failed: OPENAI_API_KEY missing")
        return None

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        image_data_url = _image_to_data_url(image_path)
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": FALLBACK_PROMPT},
                        {"type": "input_image", "image_url": image_data_url},
                    ],
                }
            ],
            max_output_tokens=180,
        )
        raw_text = (response.output_text or "").strip()
        print(f"[openai_fallback] raw response: {raw_text!r}")
        logger.info("OpenAI fallback raw response=%r", raw_text)

        data = _parse_json_payload(raw_text)
        if data is None:
            print("[openai_fallback] failed: could not parse JSON")
            logger.warning("OpenAI fallback JSON parse failed")
            return None

        title = str(data.get("title") or "").strip()
        description = str(
            data.get("image_description") or data.get("description") or ""
        ).strip()

        unusable_titles = {
            "",
            "none",
            "null",
            "n/a",
            "unknown",
            "title not clearly detected",
            "newspaper name not clearly detected",
        }
        if title.lower() in unusable_titles:
            title = ""

        if not title and not description:
            print("[openai_fallback] failed: empty title and description")
            return None

        return {
            "title": title,
            "image_description": description,
        }
    except Exception:
        logger.exception("OpenAI fallback failed for path=%r", image_path)
        print("[openai_fallback] failed: exception")
        return None

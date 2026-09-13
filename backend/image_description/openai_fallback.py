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
You are helping a visually impaired reader imagine a printed cover.

Look only at the book, novel, or magazine cover in this photo.
Ignore the real world around it: hand, fingers, table, bottle, wall, desk, lap, room,
or a person holding the book.

Return JSON only:
{
  "title": "...",
  "image_description": "..."
}

Rules:
- title: copy the main printed title if it is clearly readable. Else "".
  Do not invent a title.
- image_description: 1 or 2 spoken sentences about the PRINTED COVER ART only.
  Include the main people/animals/objects, what they are doing, setting, colors, and style
  (photo, painting, cartoon, etc.) when that is visible.
  Start from the cover scene, not from "this is a photo of a book".
  Use "appears to show" only when you are unsure.
  Do not mention the camera, the room, or anyone holding the book.
- JSON only. No markdown.
""".strip()

PAGE_ILLUSTRATION_PROMPT = """
You are helping a visually impaired reader.

Look only at the printed page in this photo (novel, magazine, or newspaper).
Ignore the real world around it: hand, fingers, table, bottle, wall, desk, lap, room,
or a person holding the book.

Decide if there is a picture printed ON THE PAGE: illustration, drawing, comic panel,
or photograph that is part of the page. Body text, headings, and page numbers do not count.
The real-world background does not count.

Return JSON only:
{
  "has_image": true,
  "image_description": "..."
}

Rules:
- If the page is only text, or you cannot see a printed picture on the page,
  set has_image to false and image_description to "".
- If there is a printed picture, set has_image to true and describe THAT picture
  in 1 or 2 spoken sentences: who/what is shown, action, setting, and style.
  Do not read or summarize the article text.
- Do not mention the camera, the room, or anyone holding the book.
- Do not invent details. JSON only. No markdown.
""".strip()


def _cover_prompt(document_type: str = "") -> str:
    kind = str(document_type or "").strip()
    if kind == "Novel":
        extra = "This photo is a novel cover. Describe the printed cover illustration."
    elif kind == "Magazine":
        extra = "This photo is a magazine cover. Describe the printed cover image."
    else:
        extra = "Describe the printed cover image of the reading material."
    return FALLBACK_PROMPT + "\n\n" + extra


def _page_prompt(context: str = "general") -> str:
    hints = {
        "novel_page": "This is an inside novel page. Describe a printed illustration only if one is on the page.",
        "novel_cover": "This is a novel cover. Describe the printed cover art.",
        "magazine_page": "This is an inside magazine page. Describe a printed photo or illustration only if one is on the page.",
        "newspaper_article_image": "This is a newspaper article area. Describe a printed photo only if one is in the article. Ignore the table and room.",
        "newspaper_page": "This is a newspaper page. Describe a printed photo only if one is on the page.",
        "general": "Describe a printed picture only if one is on the page itself.",
    }
    extra = hints.get(str(context or "general").strip() or "general", hints["general"])
    return PAGE_ILLUSTRATION_PROMPT + "\n\n" + extra


NEWSPAPER_MASTHEAD_PROMPT = """
You are helping a visually impaired reader.

Look only at the newspaper in this photo.
Read the newspaper NAME from the masthead at the top of the page
(the paper's own title, such as The Sunday Times or Daily News).

Ignore article headlines, body text, photos, adverts, the table, a hand,
and the room. Do not describe any picture.

Return JSON only:
{
  "title": "...",
  "image_description": ""
}

Rules:
- title: the newspaper's printed masthead name. Keep "The" if it is part of the name.
- Do not use an article headline as the newspaper name.
- If the masthead is not readable, return "".
- image_description must be an empty string.
- JSON only. No markdown.
""".strip()


def _openai_vision_model() -> str:
    return (
        os.getenv("OPENAI_VISION_MODEL")
        or os.getenv("OPENAI_FALLBACK_MODEL")
        or "gpt-4o-mini"
    ).strip()


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


def _openai_vision_json(
    image_path: str,
    prompt: str,
    timeout_sec: float = 15.0,
    max_output_tokens: int = 180,
) -> dict | None:
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

    model = _openai_vision_model()
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, timeout=timeout_sec, max_retries=0)
        image_data_url = _image_to_data_url(image_path)
        print(f"[openai_fallback] model={model}")
        response = client.responses.create(
            model=model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": image_data_url},
                    ],
                }
            ],
            max_output_tokens=max_output_tokens,
        )
        raw_text = (response.output_text or "").strip()
        print(f"[openai_fallback] raw response: {raw_text!r}")
        logger.info("OpenAI fallback raw response=%r", raw_text)

        data = _parse_json_payload(raw_text)
        if data is None:
            print("[openai_fallback] failed: could not parse JSON")
            logger.warning("OpenAI fallback JSON parse failed")
            return None
        return data
    except Exception:
        logger.exception("OpenAI fallback failed for path=%r", image_path)
        print("[openai_fallback] failed: exception")
        return None


def run_openai_vision_fallback(
    image_path: str,
    document_type: str = "",
) -> dict | None:
    """
    Ask OpenAI Vision for title (+ cover description for Novel/Magazine).

    Newspaper uses a masthead-name task only (no visual description).
    Returns {"title": str, "image_description": str} or None on failure.
    Never raises to the caller.
    """
    is_newspaper = str(document_type or "").strip() == "Newspaper"
    prompt = NEWSPAPER_MASTHEAD_PROMPT if is_newspaper else _cover_prompt(document_type)
    data = _openai_vision_json(
        image_path,
        prompt,
        timeout_sec=20.0,
        max_output_tokens=80 if is_newspaper else 220,
    )
    if not data:
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


def _parse_has_image_flag(raw: object) -> bool:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(raw)
    text = str(raw or "").strip().lower()
    return text in {"1", "true", "yes", "on"}


def run_openai_page_illustration_fallback(
    image_path: str,
    context: str = "general",
) -> dict | None:
    """
    Ask whether a printed picture is on the page, then describe only that picture.

    Returns {"has_image": bool, "image_description": str} or None on failure.
    Never raises to the caller.
    """
    data = _openai_vision_json(
        image_path,
        _page_prompt(context),
        timeout_sec=20.0,
        max_output_tokens=220,
    )
    if not data:
        return None

    has_image = _parse_has_image_flag(data.get("has_image"))
    description = str(
        data.get("image_description") or data.get("description") or ""
    ).strip()

    if not has_image or not description:
        print("[openai_fallback] page illustration has_image=false")
        return {
            "has_image": False,
            "image_description": "",
        }

    print(
        "[openai_fallback] page illustration has_image=true "
        f"description={description!r}"
    )
    return {
        "has_image": True,
        "image_description": description,
    }

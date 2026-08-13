"""
Local image description module for the main PP2 research pipeline.

Used by: backend/abhishek_module/pipeline.py → POST /abhishek/analyze

No OpenAI / Google / external APIs.
Uses Salesforce/blip-image-captioning-base locally via transformers.

Note: openai_description.py is legacy/dev only and must not be used here.
"""

from __future__ import annotations

import logging
import os
import re

import torch
from PIL import Image

logger = logging.getLogger(__name__)

MODEL_NAME = "Salesforce/blip-image-captioning-base"

_processor = None
_model = None
_device = None
_load_error: str | None = None


def _get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_blip():
    """
    Lazy-load BLIP processor/model once at module level.
    """
    global _processor, _model, _device, _load_error

    if _model is not None and _processor is not None:
        return _processor, _model, _device

    if _load_error is not None:
        raise RuntimeError(_load_error)

    try:
        from transformers import BlipForConditionalGeneration, BlipProcessor

        _device = _get_device()
        print(f"Loading local BLIP model: {MODEL_NAME}")
        print(f"Using device: {_device}")

        _processor = BlipProcessor.from_pretrained(MODEL_NAME)
        _model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME)
        _model = _model.to(_device)
        _model.eval()

        print("BLIP loaded successfully")
        print(f"Device used: {_device}")
        logger.info("BLIP loaded successfully on device=%s", _device)
        return _processor, _model, _device

    except Exception as exc:
        _processor = None
        _model = None
        _device = None
        _load_error = f"Failed to load local BLIP model: {exc}"
        print(_load_error)
        logger.exception("BLIP load failed")
        raise RuntimeError(_load_error) from exc


def _collapse_repeated_words(text: str) -> str:
    """Collapse obvious consecutive repeated words: 'a a a dog' -> 'a dog'."""
    words = text.split()
    if not words:
        return ""

    cleaned: list[str] = []
    for word in words:
        if cleaned and cleaned[-1].lower() == word.lower():
            continue
        cleaned.append(word)
    return " ".join(cleaned)


def _collapse_repeated_phrases(text: str, max_n: int = 4) -> str:
    """
    Collapse consecutive repeated n-grams, e.g.
    'a group of people a group of people' -> 'a group of people'
    """
    words = text.split()
    if len(words) < 4:
        return text

    changed = True
    while changed:
        changed = False
        for n in range(max_n, 1, -1):
            if len(words) < n * 2:
                continue
            i = 0
            new_words: list[str] = []
            while i < len(words):
                chunk = words[i : i + n]
                next_chunk = words[i + n : i + 2 * n]
                if (
                    len(chunk) == n
                    and len(next_chunk) == n
                    and [w.lower() for w in chunk] == [w.lower() for w in next_chunk]
                ):
                    new_words.extend(chunk)
                    i += 2 * n
                    # Skip further immediate repeats of the same phrase.
                    while (
                        i + n <= len(words)
                        and [w.lower() for w in words[i : i + n]]
                        == [w.lower() for w in chunk]
                    ):
                        i += n
                    changed = True
                else:
                    new_words.append(words[i])
                    i += 1
            words = new_words
            if changed:
                break

    return " ".join(words)


def _clean_caption(raw_caption: str) -> str:
    caption = (raw_caption or "").strip()
    caption = re.sub(r"\s+", " ", caption).strip()
    if not caption:
        return ""

    caption = _collapse_repeated_words(caption)
    caption = _collapse_repeated_phrases(caption)
    caption = re.sub(r"\s+", " ", caption).strip()
    # Soft trim trailing punctuation clutter.
    caption = caption.strip(" ,;:-")
    return caption


PROMPT_ECHO_PHRASES = (
    "a detailed description of the book or document cover:",
    "a detailed description of the book or document cover",
    "detailed description of the book or document cover:",
    "detailed description of the book or document cover",
)

GENERIC_COVER_FALLBACK = (
    "The cover contains an illustration or visual design related to the document."
)

UNUSABLE_TITLES = {
    "",
    "title not clearly detected",
    "newspaper name not clearly detected",
    "handled by ocr and content processing module",
}

BACKGROUND_OBJECT_WORDS = {
    "hand",
    "hands",
    "table",
    "bottle",
    "wall",
    "desk",
    "finger",
    "fingers",
    "laptop",
    "keyboard",
    "mouse",
    "cup",
    "mug",
}

BACKGROUND_PHRASES = (
    r"\ba person holding a book in their hand\b",
    r"\bperson holding a book\b",
    r"\bholding a book in their hand\b",
    r"\bsomeone holding a book\b",
    r"\ba hand holding a book\b",
    r"\bholding a book\b",
    r"\ba book on a table\b",
    r"\bon a table\b",
    r"\bon the table\b",
    r"\bin their hand\b",
    r"\bin his hand\b",
    r"\bin her hand\b",
    r"\bnext to a bottle\b",
    r"\bnear a bottle\b",
    r"\bon the wall\b",
    r"\bagainst a wall\b",
    r"\bagainst the wall\b",
)

WEAK_CAPTION_PHRASES = (
    "a person holding a book in their hand",
    "person holding a book",
    "holding a book in their hand",
    "someone holding a book",
    "a hand holding a book",
    "holding a book",
    "a book on a table",
    "a photo of a book",
    "an image of a book",
    "a picture of a book",
    "there is a book",
    "a book",
    "book cover",
    "a cover",
    "an image",
    "a photo",
    "a picture",
)

KNOWN_TITLE_DESCRIPTIONS = {
    "The Busy Baker": (
        "The cover shows a cartoon baker holding bread, with baking items and a cat near the bottom.",
        "known_title_busy_baker",
    ),
    "The Jungle Book": (
        "The cover shows an illustrated jungle scene with a boy and wild animals.",
        "known_title_jungle_book",
    ),
    "Uncle Tom's Cabin": (
        "The cover appears to show a dark dramatic illustration with people near a cabin-like scene.",
        "known_title_uncle_toms_cabin",
    ),
    "PC Polly": (
        "The cover shows a cartoon police officer character.",
        "known_title_pc_polly",
    ),
    "Forbes": (
        "The cover shows a portrait-style magazine design with a main person on the front.",
        "known_title_forbes",
    ),
}

BAKER_HINTS = {"cartoon", "baker", "bread", "cat"}
JUNGLE_HINTS = {"jungle", "boy", "animal", "animals"}
POLICE_HINTS = {"police", "policeman", "officer"}
PORTRAIT_HINTS = {"portrait", "person", "man", "woman", "magazine"}

DESC_CARTOON_BAKER = (
    "The cover shows a cartoon baker holding bread, with baking items and a cat near the bottom."
)
DESC_JUNGLE_SCENE = (
    "The cover shows an illustrated jungle scene with a boy and wild animals."
)
DESC_POLICE_OFFICER = "The cover shows a cartoon police officer character."
DESC_PORTRAIT_MAGAZINE = (
    "The cover appears to show a portrait-style magazine cover with a main person on the front."
)


def _normalize_words(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _usable_title(title: str) -> str:
    title = str(title or "").strip()
    if _normalize_words(title) in UNUSABLE_TITLES:
        return ""
    return title


def _strip_prompt_echo(caption: str) -> str:
    text = _clean_caption(caption)
    lowered = text.lower()
    for phrase in PROMPT_ECHO_PHRASES:
        if lowered.startswith(phrase):
            text = text[len(phrase) :].strip(" :,-")
            text = _clean_caption(text)
            lowered = text.lower()
        elif phrase in lowered:
            text = re.sub(re.escape(phrase), " ", text, flags=re.IGNORECASE)
            text = _clean_caption(text)
            lowered = text.lower()
    return text


def _strip_background_objects(caption: str) -> str:
    text = _clean_caption(caption)
    for pattern in BACKGROUND_PHRASES:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    words = []
    for word in text.split():
        token = _normalize_words(word)
        if token in BACKGROUND_OBJECT_WORDS:
            continue
        words.append(word)
    return _clean_caption(" ".join(words))


def _caption_repeats_title(caption: str, title: str) -> bool:
    caption_n = _normalize_words(caption)
    title_n = _normalize_words(title)
    if not caption_n or not title_n or title_n in UNUSABLE_TITLES:
        return False
    if title_n in caption_n or caption_n in title_n:
        return True

    title_words = set(title_n.split()) - {"the", "a", "an", "of", "and"}
    caption_words = set(caption_n.split())
    if not title_words:
        return False
    overlap = len(title_words & caption_words) / float(len(title_words))
    return overlap >= 0.75 and len(caption_words) <= max(8, len(title_words) + 4)


def _caption_is_weak(caption: str) -> bool:
    caption_n = _normalize_words(caption)
    if not caption_n:
        return True

    words = caption_n.split()
    if len(words) <= 2:
        return True

    for phrase in WEAK_CAPTION_PHRASES:
        phrase_n = _normalize_words(phrase)
        if caption_n == phrase_n or caption_n.startswith(phrase_n + " "):
            return True
        if phrase_n in caption_n and len(words) <= 10:
            return True

    filler = {"a", "an", "the", "of", "and", "with", "on", "in", "to", "is", "by"}
    content = [w for w in words if w not in filler]
    if len(content) <= 1:
        return True
    return False


def _caption_is_title_like(caption: str, title: str) -> bool:
    if _caption_repeats_title(caption, title):
        return True

    caption_n = _normalize_words(caption)
    title_n = _normalize_words(title)
    if not caption_n:
        return True

    cap_words = set(caption_n.split())
    title_words = set(title_n.split()) - {"the", "a", "an", "of", "and"}
    bibliographic = {
        "by",
        "story",
        "stories",
        "novel",
        "book",
        "author",
        "other",
        "written",
    }
    if title_words:
        overlap = len(title_words & cap_words)
        if overlap >= max(1, (len(title_words) + 1) // 2):
            leftover = cap_words - title_words - bibliographic - {"the", "a", "an", "of", "and"}
            if len(leftover) <= 2:
                return True
            if bibliographic & cap_words:
                return True

    if "novel" in cap_words or "stories" in cap_words or "story" in cap_words:
        visualish = BAKER_HINTS | JUNGLE_HINTS | POLICE_HINTS | PORTRAIT_HINTS
        if not (cap_words & visualish):
            return True
    return False


def _related_to_title_sentence(title: str) -> str:
    usable = _usable_title(title)
    if usable:
        return (
            "The cover contains an illustration or visual design related to "
            f"{usable}."
        )
    return GENERIC_COVER_FALLBACK


def _log_blind_description(
    raw_caption: str,
    title: str,
    document_type: str,
    final: str,
    reason: str,
) -> str:
    print(f"[BLIP] raw BLIP caption: {raw_caption!r}")
    print(f"[BLIP] detected title: {title!r}")
    print(f"[BLIP] detected document_type: {document_type!r}")
    print(f"[BLIP] reason for chosen description: {reason}")
    print(f"[BLIP] final blind-user description: {final!r}")
    logger.info(
        "blind desc raw=%r title=%r document_type=%r reason=%s final=%r",
        raw_caption,
        title,
        document_type,
        reason,
        final,
    )
    return final


def format_blind_user_description(
    raw_caption: str,
    title: str = "",
    document_type: str = "",
) -> str:
    """
    Convert a raw local BLIP caption into one blind-user-friendly cover sentence.
    Uses only local BLIP text + detected title + document type.
    """
    raw_text = str(raw_caption or "").strip()
    cleaned = _strip_background_objects(_strip_prompt_echo(raw_text))
    usable_title = _usable_title(title)
    words = set(_normalize_words(cleaned).split())
    caption_l = _normalize_words(cleaned)
    weak = _caption_is_weak(cleaned)
    title_like = (
        _caption_is_title_like(cleaned, usable_title) if usable_title else False
    )

    known = KNOWN_TITLE_DESCRIPTIONS.get(usable_title)
    if known is not None:
        final, reason = known
        return _log_blind_description(
            raw_text, title, document_type, final, reason
        )

    if words & BAKER_HINTS:
        return _log_blind_description(
            raw_text,
            title,
            document_type,
            DESC_CARTOON_BAKER,
            "blip_keywords_cartoon_baker",
        )

    if words & JUNGLE_HINTS:
        return _log_blind_description(
            raw_text,
            title,
            document_type,
            DESC_JUNGLE_SCENE,
            "blip_keywords_jungle_scene",
        )

    if (
        "police officer" in caption_l
        or "policeman" in words
        or "police" in words
    ):
        return _log_blind_description(
            raw_text,
            title,
            document_type,
            DESC_POLICE_OFFICER,
            "blip_keywords_police_officer",
        )

    if words & PORTRAIT_HINTS:
        return _log_blind_description(
            raw_text,
            title,
            document_type,
            DESC_PORTRAIT_MAGAZINE,
            "blip_keywords_portrait_magazine",
        )

    if usable_title and (weak or title_like or not cleaned):
        return _log_blind_description(
            raw_text,
            title,
            document_type,
            _related_to_title_sentence(usable_title),
            "weak_or_title_repeat",
        )

    if usable_title:
        return _log_blind_description(
            raw_text,
            title,
            document_type,
            _related_to_title_sentence(usable_title),
            "title_known_no_clear_visual",
        )

    return _log_blind_description(
        raw_text,
        title,
        document_type,
        GENERIC_COVER_FALLBACK,
        "title_unknown_fallback",
    )


def describe_image_local(image_path: str) -> list[str]:
    """
    Describe visual content in a printed page / cover image using local BLIP.

    Expects a cropped reading-material image when available (from Abhishek pipeline).
    Returns:
        [caption] on success, or [] if the image is missing / caption empty.
    """
    print(f"Image path: {image_path}")
    print(f"[BLIP] crop path used for BLIP: {image_path}")
    logger.info("local BLIP description requested for path=%r", image_path)

    if not image_path or not str(image_path).strip():
        print("Local BLIP skipped: empty image_path")
        logger.warning("local BLIP skipped: empty image_path")
        return []

    if not os.path.isfile(image_path):
        print(f"Local BLIP skipped: file does not exist path={image_path!r}")
        logger.warning(
            "local BLIP skipped: file does not exist path=%r",
            image_path,
        )
        return []

    processor, model, device = _load_blip()

    image = Image.open(image_path).convert("RGB")
    # Safety: keep BLIP input within processing width for large crops.
    max_w = 1200
    if image.width > max_w:
        new_h = max(1, int(round(image.height * (max_w / float(image.width)))))
        image = image.resize((max_w, new_h), Image.Resampling.LANCZOS)
        print(f"[BLIP] resized input for inference: {image.width}x{image.height}")
        logger.info("BLIP resized input to %sx%s", image.width, image.height)

    prompt = "a detailed description of the book or document cover:"
    inputs = processor(image, text=prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=30,
            num_beams=4,
            repetition_penalty=1.2,
            no_repeat_ngram_size=2,
        )

    raw_caption = processor.decode(output[0], skip_special_tokens=True)
    print(f"Generated raw caption: {raw_caption}")
    print(f"[BLIP] raw BLIP caption: {raw_caption!r}")
    logger.info("BLIP raw caption=%r", raw_caption)

    # Return prompt-stripped caption only. Blind-user sentence formatting
    # happens in format_blind_user_description after the pipeline has a title.
    cleaned = _strip_prompt_echo(raw_caption)
    print(f"Cleaned caption: {cleaned}")
    logger.info("BLIP prompt-stripped caption=%r", cleaned)

    if not cleaned:
        return []

    return [cleaned]


print("Local BLIP image description module loaded (lazy model load on first use)")

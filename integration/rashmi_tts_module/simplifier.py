"""
Apply Rashmi reading-level simplification before TTS.

Uses tts-preferences/trained_multi_level_simplifier_v3 when available.
Advanced keeps the original wording.
"""

import os
import re
from pathlib import Path
from typing import Optional


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent
MODEL_DIR = Path(
    os.getenv(
        "RASHMI_SIMPLIFIER_PATH",
        str(PROJECT_ROOT / "tts-preferences" / "trained_multi_level_simplifier_v3"),
    )
)

_MODEL = None
_TOKENIZER = None
_DEVICE = "cpu"
_LOAD_ATTEMPTED = False

LEVEL_ALIASES = {
    "simple": "simple",
    "easy": "simple",
    "very simple": "simple",
    "moderate": "moderate",
    "medium": "moderate",
    "normal level": "moderate",
    "advanced": "advanced",
    "advance": "advanced",
    "hard": "advanced",
    "light": "advanced",
}

SIMPLE_SWAPS = (
    (r"\bapproximately\b", "about"),
    (r"\btherefore\b", "so"),
    (r"\bhowever\b", "but"),
    (r"\bnevertheless\b", "but"),
    (r"\bsubsequently\b", "then"),
    (r"\bsignificant(?:ly)?\b", "important"),
    (r"\bdemonstrate(?:s|d)?\b", "show"),
    (r"\bindicate(?:s|d)?\b", "show"),
    (r"\badditional\b", "more"),
    (r"\bcommence(?:s|d)?\b", "start"),
    (r"\bterminate(?:s|d)?\b", "end"),
    (r"\butilize(?:s|d)?\b", "use"),
    (r"\bobtain(?:s|ed)?\b", "get"),
    (r"\brequire(?:s|d)?\b", "need"),
    (r"\bregarding\b", "about"),
    (r"\bin order to\b", "to"),
)


def normalize_reading_level(reading_level: Optional[str]) -> str:
    key = str(reading_level or "").strip().lower()
    return LEVEL_ALIASES.get(key, "moderate" if key else "simple")


def _load_model():
    global _MODEL, _TOKENIZER, _DEVICE, _LOAD_ATTEMPTED

    if _LOAD_ATTEMPTED:
        return _MODEL is not None

    _LOAD_ATTEMPTED = True

    if not (MODEL_DIR / "config.json").exists():
        print("Rashmi simplifier model not found:", MODEL_DIR)
        return False

    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        _TOKENIZER = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
        _MODEL = AutoModelForSeq2SeqLM.from_pretrained(str(MODEL_DIR), local_files_only=True)
        _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        _MODEL.to(_DEVICE)
        _MODEL.eval()
        print(f"Rashmi simplifier loaded from {MODEL_DIR} on {_DEVICE}")
        return True
    except Exception as error:
        print("Rashmi simplifier model could not be loaded:", error)
        _MODEL = None
        _TOKENIZER = None
        return False


def _split_chunks(text: str, max_chars: int = 420):
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks = []
    current = ""

    for part in parts:
        piece = part.strip()
        if not piece:
            continue
        if current and len(current) + 1 + len(piece) > max_chars:
            chunks.append(current.strip())
            current = piece
        else:
            current = f"{current} {piece}".strip()

    if current:
        chunks.append(current.strip())

    return chunks or [text.strip()]


def _generate_chunk(chunk: str, level: str) -> str:
    import torch

    prompt = f"simplify to {level}: {chunk}"
    encoded = _TOKENIZER(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    encoded = {key: value.to(_DEVICE) for key, value in encoded.items()}
    max_new = min(256, max(64, len(chunk.split()) * 2))

    with torch.no_grad():
        output_ids = _MODEL.generate(
            **encoded,
            max_new_tokens=max_new,
            num_beams=4,
            do_sample=False,
        )

    result = _TOKENIZER.decode(output_ids[0], skip_special_tokens=True).strip()
    result = re.sub(r"^(simplify to\s+\w+:\s*)", "", result, flags=re.IGNORECASE)
    return result


def _lexical_simplify(text: str, level: str) -> str:
    if level != "simple":
        return text

    updated = text
    for pattern, replacement in SIMPLE_SWAPS:
        updated = re.sub(pattern, replacement, updated, flags=re.IGNORECASE)

    sentences = re.split(r"(?<=[.!?])\s+", updated)
    short_sentences = []
    for sentence in sentences:
        words = sentence.split()
        if len(words) > 22:
            cut = " ".join(words[:18]).rstrip(",;")
            if not cut.endswith("."):
                cut += "."
            short_sentences.append(cut)
        elif sentence.strip():
            short_sentences.append(sentence.strip())
    return " ".join(short_sentences).strip() or text


def simplify_text(text: str, reading_level: Optional[str]) -> str:
    original = str(text or "").strip()
    if not original:
        return original

    level = normalize_reading_level(reading_level)
    if level == "advanced" or len(original.split()) < 8:
        return original

    if _load_model():
        simplified_parts = []
        for chunk in _split_chunks(original):
            try:
                piece = _generate_chunk(chunk, level)
            except Exception as error:
                print("Rashmi simplifier generate failed:", error)
                piece = chunk
            if not piece or len(piece.split()) < max(3, len(chunk.split()) // 6):
                piece = chunk
            simplified_parts.append(piece)
        simplified = " ".join(simplified_parts).strip()
        if simplified:
            print(f"Simplification source: local T5 ({level})")
            return simplified

    fallback = _lexical_simplify(original, level)
    if fallback != original:
        print(f"Simplification source: lexical fallback ({level})")
    else:
        print(f"Simplification skipped; using original text ({level})")
    return fallback

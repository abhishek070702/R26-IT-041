"""
Apply Rashmi reading-level simplification before TTS.

Live method matches Final 1.1.ipynb:
- light: original text
- very simple / moderate: CSV exact match, then OpenAI, then local word fallback
The T5 folder is not used; it copies input and does not apply these levels.
"""

import csv
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent
CSV_PATH = Path(
    os.getenv(
        "RASHMI_SIMPLIFIER_CSV",
        str(PROJECT_ROOT / "tts-preferences" / "text simplification - Copy.csv"),
    )
)

LEVEL_ALIASES = {
    "very simple": "very simple",
    "simple": "very simple",
    "easy": "very simple",
    "very easy": "very simple",
    "moderate": "moderate",
    "medium": "moderate",
    "normal level": "moderate",
    "light": "light",
    "original": "light",
    "close to original": "light",
    "advanced": "light",
    "advance": "light",
    "hard": "light",
}

CSV_LEVEL_NAMES = {
    "very simple": "Very Simple",
    "moderate": "Moderate",
    "light": "Light",
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
    (r"\binvestigation\b", "check"),
    (r"\bannounced\b", "said"),
    (r"\bresidents\b", "people"),
    (r"\bexpressed concern\b", "worried"),
    (r"\bintroduced a plan\b", "made a plan"),
    (r"\bpublic transportation\b", "buses and trains"),
    (r"\bcontinue to rise\b", "keep going up"),
    (r"\bsafety measures\b", "safety rules"),
    (r"\bexperts warned\b", "Experts said"),
)

MODERATE_SWAPS = (
    (r"\bapproximately\b", "about"),
    (r"\btherefore\b", "so"),
    (r"\bhowever\b", "but"),
    (r"\bnevertheless\b", "still"),
    (r"\bsubsequently\b", "later"),
    (r"\butilize(?:s|d)?\b", "use"),
    (r"\bcommence(?:s|d)?\b", "begin"),
    (r"\bregarding\b", "about"),
    (r"\bin order to\b", "to"),
    (r"\bannounced\b", "told people"),
    (r"\bexpressed concern\b", "were worried"),
    (r"\bintroduced a plan\b", "made a plan"),
    (r"\bpublic transportation\b", "public transport"),
    (r"\bcontinue to rise\b", "keep rising"),
    (r"\bsignificant(?:ly)?\b", "important"),
)

_ENV_LOADED = False
_OPENAI_CLIENT = None
_OPENAI_LOAD_ATTEMPTED = False
_OPENAI_UNAVAILABLE = False


def _load_env():
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _ENV_LOADED = True
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    for path in (
        PROJECT_ROOT / "backend" / ".env",
        PROJECT_ROOT / "article_detection" / ".env",
        PROJECT_ROOT / ".env",
    ):
        if path.is_file():
            load_dotenv(path, override=False)


def normalize_reading_level(reading_level: Optional[str]) -> str:
    key = str(reading_level or "").strip().lower()
    return LEVEL_ALIASES.get(key, "moderate" if key else "very simple")


def _compact_lookup_key(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


@lru_cache(maxsize=1)
def _load_csv_tables() -> Tuple[Dict[Tuple[str, str], str], Dict[str, Dict[str, str]]]:
    exact: Dict[Tuple[str, str], str] = {}
    word_maps: Dict[str, Dict[str, str]] = {
        "very simple": {},
        "moderate": {},
        "light": {},
    }
    if not CSV_PATH.exists():
        print("Rashmi simplifier CSV not found:", CSV_PATH)
        return exact, word_maps

    with CSV_PATH.open("r", encoding="utf-8", errors="replace", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            source = str(row.get("Input") or "").strip()
            csv_level = str(row.get("Reading_level") or "").strip()
            output = str(row.get("Output") or "").strip()
            if not source or not output:
                continue

            level_key = {
                "very simple": "very simple",
                "moderate": "moderate",
                "light": "light",
            }.get(csv_level.lower())
            if level_key is None:
                continue

            exact[( _compact_lookup_key(source), csv_level)] = output

            if len(source.split()) <= 3 and _compact_lookup_key(source) != _compact_lookup_key(output):
                word_maps[level_key][_compact_lookup_key(source)] = output

    print("Rashmi simplifier CSV loaded:", CSV_PATH.name)
    return exact, word_maps


def lookup_simplification(text: str, reading_level: str) -> Optional[str]:
    exact, _word_maps = _load_csv_tables()
    csv_level = CSV_LEVEL_NAMES.get(reading_level)
    if not csv_level:
        return None
    return exact.get((_compact_lookup_key(text), csv_level))


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


def _openai_client():
    global _OPENAI_CLIENT, _OPENAI_LOAD_ATTEMPTED
    if _OPENAI_LOAD_ATTEMPTED:
        return _OPENAI_CLIENT

    _OPENAI_LOAD_ATTEMPTED = True
    _load_env()
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        print("Rashmi simplifier OpenAI key not found; using local fallback.")
        return None

    try:
        from openai import OpenAI
        _OPENAI_CLIENT = OpenAI(api_key=api_key)
        return _OPENAI_CLIENT
    except Exception as error:
        print("Rashmi simplifier OpenAI client failed:", error)
        _OPENAI_CLIENT = None
        return None


def _openai_simplify_chunk(text: str, reading_level: str) -> Optional[str]:
    global _OPENAI_UNAVAILABLE
    if _OPENAI_UNAVAILABLE:
        return None

    client = _openai_client()
    if client is None:
        _OPENAI_UNAVAILABLE = True
        return None

    level_label = CSV_LEVEL_NAMES[reading_level]
    model = os.getenv("OPENAI_TEXT_MODEL", "gpt-4o-mini")
    prompt = f"""
You simplify text for a visually impaired user's selected reading level.

Rules:
- Very simple: use very easy everyday words and short sentences.
- Moderate: simplify difficult words and make the text clearly easier to understand.
- Light: keep the text close to the original, changing only difficult wording when needed.
- Preserve important meaning.
- Return only the simplified text.

Reading level: {level_label}
Text: {text}
""".strip()

    try:
        response = client.responses.create(model=model, input=prompt)
        result = str(getattr(response, "output_text", "") or "").strip()
        if result:
            return result
    except Exception as error:
        print("Rashmi simplifier OpenAI responses API failed; trying chat.")
        if "401" in str(error) or "invalid_api_key" in str(error).lower():
            print("Rashmi simplifier OpenAI key was rejected; using local fallback.")
            _OPENAI_UNAVAILABLE = True
            return None

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        result = str(completion.choices[0].message.content or "").strip()
        return result or None
    except Exception as error:
        print("Rashmi simplifier OpenAI failed:", error)
        _OPENAI_UNAVAILABLE = True
        return None


def _apply_phrase_map(text: str, mapping: Dict[str, str]) -> str:
    updated = text
    for source in sorted(mapping, key=len, reverse=True):
        if len(source) < 3:
            continue
        pattern = r"\b" + re.escape(source) + r"\b"
        updated = re.sub(pattern, mapping[source], updated, flags=re.IGNORECASE)
    return updated


def _shorten_sentences(text: str, max_words: int) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    short_sentences = []
    for sentence in sentences:
        words = sentence.split()
        if len(words) > max_words:
            cut = " ".join(words[: max_words - 2]).rstrip(",;")
            if cut and not cut.endswith("."):
                cut += "."
            short_sentences.append(cut)
        elif sentence.strip():
            short_sentences.append(sentence.strip())
    return " ".join(short_sentences).strip() or text


def _lexical_simplify(text: str, level: str) -> str:
    _exact, word_maps = _load_csv_tables()
    updated = _apply_phrase_map(text, word_maps.get(level) or {})

    if level == "very simple":
        for pattern, replacement in SIMPLE_SWAPS:
            updated = re.sub(pattern, replacement, updated, flags=re.IGNORECASE)
        updated = _shorten_sentences(updated, 16)
    elif level == "moderate":
        for pattern, replacement in MODERATE_SWAPS:
            updated = re.sub(pattern, replacement, updated, flags=re.IGNORECASE)

    return updated.strip() or text


def simplify_text(text: str, reading_level: Optional[str]) -> str:
    original = str(text or "").strip()
    if not original:
        return original

    level = normalize_reading_level(reading_level)
    if level == "light":
        print("Reading level: light — original text will be used without simplification.")
        return original

    exact = lookup_simplification(original, level)
    if exact:
        print(f"Simplification source: dataset ({level})")
        return exact

    openai_parts = []
    openai_hits = 0
    for chunk in _split_chunks(original):
        piece = _openai_simplify_chunk(chunk, level)
        if piece:
            openai_hits += 1
            openai_parts.append(piece)
        else:
            openai_parts.append(_lexical_simplify(chunk, level))

    simplified = " ".join(part for part in openai_parts if part).strip()
    if simplified and openai_hits:
        source = "OpenAI" if openai_hits == len(openai_parts) else "OpenAI + local fallback"
        print(f"Simplification source: {source} ({level})")
        return simplified

    fallback = _lexical_simplify(original, level)
    if fallback != original:
        print(f"Simplification source: local fallback ({level})")
        return fallback

    print(f"Simplification skipped; using original text ({level})")
    return original

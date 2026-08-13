"""
Optional local Florence-2 image description for the Abhishek PP2 pipeline.

Used when IMAGE_DESCRIPTION_MODEL=florence.
Default remains BLIP via local_blip_description.py (do not remove that module).

Model: microsoft/Florence-2-base-ft
No OpenAI / external APIs.
"""

from __future__ import annotations

import logging
import os
import re

import torch
from PIL import Image

logger = logging.getLogger(__name__)

MODEL_NAME = "microsoft/Florence-2-base-ft"
TASK_PROMPT = "<MORE_DETAILED_CAPTION>"

_processor = None
_model = None
_device = None
_load_error: str | None = None

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
    "background",
    "room",
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
    r"\bin the background\b",
    r"\bin the room\b",
)


def _get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _transformers_version_tuple() -> tuple[int, ...]:
    try:
        import transformers

        parts = []
        for chunk in str(transformers.__version__).split("."):
            digits = "".join(ch for ch in chunk if ch.isdigit())
            if digits:
                parts.append(int(digits))
            else:
                break
        return tuple(parts) if parts else (0,)
    except Exception:
        return (0,)


def _apply_florence_compat_patches() -> None:
    """
    Newer transformers versions may raise:
      'Florence2LanguageConfig' object has no attribute 'forced_bos_token_id'

    Patch PretrainedConfig attribute access so missing forced_bos_token_id is
    filled safely from bos_token_id (or 0).
    """
    from transformers.configuration_utils import PretrainedConfig

    if getattr(PretrainedConfig, "_florence_forced_bos_patch", False):
        return

    original_getattribute = PretrainedConfig.__getattribute__

    def patched_getattribute(self, key):  # type: ignore[no-untyped-def]
        try:
            return original_getattribute(self, key)
        except AttributeError:
            if key != "forced_bos_token_id":
                raise
            try:
                bos = original_getattribute(self, "bos_token_id")
            except Exception:
                bos = 0
            value = bos if bos is not None else 0
            object.__setattr__(self, "forced_bos_token_id", value)
            return value

    PretrainedConfig.__getattribute__ = patched_getattribute  # type: ignore[method-assign]
    PretrainedConfig._florence_forced_bos_patch = True
    print("[Florence] applied forced_bos_token_id compatibility patch")
    logger.info("applied Florence forced_bos_token_id compatibility patch")


def _ensure_forced_bos_token_id(model) -> None:
    """Ensure Florence language/config objects expose forced_bos_token_id."""
    configs = []
    cfg = getattr(model, "config", None)
    if cfg is not None:
        configs.append(cfg)
        text_cfg = getattr(cfg, "text_config", None)
        if text_cfg is not None:
            configs.append(text_cfg)
        lang_cfg = getattr(cfg, "language_config", None)
        if lang_cfg is not None:
            configs.append(lang_cfg)

    for conf in configs:
        if conf is None:
            continue
        if getattr(conf, "forced_bos_token_id", None) is None:
            bos = getattr(conf, "bos_token_id", None)
            conf.forced_bos_token_id = bos if bos is not None else 0


def _load_florence():
    """Lazy-load Florence-2 processor/model once (official HF style + compat patch)."""
    global _processor, _model, _device, _load_error

    if _model is not None and _processor is not None:
        return _processor, _model, _device

    # Allow retry after a previous failure (e.g. missing deps installed later).
    _load_error = None

    try:
        import transformers
        from transformers import AutoModelForCausalLM, AutoProcessor

        tf_version = getattr(transformers, "__version__", "unknown")
        print(f"[Florence] transformers version={tf_version}")
        logger.info("Florence transformers version=%s", tf_version)

        version_tuple = _transformers_version_tuple()
        if version_tuple and version_tuple[0] >= 5:
            print(
                "[Florence] warning: transformers "
                f"{tf_version} is often incompatible with Florence-2 "
                "(forced_bos_token_id). Prefer transformers>=4.41,<4.50 "
                "(recommended: 4.49.0)."
            )
            logger.warning(
                "transformers %s may be incompatible with Florence-2",
                tf_version,
            )

        _apply_florence_compat_patches()

        _device = _get_device()
        print(f"[Florence] Loading local Florence model: {MODEL_NAME}")
        print(f"[Florence] Using device: {_device}")

        torch_dtype = torch.float16 if _device == "cuda" else torch.float32

        _processor = AutoProcessor.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
        )
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )
        _model = _model.to(_device)
        _ensure_forced_bos_token_id(_model)
        _model.eval()

        print("[Florence] loaded successfully")
        logger.info("Florence loaded successfully on device=%s", _device)
        return _processor, _model, _device

    except Exception as exc:
        _processor = None
        _model = None
        _device = None
        message = str(exc)
        if "forced_bos_token_id" in message:
            message = (
                f"{message}. Installed transformers appears incompatible with "
                "Florence-2. Install a compatible version, e.g. "
                "`pip install 'transformers>=4.41.0,<4.50.0'` "
                "(recommended: transformers==4.49.0)."
            )
        _load_error = f"Failed to load local Florence model: {message}"
        print(f"[Florence] loaded failed: {_load_error}")
        logger.exception("Florence load failed")
        raise RuntimeError(_load_error) from exc


def _clean_text(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"\s+", " ", text).strip(" ,;:-")
    return text


def _strip_background(text: str) -> str:
    cleaned = _clean_text(text)
    for pattern in BACKGROUND_PHRASES:
        cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)
    words = []
    for word in cleaned.split():
        token = re.sub(r"[^a-z0-9]", "", word.lower())
        if token in BACKGROUND_OBJECT_WORDS:
            continue
        words.append(word)
    cleaned = _clean_text(" ".join(words))
    # Drop trailing dangling fillers left after removals: "with a", "and the", etc.
    cleaned = re.sub(
        r"\b(with|and|near|beside|on|in|at|by|next to)\s+(a|an|the)?\s*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s+,", ",", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return _clean_text(cleaned)


FLORENCE_SAFE_VISUAL_FALLBACK = (
    "The cover appears to show an illustrated scene related to the story."
)

NOVEL_FALLBACK = FLORENCE_SAFE_VISUAL_FALLBACK
MAGAZINE_FALLBACK = (
    "The magazine cover appears to show a visual design or main cover image."
)
NEWSPAPER_NO_IMAGE = "No clear main image is detected on this newspaper page."
REPORT_NO_IMAGE = "No important visual image is detected in this report."
LETTER_NO_IMAGE = "No important visual image is detected in this letter."

KNOWN_FLORENCE_TITLES = {
    "peter pan": "Peter Pan",
    "the jungle book": "The Jungle Book",
    "jungle book": "The Jungle Book",
    "the busy baker": "The Busy Baker",
    "busy baker": "The Busy Baker",
    "uncle tom's cabin": "Uncle Tom's Cabin",
    "uncle toms cabin": "Uncle Tom's Cabin",
    "pc polly": "PC Polly",
    "forbes": "Forbes",
    "shape": "Shape",
    "treasure island": "Treasure Island",
    "ulysses": "Ulysses",
    "the secret garden": "The Secret Garden",
    "secret garden": "The Secret Garden",
    "the kite runner": "The Kite Runner",
    "kite runner": "The Kite Runner",
    "dracula": "Dracula",
    "mr china": "Mr. China",
    "mr. china": "Mr. China",
}

KNOWN_NEWSPAPER_MASTHEADS = {
    "the times": "The Times",
    "times": "The Times",
    "the guardian": "The Guardian",
    "guardian": "The Guardian",
    "daily news": "Daily News",
    "the daily news": "Daily News",
    "washington post": "Washington Post",
    "the washington post": "Washington Post",
    "new york times": "New York Times",
    "the new york times": "New York Times",
    "forbes": "Forbes",
}

BAD_TITLE_TOKENS = {
    "cover",
    "image",
    "picture",
    "photo",
    "shows",
    "show",
    "appears",
    "appear",
    "illustration",
    "document",
    "background",
    "table",
    "hand",
    "bottle",
    "wall",
    "report",
    "letter",
    "newspaper",
    "magazine",
    "page",
    "title",
    "text",
    "written",
    "letters",
    "lettering",
    "words",
    "front",
}

# Full descriptive phrases that must never become titles.
DESCRIPTIVE_TITLE_PHRASES = (
    "written in red letters",
    "written in black letters",
    "written in gold letters",
    "written in blue letters",
    "written in white letters",
    "written in large letters",
    "red letters",
    "black letters",
    "gold letters",
    "blue letters",
    "white letters",
    "title written",
    "cover has",
    "cover of",
    "book cover",
    "front page",
    "picture of",
    "image of",
    "newspaper page",
    "magazine cover",
    "the words",
    "the title of the book is written",
    "title of the book is written",
)

DESCRIPTIVE_TITLE_PATTERNS = (
    r"^written in\b",
    r"\bwritten in\b.+\bletters?\b",
    r"\b(?:red|black|gold|blue|white)\s+letters?\b",
    r"\btitle written\b",
    r"\bcover has\b",
    r"^cover of\b",
    r"\bbook cover\b",
    r"\bfront page\b",
    r"\bpicture of\b",
    r"\bimage of\b",
    r"\bnewspaper page\b",
    r"\bmagazine cover\b",
    r"^the words\b",
    r"\bthe title of the book is written\b",
    r"\btitle of the book is written\b",
)

NOVEL_SUBJECT_WORDS = {
    "boy",
    "girl",
    "man",
    "woman",
    "person",
    "people",
    "child",
    "children",
    "character",
    "pirate",
    "ship",
    "sea",
    "ocean",
    "water",
    "boat",
    "jungle",
    "animal",
    "animals",
    "cat",
    "dog",
    "bird",
    "scene",
    "illustration",
    "illustrated",
    "silhouette",
    "portrait",
    "landscape",
    "forest",
    "tree",
    "island",
    "adventure",
    "map",
    "sword",
    "hat",
    "face",
    "figure",
    "figures",
    "house",
}

MAGAZINE_SUBJECT_WORDS = {
    "person",
    "people",
    "portrait",
    "model",
    "product",
    "fashion",
    "food",
    "building",
    "vehicle",
    "car",
    "design",
    "advertisement",
    "ad",
    "cover",
    "photo",
    "photograph",
    "woman",
    "man",
    "girl",
    "boy",
    "celebrity",
    "outfit",
    "dress",
    "jewelry",
    "watch",
    "phone",
    "city",
    "skyline",
}

NEWSPAPER_SUBJECT_WORDS = {
    "people",
    "person",
    "meeting",
    "sports",
    "player",
    "players",
    "vehicle",
    "car",
    "building",
    "event",
    "crowd",
    "chart",
    "scene",
    "photo",
    "photograph",
    "image",
    "picture",
    "man",
    "woman",
    "protest",
    "match",
    "game",
    "stadium",
    "police",
    "soldier",
    "firefighter",
}

REPORT_SUBJECT_WORDS = {
    "chart",
    "diagram",
    "table",
    "graph",
    "logo",
    "photo",
    "photograph",
    "image",
    "picture",
    "figure",
    "map",
    "infographic",
    "illustration",
    "seal",
    "stamp",
    "icon",
}

LETTER_SUBJECT_WORDS = {
    "logo",
    "signature",
    "stamp",
    "seal",
    "photo",
    "photograph",
    "image",
    "picture",
    "letterhead",
    "emblem",
    "crest",
    "icon",
}

BIBLIOGRAPHIC_REJECT_PHRASES = (
    "title of the book",
    "title is",
    "book by",
    "novel by",
    "written by",
    "written in",
    "author",
    "lettering",
    "gold lettering",
    "blue with gold",
    "cover is blue",
    "cover is red",
    "cover is green",
    "cover is yellow",
    "font",
    "text says",
    "the title",
    "large letters",
    "large lettering",
    "typography",
    "title text",
    "headline text",
    "body text",
    "paragraph",
    "columns of text",
)

# Prefer empty-ish visual for text-heavy docs when caption is only about text/layout.
TEXT_ONLY_REJECT_PHRASES = (
    "page of text",
    "filled with text",
    "mostly text",
    "wall of text",
    "blocks of text",
    "printed text",
    "typewritten",
    "handwritten letter",
    "letter written",
    "document with text",
    "full of text",
    "text document",
)


def _normalize_document_type(document_type: str) -> str:
    raw = (document_type or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "novel": "Novel",
        "novels": "Novel",
        "book": "Novel",
        "magazine": "Magazine",
        "magazines": "Magazine",
        "newspaper": "Newspaper",
        "newspapers": "Newspaper",
        "report": "Report",
        "reports": "Report",
        "printed_letter": "Printed_Letter",
        "printedletter": "Printed_Letter",
        "letter": "Printed_Letter",
        "letters": "Printed_Letter",
    }
    return aliases.get(raw, "Novel" if not raw else document_type.strip())


def _doc_config(document_type: str) -> dict:
    dtype = _normalize_document_type(document_type)
    configs = {
        "Novel": {
            "dtype": "Novel",
            "prefix": "The cover appears to show",
            "subjects": NOVEL_SUBJECT_WORDS,
            "fallback": NOVEL_FALLBACK,
            "allow_empty": False,
            "title_mode": "cover",
        },
        "Magazine": {
            "dtype": "Magazine",
            "prefix": "The magazine cover appears to show",
            "subjects": MAGAZINE_SUBJECT_WORDS,
            "fallback": MAGAZINE_FALLBACK,
            "allow_empty": False,
            "title_mode": "cover",
        },
        "Newspaper": {
            "dtype": "Newspaper",
            "prefix": "The newspaper page appears to contain",
            "subjects": NEWSPAPER_SUBJECT_WORDS,
            "fallback": NEWSPAPER_NO_IMAGE,
            "allow_empty": False,
            "title_mode": "masthead",
        },
        "Report": {
            "dtype": "Report",
            "prefix": "The report appears to show",
            "subjects": REPORT_SUBJECT_WORDS,
            "fallback": REPORT_NO_IMAGE,
            "allow_empty": True,
            "title_mode": "strict",
        },
        "Printed_Letter": {
            "dtype": "Printed_Letter",
            "prefix": "The letter appears to show",
            "subjects": LETTER_SUBJECT_WORDS,
            "fallback": LETTER_NO_IMAGE,
            "allow_empty": True,
            "title_mode": "strict",
        },
    }
    return configs.get(dtype, configs["Novel"])


def florence_fallback_for_document_type(document_type: str) -> str:
    return str(_doc_config(document_type)["fallback"])


def _normalize_compare(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _to_one_sentence(text: str) -> str:
    cleaned = _clean_text(text)
    if not cleaned:
        return ""

    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    sentence = _clean_text(parts[0] if parts else cleaned)
    if not sentence:
        return ""
    if not sentence.endswith((".", "!", "?")):
        sentence = sentence + "."
    return sentence


def _strip_author_phrases(text: str) -> str:
    """Remove 'by <Author Name>' style phrases."""
    cleaned = _clean_text(text)
    cleaned = re.sub(
        r"\s*,?\s*\bby\s+[A-Z][A-Za-z.'\-]+(?:\s+[A-Z][A-Za-z.'\-]+){0,4}",
        " ",
        cleaned,
    )
    cleaned = re.sub(
        r"\s*,?\s*\bby\s+[a-z][a-z.'\-]+(?:\s+[a-z][a-z.'\-]+){0,4}",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    return _clean_text(cleaned)


def _split_caption_sentences(text: str) -> list[str]:
    cleaned = _clean_text(text)
    if not cleaned:
        return []
    parts = re.split(r"(?<=[.!?])\s+|\s*;\s*|\s+\-\s+", cleaned)
    sentences = []
    for part in parts:
        item = _clean_text(part)
        if item:
            sentences.append(item)
    if not sentences and cleaned:
        sentences = [cleaned]
    return sentences


def _is_bibliographic_or_lettering_sentence(sentence: str) -> bool:
    lowered = _normalize_compare(sentence)
    if not lowered:
        return True
    if any(phrase in lowered for phrase in BIBLIOGRAPHIC_REJECT_PHRASES):
        return True
    if re.search(r"\bcover is\b.+\b(blue|red|green|gold|yellow|black|white)\b", lowered):
        return True
    if re.search(r"\b(lettering|letters|font|typography)\b", lowered):
        return True
    if re.search(r"\b(title of the book|book by|written by|author)\b", lowered):
        return True
    if re.fullmatch(r"[\"'].+[\"']\s*by\s+.+", sentence.strip(), flags=re.IGNORECASE):
        return True
    return False


def _is_text_only_non_visual(sentence: str, subjects: set[str]) -> bool:
    lowered = _normalize_compare(sentence)
    if not lowered:
        return True
    if any(phrase in lowered for phrase in TEXT_ONLY_REJECT_PHRASES):
        words = set(lowered.split())
        if not (words & subjects):
            return True
    # Reject captions that are only about title/cover wording with no subject.
    if re.search(r"\b(title text|cover text|headline only)\b", lowered):
        return True
    words = set(lowered.split())
    filler = {
        "a",
        "an",
        "the",
        "of",
        "and",
        "with",
        "on",
        "in",
        "to",
        "is",
        "by",
        "this",
        "that",
        "page",
        "document",
        "cover",
        "book",
        "novel",
        "story",
        "text",
        "title",
    }
    content = [w for w in words if w not in filler]
    if len(content) <= 2 and not (set(content) & subjects):
        return True
    return False


def _is_non_visual_sentence(sentence: str, subjects: set[str]) -> bool:
    if _is_bibliographic_or_lettering_sentence(sentence):
        return True
    if _is_text_only_non_visual(sentence, subjects):
        return True
    lowered = _normalize_compare(sentence)
    # Reject empty/self-referential cover fragments.
    if re.fullmatch(r"(the\s+)?cover(\s+image)?", lowered):
        return True
    if lowered in {"title", "title text", "text", "cover", "story", "novel"}:
        return True
    return False


def _visual_sentence_score(sentence: str, subjects: set[str], dtype: str) -> int:
    lowered = _normalize_compare(sentence)
    if not lowered:
        return -100
    if _is_non_visual_sentence(sentence, subjects):
        return -100
    words = set(lowered.split())
    score = 0
    hits = words & subjects
    score += 5 * len(hits)

    if dtype == "Novel":
        if any(token in lowered for token in ("pirate", "ship", "sea", "ocean", "boat", "jungle")):
            score += 8
        if any(token in lowered for token in ("boy", "girl", "man", "woman", "person", "character")):
            score += 6
        if "illustration" in lowered or "illustrated" in lowered or "silhouette" in lowered:
            score += 4
        if "scene" in lowered:
            score += 3
    elif dtype == "Magazine":
        if any(token in lowered for token in ("portrait", "model", "fashion", "product", "advertisement")):
            score += 8
        if any(token in lowered for token in ("person", "woman", "man", "photo", "design")):
            score += 5
    elif dtype == "Newspaper":
        if any(token in lowered for token in ("photo", "photograph", "image", "picture")):
            score += 8
        if any(
            token in lowered
            for token in ("people", "crowd", "meeting", "sports", "player", "vehicle", "building", "event", "chart")
        ):
            score += 6
        # Penalize book/cover/story wording for newspapers.
        if any(token in lowered for token in ("book cover", "novel", "story", "illustrated scene")):
            score -= 20
    elif dtype == "Report":
        if any(token in lowered for token in ("chart", "diagram", "table", "graph", "logo", "infographic")):
            score += 10
        if any(token in lowered for token in ("photo", "photograph", "image", "map", "figure")):
            score += 6
        if any(token in lowered for token in ("cover", "story", "novel")):
            score -= 20
    elif dtype == "Printed_Letter":
        if any(token in lowered for token in ("logo", "signature", "stamp", "seal", "letterhead")):
            score += 10
        if any(token in lowered for token in ("photo", "photograph", "image", "emblem")):
            score += 6
        if any(token in lowered for token in ("cover", "story", "novel")):
            score -= 20

    if '"' in sentence or "“" in sentence or "”" in sentence:
        score -= 4
    if re.search(r"\s[\"']\s*[\"']\s*", sentence):
        score -= 20
    if len(lowered) < 20:
        score -= 3
    return score


def _strip_existing_prefixes(visual: str) -> str:
    patterns = (
        r"^(?:the\s+)?(?:magazine\s+)?cover\s+(?:shows|appears to show)\s+",
        r"^(?:the\s+)?novel\s+cover\s+image\s+(?:shows|appears to show)\s+",
        r"^(?:the\s+)?page\s+illustration\s+(?:shows|appears to show)\s+",
        r"^(?:the\s+)?selected\s+article\s+image\s+(?:shows|appears to show)\s+",
        r"^(?:the\s+)?(?:image|photo)\s+(?:shows|appears to show)\s+",
        r"^(?:the\s+)?newspaper\s+page\s+(?:shows|appears to (?:show|contain)|contains)\s+",
        r"^(?:the\s+)?(?:report|letter)\s+(?:shows|appears to show|appears to contain|contains)\s+",
        r"^(?:the\s+)?(?:page|document)\s+(?:shows|appears to show|appears to contain|contains)\s+",
        r"^(?:showing|depicting|featuring|with)\s+",
    )
    cleaned = visual
    for pattern in patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    return _clean_text(cleaned)


def _normalize_selected_visual_sentence(sentence: str, document_type: str = "") -> str:
    cfg = _doc_config(document_type)
    prefix = str(cfg["prefix"])
    subjects = set(cfg["subjects"])
    dtype = str(cfg["dtype"])

    visual = _clean_text(sentence)
    visual = _strip_author_phrases(visual)
    visual = re.sub(r'[\"“”\']+', " ", visual)
    visual = re.sub(r"\s{2,}", " ", visual).strip(" ,;:-.")
    visual = _clean_text(visual)
    if not visual:
        return ""

    visual = _strip_existing_prefixes(visual)
    if not visual:
        return ""

    # Reject leftover nonsense bodies.
    body_l = _normalize_compare(visual)
    if body_l in {"", "cover", "title", "title text", "text", "story", "novel", "image"}:
        return ""
    if body_l in {"a cover", "the cover", "a title", "the title"}:
        return ""

    # Already a complete typed sentence.
    if visual.lower().startswith(prefix.lower()):
        final = _to_one_sentence(visual)
    else:
        body = visual[0].lower() + visual[1:] if visual else visual
        body = re.sub(
            r"\bappears on the (?:page|cover|letter|document)\b\.?$",
            "",
            body,
            flags=re.IGNORECASE,
        )
        body = _clean_text(body)
        verb_fixes = (
            (r"\blooks\b", "looking"),
            (r"\bstands\b", "standing"),
            (r"\bsits\b", "sitting"),
            (r"\bholds\b", "holding"),
            (r"\bsails\b", "sailing"),
            (r"\bposes\b", "posing"),
            (r"\bappears\b", "appearing"),
            (r"\bcelebrates\b", "celebrating"),
            (r"\bshows\b", "showing"),
        )
        for pattern, replacement in verb_fixes:
            body = re.sub(pattern, replacement, body, flags=re.IGNORECASE)
        if re.match(r"^(?:in|on|above|below|with)\b", body, flags=re.IGNORECASE):
            if dtype in {"Newspaper", "Report", "Printed_Letter"}:
                body = f"a visual {body}"
            else:
                body = f"a visual design {body}"
        final = f"{prefix} {body}."
        if not final.endswith("."):
            final += "."

    if not _is_valid_florence_description(final, document_type=dtype):
        return ""
    if _is_non_visual_sentence(final, subjects):
        return ""
    # Guard against book/story wording on non-novel types.
    lowered = final.lower()
    if dtype != "Novel":
        if "related to the story" in lowered:
            return ""
        if dtype in {"Newspaper", "Report", "Printed_Letter"} and re.search(
            r"\b(book cover|novel|storybook)\b",
            lowered,
        ):
            return ""
    return final


def _is_valid_florence_description(text: str, document_type: str = "") -> bool:
    cleaned = _clean_text(text)
    if not cleaned:
        return False

    # Explicit no-visual messages are valid outputs.
    no_visual = {
        NEWSPAPER_NO_IMAGE.lower(),
        REPORT_NO_IMAGE.lower(),
        LETTER_NO_IMAGE.lower(),
    }
    if cleaned.lower() in no_visual:
        return True

    compact = cleaned.strip(" .\"'")
    if not compact or compact in {".", '"', "'"}:
        return False
    if len(cleaned) < 15:
        return False

    lowered = cleaned.lower()
    if re.search(r"appears to (?:show|contain)\s*\.", lowered):
        return False
    if re.search(r"appears to (?:show|contain)\s*[\"']\s*[\"']", lowered):
        return False
    if re.search(r"appears to (?:show|contain)\s+cover\b", lowered):
        return False
    if re.search(r"appears to (?:show|contain)\s+title text\b", lowered):
        return False
    if re.fullmatch(r"the (?:magazine )?cover appears to show\s*\.?", lowered):
        return False
    if re.fullmatch(r"the cover shows\s*\.?", lowered):
        return False
    if re.fullmatch(r"the newspaper page appears to contain\s*\.?", lowered):
        return False
    if re.fullmatch(r"the (?:report|letter) appears to show\s*\.?", lowered):
        return False

    dtype = _normalize_document_type(document_type) if document_type else ""
    if dtype in {"Newspaper", "Report", "Printed_Letter"}:
        if "related to the story" in lowered:
            return False
        if re.search(r"\b(the cover appears to show|book cover|novel)\b", lowered):
            # Allow only if somehow mixed; treat as invalid for these types.
            if "newspaper" not in lowered and "report" not in lowered and "letter" not in lowered:
                return False
    return True


def format_florence_blind_caption(
    raw_caption: str,
    document_type: str = "",
) -> str:
    """
    Prefer visual sentences from Florence raw caption, using document_type
    to choose wording and subject preferences for all reading materials.
    """
    cfg = _doc_config(document_type)
    dtype = str(cfg["dtype"])
    subjects = set(cfg["subjects"])
    fallback = str(cfg["fallback"])
    allow_empty = bool(cfg["allow_empty"])

    print(f"[Florence] document_type_for_description={dtype}")
    raw = _clean_text(raw_caption)
    if not raw:
        print("[Florence] selected visual sentence=<none>")
        if allow_empty:
            print("[Florence] final document-aware description=<empty>")
            return ""
        print(f"[Florence] final document-aware description={fallback!r}")
        return fallback

    sentences = _split_caption_sentences(raw)
    best_sentence = ""
    best_score = 0

    for sentence in sentences:
        if _is_non_visual_sentence(sentence, subjects):
            print(f"[Florence] rejected non-visual sentence={sentence!r}")
            continue
        score = _visual_sentence_score(sentence, subjects, dtype)
        if score > best_score:
            best_score = score
            best_sentence = sentence

    if best_sentence and best_score > 0:
        print(f"[Florence] selected visual sentence={best_sentence!r}")
        final = _normalize_selected_visual_sentence(best_sentence, dtype)
        if final:
            print(f"[Florence] final document-aware description={final!r}")
            return final
        print(f"[Florence] rejected non-visual sentence={best_sentence!r}")

    if not _is_non_visual_sentence(raw, subjects):
        score = _visual_sentence_score(raw, subjects, dtype)
        if score > 0:
            print(f"[Florence] selected visual sentence={raw!r}")
            final = _normalize_selected_visual_sentence(raw, dtype)
            if final:
                print(f"[Florence] final document-aware description={final!r}")
                return final

    print("[Florence] selected visual sentence=<none>")
    if allow_empty:
        # Prefer informative no-visual message over a silent empty string.
        print(f"[Florence] final document-aware description={fallback!r}")
        return fallback
    print(f"[Florence] final document-aware description={fallback!r}")
    return fallback


def _blind_friendly_sentence(raw_caption: str, document_type: str = "") -> str:
    """Backward-compatible alias for Florence blind-user cleanup."""
    return format_florence_blind_caption(raw_caption, document_type=document_type)


NOVEL_PAGE_NO_IMAGE = "No important visual image is detected on this page."
GENERAL_NO_IMAGE = "No important visual image is detected."
ARTICLE_NO_IMAGE = "No clear main image is detected in this article image."

# Subject pool for general / article crops (photo + illustration cues).
GENERAL_SUBJECT_WORDS = (
    NOVEL_SUBJECT_WORDS
    | MAGAZINE_SUBJECT_WORDS
    | NEWSPAPER_SUBJECT_WORDS
    | REPORT_SUBJECT_WORDS
)


def _normalize_context(context: str) -> str:
    raw = (context or "general").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "general": "general",
        "novel_cover": "novel_cover",
        "novelcover": "novel_cover",
        "cover": "novel_cover",
        "novel_page": "novel_page",
        "novelpage": "novel_page",
        "page": "novel_page",
        "illustration": "novel_page",
        "newspaper_article_image": "newspaper_article_image",
        "newspaper_article": "newspaper_article_image",
        "article_image": "newspaper_article_image",
        "article": "newspaper_article_image",
        "newspaper_page": "newspaper_page",
        "newspaper": "newspaper_page",
    }
    return aliases.get(raw, "general")


def _context_config(context: str) -> dict:
    ctx = _normalize_context(context)
    configs = {
        "novel_cover": {
            "context": "novel_cover",
            "prefix": "The novel cover image appears to show",
            "subjects": NOVEL_SUBJECT_WORDS,
            "score_dtype": "Novel",
            "fallback": (
                "The novel cover image appears to show an illustrated scene "
                "related to the story."
            ),
            "no_image_message": None,
        },
        "novel_page": {
            "context": "novel_page",
            "prefix": "The page illustration appears to show",
            "subjects": NOVEL_SUBJECT_WORDS,
            "score_dtype": "Novel",
            "fallback": NOVEL_PAGE_NO_IMAGE,
            "no_image_message": NOVEL_PAGE_NO_IMAGE,
        },
        "newspaper_article_image": {
            "context": "newspaper_article_image",
            "prefix": "The selected article image appears to show",
            "subjects": NEWSPAPER_SUBJECT_WORDS | MAGAZINE_SUBJECT_WORDS,
            "score_dtype": "Newspaper",
            "fallback": ARTICLE_NO_IMAGE,
            "no_image_message": ARTICLE_NO_IMAGE,
        },
        "newspaper_page": {
            "context": "newspaper_page",
            "prefix": "The newspaper page appears to contain",
            "subjects": NEWSPAPER_SUBJECT_WORDS,
            "score_dtype": "Newspaper",
            "fallback": NEWSPAPER_NO_IMAGE,
            "no_image_message": NEWSPAPER_NO_IMAGE,
        },
        "general": {
            "context": "general",
            "prefix": "The image appears to show",
            "subjects": GENERAL_SUBJECT_WORDS,
            "score_dtype": "Novel",
            "fallback": GENERAL_NO_IMAGE,
            "no_image_message": GENERAL_NO_IMAGE,
        },
    }
    return configs.get(ctx, configs["general"])


def _normalize_selected_visual_sentence_with_prefix(
    sentence: str,
    prefix: str,
    subjects: set[str],
    score_dtype: str,
) -> str:
    """Like document-type normalize, but with an explicit context prefix."""
    visual = _clean_text(sentence)
    visual = _strip_author_phrases(visual)
    visual = re.sub(r'[\"“”\']+', " ", visual)
    visual = re.sub(r"\s{2,}", " ", visual).strip(" ,;:-.")
    visual = _clean_text(visual)
    if not visual:
        return ""

    visual = _strip_existing_prefixes(visual)
    if not visual:
        return ""

    body_l = _normalize_compare(visual)
    if body_l in {"", "cover", "title", "title text", "text", "story", "novel", "image"}:
        return ""
    if body_l in {"a cover", "the cover", "a title", "the title"}:
        return ""

    if visual.lower().startswith(prefix.lower()):
        final = _to_one_sentence(visual)
    else:
        body = visual[0].lower() + visual[1:] if visual else visual
        body = re.sub(
            r"\bappears on the (?:page|cover|letter|document|article)\b\.?$",
            "",
            body,
            flags=re.IGNORECASE,
        )
        body = _clean_text(body)
        verb_fixes = (
            (r"\blooks\b", "looking"),
            (r"\bstands\b", "standing"),
            (r"\bsits\b", "sitting"),
            (r"\bholds\b", "holding"),
            (r"\bsails\b", "sailing"),
            (r"\bposes\b", "posing"),
            (r"\bappears\b", "appearing"),
            (r"\bcelebrates\b", "celebrating"),
            (r"\bshows\b", "showing"),
        )
        for pattern, replacement in verb_fixes:
            body = re.sub(pattern, replacement, body, flags=re.IGNORECASE)
        if re.match(r"^(?:in|on|above|below|with)\b", body, flags=re.IGNORECASE):
            body = f"a visual {body}"
        final = f"{prefix} {body}."
        if not final.endswith("."):
            final += "."

    if not _is_valid_florence_description(final, document_type=score_dtype):
        # Allow context-specific prefixes that document-type validation may reject.
        lowered = final.lower()
        if re.search(r"appears to (?:show|contain)\s*\.", lowered):
            return ""
        if re.search(r"appears to (?:show|contain)\s*[\"']\s*[\"']", lowered):
            return ""
        if len(final) < 15:
            return ""
    if _is_non_visual_sentence(final, subjects):
        return ""
    return final


def format_florence_context_caption(
    raw_caption: str,
    context: str = "general",
) -> tuple[str, bool]:
    """
    Format Florence raw caption for image-only reuse with context wording.

    Returns:
        (description, has_image)
    """
    cfg = _context_config(context)
    ctx = str(cfg["context"])
    prefix = str(cfg["prefix"])
    subjects = set(cfg["subjects"])
    score_dtype = str(cfg["score_dtype"])
    fallback = str(cfg["fallback"])
    no_image_message = cfg.get("no_image_message")

    print(f"[image_description] context={ctx}")
    raw = _clean_text(raw_caption)
    if not raw:
        has_image = no_image_message is None
        desc = fallback if no_image_message is None else str(no_image_message)
        return desc, has_image

    sentences = _split_caption_sentences(raw)
    best_sentence = ""
    best_score = 0
    for sentence in sentences:
        if _is_non_visual_sentence(sentence, subjects):
            print(f"[Florence] rejected non-visual sentence={sentence!r}")
            continue
        score = _visual_sentence_score(sentence, subjects, score_dtype)
        if score > best_score:
            best_score = score
            best_sentence = sentence

    if best_sentence and best_score > 0:
        print(f"[Florence] selected visual sentence={best_sentence!r}")
        final = _normalize_selected_visual_sentence_with_prefix(
            best_sentence,
            prefix=prefix,
            subjects=subjects,
            score_dtype=score_dtype,
        )
        if final:
            return final, True

    if not _is_non_visual_sentence(raw, subjects):
        score = _visual_sentence_score(raw, subjects, score_dtype)
        if score > 0:
            final = _normalize_selected_visual_sentence_with_prefix(
                raw,
                prefix=prefix,
                subjects=subjects,
                score_dtype=score_dtype,
            )
            if final:
                return final, True

    if no_image_message is None:
        # novel_cover soft visual fallback still counts as an image description.
        return fallback, True
    return str(no_image_message), False


def _title_case_candidate(words: list[str]) -> str:
    out = []
    for i, word in enumerate(words):
        upper = word.upper()
        if upper in {"PC", "UK", "USA", "US", "GQ", "LMD"}:
            out.append(upper)
        elif i != 0 and upper in {"AND", "THE", "OF", "A", "AN"}:
            out.append(word.lower())
        elif word.isupper() and len(word) > 1:
            out.append(word.capitalize())
        else:
            out.append(word.capitalize() if word[:1].islower() else word)
    return " ".join(out)


def _is_descriptive_title_phrase(text: str) -> bool:
    """True when candidate is cover/layout description, not a real title."""
    cleaned = _clean_text(text)
    if not cleaned:
        return True
    lowered = _normalize_compare(cleaned)
    if not lowered:
        return True
    if any(phrase in lowered for phrase in DESCRIPTIVE_TITLE_PHRASES):
        return True
    for pattern in DESCRIPTIVE_TITLE_PATTERNS:
        if re.search(pattern, lowered, flags=re.IGNORECASE):
            return True
    # Reject "written in ..." fragments entirely.
    if re.search(r"\bwritten in\b", lowered):
        return True
    return False


def _is_bad_title_candidate(title: str) -> bool:
    cleaned = _clean_text(title)
    if not cleaned:
        return True
    if _is_descriptive_title_phrase(cleaned):
        return True
    words = cleaned.split()
    if len(words) == 0 or len(words) > 6:
        return True
    lowered = {w.lower().strip(".,;:'\"") for w in words}
    if lowered & BAD_TITLE_TOKENS:
        return True
    # Color + letters style leftovers.
    if "letters" in lowered or "lettering" in lowered:
        return True
    if any(ch.isdigit() for ch in cleaned) and len(words) <= 1:
        return True
    return False


def _reject_florence_title_candidate(candidate: str, reason: str) -> None:
    print(
        f"[title_reader] rejected florence title candidate={candidate!r} "
        f"reason={reason}"
    )
    logger.info(
        "rejected florence title candidate=%r reason=%s",
        candidate,
        reason,
    )


def _accept_florence_title_candidate(candidate: str) -> str:
    print(f"[title_reader] accepted florence title candidate={candidate!r}")
    logger.info("accepted florence title candidate=%r", candidate)
    return candidate


def _finalize_recovered_title(candidate: str) -> str | None:
    raw_candidate = _clean_text(candidate)
    if not raw_candidate:
        return None
    if _is_descriptive_title_phrase(raw_candidate):
        _reject_florence_title_candidate(raw_candidate, "descriptive_phrase")
        return None

    candidate = _strip_author_phrases(raw_candidate)
    candidate = candidate.strip(" .,;:'\"“”`-")
    # Stop before descriptive continuation clauses.
    candidate = re.split(
        r"\b(?:showing|depicting|featuring|with|where|who|that|by|written|appearing|appears)\b",
        candidate,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip(" .,;:'\"“”`-")
    words = [w for w in candidate.split() if w]
    while words and words[-1].lower() in {"the", "a", "an", "of", "and"}:
        words.pop()
    if not words:
        return None
    title = _title_case_candidate(words)
    known = KNOWN_FLORENCE_TITLES.get(title.lower())
    if known:
        title = known
    if _is_descriptive_title_phrase(title):
        _reject_florence_title_candidate(title, "descriptive_phrase")
        return None
    if _is_bad_title_candidate(title):
        _reject_florence_title_candidate(title, "bad_title_tokens")
        return None
    return _accept_florence_title_candidate(title)


def _extract_called_titled_named(raw: str) -> str | None:
    """
    Prefer explicit title naming patterns:
      book called <TITLE>
      book titled <TITLE>
      newspaper called <TITLE>
    """
    # Stop before sentence punctuation / descriptive clauses.
    # Allow honorific abbreviations: Mr. Mrs. Ms. Dr. St.
    stop = (
        r"(?="
        r"(?:(?<!Mr)(?<!Mrs)(?<!Ms)(?<!Dr)(?<!St)[.!?])|"
        r"[,;:]|"
        r"\s+\b(?:showing|depicting|featuring|with|where|"
        r"who|that|by|written|appearing|appears|and the|on the)\b|"
        r"$"
        r")"
    )
    title_body = rf"([A-Za-z][A-Za-z0-9.' \-]{{1,60}}?){stop}"

    patterns = [
        rf"cover of (?:a |the )?(?:book|novel|magazine)\s+(?:called|titled|named)\s+{title_body}",
        rf"(?:book|novel|magazine)\s+(?:called|titled|named)\s+{title_body}",
        rf"(?:newspaper|masthead)\s+(?:called|titled|named)\s+{title_body}",
        rf"(?:is|was)\s+(?:called|titled|named)\s+{title_body}",
        rf"(?:called|titled|named)\s+{title_body}",
    ]
    for pattern in patterns:
        match = re.search(pattern, raw, flags=re.IGNORECASE)
        if not match:
            continue
        candidate = match.group(1).strip(" .,;:'\"“”`-")
        # Keep trailing period for honorifics like "Mr."
        raw_group = match.group(1).strip(" ,;:'\"“”`-")
        if re.search(r"\b(?:Mr|Mrs|Ms|Dr|St)\.$", raw_group, flags=re.IGNORECASE):
            # Include the next word after honorific if present in original span.
            after = raw[match.end(1) : match.end(1) + 40]
            next_word = re.match(r"\.?\s*([A-Za-z][A-Za-z0-9'\-]*)", after)
            if next_word:
                candidate = f"{raw_group.rstrip('.')} {next_word.group(1)}".strip()
                # Prefer known "Mr. China" form
                known = KNOWN_FLORENCE_TITLES.get(_normalize_compare(candidate))
                if known:
                    candidate = known
                elif candidate.lower().startswith("mr "):
                    candidate = "Mr. " + candidate[3:]
        if _is_descriptive_title_phrase(candidate):
            _reject_florence_title_candidate(candidate, "descriptive_phrase")
            continue
        finalized = _finalize_recovered_title(candidate)
        if finalized:
            return finalized
    return None


def _recover_cover_title(raw: str) -> str | None:
    lowered = raw.lower()

    # Never recover from "title ... is written in ..." captions.
    if re.search(
        r"\btitle of the book is written\b|\btitle(?:\s+\w+){0,4}\s+written in\b",
        lowered,
    ):
        _reject_florence_title_candidate(raw, "descriptive_phrase")
        # Still allow an explicit "called/titled" title elsewhere in the caption.
        named = _extract_called_titled_named(raw)
        if named:
            return named
        return None

    if re.search(r"\bulysses\b", raw, flags=re.IGNORECASE):
        return _accept_florence_title_candidate("Ulysses")

    for key, title in sorted(
        KNOWN_FLORENCE_TITLES.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        if re.search(rf"\b{re.escape(key)}\b", lowered):
            return _accept_florence_title_candidate(title)

    # Highest-priority: explicit called/titled/named extractions.
    named = _extract_called_titled_named(raw)
    if named:
        return named

    patterns = [
        r'[\"“”\']\s*([^\"“”\']+?)\s*[\"“”\']\s*by\b',
        r'[\"“”\']\s*([^\"“”\']+?)\s*[\"“”\']',
        # Require a real title after "title of the book is", not "written...".
        r"title of the book is\s+(?!written\b)([A-Za-z][A-Za-z0-9.' \-]{1,50})"
        r"(?=\s*(?:[.!?,;:]|$|\b(?:showing|depicting|featuring|with|by|written)\b))",
    ]

    for pattern in patterns:
        match = re.search(pattern, raw, flags=re.IGNORECASE)
        if not match:
            continue
        candidate = match.group(1)
        if _is_descriptive_title_phrase(candidate):
            _reject_florence_title_candidate(candidate, "descriptive_phrase")
            continue
        finalized = _finalize_recovered_title(candidate)
        if finalized:
            return finalized

    upper_tokens = re.findall(r"\b([A-Z]{3,})\b", raw)
    for token in upper_tokens:
        if token.lower() in BAD_TITLE_TOKENS:
            continue
        if token in {
            "THE",
            "AND",
            "FOR",
            "WITH",
            "FROM",
            "THIS",
            "THAT",
            "BOOK",
            "NOVEL",
            "REPORT",
            "LETTER",
            "WRITTEN",
            "LETTERS",
            "RED",
            "BLACK",
            "GOLD",
            "COVER",
        }:
            continue
        if _is_descriptive_title_phrase(token):
            _reject_florence_title_candidate(token, "descriptive_phrase")
            continue
        finalized = _finalize_recovered_title(token)
        if finalized:
            return finalized

    return None


def _recover_newspaper_masthead(raw: str) -> str | None:
    lowered = raw.lower()
    for key, title in sorted(
        KNOWN_NEWSPAPER_MASTHEADS.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        if re.search(rf"\b{re.escape(key)}\b", lowered):
            # Require masthead-ish context when the name is short/generic.
            if key in {"times", "guardian", "forbes"} and not re.search(
                r"\b(newspaper|masthead|daily|times|guardian|post)\b",
                lowered,
            ):
                continue
            return _accept_florence_title_candidate(title)

    named = _extract_called_titled_named(raw)
    if named:
        return named

    patterns = [
        r"masthead\s+(?:reads|says)\s+([A-Za-z][A-Za-z0-9' \-]{1,40})",
        r'[\"“”\']\s*([^\"“”\']{2,40}?)\s*[\"“”\'].{0,40}\b(?:newspaper|masthead)\b',
        r"\b(?:newspaper|masthead)\b.{0,40}[\"“”\']\s*([^\"“”\']{2,40}?)\s*[\"“”\']",
    ]
    for pattern in patterns:
        match = re.search(pattern, raw, flags=re.IGNORECASE)
        if not match:
            continue
        candidate = match.group(1)
        if _is_descriptive_title_phrase(candidate):
            _reject_florence_title_candidate(candidate, "descriptive_phrase")
            continue
        finalized = _finalize_recovered_title(candidate)
        if finalized:
            return finalized
    return None


def _recover_strict_title(raw: str) -> str | None:
    """Only recover when Florence clearly names a title (reports/letters)."""
    named = _extract_called_titled_named(raw)
    if named:
        return named

    patterns = [
        r"(?:title|heading)\s+(?:is|reads|says)\s+(?!written\b)[\"“”']?"
        r"([A-Za-z][A-Za-z0-9' \-]{1,50})",
        r"(?:document|report|letter)\s+(?:titled|called|named)\s+[\"“”']?"
        r"([A-Za-z][A-Za-z0-9' \-]{1,50})",
        r'[\"“”\']\s*([^\"“”\']{3,50}?)\s*[\"“”\'].{0,30}\b(?:title|heading|report|letter)\b',
    ]
    for pattern in patterns:
        match = re.search(pattern, raw, flags=re.IGNORECASE)
        if not match:
            continue
        candidate = match.group(1)
        if _is_descriptive_title_phrase(candidate):
            _reject_florence_title_candidate(candidate, "descriptive_phrase")
            continue
        finalized = _finalize_recovered_title(candidate)
        if finalized:
            return finalized
    return None


def _match_known_florence_title(raw: str) -> str | None:
    lowered = raw.lower()
    for key, title in sorted(
        KNOWN_FLORENCE_TITLES.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        if re.search(rf"\b{re.escape(key)}\b", lowered):
            return _accept_florence_title_candidate(title)
    return None


def recover_title_from_florence_caption(
    raw_caption: str,
    document_type: str = "",
) -> str | None:
    """
    Recover a visible title/name from Florence raw caption text.
    Behavior depends on document_type.
    Rejects descriptive/non-title phrases.
    """
    raw = _clean_text(raw_caption)
    if not raw:
        return None

    # Prefer explicit named titles even when caption also has descriptive wording.
    named = _extract_called_titled_named(raw)
    if named:
        return named

    # Reject written-in / lettering-only captions with no recoverable title.
    lowered = _normalize_compare(raw)
    if re.search(r"\bwritten in\b", lowered) or re.search(
        r"\b(?:red|black|gold|blue|white)\s+letters?\b",
        lowered,
    ):
        known = _match_known_florence_title(raw)
        if known:
            return known
        _reject_florence_title_candidate(raw, "descriptive_phrase")
        return None

    cfg = _doc_config(document_type)
    mode = str(cfg["title_mode"])
    if mode == "masthead":
        return _recover_newspaper_masthead(raw)
    if mode == "strict":
        return _recover_strict_title(raw)
    return _recover_cover_title(raw)


def describe_image_florence(
    image_path: str,
    document_type: str = "",
) -> dict[str, str]:
    """
    Describe cropped reading-material cover/page using local Florence-2.

    Returns:
        {
          "florence_raw_caption": "...",
          "florence_cleaned_caption": "...",
        }
        or {} if image missing / caption empty.
    Raises:
        RuntimeError if Florence cannot be loaded or inference fails hard.
    """
    print(f"[Florence] Image path: {image_path}")
    print(f"[Florence] crop path used for Florence: {image_path}")
    dtype = str(_doc_config(document_type)["dtype"])
    print(f"[Florence] document_type_for_description={dtype}")
    logger.info(
        "local Florence description requested for path=%r document_type=%s",
        image_path,
        dtype,
    )

    if not image_path or not str(image_path).strip():
        print("[Florence] skipped: empty image_path")
        return {}

    if not os.path.isfile(image_path):
        print(f"[Florence] skipped: file does not exist path={image_path!r}")
        return {}

    processor, model, device = _load_florence()

    image = Image.open(image_path).convert("RGB")
    max_w = 1200
    if image.width > max_w:
        new_h = max(1, int(round(image.height * (max_w / float(image.width)))))
        image = image.resize((max_w, new_h), Image.Resampling.LANCZOS)
        print(f"[Florence] resized input for inference: {image.width}x{image.height}")

    inputs = processor(text=TASK_PROMPT, images=image, return_tensors="pt")
    for key, value in list(inputs.items()):
        if hasattr(value, "to"):
            if key == "pixel_values" and device == "cuda":
                inputs[key] = value.to(device, dtype=torch.float16)
            else:
                inputs[key] = value.to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=128,
            num_beams=3,
            do_sample=False,
        )

    generated_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=False,
    )[0]
    parsed = processor.post_process_generation(
        generated_text,
        task=TASK_PROMPT,
        image_size=(image.width, image.height),
    )
    florence_raw_caption = ""
    if isinstance(parsed, dict):
        florence_raw_caption = str(parsed.get(TASK_PROMPT) or "").strip()
    if not florence_raw_caption:
        florence_raw_caption = _clean_text(generated_text)

    print(f"[Florence] raw caption: {florence_raw_caption!r}")
    logger.info("Florence raw caption=%r", florence_raw_caption)

    florence_cleaned_caption = format_florence_blind_caption(
        florence_raw_caption,
        document_type=dtype,
    )
    if florence_cleaned_caption and not _is_valid_florence_description(
        florence_cleaned_caption,
        document_type=dtype,
    ):
        florence_cleaned_caption = florence_fallback_for_document_type(dtype)
    print(f"[Florence] cleaned caption: {florence_cleaned_caption!r}")
    logger.info("Florence cleaned caption=%r", florence_cleaned_caption)

    if not florence_raw_caption and not florence_cleaned_caption:
        return {}

    return {
        "florence_raw_caption": florence_raw_caption,
        "florence_cleaned_caption": florence_cleaned_caption,
    }


print("Local Florence image description module loaded (lazy model load on first use)")

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

import cv2
import torch
from PIL import Image
import numpy as np
import easyocr


logger = logging.getLogger(__name__)


# =========================
# Load EasyOCR reader
# =========================

# EasyOCR uses CPU to avoid CUDA out-of-memory issues.
reader = easyocr.Reader(
    ["en"],
    gpu=False
)

NEWSPAPER_NAME_FALLBACK = "Newspaper name not clearly detected"

# Masthead / newspaper name only — not article headlines.
NEWSPAPER_PATTERNS = {
    "DAILY NEWS": "Daily News",
    "THE DAILY NEWS": "Daily News",
    "DAILY MIRROR": "Daily Mirror",
    "THE DAILY MIRROR": "Daily Mirror",
    "DAILY MAIL": "Daily Mail",
    "THE DAILY MAIL": "Daily Mail",
    "THE GUARDIAN": "The Guardian",
    "GUARDIAN": "The Guardian",
    "THE TIMES": "The Times",
    "SUNDAY TIMES": "Sunday Times",
    "THE SUNDAY TIMES": "Sunday Times",
    "THE SUN": "The Sun",
    "NEW YORK TIMES": "New York Times",
    "THE NEW YORK TIMES": "New York Times",
    "WASHINGTON POST": "Washington Post",
    "THE WASHINGTON POST": "Washington Post",
    "WALL STREET JOURNAL": "Wall Street Journal",
    "THE WALL STREET JOURNAL": "Wall Street Journal",
    "USA TODAY": "USA Today",
    "FINANCIAL TIMES": "Financial Times",
    "THE FINANCIAL TIMES": "Financial Times",
    "THE HINDU": "The Hindu",
    "TIMES OF INDIA": "Times of India",
    "THE TIMES OF INDIA": "Times of India",
    "DAILY NEWS SRI LANKA": "Daily News",
    "SUNDAY OBSERVER": "Sunday Observer",
    "THE ISLAND": "The Island",
    "DAILY FT": "Daily FT",
    "CEYLON TODAY": "Ceylon Today",
    "THE MORNING": "The Morning",
}

# Distinctive first/key tokens that uniquely identify a masthead when OCR is partial.
NEWSPAPER_UNIQUE_TOKENS = {
    "CEYLON": "Ceylon Today",
    "GUARDIAN": "The Guardian",
    "HINDU": "The Hindu",
    "ISLAND": "The Island",
    "OBSERVER": "Sunday Observer",
}


# =========================
# Text cleaning helpers
# =========================

def clean_easyocr_title_text(text):
    """
    Clean OCR text for title extraction.
    """
    text = text.upper()
    text = text.replace("&", " AND ")
    text = text.replace("+", " AND ")
    text = re.sub(r"[^A-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def nice_title_case(title):
    """
    Convert uppercase title to readable title case.
    Preserve short acronyms such as PC, UK, USA.
    """
    small_words = [
        "AND", "THE", "OF", "IN", "ON",
        "TO", "A", "AN", "FOR", "WITH"
    ]
    acronyms = {
        "PC", "UK", "USA", "US", "UAE", "EU",
        "GQ", "LMD", "BBC", "CNN", "NBA", "NFL",
        "NY", "DC", "AI", "IT", "TV",
    }

    words = title.split()
    final_words = []

    for i, word in enumerate(words):
        upper = word.upper()
        if upper in acronyms:
            final_words.append(upper)
        elif i != 0 and upper in small_words:
            final_words.append(word.lower())
        else:
            final_words.append(word.capitalize())

    return " ".join(final_words)


COVER_PUBLISHER_NOISE = {
    "LADYBIRD", "PEARSON", "LONGMAN", "CLASSICS", "CLASSIC",
    "LIBRARY", "ISSUE", "MAGAZINE", "PAGE", "PAGES",
    "PUBLISHER", "PUBLISHING", "EDITION", "BESTSELLER",
    "BEST", "SELLER", "PENGUIN", "OXFORD", "CAMBRIDGE",
    "SCHOLASTIC", "BLOOMSBURY", "MACMILLAN", "HARPERCOLLINS",
    "HARPER", "COLLINS", "RANDOM", "HOUSE", "VINTAGE",
    "PAN", "DK", "DISNEY", "MARVEL",
    "COMICS", "PRESS", "SERIES",
    "VOLUME", "VOL", "NOVEL", "AUTHOR", "WRITTEN",
    "ILLUSTRATED", "COPYRIGHT", "RESERVED", "RIGHTS",
    "PRINT", "PRINTED", "INDIA", "LIMITED", "LTD",
    "INC", "CORP", "COMPANY", "CO",
    "SLIT", "SLIIT", "BARCODE", "ISBN", "ISSN",
}

COVER_OCR_NOISE_TOKENS = {
    "AUAT", "AUAC", "AUA", "AUAD", "AUAE",
    "IIII", "IIIIII", "WWW", "HTTP", "HTTPS",
    "COM", "ORG", "NET",
    "YSCLF", "YSCIF", "YSCL", "YSELF", "ITSELF",
    "YSS", "YSS17", "YSSI7", "YSS1", "YSS7",
    "VUNNICI", "VUNNIC", "OROWE", "DEECHER", "FARRIET",
    "BEECHER", "STOWE",  # author fragments often misread as title
}

COVER_NOISE_SUBSTRINGS = (
    "yss", "yscl", "auat", "auac", "ysclf", "yscif",
    "vunnici", "orowe", "deecher", "farriet",
)

# Known strong single-word magazine/novel brands that should NOT trigger fallback.
STRONG_SINGLE_WORD_TITLES = {
    "FORBES", "VOGUE", "TIME", "ELLE", "WIRED", "GQ", "PULSE",
    "SHAPE", "FORTUNE", "PEOPLE", "DRACULA", "FRANKENSTEIN",
    "ALLURE", "GLAMOUR", "INSTYLE", "ESQUIRE", "LMD",
}


def is_ocr_noise_token(word: str) -> bool:
    word = clean_easyocr_title_text(word)
    if word == "":
        return True
    if word in COVER_OCR_NOISE_TOKENS:
        return True
    # Odd short OCR junk: starts with AUA... / YSS...
    if re.fullmatch(r"AUA[A-Z0-9]{0,3}", word):
        return True
    if re.fullmatch(r"YSS[A-Z0-9]{0,4}", word):
        return True
    if re.fullmatch(r"YSCL[A-Z0-9]{0,3}", word):
        return True
    # Very short non-acronym consonant soup (keep PC/UK/USA via acronym set in nicer path)
    if len(word) <= 2 and word not in {"PC", "UK", "US", "NY", "DC", "AI", "IT", "TV", "GQ"}:
        # Allow common short title words
        if word not in {"OF", "TO", "IN", "ON", "A", "AN", "OR", "MY", "ME", "WE", "NO", "GO"}:
            # 2-letter random OCR often noise unless known acronym
            vowels = set("AEIOU")
            if not any(ch in vowels for ch in word) and word not in {"PC", "TV", "NY", "DC"}:
                return True
    return False


def is_barcode_or_sticker_text(text: str) -> bool:
    text = clean_easyocr_title_text(text)
    if text == "":
        return True
    if re.fullmatch(r"[\d\s]+", text):
        return True
    digits = sum(ch.isdigit() for ch in text)
    letters = sum(ch.isalpha() for ch in text)
    if digits >= 6 and digits >= letters:
        return True
    if any(tok in text.split() for tok in ("ISBN", "ISSN", "BARCODE", "SLIT", "SLIIT")):
        return True
    return False


def compose_known_title_from_tokens(tokens, document_type: str) -> str | None:
    """
    Rebuild known titles from split OCR tokens.
    e.g. JUNGLE + BOOK -> The Jungle Book
         UNCLE + TOM/TOMS + CABIN -> Uncle Tom's Cabin
         BUSY + BAKER -> The Busy Baker
         PC + POLLY -> PC Polly
         FORBES / SHAPE -> magazine brands
    """
    if not tokens:
        return None

    upper = []
    for token in tokens:
        cleaned = clean_easyocr_title_text(str(token))
        if cleaned:
            upper.extend(cleaned.split())

    # Normalize possessive OCR forms: TOMS / TOM + S -> TOM
    normalized: list[str] = []
    for token in upper:
        if token == "TOMS":
            normalized.append("TOM")
            continue
        if token == "S" and normalized and normalized[-1] == "TOM":
            # Drop possessive remnant from "Tom's"
            continue
        normalized.append(token)
    upper = normalized
    upper_set = set(upper)
    blob = " ".join(upper)

    known = detect_known_easyocr_title(blob, document_type)
    if known is not None:
        return known

    if document_type == "Novel":
        if "JUNGLE" in upper_set and "BOOK" in upper_set:
            return "The Jungle Book"
        if (
            "UNCLE" in upper_set
            and "CABIN" in upper_set
            and ("TOM" in upper_set or "TOMS" in upper_set)
        ):
            return "Uncle Tom's Cabin"
        if "UNCLE" in upper_set and "CABIN" in upper_set:
            return "Uncle Tom's Cabin"
        if "BUSY" in upper_set and "BAKER" in upper_set:
            return "The Busy Baker"
        if "PC" in upper_set and "POLLY" in upper_set:
            return "PC Polly"
        if "SECRET" in upper_set and "GARDEN" in upper_set:
            return "The Secret Garden"

    if document_type == "Magazine":
        if "FORBES" in upper_set:
            return "Forbes"
        if "SHAPE" in upper_set:
            return "Shape"
        known = detect_known_easyocr_title(blob, "Magazine")
        if known is not None:
            return known

    return None


def is_known_corrected_title(title: str, document_type: str) -> bool:
    """True when title is already a trusted known/corrected cover title."""
    title_str = str(title or "").strip()
    if not title_str or title_str == "Title not clearly detected":
        return False

    composed = compose_known_title_from_tokens(title_str.split(), document_type)
    if composed is not None and composed == title_str:
        return True

    known = detect_known_easyocr_title(title_str, document_type)
    if known is not None and known == title_str:
        return True

    return False


def apply_known_title_correction(
    text_or_tokens,
    document_type: str,
) -> str | None:
    """
    Apply known title corrections from OCR text or token lists.
    Returns the corrected title immediately when matched.
    """
    if text_or_tokens is None:
        return None

    if isinstance(text_or_tokens, str):
        tokens = text_or_tokens.split()
        blob = text_or_tokens
    else:
        tokens = list(text_or_tokens)
        blob = " ".join(str(t) for t in tokens)

    composed = compose_known_title_from_tokens(tokens, document_type)
    if composed is not None:
        print(f"[title_reader] known title correction matched: {composed}")
        logger.info("known title correction matched title=%r", composed)
        return composed

    known = detect_known_easyocr_title(blob, document_type)
    if known is not None:
        print(f"[title_reader] known title correction matched: {known}")
        logger.info("known title correction matched title=%r", known)
        return known

    return None


def is_weak_cover_title(
    title: str,
    document_type: str,
    central_group_texts: list[str] | None = None,
) -> tuple[bool, str]:
    """
    Decide whether the first-pass Novel/Magazine title needs stronger OCR fallback.
    Returns (is_weak, reason).
    """
    if title is None:
        return True, "empty_title"

    title_str = str(title).strip()
    if title_str == "" or title_str == "Title not clearly detected":
        return True, "unclear_or_empty"

    cleaned = clean_easyocr_title_text(title_str)
    words = cleaned.split()
    compact = cleaned.replace(" ", "")

    # Strong known single-word brands (Forbes, Vogue, ...) are acceptable.
    if len(words) == 1 and words[0] in STRONG_SINGLE_WORD_TITLES:
        return False, ""
    if len(words) == 1:
        known_single = detect_known_easyocr_title(words[0], document_type)
        if known_single is not None and known_single.upper().replace(" ", "") == words[0]:
            return False, ""

    # Full known multi-word title that matches exactly is strong.
    known_full = detect_known_easyocr_title(cleaned, document_type)
    if known_full is not None:
        known_norm = clean_easyocr_title_text(known_full)
        if known_norm == cleaned or cleaned in known_norm or known_norm in cleaned:
            # "JUNGLE" matching via accidental substring? detect uses pattern in text.
            # "JUNGLE BOOK" in "JUNGLE" is False, so known_full is None for Jungle alone.
            if len(words) >= 2 or words[0] in STRONG_SINGLE_WORD_TITLES:
                return False, ""

    if len(words) == 1:
        return True, "single_word"

    if len(compact) < 6:
        return True, "too_short"

    if re.search(r"(yss|yscl|auat|auac|ysclf|yscif)", title_str, flags=re.IGNORECASE):
        return True, "noise_pattern"

    if any(is_ocr_noise_token(w) for w in words):
        return True, "noise_token"

    letters = sum(ch.isalpha() for ch in compact)
    digits = sum(ch.isdigit() for ch in compact)
    if digits > 0 and digits >= max(1, letters // 2):
        return True, "mostly_numbers_or_noise"

    if is_barcode_or_sticker_text(cleaned):
        return True, "barcode_or_sticker"

    if central_group_texts is not None:
        central_blob = clean_easyocr_title_text(" ".join(central_group_texts))
        # If chosen title is not represented in the strongest central group, treat as weak.
        title_words = set(words)
        central_words = set(central_blob.split())
        if title_words and central_words:
            overlap = len(title_words & central_words) / float(len(title_words))
            if overlap < 0.5 and cleaned not in central_blob:
                return True, "not_in_central_group"

    return False, ""


def collapse_repeated_words(text: str) -> str:
    """Collapse consecutive repeats and drop later duplicate tokens."""
    words = text.split()
    if not words:
        return ""

    consecutive = []
    for word in words:
        if consecutive and consecutive[-1].upper() == word.upper():
            continue
        consecutive.append(word)

    seen = set()
    out = []
    for word in consecutive:
        key = word.upper()
        if key in seen and key not in {"THE", "AND", "OF", "A", "AN"}:
            continue
        seen.add(key)
        out.append(word)
    return " ".join(out)


def looks_like_person_name(text: str) -> bool:
    """
    Heuristic for author lines (e.g. Harriet Beecher Stowe).
    Titles usually start with THE/A/AN or contain title words.
    """
    text = clean_easyocr_title_text(text)
    words = text.split()
    if not (2 <= len(words) <= 4):
        return False
    if words[0] in {"THE", "A", "AN"}:
        return False
    titleish = {
        "OF", "AND", "BOOK", "STORY", "CABIN", "ADVENTURES",
        "TALE", "HOUSE", "GARDEN", "ISLAND", "FARM", "WOMEN",
        "POLLY", "BAKER", "BUSY", "JUNGLE",
    }
    if any(w in titleish for w in words):
        return False
    return True


def contains_cover_noise_text(text: str) -> bool:
    raw = str(text or "")
    lowered = raw.lower()
    if any(token in lowered for token in COVER_NOISE_SUBSTRINGS):
        return True
    cleaned = clean_easyocr_title_text(raw)
    for word in cleaned.split():
        if is_ocr_noise_token(word):
            return True
        if re.search(r"\d", word) and len(re.sub(r"\d", "", word)) <= 2:
            return True
    return False


def is_long_author_or_subtitle_fragment(text: str) -> bool:
    """Reject long author/subtitle OCR fragments mistaken for titles."""
    cleaned = clean_easyocr_title_text(text)
    words = cleaned.split()
    if len(words) >= 5:
        return True
    if looks_like_person_name(cleaned):
        return True
    # Multi-word fragments that are mostly person-name tokens.
    nameish = {
        "HARRIET", "FARRIET", "BEECHER", "STOWE", "RUDYARD", "KIPLING",
        "JANE", "AUSTEN", "ROWLING", "AUTHOR", "WRITTEN",
    }
    if words and sum(1 for w in words if w in nameish) >= 2:
        return True
    return False


def is_confident_cover_title(title: str, document_type: str) -> bool:
    """
    Accept only known/corrected titles or strong magazine brands.
    Never accept random OCR text.
    Known corrections always win (even if apostrophe cleanup creates short tokens).
    """
    if not title or str(title).strip() in {"", "Title not clearly detected"}:
        return False

    title_str = str(title).strip()

    # Known corrections must never be rejected by noise heuristics.
    if is_known_corrected_title(title_str, document_type):
        return True

    if contains_cover_noise_text(title_str):
        return False
    if is_long_author_or_subtitle_fragment(title_str):
        return False

    cleaned = clean_easyocr_title_text(title_str)
    words = cleaned.split()
    if not words:
        return False

    # Strong single-word magazine/novel brands only.
    if len(words) == 1 and words[0] in STRONG_SINGLE_WORD_TITLES:
        known_single = detect_known_easyocr_title(words[0], document_type)
        if known_single is not None and known_single == title_str:
            return True
        if document_type == "Magazine" and nice_title_case(words[0]) == title_str:
            return True

    return False


def finalize_cover_title(text: str, document_type: str) -> str | None:
    """
    Return only a trusted Novel/Magazine title.
    If OCR text cannot be confidently corrected, return None.
    """
    text = collapse_repeated_words(strip_cover_noise_words(text))
    if text == "":
        return None

    if contains_cover_noise_text(text) or is_long_author_or_subtitle_fragment(text):
        return None

    composed = compose_known_title_from_tokens(text.split(), document_type)
    if composed is not None:
        return composed

    known = detect_known_easyocr_title(text, document_type)
    if known is not None:
        return known

    words = text.split()
    if len(words) == 1 and words[0] in STRONG_SINGLE_WORD_TITLES:
        known_single = detect_known_easyocr_title(words[0], document_type)
        if known_single is not None:
            return known_single
        if document_type == "Magazine":
            return nice_title_case(words[0])

    # Do not invent titles from raw OCR.
    return None



def strip_cover_noise_words(text: str) -> str:
    """Remove publisher/logo/noise tokens from a title candidate line."""
    text = clean_easyocr_title_text(text)
    words = []
    for word in text.split():
        if word in COVER_PUBLISHER_NOISE:
            continue
        if is_ocr_noise_token(word):
            continue
        # Drop bare issue numbers after publisher cleanup.
        if re.fullmatch(r"\d{1,4}", word):
            continue
        words.append(word)
    return collapse_repeated_words(" ".join(words))


def is_publisher_only_line(text: str) -> bool:
    text = clean_easyocr_title_text(text)
    if text == "":
        return True
    words = text.split()
    if not words:
        return True
    meaningful = [
        w for w in words
        if w not in COVER_PUBLISHER_NOISE and not is_ocr_noise_token(w)
        and not re.fullmatch(r"\d{1,4}", w)
    ]
    return len(meaningful) == 0



def detect_known_easyocr_title(text, document_type):
    """
    Detect common magazine names and known novel titles from OCR text.
    """

    text = clean_easyocr_title_text(text)

    magazine_patterns = {
        "VOGUE": "Vogue",
        "TIME": "Time",
        "THE ECONOMIST": "The Economist",
        "ECONOMIST": "The Economist",
        "FORBES": "Forbes",
        "FORTUNE": "Fortune",
        "ELLE": "Elle",
        "PEOPLE": "People",
        "COSMOPOLITAN": "Cosmopolitan",
        "NATIONAL GEOGRAPHIC": "National Geographic",
        "NAT GEO": "National Geographic",
        "WIRED": "Wired",
        "GQ": "GQ",
        "ESQUIRE": "Esquire",
        "VANITY FAIR": "Vanity Fair",
        "ROLLING STONE": "Rolling Stone",
        "THE NEW YORKER": "The New Yorker",
        "NEW YORKER": "The New Yorker",
        "LMD": "LMD",
        "PULSE": "Pulse",
        "READER S DIGEST": "Reader's Digest",
        "READERS DIGEST": "Reader's Digest",
        "HARPER S BAZAAR": "Harper's Bazaar",
        "HARPERS BAZAAR": "Harper's Bazaar",
        "BAZAAR": "Harper's Bazaar",
        "MARIE CLAIRE": "Marie Claire",
        "GLAMOUR": "Glamour",
        "ALLURE": "Allure",
        "INSTYLE": "InStyle",
        "SCIENTIFIC AMERICAN": "Scientific American",
        "NEW SCIENTIST": "New Scientist",
        "POPULAR SCIENCE": "Popular Science",
        "POPULAR MECHANICS": "Popular Mechanics",
        "THE ATLANTIC": "The Atlantic",
        "ATLANTIC": "The Atlantic",
        "SHAPE": "Shape",
    }

    novel_patterns = {
        "THE SECRET GARDEN": "The Secret Garden",
        "SECRET GARDEN": "The Secret Garden",
        "PRIDE AND PREJUDICE": "Pride and Prejudice",
        "HARRY POTTER": "Harry Potter and the Cursed Child",
        "CURSED CHILD": "Harry Potter and the Cursed Child",
        "SIN EATER": "Sin Eater",
        "TREASURE ISLAND": "Treasure Island",
        "ALICE IN WONDERLAND": "Alice in Wonderland",
        "THE JUNGLE BOOK": "The Jungle Book",
        "JUNGLE BOOK": "The Jungle Book",
        "GREAT GATSBY": "The Great Gatsby",
        "MOBY DICK": "Moby Dick",
        "DRACULA": "Dracula",
        "FRANKENSTEIN": "Frankenstein",
        "JANE EYRE": "Jane Eyre",
        "LITTLE WOMEN": "Little Women",
        "ANIMAL FARM": "Animal Farm",
        "TO KILL A MOCKINGBIRD": "To Kill a Mockingbird",
        "LORD OF THE FLIES": "Lord of the Flies",
        "OLIVER TWIST": "Oliver Twist",
        "TOM SAWYER": "The Adventures of Tom Sawyer",
        "LITTLE FROG": "Little Frog",
        "THE STORY OF A LITTLE FROG": "The Story of a Little Frog",
        "PC POLLY": "PC Polly",
        "THE BUSY BAKER": "The Busy Baker",
        "BUSY BAKER": "The Busy Baker",
        "UNCLE TOM S CABIN": "Uncle Tom's Cabin",
        "UNCLE TOMS CABIN": "Uncle Tom's Cabin",
        "UNCLE TOM CABIN": "Uncle Tom's Cabin",
    }

    # Prefer longer magazine brand matches first via sorted check below.
    # "FORBES INDIA" / "SHAPE MAGAZINE" handled after noise strip + known match.

    # Masthead / newspaper name only — not article headlines.
    newspaper_patterns = NEWSPAPER_PATTERNS

    if document_type == "Magazine":
        for pattern, title in sorted(
            magazine_patterns.items(),
            key=lambda item: len(item[0]),
            reverse=True,
        ):
            if pattern in text:
                return title

    if document_type == "Novel":
        for pattern, title in sorted(
            novel_patterns.items(),
            key=lambda item: len(item[0]),
            reverse=True,
        ):
            if pattern in text:
                return title

    if document_type == "Newspaper":
        # Longer patterns first so "THE DAILY NEWS" wins over "DAILY NEWS".
        for pattern, title in sorted(
            newspaper_patterns.items(),
            key=lambda item: len(item[0]),
            reverse=True,
        ):
            if pattern in text:
                return title

    return None


def is_bad_easyocr_title(text):
    """
    Reject bad OCR title candidates.
    """

    if text is None:
        return True

    text = clean_easyocr_title_text(text)

    if text == "":
        return True

    words = text.split()

    if len(words) == 0:
        return True

    if len(words) > 9:
        return True

    one_letter_count = sum(1 for word in words if len(word) == 1)

    if one_letter_count >= 4:
        return True

    letters_only = re.sub(r"[^A-Z]", "", text)

    if len(letters_only) < 3:
        return True

    bad_words = [
        "NOVEL",
        "AUTHOR",
        "BESTSELLER",
        "PUBLISHER",
        "EDITION",
        "PAGE",
        "TURNER",
        "THRILLING",
        "DARK",
        "REVIEW",
        "PRAISE",
        "PRICE",
        "ISSUE",
        "DATE",
        "A",
        "AN"
    ]

    if text in bad_words:
        return True

    return False


def remove_author_and_extra_lines(text):
    """
    Remove known author or extra cover text.
    """
    text = clean_easyocr_title_text(text)

    remove_phrases = [
        "ERIC WELLS",
        "MEGAN CAMPISI",
        "EMMA DONOGHUE",
        "JANE AUSTEN",
        "FRANCES HODGSON BURNETT",
        "FRANCES HODGSON",
        "J K ROWLING",
        "JK ROWLING",
        "JOHN TIFFANY",
        "JACK THORNE",
        "RUDYARD KIPLING",
        "HARRIET BEECHER STOWE",
        "FARRIET BEECHER STOWE",
        "HARRIET BEECHER",
        "FARRIET BEECHER",
        "BEECHER STOWE",
        "A NOVEL",
        "NOVEL",
        "AUTHOR",
        "BY",
        "PUBLISHER",
        "EDITION",
        "BESTSELLER",
        "BEST SELLER",
        "INTERNATIONAL BESTSELLER"
    ]

    for phrase in remove_phrases:
        text = text.replace(phrase, " ")

    text = re.sub(r"\s+", " ", text).strip()
    return text


# =========================
# EasyOCR title extraction
# =========================

def is_bad_newspaper_masthead(text):
    """
    Reject article headlines / body text mistaken for masthead.
    Masthead names are usually short (1–4 words).
    """
    if text is None:
        return True

    text = clean_easyocr_title_text(text)

    if text == "":
        return True

    words = text.split()

    # Hard limit: mastheads are short; long lines are headlines.
    if len(words) == 0 or len(words) > 4:
        return True

    letters_only = re.sub(r"[^A-Z]", "", text)
    if len(letters_only) < 3:
        return True

    # Very long character runs without spaces are usually OCR glue / headlines.
    if len(letters_only) > 28 and len(words) <= 2:
        return True

    # Common article / section words — not newspaper names.
    headline_noise = [
        "SPORTS", "SPORT", "POLITICS", "BUSINESS", "WORLD",
        "LOCAL", "OPINION", "EDITORIAL", "WEATHER", "CLASSIFIED",
        "BREAKING", "NEWS BRIEF", "INSIDE", "PAGE", "VOL",
        "VOLUME", "EDITION", "COPYRIGHT", "CONTINUED", "PUBLIC",
        "GOVERNMENT", "MINISTER", "CRICKET", "FOOTBALL", "MARKET",
    ]
    if text in headline_noise:
        return True

    # Reject if dominated by known headline noise words.
    noise_hits = sum(1 for w in words if w in headline_noise)
    if noise_hits >= 1 and len(words) >= 2:
        # Allow "DAILY NEWS" etc. — NEWS alone in a 2-word known name is fine
        # only when paired with DAILY/SUNDAY etc. handled by known patterns later.
        if "DAILY" not in words and "SUNDAY" not in words and "TODAY" not in words:
            if any(w in ("PUBLIC", "GOVERNMENT", "MINISTER", "CRICKET") for w in words):
                return True

    return False


def _masthead_lexicon_words():
    """All distinctive words from known newspaper names, longest first."""
    words = set()
    for pattern in NEWSPAPER_PATTERNS:
        for word in pattern.split():
            if word not in {"THE", "OF", "AND", "A"}:
                words.add(word)
    return sorted(words, key=len, reverse=True)


def split_glued_masthead_text(text):
    """
    Split OCR glue like 'CEYLONPUBLIC' into known masthead tokens + leftovers.
    """
    text = clean_easyocr_title_text(text)
    if text == "":
        return []

    found = []
    remaining = text.replace(" ", "")

    for word in _masthead_lexicon_words():
        if word in remaining:
            found.append(word)
            remaining = remaining.replace(word, " ", 1)

    leftover = clean_easyocr_title_text(remaining)
    leftover_words = [
        w for w in leftover.split()
        if len(w) >= 3 and w not in {"AND", "THE", "OF"}
    ]
    return found + leftover_words


def match_newspaper_from_tokens(token_list):
    """
    Match a known newspaper when masthead words are split or partially glued.
    """
    if not token_list:
        return None

    tokens = [clean_easyocr_title_text(t) for t in token_list]
    tokens = [t for t in tokens if t]
    blob = " ".join(tokens)
    compact = re.sub(r"\s+", "", blob)

    for pattern, title in sorted(
        NEWSPAPER_PATTERNS.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        pattern_words = pattern.split()
        # All pattern words present as whole tokens or substrings in compact blob.
        if all(
            (word in tokens) or (word in compact) or (word in blob)
            for word in pattern_words
        ):
            return title

    # Partial but unique token (e.g. CEYLON from damaged 'CEYLONPUBLIC AND')
    for token in tokens:
        for unique, title in NEWSPAPER_UNIQUE_TOKENS.items():
            if unique in token or token == unique:
                return title
        # Also scan glued token interiors
        for unique, title in NEWSPAPER_UNIQUE_TOKENS.items():
            if unique in token:
                return title

    return None


def crop_for_easyocr_title(image_path, document_type):
    """
    Crop useful title areas.
    Novel/Magazine: one upper/center cover band (avoid duplicate OCR joins).
    Newspaper: top masthead region only (not article headlines).
    """

    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    crops = []

    if document_type == "Magazine":
        # Masthead / brand usually in the upper half.
        crops.append(image.crop((0, 0, width, max(1, int(height * 0.55)))))

    elif document_type == "Novel":
        # Cover title is often center / upper-center (author may sit above it).
        top = int(height * 0.08)
        bottom = int(height * 0.78)
        left = int(width * 0.04)
        right = int(width * 0.96)
        crops.append(image.crop((left, top, right, max(top + 1, bottom))))

    elif document_type == "Newspaper":
        # Single tight top band — avoid pulling article headlines.
        crops.append(image.crop((0, 0, width, max(1, int(height * 0.22)))))

    return crops


def build_full_novel_title(valid_results):
    """
    Build full title for novels from filtered OCR title lines.
    Prefers largest center/upper-center lines; ignores smaller author lines.
    """
    cluster = select_main_title_cluster(valid_results)
    if not cluster:
        return None

    title = " ".join([item["text"] for item in cluster])
    title = collapse_repeated_words(title)
    title = strip_cover_noise_words(title)
    return finalize_cover_title(title, "Novel")


def _score_cover_title_line(item: dict, crop_height: float, crop_width: float) -> float:
    """
    Higher score for LARGE center / upper-center cover text.
    Do not prefer absolute topmost text (author names often sit above the title).
    """
    y_center = (item["y_min"] + item["y_max"]) / 2.0
    x_center = (item["x_min"] + item["x_max"]) / 2.0
    y_ratio = y_center / max(crop_height, 1.0)
    x_ratio = x_center / max(crop_width, 1.0)

    area = float(item["height"]) * float(item["width"])
    # Dominant signal: glyph size / box area.
    size_score = float(item["height"]) * 5.0 + (area * 0.008)

    # Prefer center / upper-center band; soft-penalize very top author band.
    if 0.18 <= y_ratio <= 0.65:
        band_score = 40.0
    elif 0.12 <= y_ratio < 0.18:
        band_score = 8.0
    elif y_ratio < 0.12:
        band_score = -20.0
    elif y_ratio > 0.78:
        band_score = -50.0
    else:
        band_score = 5.0

    center_score = (1.0 - abs(x_ratio - 0.5) * 2.0) * 18.0
    conf_score = float(item["confidence"]) * 12.0

    # Soft penalty for author-like wording when not huge.
    name_penalty = 0.0
    if looks_like_person_name(item.get("text", "")):
        name_penalty = -35.0

    return size_score + band_score + center_score + conf_score + name_penalty


def select_main_title_cluster(candidates: list[dict]) -> list[dict]:
    """
    Keep the largest title line(s) and nearby large companions.
    Ignore smaller author/subtitle/logo lines.
    """
    if not candidates:
        return []

    ranked = sorted(
        candidates,
        key=lambda x: (float(x.get("height", 0.0)), float(x.get("cover_score", 0.0))),
        reverse=True,
    )
    best = ranked[0]
    max_h = float(best.get("height", 0.0)) or 1.0

    print(
        "[title_reader] selected title candidates (size/score): "
        f"{[(c.get('text'), round(float(c.get('height', 0)), 1), round(float(c.get('cover_score', 0)), 1)) for c in ranked[:8]]}"
    )
    logger.debug(
        "selected title candidates: %s",
        [(c.get("text"), c.get("height"), c.get("cover_score")) for c in ranked[:8]],
    )

    cluster = []
    for item in candidates:
        height = float(item.get("height", 0.0))
        # Require near-largest text.
        if height < max_h * 0.70:
            continue
        # Ignore smaller author-like lines even if somewhat large.
        if looks_like_person_name(item.get("text", "")) and height < max_h * 0.92:
            continue
        # Keep lines near the main title vertically.
        if abs(float(item["y_min"]) - float(best["y_min"])) > max(max_h * 2.8, 70.0):
            continue
        cluster.append(item)

    if not cluster:
        cluster = [best]

    cluster = sorted(cluster, key=lambda x: x["y_min"])[:4]
    print(
        "[title_reader] title cluster: "
        f"{[(c.get('text'), round(float(c.get('height', 0)), 1)) for c in cluster]}"
    )
    return cluster


def _preprocess_cover_variants(rgb_image: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """
    Limited preprocessing variants for fallback OCR (kept small for CPU speed).
    """
    variants = [("original", rgb_image)]

    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    variants.append(("gray_contrast", cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)))

    # Inverted Otsu helps white text on dark covers.
    _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inverted = cv2.bitwise_not(thresh)
    variants.append(("inverted_threshold", cv2.cvtColor(inverted, cv2.COLOR_GRAY2RGB)))

    return variants


def _cover_region_crops(full_rgb: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """
    Limited cover regions for fallback OCR (max combinations kept small).
    """
    h, w = full_rgb.shape[:2]
    regions = []

    # Center 70%
    y0 = int(h * 0.15)
    y1 = int(h * 0.85)
    x0 = int(w * 0.15)
    x1 = int(w * 0.85)
    regions.append(("center_70", full_rgb[y0:y1, x0:x1].copy()))

    # Upper-middle 60%
    y0 = int(h * 0.08)
    y1 = int(h * 0.68)
    x0 = int(w * 0.10)
    x1 = int(w * 0.90)
    regions.append(("upper_middle_60", full_rgb[y0:y1, x0:x1].copy()))

    return regions


# Max EasyOCR fallback passes for Novel/Magazine (CPU-bound).
MAX_TITLE_FALLBACK_OCR_ATTEMPTS = 3
TITLE_OCR_TIME_BUDGET_SEC = 40.0


def _easyocr_readtext_with_timeout(image_np: np.ndarray, timeout_sec: float):
    """
    Run EasyOCR with a hard wait timeout so the pipeline does not hang forever.
    Note: the worker thread may continue briefly after timeout; the caller returns.
    """
    if timeout_sec <= 0:
        raise TimeoutError("title OCR time budget exhausted")

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            reader.readtext,
            image_np,
            detail=1,
            paragraph=False,
        )
        try:
            return future.result(timeout=timeout_sec)
        except FuturesTimeoutError as exc:
            future.cancel()
            raise TimeoutError(
                f"title OCR exceeded {timeout_sec:.1f}s budget"
            ) from exc


def _ocr_lines_from_image(
    rgb_image: np.ndarray,
    source_tag: str,
    timeout_sec: float | None = None,
) -> list[dict]:
    """Run EasyOCR on one RGB image and return structured lines."""
    if rgb_image is None or rgb_image.size == 0:
        return []

    h, w = rgb_image.shape[:2]
    if timeout_sec is None:
        results = reader.readtext(rgb_image, detail=1, paragraph=False)
    else:
        results = _easyocr_readtext_with_timeout(rgb_image, timeout_sec)
    lines = []

    for bbox, text, conf in results:
        raw_text = str(text).strip()
        cleaned = clean_easyocr_title_text(raw_text)
        if cleaned == "":
            continue

        points = np.array(bbox)
        x_min = float(points[:, 0].min())
        x_max = float(points[:, 0].max())
        y_min = float(points[:, 1].min())
        y_max = float(points[:, 1].max())
        box_w = max(1.0, x_max - x_min)
        box_h = max(1.0, y_max - y_min)

        lines.append({
            "raw": raw_text,
            "text": cleaned,
            "confidence": float(conf),
            "height": box_h,
            "width": box_w,
            "area": box_h * box_w,
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
            "y_ratio": ((y_min + y_max) / 2.0) / max(h, 1),
            "x_ratio": ((x_min + x_max) / 2.0) / max(w, 1),
            "source": source_tag,
        })

    return lines


def _reject_fallback_line(text: str) -> bool:
    if is_barcode_or_sticker_text(text):
        return True
    if is_publisher_only_line(text):
        return True
    cleaned = strip_cover_noise_words(remove_author_and_extra_lines(text))
    if cleaned == "":
        return True
    words = cleaned.split()
    if words and all(is_ocr_noise_token(w) or re.fullmatch(r"\d+", w) for w in words):
        return True
    return False


def group_fallback_title_candidates(lines: list[dict]) -> list[str]:
    """
    Prefer large central boxes; group nearby words vertically/horizontally.
    """
    if not lines:
        return []

    usable = []
    for item in lines:
        text = strip_cover_noise_words(remove_author_and_extra_lines(item["text"]))
        if text == "" or _reject_fallback_line(item["text"]):
            continue
        if looks_like_person_name(text) and float(item.get("height", 0)) < 35:
            continue
        if float(item.get("y_ratio", 1.0)) > 0.82:
            continue

        cand = dict(item)
        cand["text"] = text
        cand["cover_score"] = (
            float(item["height"]) * 5.0
            + float(item["area"]) * 0.008
            + (1.0 - abs(float(item["x_ratio"]) - 0.5) * 2.0) * 20.0
            + (25.0 if 0.15 <= float(item["y_ratio"]) <= 0.70 else -10.0)
            + float(item["confidence"]) * 10.0
        )
        usable.append(cand)

    if not usable:
        return []

    usable = sorted(usable, key=lambda x: x["cover_score"], reverse=True)
    best = usable[0]
    max_h = float(best["height"]) or 1.0

    cluster = []
    for item in usable:
        if float(item["height"]) < max_h * 0.55:
            continue
        close_y = abs(float(item["y_min"]) - float(best["y_min"])) <= max(max_h * 2.5, 55.0)
        close_x = abs(float(item["x_min"]) - float(best["x_min"])) <= max(float(best["width"]) * 1.8, 80.0)
        if not (close_y or (close_x and abs(float(item["y_min"]) - float(best["y_min"])) <= max_h * 3.5)):
            continue
        if looks_like_person_name(item["text"]) and float(item["height"]) < max_h * 0.90:
            continue
        cluster.append(item)

    if not cluster:
        cluster = [best]

    cluster = sorted(cluster, key=lambda x: (x["y_min"], x["x_min"]))
    grouped_text = collapse_repeated_words(" ".join(item["text"] for item in cluster[:6]))

    candidates = [grouped_text]
    for item in usable[:8]:
        if item["text"] not in candidates:
            candidates.append(item["text"])

    print(f"[title_reader] grouped fallback candidates: {candidates}")
    logger.debug("grouped fallback candidates: %s", candidates)
    return [c for c in candidates if c]


def strong_fallback_cover_title(
    image_path: str,
    document_type: str,
    deadline: float | None = None,
) -> str | None:
    """
    Stronger OCR fallback for weak Novel/Magazine titles.

    Performance limits:
    - Max MAX_TITLE_FALLBACK_OCR_ATTEMPTS EasyOCR passes
    - Only a few region/preprocess combos
    - Early stop when a confident title is found
    - Respects optional deadline (time.monotonic) to avoid hangs
    """
    print(f"[title_reader] running strong title fallback on path={image_path!r}")
    logger.info("strong title fallback start path=%r type=%s", image_path, document_type)

    image = Image.open(image_path).convert("RGB")
    full_rgb = np.array(image)

    # Cap image size again as a safety net for CPU OCR.
    h, w = full_rgb.shape[:2]
    if w > 1200:
        scale = 1200 / float(w)
        full_rgb = cv2.resize(
            full_rgb,
            (1200, max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA,
        )

    regions = dict(_cover_region_crops(full_rgb))
    # Ordered attempts (max 3): prioritized for speed + usefulness.
    attempt_plan = [
        ("center_70", "original"),
        ("center_70", "inverted_threshold"),
        ("upper_middle_60", "gray_contrast"),
    ]

    all_lines: list[dict] = []
    attempts = 0

    for region_name, prep_name in attempt_plan:
        if attempts >= MAX_TITLE_FALLBACK_OCR_ATTEMPTS:
            break
        if deadline is not None and time.monotonic() >= deadline:
            print("[title_reader] fallback skipped: title OCR time budget exhausted")
            logger.warning("fallback skipped: title OCR time budget exhausted")
            break

        region_img = regions.get(region_name)
        if region_img is None or region_img.size == 0:
            continue

        prep_map = dict(_preprocess_cover_variants(region_img))
        prep_img = prep_map.get(prep_name)
        if prep_img is None:
            continue

        remaining = None
        if deadline is not None:
            remaining = max(0.5, deadline - time.monotonic())

        attempts += 1
        tag = f"{region_name}|{prep_name}"
        print(
            f"[title_reader] fallback OCR attempt "
            f"{attempts}/{MAX_TITLE_FALLBACK_OCR_ATTEMPTS}: {tag}"
        )
        try:
            lines = _ocr_lines_from_image(prep_img, tag, timeout_sec=remaining)
        except TimeoutError:
            print(f"[title_reader] fallback OCR timed out on {tag}")
            logger.warning("fallback OCR timed out source=%s", tag)
            break
        except Exception:
            logger.exception("fallback OCR failed source=%s", tag)
            continue
        all_lines.extend(lines)

        # Early compose check after each attempt — known corrections return immediately.
        token_bag = []
        for item in all_lines:
            cleaned = strip_cover_noise_words(remove_author_and_extra_lines(item["text"]))
            if cleaned:
                token_bag.extend(cleaned.split())
        composed = apply_known_title_correction(token_bag, document_type)
        if composed is not None:
            print(f"[title_reader] fallback early stop composed: {composed}")
            return composed

    deduped = []
    seen = set()
    for item in all_lines:
        key = (
            item["text"],
            round(float(item["y_ratio"]), 2),
            round(float(item["x_ratio"]), 2),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    print(
        "[title_reader] fallback OCR lines: "
        f"{[(l['raw'], l['source'], round(l['height'], 1), round(l['confidence'], 2)) for l in deduped[:40]]}"
    )
    logger.debug(
        "fallback OCR lines: %s",
        [(l["raw"], l["source"], l["height"], l["confidence"]) for l in deduped[:40]],
    )

    if not deduped:
        return None

    token_bag = []
    for item in deduped:
        cleaned = strip_cover_noise_words(remove_author_and_extra_lines(item["text"]))
        if cleaned:
            token_bag.extend(cleaned.split())

    composed = apply_known_title_correction(token_bag, document_type)
    if composed is not None:
        print(f"[title_reader] fallback composed known title: {composed}")
        return composed

    candidates = group_fallback_title_candidates(deduped)
    for cand in candidates:
        composed = apply_known_title_correction(cand, document_type)
        if composed is not None:
            print(f"[title_reader] fallback candidate composed: {composed}")
            return composed
        finalized = finalize_cover_title(cand, document_type)
        if finalized is not None:
            weak, reason = is_weak_cover_title(finalized, document_type)
            if not weak:
                print(f"[title_reader] fallback candidate finalized: {finalized}")
                return finalized
            print(
                f"[title_reader] fallback candidate still weak ({reason}): {finalized!r}"
            )

    return None


def extract_novel_magazine_title(image_path: str, document_type: str) -> str:
    """
    Clean Novel/Magazine title extraction using one cover-band OCR pass.
    Prefers largest center/upper-center boxes over topmost author text.
    If the first result is weak, runs a limited OCR fallback (max 3 attempts).
    If OCR is too slow or fails, returns "Title not clearly detected".
    """
    started = time.monotonic()
    deadline = started + TITLE_OCR_TIME_BUDGET_SEC

    crops = crop_for_easyocr_title(image_path, document_type)
    if not crops:
        logger.debug("final cleaned title: Title not clearly detected")
        return "Title not clearly detected"

    crop_image = crops[0]
    crop_w, crop_h = crop_image.size
    crop_np = np.array(crop_image)

    # Safety resize for CPU EasyOCR if a large image slipped through.
    if crop_w > 1200:
        scale = 1200 / float(crop_w)
        new_h = max(1, int(round(crop_h * scale)))
        crop_np = cv2.resize(crop_np, (1200, new_h), interpolation=cv2.INTER_AREA)
        crop_h, crop_w = crop_np.shape[:2]
        print(f"[title_reader] resized title crop for OCR: {crop_w}x{crop_h}")

    try:
        remaining = max(1.0, deadline - time.monotonic())
        raw_results = _easyocr_readtext_with_timeout(crop_np, remaining)
    except TimeoutError:
        print(
            "[title_reader] main OCR timed out; "
            "returning Title not clearly detected"
        )
        logger.warning("main title OCR timed out path=%r", image_path)
        return "Title not clearly detected"
    except Exception:
        logger.exception("main title OCR failed path=%r", image_path)
        print(
            "[title_reader] main OCR failed; "
            "returning Title not clearly detected"
        )
        return "Title not clearly detected"

    raw_lines = []
    all_results = []

    for bbox, text, conf in raw_results:
        raw_text = str(text).strip()
        cleaned_text = clean_easyocr_title_text(raw_text)
        if cleaned_text == "":
            continue

        points = np.array(bbox)
        x_min = float(points[:, 0].min())
        x_max = float(points[:, 0].max())
        y_min = float(points[:, 1].min())
        y_max = float(points[:, 1].max())
        box_width = x_max - x_min
        box_height = y_max - y_min

        item = {
            "raw": raw_text,
            "text": cleaned_text,
            "confidence": float(conf),
            "height": box_height,
            "width": box_width,
            "area": box_height * box_width,
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
            "score": (box_height * 2) + (box_width * 0.03) + (float(conf) * 30),
        }
        raw_lines.append(
            (
                raw_text,
                round(box_height, 1),
                round(y_min / max(crop_h, 1), 3),
                round(float(conf), 3),
            )
        )
        all_results.append(item)

    logger.debug("raw OCR lines (%s): %s", document_type, raw_lines)
    print(f"[title_reader] raw OCR lines ({document_type}): {raw_lines}")

    initial_title = "Title not clearly detected"
    central_group_texts: list[str] = []

    if all_results:
        combined_clean = collapse_repeated_words(
            strip_cover_noise_words(
                remove_author_and_extra_lines(
                    " ".join(item["text"] for item in all_results)
                )
            )
        )
        composed_blob = compose_known_title_from_tokens(
            combined_clean.split(),
            document_type,
        )
        known_title = detect_known_easyocr_title(combined_clean, document_type)
        if composed_blob is not None:
            initial_title = composed_blob
            print("[title_reader] initial title via token compose")
        elif known_title is not None:
            initial_title = known_title
            print("[title_reader] filtered candidate lines: known-match via blob")
        else:
            upper_blob_words = [w.upper() for w in combined_clean.split()]
            if "PC" in upper_blob_words and "POLLY" in upper_blob_words:
                initial_title = "PC Polly"
            else:
                max_height = max(float(item["height"]) for item in all_results)
                filtered = []
                for item in all_results:
                    y_ratio = ((item["y_min"] + item["y_max"]) / 2.0) / max(crop_h, 1.0)
                    if y_ratio > 0.82:
                        continue

                    cleaned = strip_cover_noise_words(
                        remove_author_and_extra_lines(item["text"])
                    )
                    if cleaned == "" or is_publisher_only_line(cleaned):
                        continue
                    if is_bad_easyocr_title(cleaned):
                        continue
                    if (
                        looks_like_person_name(cleaned)
                        and float(item["height"]) < max_height * 0.88
                    ):
                        continue

                    candidate = dict(item)
                    candidate["text"] = cleaned
                    candidate["cover_score"] = _score_cover_title_line(
                        candidate, crop_h, crop_w
                    )
                    filtered.append(candidate)

                filtered = sorted(
                    filtered, key=lambda x: x["cover_score"], reverse=True
                )
                print(
                    f"[title_reader] filtered candidate lines ({document_type}): "
                    f"{[(c['text'], round(float(c['height']), 1), round(float(c['cover_score']), 1)) for c in filtered]}"
                )

                if filtered:
                    cluster = select_main_title_cluster(filtered)
                    central_group_texts = [item["text"] for item in cluster]
                    cluster_text = collapse_repeated_words(
                        strip_cover_noise_words(
                            " ".join(item["text"] for item in cluster)
                        )
                    )
                    composed = compose_known_title_from_tokens(
                        cluster_text.split(), document_type
                    )
                    known = detect_known_easyocr_title(cluster_text, document_type)
                    finalized = finalize_cover_title(cluster_text, document_type)
                    if composed is not None:
                        initial_title = composed
                    elif known is not None:
                        initial_title = known
                    elif finalized is not None:
                        initial_title = finalized
                    else:
                        for item in filtered[:6]:
                            finalized = finalize_cover_title(
                                item["text"], document_type
                            )
                            if finalized is not None:
                                initial_title = finalized
                                break

    print(f"[title_reader] initial title: {initial_title!r}")
    logger.info("initial title=%r", initial_title)

    # Known corrections must be returned immediately — never overwritten later.
    if is_known_corrected_title(initial_title, document_type):
        print(f"[title_reader] known title correction matched: {initial_title}")
        logger.info("known title correction matched title=%r", initial_title)
        print(f"[title_reader] final selected title: {initial_title}")
        logger.debug("final cleaned title: %s", initial_title)
        return initial_title

    # Also try correcting weak OCR tokens from the initial candidate.
    corrected = apply_known_title_correction(initial_title, document_type)
    if corrected is not None:
        print(f"[title_reader] final selected title: {corrected}")
        logger.debug("final cleaned title: %s", corrected)
        return corrected

    weak, reason = is_weak_cover_title(
        initial_title,
        document_type,
        central_group_texts=central_group_texts or None,
    )

    if not weak:
        if is_confident_cover_title(initial_title, document_type):
            print(f"[title_reader] final selected title: {initial_title}")
            logger.debug("final cleaned title: %s", initial_title)
            return initial_title
        print(
            "[title_reader] initial title not confident; forcing fallback "
            f"title={initial_title!r}"
        )
        weak, reason = True, "not_confident"

    print(
        f"[title_reader] why fallback was triggered: {reason} "
        f"(initial title={initial_title!r})"
    )
    logger.info(
        "title fallback triggered reason=%s initial=%r",
        reason,
        initial_title,
    )

    if time.monotonic() >= deadline:
        print(
            "[title_reader] skipping fallback: title OCR time budget exhausted; "
            "returning Title not clearly detected"
        )
        logger.warning("skipping fallback: title OCR time budget exhausted")
        return "Title not clearly detected"

    fallback_title = strong_fallback_cover_title(
        image_path,
        document_type,
        deadline=deadline,
    )
    if fallback_title:
        # Known corrections from fallback always win.
        if is_known_corrected_title(fallback_title, document_type):
            print(f"[title_reader] known title correction matched: {fallback_title}")
            logger.info("known title correction matched title=%r", fallback_title)
            print(f"[title_reader] final selected title: {fallback_title}")
            logger.debug("final cleaned title: %s", fallback_title)
            return fallback_title
        corrected = apply_known_title_correction(fallback_title, document_type)
        if corrected is not None:
            print(f"[title_reader] final selected title: {corrected}")
            logger.debug("final cleaned title: %s", corrected)
            return corrected
        if is_confident_cover_title(fallback_title, document_type):
            print(f"[title_reader] final selected title: {fallback_title}")
            logger.debug("final cleaned title: %s", fallback_title)
            return fallback_title

    # Safer rule: if not confidently corrected, do not return noisy OCR text.
    print("[title_reader] final selected title: Title not clearly detected")
    logger.debug("final cleaned title: Title not clearly detected")
    return "Title not clearly detected"


def easyocr_extract_title(image_path, document_type):
    """
    Extract title using EasyOCR for Novel / Magazine / Newspaper masthead.
    For Novel/Magazine, uses cleaned cover-title extraction.
    For Newspaper, reads only the top masthead (not article headlines).
    EasyOCR runs on CPU to avoid GPU memory issues.
    """

    if document_type not in ["Novel", "Magazine", "Newspaper"]:
        return None

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Dedicated newspaper path — full-page top-band filtering.
    if document_type == "Newspaper":
        return easyocr_extract_newspaper_masthead(image_path=image_path)

    return extract_novel_magazine_title(image_path, document_type)


def _ocr_newspaper_masthead_region(image_path):
    """
    OCR only the top masthead band with full-page bounding boxes.
    """
    image = Image.open(image_path).convert("RGB")
    page_width, page_height = image.size

    top_ratio = 0.25
    crop_height = max(1, int(page_height * top_ratio))
    masthead_crop = image.crop((0, 0, page_width, crop_height))
    crop_np = np.array(masthead_crop)

    raw_results = reader.readtext(
        crop_np,
        detail=1,
        paragraph=False,
    )

    lines = []
    for bbox, text, conf in raw_results:
        raw_text = str(text).strip()
        cleaned = clean_easyocr_title_text(raw_text)
        if cleaned == "":
            continue

        points = np.array(bbox)
        x_min = float(points[:, 0].min())
        x_max = float(points[:, 0].max())
        y_min = float(points[:, 1].min())
        y_max = float(points[:, 1].max())

        # Coordinates are already in the top-band crop (= full-page top).
        box_width = x_max - x_min
        box_height = y_max - y_min
        y_center_ratio = ((y_min + y_max) / 2.0) / max(page_height, 1)
        x_center_ratio = ((x_min + x_max) / 2.0) / max(page_width, 1)

        lines.append({
            "raw": raw_text,
            "text": cleaned,
            "confidence": float(conf),
            "height": box_height,
            "width": box_width,
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
            "y_center_ratio": y_center_ratio,
            "x_center_ratio": x_center_ratio,
            "word_count": len(cleaned.split()),
        })

    logger.debug(
        "OCR raw lines from newspaper masthead area: %s",
        [(item["raw"], round(item["y_center_ratio"], 3), round(item["confidence"], 3))
         for item in lines],
    )
    return lines, page_width, page_height


def easyocr_extract_newspaper_masthead(all_results=None, image_path=None):
    """
    Pick newspaper name / masthead from top-region OCR results.
    Does not try to read article headlines or categories.

    Prefer:
    - top 25–30% band only
    - short lines (<= 4 words)
    - top / top-left larger text
    """
    if image_path is not None:
        lines, page_width, page_height = _ocr_newspaper_masthead_region(image_path)
    else:
        lines = all_results or []
        page_width = 1.0
        page_height = 1.0
        # Legacy crop-relative results: treat y as already top-band.
        for item in lines:
            item.setdefault("raw", item.get("text", ""))
            item.setdefault(
                "y_center_ratio",
                (item.get("y_min", 0) + item.get("y_max", 0)) / 2.0 / 1000.0,
            )
            item.setdefault(
                "x_center_ratio",
                (item.get("x_min", 0) + item.get("x_max", 0)) / 2.0 / 1000.0,
            )
            item.setdefault("word_count", len(str(item.get("text", "")).split()))

    if len(lines) == 0:
        logger.debug("candidate masthead text: []")
        logger.debug("final newspaper title: %s", NEWSPAPER_NAME_FALLBACK)
        return NEWSPAPER_NAME_FALLBACK

    # Keep only boxes in the upper masthead band of the page.
    top_band = [
        item for item in lines
        if item.get("y_center_ratio", 1.0) <= 0.28
    ]
    if not top_band:
        top_band = lines

    # Prefer the very top strip first (true masthead).
    strict_top = [
        item for item in top_band
        if item.get("y_center_ratio", 1.0) <= 0.16
    ]
    search_pool = strict_top if strict_top else top_band

    candidates = []
    for item in search_pool:
        text = item["text"]
        if item.get("word_count", 99) > 4:
            continue
        if is_bad_newspaper_masthead(text):
            continue

        # Soft preference: top-left / top band.
        top_score = 1.0 - float(item.get("y_center_ratio", 1.0))
        left_score = 1.0 - float(item.get("x_center_ratio", 0.5))
        size_score = float(item.get("height", 0.0))
        conf_score = float(item.get("confidence", 0.0))
        score = (size_score * 2.5) + (top_score * 40) + (left_score * 12) + (conf_score * 20)

        candidate = dict(item)
        candidate["masthead_score"] = score
        candidates.append(candidate)

    logger.debug(
        "candidate masthead text: %s",
        [(c["text"], round(c["masthead_score"], 2), round(c.get("y_center_ratio", 0), 3))
         for c in candidates],
    )

    # 1) Exact / substring known-name match on each short candidate.
    for item in sorted(candidates, key=lambda x: x["masthead_score"], reverse=True):
        known = detect_known_easyocr_title(item["text"], "Newspaper")
        if known is not None:
            logger.debug("final newspaper title: %s", known)
            return known

    # 2) Token match across top short lines (split glued OCR).
    token_bag = []
    for item in sorted(search_pool, key=lambda x: x.get("y_min", 0)):
        if item.get("word_count", 99) > 4:
            continue
        token_bag.extend(split_glued_masthead_text(item["text"]))
        token_bag.extend(item["text"].split())

    # Deduplicate tokens while preserving order
    seen = set()
    unique_tokens = []
    for token in token_bag:
        if token not in seen:
            seen.add(token)
            unique_tokens.append(token)

    known = match_newspaper_from_tokens(unique_tokens)
    if known is not None:
        logger.debug("final newspaper title: %s", known)
        return known

    # 3) Join only the top 2 short candidates by vertical order (not headlines).
    short_top = sorted(
        [c for c in candidates if c.get("word_count", 99) <= 3],
        key=lambda x: (x.get("y_min", 0), x.get("x_min", 0)),
    )[:2]
    if short_top:
        joined = " ".join(item["text"] for item in short_top)
        logger.debug("joined short masthead candidates: %r", joined)
        known = detect_known_easyocr_title(joined, "Newspaper")
        if known is not None:
            logger.debug("final newspaper title: %s", known)
            return known
        known = match_newspaper_from_tokens(split_glued_masthead_text(joined) + joined.split())
        if known is not None:
            logger.debug("final newspaper title: %s", known)
            return known

    # 4) Best single short candidate after cleaning glued leftover words.
    if candidates:
        best = sorted(candidates, key=lambda x: x["masthead_score"], reverse=True)[0]
        pieces = split_glued_masthead_text(best["text"])
        known = match_newspaper_from_tokens(pieces)
        if known is not None:
            logger.debug("final newspaper title: %s", known)
            return known

        # Keep only lexicon masthead words from the best box.
        lexicon = set(_masthead_lexicon_words())
        cleaned_words = [w for w in pieces if w in lexicon][:4]
        if cleaned_words:
            cleaned_title = nice_title_case(" ".join(cleaned_words))
            logger.debug("final newspaper title: %s", cleaned_title)
            return cleaned_title

    logger.debug("final newspaper title: %s", NEWSPAPER_NAME_FALLBACK)
    return NEWSPAPER_NAME_FALLBACK


def create_device_message(document_type, title=None):
    """
    Create final message for user.
    """

    pretty_names = {
        "Magazine": "Magazine",
        "Newspaper": "Newspaper",
        "Novel": "Novel",
        "Printed_Letter": "Printed Letter",
        "Report": "Report"
    }

    readable_name = pretty_names.get(document_type, document_type)

    if document_type in ["Novel", "Magazine"]:
        if title is None or title == "Title not clearly detected":
            return (
                f"This is a {readable_name}. "
                f"Title was not clearly detected. "
                f"Please turn the next page."
            )

        return (
            f"This is a {readable_name}. "
            f"The title is {title}. "
            f"Please turn the next page."
        )

    return f"This is a {readable_name}."


print("EasyOCR title reader loaded successfully")
print("EasyOCR GPU: False")
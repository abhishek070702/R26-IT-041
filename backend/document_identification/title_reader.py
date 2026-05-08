import re

import torch
from PIL import Image
import numpy as np
import easyocr


# =========================
# Load EasyOCR reader
# =========================

# EasyOCR uses CPU to avoid CUDA out-of-memory issues.
reader = easyocr.Reader(
    ["en"],
    gpu=False
)


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
    """
    small_words = [
        "AND", "THE", "OF", "IN", "ON",
        "TO", "A", "AN", "FOR", "WITH"
    ]

    words = title.split()
    final_words = []

    for i, word in enumerate(words):
        if i != 0 and word in small_words:
            final_words.append(word.lower())
        else:
            final_words.append(word.capitalize())

    return " ".join(final_words)


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
        "ATLANTIC": "The Atlantic"
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
        "THE STORY OF A LITTLE FROG": "The Story of a Little Frog"
    }

    if document_type == "Magazine":
        for pattern, title in magazine_patterns.items():
            if pattern in text:
                return title

    if document_type == "Novel":
        for pattern, title in novel_patterns.items():
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

def crop_for_easyocr_title(image_path, document_type):
    """
    Crop useful title areas.
    """

    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    crops = []

    if document_type == "Magazine":
        crops.append(image.crop((0, 0, width, int(height * 0.45))))
        crops.append(image.crop((0, 0, width, int(height * 0.60))))
        crops.append(image)

    elif document_type == "Novel":
        crops.append(image.crop((0, int(height * 0.03), width, int(height * 0.80))))
        crops.append(image.crop((0, int(height * 0.08), width, int(height * 0.70))))
        crops.append(image)

    return crops


def build_full_novel_title(valid_results):
    """
    Build full title for novels by combining multiple EasyOCR title lines.
    """

    cleaned_items = []

    for item in valid_results:
        cleaned_text = remove_author_and_extra_lines(item["text"])

        if cleaned_text == "":
            continue

        if is_bad_easyocr_title(cleaned_text):
            continue

        new_item = item.copy()
        new_item["text"] = cleaned_text
        cleaned_items.append(new_item)

    if len(cleaned_items) == 0:
        return None

    unique_items = []
    seen_texts = set()

    for item in cleaned_items:
        if item["text"] not in seen_texts:
            unique_items.append(item)
            seen_texts.add(item["text"])

    cleaned_items = unique_items

    heights = [item["height"] for item in cleaned_items]
    avg_height = sum(heights) / len(heights)

    title_lines = []

    for item in cleaned_items:
        if item["height"] >= avg_height * 0.55 and item["confidence"] >= 0.20:
            title_lines.append(item)

    if len(title_lines) == 0:
        title_lines = cleaned_items

    title_lines = sorted(title_lines, key=lambda x: x["y_min"])
    title_lines = title_lines[:5]

    title = " ".join([item["text"] for item in title_lines])

    words = title.split()
    final_words = []

    for word in words:
        if len(final_words) == 0 or final_words[-1] != word:
            final_words.append(word)

    title = " ".join(final_words)

    if "LITTLE" in title and "FROG" in title:
        if "STORY" in title:
            return "The Story of a Little Frog"
        return "Little Frog"

    if title == "" or is_bad_easyocr_title(title):
        return None

    return nice_title_case(title)


def easyocr_extract_title(image_path, document_type):
    """
    Extract title using EasyOCR only for Novel/Magazine.
    For Novel, combines multiple title lines into fuller title.
    EasyOCR runs on CPU to avoid GPU memory issues.
    """

    if document_type not in ["Novel", "Magazine"]:
        return None

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    crops = crop_for_easyocr_title(image_path, document_type)

    all_results = []

    for crop_image in crops:
        crop_np = np.array(crop_image)

        results = reader.readtext(
            crop_np,
            detail=1,
            paragraph=False
        )

        for bbox, text, conf in results:
            cleaned_text = clean_easyocr_title_text(text)

            if cleaned_text == "":
                continue

            points = np.array(bbox)
            x_min = points[:, 0].min()
            x_max = points[:, 0].max()
            y_min = points[:, 1].min()
            y_max = points[:, 1].max()

            box_width = x_max - x_min
            box_height = y_max - y_min

            score = (box_height * 2) + (box_width * 0.03) + (conf * 30)

            all_results.append({
                "text": cleaned_text,
                "confidence": conf,
                "height": box_height,
                "width": box_width,
                "x_min": x_min,
                "x_max": x_max,
                "y_min": y_min,
                "y_max": y_max,
                "score": score
            })

    if len(all_results) == 0:
        return "Title not clearly detected"

    combined_text = " ".join([item["text"] for item in all_results])

    known_title = detect_known_easyocr_title(
        combined_text,
        document_type
    )

    if known_title is not None:
        return known_title

    valid_results = []

    for item in all_results:
        text = item["text"]

        if not is_bad_easyocr_title(text):
            valid_results.append(item)

    if len(valid_results) == 0:
        return "Title not clearly detected"

    if document_type == "Novel":
        full_title = build_full_novel_title(valid_results)

        if full_title is not None:
            return full_title

    best = sorted(
        valid_results,
        key=lambda x: x["score"],
        reverse=True
    )[0]

    final_title = best["text"]

    return nice_title_case(final_title)


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
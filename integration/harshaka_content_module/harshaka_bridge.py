import json
import os
from pathlib import Path
from typing import Any, Dict, List

import requests


HARSHAAKA_ANALYZE_URL = os.getenv(
    "HARSHAAKA_ANALYZE_URL",
    "http://127.0.0.1:8001/analyze"
)

HARSHAAKA_GENERATE_URL = os.getenv(
    "HARSHAAKA_GENERATE_URL",
    "http://127.0.0.1:8001/generate-output"
)


def analyze_page_with_harshaka(
    image_path: str,
    document_type: str,
    timeout: int = 180,
) -> Dict[str, Any]:
    """
    Send captured page image + Abhishek document_type to Harshaka module.
    document_type examples: Novel, Newspaper
    """
    image = Path(image_path)

    if not image.exists():
        return {
            "status": "failed",
            "error": f"Image not found: {image}",
            "document_type": document_type,
        }

    with open(image, "rb") as image_file:
        response = requests.post(
            HARSHAAKA_ANALYZE_URL,
            data={"document_type": document_type},
            files={"file": (image.name, image_file, "image/jpeg")},
            timeout=timeout,
        )

    response.raise_for_status()

    result = response.json()
    result["status"] = result.get("status", "success")
    result["document_type"] = result.get("document_type", document_type)
    result["image_path"] = str(image)

    return result


def get_available_categories(analysis_result: Dict[str, Any]) -> List[str]:
    """
    Extract category names from Harshaka analysis result.
    Works with categories, category_counts, articles, or results.
    """
    categories = []

    direct_categories = analysis_result.get("categories") or analysis_result.get("detected_categories")
    if isinstance(direct_categories, list):
        categories.extend(str(item) for item in direct_categories if str(item).strip())

    category_counts = analysis_result.get("category_counts")
    if isinstance(category_counts, dict):
        categories.extend(str(key) for key in category_counts.keys() if str(key).strip())

    for key in ["articles", "results", "local_results"]:
        items = analysis_result.get(key)
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict):
                    category = item.get("category")
                    if category:
                        categories.append(str(category))

    unique_categories = []
    seen = set()

    for category in categories:
        normalized = category.strip()
        key = normalized.lower()

        if normalized and key not in seen:
            unique_categories.append(normalized)
            seen.add(key)

    return unique_categories


def generate_selected_harshaka_output(
    analysis_result: Dict[str, Any],
    selected_category: str,
    depth: str,
    timeout: int = 180,
) -> Dict[str, Any]:
    """
    Ask Harshaka module to generate final summary/full text for selected category.
    depth must be: summary or full
    """
    payload = {
        "analysis": analysis_result.get("analysis", analysis_result),
        "selected_category": selected_category,
        "depth": depth,
    }

    response = requests.post(
        HARSHAAKA_GENERATE_URL,
        json=payload,
        timeout=timeout,
    )

    response.raise_for_status()

    result = response.json()
    result["status"] = result.get("status", "success")
    result["selected_category"] = selected_category
    result["content_depth"] = depth

    return result


def get_text_for_tts(harshaka_result: Dict[str, Any]) -> str:
    """
    Extract final text that Rashmi should speak.
    """
    if not isinstance(harshaka_result, dict):
        return str(harshaka_result)

    text = (
        harshaka_result.get("text")
        or harshaka_result.get("final_output_text")
        or harshaka_result.get("next_module_payload")
        or ""
    )

    if isinstance(text, dict):
        text = text.get("text", "")

    return str(text).strip()


def run_laptop_test(image_path: str, document_type: str):
    """
    Laptop-only test.
    Later main_controller will call these functions automatically.
    """
    print("Sending image to Harshaka...")
    analysis_result = analyze_page_with_harshaka(image_path, document_type)

    print("\nAnalysis preview:")
    print(json.dumps({
        "status": analysis_result.get("status"),
        "document_type": analysis_result.get("document_type"),
        "categories": get_available_categories(analysis_result),
        "extracted_text_preview": (analysis_result.get("extracted_text") or "")[:500],
    }, indent=4))

    categories = get_available_categories(analysis_result)

    if categories:
        print("\nDetected categories:")
        for index, category in enumerate(categories, start=1):
            print(f"{index}. {category}")

        selected_category = input("Type category name or number: ").strip()

        if selected_category.isdigit():
            idx = int(selected_category) - 1
            if 0 <= idx < len(categories):
                selected_category = categories[idx]

        if not selected_category:
            selected_category = categories[0]
    else:
        selected_category = "Story" if "novel" in document_type.lower() else "General Text"

    depth = input("Type summary or full: ").strip().lower()
    if depth not in ["summary", "full"]:
        depth = "summary"

    final_result = generate_selected_harshaka_output(
        analysis_result=analysis_result,
        selected_category=selected_category,
        depth=depth,
    )

    print("\nGenerated Harshaka output:")
    print(json.dumps(final_result, indent=4))

    print("\nText Rashmi should speak:")
    print(get_text_for_tts(final_result))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--document-type", default="Novel")

    args = parser.parse_args()

    run_laptop_test(args.image, args.document_type)

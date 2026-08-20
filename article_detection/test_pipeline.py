"""
Quick local test for the V7 pipeline.

Usage (from the backend folder):
  python test_pipeline.py path\\to\\page.jpg --type Newspaper
  python test_pipeline.py path\\to\\page.jpg --type Textbook --depth summary
  python test_pipeline.py path\\to\\page.jpg --type Magazine --category Sports --depth full
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Test analyze_content + generate_selected_output on one image."
    )
    parser.add_argument("image", help="Path to a document image (jpg/png/webp).")
    parser.add_argument(
        "--type",
        default="Newspaper",
        help="Newspaper, Magazine, Novel/Story, Textbook, or General Document",
    )
    parser.add_argument("--category", default=None, help="Optional category to generate.")
    parser.add_argument(
        "--depth",
        default="full",
        choices=("summary", "full"),
        help="Output depth for generate_selected_output.",
    )
    parser.add_argument(
        "--save-json",
        default="",
        help="Optional path to write the analysis JSON.",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.is_file():
        print(f"Image not found: {image_path.resolve()}", file=sys.stderr)
        return 1

    from pipeline import analyze_content, generate_selected_output

    print("Analyzing:", image_path.resolve())
    print("Document type:", args.type)

    analysis = analyze_content(str(image_path), args.type, debug=True)

    print("\n========== ANALYSIS ==========")
    print("Pipeline:", analysis.get("pipeline_version", "(not set)"))
    print("Document type:", analysis.get("document_type"))
    print("Mode:", analysis.get("mode"))
    print("Categories:", analysis.get("categories") or analysis.get("detected_categories"))
    print("Category counts:", analysis.get("category_counts"))
    print("Items:", len(analysis.get("results") or []))
    print("Extracted chars:", len(analysis.get("full_page_text") or ""))
    warnings = analysis.get("warnings") or []
    if warnings:
        print("Warnings:")
        for warning in warnings[:12]:
            print(" -", warning)

    print("\n========== ITEMS ==========")
    for item in analysis.get("results") or []:
        text = (item.get("full_text") or "").strip()
        preview = text[:180].replace("\n", " ")
        print(
            f"- #{item.get('article_number')}  "
            f"{item.get('title')!r}  "
            f"[{item.get('category')}]  "
            f"words={len(text.split())}  "
            f"ocr={item.get('ocr_quality')}"
        )
        if preview:
            print("  ", preview, "...")

    if args.save_json:
        out_path = Path(args.save_json)
        out_path.write_text(json.dumps(analysis, default=str, indent=2), encoding="utf-8")
        print("\nSaved analysis JSON:", out_path.resolve())

    categories = analysis.get("categories") or analysis.get("detected_categories") or []
    selected = args.category
    if selected is None and len(categories) == 1:
        selected = categories[0]
        print("\nSingle category auto-selected:", selected)
    elif selected is None and len(categories) > 1:
        print("\nMultiple categories. Re-run with --category one of:", categories)
        return 0

    if not selected:
        print("\nNo category to generate output.")
        return 0

    output = generate_selected_output(analysis, selected, args.depth)
    print("\n========== GENERATED OUTPUT ==========")
    print("Status:", output.get("status"))
    if output.get("status") != "READY":
        print("Message:", output.get("message"))
        return 1

    print("Category:", output.get("selected_category"))
    print("Depth:", output.get("content_depth") or args.depth)
    print("\n--- final_output_text ---\n")
    print(output.get("final_output_text") or "")
    payload = output.get("next_module_payload") or {}
    payload_text = payload.get("text") if isinstance(payload, dict) else payload
    print("\n--- next_module_payload text ---\n")
    print(payload_text or "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

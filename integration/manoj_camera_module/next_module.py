import json
import os
import sys
from pathlib import Path

import requests


CURRENT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = CURRENT_DIR / "integration_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

LATEST_IMAGE_PATH_FILE = CURRENT_DIR / "latest_captured_image.txt"

ABHISHEK_API_URL = os.getenv(
    "ABHISHEK_API_URL",
    "http://127.0.0.1:8000/abhishek/analyze"
)

TARGET_MODULE = os.getenv("NEXT_MODULE_TARGET", "abhishek").lower()


def save_latest_image_path(image_path: Path):
    LATEST_IMAGE_PATH_FILE.write_text(str(image_path), encoding="utf-8")
    print("Latest captured image path saved:")
    print(LATEST_IMAGE_PATH_FILE)


def save_json(result, output_path: Path):
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=4)

    print("Result saved to:")
    print(output_path)


def send_to_abhishek(image_path: Path):
    print("Sending image to Abhishek module...")
    print("URL:", ABHISHEK_API_URL)

    with open(image_path, "rb") as image_file:
        response = requests.post(
            ABHISHEK_API_URL,
            files={
                "file": (
                    image_path.name,
                    image_file,
                    "image/jpeg"
                )
            },
            timeout=180,
        )

    response.raise_for_status()
    result = response.json()

    result["captured_image_path"] = str(image_path)
    result["target_module"] = "abhishek"

    output_path = OUTPUT_DIR / "last_abhishek_result.json"
    save_json(result, output_path)

    print("Abhishek result:")
    print(json.dumps(result, indent=4))

    return result


def main():
    if len(sys.argv) < 2:
        print("No image path received.")
        return

    image_path = Path(sys.argv[1])

    if not image_path.exists():
        print("Image not found:", image_path)
        return

    print("Next module received image:")
    print(image_path)

    save_latest_image_path(image_path)

    if TARGET_MODULE in ["capture_only", "capture", "none"]:
        print("Capture-only mode. Image saved for main controller.")
        return

    if TARGET_MODULE in ["abhishek", "cover", "first"]:
        try:
            send_to_abhishek(image_path)
        except Exception as error:
            print("Failed to send image to Abhishek:")
            print(error)

            save_json(
                {
                    "status": "failed",
                    "target_module": "abhishek",
                    "captured_image_path": str(image_path),
                    "error": str(error),
                },
                OUTPUT_DIR / "last_abhishek_result.json"
            )
        return

    print("Unknown NEXT_MODULE_TARGET:", TARGET_MODULE)


if __name__ == "__main__":
    main()

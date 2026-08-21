import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

RASHMI_DIR = CURRENT_DIR / "rashmi_tts_module"

# Support both repo structures:
# 1. integration/manoj_camera_module
# 2. root/manoj_camera_module on Raspberry Pi old setup
MANOJ_DIR = CURRENT_DIR / "manoj_camera_module"
if not MANOJ_DIR.exists():
    MANOJ_DIR = PROJECT_ROOT / "manoj_camera_module"

CAMERA_SCRIPT = MANOJ_DIR / "final_yolo_opencv_guidance.py"
OUTPUT_DIR = MANOJ_DIR / "integration_outputs"

ABHISHEK_RESULT_PATH = OUTPUT_DIR / "last_abhishek_result.json"
LATEST_IMAGE_PATH_FILE = MANOJ_DIR / "latest_captured_image.txt"

sys.path.insert(0, str(CURRENT_DIR))

from rashmi_tts_module.rashmi_bridge import (
    ask_by_voice,
    ask_category_by_voice,
    ask_ready_for_next_page,
    ask_summary_or_full,
    get_user_preferences,
    save_current_preferences,
    speak,
    speak_abhishek_result,
    speak_harshaka_result,
)

from harshaka_content_module.harshaka_bridge import (
    analyze_page_with_harshaka,
    generate_selected_harshaka_output,
    get_available_categories,
)


def read_json_file(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def get_latest_captured_image_path() -> Optional[Path]:
    if not LATEST_IMAGE_PATH_FILE.exists():
        print("Latest image path file not found:", LATEST_IMAGE_PATH_FILE)
        return None

    image_path = Path(LATEST_IMAGE_PATH_FILE.read_text(encoding="utf-8").strip())

    if not image_path.exists():
        print("Latest captured image not found:", image_path)
        return None

    return image_path


def run_camera_once(preferences: Dict, target_module: str) -> bool:
    """
    Start Manoj camera guidance once.
    target_module:
      - abhishek: next_module sends captured image to Abhishek backend
      - capture_only: next_module only saves captured image path
    """
    if not CAMERA_SCRIPT.exists():
        print("Camera script not found:", CAMERA_SCRIPT)
        speak("Camera guidance module was not found.", preferences)
        return False

    OUTPUT_DIR.mkdir(exist_ok=True)

    env = os.environ.copy()
    env["NEXT_MODULE_TARGET"] = target_module

    if target_module == "abhishek":
        speak("Opening camera guidance to capture the first page.", preferences)
    else:
        speak("Opening camera guidance to capture the page.", preferences)

    print("Starting camera script:", CAMERA_SCRIPT)
    print("Target module:", target_module)

    subprocess.run(
        [sys.executable, str(CAMERA_SCRIPT)],
        cwd=str(MANOJ_DIR),
        env=env,
        check=False,
    )

    print("Camera module closed.")
    return True


def ask_continue_or_stop(preferences: Dict) -> str:
    return ask_by_voice(
        question="Say ready to continue to the next page, or say stop to finish.",
        valid_answers=["ready", "stop"],
        default_answer="ready",
        preferences=preferences,
        attempts=3,
        max_seconds=8,
    )


def select_harshaka_category(
    analysis_result: Dict,
    document_type: str,
    preferences: Dict,
) -> str:
    categories = get_available_categories(analysis_result)

    if categories:
        if len(categories) == 1:
            selected_category = categories[0]
            speak(f"Single category detected: {selected_category}.", preferences)
            return selected_category

        return ask_category_by_voice(categories, preferences)

    if "novel" in document_type.lower() or "story" in document_type.lower():
        speak("Single category detected: Story.", preferences)
        return "Story"

    speak("No clear category was detected. I will use General Text.", preferences)
    return "General Text"


def run_harshaka_reading_for_image(
    image_path: Path,
    document_type: str,
    preferences: Dict,
):
    """
    Send image + document type to Harshaka.
    Ask category if needed.
    Ask summary/full.
    Speak final output using Rashmi TTS.
    """
    try:
        speak("Sending page to content reading module.", preferences)

        analysis_result = analyze_page_with_harshaka(
            image_path=str(image_path),
            document_type=document_type,
        )

        print("Harshaka analysis result preview:")
        print(json.dumps({
            "status": analysis_result.get("status"),
            "document_type": analysis_result.get("document_type"),
            "categories": get_available_categories(analysis_result),
            "extracted_text_preview": (analysis_result.get("extracted_text") or "")[:500],
        }, indent=4))

        selected_category = select_harshaka_category(
            analysis_result=analysis_result,
            document_type=document_type,
            preferences=preferences,
        )

        depth = ask_summary_or_full(preferences)

        speak(
            f"Preparing {depth} for {selected_category}.",
            preferences,
        )

        final_result = generate_selected_harshaka_output(
            analysis_result=analysis_result,
            selected_category=selected_category,
            depth=depth,
        )

        print("Harshaka final result:")
        print(json.dumps(final_result, indent=4))

        speak_harshaka_result(final_result, preferences)

    except Exception as error:
        print("Harshaka reading failed:")
        print(error)
        speak("Content reading module failed for this page.", preferences)


def run_novel_loop(preferences: Dict):
    """
    Novel flow:
    Cover page was already processed by Abhishek.
    Loop starts from inside pages.
    """
    speak("Please turn to the next page of the novel.", preferences)

    while True:
        choice = ask_continue_or_stop(preferences)

        if choice == "stop":
            speak("Novel reading stopped.", preferences)
            break

        ok = run_camera_once(preferences, target_module="capture_only")
        if not ok:
            break

        image_path = get_latest_captured_image_path()
        if image_path is None:
            speak("Captured page image was not found.", preferences)
            continue

        run_harshaka_reading_for_image(
            image_path=image_path,
            document_type="Novel",
            preferences=preferences,
        )

        speak("Page reading completed.", preferences)


def run_newspaper_loop(first_image_path: Optional[Path], preferences: Dict):
    """
    Newspaper flow:
    First captured image is used only for Abhishek newspaper-name detection.
    It is NOT sent to Harshaka, because full newspaper captures often make text too small.
    Harshaka needs a second close capture of one clear article area.
    """
    if first_image_path is not None:
        print("First newspaper image was used for Abhishek only:", first_image_path)

    speak(
        "Newspaper name detection completed. Now move the camera closer to one article area.",
        preferences,
    )
    speak(
        "Make sure the article headline and body letters are clear before capture.",
        preferences,
    )

    while True:
        choice = ask_continue_or_stop(preferences)

        if choice == "stop":
            speak("Newspaper reading stopped.", preferences)
            break

        ok = run_camera_once(preferences, target_module="capture_only")
        if not ok:
            break

        image_path = get_latest_captured_image_path()
        if image_path is None:
            speak("Captured newspaper article image was not found.", preferences)
            continue

        run_harshaka_reading_for_image(
            image_path=image_path,
            document_type="Newspaper",
            preferences=preferences,
        )

        speak("Newspaper article reading completed.", preferences)
        speak("Move to another article and say ready, or say stop to finish.", preferences)


def main():
    print("==========================================")
    print("SMART READING ASSISTANT FULL CONTROLLER")
    print("Rashmi + Manoj + Abhishek + Harshaka")
    print("==========================================")

    preferences = get_user_preferences()
    save_current_preferences(preferences)

    print("\nCurrent user preferences:")
    print(json.dumps(preferences, indent=4))

    # First capture always goes to Abhishek.
    ok = run_camera_once(preferences, target_module="abhishek")

    if not ok:
        return

    abhishek_result = read_json_file(ABHISHEK_RESULT_PATH)

    if not abhishek_result:
        speak("Abhishek analysis result was not found.", preferences)
        return

    print("Abhishek result:")
    print(json.dumps(abhishek_result, indent=4))

    speak_abhishek_result(abhishek_result, preferences)

    document_type = str(abhishek_result.get("document_type", "")).strip()
    first_image_path_text = abhishek_result.get("captured_image_path")
    first_image_path = Path(first_image_path_text) if first_image_path_text else get_latest_captured_image_path()

    document_type_lower = document_type.lower()

    if "novel" in document_type_lower:
        run_novel_loop(preferences)

    elif "newspaper" in document_type_lower:
        run_newspaper_loop(first_image_path, preferences)

    else:
        speak(
            "This document type is detected, but full content reading is currently supported for novels and newspapers only.",
            preferences,
        )

    speak("Smart Reading Assistant session completed.", preferences)


if __name__ == "__main__":
    main()

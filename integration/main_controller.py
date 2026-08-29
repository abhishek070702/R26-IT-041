import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import requests


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

RASHMI_DIR = CURRENT_DIR / "rashmi_tts_module"


def configure_rfid_for_pi():
    """
    Full system uses a PN532 tap on the Raspberry Pi.
    Keyboard card entry stays for Windows / RFID_ALLOW_KEYBOARD=1 laptop tests.
    """
    allow_keyboard = os.getenv("RFID_ALLOW_KEYBOARD", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    if sys.platform.startswith("win") or allow_keyboard:
        print("RFID: keyboard entry is allowed (PN532 tap is not required).")
        return

    os.environ.setdefault("PN532_INTERFACE", "i2c")
    os.environ.setdefault("PN532_I2C_BUS", "1")
    print(
        "RFID: tap a card on the PN532 "
        f"(I2C bus {os.environ['PN532_I2C_BUS']})."
    )


def resolve_camera_paths():
    """
    Run final_yolo_opencv_guidance.py from the folder that has best.pt.
    On the Pi that is repo-root manoj_camera_module, not integration/.
    """
    candidates = [
        PROJECT_ROOT / "manoj_camera_module" / "final_yolo_opencv_guidance.py",
        CURRENT_DIR / "manoj_camera_module" / "final_yolo_opencv_guidance.py",
    ]
    with_model = []
    with_script = []
    for script in candidates:
        if not script.exists():
            continue
        with_script.append(script)
        if (script.parent / "best.pt").exists():
            with_model.append(script)

    chosen = with_model or with_script
    if chosen:
        script = chosen[0]
        return script.parent, script

    fallback_dir = PROJECT_ROOT / "manoj_camera_module"
    if not fallback_dir.exists():
        fallback_dir = CURRENT_DIR / "manoj_camera_module"
    return fallback_dir, fallback_dir / "final_yolo_opencv_guidance.py"


MANOJ_DIR, CAMERA_SCRIPT = resolve_camera_paths()
OUTPUT_DIR = MANOJ_DIR / "integration_outputs"

ABHISHEK_RESULT_PATH = OUTPUT_DIR / "last_abhishek_result.json"
LATEST_IMAGE_PATH_FILE = MANOJ_DIR / "latest_captured_image.txt"

ABHISHEK_ANALYZE_URL = os.getenv(
    "ABHISHEK_API_URL",
    "http://127.0.0.1:8000/abhishek/analyze",
)
ABHISHEK_DESCRIBE_URL = os.getenv(
    "ABHISHEK_DESCRIBE_URL",
    ABHISHEK_ANALYZE_URL.replace("/abhishek/analyze", "/abhishek/describe-image"),
)

WEAK_TITLE_MARKERS = (
    "not clearly",
    "handled by",
    "unknown",
)
WEAK_DESCRIPTION_MARKERS = (
    "no important visual",
    "no visual image",
    "not clearly",
    "could not describe",
    "description failed",
    "handled by",
)

DOCUMENT_TYPE_ALIASES = {
    "novel": "Novel",
    "magazine": "Magazine",
    "newspaper": "Newspaper",
    "report": "Report",
    "printed_letter": "Printed_Letter",
    "printed letter": "Printed_Letter",
    "printedletter": "Printed_Letter",
    "letter": "Printed_Letter",
}

sys.path.insert(0, str(CURRENT_DIR))

from rashmi_tts_module.rashmi_bridge import (
    ask_by_voice,
    ask_category_by_voice,
    ask_summary_or_full,
    get_user_preferences,
    save_current_preferences,
    speak,
    speak_harshaka_result,
    speak_reading_text,
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


def resolve_image_path(abhishek_result: Optional[Dict] = None) -> Optional[Path]:
    if abhishek_result:
        captured = str(abhishek_result.get("captured_image_path") or "").strip()
        if captured:
            captured_path = Path(captured)
            if captured_path.exists():
                return captured_path

    return get_latest_captured_image_path()


def normalize_document_type(document_type: Optional[str]) -> str:
    raw = str(document_type or "").strip().lower().replace("-", "_")
    raw = " ".join(raw.split())

    if raw in DOCUMENT_TYPE_ALIASES:
        return DOCUMENT_TYPE_ALIASES[raw]

    compact = raw.replace(" ", "_")
    if compact in DOCUMENT_TYPE_ALIASES:
        return DOCUMENT_TYPE_ALIASES[compact]

    for key, value in DOCUMENT_TYPE_ALIASES.items():
        if key in raw:
            return value

    return ""


def is_usable_title(title: Optional[str]) -> bool:
    text = str(title or "").strip()
    if not text:
        return False

    lowered = text.lower()
    return not any(marker in lowered for marker in WEAK_TITLE_MARKERS)


def is_meaningful_image_description(
    description: Optional[str],
    has_image: Optional[bool] = None,
) -> bool:
    if has_image is False:
        return False

    text = str(description or "").strip()
    if len(text) < 8:
        return False

    lowered = text.lower()
    return not any(marker in lowered for marker in WEAK_DESCRIPTION_MARKERS)


def first_cover_description(abhishek_result: Dict) -> str:
    descriptions = abhishek_result.get("image_descriptions") or []
    if isinstance(descriptions, str):
        descriptions = [descriptions]

    if not isinstance(descriptions, list):
        return ""

    for item in descriptions:
        text = str(item or "").strip()
        if is_meaningful_image_description(text):
            return text

    return ""


def speak_first_page(abhishek_result: Dict, document_type: str, preferences: Dict):
    title = abhishek_result.get("title")
    cover_description = first_cover_description(abhishek_result)

    if document_type == "Novel":
        parts = ["This is a Novel."]
        if is_usable_title(title):
            parts.append(f"The title is {title}.")
        if cover_description:
            parts.append(cover_description)
        speak_reading_text(" ".join(parts), preferences)
        return

    if document_type == "Magazine":
        parts = ["This is a Magazine."]
        if is_usable_title(title):
            parts.append(f"The title is {title}.")
        if cover_description:
            parts.append(cover_description)
        speak_reading_text(" ".join(parts), preferences)
        return

    if document_type == "Newspaper":
        parts = ["This is a Newspaper."]
        if is_usable_title(title):
            parts.append(f"The newspaper name is {title}.")
        speak_reading_text(" ".join(parts), preferences)
        return

    if document_type == "Report":
        speak("This is a Report.", preferences)
        return

    if document_type == "Printed_Letter":
        speak("This is a Printed Letter.", preferences)
        return

    speak("The document type could not be identified clearly.", preferences)


def describe_page_visual(
    image_path: Path,
    context: str,
    preferences: Dict,
):
    """
    Call Abhishek /abhishek/describe-image and speak only a meaningful caption.
    """
    if image_path is None or not Path(image_path).exists():
        return

    try:
        with open(image_path, "rb") as image_file:
            response = requests.post(
                ABHISHEK_DESCRIBE_URL,
                files={
                    "file": (Path(image_path).name, image_file, "image/jpeg"),
                },
                data={"context": context or "general"},
                timeout=180,
            )
        response.raise_for_status()
        result = response.json()
    except Exception as error:
        print("Abhishek describe-image failed:")
        print(error)
        return

    description = (
        result.get("image_description")
        or result.get("description")
        or ""
    )
    has_image = result.get("has_image")

    if is_meaningful_image_description(description, has_image=has_image):
        speak_reading_text(str(description).strip(), preferences)


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
    env["RASHMI_VOICE"] = str(preferences.get("voice") or preferences.get("voice_type") or "female")
    env["RASHMI_PACE"] = str(preferences.get("pace") or "normal")
    env["RASHMI_TONE"] = str(preferences.get("tone") or "friendly")

    if target_module == "abhishek":
        speak("Opening camera guidance to capture the first page.", preferences)
    else:
        speak("Opening camera guidance to capture the page.", preferences)

    speak("Please wait. The camera model is loading.", preferences)

    print("Starting camera script:", CAMERA_SCRIPT)
    print("Target module:", target_module)

    completed = subprocess.run(
        [sys.executable, str(CAMERA_SCRIPT)],
        cwd=str(MANOJ_DIR),
        env=env,
        check=False,
    )

    print("Camera module closed. Exit code:", completed.returncode)
    if completed.returncode != 0:
        speak("Camera guidance failed.", preferences)
        return False

    return True


def ask_continue_or_stop(preferences: Dict, question: str) -> str:
    return ask_by_voice(
        question=question,
        valid_answers=["next", "stop"],
        default_answer="next",
        preferences=preferences,
        attempts=3,
        max_seconds=8,
    )


def select_harshaka_category(
    analysis_result: Dict,
    document_type: str,
    preferences: Dict,
) -> Optional[str]:
    categories: List[str] = get_available_categories(analysis_result)

    if not categories:
        speak("No readable content was found for this page.", preferences)
        return None

    if len(categories) == 1:
        selected_category = categories[0]
        speak(f"Single category detected: {selected_category}.", preferences)
        return selected_category

    selected_category = ask_category_by_voice(categories, preferences)
    selected_normalized = str(selected_category or "").strip().lower()
    valid_map = {item.lower(): item for item in categories}

    if selected_normalized not in valid_map:
        speak("No readable content was found for this page.", preferences)
        return None

    return valid_map[selected_normalized]


def run_harshaka_reading_for_image(
    image_path: Path,
    document_type: str,
    preferences: Dict,
    ask_depth: bool = True,
    default_depth: str = "summary",
    describe_context: Optional[str] = None,
):
    """
    Send image + document type to Harshaka.
    Select a real detected category only.
    Speak Harshaka text, then optional Abhishek image description.
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

        if str(analysis_result.get("status", "")).lower() == "failed":
            speak("No readable content was found for this page.", preferences)
            return

        selected_category = select_harshaka_category(
            analysis_result=analysis_result,
            document_type=document_type,
            preferences=preferences,
        )

        if not selected_category:
            return

        if ask_depth:
            depth = ask_summary_or_full(preferences)
        else:
            depth = default_depth

        if depth not in {"summary", "full"}:
            depth = default_depth if default_depth in {"summary", "full"} else "summary"

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

        if describe_context:
            describe_page_visual(
                image_path=image_path,
                context=describe_context,
                preferences=preferences,
            )

    except Exception as error:
        print("Harshaka reading failed:")
        print(error)
        speak("Content reading module failed for this page.", preferences)


def run_page_loop(
    preferences: Dict,
    document_type: str,
    turn_page_message: str,
    continue_question: str,
    stop_message: str,
    describe_context: str,
    missing_image_message: str,
):
    speak(turn_page_message, preferences)

    while True:
        choice = ask_continue_or_stop(preferences, continue_question)

        if choice == "stop":
            speak(stop_message, preferences)
            break

        ok = run_camera_once(preferences, target_module="capture_only")
        if not ok:
            break

        image_path = get_latest_captured_image_path()
        if image_path is None:
            speak(missing_image_message, preferences)
            continue

        run_harshaka_reading_for_image(
            image_path=image_path,
            document_type=document_type,
            preferences=preferences,
            ask_depth=True,
            default_depth="summary",
            describe_context=describe_context,
        )


def run_novel_loop(preferences: Dict):
    run_page_loop(
        preferences=preferences,
        document_type="Novel",
        turn_page_message="Please turn to the next page of the novel.",
        continue_question="Say next to continue to the next page, or say stop to finish.",
        stop_message="Novel reading stopped.",
        describe_context="novel_page",
        missing_image_message="Captured page image was not found.",
    )


def run_magazine_loop(preferences: Dict):
    run_page_loop(
        preferences=preferences,
        document_type="Magazine",
        turn_page_message="Please turn to the next page of the magazine.",
        continue_question="Say next to continue to the next page, or say stop to finish.",
        stop_message="Magazine reading stopped.",
        describe_context="general",
        missing_image_message="Captured page image was not found.",
    )


def run_newspaper_loop(first_image_path: Optional[Path], preferences: Dict):
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
        choice = ask_continue_or_stop(
            preferences,
            "Say next to capture one article, or say stop to finish.",
        )

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
            ask_depth=True,
            default_depth="summary",
            describe_context="newspaper_article_image",
        )

        speak("Move to another article and say next, or say stop to finish.", preferences)


def run_single_image_document(
    image_path: Optional[Path],
    document_type: str,
    preferences: Dict,
    ask_depth: bool,
    default_depth: str,
    describe_context: str,
):
    if image_path is None or not image_path.exists():
        speak("Captured page image was not found.", preferences)
        return

    run_harshaka_reading_for_image(
        image_path=image_path,
        document_type=document_type,
        preferences=preferences,
        ask_depth=ask_depth,
        default_depth=default_depth,
        describe_context=describe_context,
    )


def main():
    print("==========================================")
    print("SMART READING ASSISTANT FULL CONTROLLER")
    print("Rashmi + Manoj + Abhishek + Harshaka")
    print("==========================================")

    configure_rfid_for_pi()
    preferences = get_user_preferences()
    save_current_preferences(preferences)

    print("\nCurrent user preferences:")
    print(json.dumps(preferences, indent=4))

    ok = run_camera_once(preferences, target_module="abhishek")
    if not ok:
        return

    abhishek_result = read_json_file(ABHISHEK_RESULT_PATH)
    if not abhishek_result:
        speak("Abhishek analysis result was not found.", preferences)
        return

    print("Abhishek result:")
    print(json.dumps(abhishek_result, indent=4))

    if str(abhishek_result.get("status", "")).lower() == "failed":
        speak("Document analysis failed.", preferences)
        return

    document_type = normalize_document_type(abhishek_result.get("document_type"))
    first_image_path = resolve_image_path(abhishek_result)

    speak_first_page(abhishek_result, document_type, preferences)

    if document_type == "Novel":
        run_novel_loop(preferences)
    elif document_type == "Magazine":
        run_magazine_loop(preferences)
    elif document_type == "Newspaper":
        run_newspaper_loop(first_image_path, preferences)
    elif document_type == "Report":
        run_single_image_document(
            image_path=first_image_path,
            document_type="Report",
            preferences=preferences,
            ask_depth=True,
            default_depth="summary",
            describe_context="general",
        )
    elif document_type == "Printed_Letter":
        run_single_image_document(
            image_path=first_image_path,
            document_type="Printed_Letter",
            preferences=preferences,
            ask_depth=False,
            default_depth="full",
            describe_context="general",
        )
    else:
        speak(
            "The document type could not be identified clearly.",
            preferences,
        )

    speak("Smart Reading Assistant session completed.", preferences)


if __name__ == "__main__":
    main()

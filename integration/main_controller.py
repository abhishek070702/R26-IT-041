import json
import logging
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

import requests

# Hide OpenAI/httpx HTTP Request lines on the Pi terminal.
for _log_name in (
    "openai",
    "openai._base_client",
    "httpx",
    "httpcore",
    "httpcore.connection",
    "httpcore.http11",
    "urllib3",
    "urllib3.connectionpool",
):
    logging.getLogger(_log_name).setLevel(logging.WARNING)

_THREAD_HTTP = threading.local()
_HTTP_TIMEOUT = (5, 180)


def _http_session() -> requests.Session:
    session = getattr(_THREAD_HTTP, "session", None)
    if session is None:
        session = requests.Session()
        _THREAD_HTTP.session = session
    return session


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
ABHISHEK_DESCRIBE_URL = (
    os.getenv("ABHISHEK_DESCRIBE_IMAGE_URL")
    or os.getenv("ABHISHEK_DESCRIBE_URL")
    or ABHISHEK_ANALYZE_URL.replace("/abhishek/analyze", "/abhishek/describe-image")
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
    ask_article_choice_by_voice,
    ask_by_voice,
    ask_category_by_voice,
    ask_summary_or_full,
    ask_yes_no,
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


def speak_first_page(
    abhishek_result: Dict,
    document_type: str,
    preferences: Dict,
    since_capture: Optional[float] = None,
):
    """
    Speak type/title first so the user hears useful audio immediately,
    then the cover description (same content, progressive delivery).
    """
    title = abhishek_result.get("title")
    cover_description = first_cover_description(abhishek_result)
    mark = since_capture if since_capture is not None else time.perf_counter()

    def _log_first_audio(stage: str) -> None:
        print(
            "[controller] timing capture_to_first_audio="
            f"{time.perf_counter() - mark:.2f}s stage={stage}"
        )

    if document_type == "Novel":
        parts = ["This is a Novel."]
        if is_usable_title(title):
            parts.append(f"The title is {title}.")
        _log_first_audio("first_page_title_start")
        speak_reading_text(" ".join(parts), preferences)
        if cover_description:
            speak_reading_text(cover_description, preferences)
        return

    if document_type == "Magazine":
        parts = ["This is a Magazine."]
        if is_usable_title(title):
            parts.append(f"The title is {title}.")
        _log_first_audio("first_page_title_start")
        speak_reading_text(" ".join(parts), preferences)
        if cover_description:
            speak_reading_text(cover_description, preferences)
        return

    if document_type == "Newspaper":
        parts = ["This is a Newspaper."]
        if is_usable_title(title):
            parts.append(f"The newspaper name is {title}.")
        _log_first_audio("first_page_title_start")
        speak_reading_text(" ".join(parts), preferences)
        return

    if document_type == "Report":
        _log_first_audio("first_page_title_start")
        speak("This is a Report.", preferences)
        return

    if document_type == "Printed_Letter":
        _log_first_audio("first_page_title_start")
        speak("This is a Printed Letter.", preferences)
        return

    speak("The document type could not be identified clearly.", preferences)


def fetch_page_visual_description(
    image_path: Path,
    context: str,
) -> Optional[Dict]:
    """HTTP-only /describe-image fetch for prefetch during reading speech."""
    if image_path is None or not Path(image_path).exists():
        return None

    try:
        started = time.perf_counter()
        with open(image_path, "rb") as image_file:
            response = _http_session().post(
                ABHISHEK_DESCRIBE_URL,
                files={
                    "file": (Path(image_path).name, image_file, "image/jpeg"),
                },
                data={"context": context or "general"},
                timeout=_HTTP_TIMEOUT,
            )
        print(
            f"[controller] timing abhishek_describe={time.perf_counter() - started:.2f}s"
        )
        response.raise_for_status()
        return response.json()
    except Exception as error:
        print("Abhishek describe-image failed:")
        print(error)
        return None


def speak_page_visual_result(
    result: Optional[Dict],
    preferences: Dict,
) -> bool:
    if not result or not isinstance(result, dict):
        return False

    description = (
        result.get("image_description")
        or result.get("description")
        or ""
    )
    has_image = result.get("has_image")

    if is_meaningful_image_description(description, has_image=has_image):
        speak_reading_text(str(description).strip(), preferences)
        return True

    print("Describe skipped: no picture on the page.")
    return False


def describe_page_visual(
    image_path: Path,
    context: str,
    preferences: Dict,
) -> bool:
    """
    Call Abhishek /abhishek/describe-image and speak only if a picture is on the page.
    Returns True when a caption was spoken.
    """
    return speak_page_visual_result(
        fetch_page_visual_description(image_path, context),
        preferences,
    )


def print_runtime_endpoints() -> None:
    print("Abhishek analyze:", ABHISHEK_ANALYZE_URL)
    print("Abhishek describe:", ABHISHEK_DESCRIBE_URL)
    print("Harshaka analyze:", os.getenv("HARSHAAKA_ANALYZE_URL", "http://127.0.0.1:8001/analyze"))
    print("Harshaka generate:", os.getenv("HARSHAAKA_GENERATE_URL", "http://127.0.0.1:8001/generate-output"))
    print("Rashmi TTS:", os.getenv("RASHMI_TTS_URL") or "(local only)")


def send_image_to_abhishek(image_path: Path) -> Optional[Dict]:
    print("Controller sending image to Abhishek:", ABHISHEK_ANALYZE_URL)
    try:
        started = time.perf_counter()
        with open(image_path, "rb") as image_file:
            response = _http_session().post(
                ABHISHEK_ANALYZE_URL,
                files={"file": (image_path.name, image_file, "image/jpeg")},
                timeout=_HTTP_TIMEOUT,
            )
        print(
            "[controller] timing capture_to_abhishek_response="
            f"{time.perf_counter() - started:.2f}s"
        )
        response.raise_for_status()
        result = response.json()
        result["captured_image_path"] = str(image_path)
        OUTPUT_DIR.mkdir(exist_ok=True)
        with open(ABHISHEK_RESULT_PATH, "w", encoding="utf-8") as file:
            json.dump(result, file, indent=4)
        return result
    except Exception as error:
        print("Controller Abhishek request failed:", error)
        return None


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
    env["RASHMI_VOICE"] = "female"
    env["RASHMI_PACE"] = "normal"
    env["RASHMI_TONE"] = "natural"

    print("Starting camera script:", CAMERA_SCRIPT)
    print("Target module:", target_module)
    camera_started = time.perf_counter()

    # Start YOLO/camera while speaking status so load overlaps with audio.
    process = subprocess.Popen(
        [sys.executable, str(CAMERA_SCRIPT)],
        cwd=str(MANOJ_DIR),
        env=env,
    )

    if target_module == "abhishek":
        speak("Now I will turn on the camera.", preferences)
        speak("Please wait. The camera model is loading.", preferences)
    else:
        speak("Opening the camera to capture the page.", preferences)
        speak("Please wait. The camera model is loading.", preferences)

    returncode = process.wait()
    print(
        f"[controller] timing camera_session={time.perf_counter() - camera_started:.2f}s "
        f"exit={returncode}"
    )
    print("Camera module closed. Exit code:", returncode)
    if returncode != 0:
        speak("Camera guidance failed.", preferences)
        return False

    return True


def ask_continue_or_stop(preferences: Dict, question: str) -> str:
    return ask_by_voice(
        question=question,
        valid_answers=["next", "stop"],
        default_answer="stop",
        preferences=preferences,
        attempts=3,
        max_seconds=4,
    )


def ask_new_material_or_end(preferences: Dict) -> str:
    return ask_by_voice(
        question=(
            "Do you want to start a new novel, newspaper, magazine, report, or letter? "
            "Say new to start a new material, or say end to finish the session."
        ),
        valid_answers=["new", "end"],
        default_answer="end",
        preferences=preferences,
        attempts=3,
        max_seconds=4,
    )


def list_capture_articles(
    analysis_result: Dict,
    document_type: str,
) -> List[str]:
    """
    Real detected Harshaka categories from one capture.
    These are the selectable 'articles' for voice choice. Never invent names.
    """
    categories: List[str] = get_available_categories(analysis_result)
    if str(document_type or "").strip().lower() == "newspaper":
        categories = [
            item for item in categories if str(item).strip().lower() != "story"
        ]
    return categories


def select_harshaka_category(
    analysis_result: Dict,
    document_type: str,
    preferences: Dict,
    announce_empty: bool = True,
) -> Optional[str]:
    categories = list_capture_articles(analysis_result, document_type)

    if not categories:
        if announce_empty:
            speak("No readable content was found for this page.", preferences)
        return None

    if len(categories) == 1:
        selected_category = categories[0]
        speak(f"Single article detected: {selected_category}.", preferences)
        return selected_category

    selected_category = ask_article_choice_by_voice(
        categories,
        preferences,
        remaining=False,
    )
    selected_normalized = str(selected_category or "").strip().lower()
    valid_map = {item.lower(): item for item in categories}

    if selected_normalized not in valid_map:
        if announce_empty:
            speak("No readable content was found for this page.", preferences)
        return None

    return valid_map[selected_normalized]


def _generate_and_speak_article(
    analysis_result: Dict,
    selected_category: str,
    preferences: Dict,
    ask_depth: bool,
    default_depth: str,
    image_path: Path,
    describe_context: Optional[str],
    prefetch_describe: bool,
    flow_started: float,
) -> tuple[bool, bool]:
    """
    Generate one selected article and speak it.
    Returns (spoke_text, described).
    """
    if ask_depth:
        depth = ask_summary_or_full(preferences)
    else:
        depth = default_depth

    if depth not in {"summary", "full"}:
        depth = default_depth if default_depth in {"summary", "full"} else "summary"

    generate_started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=1) as pool:
        generate_future = pool.submit(
            generate_selected_harshaka_output,
            analysis_result,
            selected_category,
            depth,
        )
        speak(
            f"Preparing {depth} for {selected_category}.",
            preferences,
        )
        final_result = generate_future.result()
    print(
        "[controller] timing harshaka_generate="
        f"{time.perf_counter() - generate_started:.2f}s"
    )

    print("Harshaka final result:")
    print(json.dumps(final_result, indent=4))

    describe_future = None
    describe_pool = None
    if prefetch_describe and describe_context:
        describe_pool = ThreadPoolExecutor(max_workers=1)
        describe_future = describe_pool.submit(
            fetch_page_visual_description,
            image_path,
            describe_context,
        )

    spoke_text = False
    described = False
    try:
        print(
            "[controller] timing page_to_reading_speech_start="
            f"{time.perf_counter() - flow_started:.2f}s"
        )
        speak_harshaka_result(final_result, preferences)
        spoke_text = True

        if describe_future is not None:
            described = speak_page_visual_result(
                describe_future.result(),
                preferences,
            )
    finally:
        if describe_pool is not None:
            describe_pool.shutdown(wait=False)

    return spoke_text, described


def run_harshaka_reading_for_image(
    image_path: Path,
    document_type: str,
    preferences: Dict,
    ask_depth: bool = True,
    default_depth: str = "summary",
    describe_context: Optional[str] = None,
) -> str:
    """
    Send image + document type to Harshaka.
    Select real detected articles/categories only. Never invent names.

    When multiple articles are found in one capture:
      1) ask which to read first
      2) after reading, ask about remaining articles before any new capture
      3) continue until the user declines or none remain

    Returns:
      "read" | "no_content" | "failed"
    """
    spoke_text = False
    described = False
    flow_started = time.perf_counter()
    try:
        harshaka_analyze_started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=1) as pool:
            analyze_future = pool.submit(
                analyze_page_with_harshaka,
                str(image_path),
                document_type,
            )
            speak("Sending page to content reading module.", preferences)
            analysis_result = analyze_future.result()
        print(
            "[controller] timing harshaka_analyze="
            f"{time.perf_counter() - harshaka_analyze_started:.2f}s"
        )

        articles = list_capture_articles(analysis_result, document_type)
        print("Harshaka analysis result preview:")
        print(json.dumps({
            "status": analysis_result.get("status"),
            "document_type": analysis_result.get("document_type"),
            "categories": articles,
            "extracted_text_preview": (analysis_result.get("extracted_text") or "")[:500],
        }, indent=4))

        status_failed = str(analysis_result.get("status", "")).lower() == "failed"
        if status_failed or not articles:
            if describe_context:
                described = describe_page_visual(
                    image_path=image_path,
                    context=describe_context,
                    preferences=preferences,
                )
            if not described:
                if not bool(describe_context):
                    speak("No readable content was found for this page.", preferences)
                elif not articles and not status_failed:
                    speak("No readable content was found for this page.", preferences)
            print(
                "[controller] timing harshaka_page_total="
                f"{time.perf_counter() - flow_started:.2f}s"
            )
            return "no_content"

        remaining = list(articles)
        first_choice = True
        while remaining:
            if len(remaining) == 1 and first_choice:
                selected_category = remaining[0]
                speak(
                    f"Single article detected: {selected_category}.",
                    preferences,
                )
            else:
                selected_raw = ask_article_choice_by_voice(
                    remaining,
                    preferences,
                    remaining=not first_choice,
                )
                selected_normalized = str(selected_raw or "").strip().lower()
                valid_map = {item.lower(): item for item in remaining}
                selected_category = valid_map.get(selected_normalized)
                if not selected_category:
                    speak(
                        "I could not match that article. Please try again.",
                        preferences,
                    )
                    continue

            article_spoke, article_described = _generate_and_speak_article(
                analysis_result=analysis_result,
                selected_category=selected_category,
                preferences=preferences,
                ask_depth=ask_depth,
                default_depth=default_depth,
                image_path=image_path,
                describe_context=describe_context,
                prefetch_describe=first_choice and bool(describe_context),
                flow_started=flow_started,
            )
            spoke_text = spoke_text or article_spoke
            described = described or article_described

            remaining = [
                item
                for item in remaining
                if item.lower() != selected_category.lower()
            ]
            first_choice = False

            if not remaining:
                break

            if len(remaining) == 1:
                more_question = (
                    f"Do you want to read the remaining article, "
                    f"{remaining[0]}? Say yes or no."
                )
            else:
                more_question = (
                    "Do you want to read either of the remaining articles? "
                    "Say yes or no."
                )

            more = ask_yes_no(more_question, preferences, default_answer="no")
            if more != "yes":
                break

        if not spoke_text and not described:
            speak("No readable content was found for this page.", preferences)

        print(
            "[controller] timing harshaka_page_total="
            f"{time.perf_counter() - flow_started:.2f}s"
        )
        return "read" if spoke_text or described else "no_content"

    except Exception as error:
        print("Harshaka reading failed:")
        print(error)
        described = False
        if describe_context:
            described = describe_page_visual(
                image_path=image_path,
                context=describe_context,
                preferences=preferences,
            )
        if not described:
            speak("Content reading module failed for this page.", preferences)
        return "failed"


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
        turn_page_message="This page is finished.",
        continue_question=(
            "Turn to the next page and say next, or say stop."
        ),
        stop_message="Novel reading stopped.",
        describe_context="novel_page",
        missing_image_message="Captured page image was not found.",
    )


def run_magazine_loop(preferences: Dict):
    run_page_loop(
        preferences=preferences,
        document_type="Magazine",
        turn_page_message="This page is finished.",
        continue_question=(
            "Turn to the next page and say next, or say stop."
        ),
        stop_message="Magazine reading stopped.",
        describe_context="magazine_page",
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

    confirm_before_capture = True
    while True:
        if confirm_before_capture:
            choice = ask_continue_or_stop(
                preferences,
                "Say next to capture articles from the page, or say stop.",
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
            confirm_before_capture = True
            continue

        # Read all chosen articles from THIS capture before any new capture.
        run_harshaka_reading_for_image(
            image_path=image_path,
            document_type="Newspaper",
            preferences=preferences,
            ask_depth=True,
            default_depth="summary",
            describe_context="newspaper_article_image",
        )

        other = ask_yes_no(
            "Would you like to read any other material? Say yes or no.",
            preferences,
            default_answer="no",
        )
        if other != "yes":
            speak("Newspaper reading stopped.", preferences)
            break

        speak(
            "Please place the next reading material in front of the camera.",
            preferences,
        )
        # User already said yes to more material — open camera again without
        # asking for another next/stop, and without discarding unread articles
        # from a prior capture (those were already offered above).
        confirm_before_capture = False


def run_report_flow(image_path: Optional[Path], preferences: Dict):
    run_single_image_document(
        image_path=image_path,
        document_type="Report",
        preferences=preferences,
        ask_depth=True,
        default_depth="summary",
        describe_context="general",
    )


def run_printed_letter_flow(image_path: Optional[Path], preferences: Dict):
    run_single_image_document(
        image_path=image_path,
        document_type="Printed_Letter",
        preferences=preferences,
        ask_depth=False,
        default_depth="full",
        describe_context="general",
    )


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

    while True:
        choice = ask_continue_or_stop(
            preferences,
            "This page is finished. Say next to read the next page, or say stop.",
        )
        if choice == "stop":
            break

        ok = run_camera_once(preferences, target_module="capture_only")
        if not ok:
            break

        next_image = get_latest_captured_image_path()
        if next_image is None:
            speak("Captured page image was not found.", preferences)
            continue

        run_harshaka_reading_for_image(
            image_path=next_image,
            document_type=document_type,
            preferences=preferences,
            ask_depth=ask_depth,
            default_depth=default_depth,
            describe_context=describe_context,
        )


def capture_and_identify_document(preferences: Dict) -> Optional[Dict]:
    ok = run_camera_once(preferences, target_module="abhishek")
    if not ok:
        return None

    # Camera subprocess already ran next_module analyze for abhishek target.
    capture_ready_at = time.perf_counter()

    abhishek_result = read_json_file(ABHISHEK_RESULT_PATH)
    if not abhishek_result or str(abhishek_result.get("status", "")).lower() == "failed":
        image_path = get_latest_captured_image_path()
        if image_path is not None:
            speak("Sending the page for analysis again.", preferences)
            retried = send_image_to_abhishek(image_path)
            if retried:
                abhishek_result = retried
                capture_ready_at = time.perf_counter()

    if abhishek_result and isinstance(abhishek_result, dict):
        abhishek_result["_capture_ready_at"] = capture_ready_at
    return abhishek_result


def run_document_workflow(abhishek_result: Dict, preferences: Dict) -> None:
    document_type = normalize_document_type(abhishek_result.get("document_type"))
    first_image_path = resolve_image_path(abhishek_result)
    since_capture = abhishek_result.get("_capture_ready_at")
    if not isinstance(since_capture, (int, float)):
        since_capture = None

    workflow_started = time.perf_counter()
    speak_first_page(
        abhishek_result,
        document_type,
        preferences,
        since_capture=since_capture,
    )

    if document_type == "Novel":
        run_novel_loop(preferences)
    elif document_type == "Magazine":
        run_magazine_loop(preferences)
    elif document_type == "Newspaper":
        run_newspaper_loop(first_image_path, preferences)
    elif document_type == "Report":
        run_report_flow(first_image_path, preferences)
    elif document_type == "Printed_Letter":
        run_printed_letter_flow(first_image_path, preferences)
    else:
        speak(
            "The document type could not be identified clearly.",
            preferences,
        )

    print(
        "[controller] timing document_workflow_total="
        f"{time.perf_counter() - workflow_started:.2f}s type={document_type or 'unknown'}"
    )


def main():
    print("==========================================")
    print("SMART READING ASSISTANT FULL CONTROLLER")
    print("Rashmi + Manoj + Abhishek + Harshaka")
    print("==========================================")

    configure_rfid_for_pi()
    print_runtime_endpoints()
    preferences = get_user_preferences()
    save_current_preferences(preferences)

    print("\nCurrent user preferences:")
    print(json.dumps(preferences, indent=4))

    while True:
        abhishek_result = capture_and_identify_document(preferences)

        if not abhishek_result:
            speak("Abhishek analysis result was not found.", preferences)
        elif str(abhishek_result.get("status", "")).lower() == "failed":
            speak("Document analysis failed.", preferences)
        else:
            print("Abhishek result:")
            printable = {
                key: value
                for key, value in abhishek_result.items()
                if not str(key).startswith("_")
            }
            print(json.dumps(printable, indent=4))
            run_document_workflow(abhishek_result, preferences)

        choice = ask_new_material_or_end(preferences)
        if choice != "new":
            break

        speak(
            "Please place the new reading material in front of the camera.",
            preferences,
        )

    speak("Smart Reading Assistant session completed.", preferences)


if __name__ == "__main__":
    main()

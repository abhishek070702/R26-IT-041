import json
import subprocess
import sys
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

RASHMI_DIR = CURRENT_DIR / "rashmi_tts_module"
MANOJ_DIR = CURRENT_DIR / "manoj_camera_module"

PREFERENCES_PATH = RASHMI_DIR / "current_preferences.json"
CAMERA_SCRIPT = MANOJ_DIR / "final_yolo_opencv_guidance.py"
ABHISHEK_RESULT_PATH = MANOJ_DIR / "integration_outputs" / "last_abhishek_result.json"

sys.path.insert(0, str(CURRENT_DIR))

from rashmi_tts_module.rashmi_bridge import (
    get_user_preferences,
    speak,
    speak_abhishek_result_from_json,
)


def save_current_preferences(preferences):
    """
    Save current user preferences so next modules can reuse them.
    """
    RASHMI_DIR.mkdir(exist_ok=True)

    with open(PREFERENCES_PATH, "w", encoding="utf-8") as file:
        json.dump(preferences, file, indent=4)

    print("Current preferences saved to:", PREFERENCES_PATH)


def start_camera_guidance():
    """
    Start Manoj camera guidance and auto-capture module.
    """
    if not CAMERA_SCRIPT.exists():
        print("Camera script not found:", CAMERA_SCRIPT)
        speak("Camera guidance module was not found.")
        return

    speak("Opening camera guidance module.")

    subprocess.run(
        [sys.executable, str(CAMERA_SCRIPT)],
        cwd=str(MANOJ_DIR),
    )


def speak_latest_abhishek_result(preferences):
    """
    Speak latest Abhishek result after camera module closes.
    """
    if ABHISHEK_RESULT_PATH.exists():
        print("Speaking latest Abhishek result:", ABHISHEK_RESULT_PATH)
        speak_abhishek_result_from_json(ABHISHEK_RESULT_PATH, preferences)
    else:
        print("No Abhishek result JSON found yet.")
        speak("No analysis result was found yet.", preferences)


def main():
    print("==========================================")
    print("SMART READING ASSISTANT MAIN CONTROLLER")
    print("==========================================")

    preferences = get_user_preferences()
    save_current_preferences(preferences)

    print("\nCurrent user preferences:")
    print(json.dumps(preferences, indent=4))

    answer = input("\nStart camera guidance now? yes/no: ").strip().lower()

    if answer not in ["yes", "y", ""]:
        speak("Preferences saved. Camera guidance was not started.", preferences)
        print("Camera not started. Test completed.")
        return

    start_camera_guidance()

    print("\nCamera module closed.")
    speak_latest_abhishek_result(preferences)


if __name__ == "__main__":
    main()
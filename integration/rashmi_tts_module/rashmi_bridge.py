import json
import platform
import sqlite3
import subprocess
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "users.db"


def speak(text, preferences=None):
    """
    Simple TTS bridge.
    Windows: PowerShell voice
    Raspberry Pi/Linux: espeak-ng
    """
    if not text:
        return

    print("TTS:", text)

    system_name = platform.system().lower()

    try:
        if "windows" in system_name:
            safe_text = text.replace("'", "''")

            command = (
                "Add-Type -AssemblyName System.Speech; "
                "$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                "$speaker.Volume = 100; "
                "$speaker.Rate = 0; "
                f"$speaker.Speak('{safe_text}');"
            )

            subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-Command",
                    command,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        else:
            speed = "140"

            if preferences:
                pace = preferences.get("pace", "").lower()

                if pace == "slow":
                    speed = "115"
                elif pace == "fast":
                    speed = "170"

            subprocess.run(
                ["espeak-ng", "-s", speed, text],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

    except Exception as error:
        print("TTS error:", error)


def init_database():
    """
    Create SQLite database for RFID user preferences.
    """
    connection = sqlite3.connect(DB_PATH)
    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            rfid_id TEXT UNIQUE NOT NULL,
            reading_level TEXT,
            voice TEXT,
            pace TEXT,
            tone TEXT
        )
        """
    )

    connection.commit()
    connection.close()


def find_user(rfid_id):
    """
    Find registered user by RFID ID.
    """
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    cursor = connection.cursor()

    cursor.execute("SELECT * FROM users WHERE rfid_id = ?", (rfid_id,))
    row = cursor.fetchone()

    connection.close()

    if row is None:
        return None

    return dict(row)


def save_user_preferences(rfid_id, reading_level, voice, pace, tone):
    """
    Save or update user preferences.
    """
    connection = sqlite3.connect(DB_PATH)
    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT OR REPLACE INTO users
        (rfid_id, reading_level, voice, pace, tone)
        VALUES (?, ?, ?, ?, ?)
        """,
        (rfid_id, reading_level, voice, pace, tone),
    )

    connection.commit()
    connection.close()


def ask_new_preferences(rfid_id):
    """
    Register new user preferences.
    """
    speak("New user detected. Please provide your preferences.")

    print("\nNew User Registration")
    print("---------------------")

    reading_level = input("Reading level simple/moderate/advanced: ").strip() or "simple"
    voice = input("Voice male/female: ").strip() or "female"
    pace = input("Pace slow/normal/fast: ").strip() or "normal"
    tone = input("Tone friendly/calm/supportive: ").strip() or "friendly"

    save_user_preferences(
        rfid_id=rfid_id,
        reading_level=reading_level,
        voice=voice,
        pace=pace,
        tone=tone,
    )

    preferences = {
        "rfid_id": rfid_id,
        "reading_level": reading_level,
        "voice": voice,
        "pace": pace,
        "tone": tone,
    }

    speak("Your preferences have been saved. Opening camera.", preferences)
    return preferences


def get_user_preferences():
    """
    RFID + preference flow.
    For now, RFID is simulated using keyboard input.
    Later, this input can be replaced with real RFID reader code.
    """
    init_database()

    speak("Hello! Welcome to the Smart Reading Assistant System.")
    speak("Please tap your RFID card.")

    rfid_id = input("Enter RFID card ID: ").strip()

    if not rfid_id:
        rfid_id = "TEST_RFID_001"

    user = find_user(rfid_id)

    if user is None:
        return ask_new_preferences(rfid_id)

    speak("Known RFID card detected. Do you want to use your previous preferences?")

    print("\nKnown RFID card detected")
    print("------------------------")
    print("Saved preferences:")
    print("Reading level:", user["reading_level"])
    print("Voice:", user["voice"])
    print("Pace:", user["pace"])
    print("Tone:", user["tone"])

    answer = input("Use previous preferences? yes/no: ").strip().lower()

    if answer in ["yes", "y", ""]:
        preferences = {
            "rfid_id": user["rfid_id"],
            "reading_level": user["reading_level"],
            "voice": user["voice"],
            "pace": user["pace"],
            "tone": user["tone"],
        }

        speak("Previous preferences loaded. Opening camera.", preferences)
        return preferences

    return ask_new_preferences(rfid_id)


def format_abhishek_result_for_speech(result):
    """
    Convert Abhishek JSON result into simple speech text.
    """
    document_type = result.get("document_type", "document")
    title = result.get("title", "")
    descriptions = result.get("image_descriptions", [])

    lines = []

    lines.append("Page captured.")
    lines.append(f"This is a {document_type}.")

    if title and "not clearly detected" not in title.lower():
        lines.append(f"The title is {title}.")

    if descriptions:
        first_description = descriptions[0]

        if first_description:
            lines.append(f"Image description. {first_description}")

    return " ".join(lines)


def speak_abhishek_result(result, preferences=None):
    """
    Speak Abhishek module result.
    """
    speech_text = format_abhishek_result_for_speech(result)
    speak(speech_text, preferences)
    return speech_text


def speak_abhishek_result_from_json(json_path, preferences=None):
    """
    Read Abhishek result JSON and speak it.
    """
    json_path = Path(json_path)

    if not json_path.exists():
        print("Result JSON not found:", json_path)
        speak("Analysis result was not found.", preferences)
        return None

    with open(json_path, "r", encoding="utf-8") as file:
        result = json.load(file)

    return speak_abhishek_result(result, preferences)


if __name__ == "__main__":
    prefs = get_user_preferences()

    sample_result = {
        "document_type": "Novel",
        "confidence": 0.93,
        "title": "The Jungle Book",
        "image_descriptions": [
            "The cover shows a boy and animals in a forest scene."
        ],
        "status": "success",
        "warnings": []
    }

    speak_abhishek_result(sample_result, prefs)
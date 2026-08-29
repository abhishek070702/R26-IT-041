import json
import os
import platform
import re
import sqlite3
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

try:
    from .simplifier import simplify_text
    from .pn532_rfid import read_rfid_uid
except ImportError:
    from simplifier import simplify_text
    from pn532_rfid import read_rfid_uid


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent

DB_PATH = BASE_DIR / "users.db"
CURRENT_PREF_PATH = BASE_DIR / "current_preferences.json"

VOSK_MODEL_PATH = Path(
    os.getenv(
        "VOSK_MODEL_PATH",
        str(PROJECT_ROOT / "tts-preferences" / "vosk-model-small-en-us-0.15")
    )
)

MIC_DEVICE = os.getenv("RASHMI_MIC_DEVICE", "plughw:2,0")
SAMPLE_RATE = int(os.getenv("RASHMI_SAMPLE_RATE", "16000"))

_VOSK_MODEL = None


def _voice_name(preferences: Optional[Dict] = None) -> str:
    if not preferences:
        return "female"
    voice = str(
        preferences.get("voice")
        or preferences.get("voice_type")
        or "female"
    ).lower()
    return "male" if voice == "male" else "female"


def _espeak_command(text: str, preferences: Optional[Dict] = None) -> list:
    speed = "140"
    pitch = "50"
    voice = "en+f3"

    if preferences:
        pace = str(preferences.get("pace", "normal")).lower()
        if pace == "slow":
            speed = "115"
        elif pace == "fast":
            speed = "170"

        if _voice_name(preferences) == "male":
            voice = "en+m3"
            pitch = "35"
        else:
            voice = "en+f3"
            pitch = "72"

        tone = str(preferences.get("tone", "")).lower()
        if tone == "calm":
            speed = str(max(int(speed) - 15, 90))
        elif tone == "friendly":
            pitch = str(min(int(pitch) + 8, 90))

    return ["espeak-ng", "-v", voice, "-s", speed, "-p", pitch, text]


def speak(text: str, preferences: Optional[Dict] = None):
    """
    Rashmi TTS bridge.
    Windows: PowerShell voice for laptop testing.
    Linux/Raspberry Pi: espeak-ng with male/female voice from preferences.
    """
    if not text:
        return

    text = str(text).strip()
    if not text:
        return

    print("TTS:", text)

    system_name = platform.system().lower()

    try:
        if "windows" in system_name:
            safe_text = text.replace("'", "''")
            gender = "Female" if _voice_name(preferences) == "female" else "Male"

            command = (
                "Add-Type -AssemblyName System.Speech; "
                "$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                "$speaker.Volume = 100; "
                "$speaker.Rate = 0; "
                f"$gender = [System.Speech.Synthesis.VoiceGender]::{gender}; "
                "foreach ($v in $speaker.GetInstalledVoices()) { "
                "  if ($v.VoiceInfo.Gender -eq $gender) { "
                "    $speaker.SelectVoice($v.VoiceInfo.Name); break "
                "  } "
                "} "
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
                check=False,
            )

        else:
            subprocess.run(
                _espeak_command(text, preferences),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )

    except Exception as error:
        print("TTS error:", error)


def speak_reading_text(text: str, preferences: Optional[Dict] = None):
    """
    Speak document content using saved reading_level, voice, pace, and tone.
    """
    text = str(text or "").strip()
    if not text:
        return ""

    level = ""
    if preferences:
        level = str(preferences.get("reading_level") or "")

    adapted = simplify_text(text, level)
    if adapted != text:
        print("Reading text adapted for level:", level or "simple")

    speak(adapted, preferences)
    return adapted


def init_database():
    """
    Create a stable user preference table.
    This uses user_preferences table to avoid old users table schema problems.
    """
    connection = sqlite3.connect(DB_PATH)
    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS user_preferences (
            rfid_id TEXT PRIMARY KEY,
            reading_level TEXT,
            voice TEXT,
            pace TEXT,
            tone TEXT
        )
        """
    )

    connection.commit()
    connection.close()


def find_user(rfid_id: str):
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    cursor = connection.cursor()

    cursor.execute(
        "SELECT * FROM user_preferences WHERE rfid_id = ?",
        (rfid_id,)
    )
    row = cursor.fetchone()

    connection.close()

    if row is None:
        return None

    return dict(row)


def _preferences_from_user(user: Dict) -> Dict:
    return {
        "rfid_id": user.get("rfid_id") or "DEFAULT_USER",
        "reading_level": user.get("reading_level") or "simple",
        "voice": user.get("voice") or "female",
        "voice_type": user.get("voice") or user.get("voice_type") or "female",
        "pace": user.get("pace") or "normal",
        "tone": user.get("tone") or "friendly",
    }


def find_any_saved_user():
    saved = load_current_preferences()
    if isinstance(saved, dict) and saved.get("reading_level"):
        return _preferences_from_user(saved)

    for user_id in (
        os.getenv("RASHMI_USER_ID", "DEFAULT_USER"),
        "CARD0001",
        "DEFAULT_USER",
    ):
        user = find_user(user_id)
        if user is not None:
            return _preferences_from_user(user)

    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM user_preferences ORDER BY rowid DESC LIMIT 1")
    row = cursor.fetchone()
    connection.close()
    if row is None:
        return None
    return _preferences_from_user(dict(row))


def save_user_preferences(
    rfid_id: str,
    reading_level: str,
    voice: str,
    pace: str,
    tone: str,
):
    connection = sqlite3.connect(DB_PATH)
    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT OR REPLACE INTO user_preferences
        (rfid_id, reading_level, voice, pace, tone)
        VALUES (?, ?, ?, ?, ?)
        """,
        (rfid_id, reading_level, voice, pace, tone),
    )

    connection.commit()
    connection.close()


def save_current_preferences(preferences: Dict):
    BASE_DIR.mkdir(exist_ok=True)

    with open(CURRENT_PREF_PATH, "w", encoding="utf-8") as file:
        json.dump(preferences, file, indent=4)

    print("Current preferences saved to:", CURRENT_PREF_PATH)


def load_current_preferences():
    if not CURRENT_PREF_PATH.exists():
        return None

    with open(CURRENT_PREF_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def _compact(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def _get_vosk_model():
    global _VOSK_MODEL

    if _VOSK_MODEL is not None:
        return _VOSK_MODEL

    if not VOSK_MODEL_PATH.exists():
        print("Vosk model not found:", VOSK_MODEL_PATH)
        return None

    try:
        from vosk import Model
        _VOSK_MODEL = Model(str(VOSK_MODEL_PATH))
        return _VOSK_MODEL
    except Exception as error:
        print("Could not load Vosk model:", error)
        return None


def listen_once(max_seconds: int = 8) -> str:
    """
    Listen for one answer.
    On Windows, use keyboard fallback.
    On Raspberry Pi/Linux, use arecord + Vosk.
    """
    system_name = platform.system().lower()

    if "windows" in system_name:
        return input("Type answer: ").strip().lower()

    model = _get_vosk_model()

    if model is None:
        return input("Voice model unavailable. Type answer: ").strip().lower()

    try:
        from vosk import KaldiRecognizer

        recognizer = KaldiRecognizer(model, SAMPLE_RATE)

        process = subprocess.Popen(
            [
                "arecord",
                "-D", MIC_DEVICE,
                "-f", "S16_LE",
                "-r", str(SAMPLE_RATE),
                "-c", "1",
                "-t", "raw",
                "-d", str(max_seconds),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

        detected_text = ""

        try:
            while True:
                data = process.stdout.read(4000)

                if not data:
                    break

                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    text = result.get("text", "").strip().lower()

                    if text:
                        detected_text = text
                        break

            if not detected_text:
                final_result = json.loads(recognizer.FinalResult())
                detected_text = final_result.get("text", "").strip().lower()

        finally:
            process.terminate()
            process.wait()

        print("Recognized speech:", detected_text)
        return detected_text

    except Exception as error:
        print("Voice input error:", error)
        return input("Voice input failed. Type answer: ").strip().lower()


def _phrase_in_answer(phrase: str, answer_text: str) -> bool:
    phrase = str(phrase).lower().strip()
    answer_text = str(answer_text).lower().strip()

    if not phrase:
        return False

    pattern = r"\b" + re.escape(phrase) + r"\b"
    return re.search(pattern, answer_text) is not None


def _match_answer(answer_text: str, valid_answers: List[str]):
    """
    Match recognized/typed answer to allowed answers safely.
    Avoids bug where 'female' becomes 'male'.
    """
    answer_text = str(answer_text).lower().strip()
    answer_compact = _compact(answer_text)

    aliases = {
        "summary": ["summary", "summery", "summarize", "short", "brief"],
        "full": ["full", "all", "read it", "read full", "full text", "whole", "complete"],
        "next": [
            "next",
            "nest",
            "necks",
            "go",
            "continue",
            "proceed",
            "start",
            "ready",
            "really",
        ],
        "stop": ["stop", "end", "finish", "quit"],
        "yes": ["yes", "yeah", "yep", "ok", "okay"],
        "no": ["no", "nope"],
        "ready": ["ready", "done", "finished"],
        "simple": ["simple", "easy"],
        "moderate": ["moderate", "medium", "normal level"],
        "advanced": ["advanced", "advance", "hard"],
        "male": ["male", "mail", "man"],
        "female": ["female", "email", "woman"],
        "slow": ["slow"],
        "normal": ["normal"],
        "fast": ["fast"],
        "calm": ["calm", "come"],
        "friendly": ["friendly", "friend"],
        "supportive": ["supportive", "support"],
        "story": ["story"],
        "sports": ["sports", "sport"],
        "politics": ["politics", "politic", "political"],
        "science": ["science"],
        "social": ["social"],
        "general": ["general"],
        "general text": ["general", "general text"],
        "mathematics": ["mathematics", "maths", "math"],
    }

    # 1. Exact match first
    for valid in valid_answers:
        valid_lower = str(valid).lower().strip()
        if answer_text == valid_lower or answer_compact == _compact(valid_lower):
            return valid

    # 2. Exact alias match
    for valid in valid_answers:
        valid_lower = str(valid).lower().strip()
        for alias in aliases.get(valid_lower, []):
            if answer_text == alias or answer_compact == _compact(alias):
                return valid

    # 3. Word/phrase match only. No unsafe substring match.
    for valid in valid_answers:
        valid_lower = str(valid).lower().strip()

        if _phrase_in_answer(valid_lower, answer_text):
            return valid

        for alias in aliases.get(valid_lower, []):
            if _phrase_in_answer(alias, answer_text):
                return valid

    return None


def ask_by_voice(
    question: str,
    valid_answers: List[str],
    default_answer: Optional[str] = None,
    preferences: Optional[Dict] = None,
    attempts: int = 3,
    max_seconds: int = 8,
):
    """
    Ask user by Rashmi TTS and capture answer.
    On laptop, this uses typed input.
    On Raspberry Pi, this uses Vosk voice recognition.
    """
    valid_answers = [str(answer) for answer in valid_answers]

    for attempt in range(1, attempts + 1):
        speak(question, preferences)

        allowed_text = ", ".join(valid_answers)
        print(f"Allowed answers: {allowed_text}")
        print(f"Listening... attempt {attempt}/{attempts}")

        answer_text = listen_once(max_seconds=max_seconds)
        selected = _match_answer(answer_text, valid_answers)

        if selected is not None:
            speak(f"You selected {selected}.", preferences)
            return selected

        speak("Sorry, I did not understand. Please try again.", preferences)

    if default_answer is None:
        default_answer = valid_answers[0] if valid_answers else ""

    speak(f"I will use {default_answer}.", preferences)
    return default_answer


def ask_category_by_voice(categories: List[str], preferences: Optional[Dict] = None):
    """
    Ask user to select a detected Harshaka category by category name.
    """
    clean_categories = [str(category) for category in categories if str(category).strip()]

    if not clean_categories:
        return "general"

    category_text = ", ".join(clean_categories)
    question = f"I found these categories: {category_text}. Which category do you want to hear?"

    return ask_by_voice(
        question=question,
        valid_answers=clean_categories,
        default_answer=clean_categories[0],
        preferences=preferences,
        attempts=3,
        max_seconds=8,
    )


def ask_summary_or_full(preferences: Optional[Dict] = None):
    return ask_by_voice(
        question="Do you want a summary or full text?",
        valid_answers=["summary", "full"],
        default_answer="summary",
        preferences=preferences,
        attempts=3,
        max_seconds=8,
    )


def ask_ready_for_next_page(preferences: Optional[Dict] = None):
    return ask_by_voice(
        question="Please turn to the next page. Say next when done.",
        valid_answers=["next"],
        default_answer="next",
        preferences=preferences,
        attempts=3,
        max_seconds=8,
    )


def ask_new_preferences(rfid_id: str):
    speak("New user detected. I will ask your preferences now.")

    temp_preferences = {"pace": "normal"}

    reading_level = ask_by_voice(
        "Please say your reading level. Say simple, moderate, or advanced.",
        ["simple", "moderate", "advanced"],
        "simple",
        temp_preferences,
    )

    voice = ask_by_voice(
        "Please say your voice type. Say male or female.",
        ["male", "female"],
        "female",
        temp_preferences,
    )

    pace = ask_by_voice(
        "Please say your reading speed. Say slow, normal, or fast.",
        ["slow", "normal", "fast"],
        "normal",
        temp_preferences,
    )

    temp_preferences["pace"] = pace

    tone = ask_by_voice(
        "Please say your tone. Say friendly, calm, or supportive.",
        ["friendly", "calm", "supportive"],
        "friendly",
        temp_preferences,
    )

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
        "voice_type": voice,
        "pace": pace,
        "tone": tone,
    }

    save_current_preferences(preferences)
    speak("Your preferences have been saved.", preferences)
    return preferences


def get_user_preferences():
    """
    RFID + preference flow.
    Wait for a PN532 card tap on the Pi. Keyboard entry is only a fallback.
    After the card id, preferences are collected by Rashmi voice input.
    """
    init_database()

    speak("Hello. Welcome to the Smart Reading Assistant System.")
    speak("Please tap your RFID card on the reader.")

    rfid_id = read_rfid_uid().strip()

    if not rfid_id:
        rfid_id = "TEST_RFID_001"

    print("RFID card ID:", rfid_id)

    user = find_user(rfid_id)

    if user is None:
        return ask_new_preferences(rfid_id)

    preferences = _preferences_from_user(user)

    speak("Known RFID card detected.", preferences)

    answer = ask_by_voice(
        "Do you want to use your previous preferences? Say yes or no.",
        ["yes", "no"],
        "yes",
        preferences,
    )

    if answer == "yes":
        save_current_preferences(preferences)
        speak("Previous preferences loaded.", preferences)
        return preferences

    return ask_new_preferences(rfid_id)


def format_abhishek_result_for_speech(result: Dict):
    """
    Convert Abhishek JSON result into simple speech text.
    """
    document_type = result.get("document_type", "document")
    title = result.get("title", "")
    descriptions = result.get("image_descriptions", [])

    lines = ["Page captured.", f"This is a {document_type}."]

    if title and "not clearly" not in str(title).lower() and "handled by" not in str(title).lower():
        if str(document_type).lower() == "newspaper":
            lines.append(f"The newspaper name is {title}.")
        else:
            lines.append(f"The title is {title}.")

    if descriptions:
        first_description = descriptions[0]
        if first_description:
            lines.append(f"Image description. {first_description}")

    return " ".join(lines)


def speak_abhishek_result(result: Dict, preferences: Optional[Dict] = None):
    speech_text = format_abhishek_result_for_speech(result)
    speak_reading_text(speech_text, preferences)
    return speech_text


def speak_abhishek_result_from_json(json_path, preferences: Optional[Dict] = None):
    json_path = Path(json_path)

    if not json_path.exists():
        print("Result JSON not found:", json_path)
        speak("Analysis result was not found.", preferences)
        return None

    with open(json_path, "r", encoding="utf-8") as file:
        result = json.load(file)

    return speak_abhishek_result(result, preferences)


def format_harshaka_result_for_speech(result: Dict):
    """
    Harshaka final output should speak the text field.
    Also supports final_output_text and next_module_payload formats.
    """
    if not isinstance(result, dict):
        return str(result)

    text = (
        result.get("text")
        or result.get("final_output_text")
        or result.get("next_module_payload")
        or ""
    )

    if isinstance(text, dict):
        text = text.get("text", "")

    return str(text).strip()


def speak_harshaka_result(result: Dict, preferences: Optional[Dict] = None):
    speech_text = format_harshaka_result_for_speech(result)

    if not speech_text:
        speech_text = "No readable content was found."

    speak_reading_text(speech_text, preferences)
    return speech_text


if __name__ == "__main__":
    prefs = get_user_preferences()
    speak("Preference test completed.", prefs)

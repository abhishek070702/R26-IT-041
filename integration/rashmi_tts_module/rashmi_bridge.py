import json
import math
import os
import platform
import queue
import re
import shutil
import sqlite3
import struct
import subprocess
import threading
import time
import wave
from pathlib import Path
from typing import Dict, List, Optional

try:
    from .simplifier import simplify_text
    from .pn532_rfid import read_rfid_uid
    from .preference_tts import (
        speak_preference_voice,
        speak_reading_voice,
        system_speech_preferences,
    )
    from .tone_classifier import classify_live_tone, classifier_ready
except ImportError:
    from simplifier import simplify_text
    from pn532_rfid import read_rfid_uid
    from preference_tts import (
        speak_preference_voice,
        speak_reading_voice,
        system_speech_preferences,
    )
    from tone_classifier import classify_live_tone, classifier_ready


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

MIC_DEVICE = os.getenv("RASHMI_MIC_DEVICE", "plughw:3,0")
SAMPLE_RATE = int(os.getenv("RASHMI_SAMPLE_RATE", "16000"))
BEEP_PATH = BASE_DIR / "listen_beep.wav"

_VOSK_MODEL = None
_WORKING_MIC = None


def _voice_name(preferences: Optional[Dict] = None) -> str:
    if not preferences:
        return "female"
    voice = str(
        preferences.get("voice")
        or preferences.get("voice_type")
        or "female"
    ).lower()
    return "male" if voice == "male" else "female"


def speak(text: str, preferences: Optional[Dict] = None):
    """
    System guidance voice. Same as the welcome message.
    User reading prefs (voice, pace, tone, reading_level) are ignored.
    """
    if not text:
        return

    text = str(text).strip()
    if not text:
        return

    print("TTS:", text)
    try:
        speak_preference_voice(text, system_speech_preferences())
    except Exception as error:
        print("TTS error:", error)


def speak_reading_text(text: str, preferences: Optional[Dict] = None):
    """
    Speak actual reading material only: title, cover/page pictures, Harshaka text.
    Applies saved reading_level, voice, pace, and tone.
    """
    text = str(text or "").strip()
    if not text:
        return ""

    level = ""
    if preferences:
        level = str(preferences.get("reading_level") or "")

    adapted = simplify_text(text, level)
    if adapted != text:
        print("Reading text adapted for level:", level or "very simple")

    try:
        speak_reading_voice(adapted, preferences)
    except Exception as error:
        print("Reading TTS error:", error)
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
        "reading_level": user.get("reading_level") or "very simple",
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
        from vosk import Model, SetLogLevel

        SetLogLevel(-1)
        print("Loading Vosk speech model...")
        _VOSK_MODEL = Model(str(VOSK_MODEL_PATH))
        print("Vosk speech model ready.")
        return _VOSK_MODEL
    except Exception as error:
        print("Could not load Vosk model:", error)
        return None


def _preload_speech_models():
    try:
        _get_vosk_model()
    except Exception as error:
        print("Vosk preload failed:", error)
    try:
        classifier_ready()
    except Exception as error:
        print("Tone classifier preload failed:", error)


def _typed_input_requested() -> bool:
    return os.getenv("RASHMI_TYPED_INPUT", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _windows_mic_device():
    raw = os.getenv("RASHMI_WINDOWS_MIC_DEVICE", "").strip()
    if raw.isdigit():
        return int(raw)
    return None


ANSWER_ALIASES = {
    "summary": ["summary", "summery", "summarize", "short", "brief"],
    "full": ["full", "all", "read it", "read full", "full text", "whole", "complete"],
    "next": [
        "next",
        "nest",
        "necks",
        "go",
        "continue",
        "proceed",
        "ready",
        "really",
        "next page",
    ],
    "stop": ["stop", "quit"],
    "new": ["new", "another", "material", "restart", "again", "start"],
    "end": ["end", "finish", "quit", "exit", "close", "done", "stop"],
    "yes": ["yes", "yeah", "yep", "ok", "okay"],
    "no": ["no", "nope"],
    "ready": ["ready", "done", "finished"],
    "very simple": ["very simple", "simple", "very easy", "easy"],
    "simple": ["simple", "easy", "very simple"],
    "moderate": ["moderate", "medium", "normal level"],
    "light": ["light", "original", "close to original", "advanced", "advance", "hard"],
    "advanced": ["advanced", "advance", "hard", "light"],
    "male": ["male", "mail", "man"],
    "female": ["female", "email", "woman"],
    "slow": ["slow"],
    "normal": ["normal"],
    "fast": ["fast"],
    "calm": ["calm", "come", "peaceful", "soft"],
    "friendly": ["friendly", "friend", "warm", "kind"],
    "natural": ["natural", "neutral"],
    "emotional": [
        "emotional",
        "emotion",
        "imotional",
        "dramatic",
        "expressive",
    ],
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


def _make_vosk_recognizer(model, sample_rate: int, valid_answers: Optional[List[str]] = None):
    from vosk import KaldiRecognizer

    phrases = set()
    for answer in valid_answers or []:
        valid = str(answer).lower().strip()
        if not valid:
            continue
        phrases.add(valid)
        phrases.update(ANSWER_ALIASES.get(valid, []))

    if phrases:
        grammar = sorted(phrases) + ["[unk]"]
        return KaldiRecognizer(model, sample_rate, json.dumps(grammar))

    return KaldiRecognizer(model, sample_rate)


def _clean_recognized_text(text: str) -> str:
    cleaned = str(text or "").strip().lower()
    if cleaned in {"[unk]", "unk"}:
        return ""
    return cleaned


def _ensure_beep_wav() -> Path:
    if BEEP_PATH.exists() and BEEP_PATH.stat().st_size > 400:
        return BEEP_PATH

    rate = 16000
    duration = 0.22
    freq = 880.0
    n_samples = int(rate * duration)
    fade = 240
    frames = bytearray()
    for index in range(n_samples):
        envelope = min(1.0, index / fade, (n_samples - index) / fade)
        sample = int(
            0.38 * envelope * 32767 * math.sin(2.0 * math.pi * freq * index / rate)
        )
        frames += struct.pack("<h", max(-32767, min(32767, sample)))

    BEEP_PATH.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(BEEP_PATH), "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(rate)
        wav_file.writeframes(frames)
    return BEEP_PATH


def _play_listen_beep() -> None:
    """Play a short cue, then wait so the speaker releases before recording."""
    beep_path = _ensure_beep_wav()
    played = False

    if "windows" in platform.system().lower():
        try:
            import winsound

            winsound.PlaySound(str(beep_path), winsound.SND_FILENAME)
            played = True
        except Exception as error:
            print("Beep playback failed:", error)

    if not played:
        for command in (
            ["aplay", "-q", str(beep_path)],
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(beep_path)],
            ["paplay", str(beep_path)],
            ["mpv", "--no-video", "--really-quiet", str(beep_path)],
        ):
            if shutil.which(command[0]) is None:
                continue
            result = subprocess.run(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if result.returncode == 0:
                played = True
                break

    if not played:
        print("Listen beep could not be played; listening anyway.")
    else:
        print("Listen beep played. Speak now.")
    time.sleep(0.12)


def _discover_alsa_capture_devices() -> List[str]:
    devices: List[str] = []
    try:
        listed = subprocess.run(
            ["arecord", "-l"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception as error:
        print("Could not list ALSA capture devices:", error)
        return devices

    output = listed.stdout or ""
    print("ALSA capture devices:\n" + (output.strip() or "(none)"))
    for line in output.splitlines():
        match = re.search(
            r"card\s+(\d+)\s*:\s*(\S+).*?device\s+(\d+)\s*:",
            line,
            flags=re.IGNORECASE,
        )
        if not match:
            continue
        card, card_name, dev = match.group(1), match.group(2), match.group(3)
        devices.extend(
            [
                f"plughw:{card},{dev}",
                f"hw:{card},{dev}",
                f"plughw:CARD={card_name},DEV={dev}",
            ]
        )
    return devices


def _mic_device_candidates() -> List[str]:
    if _WORKING_MIC:
        return [_WORKING_MIC]
    seen: List[str] = []
    for device in (
        [MIC_DEVICE]
        + _discover_alsa_capture_devices()
        + [
            "pulse",
            "pipewire",
            "sysdefault",
            "plughw:3,0",
            "plughw:2,0",
            "plughw:1,0",
            "plughw:0,0",
        ]
    ):
        name = str(device or "").strip()
        if name and name not in seen:
            seen.append(name)
    return seen


def _pcm_rms(pcm: bytes) -> float:
    if len(pcm) < 4:
        return 0.0
    count = len(pcm) // 2
    total = 0.0
    for index in range(0, len(pcm) - 1, 2):
        sample = struct.unpack_from("<h", pcm, index)[0]
        total += float(sample) * float(sample)
    return math.sqrt(total / max(count, 1)) / 32768.0


def _recognize_pcm(
    model,
    pcm: bytes,
    valid_answers: Optional[List[str]] = None,
) -> str:
    if not pcm:
        return ""

    recognizer = _make_vosk_recognizer(model, SAMPLE_RATE, valid_answers)
    detected_text = ""
    chunk_size = 4000
    for offset in range(0, len(pcm), chunk_size):
        chunk = pcm[offset : offset + chunk_size]
        if recognizer.AcceptWaveform(chunk):
            result = json.loads(recognizer.Result())
            text = _clean_recognized_text(result.get("text", ""))
            if text:
                detected_text = text
                break
    if not detected_text:
        final_result = json.loads(recognizer.FinalResult())
        detected_text = _clean_recognized_text(final_result.get("text", ""))

    if detected_text or not valid_answers:
        return detected_text

    # Grammar pass missed the words; try free recognition on the same clip.
    recognizer = _make_vosk_recognizer(model, SAMPLE_RATE, None)
    for offset in range(0, len(pcm), chunk_size):
        chunk = pcm[offset : offset + chunk_size]
        if recognizer.AcceptWaveform(chunk):
            result = json.loads(recognizer.Result())
            text = _clean_recognized_text(result.get("text", ""))
            if text:
                return text
    final_result = json.loads(recognizer.FinalResult())
    return _clean_recognized_text(final_result.get("text", ""))


def _wav_to_mono_pcm(wav_path: Path) -> bytes:
    with wave.open(str(wav_path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        frames = wav_file.readframes(wav_file.getnframes())
        sample_width = wav_file.getsampwidth()
    if sample_width != 2:
        return frames
    if channels <= 1:
        return frames
    mono = bytearray()
    frame_count = len(frames) // (2 * channels)
    for index in range(frame_count):
        total = 0
        for channel in range(channels):
            offset = (index * channels + channel) * 2
            total += struct.unpack_from("<h", frames, offset)[0]
        sample = int(total / channels)
        mono += struct.pack("<h", max(-32767, min(32767, sample)))
    return bytes(mono)


def _record_pi_pcm(max_seconds: int, device: str) -> tuple[bytes, str]:
    wav_path = BASE_DIR / "last_listen.wav"
    last_error = ""
    for channels in (1, 2):
        command = [
            "arecord",
            "-D",
            device,
            "-f",
            "S16_LE",
            "-r",
            str(SAMPLE_RATE),
            "-c",
            str(channels),
            "-t",
            "wav",
            "-d",
            str(max(1, int(max_seconds))),
            str(wav_path),
        ]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=max(max_seconds + 4, 8),
            check=False,
        )
        error_text = (result.stderr or result.stdout or "").strip()
        if result.returncode != 0:
            last_error = error_text or f"arecord exited {result.returncode}"
            if channels == 1 and "channel" in last_error.lower():
                continue
            return b"", last_error
        if not wav_path.exists():
            last_error = error_text or "arecord did not write a recording"
            continue
        try:
            return _wav_to_mono_pcm(wav_path), error_text
        except Exception as error:
            last_error = str(error)
    return b"", last_error or "arecord failed"


def _downmix_stereo_chunk(data: bytes) -> bytes:
    usable = data[: len(data) // 4 * 4]
    if not usable:
        return b""
    mono = bytearray()
    for index in range(0, len(usable), 4):
        left = struct.unpack_from("<h", usable, index)[0]
        right = struct.unpack_from("<h", usable, index + 2)[0]
        mono += struct.pack("<h", int((left + right) / 2))
    return bytes(mono)


def _listen_pi_stream(
    model,
    max_seconds: int,
    valid_answers: Optional[List[str]],
    device: str,
) -> tuple[str, str]:
    last_error = ""
    for channels in (1, 2):
        process = subprocess.Popen(
            [
                "arecord",
                "-D",
                device,
                "-f",
                "S16_LE",
                "-r",
                str(SAMPLE_RATE),
                "-c",
                str(channels),
                "-t",
                "raw",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        time.sleep(0.08)
        if process.poll() is not None:
            err = (process.stderr.read() or b"").decode("utf-8", "ignore").strip()
            last_error = err or f"arecord exited {process.returncode}"
            continue

        recognizer = _make_vosk_recognizer(model, SAMPLE_RATE, valid_answers)
        detected_text = ""
        pcm = bytearray()
        started = time.monotonic()
        read_size = 4000 if channels == 1 else 8000
        try:
            while time.monotonic() - started < max_seconds:
                data = process.stdout.read(read_size)
                if not data:
                    break
                if channels == 2:
                    data = _downmix_stereo_chunk(data)
                if not data:
                    continue
                pcm.extend(data)
                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    text = _clean_recognized_text(result.get("text", ""))
                    if text:
                        detected_text = text
                        break
            if not detected_text:
                final_result = json.loads(recognizer.FinalResult())
                detected_text = _clean_recognized_text(final_result.get("text", ""))
            if not detected_text and valid_answers and pcm:
                detected_text = _recognize_pcm(model, bytes(pcm), None)
        finally:
            process.terminate()
            try:
                process.wait(timeout=0.8)
            except Exception:
                process.kill()

        level = _pcm_rms(bytes(pcm))
        print(f"Microphone level: {level:.4f}")
        if detected_text:
            return detected_text, ""
        if level < 0.004:
            last_error = f"silent recording on {device}"
            continue
        return "", last_error
    return "", last_error or "arecord failed"


def _listen_with_sounddevice(
    max_seconds: int,
    valid_answers: Optional[List[str]] = None,
) -> str:
    import sounddevice as sd

    model = _get_vosk_model()
    if model is None:
        raise RuntimeError("Vosk model not available")

    device = _windows_mic_device()
    try:
        info = sd.query_devices(device, "input")
        print("Microphone:", info.get("name", "default"))
    except Exception as error:
        print("Microphone query failed:", error)

    recognizer = _make_vosk_recognizer(model, SAMPLE_RATE, valid_answers)
    audio_queue: queue.Queue = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print("Audio status:", status)
        audio_queue.put(bytes(indata))

    _play_listen_beep()
    print("Speak now...")
    detected_text = ""
    started = time.monotonic()

    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=4000,
        device=device,
        dtype="int16",
        channels=1,
        callback=callback,
    ):
        while time.monotonic() - started < max_seconds:
            try:
                data = audio_queue.get(timeout=0.25)
            except queue.Empty:
                continue

            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text = _clean_recognized_text(result.get("text", ""))
                if text:
                    detected_text = text
                    break

    if not detected_text:
        final_result = json.loads(recognizer.FinalResult())
        detected_text = _clean_recognized_text(final_result.get("text", ""))

    print("Recognized speech:", detected_text)
    return detected_text


def listen_once(
    max_seconds: int = 4,
    valid_answers: Optional[List[str]] = None,
) -> str:
    """
    Listen for one answer.
    Windows uses the laptop microphone + Vosk.
    Raspberry Pi/Linux uses arecord + Vosk.
    Set RASHMI_TYPED_INPUT=1 to type instead.
    """
    if _typed_input_requested():
        return input("Type answer: ").strip().lower()

    system_name = platform.system().lower()

    if "windows" in system_name:
        try:
            return _listen_with_sounddevice(max_seconds, valid_answers)
        except Exception as error:
            print("Voice input error:", error)
            return input("Voice input failed. Type answer: ").strip().lower()

    model = _get_vosk_model()

    if model is None:
        return input("Voice model unavailable. Type answer: ").strip().lower()

    try:
        global _WORKING_MIC
        _play_listen_beep()
        last_error = ""
        for device in _mic_device_candidates():
            print("Listening on microphone:", device)
            detected_text, error_text = _listen_pi_stream(
                model, max_seconds, valid_answers, device
            )
            if detected_text:
                _WORKING_MIC = device
                print("Recognized speech:", detected_text)
                return detected_text
            if error_text:
                last_error = error_text
                print("arecord failed on", device, ":", error_text)

        print("Voice input error:", last_error or "no usable microphone")
        return ""

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

    # 1. Exact match first
    for valid in valid_answers:
        valid_lower = str(valid).lower().strip()
        if answer_text == valid_lower or answer_compact == _compact(valid_lower):
            return valid

    # 2. Exact alias match
    for valid in valid_answers:
        valid_lower = str(valid).lower().strip()
        for alias in ANSWER_ALIASES.get(valid_lower, []):
            if answer_text == alias or answer_compact == _compact(alias):
                return valid

    # 3. Word/phrase match only. No unsafe substring match.
    for valid in valid_answers:
        valid_lower = str(valid).lower().strip()

        if _phrase_in_answer(valid_lower, answer_text):
            return valid

        for alias in ANSWER_ALIASES.get(valid_lower, []):
            if _phrase_in_answer(alias, answer_text):
                return valid

    return None


def ask_by_voice(
    question: str,
    valid_answers: List[str],
    default_answer: Optional[str] = None,
    preferences: Optional[Dict] = None,
    attempts: int = 3,
    max_seconds: int = 4,
):
    """
    Ask user by Rashmi TTS and capture answer with Vosk.
    Set RASHMI_TYPED_INPUT=1 to type instead of speaking.
    """
    valid_answers = [str(answer) for answer in valid_answers]

    for attempt in range(1, attempts + 1):
        speak(question, preferences)

        allowed_text = ", ".join(valid_answers)
        print(f"Allowed answers: {allowed_text}")
        print(f"Listening... attempt {attempt}/{attempts}")
        print("Wait for the beep, then speak.")

        answer_text = listen_once(
            max_seconds=max_seconds,
            valid_answers=valid_answers,
        )
        selected = _match_answer(answer_text, valid_answers)

        if selected is not None:
            print("Voice answer:", selected)
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
        max_seconds=4,
    )


def ask_summary_or_full(preferences: Optional[Dict] = None):
    return ask_by_voice(
        question="Do you want a summary or full text?",
        valid_answers=["summary", "full"],
        default_answer="summary",
        preferences=preferences,
        attempts=3,
        max_seconds=4,
    )


def ask_ready_for_next_page(preferences: Optional[Dict] = None):
    return ask_by_voice(
        question="Please turn to the next page. Say next when done.",
        valid_answers=["next"],
        default_answer="next",
        preferences=preferences,
        attempts=3,
        max_seconds=4,
    )


def ask_tone_preference(preferences: Optional[Dict] = None) -> str:
    """
    Ask the user to say natural, friendly, calm, or emotional.
    Then load mfcc_svm_voice_classifier.joblib to check a short spoken sample.
    The spoken word is saved even if the SVM guess differs.
    """
    selected_tone = ask_by_voice(
        "Please choose your preferred tone. Say natural, friendly, calm, or emotional.",
        ["natural", "friendly", "calm", "emotional"],
        "natural",
        preferences,
    )

    if not classifier_ready():
        return selected_tone

    speak(
        "Please say: I am ready to listen to my reading, using your selected tone.",
        preferences,
    )
    detected = classify_live_tone(seconds=3.0)
    if detected and detected == selected_tone:
        speak(f"{selected_tone} tone confirmed.", preferences)
    elif detected:
        print(
            "MFCC+SVM detected",
            detected,
            "but the spoken choice",
            selected_tone,
            "will be saved.",
        )

    return selected_tone


def ask_new_preferences(rfid_id: str, is_new_user: bool = True):
    if is_new_user:
        speak("New user detected. I will ask your preferences now.")
    else:
        speak("Okay. Please provide your preferences again.")

    temp_preferences = {"pace": "normal"}

    reading_level = ask_by_voice(
        "Please say your reading level. Say very simple, moderate, or light.",
        ["very simple", "moderate", "light"],
        "very simple",
        temp_preferences,
        max_seconds=10,
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

    tone = ask_tone_preference(temp_preferences)

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
    if is_new_user:
        speak("Your preferences have been saved.", preferences)
    else:
        speak("Your preferences have been updated.", preferences)
    return preferences


def get_user_preferences():
    """
    RFID + preference flow.
    Wait for a PN532 card tap on the Pi. Keyboard entry is only a fallback.
    After the card id, preferences are collected by Rashmi voice input.
    """
    init_database()

    print("Loading speech models in the background...")
    preload = threading.Thread(target=_preload_speech_models, daemon=True)
    preload.start()

    speak("Hello. Welcome to the Smart Reading Assistant System.")
    speak(
        "I will ask questions by voice. Wait until you hear a beep, then speak your answer."
    )
    speak("Please tap your RFID card on the right side of the device.")

    rfid_id = read_rfid_uid().strip()

    if not rfid_id:
        rfid_id = "TEST_RFID_001"

    print("RFID card ID:", rfid_id)
    preload.join(timeout=60)

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

    return ask_new_preferences(rfid_id, is_new_user=False)


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

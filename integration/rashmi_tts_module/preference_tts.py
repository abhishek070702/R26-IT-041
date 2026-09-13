"""
Rashmi Final 1.1.ipynb voices for the live device.

System guidance (welcome, camera, instructions): fixed female / normal / natural.
Document reading: user voice, pace, tone, and reading_level.
espeak-ng / Windows Speech is only the last offline fallback.
"""

from __future__ import annotations

import asyncio
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, Optional

# Welcome-message voice. Used for guidance, camera cues, and other system talk.
# User reading prefs (voice/pace/tone/level) must not change this.
SYSTEM_SPEECH_PREFERENCES = {
    "voice": "female",
    "voice_type": "female",
    "pace": "normal",
    "tone": "natural",
}


def system_speech_preferences() -> Dict[str, str]:
    return dict(SYSTEM_SPEECH_PREFERENCES)


# Same maps as tts-preferences/Final 1.1.ipynb
VOICE_MAP = {
    "female": "en-US-JennyNeural",
    "male": "en-US-GuyNeural",
}

PACE_MAP = {
    "slow": "-45%",
    "normal": "-20%",
    "fast": "+0%",
}

TONE_MAP = {
    "natural": {"pitch": "+0Hz", "volume": "+0%"},
    "friendly": {"pitch": "+18Hz", "volume": "+6%"},
    "calm": {"pitch": "-15Hz", "volume": "-10%"},
    "emotional": {"pitch": "+28Hz", "volume": "+8%"},
    "clear": {"pitch": "+0Hz", "volume": "+8%"},
    "supportive": {"pitch": "+6Hz", "volume": "+0%"},
}

TONE_ALIASES = {
    "natural": ("natural", "neutral"),
    "friendly": ("friendly", "warm", "kind"),
    "calm": ("calm", "peaceful", "soft"),
    "emotional": ("emotional", "emotion", "immotional"),
}

# Same idea as Final 1.1 OpenAI profiles: a different neural voice per tone.
EDGE_VOICE_PROFILES = {
    "female": {
        "natural": "en-US-JennyNeural",
        "friendly": "en-US-AriaNeural",
        "calm": "en-US-JennyNeural",
        "emotional": "en-US-MichelleNeural",
        "clear": "en-US-JennyNeural",
        "supportive": "en-US-AriaNeural",
    },
    "male": {
        "natural": "en-US-GuyNeural",
        "friendly": "en-US-AndrewNeural",
        "calm": "en-US-BrianNeural",
        "emotional": "en-US-ChristopherNeural",
        "clear": "en-US-GuyNeural",
        "supportive": "en-US-AndrewNeural",
    },
}

EDGE_TONE_RATE = {
    "natural": "+0%",
    "friendly": "+18%",
    "calm": "-22%",
    "emotional": "+8%",
    "clear": "+0%",
    "supportive": "+6%",
}

OPENAI_VOICE_PROFILES = {
    "female": {
        "natural": "marin",
        "friendly": "coral",
        "calm": "shimmer",
        "emotional": "nova",
    },
    "male": {
        "natural": "cedar",
        "friendly": "echo",
        "calm": "onyx",
        "emotional": "ballad",
    },
}

OPENAI_SPEED_MAP = {
    "slow": 0.85,
    "normal": 1.00,
    "fast": 1.15,
}

OPENAI_TONE_SPEED_MULTIPLIER = {
    "natural": 1.00,
    "friendly": 1.10,
    "calm": 0.80,
    "emotional": 0.93,
}

OPENAI_TONE_INSTRUCTIONS = {
    "natural": (
        "Read in a neutral, everyday speaking style. "
        "Use balanced pitch, medium energy, steady rhythm, and ordinary pauses. "
        "Keep the delivery clear and natural. "
        "Avoid sounding especially cheerful, soothing, or dramatic."
    ),
    "friendly": (
        "Read in a clearly cheerful, warm, upbeat, and welcoming style. "
        "Sound like you are smiling while speaking. "
        "Use brighter intonation, lively rhythm, positive energy, and noticeable friendly emphasis. "
        "Use shorter pauses and a more energetic conversational delivery. "
        "Make this sound obviously happier and more animated than the natural voice."
    ),
    "calm": (
        "Read in a very soft, peaceful, soothing, and relaxed style. "
        "Use noticeably lower energy, gentle intonation, a smooth slow rhythm, and longer pauses. "
        "Keep emphasis soft and controlled. "
        "Avoid excitement, strong emotional swings, and energetic delivery. "
        "Make this sound clearly quieter, slower, and more relaxing than the other voice types."
    ),
    "emotional": (
        "Read like an expressive storyteller with strong emotional range. "
        "Use obvious rises and falls in intonation, noticeable changes in intensity, "
        "strong emphasis on emotionally important words, and dramatic but natural pauses. "
        "Let the emotion of each sentence strongly affect the delivery. "
        "Make this sound clearly more expressive and dramatic than natural, friendly, or calm."
    ),
}

AUDIO_DIR = Path(__file__).resolve().parent / "tts_outputs"
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MAX_EDGE_CHARS = 3500
MAX_OPENAI_CHARS = 3500

_ENV_LOADED = False
_OPENAI_CLIENT = None
_OPENAI_LOAD_ATTEMPTED = False
_OPENAI_UNAVAILABLE = False
_TTS_SERVER_UNAVAILABLE = False
_EDGE_UNAVAILABLE = False
_EDGE_LOCK = threading.Lock()


def _voice_key(preferences: Optional[Dict] = None) -> str:
    if not preferences:
        return "female"
    voice = str(
        preferences.get("voice") or preferences.get("voice_type") or "female"
    ).lower()
    return "male" if voice == "male" else "female"


def _pace_key(preferences: Optional[Dict] = None) -> str:
    pace = str((preferences or {}).get("pace") or "normal").lower()
    return pace if pace in PACE_MAP else "normal"


def _tone_key(preferences: Optional[Dict] = None) -> str:
    tone = str((preferences or {}).get("tone") or "natural").lower().strip()
    for canonical, aliases in TONE_ALIASES.items():
        if tone == canonical or tone in aliases:
            return canonical
    if tone in TONE_MAP:
        return tone
    return "natural"


def _percent_int(value: str) -> int:
    return int(str(value).strip().replace("%", "").replace("Hz", "") or "0")


def _edge_voice_name(preferences: Optional[Dict] = None) -> str:
    voice_key = _voice_key(preferences)
    tone_key = _tone_key(preferences)
    return EDGE_VOICE_PROFILES[voice_key].get(
        tone_key,
        VOICE_MAP[voice_key],
    )


def _edge_voice_candidates(preferences: Optional[Dict] = None) -> list[str]:
    primary = _edge_voice_name(preferences)
    fallback = VOICE_MAP[_voice_key(preferences)]
    candidates = [primary]
    if fallback != primary:
        candidates.append(fallback)
    return candidates


def _edge_rate(preferences: Optional[Dict] = None) -> str:
    pace = _percent_int(PACE_MAP[_pace_key(preferences)])
    tone = _percent_int(EDGE_TONE_RATE.get(_tone_key(preferences), "+0%"))
    total = max(-50, min(50, pace + tone))
    return f"{total:+d}%"


def _load_env() -> None:
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _ENV_LOADED = True
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    for path in (
        PROJECT_ROOT / ".env",
        PROJECT_ROOT / "tts-preferences" / ".env",
        Path(__file__).resolve().parent / ".env",
        PROJECT_ROOT / "backend" / ".env",
        PROJECT_ROOT / "article_detection" / ".env",
    ):
        if path.is_file():
            load_dotenv(path, override=False)
            print("Loaded TTS env from:", path)


def _openai_client():
    global _OPENAI_CLIENT, _OPENAI_LOAD_ATTEMPTED, _OPENAI_UNAVAILABLE
    if _OPENAI_UNAVAILABLE:
        return None
    if _OPENAI_LOAD_ATTEMPTED:
        return _OPENAI_CLIENT

    _OPENAI_LOAD_ATTEMPTED = True
    _load_env()
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        print("OpenAI TTS key not found; reading will use Edge TTS.")
        _OPENAI_UNAVAILABLE = True
        return None

    try:
        from openai import OpenAI

        _OPENAI_CLIENT = OpenAI(api_key=api_key, timeout=20.0)
        return _OPENAI_CLIENT
    except Exception as error:
        print("OpenAI TTS client failed:", error)
        _OPENAI_UNAVAILABLE = True
        _OPENAI_CLIENT = None
        return None


def _engine_name() -> str:
    return os.getenv("RASHMI_TTS_ENGINE", "edge").strip().lower()


def _split_chunks(text: str, max_chars: int = MAX_EDGE_CHARS) -> list[str]:
    text = str(text or "").strip()
    if len(text) <= max_chars:
        return [text] if text else []

    chunks: list[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= max_chars:
            chunks.append(remaining)
            break
        window = remaining[:max_chars]
        split_at = max(window.rfind(". "), window.rfind("? "), window.rfind("! "))
        if split_at < max_chars // 2:
            split_at = window.rfind(" ")
        if split_at < 40:
            split_at = max_chars
        chunks.append(remaining[: split_at + 1].strip())
        remaining = remaining[split_at + 1 :].strip()
    return [item for item in chunks if item]


async def _edge_save(text: str, voice_name: str, rate: str, pitch: str, volume: str, out_path: Path):
    import edge_tts

    communicate = edge_tts.Communicate(
        text,
        voice_name,
        rate=rate,
        pitch=pitch,
        volume=volume,
    )
    await communicate.save(str(out_path))


def _run_asyncio(coro):
    """Run a coroutine even if FastAPI already has an event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result = {}

    def worker():
        try:
            result["value"] = asyncio.run(coro)
        except Exception as error:
            result["error"] = error

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join()
    if "error" in result:
        raise result["error"]
    return result.get("value")


def _synthesize_edge_cli(
    text: str,
    voice_name: str,
    rate: str,
    pitch: str,
    volume: str,
    out_path: Path,
) -> None:
    # Same CLI path as tts-preferences/Final 1.1.ipynb
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "edge_tts",
            "--voice",
            voice_name,
            f"--rate={rate}",
            f"--pitch={pitch}",
            f"--volume={volume}",
            "--text",
            text,
            "--write-media",
            str(out_path),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "edge-tts CLI failed").strip())


def _synthesize_edge_mp3(text: str, preferences: Optional[Dict] = None) -> Path:
    global _EDGE_UNAVAILABLE
    if _EDGE_UNAVAILABLE:
        raise RuntimeError("Edge TTS is not available.")

    try:
        import edge_tts  # noqa: F401
    except Exception as error:
        _EDGE_UNAVAILABLE = True
        raise RuntimeError("No module named 'edge_tts'") from error

    AUDIO_DIR.mkdir(exist_ok=True)
    voice_key = _voice_key(preferences)
    tone_key = _tone_key(preferences)
    rate = _edge_rate(preferences)
    tone_cfg = TONE_MAP.get(tone_key, TONE_MAP["natural"])
    pitch = tone_cfg["pitch"]
    volume = tone_cfg["volume"]
    voice_candidates = _edge_voice_candidates(preferences)

    last_error: Exception | None = None
    with _EDGE_LOCK:
        for voice_name in voice_candidates:
            print(
                "TTS engine=edge "
                f"voice={voice_key} tone={tone_key} "
                f"edge_voice={voice_name} rate={rate} pitch={pitch}"
            )
            for attempt in range(2):
                out_path = AUDIO_DIR / f"rashmi_{uuid.uuid4().hex[:10]}.mp3"
                try:
                    try:
                        _synthesize_edge_cli(
                            text, voice_name, rate, pitch, volume, out_path
                        )
                    except Exception:
                        _run_asyncio(
                            _edge_save(
                                text,
                                voice_name,
                                rate,
                                pitch,
                                volume,
                                out_path,
                            )
                        )
                    if out_path.exists() and out_path.stat().st_size > 0:
                        return out_path
                    last_error = RuntimeError("Edge TTS did not write audio.")
                except Exception as error:
                    last_error = error
                    try:
                        out_path.unlink(missing_ok=True)
                    except OSError:
                        pass
                    time.sleep(0.35 * (attempt + 1))
            print(f"Edge voice failed ({voice_name}); trying fallback if available.")

    raise RuntimeError(f"Edge TTS failed: {last_error}") from last_error


def _play_mp3(audio_path: Path) -> None:
    path = str(audio_path.resolve())

    for binary, extra in (
        ("ffplay", ["-nodisp", "-autoexit", "-loglevel", "quiet", path]),
        ("mpg123", ["-q", path]),
        ("mpv", ["--no-video", "--really-quiet", path]),
        ("cvlc", ["--play-and-exit", "--quiet", path]),
    ):
        if shutil.which(binary):
            subprocess.run(
                [binary, *extra],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            return

    try:
        from playsound import playsound

        playsound(path)
        return
    except Exception:
        pass

    if "windows" in platform.system().lower():
        safe = path.replace("'", "''")
        command = (
            "Add-Type -AssemblyName presentationCore; "
            "$m = New-Object System.Windows.Media.MediaPlayer; "
            f"$m.Open([uri]((Get-Item '{safe}').FullName)); "
            "Start-Sleep -Milliseconds 250; "
            "$m.Play(); "
            "while (-not $m.NaturalDuration.HasTimeSpan) { Start-Sleep -Milliseconds 50 }; "
            "Start-Sleep -Milliseconds ([int]($m.NaturalDuration.TimeSpan.TotalMilliseconds)); "
            "$m.Close();"
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
        return

    raise RuntimeError("No MP3 player found (install mpg123, ffmpeg, or playsound).")


def _espeak_binary() -> Optional[str]:
    return shutil.which("espeak-ng") or shutil.which("espeak")


def _espeak_command(text: str, preferences: Optional[Dict] = None) -> list:
    binary = _espeak_binary() or "espeak-ng"
    speed = "140"
    pitch = "50"
    voice = "en+f3"

    if preferences:
        pace = _pace_key(preferences)
        if pace == "slow":
            speed = "115"
        elif pace == "fast":
            speed = "170"

        if _voice_key(preferences) == "male":
            voice = "en+m3"
            pitch = "35"
        else:
            voice = "en+f3"
            pitch = "72"

        tone = _tone_key(preferences)
        if tone == "calm":
            speed = str(max(int(speed) - 15, 90))
        elif tone in {"friendly", "emotional"}:
            pitch = str(min(int(pitch) + 8, 90))

    return [binary, "-v", voice, "-s", speed, "-p", pitch, text]


def _speak_espeak_or_windows(text: str, preferences: Optional[Dict] = None) -> None:
    system_name = platform.system().lower()
    if "windows" in system_name:
        safe_text = text.replace("'", "''")
        gender = "Female" if _voice_key(preferences) == "female" else "Male"
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
        return

    command = _espeak_command(text, preferences)
    if shutil.which(command[0]) is None and shutil.which("spd-say"):
        subprocess.run(
            ["spd-say", "-w", str(text)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return
    if shutil.which(command[0]) is None:
        raise FileNotFoundError(
            "No local voice on this device (install espeak-ng or set RASHMI_TTS_URL)."
        )
    subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def _synthesize_openai_mp3(text: str, preferences: Optional[Dict] = None) -> Path:
    global _OPENAI_UNAVAILABLE

    client = _openai_client()
    if client is None:
        raise RuntimeError("OpenAI TTS is not available.")

    AUDIO_DIR.mkdir(exist_ok=True)
    out_path = AUDIO_DIR / f"rashmi_openai_{uuid.uuid4().hex[:10]}.mp3"

    voice_key = _voice_key(preferences)
    tone_key = _tone_key(preferences)
    if tone_key not in OPENAI_TONE_INSTRUCTIONS:
        tone_key = "natural"

    openai_voice = OPENAI_VOICE_PROFILES[voice_key][tone_key]
    pace_key = _pace_key(preferences)
    speed = max(
        0.25,
        min(
            4.0,
            OPENAI_SPEED_MAP.get(pace_key, 1.0)
            * OPENAI_TONE_SPEED_MULTIPLIER.get(tone_key, 1.0),
        ),
    )
    model = os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
    instructions = OPENAI_TONE_INSTRUCTIONS[tone_key]

    try:
        with client.audio.speech.with_streaming_response.create(
            model=model,
            voice=openai_voice,
            input=text,
            instructions=instructions,
            speed=speed,
            response_format="mp3",
        ) as response:
            response.stream_to_file(out_path)
    except Exception as error:
        message = str(error)
        if "401" in message or "invalid_api_key" in message.lower():
            print("OpenAI TTS key was rejected; reading will use Edge TTS.")
            _OPENAI_UNAVAILABLE = True
        raise

    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError("OpenAI TTS did not write audio.")

    print(
        "TTS engine=openai "
        f"voice={voice_key} tone={tone_key} "
        f"openai_voice={openai_voice} speed={speed:.2f}"
    )
    return out_path


def _speak_openai(text: str, preferences: Optional[Dict] = None) -> None:
    chunks = _split_chunks(text, MAX_OPENAI_CHARS)
    for chunk in chunks:
        audio_path = None
        try:
            audio_path = _synthesize_openai_mp3(chunk, preferences)
            _play_mp3(audio_path)
        finally:
            if audio_path is not None:
                try:
                    audio_path.unlink(missing_ok=True)
                except OSError:
                    pass


def _speak_edge(text: str, preferences: Optional[Dict] = None) -> None:
    chunks = _split_chunks(text)
    for chunk in chunks:
        audio_path = None
        try:
            audio_path = _synthesize_edge_mp3(chunk, preferences)
            _play_mp3(audio_path)
        finally:
            if audio_path is not None:
                try:
                    audio_path.unlink(missing_ok=True)
                except OSError:
                    pass


def _combine_mp3s(paths: list[Path]) -> Path:
    AUDIO_DIR.mkdir(exist_ok=True)
    out_path = AUDIO_DIR / f"rashmi_combined_{uuid.uuid4().hex[:10]}.mp3"
    with out_path.open("wb") as dest:
        for path in paths:
            dest.write(path.read_bytes())
    return out_path


def _synthesize_chunks(
    text: str,
    preferences: Optional[Dict],
    max_chars: int,
    synth_fn,
) -> Path:
    chunks = _split_chunks(text, max_chars)
    paths: list[Path] = []
    try:
        for chunk in chunks:
            paths.append(synth_fn(chunk, preferences))
        if len(paths) == 1:
            return paths[0]
        combined = _combine_mp3s(paths)
        for path in paths:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        return combined
    except Exception:
        for path in paths:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        raise


def _tts_server_url() -> str:
    return (os.getenv("RASHMI_TTS_URL") or "").strip()


def _speak_via_tts_server(text: str, preferences: Optional[Dict] = None) -> bool:
    global _TTS_SERVER_UNAVAILABLE
    if _TTS_SERVER_UNAVAILABLE:
        return False

    url = _tts_server_url()
    if not url:
        return False

    try:
        import requests
    except Exception as error:
        print("Laptop TTS client missing:", error)
        return False

    AUDIO_DIR.mkdir(exist_ok=True)
    out_path = AUDIO_DIR / f"rashmi_remote_{uuid.uuid4().hex[:10]}.mp3"
    try:
        response = requests.post(
            url,
            json={
                "text": text,
                "voice": _voice_key(preferences),
                "pace": _pace_key(preferences),
                "tone": _tone_key(preferences),
            },
            timeout=(2.0, 45.0),
        )
        if response.status_code >= 400:
            detail = (response.text or "").strip().replace("\n", " ")[:300]
            raise RuntimeError(
                f"{response.status_code} {response.reason} for url: {url}"
                + (f" ({detail})" if detail else "")
            )
        if not response.content:
            raise RuntimeError("TTS server returned empty audio")
        out_path.write_bytes(response.content)
        print("TTS engine=laptop-server", url)
        _play_mp3(out_path)
        return True
    except Exception as error:
        message = str(error).lower()
        if any(
            token in message
            for token in (
                "no route to host",
                "failed to establish",
                "connection refused",
                "nameresolutionerror",
                "timed out",
                "max retries",
            )
        ):
            _TTS_SERVER_UNAVAILABLE = True
            print("Laptop TTS server unreachable; not retrying until restart.")
        print("Laptop TTS server failed:", error)
        return False
    finally:
        try:
            out_path.unlink(missing_ok=True)
        except OSError:
            pass


def synthesize_speech_mp3(text: str, preferences: Optional[Dict] = None) -> Path:
    """
    Build an MP3 for the laptop TTS server. Does not play locally.
    OpenAI first, then Edge. Caller deletes the returned file.
    """
    text = str(text or "").strip()
    if not text:
        raise ValueError("text is required")

    if not _OPENAI_UNAVAILABLE:
        try:
            return _synthesize_chunks(
                text, preferences, MAX_OPENAI_CHARS, _synthesize_openai_mp3
            )
        except Exception as error:
            print("OpenAI TTS failed, using Edge voice:", error)

    print(
        "TTS engine=edge "
        f"voice={_voice_key(preferences)} "
        f"pace={_pace_key(preferences)} "
        f"tone={_tone_key(preferences)}"
    )
    return _synthesize_chunks(text, preferences, MAX_EDGE_CHARS, _synthesize_edge_mp3)


def speak_reading_voice(text: str, preferences: Optional[Dict] = None) -> None:
    """
    Final 1.1 document reading: OpenAI TTS with selected tone.
    Falls back to Edge, then espeak / Windows Speech.
    """
    text = str(text or "").strip()
    if not text:
        return

    if _speak_via_tts_server(text, preferences):
        return

    engine = _engine_name()
    if engine in {"espeak", "espeak-ng", "local"}:
        speak_preference_voice(text, preferences, allow_remote=False)
        return

    try:
        _speak_openai(text, preferences)
        return
    except Exception as error:
        print("OpenAI TTS failed, using Edge voice:", error)

    speak_preference_voice(text, preferences, allow_remote=False)


def speak_preference_voice(
    text: str,
    preferences: Optional[Dict] = None,
    allow_remote: bool = True,
) -> None:
    """
    Final 1.1 voice: OpenAI TTS (tone voices + style) when a key is available.
    Edge Jenny/Guy is only the fallback.
    """
    text = str(text or "").strip()
    if not text:
        return

    if allow_remote and _speak_via_tts_server(text, preferences):
        return

    engine = _engine_name()
    if engine in {"espeak", "espeak-ng", "local"}:
        _speak_espeak_or_windows(text, preferences)
        return

    try:
        _speak_openai(text, preferences)
        return
    except Exception as error:
        print("OpenAI TTS failed, using Edge voice:", error)

    try:
        print(
            "TTS engine=edge "
            f"voice={_voice_key(preferences)} "
            f"pace={_pace_key(preferences)} "
            f"tone={_tone_key(preferences)}"
        )
        _speak_edge(text, preferences)
    except Exception as error:
        print("Edge TTS failed, using offline voice:", error)
        _speak_espeak_or_windows(text, preferences)


def preferences_from_env() -> Dict[str, str]:
    """Camera / env helpers. Guidance always uses the system welcome voice."""
    return system_speech_preferences()

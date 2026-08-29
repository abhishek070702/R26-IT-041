"""
Rashmi Final 1.1.ipynb voices for the live device.

Edge neural TTS (Jenny / Guy) using the same voice, pace, and tone maps
as the notebook. espeak-ng / Windows Speech is only the offline fallback.
"""

from __future__ import annotations

import asyncio
import os
import platform
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Dict, Optional

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
    "friendly": {"pitch": "+10Hz", "volume": "+0%"},
    "calm": {"pitch": "-5Hz", "volume": "-5%"},
    "emotional": {"pitch": "+18Hz", "volume": "+0%"},
    "clear": {"pitch": "+0Hz", "volume": "+5%"},
    "supportive": {"pitch": "+4Hz", "volume": "+0%"},
}

AUDIO_DIR = Path(__file__).resolve().parent / "tts_outputs"
MAX_EDGE_CHARS = 3500


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
    tone = str((preferences or {}).get("tone") or "friendly").lower()
    if tone in TONE_MAP:
        return tone
    return "friendly"


def _engine_name() -> str:
    return os.getenv("RASHMI_TTS_ENGINE", "edge").strip().lower()


def _split_chunks(text: str) -> list[str]:
    text = str(text or "").strip()
    if len(text) <= MAX_EDGE_CHARS:
        return [text] if text else []

    chunks: list[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= MAX_EDGE_CHARS:
            chunks.append(remaining)
            break
        window = remaining[:MAX_EDGE_CHARS]
        split_at = max(window.rfind(". "), window.rfind("? "), window.rfind("! "))
        if split_at < MAX_EDGE_CHARS // 2:
            split_at = window.rfind(" ")
        if split_at < 40:
            split_at = MAX_EDGE_CHARS
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


def _synthesize_edge_mp3(text: str, preferences: Optional[Dict] = None) -> Path:
    AUDIO_DIR.mkdir(exist_ok=True)
    out_path = AUDIO_DIR / f"rashmi_{uuid.uuid4().hex[:10]}.mp3"
    voice_name = VOICE_MAP[_voice_key(preferences)]
    rate = PACE_MAP[_pace_key(preferences)]
    tone_cfg = TONE_MAP[_tone_key(preferences)]

    asyncio.run(
        _edge_save(
            text,
            voice_name,
            rate,
            tone_cfg["pitch"],
            tone_cfg["volume"],
            out_path,
        )
    )
    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError("Edge TTS did not write audio.")
    return out_path


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


def _espeak_command(text: str, preferences: Optional[Dict] = None) -> list:
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

    return ["espeak-ng", "-v", voice, "-s", speed, "-p", pitch, text]


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

    subprocess.run(
        _espeak_command(text, preferences),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


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


def speak_preference_voice(text: str, preferences: Optional[Dict] = None) -> None:
    """
    Speak with Rashmi Final 1.1 Edge voices when possible.
    Falls back to espeak / Windows Speech if Edge is unavailable.
    """
    text = str(text or "").strip()
    if not text:
        return

    engine = _engine_name()
    if engine in {"espeak", "espeak-ng", "local"}:
        _speak_espeak_or_windows(text, preferences)
        return

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
    return {
        "voice": os.getenv("RASHMI_VOICE", "female"),
        "voice_type": os.getenv("RASHMI_VOICE", "female"),
        "pace": os.getenv("RASHMI_PACE", "normal"),
        "tone": os.getenv("RASHMI_TONE", "friendly"),
    }

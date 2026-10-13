"""
Laptop Rashmi voice API. Same pattern as Abhishek :8000 and Harshaka :8001.

The Pi POSTs text + prefs here, plays the MP3 on the device speaker.
OpenAI / Edge synthesis stays on the laptop so the Pi stays light.

  python tts_server.py

Pi:
  export RASHMI_TTS_URL=http://192.168.0.200:8002/rashmi/speak
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from preference_tts import _load_env, _openai_client, synthesize_speech_mp3

_load_env()

app = FastAPI(title="Rashmi TTS API", version="1.0.0")
_SYNTH_LOCK = threading.Lock()


class SpeakRequest(BaseModel):
    text: str
    voice: str = Field(default="female")
    pace: str = Field(default="normal")
    tone: str = Field(default="natural")


@app.on_event("startup")
def _warmup() -> None:
    client = _openai_client()
    if client is None:
        print("Rashmi TTS server: OpenAI not ready; Edge fallback will be used.")
    else:
        print("Rashmi TTS server: OpenAI client ready.")


@app.get("/health")
def health() -> dict:
    return {"ok": True, "service": "rashmi-tts"}


@app.post("/rashmi/speak")
def speak(req: SpeakRequest) -> Response:
    text = str(req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")

    preferences = {
        "voice": req.voice,
        "voice_type": req.voice,
        "pace": req.pace,
        "tone": req.tone,
    }
    audio_path: Path | None = None
    try:
        with _SYNTH_LOCK:
            audio_path = synthesize_speech_mp3(text, preferences)
            data = audio_path.read_bytes()
    except Exception as error:
        print("TTS failed:", error)
        raise HTTPException(status_code=502, detail=f"TTS failed: {error}") from error
    finally:
        if audio_path is not None:
            try:
                audio_path.unlink(missing_ok=True)
            except OSError:
                pass

    if not data:
        raise HTTPException(status_code=502, detail="TTS returned empty audio")
    return Response(content=data, media_type="audio/mpeg")


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("RASHMI_TTS_HOST", "0.0.0.0")
    port = int(os.getenv("RASHMI_TTS_PORT", "8002"))
    uvicorn.run(app, host=host, port=port)

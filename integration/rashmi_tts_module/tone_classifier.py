"""
Rashmi MFCC + SVM tone classifier.

Loads tts-preferences/mfcc_svm_voice_classifier.joblib and predicts
natural / friendly / calm / emotional from a short microphone sample.
This selects the tone. Edge TTS still speaks the page using that saved tone.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent
CLASSIFIER_PATH = Path(
    os.getenv(
        "RASHMI_TONE_CLASSIFIER_PATH",
        str(PROJECT_ROOT / "tts-preferences" / "mfcc_svm_voice_classifier.joblib"),
    )
)
VALID_TONES = ("natural", "friendly", "calm", "emotional")

_BUNDLE = None
_MODEL = None
_LOAD_ATTEMPTED = False


def _load_classifier():
    global _BUNDLE, _MODEL, _LOAD_ATTEMPTED
    if _LOAD_ATTEMPTED:
        return _MODEL is not None
    _LOAD_ATTEMPTED = True

    if not CLASSIFIER_PATH.exists():
        print("MFCC+SVM tone classifier not found:", CLASSIFIER_PATH)
        return False

    try:
        import joblib

        bundle = joblib.load(CLASSIFIER_PATH)
        model = bundle["model"]
        labels = [
            str(item).strip().lower()
            for item in bundle.get("labels", getattr(model, "classes_", []))
        ]
        if set(labels) != set(VALID_TONES):
            print("MFCC+SVM labels were unexpected:", labels)
            return False

        _BUNDLE = bundle
        _MODEL = model
        print("MFCC+SVM tone classifier loaded:", CLASSIFIER_PATH.name)
        print("Tone labels:", labels)
        return True
    except Exception as error:
        print("MFCC+SVM tone classifier could not be loaded:", error)
        _BUNDLE = None
        _MODEL = None
        return False


def classifier_ready() -> bool:
    return _load_classifier()


def _mean_std(values) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    if flat.size == 0:
        return np.zeros(2, dtype=np.float32)
    return np.array([float(np.mean(flat)), float(np.std(flat))], dtype=np.float32)


def _pitch_stats(y: np.ndarray, sample_rate: int, hop_length: int) -> np.ndarray:
    import librosa

    f0 = librosa.yin(
        y,
        fmin=50,
        fmax=400,
        sr=sample_rate,
        hop_length=hop_length,
    )
    voiced = f0[np.isfinite(f0) & (f0 > 0)]
    if voiced.size == 0:
        return np.zeros(5, dtype=np.float32)
    return np.array(
        [
            float(np.mean(voiced)),
            float(np.std(voiced)),
            float(np.min(voiced)),
            float(np.max(voiced)),
            float(voiced.size / max(len(f0), 1)),
        ],
        dtype=np.float32,
    )


def extract_voice_features(y: np.ndarray, sample_rate: Optional[int] = None) -> np.ndarray:
    import librosa

    if not _load_classifier() or _BUNDLE is None:
        raise RuntimeError("Voice classifier is not loaded.")

    sample_rate = int(sample_rate or _BUNDLE.get("sample_rate", 16000))
    target_duration = float(_BUNDLE.get("target_duration", 3.0))
    n_mfcc = int(_BUNDLE.get("n_mfcc", 20))
    n_fft = int(_BUNDLE.get("n_fft", 400))
    hop_length = int(_BUNDLE.get("hop_length", 160))
    n_mels = int(_BUNDLE.get("n_mels", 40))

    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if y.size == 0:
        y = np.zeros(int(sample_rate * target_duration), dtype=np.float32)

    y, _ = librosa.effects.trim(y, top_db=30)
    if y.size == 0:
        y = np.zeros(int(sample_rate * target_duration), dtype=np.float32)
    if np.max(np.abs(y)) > 0:
        y = librosa.util.normalize(y)

    target_samples = int(sample_rate * target_duration)
    if y.size > target_samples:
        y = y[:target_samples]
    else:
        y = librosa.util.fix_length(y, size=target_samples)

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sample_rate,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    rms = librosa.feature.rms(y=y, hop_length=hop_length)
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)
    centroid = librosa.feature.spectral_centroid(
        y=y, sr=sample_rate, n_fft=n_fft, hop_length=hop_length
    )
    bandwidth = librosa.feature.spectral_bandwidth(
        y=y, sr=sample_rate, n_fft=n_fft, hop_length=hop_length
    )
    rolloff = librosa.feature.spectral_rolloff(
        y=y, sr=sample_rate, n_fft=n_fft, hop_length=hop_length
    )

    features = np.concatenate(
        [
            np.mean(mfcc, axis=1),
            np.std(mfcc, axis=1),
            np.mean(delta, axis=1),
            np.std(delta, axis=1),
            np.mean(delta2, axis=1),
            np.std(delta2, axis=1),
            _mean_std(rms),
            _mean_std(zcr),
            _mean_std(centroid),
            _mean_std(bandwidth),
            _mean_std(rolloff),
            _pitch_stats(y, sample_rate, hop_length),
        ]
    ).astype(np.float32)

    expected = int(_BUNDLE.get("feature_count") or getattr(_MODEL, "n_features_in_", 0) or 0)
    if expected and features.size != expected:
        if features.size > expected:
            features = features[:expected]
        else:
            features = np.pad(features, (0, expected - features.size))

    return features


def _record_tone_sample(seconds: float = 3.0) -> np.ndarray:
    import sounddevice as sd

    if not _load_classifier() or _BUNDLE is None:
        raise RuntimeError("Voice classifier is not loaded.")

    sample_rate = int(_BUNDLE.get("sample_rate", 16000))
    device = None
    raw = os.getenv("RASHMI_WINDOWS_MIC_DEVICE", "").strip()
    if raw.isdigit():
        device = int(raw)

    print(f"Recording tone sample for {seconds:.0f} seconds...")
    audio = sd.rec(
        int(seconds * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        device=device,
    )
    sd.wait()
    return np.asarray(audio, dtype=np.float32).reshape(-1)


def classify_live_tone(seconds: float = 3.0) -> Optional[str]:
    """
    Record a short clip and predict tone with the MFCC+SVM joblib.
    """
    if not _load_classifier() or _MODEL is None:
        return None

    try:
        waveform = _record_tone_sample(seconds=seconds)
        if float(np.max(np.abs(waveform))) < 0.001:
            print("Tone sample was too quiet.")
            return None

        features = extract_voice_features(waveform).reshape(1, -1)
        predicted = str(_MODEL.predict(features)[0]).strip().lower()
        print("MFCC+SVM detected tone:", predicted)
        if predicted in VALID_TONES:
            return predicted
        print("MFCC+SVM returned an unknown tone:", predicted)
        return None
    except Exception as error:
        print("MFCC+SVM tone classification failed:", error)
        return None

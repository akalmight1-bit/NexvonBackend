"""Local TTS via Piper. Voice .onnx + .json live under models/piper/."""

from __future__ import annotations

import io
import wave
from functools import lru_cache
from pathlib import Path

from app.config import get_settings


@lru_cache
def _get_piper():
    settings = get_settings()
    model_path = Path(settings.piper_model_path)
    config_path = Path(settings.piper_config_path)

    if not model_path.is_file():
        raise RuntimeError(
            f"Piper model not found at {model_path}. "
            "Run: python scripts/download_speech_models.py"
        )
    if not config_path.is_file():
        raise RuntimeError(
            f"Piper config not found at {config_path}. "
            "Run: python scripts/download_speech_models.py"
        )

    try:
        from piper import PiperVoice
    except ImportError as e:
        raise RuntimeError(
            "piper-tts is not installed. Run: pip install piper-tts"
        ) from e

    return PiperVoice.load(str(model_path), config_path=str(config_path))


def synthesize_speech(text: str) -> bytes:
    """Synthesize text to WAV bytes (mono PCM)."""
    settings = get_settings()
    if not settings.tts_enabled:
        raise RuntimeError("TTS is disabled (TTS_ENABLED=false)")

    text = (text or "").strip()
    if not text:
        raise ValueError("Empty text")
    if len(text) > 4000:
        text = text[:4000]

    voice = _get_piper()

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        voice.synthesize(
            text,
            wf,
            speaker_id=settings.piper_speaker if settings.piper_speaker else None,
            length_scale=settings.piper_length_scale,
        )

    return buf.getvalue()

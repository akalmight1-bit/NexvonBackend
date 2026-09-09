"""Local STT via faster-whisper. Models download once into models/whisper/."""

from __future__ import annotations

import io
import tempfile
from functools import lru_cache
from pathlib import Path

from app.config import get_settings


@lru_cache
def _get_whisper():
    settings = get_settings()
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise RuntimeError(
            "faster-whisper is not installed. Run: pip install faster-whisper"
        ) from e

    root = Path(settings.whisper_download_root)
    root.mkdir(parents=True, exist_ok=True)

    return WhisperModel(
        settings.whisper_model,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
        download_root=str(root),
    )


def transcribe_audio(
    data: bytes,
    *,
    filename: str = "audio.webm",
    language: str | None = None,
) -> dict:
    """Transcribe raw audio bytes. Returns {text, language, duration}."""
    settings = get_settings()
    if not settings.stt_enabled:
        raise RuntimeError("STT is disabled (STT_ENABLED=false)")

    if not data:
        raise ValueError("Empty audio")

    # faster-whisper wants a file path or file-like with a real format
    suffix = Path(filename).suffix or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(data)
        tmp.flush()

        model = _get_whisper()
        segments, info = model.transcribe(
            tmp.name,
            language=language,
            beam_size=1,
            vad_filter=True,
        )
        parts = [seg.text.strip() for seg in segments if seg.text.strip()]
        text = " ".join(parts).strip()

    return {
        "text": text,
        "language": getattr(info, "language", language) or "unknown",
        "duration": float(getattr(info, "duration", 0) or 0),
    }

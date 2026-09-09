#!/usr/bin/env python3
"""Download local speech models (Whisper via faster-whisper cache + Piper voice)."""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PIPER_DIR = ROOT / "models" / "piper"
WHISPER_DIR = ROOT / "models" / "whisper"

# Piper lessac medium (en_US) — good default voice
PIPER_BASE = (
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
    "en/en_US/lessac/medium"
)
PIPER_FILES = {
    "en_US-lessac-medium.onnx": f"{PIPER_BASE}/en_US-lessac-medium.onnx",
    "en_US-lessac-medium.onnx.json": f"{PIPER_BASE}/en_US-lessac-medium.onnx.json",
}


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0:
        print(f"  skip (exists): {dest.name}")
        return
    print(f"  downloading {dest.name} ...")
    urllib.request.urlretrieve(url, dest)
    print(f"  ok: {dest} ({dest.stat().st_size // 1024} KB)")


def download_piper() -> None:
    print("\n[Piper TTS]")
    for name, url in PIPER_FILES.items():
        download(url, PIPER_DIR / name)


def download_whisper() -> None:
    print("\n[Whisper STT]")
    WHISPER_DIR.mkdir(parents=True, exist_ok=True)
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("  faster-whisper not installed — run: pip install faster-whisper")
        print("  Skipping Whisper download.")
        return

    model_name = "base"
    print(f"  pulling faster-whisper '{model_name}' into {WHISPER_DIR} ...")
    WhisperModel(
        model_name,
        device="cpu",
        compute_type="int8",
        download_root=str(WHISPER_DIR),
    )
    print("  ok: Whisper model cached")


def main() -> int:
    print("Nexvon local speech model installer")
    print(f"Root: {ROOT}")
    download_piper()
    download_whisper()
    print("\nDone. Start the API with: uvicorn app.main:app --reload")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# Local speech models (STT + TTS)

Nexvon voice runs **fully offline** once models are on disk.

| Role | Engine | Models dir |
|------|--------|------------|
| STT | faster-whisper | `models/whisper/` |
| TTS | Piper | `models/piper/` |

## One-time setup

```bash
# from NexvonBackend root, with venv active
pip install -r requirements.txt
python scripts/download_speech_models.py
```

This will:

1. Download Piper voice `en_US-lessac-medium` → `models/piper/`
2. Cache Whisper `base` → `models/whisper/`

## Test

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000

# STT
curl -s -F "file=@sample.wav" http://127.0.0.1:8000/v1/stt

# TTS → play with any audio player
curl -s -X POST http://127.0.0.1:8000/v1/tts \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello from Nexvon."}' \
  --output nexvon.wav
```

## GPU (optional)

In `.env`:

```env
WHISPER_DEVICE=cuda
WHISPER_COMPUTE_TYPE=float16
```

Requires a CUDA-capable GPU and the matching `faster-whisper` / PyTorch build.

## Other Piper voices

Browse: https://huggingface.co/rhasspy/piper-voices

Download `.onnx` + `.onnx.json` into `models/piper/` and point `.env`:

```env
PIPER_MODEL_PATH=models/piper/your-voice.onnx
PIPER_CONFIG_PATH=models/piper/your-voice.onnx.json
```

## Disable speech

```env
STT_ENABLED=false
TTS_ENABLED=false
```

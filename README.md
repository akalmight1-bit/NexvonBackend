# NexvonBackend

Multi-provider orchestrator + API for **Nexvon**.

Chat via local Ollama, NVIDIA, or xAI. Voice via **local** Whisper (STT) and Piper (TTS) — models download once, no cloud required for speech.

## Features

| API | Purpose |
|-----|---------|
| `POST /v1/chat` | Streaming chat (SSE) |
| `POST /v1/stt` | Local speech → text |
| `POST /v1/tts` | Local text → WAV |
| `GET /v1/models` | Providers + speech config |
| `GET /health` | Liveness |

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env

# Optional: local chat models
cd hosting/ollama && docker compose up -d && cd ../..

# Download local voice models (Whisper + Piper)
python scripts/download_speech_models.py

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Local voice (offline)

After `python scripts/download_speech_models.py`:

```bash
# Speech → text
curl -s -F "file=@recording.webm" http://127.0.0.1:8000/v1/stt

# Text → speech (WAV)
curl -s -X POST http://127.0.0.1:8000/v1/tts \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello from Nexvon."}' \
  --output nexvon.wav
```

Models live under `models/whisper/` and `models/piper/`. Details: `hosting/speech/README.md`.

## Chat API

```json
POST /v1/chat
{
  "messages": [{ "role": "user", "content": "Hello" }],
  "model": "auto",
  "stream": true
}
```

`model`: `auto` | `xai` | `nvidia` | `nvidia/<id>` | `ollama` | `ollama/<name>` | `local`

SSE:

```text
data: {"text":"..."}
data: [DONE]
```

## Point NexvonUI here

```env
VITE_API_URL=http://127.0.0.1:8000
```

Use `${VITE_API_URL}/v1/chat`, `/v1/stt`, `/v1/tts`.

## Layout

```text
app/                 FastAPI + orchestrator + providers
app/speech/          Local Whisper STT + Piper TTS
models/              Downloaded voice models (gitignored)
hosting/ollama/      Docker for local LLMs
hosting/speech/      Voice setup notes
scripts/             download_speech_models.py
orchestrator.py      CLI multi-model router (Ollama)
```

## Env

See `.env.example` for all keys (chat providers + `STT_*` / `TTS_*` / `WHISPER_*` / `PIPER_*`).

# NexvonBackend

Multi-provider orchestrator + API for **Nexvon**.

Routes chat through local Ollama, NVIDIA NIM, or xAI — same streaming contract for the frontend. Includes self-host backup so you can run models yourself when needed.

## What this is

| Piece | Role |
|-------|------|
| `app/` | FastAPI HTTP API (SSE streaming) |
| `orchestrator.py` | Your multi-model router (tools / chat / vision) |
| `providers/` | xAI · NVIDIA · Ollama · OpenAI-compatible |
| `hosting/` | Docker Compose to run Ollama locally as backup |

## Quick start

```bash
# 1. Python env
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Config
cp .env.example .env
# edit .env — at least set one of: XAI_API_KEY, NVIDIA_API_KEY, or run Ollama

# 3. (Optional) local models
cd hosting/ollama && docker compose up -d
ollama pull llama3.2:3b

# 4. Run API
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Health check: `GET http://127.0.0.1:8000/health`

## API

### `POST /v1/chat` (streaming SSE)

```json
{
  "messages": [
    { "role": "user", "content": "Hello" }
  ],
  "model": "auto",
  "stream": true
}
```

`model` options:
- `auto` — uses `DEFAULT_PROVIDER`
- `xai` / `grok` — xAI Grok
- `nvidia` or `nvidia/<model-id>` — NVIDIA API
- `ollama` or `ollama/<model>` — local Ollama
- `local` — alias for default local model

SSE response (same as NexvonUI expects):

```text
data: {"text":"Hello"}
data: {"text":" world"}
data: [DONE]
```

### `GET /v1/models`

Lists configured providers and default models.

### `GET /health`

Liveness probe.

## Point NexvonUI at this backend

In the UI repo, set:

```env
VITE_API_URL=http://127.0.0.1:8000
```

and call `${VITE_API_URL}/v1/chat` instead of `/api/chat`.

## Self-host backup

When cloud APIs are down or you want private inference:

```bash
cd hosting/ollama
docker compose up -d
ollama pull llama3.2:3b
```

Then in `.env`:

```env
DEFAULT_PROVIDER=ollama
FALLBACK_PROVIDER=ollama
OLLAMA_BASE_URL=http://127.0.0.1:11434
```

## Env vars

See `.env.example`.

| Variable | Purpose |
|----------|---------|
| `XAI_API_KEY` | xAI Grok |
| `NVIDIA_API_KEY` | NVIDIA NIM / API Catalog |
| `NVIDIA_BASE_URL` | Default `https://integrate.api.nvidia.com/v1` |
| `OLLAMA_BASE_URL` | Default `http://127.0.0.1:11434` |
| `DEFAULT_PROVIDER` | `xai` \| `nvidia` \| `ollama` |
| `FALLBACK_PROVIDER` | Tried once if primary fails |
| `CORS_ORIGINS` | Comma-separated frontend origins |

## CLI orchestrator (your original script)

```bash
python orchestrator.py
```

Routes naturally between coder tools, chat, and vision via local Ollama models.

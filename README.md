# NexvonBackend

API for **Nexvon**: chat orchestration, dual web search (Brave + Serper), file/image handling, and RAG.

Point NexvonUI at this service with `NEXVON_API_URL` (server-only). The UI always talks to its own `/api/*` routes; those hop here so CORS and secrets stay off the browser.

## Features

| API | Purpose |
|-----|---------|
| `POST /v1/chat` | Streaming chat (SSE). Accepts `provider` or `model`, multimodal content, optional `knowledge`. |
| `GET /v1/chat` / `GET /v1/models` | Providers + search/RAG status (UI-compatible). |
| `POST /v1/search` | Brave + Serper, merged. |
| `POST /v1/files` | Images (vision data URLs) and text extraction. |
| `POST /v1/rag/ingest` | Add a document to the knowledge store. |
| `POST /v1/rag/query` | Retrieve relevant chunks (BM25). |
| `GET /v1/rag/docs` | List ingested documents. |
| `POST /v1/stt` / `POST /v1/tts` | Local Whisper + Piper. |
| `GET /health` | Liveness. |

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Docker:

```bash
docker build -t nexvon-backend .
docker run --env-file .env -p 8000:8000 nexvon-backend
```

## Connect NexvonUI

On the **UI server** (not `VITE_`):

```env
NEXVON_API_URL=https://your-backend.example.com
```

Same-origin `/api/chat`, `/api/search`, `/api/files`, and `/api/rag` will proxy here. CORS also allows `*.grok.me`, `*.vercel.app`, and `*.netlify.app`.

Add your UI origin to `CORS_ORIGINS` if it is a custom domain.

## Search

Set one or both:

- `BRAVE_API_KEY` — [Brave Search API](https://brave.com/search/api/)
- `SERPER_API_KEY` — [Serper](https://serper.dev)

When both are set they run in parallel and results are merged. `SEARCH_PROVIDER=brave|serper|auto`.

## Files

`POST /v1/files` (multipart). Images become `data:` URLs for vision models. Text-like files are inlined. Binary files are metadata only.

## RAG

Lexical BM25 over ingested documents (stored under `data/rag/`, gitignored). Optional OpenAI-compatible embeddings via `EMBEDDING_*` later — retrieval works without them.

```bash
curl -s -X POST http://127.0.0.1:8000/v1/rag/ingest \
  -H "Content-Type: application/json" \
  -d '{"name":"notes.md","text":"# Project\nNexvon uses dual search."}'
```

Chat turns automatically retrieve relevant chunks.

## Env

See `.env.example`.

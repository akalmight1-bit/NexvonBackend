from __future__ import annotations

import json
from typing import Any, Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from app.config import get_settings
from app.orchestrator import stream_chat
from app.providers import ChatMessage, list_providers
from app.speech import synthesize_speech, transcribe_audio
from app.tools.files import process_bytes
from app.tools.rag import delete_doc, ingest, list_docs, retrieve, format_rag_context
from app.tools.search import search_web

app = FastAPI(title="NexvonBackend", version="0.4.0")

settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.origin_list or ["*"],
    allow_origin_regex=r"https://.*\.(grok\.me|vercel\.app|netlify\.app)$",
    allow_credentials="*" not in (settings.origin_list or []),
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(default_factory=list)
    model: str | None = "auto"
    provider: str | None = None
    stream: bool = True
    use_search: bool | Literal["auto"] | None = "auto"
    knowledge: list[dict[str, Any]] | None = None


class TtsRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    num: int = Field(default=6, ge=1, le=10)


class RagIngestRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    text: str = Field(..., min_length=1)
    mimeType: str | None = "text/plain"


class RagQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    k: int | None = None
    documents: list[dict[str, Any]] | None = None


def _status_payload() -> dict[str, Any]:
    s = get_settings()
    return {
        "status": "ok",
        "service": "nexvon-backend",
        "stt": s.stt_enabled,
        "tts": s.tts_enabled,
        "search": bool(s.search_engines) and s.search_enabled,
        "files": True,
        "rag": s.rag_enabled,
    }


@app.get("/health")
async def health() -> dict[str, Any]:
    return _status_payload()


@app.get("/v1/models")
@app.get("/v1/chat")
async def models() -> dict[str, Any]:
    s = get_settings()
    providers = list_providers()
    return {
        "default_provider": s.default_provider,
        "fallback_provider": s.fallback_provider,
        "providers": [
            {
                "id": p["id"],
                "label": p["id"],
                "model": p.get("default_model") or "",
                "default_model": p.get("default_model") or "",
                "configured": p.get("configured", True),
            }
            for p in providers
        ],
        "search": {
            "enabled": s.search_enabled and bool(s.search_engines),
            "engines": s.search_engines,
            "mode": s.search_mode,
            "num_results": s.search_num_results,
        },
        "files": True,
        "rag": {"enabled": s.rag_enabled, "top_k": s.rag_top_k},
        "speech": {
            "stt": {
                "enabled": s.stt_enabled,
                "engine": "faster-whisper",
                "model": s.whisper_model,
                "device": s.whisper_device,
            },
            "tts": {
                "enabled": s.tts_enabled,
                "engine": "piper",
                "model_path": s.piper_model_path,
            },
        },
    }


@app.post("/v1/chat")
async def chat(body: ChatRequest):
    if not body.messages:
        raise HTTPException(status_code=400, detail="Send a message first.")

    cleaned: list[ChatMessage] = []
    for m in body.messages:
        if m.role not in ("user", "assistant", "system"):
            continue
        text = m.text()
        images = m.image_urls()
        if not text.strip() and not images:
            continue
        cleaned.append(m)

    if not cleaned or cleaned[-1].role != "user":
        raise HTTPException(status_code=400, detail="Last message must be from the user.")

    model = body.provider or body.model or "auto"

    async def event_stream():
        try:
            async for text in stream_chat(
                cleaned,
                model=model,
                use_search=body.use_search,
                knowledge=body.knowledge,
            ):
                yield f"data: {json.dumps({'text': text})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/v1/search")
async def search(body: SearchRequest):
    try:
        return await search_web(body.query, num=body.num)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {e}") from e


@app.post("/v1/files")
async def upload_files(files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")
    out = []
    for f in files[:6]:
        data = await f.read()
        try:
            out.append(process_bytes(f.filename or "upload", f.content_type or "application/octet-stream", data))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
    return {"files": out}


@app.post("/v1/rag/ingest")
async def rag_ingest(body: RagIngestRequest):
    try:
        return ingest(body.name, body.text, body.mimeType or "text/plain")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/v1/rag/query")
async def rag_query(body: RagQueryRequest):
    hits = retrieve(body.query, k=body.k, documents=body.documents)
    return {"query": body.query, "results": hits, "context": format_rag_context(hits)}


@app.get("/v1/rag/docs")
async def rag_docs():
    return {"documents": list_docs()}


@app.delete("/v1/rag/docs/{doc_id}")
async def rag_delete(doc_id: str):
    if not delete_doc(doc_id):
        raise HTTPException(status_code=404, detail="Document not found")
    return {"ok": True}


@app.post("/v1/stt")
async def stt(file: UploadFile = File(...), language: str | None = None):
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty audio file")
    if len(data) > 25 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Audio too large (max 25MB)")

    try:
        result = transcribe_audio(
            data,
            filename=file.filename or "audio.webm",
            language=language or None,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"STT failed: {e}") from e

    return result


@app.post("/v1/tts")
async def tts(body: TtsRequest):
    try:
        wav = synthesize_speech(body.text)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS failed: {e}") from e

    return Response(
        content=wav,
        media_type="audio/wav",
        headers={"Content-Disposition": 'inline; filename="nexvon.wav"'},
    )

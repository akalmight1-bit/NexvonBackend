from __future__ import annotations

import json
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from app.config import get_settings
from app.orchestrator import stream_chat
from app.providers import ChatMessage, list_providers
from app.speech import synthesize_speech, transcribe_audio

app = FastAPI(title="NexvonBackend", version="0.2.0")

settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.origin_list or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(default_factory=list)
    model: str | None = "auto"
    stream: bool = True


class TtsRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000)


@app.get("/health")
async def health() -> dict[str, Any]:
    s = get_settings()
    return {
        "status": "ok",
        "service": "nexvon-backend",
        "stt": s.stt_enabled,
        "tts": s.tts_enabled,
    }


@app.get("/v1/models")
async def models() -> dict[str, Any]:
    s = get_settings()
    return {
        "default_provider": s.default_provider,
        "fallback_provider": s.fallback_provider,
        "providers": list_providers(),
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
        content = m.content.strip()
        if not content:
            continue
        cleaned.append(ChatMessage(role=m.role, content=content[:8000]))

    if not cleaned or cleaned[-1].role != "user":
        raise HTTPException(status_code=400, detail="Last message must be from the user.")

    async def event_stream():
        try:
            async for text in stream_chat(cleaned, model=body.model):
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


@app.post("/v1/stt")
async def stt(file: UploadFile = File(...), language: str | None = None):
    """Local speech-to-text. Upload audio (webm/wav/mp3/ogg)."""
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
    """Local text-to-speech. Returns WAV audio."""
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

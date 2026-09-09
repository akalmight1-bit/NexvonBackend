from __future__ import annotations

import json
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.config import get_settings
from app.orchestrator import stream_chat
from app.providers import ChatMessage, list_providers

app = FastAPI(title="NexvonBackend", version="0.1.0")

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


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok", "service": "nexvon-backend"}


@app.get("/v1/models")
async def models() -> dict[str, Any]:
    s = get_settings()
    return {
        "default_provider": s.default_provider,
        "fallback_provider": s.fallback_provider,
        "providers": list_providers(),
    }


@app.post("/v1/chat")
async def chat(body: ChatRequest):
    if not body.messages:
        raise HTTPException(status_code=400, detail="Send a message first.")

    # Validate roles lightly
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

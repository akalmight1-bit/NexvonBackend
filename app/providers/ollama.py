"""Local Ollama provider — primary self-host / backup path."""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx

from app.config import Settings
from app.providers.base import ChatMessage


class OllamaProvider:
    def __init__(self, settings: Settings):
        self.id = "ollama"
        self.base_url = settings.ollama_base_url.rstrip("/")
        self.default_model = settings.ollama_model

    async def stream(
        self,
        messages: list[ChatMessage],
        model: str,
        *,
        system: str | None = None,
    ) -> AsyncIterator[str]:
        model = model or self.default_model
        payload_messages = []
        if system:
            payload_messages.append({"role": "system", "content": system})
        payload_messages.extend({"role": m.role, "content": m.content} for m in messages)

        body = {
            "model": model,
            "messages": payload_messages,
            "stream": True,
        }

        url = f"{self.base_url}/api/chat"
        async with httpx.AsyncClient(timeout=httpx.Timeout(180.0, connect=10.0)) as client:
            async with client.stream("POST", url, json=body) as resp:
                if resp.status_code >= 400:
                    detail = (await resp.aread()).decode("utf-8", errors="replace")[:500]
                    raise RuntimeError(f"[ollama] HTTP {resp.status_code}: {detail}")

                import json

                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    msg = obj.get("message") or {}
                    text = msg.get("content") or ""
                    if text:
                        yield text
                    if obj.get("done"):
                        break


def make_ollama(settings: Settings) -> OllamaProvider:
    return OllamaProvider(settings)

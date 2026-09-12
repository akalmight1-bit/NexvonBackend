"""OpenAI-compatible streaming client — used by NVIDIA, xAI, local vLLM/TGI, etc."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx

from app.providers.base import ChatMessage


class OpenAICompatProvider:
    def __init__(self, *, provider_id: str, base_url: str, api_key: str = "", default_model: str = ""):
        self.id = provider_id
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.default_model = default_model

    def _payload_content(self, message: ChatMessage) -> Any:
        if isinstance(message.content, str):
            return message.content
        return message.content

    async def stream(
        self,
        messages: list[ChatMessage],
        model: str,
        *,
        system: str | None = None,
    ) -> AsyncIterator[str]:
        model = model or self.default_model
        if not model:
            raise RuntimeError(f"[{self.id}] No model specified")

        payload_messages: list[dict[str, Any]] = []
        if system:
            payload_messages.append({"role": "system", "content": system})
        payload_messages.extend({"role": m.role, "content": self._payload_content(m)} for m in messages)

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        body = {
            "model": model,
            "messages": payload_messages,
            "stream": True,
            "max_tokens": 2048,
            "temperature": 0.7,
        }

        url = f"{self.base_url}/chat/completions"
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=15.0)) as client:
            async with client.stream("POST", url, headers=headers, json=body) as resp:
                if resp.status_code >= 400:
                    detail = (await resp.aread()).decode("utf-8", errors="replace")[:500]
                    raise RuntimeError(f"[{self.id}] HTTP {resp.status_code}: {detail}")

                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    if line.startswith("data:"):
                        data = line[5:].strip()
                    else:
                        data = line.strip()
                    if not data or data == "[DONE]":
                        continue
                    try:
                        import json

                        obj = json.loads(data)
                    except Exception:
                        continue
                    choices = obj.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    text = delta.get("content") or ""
                    if text:
                        yield text

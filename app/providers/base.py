from collections.abc import AsyncIterator
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field


Role = Literal["system", "user", "assistant"]
ContentPart = dict[str, Any]
MessageContent = str | list[ContentPart]


class ChatMessage(BaseModel):
    role: Role
    content: MessageContent = Field(...)

    def text(self) -> str:
        if isinstance(self.content, str):
            return self.content
        bits: list[str] = []
        for part in self.content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text" and isinstance(part.get("text"), str):
                bits.append(part["text"])
        return "\n".join(bits)

    def image_urls(self) -> list[str]:
        if isinstance(self.content, str):
            return []
        urls: list[str] = []
        for part in self.content:
            if not isinstance(part, dict):
                continue
            if part.get("type") != "image_url":
                continue
            image = part.get("image_url") or {}
            url = image.get("url") if isinstance(image, dict) else None
            if isinstance(url, str) and url.startswith("data:image/"):
                urls.append(url)
        return urls


class Provider(Protocol):
    id: str

    async def stream(
        self,
        messages: list[ChatMessage],
        model: str,
        *,
        system: str | None = None,
    ) -> AsyncIterator[str]:
        """Yield text deltas."""
        ...

from collections.abc import AsyncIterator
from typing import Literal, Protocol

from pydantic import BaseModel, Field


Role = Literal["system", "user", "assistant"]


class ChatMessage(BaseModel):
    role: Role
    content: str = Field(min_length=1)


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

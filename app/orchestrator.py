"""HTTP-facing orchestrator: pick provider, stream, optional failover."""

from __future__ import annotations

from collections.abc import AsyncIterator

from app.config import get_settings
from app.providers import ChatMessage, get_provider, resolve_model


async def stream_chat(
    messages: list[ChatMessage],
    *,
    model: str | None = None,
) -> AsyncIterator[str]:
    settings = get_settings()
    provider_id, model_name = resolve_model(model)

    # Keep only user/assistant for providers; system is injected separately
    history = [m for m in messages if m.role in ("user", "assistant")]
    if not history:
        raise ValueError("Send at least one user message")
    if history[-1].role != "user":
        raise ValueError("Last message must be from the user")

    # Cap context a bit
    history = history[-32:]

    async def _run(pid: str, mname: str) -> AsyncIterator[str]:
        provider = get_provider(pid)
        async for chunk in provider.stream(history, mname, system=settings.system_prompt):
            yield chunk

    try:
        async for chunk in _run(provider_id, model_name):
            yield chunk
        return
    except Exception as primary_err:
        fallback = settings.fallback_provider
        if not fallback or fallback == provider_id:
            raise primary_err
        # one-shot failover
        fb_model = {
            "xai": settings.xai_model,
            "nvidia": settings.nvidia_model,
            "ollama": settings.ollama_model,
            "local": settings.local_model,
        }.get(fallback, settings.ollama_model)
        async for chunk in _run(fallback, fb_model):
            yield chunk

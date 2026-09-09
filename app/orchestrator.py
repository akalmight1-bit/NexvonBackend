"""HTTP-facing orchestrator: optional Serper search, then stream from a provider."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Literal

from app.config import get_settings
from app.providers import ChatMessage, get_provider, resolve_model
from app.tools.serper import format_search_context, needs_web_search, search_web

SearchMode = Literal["auto", "always", "never", "true", "false"]


def _should_search(last_user: str, use_search: bool | str | None) -> bool:
    settings = get_settings()
    if not settings.search_enabled or not settings.serper_api_key:
        return False

    if use_search is True or use_search in ("true", "always", "1", "yes"):
        return True
    if use_search is False or use_search in ("false", "never", "0", "no"):
        return False

    # respect global default
    mode = (settings.search_mode or "auto").lower()
    if mode == "always":
        return True
    if mode == "never":
        return False
    return needs_web_search(last_user)


async def stream_chat(
    messages: list[ChatMessage],
    *,
    model: str | None = None,
    use_search: bool | str | None = None,
) -> AsyncIterator[str]:
    settings = get_settings()
    provider_id, model_name = resolve_model(model)

    history = [m for m in messages if m.role in ("user", "assistant")]
    if not history:
        raise ValueError("Send at least one user message")
    if history[-1].role != "user":
        raise ValueError("Last message must be from the user")

    history = history[-32:]
    last_user = history[-1].content

    system = settings.system_prompt

    if _should_search(last_user, use_search):
        try:
            results = await search_web(last_user)
            ctx = format_search_context(results)
            system = (
                f"{settings.system_prompt}\n\n"
                "You have live web search results below. Prefer them for "
                "current facts, news, prices, and anything time-sensitive. "
                "If results conflict with your prior knowledge, trust the results "
                "and mention the source when it matters.\n\n"
                f"{ctx}"
            )
        except Exception as e:
            # Search failed — still answer without web context
            system = (
                f"{settings.system_prompt}\n\n"
                f"(Web search was attempted but failed: {e}. Answer from knowledge.)"
            )

    async def _run(pid: str, mname: str) -> AsyncIterator[str]:
        provider = get_provider(pid)
        async for chunk in provider.stream(history, mname, system=system):
            yield chunk

    try:
        async for chunk in _run(provider_id, model_name):
            yield chunk
        return
    except Exception as primary_err:
        fallback = settings.fallback_provider
        if not fallback or fallback == provider_id:
            raise primary_err
        fb_model = {
            "xai": settings.xai_model,
            "nvidia": settings.nvidia_model,
            "ollama": settings.ollama_model,
            "local": settings.local_model,
        }.get(fallback, settings.ollama_model)
        async for chunk in _run(fallback, fb_model):
            yield chunk

"""HTTP-facing orchestrator: search + RAG, then stream from a provider."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from app.config import get_settings
from app.providers import ChatMessage, get_provider, resolve_model
from app.tools.rag import format_rag_context, retrieve
from app.tools.search import format_search_context, needs_web_search, search_web


def _should_search(last_user: str, use_search: bool | str | None) -> bool:
    settings = get_settings()
    if not settings.search_enabled or not settings.search_engines:
        return False

    if use_search is True or use_search in ("true", "always", "1", "yes"):
        return True
    if use_search is False or use_search in ("false", "never", "0", "no"):
        return False

    mode = (settings.search_mode or "auto").lower()
    if mode == "always":
        return True
    if mode == "never":
        return False
    return needs_web_search(last_user)


def _last_user_text(history: list[ChatMessage]) -> str:
    last = history[-1]
    return last.text() if hasattr(last, "text") else str(last.content)


async def stream_chat(
    messages: list[ChatMessage],
    *,
    model: str | None = None,
    use_search: bool | str | None = None,
    knowledge: list[dict[str, Any]] | None = None,
) -> AsyncIterator[str]:
    settings = get_settings()
    provider_id, model_name = resolve_model(model)

    history = [m for m in messages if m.role in ("user", "assistant")]
    if not history:
        raise ValueError("Send at least one user message")
    if history[-1].role != "user":
        raise ValueError("Last message must be from the user")

    history = history[-32:]
    last_user = _last_user_text(history)
    system = settings.system_prompt

    if _should_search(last_user, use_search):
        try:
            results = await search_web(last_user)
            ctx = format_search_context(results)
            system = (
                f"{system}\n\n"
                "You have live web search results below. Prefer them for "
                "current facts, news, prices, and anything time-sensitive. "
                "If results conflict with your prior knowledge, trust the results "
                "and mention the source when it matters.\n\n"
                f"{ctx}"
            )
        except Exception as e:
            system = f"{system}\n\n(Web search was attempted but failed: {e}. Answer from knowledge.)"

    if settings.rag_enabled:
        try:
            hits = retrieve(last_user, documents=knowledge if knowledge else None)
            rag = format_rag_context(hits)
            if rag:
                system = f"{system}\n\n{rag}"
        except Exception:
            pass

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

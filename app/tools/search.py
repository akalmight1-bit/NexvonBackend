"""Unified web search: Brave + Serper, in parallel, merged by URL."""

from __future__ import annotations

import asyncio
import re
from typing import Any
from urllib.parse import urlparse

from app.config import get_settings
from app.tools.brave import brave_search
from app.tools.serper import search_web as serper_search

_WEB_HINTS = re.compile(
    r"\b("
    r"who is|what is|what are|when did|when is|where is|how many|"
    r"latest|recent|today|yesterday|this week|this year|current|"
    r"news|price|stock|weather|score|release date|"
    r"search|look up|find out|google|according to|"
    r"as of|right now|live|trending"
    r")\b",
    re.IGNORECASE,
)


def needs_web_search(text: str) -> bool:
    t = (text or "").strip()
    if len(t) < 8:
        return False
    if _WEB_HINTS.search(t):
        return True
    if "?" in t and len(t.split()) <= 20:
        return True
    return False


def _norm_url(url: str) -> str:
    try:
        u = urlparse(url)
        host = (u.hostname or "").removeprefix("www.")
        path = (u.path or "").rstrip("/")
        return f"{host}{path}"
    except Exception:
        return url


def _merge(groups: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for group in groups:
        for item in group:
            key = _norm_url(item.get("link") or "") or f"{item.get('title')}:{item.get('snippet', '')[:40]}"
            if key in seen:
                continue
            seen.add(key)
            out.append(item)
    return out


async def search_web(query: str, *, num: int | None = None) -> dict[str, Any]:
    settings = get_settings()
    if not settings.search_enabled:
        raise RuntimeError("Search is disabled (SEARCH_ENABLED=false)")
    query = (query or "").strip()
    if not query:
        raise ValueError("Empty search query")

    n = num if num is not None else settings.search_num_results
    n = max(1, min(n, 10))
    preferred = (settings.search_provider or "auto").lower()

    tasks: list[asyncio.Task] = []
    engines: list[str] = []

    async def _brave() -> list[dict[str, Any]]:
        return await brave_search(query, num=n)

    async def _serper() -> list[dict[str, Any]]:
        raw = await serper_search(query, num=n)
        hits = []
        for item in raw.get("organic") or []:
            hits.append(
                {
                    "title": item.get("title") or "",
                    "link": item.get("link") or "",
                    "snippet": item.get("snippet") or "",
                    "engine": "serper",
                }
            )
        ab = raw.get("answer_box")
        if ab:
            hits.insert(
                0,
                {
                    "title": ab.get("title") or "Answer",
                    "link": ab.get("link") or "",
                    "snippet": ab.get("answer") or "",
                    "engine": "serper",
                    "source": "answer box",
                },
            )
        return hits

    use_brave = bool(settings.brave_api_key) and preferred != "serper"
    use_serper = bool(settings.serper_api_key) and preferred != "brave"
    if not use_brave and not use_serper:
        raise RuntimeError("No search API key. Set BRAVE_API_KEY and/or SERPER_API_KEY.")

    if use_brave:
        engines.append("brave")
        tasks.append(asyncio.create_task(_brave()))
    if use_serper:
        engines.append("serper")
        tasks.append(asyncio.create_task(_serper()))

    results = await asyncio.gather(*tasks, return_exceptions=True)
    groups: list[list[dict[str, Any]]] = []
    errors: list[str] = []
    for r in results:
        if isinstance(r, Exception):
            errors.append(str(r))
        else:
            groups.append(r)

    organic = _merge(groups)[:12]
    if not organic and errors:
        raise RuntimeError("; ".join(errors))

    return {
        "query": query,
        "organic": organic,
        "engines": engines,
        "errors": errors,
    }


def format_search_context(results: dict[str, Any], *, max_chars: int = 3500) -> str:
    lines: list[str] = [
        "Web search results (use these facts; cite links when relevant):",
        f"Query: {results.get('query', '')}",
        "",
    ]
    for i, item in enumerate(results.get("organic") or [], 1):
        title = item.get("title") or "Untitled"
        snippet = item.get("snippet") or ""
        link = item.get("link") or ""
        engine = item.get("engine") or ""
        extra = f" · {engine}" if engine else ""
        lines.append(f"{i}. {title}{extra}")
        if snippet:
            lines.append(f"   {snippet}")
        if link:
            lines.append(f"   {link}")
        lines.append("")
    text = "\n".join(lines).strip()
    if len(text) > max_chars:
        text = text[: max_chars - 20] + "\n…[truncated]"
    return text

"""Serper.dev Google Search — live web results for the orchestrator."""

from __future__ import annotations

import re
from typing import Any

import httpx

from app.config import get_settings

# Heuristics: questions that usually need current / external facts
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
    """Cheap classifier: does this user message likely need the internet?"""
    t = (text or "").strip()
    if len(t) < 8:
        return False
    if _WEB_HINTS.search(t):
        return True
    # Question mark + short factual phrasing
    if "?" in t and len(t.split()) <= 20:
        return True
    return False


async def search_web(
    query: str,
    *,
    num: int | None = None,
    gl: str = "us",
    hl: str = "en",
) -> dict[str, Any]:
    """Call Serper Google Search API. Returns normalized results."""
    settings = get_settings()
    if not settings.serper_api_key:
        raise RuntimeError(
            "SERPER_API_KEY is not set. Get a key at https://serper.dev and add it to .env"
        )
    if not settings.search_enabled:
        raise RuntimeError("Search is disabled (SEARCH_ENABLED=false)")

    query = (query or "").strip()
    if not query:
        raise ValueError("Empty search query")

    n = num if num is not None else settings.search_num_results
    n = max(1, min(n, 10))

    url = f"{settings.serper_base_url.rstrip('/')}/search"
    headers = {
        "X-API-KEY": settings.serper_api_key,
        "Content-Type": "application/json",
    }
    body = {"q": query, "num": n, "gl": gl, "hl": hl}

    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.post(url, headers=headers, json=body)
        if resp.status_code >= 400:
            detail = resp.text[:400]
            raise RuntimeError(f"Serper HTTP {resp.status_code}: {detail}")
        data = resp.json()

    organic = []
    for item in data.get("organic") or []:
        organic.append(
            {
                "title": item.get("title") or "",
                "link": item.get("link") or "",
                "snippet": item.get("snippet") or "",
                "date": item.get("date") or "",
            }
        )

    answer_box = data.get("answerBox") or {}
    knowledge = data.get("knowledgeGraph") or {}

    return {
        "query": query,
        "organic": organic,
        "answer_box": {
            "title": answer_box.get("title") or answer_box.get("snippet") or "",
            "answer": answer_box.get("answer") or answer_box.get("snippet") or "",
            "link": answer_box.get("link") or "",
        }
        if answer_box
        else None,
        "knowledge_graph": {
            "title": knowledge.get("title") or "",
            "type": knowledge.get("type") or "",
            "description": knowledge.get("description") or "",
            "description_source": (knowledge.get("descriptionSource") or ""),
        }
        if knowledge
        else None,
        "raw_keys": list(data.keys()),
    }


def format_search_context(results: dict[str, Any], *,
 max_chars: int = 3500) -> str:
    """Turn Serper results into a compact block the LLM can cite."""
    lines: list[str] = [
        "Web search results (use these facts; cite links when relevant):",
        f"Query: {results.get('query', '')}",
        "",
    ]

    ab = results.get("answer_box")
    if ab and (ab.get("answer") or ab.get("title")):
        lines.append(f"Answer box: {ab.get('answer') or ab.get('title')}")
        if ab.get("link"):
            lines.append(f"  Source: {ab['link']}")
        lines.append("")

    kg = results.get("knowledge_graph")
    if kg and kg.get("title"):
        lines.append(f"Knowledge: {kg['title']} — {kg.get('description', '')}")
        lines.append("")

    for i, item in enumerate(results.get("organic") or [], 1):
        title = item.get("title") or "Untitled"
        snippet = item.get("snippet") or ""
        link = item.get("link") or ""
        date = item.get("date") or ""
        extra = f" ({date})" if date else ""
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

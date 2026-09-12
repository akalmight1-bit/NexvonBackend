"""Brave Search API — live web results."""

from __future__ import annotations

from typing import Any

import httpx

from app.config import get_settings


async def brave_search(query: str, *, num: int = 6) -> list[dict[str, Any]]:
    settings = get_settings()
    if not settings.brave_api_key:
        raise RuntimeError("BRAVE_API_KEY is not set. Get a key at https://brave.com/search/api/")

    n = max(1, min(num, 10))
    url = f"{settings.brave_base_url.rstrip('/')}/web/search"
    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": settings.brave_api_key,
    }
    params = {"q": query, "count": n, "text_decorations": "false"}

    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.get(url, headers=headers, params=params)
        if resp.status_code >= 400:
            raise RuntimeError(f"Brave HTTP {resp.status_code}: {resp.text[:400]}")
        data = resp.json()

    hits: list[dict[str, Any]] = []
    for item in (data.get("infobox") or {}).get("results") or []:
        hits.append(
            {
                "title": item.get("title") or "",
                "link": item.get("url") or "",
                "snippet": item.get("description") or "",
                "engine": "brave",
                "source": "infobox",
            }
        )
    for item in (data.get("web") or {}).get("results") or []:
        hits.append(
            {
                "title": item.get("title") or "",
                "link": item.get("url") or "",
                "snippet": item.get("description") or "",
                "engine": "brave",
            }
        )
    return hits

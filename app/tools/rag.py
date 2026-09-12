"""Local RAG store: chunk documents, lexical retrieval, optional embeddings."""

from __future__ import annotations

import json
import math
import re
import uuid
from pathlib import Path
from typing import Any

from app.config import get_settings

STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "it", "as",
    "at", "by", "be", "this", "that", "with", "from", "are", "was", "were", "but",
}
TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if len(t) > 1 and t not in STOP]


def chunk_text(text: str, max_chars: int = 900, overlap: int = 120) -> list[str]:
    clean = text.replace("\r\n", "\n").strip()
    if not clean:
        return []
    if len(clean) <= max_chars:
        return [clean]
    paragraphs = re.split(r"\n{2,}", clean)
    chunks: list[str] = []
    buf = ""

    def flush() -> None:
        nonlocal buf
        t = buf.strip()
        if t:
            chunks.append(t)
        buf = t[max(0, len(t) - overlap) :]

    for p in paragraphs:
        if buf and len(buf) + 2 + len(p) > max_chars:
            flush()
        if len(p) > max_chars:
            for w in p.split():
                if buf and len(buf) + 1 + len(w) > max_chars:
                    flush()
                buf = f"{buf} {w}".strip()
        else:
            buf = f"{buf}\n\n{p}".strip() if buf else p
    flush()
    return [c for c in chunks if c]


def _store_path() -> Path:
    settings = get_settings()
    path = Path(settings.rag_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path / "docs.json"


def _load() -> list[dict[str, Any]]:
    p = _store_path()
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save(docs: list[dict[str, Any]]) -> None:
    p = _store_path()
    p.write_text(json.dumps(docs, ensure_ascii=False, indent=2), encoding="utf-8")


def ingest(name: str, text: str, mime_type: str = "text/plain") -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise ValueError("Nothing to ingest")
    chunks = chunk_text(text[:200_000])
    doc = {
        "id": str(uuid.uuid4()),
        "name": (name or "Document")[:120],
        "mimeType": mime_type or "text/plain",
        "text": text[:200_000],
        "chunks": chunks,
        "size": len(text),
    }
    docs = _load()
    docs.insert(0, doc)
    _save(docs[:200])
    return {"id": doc["id"], "name": doc["name"], "chunkCount": len(chunks), "size": doc["size"]}


def list_docs() -> list[dict[str, Any]]:
    return [
        {"id": d["id"], "name": d.get("name"), "chunkCount": len(d.get("chunks") or []), "size": d.get("size")}
        for d in _load()
    ]


def delete_doc(doc_id: str) -> bool:
    docs = _load()
    next_docs = [d for d in docs if d.get("id") != doc_id]
    if len(next_docs) == len(docs):
        return False
    _save(next_docs)
    return True


def retrieve(query: str, k: int | None = None, documents: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    settings = get_settings()
    top_k = k if k is not None else settings.rag_top_k
    top_k = max(1, min(top_k, 12))
    q_terms = tokenize(query)
    if not q_terms:
        return []

    corpus: list[dict[str, Any]] = []
    source = documents if documents is not None else _load()
    for doc in source:
        title = doc.get("name") or doc.get("title") or "Document"
        pieces = doc.get("chunks") or chunk_text(doc.get("text") or "")
        for i, text in enumerate(pieces):
            if not (text or "").strip():
                continue
            corpus.append({"id": f"{doc.get('id', title)}:{i}", "title": title, "text": text})

    if not corpus:
        return []

    tokenized = [tokenize(c["text"]) for c in corpus]
    avgdl = sum(len(t) for t in tokenized) / max(len(tokenized), 1)
    df: dict[str, int] = {}
    for terms in tokenized:
        for t in set(terms):
            df[t] = df.get(t, 0) + 1
    n_docs = len(corpus)
    k1, b = 1.5, 0.75

    scored: list[tuple[float, dict[str, Any]]] = []
    for doc, terms in zip(corpus, tokenized):
        tf: dict[str, int] = {}
        for t in terms:
            tf[t] = tf.get(t, 0) + 1
        score = 0.0
        for qt in q_terms:
            f = tf.get(qt, 0)
            if not f:
                continue
            n = df.get(qt, 0)
            idf = math.log(1 + (n_docs - n + 0.5) / (n + 0.5))
            denom = f + k1 * (1 - b + b * (len(terms) / avgdl))
            score += idf * ((f * (k1 + 1)) / denom)
        if score > 0:
            scored.append((score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored[:top_k]]


def format_rag_context(hits: list[dict[str, Any]]) -> str:
    if not hits:
        return ""
    lines = [
        "Retrieved knowledge from the user's library. Prefer these when they answer the question. Cite the document name.",
        "",
    ]
    for i, hit in enumerate(hits, 1):
        lines.append(f"[{i}] {hit.get('title') or 'Document'}")
        lines.append((hit.get("text") or "")[:1200])
        lines.append("")
    return "\n".join(lines)[:6000]

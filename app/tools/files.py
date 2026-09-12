"""Image and file handling — validate, extract text, return a chat-ready payload."""

from __future__ import annotations

import base64
from typing import Any

MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_TEXT_BYTES = 300 * 1024
MAX_FILE_BYTES = 25 * 1024 * 1024

TEXT_EXTENSIONS = {
    "txt", "md", "csv", "json", "xml", "yml", "yaml", "log",
    "js", "jsx", "ts", "tsx", "py", "java", "c", "cpp", "h",
    "go", "rs", "rb", "php", "html", "css", "scss", "sh",
    "sql", "toml", "ini", "env", "conf",
}
IMAGE_MIMES = {"image/png", "image/jpeg", "image/jpg", "image/gif", "image/webp"}


def _ext(name: str) -> str:
    if "." not in name:
        return ""
    return name.rsplit(".", 1)[-1].lower()


def process_bytes(name: str, mime: str, data: bytes) -> dict[str, Any]:
    mime = mime or "application/octet-stream"
    size = len(data)
    if size > MAX_FILE_BYTES:
        raise ValueError(f"{name} is too large (max 25MB)")

    if mime in IMAGE_MIMES or mime.startswith("image/"):
        if size > MAX_IMAGE_BYTES:
            raise ValueError(f"{name} is too large for an image (max 8MB)")
        if mime not in IMAGE_MIMES and mime != "image/jpg":
            raise ValueError(f"Unsupported image type: {mime}")
        b64 = base64.b64encode(data).decode("ascii")
        data_url = f"data:{mime};base64,{b64}"
        return {
            "name": name,
            "mimeType": mime,
            "size": size,
            "kind": "image",
            "dataUrl": data_url,
        }

    text_like = mime.startswith("text/") or mime in {"application/json", "application/xml"} or _ext(name) in TEXT_EXTENSIONS
    if text_like and size <= MAX_TEXT_BYTES:
        text = data.decode("utf-8", errors="replace")
        return {
            "name": name,
            "mimeType": mime,
            "size": size,
            "kind": "text",
            "textContent": text,
        }

    return {
        "name": name,
        "mimeType": mime,
        "size": size,
        "kind": "file",
    }

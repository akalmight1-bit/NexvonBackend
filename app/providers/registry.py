from __future__ import annotations

from app.config import Settings, get_settings
from app.providers.base import Provider
from app.providers.nvidia import make_nvidia
from app.providers.ollama import make_ollama
from app.providers.openai_compat import OpenAICompatProvider
from app.providers.xai import make_xai


def _build_registry(settings: Settings) -> dict[str, Provider]:
    reg: dict[str, Provider] = {
        "ollama": make_ollama(settings),
        "xai": make_xai(settings),
        "nvidia": make_nvidia(settings),
    }
    if settings.local_base_url:
        reg["local"] = OpenAICompatProvider(
            provider_id="local",
            base_url=settings.local_base_url,
            api_key=settings.local_api_key,
            default_model=settings.local_model,
        )
    return reg


def list_providers() -> list[dict]:
    s = get_settings()
    reg = _build_registry(s)
    out = []
    for pid, p in reg.items():
        default_model = getattr(p, "default_model", "")
        out.append(
            {
                "id": pid,
                "default_model": default_model,
                "configured": _is_configured(pid, s),
            }
        )
    return out


def _is_configured(pid: str, s: Settings) -> bool:
    if pid == "xai":
        return bool(s.xai_api_key)
    if pid == "nvidia":
        return bool(s.nvidia_api_key)
    if pid == "ollama":
        return True  # always reachable if daemon is up
    if pid == "local":
        return bool(s.local_base_url)
    return False


def resolve_model(model: str | None) -> tuple[str, str]:
    """Return (provider_id, model_name)."""
    s = get_settings()
    raw = (model or "auto").strip().lower()

    if raw in ("", "auto"):
        return s.default_provider, _default_model_for(s.default_provider, s)

    if raw in ("grok", "xai"):
        return "xai", s.xai_model

    if raw == "nvidia":
        return "nvidia", s.nvidia_model

    if raw.startswith("nvidia/"):
        return "nvidia", raw.split("/", 1)[1] or s.nvidia_model

    if raw in ("ollama", "local-ollama"):
        return "ollama", s.ollama_model

    if raw.startswith("ollama/"):
        return "ollama", raw.split("/", 1)[1] or s.ollama_model

    if raw == "local":
        if s.local_base_url:
            return "local", s.local_model
        return "ollama", s.ollama_model

    # bare model name → default provider
    return s.default_provider, model or _default_model_for(s.default_provider, s)


def _default_model_for(provider: str, s: Settings) -> str:
    return {
        "xai": s.xai_model,
        "nvidia": s.nvidia_model,
        "ollama": s.ollama_model,
        "local": s.local_model,
    }.get(provider, s.ollama_model)


def get_provider(provider_id: str) -> Provider:
    s = get_settings()
    reg = _build_registry(s)
    if provider_id not in reg:
        raise KeyError(f"Unknown provider: {provider_id}")
    return reg[provider_id]

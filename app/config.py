from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # xAI
    xai_api_key: str = ""
    xai_model: str = "grok-4.5"
    xai_base_url: str = "https://api.x.ai/v1"

    # NVIDIA
    nvidia_api_key: str = ""
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    nvidia_model: str = "meta/llama-3.1-70b-instruct"

    # Ollama
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "llama3.2:3b"

    # Generic OpenAI-compatible
    local_base_url: str = ""
    local_api_key: str = ""
    local_model: str = "llama3.1"

    default_provider: str = "ollama"
    fallback_provider: str = "ollama"

    # Search — Brave and/or Serper (parallel when both keys exist)
    serper_api_key: str = ""
    serper_base_url: str = "https://google.serper.dev"
    brave_api_key: str = ""
    brave_base_url: str = "https://api.search.brave.com/res/v1"
    search_enabled: bool = True
    search_num_results: int = 6
    # auto | brave | serper | always | never
    search_mode: str = "auto"
    search_provider: str = "auto"

    # RAG
    rag_enabled: bool = True
    rag_dir: str = str(ROOT / "data" / "rag")
    rag_top_k: int = 6
    embedding_api_key: str = ""
    embedding_base_url: str = ""
    embedding_model: str = "text-embedding-3-small"

    # Local speech
    stt_enabled: bool = True
    whisper_model: str = "base"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    whisper_download_root: str = str(ROOT / "models" / "whisper")

    tts_enabled: bool = True
    piper_model_path: str = str(ROOT / "models" / "piper" / "en_US-lessac-medium.onnx")
    piper_config_path: str = str(ROOT / "models" / "piper" / "en_US-lessac-medium.onnx.json")
    piper_speaker: int = 0
    piper_length_scale: float = 1.0

    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: str = (
        "http://localhost:8080,http://127.0.0.1:8080,"
        "http://localhost:5173,http://127.0.0.1:5173"
    )

    system_prompt: str = (
        "You are Nexvon, a precise cinematic AI assistant. Speak clearly, with warmth "
        "but without fluff or emoji. Help with anything — code, writing, reasoning, "
        "science, decisions. You may occasionally draw on gravity, light, and time as "
        "metaphors, but never force the black-hole theme. Structure answers with "
        "markdown when it helps. Be concise unless the user asks for depth."
    )

    @property
    def origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def search_engines(self) -> list[str]:
        engines: list[str] = []
        if self.brave_api_key:
            engines.append("brave")
        if self.serper_api_key:
            engines.append("serper")
        return engines


@lru_cache
def get_settings() -> Settings:
    return Settings()

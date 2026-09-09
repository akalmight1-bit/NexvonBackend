from app.config import Settings
from app.providers.openai_compat import OpenAICompatProvider


def make_nvidia(settings: Settings) -> OpenAICompatProvider:
    return OpenAICompatProvider(
        provider_id="nvidia",
        base_url=settings.nvidia_base_url,
        api_key=settings.nvidia_api_key,
        default_model=settings.nvidia_model,
    )

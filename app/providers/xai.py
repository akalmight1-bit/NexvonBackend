from app.config import Settings
from app.providers.openai_compat import OpenAICompatProvider


def make_xai(settings: Settings) -> OpenAICompatProvider:
    return OpenAICompatProvider(
        provider_id="xai",
        base_url=settings.xai_base_url,
        api_key=settings.xai_api_key,
        default_model=settings.xai_model,
    )

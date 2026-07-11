from router.providers.base import OpenAICompatProvider


class OpenRouterProvider(OpenAICompatProvider):
    """OpenRouter free-tier — primary critic for dataset labeling only."""

    name = "openrouter"
    base_url = "https://openrouter.ai/api/v1"

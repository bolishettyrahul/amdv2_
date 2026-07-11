from router.providers.base import OpenAICompatProvider


class FireworksProvider(OpenAICompatProvider):
    name = "fireworks"
    base_url = "https://api.fireworks.ai/inference/v1"

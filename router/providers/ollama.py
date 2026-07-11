from router.providers.base import OpenAICompatProvider, Transport


class OllamaProvider(OpenAICompatProvider):
    """Local Ollama via its OpenAI-compatible endpoint. Free — GPU compute only."""

    name = "ollama"

    def __init__(self, host: str = "http://localhost:11434", transport: Transport | None = None,
                 max_retries: int = 2, retry_delay: float = 0.5):
        super().__init__(api_key="", transport=transport,
                         max_retries=max_retries, retry_delay=retry_delay)
        self.base_url = f"{host}/v1"

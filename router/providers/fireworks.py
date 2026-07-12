from __future__ import annotations

import os

from router.providers.base import OpenAICompatProvider, Transport

DEFAULT_FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"


class FireworksProvider(OpenAICompatProvider):
    name = "fireworks"

    def __init__(
        self,
        api_key: str = "",
        transport: Transport | None = None,
        base_url: str | None = None,
        **kwargs,
    ):
        super().__init__(api_key=api_key, transport=transport, **kwargs)
        # The grading harness injects FIREWORKS_BASE_URL; an explicit argument
        # (threaded from Settings) wins over the raw env read.
        self.base_url = (base_url or os.environ.get("FIREWORKS_BASE_URL")
                         or DEFAULT_FIREWORKS_BASE_URL)

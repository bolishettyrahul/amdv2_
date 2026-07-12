from router.providers.base import ChatResponse, OpenAICompatProvider

_GROQ_MAPPINGS = {
    # Live Fireworks catalog -> closest Groq test-time stand-in.
    "accounts/fireworks/models/gpt-oss-120b": "openai/gpt-oss-120b",
    "accounts/fireworks/models/kimi-k2p6": "moonshotai/kimi-k2-instruct",
    "accounts/fireworks/models/glm-5p1": "llama-3.3-70b-versatile",
    "accounts/fireworks/models/glm-5p2": "llama-3.3-70b-versatile",
    "accounts/fireworks/models/deepseek-v4-pro": "llama-3.3-70b-versatile",
    # Retired Fireworks catalog generation.
    "accounts/fireworks/models/llama-v3p1-8b-instruct": "llama-3.1-8b-instant",
    "accounts/fireworks/models/gemma2-9b-it": "gemma2-9b-it",
    "accounts/fireworks/models/gemma2-27b-it": "llama-3.3-70b-versatile",
    "accounts/fireworks/models/llama-v3p1-70b-instruct": "llama-3.3-70b-versatile",
    "accounts/fireworks/models/qwen2p5-72b-instruct": "llama-3.3-70b-versatile",
    "accounts/fireworks/models/llama-v3p1-405b-instruct": "llama-3.3-70b-versatile",
    "accounts/fireworks/models/llama4-maverick-instruct-basic": "llama-3.3-70b-versatile",
}


class GroqProvider(OpenAICompatProvider):
    name = "groq"
    base_url = "https://api.groq.com/openai/v1"

    def chat(
        self,
        messages: list[dict],
        model: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> ChatResponse:
        mapped_model = _GROQ_MAPPINGS.get(model, model)
        return super().chat(messages, mapped_model, temperature, max_tokens)

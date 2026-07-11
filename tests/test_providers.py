"""One OpenAI-chat-compatible interface for every generate-text need, local or paid."""

import pytest

from router.providers.base import ChatResponse, LLMProvider, ProviderError
from router.providers.fireworks import FireworksProvider
from router.providers.groq import GroqProvider
from router.providers.ollama import OllamaProvider
from router.providers.openrouter import OpenRouterProvider


def ok_response(text="hello", tokens_in=10, tokens_out=5):
    return (200, {
        "choices": [{"message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": tokens_in, "completion_tokens": tokens_out},
    })


def test_chat_sends_openai_payload_and_parses_usage():
    seen = {}

    def transport(url, headers, payload):
        seen.update(url=url, headers=headers, payload=payload)
        return ok_response("Paris", 12, 3)

    p = FireworksProvider(api_key="fw-key", transport=transport)
    resp = p.chat([{"role": "user", "content": "Capital of France?"}],
                  model="accounts/fireworks/models/gemma2-9b-it", temperature=0.2)

    assert isinstance(resp, ChatResponse)
    assert resp.text == "Paris"
    assert (resp.tokens_in, resp.tokens_out) == (12, 3)
    assert seen["url"] == "https://api.fireworks.ai/inference/v1/chat/completions"
    assert seen["headers"]["Authorization"] == "Bearer fw-key"
    assert seen["payload"]["model"] == "accounts/fireworks/models/gemma2-9b-it"
    assert seen["payload"]["temperature"] == 0.2
    assert seen["payload"]["messages"][0]["content"] == "Capital of France?"


def test_groq_ollama_openrouter_base_urls():
    urls = []

    def transport(url, headers, payload):
        urls.append(url)
        return ok_response()

    GroqProvider(api_key="k", transport=transport).chat([], model="m")
    OllamaProvider(transport=transport).chat([], model="qwen2.5:7b")
    OpenRouterProvider(api_key="k", transport=transport).chat([], model="m")
    assert urls[0] == "https://api.groq.com/openai/v1/chat/completions"
    assert urls[1] == "http://localhost:11434/v1/chat/completions"
    assert urls[2] == "https://openrouter.ai/api/v1/chat/completions"


def test_retries_on_429_then_succeeds():
    calls = {"n": 0}

    def transport(url, headers, payload):
        calls["n"] += 1
        if calls["n"] < 3:
            return (429, {"error": "rate limited"})
        return ok_response("ok")

    p = GroqProvider(api_key="k", transport=transport, retry_delay=0.0)
    assert p.chat([], model="m").text == "ok"
    assert calls["n"] == 3


def test_raises_provider_error_after_max_retries():
    def transport(url, headers, payload):
        return (500, {"error": "boom"})

    p = GroqProvider(api_key="k", transport=transport, retry_delay=0.0, max_retries=2)
    with pytest.raises(ProviderError):
        p.chat([], model="m")


def test_missing_usage_falls_back_to_estimate():
    def transport(url, headers, payload):
        return (200, {"choices": [{"message": {"content": "four words of text"}}]})

    resp = OllamaProvider(transport=transport).chat(
        [{"role": "user", "content": "x" * 40}], model="m")
    assert resp.tokens_in > 0
    assert resp.tokens_out > 0


def test_llmprovider_is_abstract():
    with pytest.raises(TypeError):
        LLMProvider()  # type: ignore[abstract]

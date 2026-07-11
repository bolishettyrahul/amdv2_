"""LLMProvider ABC + shared OpenAI-chat-compatible HTTP implementation."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

# transport(url, headers, payload) -> (status_code, response_json)
Transport = Callable[[str, dict, dict], tuple[int, dict]]


class ProviderError(RuntimeError):
    pass


@dataclass(frozen=True)
class ChatResponse:
    text: str
    tokens_in: int
    tokens_out: int
    model: str


def _requests_transport(url: str, headers: dict, payload: dict) -> tuple[int, dict]:
    import requests

    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    try:
        body = resp.json()
    except ValueError:
        body = {"error": resp.text}
    return resp.status_code, body


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class LLMProvider(ABC):
    name: str

    @abstractmethod
    def chat(
        self,
        messages: list[dict],
        model: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> ChatResponse: ...


class OpenAICompatProvider(LLMProvider):
    """Any endpoint speaking POST {base_url}/chat/completions."""

    base_url: str = ""
    name: str = "openai-compat"

    def __init__(
        self,
        api_key: str = "",
        transport: Transport | None = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        self.api_key = api_key
        self.transport = transport or _requests_transport
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def chat(
        self,
        messages: list[dict],
        model: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> ChatResponse:
        payload: dict = {"model": model, "messages": messages, "temperature": temperature}
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        url = f"{self.base_url}/chat/completions"
        last_error = ""
        for attempt in range(self.max_retries):
            status, body = self.transport(url, headers, payload)
            if status == 200:
                return self._parse(body, messages, model)
            last_error = f"HTTP {status}: {body}"
            if status in (408, 429) or status >= 500:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2**attempt))
                continue
            break  # non-retryable client error
        raise ProviderError(f"{self.name} chat failed after retries: {last_error}")

    def _parse(self, body: dict, messages: list[dict], model: str) -> ChatResponse:
        try:
            text = body["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError) as exc:
            raise ProviderError(f"{self.name} malformed response: {body}") from exc
        usage = body.get("usage") or {}
        tokens_in = usage.get("prompt_tokens") or _estimate_tokens(
            "".join(m.get("content", "") for m in messages)
        )
        tokens_out = usage.get("completion_tokens") or _estimate_tokens(text)
        return ChatResponse(text=text, tokens_in=tokens_in, tokens_out=tokens_out, model=model)

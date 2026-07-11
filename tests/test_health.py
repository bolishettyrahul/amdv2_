"""Startup health check: is local Ollama inference available and fast enough?"""

from router.health import check_local_inference
from router.providers.base import ChatResponse, LLMProvider, ProviderError


class FakeOllama(LLMProvider):
    name = "ollama"

    def __init__(self, fail=False):
        self.fail = fail

    def chat(self, messages, model, temperature=0.0, max_tokens=None):
        if self.fail:
            raise ProviderError("connection refused")
        return ChatResponse("pong", 5, 1, model)


def test_healthy_when_fast():
    clock = iter([0.0, 0.5])
    status = check_local_inference(FakeOllama(), timer=lambda: next(clock))
    assert status.ok
    assert status.latency_s == 0.5


def test_unhealthy_when_slow():
    clock = iter([0.0, 5.0])
    status = check_local_inference(FakeOllama(), max_latency_s=3.0, timer=lambda: next(clock))
    assert not status.ok
    assert "slow" in status.reason


def test_unhealthy_when_unreachable():
    status = check_local_inference(FakeOllama(fail=True))
    assert not status.ok
    assert "connection refused" in status.reason

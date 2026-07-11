"""LLM critique: dataset labeling only, never live routing."""

import pytest

from router.critique import Critic
from router.providers.base import ChatResponse, LLMProvider, ProviderError
from router.types import Domain, Task


class FakeProvider(LLMProvider):
    name = "fake"

    def __init__(self, responses, fail=False):
        self.responses = list(responses)
        self.fail = fail
        self.calls = []

    def chat(self, messages, model, temperature=0.0, max_tokens=None):
        if self.fail:
            raise ProviderError("rate limited")
        self.calls.append({"messages": messages, "model": model})
        return ChatResponse(self.responses.pop(0), 10, 5, model)


TASK = Task("t", "Capital of Australia?")


def make_critic(primary, fallback):
    return Critic(primary_provider=primary, primary_model="meta-llama/llama-3.3-70b:free",
                  fallback_provider=fallback, fallback_model="mistral-nemo:12b-instruct")


def test_skipped_for_deterministic_domains():
    primary = FakeProvider([])
    critic = make_critic(primary, FakeProvider([]))
    for domain in (Domain.MATH, Domain.SENTIMENT, Domain.NER,
                   Domain.CODE_DEBUG, Domain.CODE_GEN):
        assert critic.critique(TASK, domain, "answer", actor_model="ollama/qwen2.5:7b") is None
    assert primary.calls == []


def test_required_for_weak_domains_verdict_parsed():
    primary = FakeProvider(["VERDICT: correct\nREASONING: Canberra is right."])
    critic = make_critic(primary, FakeProvider([]))
    result = critic.critique(TASK, Domain.FACTUAL, "Canberra", actor_model="ollama/qwen2.5:7b")
    assert result.verdict == "correct"
    assert "Canberra" in result.reasoning
    assert result.critic_model == "meta-llama/llama-3.3-70b:free"


def test_incorrect_verdict_parsed():
    primary = FakeProvider(["VERDICT: incorrect\nREASONING: The capital is Canberra, not Sydney."])
    critic = make_critic(primary, FakeProvider([]))
    result = critic.critique(TASK, Domain.FACTUAL, "Sydney", actor_model="ollama/qwen2.5:7b")
    assert result.verdict == "incorrect"


def test_falls_back_when_primary_unavailable():
    fallback = FakeProvider(["VERDICT: correct\nREASONING: fine."])
    critic = make_critic(FakeProvider([], fail=True), fallback)
    result = critic.critique(TASK, Domain.SUMMARIZATION, "some summary",
                             actor_model="ollama/qwen2.5:7b")
    assert result.critic_model == "mistral-nemo:12b-instruct"
    assert len(fallback.calls) == 1


def test_critic_never_critiques_its_own_answer():
    fallback = FakeProvider(["VERDICT: correct\nREASONING: ok."])
    critic = make_critic(FakeProvider(["should not be used"]), fallback)
    # Actor was the same model as the primary critic -> must switch to fallback.
    result = critic.critique(TASK, Domain.FACTUAL, "Canberra",
                             actor_model="meta-llama/llama-3.3-70b:free")
    assert result.critic_model == "mistral-nemo:12b-instruct"


def test_raises_if_no_distinct_critic_exists():
    # Primary is down and the fallback critic was the actor -> hard rule violated.
    critic = make_critic(FakeProvider([], fail=True), FakeProvider([]))
    with pytest.raises(ValueError):
        critic.critique(TASK, Domain.FACTUAL, "x", actor_model="mistral-nemo:12b-instruct")

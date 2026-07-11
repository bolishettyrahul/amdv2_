"""Stage 3: dynamic paid routing — cheapest sufficient model + self-correction."""

from router.cost import CostTracker
from router.providers.base import ChatResponse, LLMProvider
from router.stage3 import ESCALATION, ROUTING, Stage3Paid, choose_model, estimate_complexity
from router.types import Domain, Task

M8 = "accounts/fireworks/models/llama-v3p1-8b-instruct"
G9 = "accounts/fireworks/models/gemma2-9b-it"
L70 = "accounts/fireworks/models/llama-v3p1-70b-instruct"


class FakeProvider(LLMProvider):
    name = "fireworks"

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat(self, messages, model, temperature=0.0, max_tokens=None):
        self.calls.append({"messages": messages, "model": model})
        return ChatResponse(self.responses.pop(0), tokens_in=100, tokens_out=50, model=model)


def test_routing_matrix_prefers_cheap_models_for_simple_tasks():
    assert choose_model(Domain.FACTUAL, "simple") == G9
    assert choose_model(Domain.FACTUAL, "complex") == L70
    assert choose_model(Domain.SENTIMENT, "simple") == M8
    assert choose_model(Domain.NER, "complex") == M8  # NER never needs a big model


def test_complexity_estimation():
    assert estimate_complexity(Task("t", "What is the capital of France?"), Domain.FACTUAL) == "simple"
    assert estimate_complexity(Task("t", "Prove that sqrt(2) is irrational."), Domain.MATH) == "complex"
    long_prompt = "word " * 800
    assert estimate_complexity(Task("t", long_prompt), Domain.SUMMARIZATION) == "complex"


def test_verified_first_try_single_call():
    p = FakeProvider(["```python\ndef add(a, b):\n    return a + b\n```"])
    tracker = CostTracker()
    s3 = Stage3Paid(p, cost_tracker=tracker)
    task = Task("t", "Write add.", metadata={"tests": "assert add(1, 2) == 3"})
    r = s3.attempt(task, Domain.CODE_GEN)
    assert r.verified
    assert r.stage == "stage3_paid"
    assert len(p.calls) == 1
    assert r.cost_usd > 0.0
    assert tracker.total_cost_usd == r.cost_usd


def test_self_correction_retry_includes_failure_context():
    bad = "```python\ndef add(a, b):\n    return a - b\n```"
    good = "```python\ndef add(a, b):\n    return a + b\n```"
    p = FakeProvider([bad, good])
    task = Task("t", "Write add.", metadata={"tests": "assert add(1, 2) == 3"})
    r = Stage3Paid(p, cost_tracker=CostTracker()).attempt(task, Domain.CODE_GEN)
    assert r.verified
    assert len(p.calls) == 2
    retry_text = str(p.calls[1]["messages"])
    assert "fail" in retry_text.lower()  # failure context fed back


def test_exhausted_corrections_escalate_model_then_return_last_answer():
    bad = "```python\ndef add(a, b):\n    return a - b\n```"
    p = FakeProvider([bad, bad, bad])
    task = Task("t", "Write add.", metadata={"tests": "assert add(1, 2) == 3"})
    r = Stage3Paid(p, cost_tracker=CostTracker()).attempt(task, Domain.CODE_GEN)
    assert not r.verified
    assert r.answer is not None  # final stage always answers
    assert len(p.calls) == 3
    routed = choose_model(Domain.CODE_GEN, "simple")
    assert p.calls[-1]["model"] == ESCALATION[routed]  # last try on the stronger model
    assert p.calls[0]["model"] == routed


def test_weak_domain_accepts_single_answer_no_retry():
    p = FakeProvider(["Canberra"])
    r = Stage3Paid(p, cost_tracker=CostTracker()).attempt(
        Task("t", "Capital of Australia?"), Domain.FACTUAL)
    assert r.answer == "Canberra"
    assert len(p.calls) == 1


def test_every_domain_and_complexity_has_a_route():
    for domain in Domain:
        for complexity in ("simple", "complex"):
            assert choose_model(domain, complexity) in {
                m for tiers in ROUTING.values() for m in tiers.values()
            }

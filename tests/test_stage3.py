"""Stage 3: dynamic paid routing — cheapest sufficient model + self-correction."""

from router.cost import CostTracker
from router.providers.base import ChatResponse, LLMProvider
from router.stage3 import (DSV4, ESCALATION, GLM52, KIMI, OSS120, ROUTING,
                           Stage3Paid, choose_model, estimate_complexity,
                           resolve_model)
from router.types import Domain, Task


class FakeProvider(LLMProvider):
    name = "fireworks"

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat(self, messages, model, temperature=0.0, max_tokens=None):
        self.calls.append({"messages": messages, "model": model})
        return ChatResponse(self.responses.pop(0), tokens_in=100, tokens_out=50, model=model)


def test_routing_matrix_prefers_cheap_models_for_simple_tasks():
    assert choose_model(Domain.FACTUAL, "simple") == OSS120
    assert choose_model(Domain.FACTUAL, "complex") == KIMI
    assert choose_model(Domain.SENTIMENT, "simple") == OSS120
    assert choose_model(Domain.NER, "complex") == OSS120  # NER never needs a big model
    assert choose_model(Domain.CODE_GEN, "complex") == GLM52


def test_complexity_estimation():
    assert estimate_complexity(Task("t", "What is the capital of France?"), Domain.FACTUAL) == "simple"
    assert estimate_complexity(Task("t", "Prove that sqrt(2) is irrational."), Domain.MATH) == "complex"
    long_prompt = "word " * 800
    assert estimate_complexity(Task("t", long_prompt), Domain.SUMMARIZATION) == "complex"


class DeadModelProvider(LLMProvider):
    """404s for models in `dead`, answers from any other model."""

    name = "fireworks"

    def __init__(self, dead, text="Paris."):
        self.dead = set(dead)
        self.text = text
        self.calls = []

    def chat(self, messages, model, temperature=0.0, max_tokens=None):
        self.calls.append(model)
        if model in self.dead:
            from router.providers.base import ProviderError
            err = ProviderError(f"fireworks chat failed: HTTP 404 model not found: {model}")
            err.status = 404
            raise err
        return ChatResponse(self.text, tokens_in=10, tokens_out=5, model=model)


def test_dead_allowed_model_fails_over_to_next_allowed():
    # The real judging list shape: first allowed model is not deployed.
    allowed = ["dead-model", "live-model"]
    p = DeadModelProvider(dead=["dead-model"])
    s3 = Stage3Paid(p, cost_tracker=CostTracker(), allowed_models=allowed)
    r = s3.attempt(Task("t", "What is the capital of France?"), Domain.FACTUAL)
    assert r.answer == "Paris."
    assert r.model == "live-model"


def test_404_model_is_blacklisted_for_subsequent_tasks():
    allowed = ["dead-model", "live-model"]
    p = DeadModelProvider(dead=["dead-model"])
    s3 = Stage3Paid(p, cost_tracker=CostTracker(), allowed_models=allowed)
    s3.attempt(Task("t1", "What is the capital of France?"), Domain.FACTUAL)
    s3.attempt(Task("t2", "What is the capital of Peru?"), Domain.FACTUAL)
    # The dead model is tried at most once across the whole batch.
    assert p.calls.count("dead-model") == 1


def test_all_models_dead_raises_so_pipeline_reports_stage_error():
    import pytest

    p = DeadModelProvider(dead=["a", "b"])
    s3 = Stage3Paid(p, cost_tracker=CostTracker(), allowed_models=["a", "b"])
    with pytest.raises(Exception):
        s3.attempt(Task("t", "capital?"), Domain.FACTUAL)


def test_transient_error_fails_over_without_blacklisting():
    class FlakyProvider(LLMProvider):
        name = "fireworks"

        def __init__(self):
            self.calls = []

        def chat(self, messages, model, temperature=0.0, max_tokens=None):
            self.calls.append(model)
            if model == "flaky-model" and self.calls.count("flaky-model") == 1:
                from router.providers.base import ProviderError
                raise ProviderError("fireworks chat failed after retries: HTTP 503")
            return ChatResponse("Paris.", tokens_in=10, tokens_out=5, model=model)

    p = FlakyProvider()
    s3 = Stage3Paid(p, cost_tracker=CostTracker(),
                    allowed_models=["flaky-model", "backup-model"])
    r1 = s3.attempt(Task("t1", "capital?"), Domain.FACTUAL)
    assert r1.answer == "Paris."
    # Next task tries the preferred model again: a 503 is not a dead model.
    s3.attempt(Task("t2", "capital?"), Domain.FACTUAL)
    assert p.calls.count("flaky-model") == 2


def test_provider_error_carries_http_status():
    from router.providers.base import OpenAICompatProvider, ProviderError

    class P(OpenAICompatProvider):
        base_url = "http://x"
        name = "fireworks"

    def transport_404(url, headers, payload):
        return 404, {"error": "Model not found"}

    import pytest
    with pytest.raises(ProviderError) as exc_info:
        P(transport=transport_404).chat([{"role": "user", "content": "hi"}], model="m")
    assert exc_info.value.status == 404


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


def test_restrictive_allowed_models_yields_allowed_choice_for_every_domain():
    """The harness pins ALLOWED_MODELS; routing must never pick a rejected model."""
    allowed = [OSS120, DSV4]
    for domain in Domain:
        for complexity in ("simple", "complex"):
            model, _ = resolve_model(domain, complexity, allowed)
            assert model in allowed, f"{domain} {complexity} routed to {model}"


def test_allowed_models_fallback_order():
    # Routed model itself allowed: no change, no flag.
    assert resolve_model(Domain.FACTUAL, "simple", [OSS120, DSV4]) == (OSS120, False)
    # Routed (KIMI) not allowed, other tier (OSS120) is: swap tiers, no flag.
    assert resolve_model(Domain.MATH, "complex", [OSS120, DSV4]) == (OSS120, False)
    # Neither tier (KIMI/OSS120) allowed, escalation of routed (KIMI -> DSV4) is.
    assert resolve_model(Domain.MATH, "complex", [DSV4]) == (DSV4, False)
    # Nothing in ROUTING/ESCALATION for the domain allowed: first allowed model,
    # flagged as a last-resort fallback so it's visible in results.json.
    other = "accounts/fireworks/models/some-harness-model"
    assert resolve_model(Domain.SENTIMENT, "simple", [other]) == (other, True)
    # Empty allowlist = local dev, anything goes.
    assert resolve_model(Domain.MATH, "complex", []) == (choose_model(Domain.MATH, "complex"), False)


def test_attempt_under_allowlist_only_calls_allowed_models_and_flags_fallback():
    bad = "```python\ndef add(a, b):\n    return a - b\n```"
    # CODE_GEN routes OSS120/GLM52, escalations KIMI/DSV4: none of them allowed.
    unknown = "accounts/fireworks/models/some-harness-model"
    p = FakeProvider([bad, bad, bad])
    task = Task("t", "Write add.", metadata={"tests": "assert add(1, 2) == 3"})
    r = Stage3Paid(p, cost_tracker=CostTracker(), allowed_models=[unknown]).attempt(
        task, Domain.CODE_GEN)
    assert all(c["model"] == unknown for c in p.calls)  # incl. the final escalation try
    assert r.allowed_models_fallback
    assert r.answer is not None


def test_attempt_without_allowlist_does_not_flag_fallback():
    p = FakeProvider(["Canberra"])
    r = Stage3Paid(p, cost_tracker=CostTracker()).attempt(
        Task("t", "Capital of Australia?"), Domain.FACTUAL)
    assert not r.allowed_models_fallback


def test_unpriced_allowed_model_does_not_crash_cost_tracking():
    unknown = "accounts/fireworks/models/some-harness-model"
    p = FakeProvider(["Canberra"])
    r = Stage3Paid(p, cost_tracker=CostTracker(), allowed_models=[unknown]).attempt(
        Task("t", "Capital of Australia?"), Domain.FACTUAL)
    assert p.calls[0]["model"] == unknown
    assert r.answer == "Canberra"
    assert r.tokens_in > 0  # tokens still counted even with unknown pricing

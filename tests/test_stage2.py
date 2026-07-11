"""Stage 2: local model (free) with per-domain free verification gates."""

from router.providers.base import ChatResponse, LLMProvider
from router.stage2 import Stage2Local
from router.types import Domain, Task


class FakeProvider(LLMProvider):
    name = "ollama"

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat(self, messages, model, temperature=0.0, max_tokens=None):
        self.calls.append({"messages": messages, "model": model, "temperature": temperature})
        return ChatResponse(self.responses.pop(0), tokens_in=10, tokens_out=5, model=model)


def test_factual_self_consistency_pass():
    p = FakeProvider(["Paris", "paris."])
    r = Stage2Local(p, factual_k=2).attempt(Task("t", "Capital of France?"), Domain.FACTUAL)
    assert r.verified
    assert r.answer == "Paris"
    assert r.stage == "stage2_local"
    assert r.cost_usd == 0.0
    assert r.tokens_in == 20  # both samples counted
    assert len(p.calls) == 2


def test_factual_self_consistency_fail():
    p = FakeProvider(["Paris", "Lyon"])
    r = Stage2Local(p, factual_k=2).attempt(Task("t", "Capital of France?"), Domain.FACTUAL)
    assert not r.verified


def test_math_model_expression_computed_by_tool():
    p = FakeProvider(["12 * 7"])
    r = Stage2Local(p).attempt(Task("t", "What is twelve times seven?"), Domain.MATH)
    assert r.verified
    assert r.answer == "84"


def test_math_unparseable_output_escalates():
    p = FakeProvider(["I cannot compute that."])
    r = Stage2Local(p).attempt(Task("t", "hard math"), Domain.MATH)
    assert not r.verified


def test_sentiment_label_schema():
    p = FakeProvider(["Positive"])
    r = Stage2Local(p).attempt(Task("t", "Sentiment: 'meh but fine'"), Domain.SENTIMENT)
    assert r.verified
    assert r.answer == "positive"


def test_sentiment_bad_label_escalates():
    p = FakeProvider(["It is complicated."])
    r = Stage2Local(p).attempt(Task("t", "Sentiment: 'meh'"), Domain.SENTIMENT)
    assert not r.verified


def test_summarization_heuristic_gate():
    source = ("NASA launched the Artemis mission from Florida. " * 10)
    p = FakeProvider(["NASA launched the Artemis mission from Florida; a success overall."])
    r = Stage2Local(p).attempt(Task("t", f"Summarize: {source}"), Domain.SUMMARIZATION)
    assert r.verified


def test_ner_json_schema_gate():
    p = FakeProvider(['```json\n[{"text": "Paris", "label": "GPE"}]\n```'])
    r = Stage2Local(p).attempt(Task("t", "Extract entities: Paris is nice."), Domain.NER)
    assert r.verified
    assert '"Paris"' in r.answer


def test_code_generation_runs_tests_in_sandbox():
    good = "Here:\n```python\ndef add(a, b):\n    return a + b\n```"
    p = FakeProvider([good])
    task = Task("t", "Write add(a, b).", metadata={"tests": "assert add(1, 2) == 3"})
    r = Stage2Local(p).attempt(task, Domain.CODE_GEN)
    assert r.verified
    assert "def add" in r.answer


def test_code_generation_failing_tests_escalates():
    p = FakeProvider(["```python\ndef add(a, b):\n    return a - b\n```"])
    task = Task("t", "Write add(a, b).", metadata={"tests": "assert add(1, 2) == 3"})
    r = Stage2Local(p).attempt(task, Domain.CODE_GEN)
    assert not r.verified


def test_logic_majority_of_three():
    p = FakeProvider(["Yes", "yes.", "No"])
    r = Stage2Local(p, logic_k=3).attempt(Task("t", "puzzle"), Domain.LOGIC)
    assert r.verified
    assert r.answer == "Yes"


def test_paid_cloud_fallback_mode_costs_tokens():
    # Standardized-env fallback: same stage logic, cheap Fireworks model, real cost.
    p = FakeProvider(["Paris", "Paris"])
    p.name = "fireworks"
    model = "accounts/fireworks/models/llama-v3p1-8b-instruct"
    s2 = Stage2Local(p, models={Domain.FACTUAL: model}, factual_k=2, paid=True)
    r = s2.attempt(Task("t", "Capital of France?"), Domain.FACTUAL)
    assert r.verified
    assert r.cost_usd > 0.0
    assert r.model == model

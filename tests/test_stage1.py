"""Stage 1: deterministic tools per domain, free verification gate."""

from router.stage1 import Stage1Deterministic
from router.tools.ner_tool import NerTool
from router.types import Domain, Task


def stage1(**kwargs) -> Stage1Deterministic:
    kwargs.setdefault("ner_tool", NerTool(None))
    return Stage1Deterministic(**kwargs)


def test_math_resolved_and_verified():
    r = stage1().attempt(Task("t", "What is 6 * 7?"), Domain.MATH)
    assert r.verified
    assert r.answer == "42"
    assert r.stage == "stage1_tool"
    assert r.cost_usd == 0.0


def test_strong_sentiment_resolved():
    r = stage1().attempt(
        Task("t", "Sentiment of this review: 'Absolutely wonderful, I loved every minute!'"),
        Domain.SENTIMENT,
    )
    assert r.verified
    assert r.answer == "positive"


def test_solver_shaped_logic_resolved():
    prompt = "```smtlib\n(declare-const x Int)\n(assert (> x 5))\n(assert (< x 3))\n```"
    r = stage1().attempt(Task("t", prompt), Domain.LOGIC)
    assert r.verified
    assert r.answer == "unsat"


def test_code_debug_returns_code_unchanged_when_tests_already_pass():
    code = "def add(a, b):\n    return a + b\n"
    task = Task("t", "Fix this.", metadata={"code": code, "tests": "assert add(1, 2) == 3"})
    r = stage1().attempt(task, Domain.CODE_DEBUG)
    assert r.verified
    assert r.answer == code


def test_code_debug_with_failing_tests_escalates():
    task = Task("t", "Fix this.", metadata={"code": "def add(a, b):\n    return a - b\n",
                                            "tests": "assert add(1, 2) == 3"})
    r = stage1().attempt(task, Domain.CODE_DEBUG)
    assert not r.verified


def test_domains_without_stage1_tool_escalate():
    s1 = stage1()
    for domain in (Domain.FACTUAL, Domain.SUMMARIZATION, Domain.CODE_GEN):
        r = s1.attempt(Task("t", "whatever"), domain)
        assert not r.verified


def test_weak_sentiment_escalates():
    r = stage1().attempt(Task("t", "Sentiment: 'It arrived on Tuesday.'"), Domain.SENTIMENT)
    assert not r.verified

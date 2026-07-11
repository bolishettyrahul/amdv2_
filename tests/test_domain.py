"""Stage 0: infer the (unlabeled) domain of each task."""

from router.domain import DomainClassifier
from router.types import Domain


def heuristic_clf() -> DomainClassifier:
    # embed_fn=None disables the embedding path -> pure keyword heuristic.
    return DomainClassifier(embed_fn=None)


def test_heuristic_code_debugging():
    d, _ = heuristic_clf().classify(
        "Fix the bug in this function:\n```python\ndef f():\n    return 1/0\n```\n"
        "It raises ZeroDivisionError."
    )
    assert d == Domain.CODE_DEBUG


def test_heuristic_code_generation():
    d, _ = heuristic_clf().classify(
        "Write a Python function that returns the nth Fibonacci number."
    )
    assert d == Domain.CODE_GEN


def test_heuristic_math():
    d, _ = heuristic_clf().classify("What is the derivative of x**2 + 3*x?")
    assert d == Domain.MATH


def test_heuristic_summarization():
    d, _ = heuristic_clf().classify("Summarize the following article in two sentences: ...")
    assert d == Domain.SUMMARIZATION


def test_heuristic_sentiment():
    d, _ = heuristic_clf().classify(
        "Classify the sentiment of this review as positive or negative: 'I loved it.'"
    )
    assert d == Domain.SENTIMENT


def test_heuristic_ner():
    d, _ = heuristic_clf().classify(
        "Extract all named entities from: Barack Obama visited Paris in 2015."
    )
    assert d == Domain.NER


def test_heuristic_logic():
    d, _ = heuristic_clf().classify(
        "If all bloops are razzies and all razzies are lazzies, are all bloops lazzies?"
    )
    assert d == Domain.LOGIC


def test_heuristic_factual_default():
    d, _ = heuristic_clf().classify("What is the capital of France?")
    assert d == Domain.FACTUAL


def test_embedding_path_picks_most_similar_exemplar_domain():
    def fake_embed(texts):
        return [[1.0, 0.0] if "math" in t.lower() else [0.0, 1.0] for t in texts]

    clf = DomainClassifier(
        embed_fn=fake_embed,
        exemplars={Domain.MATH: ["solve math problems"], Domain.FACTUAL: ["trivia question"]},
    )
    d, conf = clf.classify("a math question")
    assert d == Domain.MATH
    assert conf > 0.9


def test_ambiguous_top2_falls_back_to_llm_classifier():
    calls = []

    def fake_embed(texts):
        return [[1.0, 0.0] for _ in texts]  # everything identical -> tie

    def llm_classify(prompt):
        calls.append(prompt)
        return Domain.LOGIC

    clf = DomainClassifier(
        embed_fn=fake_embed,
        exemplars={Domain.MATH: ["a"], Domain.LOGIC: ["b"]},
        llm_classify=llm_classify,
    )
    d, _ = clf.classify("ambiguous prompt")
    assert d == Domain.LOGIC
    assert calls == ["ambiguous prompt"]

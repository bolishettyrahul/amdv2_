"""Stage 0 — domain classification (free, local).

Primary path: sentence-embedding cosine similarity against per-domain exemplar
prompts (sentence-transformers all-MiniLM-L6-v2, lazy-loaded). If the top-2
domain scores are too close, an optional $0 local LLM classification call
breaks the tie. If sentence-transformers is unavailable (CPU-only sandbox,
no wheels), a keyword/structure heuristic keeps Stage 0 functional.
"""

from __future__ import annotations

import math
import re
from typing import Callable

from router.types import Domain

EXEMPLARS: dict[Domain, list[str]] = {
    Domain.FACTUAL: [
        "What is the capital of Australia?",
        "Who wrote the novel 1984?",
        "In what year did the Berlin Wall fall?",
    ],
    Domain.MATH: [
        "Solve for x: 3x + 7 = 22.",
        "What is the integral of x^2 dx?",
        "A train travels 60 km in 45 minutes. What is its average speed?",
    ],
    Domain.SENTIMENT: [
        "Classify the sentiment of this review as positive or negative: 'Terrible service.'",
        "Is the following tweet positive, negative, or neutral?",
    ],
    Domain.SUMMARIZATION: [
        "Summarize the following article in three sentences.",
        "Provide a concise summary of the passage below.",
    ],
    Domain.NER: [
        "Extract all named entities (person, organization, location) from the text.",
        "List the people and places mentioned in this paragraph.",
    ],
    Domain.CODE_DEBUG: [
        "This Python function raises an IndexError. Find and fix the bug.",
        "The following code fails its unit tests. Debug it.",
    ],
    Domain.LOGIC: [
        "If all A are B and some B are C, does it follow that some A are C?",
        "Three friends each own one pet. Use the clues to determine who owns the cat.",
    ],
    Domain.CODE_GEN: [
        "Write a Python function that reverses a linked list.",
        "Implement a function to check whether a string is a palindrome.",
    ],
}

_CODE_MARKERS = re.compile(r"```|\bdef \w+\s*\(|\bclass \w+\s*[:(]|\bfunction\s+\w+\s*\(|;\s*$", re.M)
_CODE_WORDS = ("python", "javascript", "code", "function", "script", "program")
_DEBUG_WORDS = ("bug", "fix", "error", "traceback", "debug", "fails", "doesn't work", "broken", "incorrect output")
_GEN_WORDS = ("write", "implement", "generate", "create", "build")
_MATH_WORDS = ("solve", "derivative", "integral", "equation", "calculate", "compute",
               "how many", "how much", "sum of", "average", "percent", "probability")
_LOGIC_WORDS = ("if all", "if some", "premise", "conclusion", "syllogism", "logically",
                "deduce", "valid argument", "follows that", "knights", "truth-teller")
_NER_WORDS = ("entit", "named entity", "people and places", "person, organization")
_SENTIMENT_WORDS = ("sentiment", "positive or negative", "positive, negative")


def heuristic_domain(prompt: str) -> Domain:
    p = prompt.lower()
    has_code = bool(_CODE_MARKERS.search(prompt)) or any(w in p for w in _CODE_WORDS)
    if has_code:
        if any(w in p for w in _DEBUG_WORDS):
            return Domain.CODE_DEBUG
        if any(w in p for w in _GEN_WORDS):
            return Domain.CODE_GEN
        return Domain.CODE_DEBUG if "```" in prompt else Domain.CODE_GEN
    if "summar" in p:
        return Domain.SUMMARIZATION
    if any(w in p for w in _SENTIMENT_WORDS):
        return Domain.SENTIMENT
    if any(w in p for w in _NER_WORDS) or ("extract" in p and ("names" in p or "location" in p)):
        return Domain.NER
    if any(w in p for w in _MATH_WORDS) or re.search(r"\d+\s*[-+*/^=]\s*\w", p):
        return Domain.MATH
    if any(w in p for w in _LOGIC_WORDS):
        return Domain.LOGIC
    return Domain.FACTUAL


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def _load_default_embedder() -> Callable[[list[str]], list[list[float]]] | None:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return lambda texts: model.encode(texts).tolist()


class DomainClassifier:
    """classify(prompt) -> (Domain, confidence in [0, 1])."""

    def __init__(
        self,
        embed_fn: Callable[[list[str]], list[list[float]]] | None | str = "auto",
        exemplars: dict[Domain, list[str]] | None = None,
        llm_classify: Callable[[str], Domain] | None = None,
        ambiguity_margin: float = 0.05,
    ):
        self.embed_fn = _load_default_embedder() if embed_fn == "auto" else embed_fn
        self.exemplars = exemplars or EXEMPLARS
        self.llm_classify = llm_classify
        self.ambiguity_margin = ambiguity_margin
        self._exemplar_vecs: dict[Domain, list[list[float]]] | None = None

    def _exemplar_embeddings(self) -> dict[Domain, list[list[float]]]:
        if self._exemplar_vecs is None:
            assert self.embed_fn is not None
            self._exemplar_vecs = {
                d: self.embed_fn(texts) for d, texts in self.exemplars.items()
            }
        return self._exemplar_vecs

    def classify(self, prompt: str) -> tuple[Domain, float]:
        if self.embed_fn is None:
            return heuristic_domain(prompt), 0.6
        [pvec] = self.embed_fn([prompt])
        scores = {
            d: max(_cosine(pvec, v) for v in vecs)
            for d, vecs in self._exemplar_embeddings().items()
        }
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        (top_domain, top), runner_up = ranked[0], (ranked[1][1] if len(ranked) > 1 else -1.0)
        if top - runner_up < self.ambiguity_margin and self.llm_classify is not None:
            return self.llm_classify(prompt), max(top, 0.0)
        return top_domain, max(top, 0.0)

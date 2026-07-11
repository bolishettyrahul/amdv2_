"""Free verification gates for LLM answers.

These are the weak-proxy checks (self-consistency, schema validation, summary
heuristics) used by Stages 2 and 3 for domains without a deterministic tool.
All pure functions — no network, no LLM.
"""

from __future__ import annotations

import json
import re
from collections import Counter

_ARTICLES = re.compile(r"^(the|a|an)\s+", re.I)
_PUNCT = re.compile(r"[^\w\s]")
_FENCE = re.compile(r"```[a-zA-Z0-9]*\s*\n(.*?)```", re.S)
_STOPWORDS = {"the", "a", "an", "this", "that", "it", "he", "she", "they", "we", "i"}

SENTIMENT_LABELS = ("positive", "negative", "neutral")


def normalize_answer(text: str) -> str:
    text = _PUNCT.sub(" ", text.strip().lower())
    text = re.sub(r"\s+", " ", text).strip()
    return _ARTICLES.sub("", text)


def answers_agree(a: str, b: str) -> bool:
    na, nb = normalize_answer(a), normalize_answer(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    shorter, longer = sorted((na, nb), key=len)
    return len(shorter) >= 3 and shorter in longer


def majority_answer(samples: list[str]) -> tuple[str, float]:
    """Most common (normalized) answer and its agreement ratio."""
    groups: dict[str, list[str]] = {}
    for s in samples:
        groups.setdefault(normalize_answer(s), []).append(s)
    key, members = max(groups.items(), key=lambda kv: len(kv[1]))
    return members[0], len(members) / len(samples)


def valid_sentiment_label(text: str) -> str | None:
    """Label schema check: exactly one sentiment label mentioned."""
    low = text.lower()
    found = [lbl for lbl in SENTIMENT_LABELS if lbl in low]
    return found[0] if len(found) == 1 else None


def _key_entities(source: str, top_n: int = 3) -> list[str]:
    tokens = re.findall(r"\b(?:[A-Z][a-z]+|[A-Z]{2,})\b", source)
    counts = Counter(t for t in tokens if t.lower() not in _STOPWORDS)
    return [t for t, _ in counts.most_common(top_n)]


def summary_ok(
    source: str,
    summary: str,
    min_ratio: float = 0.02,
    max_ratio: float = 0.5,
    min_words: int = 5,
    min_entity_coverage: float = 0.5,
) -> tuple[bool, str]:
    """Heuristic: length bounds + key-entity coverage."""
    src_words, sum_words = len(source.split()), len(summary.split())
    if src_words == 0 or sum_words < min_words:
        return False, f"summary too short ({sum_words} words)"
    ratio = sum_words / src_words
    if not (min_ratio <= ratio <= max_ratio):
        return False, f"compression ratio {ratio:.2f} outside [{min_ratio}, {max_ratio}]"
    entities = _key_entities(source)
    if entities:
        low = summary.lower()
        covered = sum(1 for e in entities if e.lower() in low)
        if covered / len(entities) < min_entity_coverage:
            return False, f"key entities missing: covered {covered}/{len(entities)} of {entities}"
    return True, f"ratio {ratio:.2f} in bounds, entity coverage ok"


def valid_ner_entities(text: str) -> list[dict] | None:
    """Parse and schema-check an LLM's NER output (possibly fenced JSON)."""
    fenced = _FENCE.search(text)
    candidate = fenced.group(1) if fenced else text
    start, end = candidate.find("["), candidate.rfind("]")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(candidate[start : end + 1])
    except ValueError:
        return None
    if not isinstance(parsed, list):
        return None
    for item in parsed:
        if not isinstance(item, dict):
            return None
        if not isinstance(item.get("text"), str) or not item["text"]:
            return None
        if not isinstance(item.get("label"), str) or not item["label"]:
            return None
    return parsed


def extract_code_block(text: str) -> str:
    match = _FENCE.search(text)
    return (match.group(1) if match else text).strip()

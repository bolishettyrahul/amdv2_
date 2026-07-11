"""Stage 1 sentiment tool — VADER lexicon, verifier = confidence >= threshold."""

from __future__ import annotations

import re

from router.types import ToolResult

_analyzer = None


def _get_analyzer():
    global _analyzer
    if _analyzer is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        _analyzer = SentimentIntensityAnalyzer()
    return _analyzer


_QUOTED = re.compile(r"[\"'](.+?)[\"']", re.S)


def extract_target_text(prompt: str) -> str:
    """Pull the text-to-classify out of an instruction-style prompt."""
    quoted = sorted(_QUOTED.findall(prompt), key=len, reverse=True)
    if quoted and len(quoted[0]) >= 10:
        return quoted[0].strip()
    head, sep, tail = prompt.partition(":")
    if sep and tail.strip():
        return tail.strip()
    return prompt.strip()


def classify_sentiment(text: str, threshold: float = 0.5) -> ToolResult:
    compound = _get_analyzer().polarity_scores(text)["compound"]
    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"
    confidence = abs(compound)
    return ToolResult(
        label,
        verified=confidence >= threshold,
        confidence=confidence,
        reason=f"VADER compound={compound:.3f}, threshold={threshold}",
    )

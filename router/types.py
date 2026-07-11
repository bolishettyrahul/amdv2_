"""Core types shared across the cascade."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Domain(str, Enum):
    FACTUAL = "factual_knowledge"
    MATH = "math_reasoning"
    SENTIMENT = "sentiment_classification"
    SUMMARIZATION = "summarization"
    NER = "ner"
    CODE_DEBUG = "code_debugging"
    LOGIC = "logical_reasoning"
    CODE_GEN = "code_generation"


@dataclass
class Task:
    task_id: str
    prompt: str
    # Extra dataset columns (e.g. test cases for code domains, reference text).
    metadata: dict = field(default_factory=dict)


@dataclass
class ToolResult:
    """Outcome of a Stage 1 deterministic tool. verified=False means escalate."""

    answer: str | None
    verified: bool
    confidence: float = 0.0
    reason: str = ""


@dataclass
class StageResult:
    """Outcome of one cascade stage attempt for a task."""

    answer: str | None
    stage: str  # "stage1_tool" | "stage2_local" | "stage3_paid"
    verified: bool
    verifier: str = ""
    verifier_reason: str = ""
    model: str | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0

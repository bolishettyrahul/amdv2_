"""Token cost accounting.

Every paid-provider call is recorded against this price table so the
accuracy / total-paid-token-cost score is measurable during development.
Prices are USD per 1M tokens (input, output), from plan/fireworks-model-catalog.md.
Models with a (0.0, 0.0) entry are free tiers (local Ollama, Groq test credits
are tracked separately but priced at their Fireworks-equivalent during eval).
"""

from __future__ import annotations

from dataclasses import dataclass, field

# model id -> (usd per 1M input tokens, usd per 1M output tokens)
PRICE_TABLE: dict[str, tuple[float, float]] = {
    # Live serverless catalog (docs.fireworks.ai/serverless/pricing, 2026-07).
    "accounts/fireworks/models/gpt-oss-120b": (0.15, 0.60),
    "accounts/fireworks/models/kimi-k2p6": (0.95, 4.00),
    "accounts/fireworks/models/glm-5p1": (1.40, 4.40),
    "accounts/fireworks/models/glm-5p2": (1.40, 4.40),
    "accounts/fireworks/models/deepseek-v4-pro": (1.74, 3.48),
    # Retired catalog generation, kept for replaying old logs.
    "accounts/fireworks/models/llama-v3p1-8b-instruct": (0.20, 0.20),
    "accounts/fireworks/models/gemma2-9b-it": (0.20, 0.20),
    "accounts/fireworks/models/gemma2-27b-it": (0.80, 0.80),
    "accounts/fireworks/models/llama-v3p1-70b-instruct": (0.90, 0.90),
    "accounts/fireworks/models/qwen2p5-72b-instruct": (0.90, 0.90),
    "accounts/fireworks/models/llama-v3p1-405b-instruct": (4.00, 4.00),
    "accounts/fireworks/models/llama4-maverick-instruct-basic": (0.22, 0.88),
}

_FREE_PREFIXES = ("ollama/",)


def cost_usd(model: str, *, tokens_in: int, tokens_out: int) -> float:
    """Cost of one call. Free local models cost 0; unknown paid models raise KeyError."""
    if model.startswith(_FREE_PREFIXES):
        return 0.0
    price_in, price_out = PRICE_TABLE[model]
    return (tokens_in * price_in + tokens_out * price_out) / 1_000_000


@dataclass(frozen=True)
class CostRecord:
    model: str
    tokens_in: int
    tokens_out: int
    cost_usd: float


@dataclass
class CostTracker:
    records: list[CostRecord] = field(default_factory=list)

    def record(self, model: str, *, tokens_in: int, tokens_out: int) -> CostRecord:
        rec = CostRecord(model, tokens_in, tokens_out,
                         cost_usd(model, tokens_in=tokens_in, tokens_out=tokens_out))
        self.records.append(rec)
        return rec

    @property
    def total_cost_usd(self) -> float:
        return sum(r.cost_usd for r in self.records)

    @property
    def total_tokens_in(self) -> int:
        return sum(r.tokens_in for r in self.records)

    @property
    def total_tokens_out(self) -> int:
        return sum(r.tokens_out for r in self.records)

"""Post-hoc LLM critique — dataset labeling only, never live routing.

Skipped for the 5 deterministically-verified domains. For the 3 weak-verifier
domains it labels the returned answer: primary critic = an OpenRouter
free-tier model, fallback = local mistral-nemo (distinct lineage from every
actor model). Hard rule enforced here: the critic must never be the model
that produced the answer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from router.providers.base import LLMProvider, ProviderError
from router.types import Domain, Task

DETERMINISTIC_DOMAINS = frozenset(
    {Domain.MATH, Domain.SENTIMENT, Domain.NER, Domain.CODE_DEBUG, Domain.CODE_GEN}
)

_VERDICT = re.compile(r"verdict\s*[:\-]?\s*(correct|incorrect)", re.I)

_CRITIC_PROMPT = (
    "You are grading another model's answer.\n\nTASK:\n{prompt}\n\nANSWER:\n{answer}\n\n"
    "Reply in exactly this format:\nVERDICT: correct|incorrect\nREASONING: <one short paragraph>"
)


@dataclass(frozen=True)
class CritiqueResult:
    verdict: str  # "correct" | "incorrect" | "unparseable"
    reasoning: str
    critic_model: str


def _bare_model(model: str | None) -> str:
    return (model or "").split("/")[-1].lower()


class Critic:
    def __init__(self, primary_provider: LLMProvider, primary_model: str,
                 fallback_provider: LLMProvider, fallback_model: str):
        self.primary = (primary_provider, primary_model)
        self.fallback = (fallback_provider, fallback_model)

    def critique(self, task: Task, domain: Domain, answer: str,
                 actor_model: str | None) -> CritiqueResult | None:
        if domain in DETERMINISTIC_DOMAINS:
            return None  # tool result already is ground truth

        candidates = [c for c in (self.primary, self.fallback)
                      if _bare_model(c[1]) != _bare_model(actor_model)]
        if not candidates:
            raise ValueError(f"no critic distinct from actor model {actor_model!r}")

        message = [{"role": "user", "content":
                    _CRITIC_PROMPT.format(prompt=task.prompt, answer=answer)}]
        last_error: Exception | None = None
        for provider, model in candidates:
            try:
                resp = provider.chat(message, model=model, temperature=0.0)
            except ProviderError as exc:
                last_error = exc
                continue
            match = _VERDICT.search(resp.text)
            verdict = match.group(1).lower() if match else "unparseable"
            reasoning = resp.text.split("REASONING:", 1)[-1].strip() if "REASONING:" in resp.text \
                else resp.text.strip()
            return CritiqueResult(verdict=verdict, reasoning=reasoning, critic_model=model)
        raise ValueError(f"all critics unavailable (last error: {last_error})")

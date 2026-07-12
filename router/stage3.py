"""Stage 3 — paid escalation via Dynamic Fireworks Routing.

Picks the cheapest model capable of the task (domain x estimated complexity,
per plan/fireworks-model-catalog.md), verifies the answer with the same free
gates, and on failure runs an agentic self-correction loop: retry with the
failure context, then one attempt on a stronger model. The final stage always
returns an answer, verified or not.
"""

from __future__ import annotations

import re

from router.cost import CostTracker
from router.domain_verify import verify_domain_answer
from router.providers.base import LLMProvider
from router.types import Domain, StageResult, Task

STAGE = "stage3_paid"

M8 = "accounts/fireworks/models/llama-v3p1-8b-instruct"
G9 = "accounts/fireworks/models/gemma2-9b-it"
G27 = "accounts/fireworks/models/gemma2-27b-it"
L70 = "accounts/fireworks/models/llama-v3p1-70b-instruct"
Q72 = "accounts/fireworks/models/qwen2p5-72b-instruct"
L405 = "accounts/fireworks/models/llama-v3p1-405b-instruct"
MAVERICK = "accounts/fireworks/models/llama4-maverick-instruct-basic"

ROUTING: dict[Domain, dict[str, str]] = {
    Domain.FACTUAL: {"simple": G9, "complex": L70},
    Domain.MATH: {"simple": G9, "complex": Q72},
    Domain.SENTIMENT: {"simple": M8, "complex": M8},
    Domain.SUMMARIZATION: {"simple": M8, "complex": L70},
    Domain.NER: {"simple": M8, "complex": M8},
    Domain.CODE_DEBUG: {"simple": M8, "complex": L70},
    Domain.LOGIC: {"simple": G27, "complex": L70},
    Domain.CODE_GEN: {"simple": M8, "complex": Q72},
}

# Self-correction escalation ladder: where to go when the routed model keeps
# failing the verification gate. Use the highly cost-effective 400B MAVERICK
# model instead of the expensive L405.
ESCALATION: dict[str, str] = {M8: L70, G9: L70, G27: L70, Q72: MAVERICK, L70: MAVERICK, L405: MAVERICK, MAVERICK: MAVERICK}

_COMPLEX_CUES = re.compile(
    r"\bprove\b|\bproof\b|\bcompare\b|\bmulti-?hop\b|\bconcurrenc|\bthread|\barchitect"
    r"|\bstep by step\b|\bboth\b.*\band\b", re.I,
)
_COMPLEX_PROMPT_CHARS = 1500


def estimate_complexity(task: Task, domain: Domain) -> str:
    if len(task.prompt) > _COMPLEX_PROMPT_CHARS:
        return "complex"
    if _COMPLEX_CUES.search(task.prompt):
        return "complex"
    if len(task.metadata.get("code") or "") > 800:
        return "complex"
    return "simple"


def choose_model(domain: Domain, complexity: str) -> str:
    return ROUTING[domain][complexity]


_SYSTEM_PROMPTS: dict[Domain, str] = {
    Domain.FACTUAL: "Answer the question directly and concisely.",
    Domain.MATH: ("Solve the problem. Rewrite it as a single arithmetic expression or "
                  "equation when possible and output ONLY that expression; otherwise give "
                  "the final numeric answer only."),
    Domain.SENTIMENT: "Classify the sentiment. Answer with exactly one word: "
                      "positive, negative, or neutral.",
    Domain.SUMMARIZATION: "Summarize the text concisely, keeping the key names and facts.",
    Domain.NER: ('Extract named entities. Output ONLY a JSON list like '
                 '[{"text": "...", "label": "PERSON|ORG|GPE|DATE|..."}].'),
    Domain.CODE_DEBUG: "Fix the code. Output ONLY the corrected code in a ```python``` block.",
    Domain.LOGIC: "Solve the problem carefully. State only the final answer.",
    Domain.CODE_GEN: "Write the requested code. Output ONLY the code in a ```python``` block.",
}

_WEAK_DOMAINS = (Domain.FACTUAL, Domain.LOGIC)


class Stage3Paid:
    def __init__(self, provider: LLMProvider, cost_tracker: CostTracker | None = None,
                 max_attempts: int = 3, sandbox_timeout: float = 2.0):
        self.provider = provider
        self.cost_tracker = cost_tracker or CostTracker()
        self.max_attempts = max_attempts
        self.sandbox_timeout = sandbox_timeout

    def attempt(self, task: Task, domain: Domain) -> StageResult:
        routed = choose_model(domain, estimate_complexity(task, domain))
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPTS[domain]},
            {"role": "user", "content": self._user_content(task, domain)},
        ]
        model = routed
        tokens_in = tokens_out = 0
        cost = 0.0
        answer: str | None = None
        verified, verifier, reason = False, "", ""

        for attempt_idx in range(self.max_attempts):
            if attempt_idx == self.max_attempts - 1:
                model = ESCALATION[routed]  # last try: stronger model
            resp = self.provider.chat(messages, model=model, temperature=0.0)
            rec = self.cost_tracker.record(model, tokens_in=resp.tokens_in,
                                           tokens_out=resp.tokens_out)
            tokens_in += resp.tokens_in
            tokens_out += resp.tokens_out
            cost += rec.cost_usd

            answer, verified, verifier, reason = verify_domain_answer(
                task, domain, [resp.text], sandbox_timeout=self.sandbox_timeout)
            if answer is None:
                answer = resp.text
            if verified or domain in _WEAK_DOMAINS:
                break
            # Agentic self-correction: feed the failure back and retry.
            messages = messages + [
                {"role": "assistant", "content": resp.text},
                {"role": "user", "content":
                    f"Your answer failed verification: {reason}\n"
                    "Correct the problem and answer again in the same format."},
            ]

        return StageResult(answer, STAGE, verified=verified, verifier=verifier,
                           verifier_reason=reason, model=model,
                           tokens_in=tokens_in, tokens_out=tokens_out, cost_usd=cost)

    @staticmethod
    def _user_content(task: Task, domain: Domain) -> str:
        if domain in (Domain.CODE_DEBUG, Domain.CODE_GEN):
            parts = [task.prompt]
            if task.metadata.get("code"):
                parts.append(f"```python\n{task.metadata['code']}\n```")
            if task.metadata.get("tests"):
                parts.append(f"It must pass these tests:\n{task.metadata['tests']}")
            return "\n\n".join(parts)
        return task.prompt

"""Stage 3 — paid escalation via Dynamic Fireworks Routing.

Picks the cheapest model capable of the task (domain x estimated complexity,
per plan/fireworks-model-catalog.md), verifies the answer with the same free
gates, and on failure runs an agentic self-correction loop: retry with the
failure context, then one attempt on a stronger model. The final stage always
returns an answer, verified or not.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from router.cost import CostTracker
from router.domain_verify import verify_domain_answer
from router.providers.base import LLMProvider
from router.types import Domain, StageResult, Task

STAGE = "stage3_paid"

# Live Fireworks serverless catalog (verified 2026-07 against GET /v1/models —
# the entire llama/gemma/qwen generation was retired). Standard-tier prices
# per 1M tokens (in/out) from docs.fireworks.ai/serverless/pricing:
OSS120 = "accounts/fireworks/models/gpt-oss-120b"    # $0.15/$0.60 — cheap tier
KIMI = "accounts/fireworks/models/kimi-k2p6"         # $0.95/$4.00 — mid reasoning
GLM51 = "accounts/fireworks/models/glm-5p1"          # $1.40/$4.40
GLM52 = "accounts/fireworks/models/glm-5p2"          # $1.40/$4.40 — strong coder
DSV4 = "accounts/fireworks/models/deepseek-v4-pro"   # $1.74/$3.48 — apex

ROUTING: dict[Domain, dict[str, str]] = {
    Domain.FACTUAL: {"simple": OSS120, "complex": KIMI},
    Domain.MATH: {"simple": OSS120, "complex": KIMI},
    Domain.SENTIMENT: {"simple": OSS120, "complex": OSS120},
    # Summarization is input-heavy, so the cheap-input mid tier wins there.
    Domain.SUMMARIZATION: {"simple": OSS120, "complex": KIMI},
    Domain.NER: {"simple": OSS120, "complex": OSS120},
    Domain.CODE_DEBUG: {"simple": OSS120, "complex": GLM52},
    Domain.LOGIC: {"simple": OSS120, "complex": KIMI},
    Domain.CODE_GEN: {"simple": OSS120, "complex": GLM52},
}

# Self-correction escalation ladder: where to go when the routed model keeps
# failing the verification gate. DSV4 is the apex — cheapest output $/M among
# the frontier tier, which matters because retries carry long outputs.
ESCALATION: dict[str, str] = {OSS120: KIMI, KIMI: DSV4, GLM51: GLM52, GLM52: DSV4, DSV4: DSV4}

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


def resolve_model(domain: Domain, complexity: str,
                  allowed: Sequence[str]) -> tuple[str, bool]:
    """Constrain the routed model to the harness allowlist.

    Empty `allowed` means no restriction (local dev). Otherwise fall back in
    order: routed model -> the domain's other complexity tier -> the routed
    model's escalation target -> the first allowed model as a flagged last
    resort. Returns (model, last_resort_fallback_used).
    """
    routed = choose_model(domain, complexity)
    if not allowed:
        return routed, False
    allowed_set = set(allowed)
    if routed in allowed_set:
        return routed, False
    other_tier = ROUTING[domain]["complex" if complexity == "simple" else "simple"]
    if other_tier in allowed_set:
        return other_tier, False
    if ESCALATION.get(routed) in allowed_set:
        return ESCALATION[routed], False
    return allowed[0], True


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
                 max_attempts: int = 3, sandbox_timeout: float = 2.0,
                 allowed_models: Sequence[str] | None = None):
        self.provider = provider
        self.cost_tracker = cost_tracker or CostTracker()
        self.max_attempts = max_attempts
        self.sandbox_timeout = sandbox_timeout
        self.allowed_models = list(allowed_models or [])

    def _escalation_target(self, model: str) -> str:
        target = ESCALATION.get(model, model)
        if self.allowed_models and target not in self.allowed_models:
            return model  # stronger model not allowed: stay on the allowed one
        return target

    def attempt(self, task: Task, domain: Domain) -> StageResult:
        routed, allowed_fallback = resolve_model(
            domain, estimate_complexity(task, domain), self.allowed_models)
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
                model = self._escalation_target(routed)  # last try: stronger model
            resp = self.provider.chat(messages, model=model, temperature=0.0)
            tokens_in += resp.tokens_in
            tokens_out += resp.tokens_out
            try:
                rec = self.cost_tracker.record(model, tokens_in=resp.tokens_in,
                                               tokens_out=resp.tokens_out)
                cost += rec.cost_usd
            except KeyError:
                # Allowlisted model outside our price table: an unknown price
                # must not sink the task; tokens are still counted above.
                pass

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
                           tokens_in=tokens_in, tokens_out=tokens_out, cost_usd=cost,
                           allowed_models_fallback=allowed_fallback)

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

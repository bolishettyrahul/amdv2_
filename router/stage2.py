"""Stage 2 — local model attempt (free: GPU compute only) with free verification.

In cloud-fallback mode (grading sandbox without a working GPU) the same stage
runs against a cheap paid model (`paid=True`), and its tokens are costed.
"""

from __future__ import annotations

from router.cost import cost_usd
from router.domain_verify import verify_domain_answer
from router.providers.base import LLMProvider
from router.types import Domain, StageResult, Task

STAGE = "stage2_local"

DEFAULT_MODELS: dict[Domain, str] = {
    Domain.FACTUAL: "qwen2.5:7b",
    Domain.MATH: "qwen2.5:7b",
    Domain.SENTIMENT: "phi3.5:3.8b",
    Domain.SUMMARIZATION: "llama3.2:3b",
    Domain.NER: "qwen2.5:7b",
    Domain.CODE_DEBUG: "qwen2.5:7b",
    Domain.LOGIC: "qwen2.5:7b",
    Domain.CODE_GEN: "qwen2.5:7b",
}

_PROMPTS: dict[Domain, str] = {
    Domain.FACTUAL: "Answer the question directly and concisely. No explanation.",
    Domain.MATH: ("Rewrite the problem as a single arithmetic expression or equation and "
                  "output ONLY that expression, nothing else."),
    Domain.SENTIMENT: "Classify the sentiment. Answer with exactly one word: "
                      "positive, negative, or neutral.",
    Domain.SUMMARIZATION: "Summarize the text concisely, keeping the key names and facts.",
    Domain.NER: ('Extract named entities. Output ONLY a JSON list like '
                 '[{"text": "...", "label": "PERSON|ORG|GPE|DATE|..."}].'),
    Domain.CODE_DEBUG: "Fix the code. Output ONLY the corrected code in a ```python``` block.",
    Domain.LOGIC: "Solve the problem. State the final answer on the last line.",
    Domain.CODE_GEN: "Write the requested code. Output ONLY the code in a ```python``` block.",
}


class Stage2Local:
    def __init__(
        self,
        provider: LLMProvider,
        models: dict[Domain, str] | None = None,
        factual_k: int = 2,
        logic_k: int = 3,
        factual_min_agreement: float = 1.0,
        logic_min_agreement: float = 0.6,
        sandbox_timeout: float = 2.0,
        paid: bool = False,
    ):
        self.provider = provider
        self.models = {**DEFAULT_MODELS, **(models or {})}
        self.factual_k = factual_k
        self.logic_k = logic_k
        self.factual_min_agreement = factual_min_agreement
        self.logic_min_agreement = logic_min_agreement
        self.sandbox_timeout = sandbox_timeout
        self.paid = paid

    def attempt(self, task: Task, domain: Domain) -> StageResult:
        model = self.models[domain]
        samples, tokens_in, tokens_out = self._sample(task, domain, model)
        answer, verified, verifier, reason = verify_domain_answer(
            task, domain, samples,
            sandbox_timeout=self.sandbox_timeout,
            factual_min_agreement=self.factual_min_agreement,
            logic_min_agreement=self.logic_min_agreement,
        )
        qualified = model if self.paid else f"ollama/{model}"
        return StageResult(
            answer, STAGE, verified=verified, verifier=verifier, verifier_reason=reason,
            model=qualified, tokens_in=tokens_in, tokens_out=tokens_out,
            cost_usd=cost_usd(qualified, tokens_in=tokens_in, tokens_out=tokens_out),
        )

    def _sample(self, task: Task, domain: Domain, model: str) -> tuple[list[str], int, int]:
        k = {Domain.FACTUAL: self.factual_k, Domain.LOGIC: self.logic_k}.get(domain, 1)
        messages = [
            {"role": "system", "content": _PROMPTS[domain]},
            {"role": "user", "content": self._user_content(task, domain)},
        ]
        samples: list[str] = []
        tokens_in = tokens_out = 0
        for i in range(k):
            resp = self.provider.chat(messages, model=model,
                                      temperature=0.0 if i == 0 else 0.7)
            samples.append(resp.text)
            tokens_in += resp.tokens_in
            tokens_out += resp.tokens_out
        return samples, tokens_in, tokens_out

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

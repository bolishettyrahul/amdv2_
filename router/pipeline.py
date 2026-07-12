"""The Phase 1 cascade orchestrator.

Stage 0 classify -> Stage 1 tool -> Stage 2 local -> Stage 3 paid, with a free
verification gate after Stages 1 and 2 (pass = return at $0). Every task
appends exactly one structured JSONL record. Critique labels weak-verifier
domains post-hoc; it never alters or blocks the answer already produced.
"""

from __future__ import annotations

import time

from router.critique import DETERMINISTIC_DOMAINS
from router.task_log import TaskLogger
from router.types import Domain, StageResult, Task


class Pipeline:
    def __init__(self, classifier, stage1, stage2, stage3, critic, logger: TaskLogger):
        self.classifier = classifier
        self.stage1 = stage1
        self.stage2 = stage2
        self.stage3 = stage3
        self.critic = critic
        self.logger = logger

    def process(self, task: Task) -> dict:
        started = time.perf_counter()
        domain, domain_confidence = self.classifier.classify(task.prompt)

        attempts: list[StageResult] = []
        final: StageResult | None = None
        for stage in (self.stage1, self.stage2, self.stage3):
            try:
                result = stage.attempt(task, domain)
            except Exception as exc:  # a broken stage escalates, never crashes the batch
                result = StageResult(None, getattr(stage, "STAGE", stage.__class__.__name__),
                                     verified=False, verifier_reason=f"stage error: {exc}")
            attempts.append(result)
            if result.verified:
                final = result
                break
        if final is None:
            # Nothing verified: return the last stage's (paid) answer, else the
            # best unverified answer any stage produced.
            final = next((r for r in reversed(attempts) if r.answer is not None), attempts[-1])

        critique = self._critique(task, domain, final)
        record = {
            "task_id": task.task_id,
            "prompt": task.prompt,
            "domain": domain.value,
            "domain_confidence": round(float(domain_confidence), 4),
            "stage": final.stage,
            "answer": final.answer,
            "verified": final.verified,
            "verifier": final.verifier,
            "verifier_reason": final.verifier_reason,
            "model": final.model,
            "critique": critique,
            "stages_attempted": [r.stage for r in attempts],
            "tokens_in": sum(r.tokens_in for r in attempts),
            "tokens_out": sum(r.tokens_out for r in attempts),
            "cost_usd": sum(r.cost_usd for r in attempts),
            "latency_s": round(time.perf_counter() - started, 3),
        }
        if any(getattr(r, "allowed_models_fallback", False) for r in attempts):
            # Warning, not silent: the ALLOWED_MODELS last-resort fallback fired.
            record["allowed_models_fallback"] = True
        self.logger.append(record)
        return record

    def _critique(self, task: Task, domain: Domain, final: StageResult) -> dict | None:
        if domain in DETERMINISTIC_DOMAINS or final.answer is None:
            return None
        try:
            result = self.critic.critique(task, domain, final.answer, actor_model=final.model)
        except Exception as exc:
            return {"error": str(exc)}
        if result is None:
            return None
        return {"verdict": result.verdict, "reasoning": result.reasoning,
                "critic_model": result.critic_model}

"""Per-domain free verification of LLM answers — shared by Stages 2 and 3."""

from __future__ import annotations

import json

from router.tools.math_tool import solve_math
from router.tools.sentiment_tool import extract_target_text
from router.types import Domain, Task
from router.verifiers import (
    extract_code_block,
    majority_answer,
    summary_ok,
    valid_ner_entities,
    valid_sentiment_label,
)

Verdict = tuple[str | None, bool, str, str]  # answer, verified, verifier, reason


def verify_domain_answer(
    task: Task,
    domain: Domain,
    samples: list[str],
    *,
    sandbox_timeout: float = 2.0,
    factual_min_agreement: float = 1.0,
    logic_min_agreement: float = 0.6,
) -> Verdict:
    if domain in (Domain.FACTUAL, Domain.LOGIC):
        min_agree = factual_min_agreement if domain == Domain.FACTUAL else logic_min_agreement
        answer, ratio = majority_answer(samples)
        if len(samples) == 1:
            return answer, False, "none (weak-verifier domain, single sample)", "accepted as-is"
        return (answer, ratio >= min_agree, "self-consistency",
                f"agreement {ratio:.2f} (need >= {min_agree})")

    text = samples[0]
    if domain == Domain.MATH:
        result = solve_math(text)
        return result.answer, result.verified, "tool re-verification", result.reason
    if domain == Domain.SENTIMENT:
        label = valid_sentiment_label(text)
        return label, label is not None, "label schema check", f"raw: {text[:80]!r}"
    if domain == Domain.SUMMARIZATION:
        ok, reason = summary_ok(extract_target_text(task.prompt), text)
        return text, ok, "length bounds + key-entity coverage", reason
    if domain == Domain.NER:
        entities = valid_ner_entities(text)
        if entities is None:
            return None, False, "NER schema validation", "output is not a valid entity list"
        return (json.dumps(entities, ensure_ascii=False), True,
                "NER schema validation", f"{len(entities)} entities")

    # Code domains: ground truth = sandboxed test execution.
    from router.tools.sandbox import run_python

    code = extract_code_block(text)
    tests = task.metadata.get("tests")
    if not tests:
        return code, False, "sandboxed test execution", "no tests provided in task"
    run = run_python(code, tests=tests, timeout=sandbox_timeout)
    reason = "tests pass" if run.passed else f"tests fail: {run.stderr[-400:]}"
    return code, run.passed, "sandboxed test execution", reason

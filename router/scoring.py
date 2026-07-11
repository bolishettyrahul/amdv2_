"""Offline scoring for threshold calibration: Score = accuracy / total paid cost."""

from __future__ import annotations

import re

from router.verifiers import normalize_answer


def grade_answer(produced: str | None, expected: str) -> bool:
    if not produced:
        return False
    got, want = normalize_answer(produced), normalize_answer(expected)
    if not got or not want:
        return False
    if got == want:
        return True
    # The expected answer appearing as a whole word inside the produced answer
    # counts ("The answer is 42" vs "42"); the reverse does not.
    return re.search(rf"(?<!\w){re.escape(want)}(?!\w)", got) is not None


def score_run(records: list[dict], expected: list[str]) -> dict:
    correct = sum(1 for rec, exp in zip(records, expected)
                  if grade_answer(rec.get("answer"), exp))
    accuracy = correct / len(records) if records else 0.0
    total_cost = sum(rec.get("cost_usd", 0.0) for rec in records)
    score = accuracy / total_cost if total_cost > 0 else (
        float("inf") if accuracy > 0 else 0.0)
    return {"accuracy": accuracy, "correct": correct, "tasks": len(records),
            "total_cost_usd": total_cost, "score": score}

"""Offline evaluation: accuracy / total paid token cost on a labeled dataset."""

import pytest

from router.scoring import grade_answer, score_run


def test_grade_answer_normalized_match():
    assert grade_answer("Canberra", " canberra. ")
    assert grade_answer("The answer is 42", "42")
    assert not grade_answer("Sydney", "Canberra")


def test_score_run_computes_accuracy_over_cost():
    records = [
        {"answer": "4", "cost_usd": 0.0},
        {"answer": "Canberra", "cost_usd": 0.002},
        {"answer": "wrong", "cost_usd": 0.001},
    ]
    expected = ["4", "Canberra", "right"]
    result = score_run(records, expected)
    assert result["accuracy"] == pytest.approx(2 / 3)
    assert result["total_cost_usd"] == pytest.approx(0.003)
    assert result["score"] == pytest.approx((2 / 3) / 0.003)


def test_score_run_zero_cost_scores_infinite():
    result = score_run([{"answer": "4", "cost_usd": 0.0}], ["4"])
    assert result["accuracy"] == 1.0
    assert result["score"] == float("inf")

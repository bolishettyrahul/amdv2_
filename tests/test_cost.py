"""Cost tracking: every paid call records tokens and computed $ cost against a price table."""

import pytest

from router.cost import PRICE_TABLE, CostTracker, cost_usd


def test_price_table_covers_all_catalog_models():
    expected = {
        "accounts/fireworks/models/llama-v3p1-8b-instruct",
        "accounts/fireworks/models/gemma2-9b-it",
        "accounts/fireworks/models/gemma2-27b-it",
        "accounts/fireworks/models/llama-v3p1-70b-instruct",
        "accounts/fireworks/models/qwen2p5-72b-instruct",
        "accounts/fireworks/models/llama-v3p1-405b-instruct",
    }
    assert expected <= set(PRICE_TABLE)


def test_cost_usd_computes_from_per_million_prices():
    # Llama 3.1 8B: $0.20 in / $0.20 out per 1M tokens
    model = "accounts/fireworks/models/llama-v3p1-8b-instruct"
    assert cost_usd(model, tokens_in=1_000_000, tokens_out=0) == pytest.approx(0.20)
    assert cost_usd(model, tokens_in=500_000, tokens_out=500_000) == pytest.approx(0.20)


def test_cost_usd_unknown_model_raises():
    with pytest.raises(KeyError):
        cost_usd("not-a-model", tokens_in=10, tokens_out=10)


def test_free_models_cost_zero():
    # Local Ollama models are free regardless of token counts.
    assert cost_usd("ollama/qwen2.5:7b", tokens_in=1_000_000, tokens_out=1_000_000) == 0.0


def test_tracker_accumulates_records():
    tracker = CostTracker()
    model = "accounts/fireworks/models/gemma2-9b-it"
    rec1 = tracker.record(model, tokens_in=1000, tokens_out=2000)
    rec2 = tracker.record(model, tokens_in=1000, tokens_out=0)
    assert rec1.cost_usd == pytest.approx((1000 * 0.20 + 2000 * 0.20) / 1_000_000)
    assert tracker.total_cost_usd == pytest.approx(rec1.cost_usd + rec2.cost_usd)
    assert tracker.total_tokens_in == 2000
    assert tracker.total_tokens_out == 2000
    assert len(tracker.records) == 2

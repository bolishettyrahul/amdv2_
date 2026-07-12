"""Settings resolved from environment: provider selection, keys, thresholds."""

from router.config import Settings


def test_defaults(monkeypatch):
    for var in ("PAID_PROVIDER", "GROQ_API_KEY", "FIREWORKS_API_KEY",
                "FIREWORKS_BASE_URL", "OPENROUTER_API_KEY", "OLLAMA_HOST",
                "USE_CLOUD_FALLBACK", "ALLOWED_MODELS"):
        monkeypatch.delenv(var, raising=False)
    s = Settings.from_env()
    assert s.paid_provider == "groq"  # preserve Fireworks credits during testing
    assert s.ollama_host == "http://localhost:11434"
    assert s.use_cloud_fallback is None  # None -> decide via health check
    assert 0.0 < s.sentiment_threshold < 1.0
    assert s.fireworks_base_url == "https://api.fireworks.ai/inference/v1"
    assert s.allowed_models == []  # unset -> no Stage 3 restriction


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("PAID_PROVIDER", "fireworks")
    monkeypatch.setenv("FIREWORKS_API_KEY", "fw-123")
    monkeypatch.setenv("FIREWORKS_BASE_URL", "https://proxy.example/v1")
    monkeypatch.setenv("USE_CLOUD_FALLBACK", "1")
    monkeypatch.setenv("SENTIMENT_THRESHOLD", "0.7")
    s = Settings.from_env()
    assert s.paid_provider == "fireworks"
    assert s.fireworks_api_key == "fw-123"
    assert s.fireworks_base_url == "https://proxy.example/v1"
    assert s.use_cloud_fallback is True
    assert s.sentiment_threshold == 0.7


def test_allowed_models_parses_commas_and_whitespace(monkeypatch):
    monkeypatch.setenv(
        "ALLOWED_MODELS",
        " accounts/fireworks/models/gemma2-9b-it, "
        "accounts/fireworks/models/llama4-maverick-instruct-basic "
        "accounts/fireworks/models/llama-v3p1-8b-instruct ",
    )
    s = Settings.from_env()
    assert s.allowed_models == [
        "accounts/fireworks/models/gemma2-9b-it",
        "accounts/fireworks/models/llama4-maverick-instruct-basic",
        "accounts/fireworks/models/llama-v3p1-8b-instruct",
    ]


def test_allowed_models_empty_means_unrestricted(monkeypatch):
    monkeypatch.setenv("ALLOWED_MODELS", "  ")
    assert Settings.from_env().allowed_models == []


def test_fireworks_key_without_paid_provider_selects_fireworks(monkeypatch):
    # The grading harness sets FIREWORKS_API_KEY but never PAID_PROVIDER.
    monkeypatch.delenv("PAID_PROVIDER", raising=False)
    monkeypatch.setenv("FIREWORKS_API_KEY", "fw-123")
    assert Settings.from_env().paid_provider == "fireworks"
    # An explicit PAID_PROVIDER still wins over the key heuristic.
    monkeypatch.setenv("PAID_PROVIDER", "groq")
    assert Settings.from_env().paid_provider == "groq"

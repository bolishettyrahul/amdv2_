"""Settings resolved from environment: provider selection, keys, thresholds."""

from router.config import Settings


def test_defaults(monkeypatch):
    for var in ("PAID_PROVIDER", "GROQ_API_KEY", "FIREWORKS_API_KEY",
                "OPENROUTER_API_KEY", "OLLAMA_HOST", "USE_CLOUD_FALLBACK"):
        monkeypatch.delenv(var, raising=False)
    s = Settings.from_env()
    assert s.paid_provider == "groq"  # preserve Fireworks credits during testing
    assert s.ollama_host == "http://localhost:11434"
    assert s.use_cloud_fallback is None  # None -> decide via health check
    assert 0.0 < s.sentiment_threshold < 1.0


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("PAID_PROVIDER", "fireworks")
    monkeypatch.setenv("FIREWORKS_API_KEY", "fw-123")
    monkeypatch.setenv("USE_CLOUD_FALLBACK", "1")
    monkeypatch.setenv("SENTIMENT_THRESHOLD", "0.7")
    s = Settings.from_env()
    assert s.paid_provider == "fireworks"
    assert s.fireworks_api_key == "fw-123"
    assert s.use_cloud_fallback is True
    assert s.sentiment_threshold == 0.7

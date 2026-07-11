"""build_pipeline: the frozen wiring contract every feature branch builds against."""

from router.config import Settings
from router.factory import build_pipeline
from router.pipeline import Pipeline
from router.stage3 import M8


def ok_transport(log):
    def transport(url, headers, payload):
        log.append(url)
        return (200, {
            "choices": [{"message": {"role": "assistant", "content": "pong"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1},
        })
    return transport


def settings(tmp_path, **kwargs) -> Settings:
    kwargs.setdefault("log_path", str(tmp_path / "tasks.jsonl"))
    return Settings(groq_api_key="gk", fireworks_api_key="fk", openrouter_api_key="ok",
                    **kwargs)


def test_builds_pipeline_without_network(tmp_path):
    calls = []
    p = build_pipeline(settings(tmp_path, use_cloud_fallback=False),
                       transport=ok_transport(calls))
    assert isinstance(p, Pipeline)
    assert calls == []  # construction alone must not hit any endpoint


def test_healthy_local_inference_keeps_stage2_free(tmp_path):
    calls = []
    p = build_pipeline(settings(tmp_path, use_cloud_fallback=None),
                       transport=ok_transport(calls))
    # health check ran against local ollama only
    assert all("localhost:11434" in url for url in calls) and calls
    assert p.stage2.paid is False
    assert p.stage2.provider.name == "ollama"


def test_unreachable_ollama_flips_to_cloud_fallback(tmp_path):
    def transport(url, headers, payload):
        if "localhost:11434" in url:
            return (500, {"error": "no gpu"})
        raise AssertionError(f"unexpected call to {url}")

    p = build_pipeline(settings(tmp_path, use_cloud_fallback=None), transport=transport)
    assert p.stage2.paid is True
    assert p.stage2.provider.name == "groq"  # default paid provider during testing
    assert set(p.stage2.models.values()) == {M8}  # ultra-cheap tier serves as Stage 2


def test_paid_provider_selection_and_stage3_retry_wiring(tmp_path):
    s = settings(tmp_path, paid_provider="fireworks", use_cloud_fallback=False,
                 code_retry_limit=1, sentiment_threshold=0.66)
    p = build_pipeline(s, transport=ok_transport([]))
    assert p.stage3.provider.name == "fireworks"
    assert p.stage3.provider.api_key == "fk"
    assert p.stage3.max_attempts == s.code_retry_limit + 1  # driven by Settings, not default
    assert p.stage1.sentiment_threshold == 0.66

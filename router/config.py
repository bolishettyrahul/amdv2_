"""Runtime settings, resolved from environment variables.

Thresholds default to the midpoints of the sweep ranges in
plan/evaluation-tuning-strategy.md; calibrate with scripts/evaluate_routing.py.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_bool(name: str) -> bool | None:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Settings:
    # Providers. Groq during testing (preserve Fireworks credits), Fireworks in production.
    paid_provider: str = "groq"
    groq_api_key: str = ""
    fireworks_api_key: str = ""
    openrouter_api_key: str = ""
    ollama_host: str = "http://localhost:11434"
    # None -> decide at startup via health check; True/False -> forced.
    use_cloud_fallback: bool | None = None

    # Verification-gate parameters (sweepable).
    sentiment_threshold: float = 0.5
    factual_k: int = 2
    logic_k: int = 3
    code_retry_limit: int = 2
    sandbox_timeout_s: float = 2.0
    local_max_latency_s: float = 3.0

    # Paths
    log_path: str = "logs/tasks.jsonl"

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            paid_provider=os.environ.get("PAID_PROVIDER", "groq").lower(),
            groq_api_key=os.environ.get("GROQ_API_KEY", ""),
            fireworks_api_key=os.environ.get("FIREWORKS_API_KEY", ""),
            openrouter_api_key=os.environ.get("OPENROUTER_API_KEY", ""),
            ollama_host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
            use_cloud_fallback=_env_bool("USE_CLOUD_FALLBACK"),
            sentiment_threshold=float(os.environ.get("SENTIMENT_THRESHOLD", "0.5")),
            factual_k=int(os.environ.get("FACTUAL_K", "2")),
            logic_k=int(os.environ.get("LOGIC_K", "3")),
            code_retry_limit=int(os.environ.get("CODE_RETRY_LIMIT", "2")),
            sandbox_timeout_s=float(os.environ.get("SANDBOX_TIMEOUT_S", "2.0")),
            local_max_latency_s=float(os.environ.get("LOCAL_MAX_LATENCY_S", "3.0")),
            log_path=os.environ.get("TASK_LOG_PATH", "logs/tasks.jsonl"),
        )

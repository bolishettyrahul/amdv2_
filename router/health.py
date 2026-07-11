"""Startup health check for local (ROCm/GPU) inference.

If Ollama is unreachable or a dummy generation exceeds the latency budget,
the orchestration layer flips to cloud-fallback mode (cheap Fireworks model
as Stage 2). Only Ollama ever touches the GPU.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from router.providers.base import LLMProvider

DEFAULT_MAX_LATENCY_S = 3.0


@dataclass(frozen=True)
class HealthStatus:
    ok: bool
    latency_s: float
    reason: str


def check_local_inference(
    provider: LLMProvider,
    model: str = "llama3.2:3b",
    max_latency_s: float = DEFAULT_MAX_LATENCY_S,
    timer: Callable[[], float] = time.perf_counter,
) -> HealthStatus:
    start = timer()
    try:
        provider.chat([{"role": "user", "content": "Reply with the single word: pong"}],
                      model=model, temperature=0.0, max_tokens=5)
    except Exception as exc:
        return HealthStatus(False, 0.0, f"local inference unavailable: {exc}")
    latency = timer() - start
    if latency > max_latency_s:
        return HealthStatus(False, latency,
                            f"local inference too slow: {latency:.1f}s > {max_latency_s:.1f}s")
    return HealthStatus(True, latency, "local inference healthy")

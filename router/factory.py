"""Assemble the full Phase 1 pipeline from Settings.

This is the single wiring point: provider selection (Groq for testing,
Fireworks for production), the startup GPU/Ollama health check that decides
standard vs cloud-fallback mode, and every threshold that
scripts/evaluate_routing.py sweeps. `transport` is injectable for tests.
"""

from __future__ import annotations

from router.config import Settings
from router.cost import CostTracker
from router.critique import Critic
from router.domain import DomainClassifier
from router.health import check_local_inference
from router.pipeline import Pipeline
from router.providers import FireworksProvider, GroqProvider, OllamaProvider
from router.providers.base import Transport
from router.providers.openrouter import OpenRouterProvider
from router.stage1 import Stage1Deterministic
from router.stage2 import Stage2Local
from router.stage3 import M8, Stage3Paid
from router.task_log import TaskLogger
from router.types import Domain

OPENROUTER_CRITIC_MODEL = "meta-llama/llama-3.3-70b-instruct:free"
LOCAL_CRITIC_MODEL = "mistral-nemo:12b-instruct"


def build_pipeline(settings: Settings, transport: Transport | None = None) -> Pipeline:
    ollama = OllamaProvider(host=settings.ollama_host, transport=transport)
    if settings.paid_provider == "fireworks":
        paid = FireworksProvider(api_key=settings.fireworks_api_key, transport=transport)
    else:
        paid = GroqProvider(api_key=settings.groq_api_key, transport=transport)

    use_cloud_fallback = settings.use_cloud_fallback
    if use_cloud_fallback is None:
        use_cloud_fallback = not check_local_inference(
            ollama, max_latency_s=settings.local_max_latency_s).ok

    if use_cloud_fallback:
        # Grading sandbox without a usable GPU: the ultra-cheap paid tier
        # stands in for local inference (see plan/standardized-env-strategy.md).
        stage2 = Stage2Local(
            paid, models={d: M8 for d in Domain}, paid=True,
            factual_k=settings.factual_k, logic_k=settings.logic_k,
            sandbox_timeout=settings.sandbox_timeout_s,
        )
    else:
        stage2 = Stage2Local(
            ollama, paid=False,
            factual_k=settings.factual_k, logic_k=settings.logic_k,
            sandbox_timeout=settings.sandbox_timeout_s,
        )

    return Pipeline(
        classifier=DomainClassifier(),
        stage1=Stage1Deterministic(sentiment_threshold=settings.sentiment_threshold,
                                   sandbox_timeout=settings.sandbox_timeout_s),
        stage2=stage2,
        stage3=Stage3Paid(paid, cost_tracker=CostTracker(),
                          max_attempts=settings.code_retry_limit + 1,
                          sandbox_timeout=settings.sandbox_timeout_s),
        critic=Critic(
            primary_provider=OpenRouterProvider(api_key=settings.openrouter_api_key,
                                                transport=transport) if settings.openrouter_api_key else ollama,
            primary_model=OPENROUTER_CRITIC_MODEL if settings.openrouter_api_key else LOCAL_CRITIC_MODEL,
            fallback_provider=ollama,
            fallback_model=LOCAL_CRITIC_MODEL,
        ),
        logger=TaskLogger(settings.log_path),
    )

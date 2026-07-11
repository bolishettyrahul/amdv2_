#!/usr/bin/env python3
"""CLI wrapper around router.health.check_local_inference()."""

import sys

from router.config import Settings
from router.health import check_local_inference
from router.providers.ollama import OllamaProvider


def main() -> int:
    settings = Settings.from_env()
    provider = OllamaProvider(host=settings.ollama_host)
    status = check_local_inference(provider, max_latency_s=settings.local_max_latency_s)

    print(f"Health check: {status.reason} (latency: {status.latency_s:.2f}s)")
    return 0 if status.ok else 1


if __name__ == "__main__":
    sys.exit(main())

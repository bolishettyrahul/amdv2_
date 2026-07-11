# Standardized Environment & Sandbox Strategy

This document outlines the strategy for handling constraints, resource limits, and environment parity in the standardized grading sandbox environment.

## 1. Environment Constraints & Resource Limits

Because the agent is executed in a standardized, automated grading environment, we must assume the following potential limitations:
1.  **CPU-Only/Low-RAM Execution:** The grading runner might not have GPU resources attached, or GPU passthrough might fail. Running local 7B models on CPU will cause massive latency and potentially timeout.
2.  **Strict Latency / Timeout Limits:** Batch jobs often have a timeout limit per sample (e.g., 30–60 seconds) or for the entire run. If our local pipeline takes too long (e.g., executing multiple local Ollama steps or self-consistency loops), the script may be forcefully terminated, leading to a score of `0`.
3.  **Network Constraints:** The environment might block outgoing internet traffic except to specific allowed API endpoints (like Fireworks AI). If we try to fetch external resources or use APIs like OpenRouter for critiques, they will fail.

---

## 2. Parity & Failover Architecture

To protect against these environmental risks, the agent must be built with a **resilient fall-back configuration system**:

```
                  ┌───────────────────────────────┐
                  │      Environment Check        │
                  └───────────────┬───────────────┘
                                  │
                 Is GPU/Ollama active and fast?
                   /                             \
                 YES                              NO
                 /                                 \
  ┌─────────────────────────────┐    ┌─────────────────────────────┐
  │     Standard Mode           │    │    Cloud-Fallback Mode      │
  │ - Stage 2: Local Ollama     │    │ - Stage 2: Route to cheap   │
  │ - CPU-only tools            │    │   Fireworks model           │
  │                             │    │   (e.g., Llama 3.1 8B)      │
  └─────────────────────────────┘    └─────────────────────────────┘
```

### Action Plan
*   **Startup Verification check:** During agent initialization, run a quick health-check script (`scripts/health_check.py`) to verify Ollama status and GPU acceleration.
*   **Dynamic Fallback Switch:** If the health-check fails or local latency on a dummy task exceeds 3 seconds, automatically flip the configuration flag `USE_CLOUD_FALLBACK = True`.
*   **Cloud-Fallback behavior:** Under this mode, Stage 2 (Ollama local model) is bypassed. Instead, the task is sent to an ultra-cheap Fireworks model (e.g., `llama-v3p1-8b-instruct`), which serves as the "free/cheap" tier, before escalating to the larger models in Stage 3 if verification fails.

---

## 3. Execution Sandboxing for Code Domains

For the `code_debugging` and `code_generation` domains, the agent executes generated code and test suites. Running arbitrary code is dangerous and must be sandboxed:
*   **Subprocess Constraints:** Execute test runs via Python's `subprocess` module with:
    *   `timeout` limit (e.g., max 2 seconds per test execution).
    *   System resource constraints (limiting memory allocation).
    *   Restricted permissions (avoiding execution as root/admin).
*   **No State Leakage:** Ensure test executions do not write persistent changes to the disk that could corrupt subsequent task executions in the batch run.

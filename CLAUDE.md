# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

This repository contains the fully implemented Phase 1 cascade routing pipeline and testing harness. It is no longer pre-implementation. There is a complete test suite of 120 tests and command-line execution entrypoints.

## Build, Test & Run Commands

* **Activate Virtual Environment**:
  * Windows PowerShell: `.venv\Scripts\Activate.ps1`
  * Command Prompt: `.venv\Scripts\activate.bat`
* **Install Dependencies**: `pip install -r requirements.txt` (and optionally `-r requirements-optional.txt`)
* **Run Unit Tests**: `python -m pytest`
* **Run Batch Routing CLI**: `python -m router.main --input <path_to_input> --output <path_to_output>`
* **Run Offline Routing Evaluator**: `python scripts/evaluate_routing.py`
* **Run GPU/Ollama Health Check**: `python scripts/health_check.py`

## What this project is

A **hybrid token-efficient LLM routing agent** for a competition on AMD Developer Cloud. It's a **batch
job** (reads a JSONL/CSV dataset, writes results — not an HTTP service) scored on:

```
Score = Accuracy / Total Paid Token Cost
```

across 8 domains (factual knowledge, math reasoning, sentiment classification, summarization, NER, code
debugging, logical reasoning, code generation). The domain is not labeled in the input and must be
inferred per-task.

Full details live in `plan/`:
- `plan/master-plan.md` — overall architecture and build status (read this first)
- `plan/fireworks-model-catalog.md` — Stage 3 paid model catalog, pricing, and routing matrix
- `plan/evaluation-tuning-strategy.md` — threshold/parameter calibration methodology (Pareto frontier
  on accuracy vs. cost)
- `plan/standardized-env-strategy.md` — grading-sandbox constraints, GPU fallback, code-execution
  sandboxing

## Core architecture — Phase 1 cascade

Every task flows through an ordered pipeline. Each stage has a free verification gate; passing returns
the answer immediately at $0, failing escalates to the next (more expensive) stage. Paid tokens are only
ever spent at the final stage.

```
task ──► Stage 0: domain classify (free, local embedding: sentence-transformers all-MiniLM-L6-v2)
              │
              ▼
      Stage 1: deterministic tool (free, no LLM — sympy, VADER, spaCy, z3, sandboxed test execution)
              │ verifier pass? ──yes──► return ($0)
              ▼
      Stage 2: local Ollama model (free, GPU compute only — qwen2.5:7b, phi3.5:3.8b, llama3.2:3b)
              │ verifier pass? ──yes──► return ($0)
              ▼
      Stage 3: paid model via Dynamic Fireworks Routing (only stage that costs tokens)
              │ verification fails? ──► agentic self-correction retry with failure context
              ▼
         return
```

Key architectural points (see `plan/master-plan.md` for the full per-domain tool/verifier table):

- **5 of 8 domains** (math, sentiment, NER, code debugging, code generation) have ground-truth or
  near-ground-truth deterministic verifiers and should resolve at or near $0. The other 3 (factual
  knowledge, summarization, logical reasoning) have only weak confidence proxies (self-consistency,
  heuristics) and carry most of the expected token cost.
- **Stage 3 routing is dynamic, not a single static model** — it picks the cheapest Fireworks model
  capable of the task based on domain + complexity (see the routing matrix in
  `fireworks-model-catalog.md`). Escalating a simple task straight to the largest/most expensive model
  is treated as a scoring bug, not just inefficiency.
- **Provider abstraction**: a single `LLMProvider` ABC with `GroqProvider`, `FireworksProvider`,
  `OllamaProvider` implementations, all OpenAI-chat-compatible. Groq is used during testing to preserve
  Fireworks credits; Fireworks is used in production.
- **Every task appends one structured JSONL log record** (input, inferred domain, which stage answered,
  the answer, verifier/critique verdict + reasoning, tokens/cost). This log is Phase 1's deliverable and
  the entire training set Phase 2 depends on.
- **LLM critique is for dataset labeling only, never live routing.** It's skipped for the 5
  deterministically-verified domains, and required for the 3 weak-verifier domains, using an OpenRouter
  free-tier model as primary critic and local `mistral-nemo:12b-instruct` as fallback. Hard rule: the
  critic must never be the same model that produced the answer being critiqued.
- **Environment resiliency**: only Ollama touches ROCm/GPU. A startup health check determines whether
  local GPU inference is available and fast enough; if not, the orchestration layer falls back to a
  cheap Fireworks model (Llama 3.1 8B) for what would otherwise be Stage 2. The grading sandbox may be
  CPU-only, have a per-sample timeout, and may block network access to anything but approved API
  endpoints — design for that, not for a guaranteed local GPU.
- **Code domains execute untrusted generated code.** Use subprocess-based sandboxing with a timeout
  (target ~2s per test execution), memory limits, restricted permissions, and no persistent state
  leakage between tasks in the same batch run.
- **Docker/ROCm**: AMD GPU devices are passed at `docker run` time
  (`--device=/dev/kfd --device=/dev/dri --group-add video --group-add render`), never baked into the
  image. Do not use `--gpus all` (that's NVIDIA-only).

## Phase 2 (conditional, not the starting point)

A LightGBM classifier trained on Phase 1's logged dataset (features: domain, prompt embedding,
task-shape signals; label: cheapest tier verified/critiqued as correct), deployed in hybrid mode — low
confidence predictions still fall back through the full Phase 1 cascade. This has a hard dependency on
Phase 1 producing real logs first; it is scaffolded under `router/phase2/` but not implemented until
Phase 1 is running, and only pursued if time/credits remain.

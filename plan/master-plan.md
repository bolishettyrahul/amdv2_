# Master Plan — Hybrid Token-Efficient LLM Routing Agent

## Objective

Competition hosted on AMD Developer Cloud. Scored on **accuracy ÷ external paid API token cost** across 8 domains: factual knowledge, mathematical reasoning, sentiment classification, text summarization, named entity recognition, code debugging, logical reasoning, code generation. Domain is not labeled in the input — must be inferred. Harness runs the agent as a **batch job** over a dataset file (JSONL/CSV in, results out), not an HTTP service.

Local Ollama inference runs on judging-attached AMD Instinct GPUs (MI300X / Radeon AI PRO, ROCm/Vulkan passthrough) and is effectively free. Paid inference goes through Groq during testing (to preserve Fireworks credits) and Fireworks in production, serving a multi-tier hierarchy of models behind one OpenAI-compatible provider interface. **The core strategy: resolve as many tasks as possible with deterministic tools or local models, verified for free, and only pay for a model call when free verification actually fails—routing to the cheapest sufficient model on Fireworks AI when paid escalation is required.**

## Phase 1 — Cascade with verification gates (build now)

Every task flows through an ordered pipeline; each stage has a free verification gate. Passing the gate returns the answer immediately at $0. Failing escalates to the next stage. Paid tokens are spent only at the final stage, only for tasks where free verification failed everywhere else.

```
task ──► Stage 0: domain classify (free, local embedding)
              │
              ▼
      Stage 1: deterministic tool (free, no LLM)
              │ verifier pass? ──yes──► return  ($0)
              │ no
              ▼
      Stage 2: local Ollama model (free, GPU compute only)
              │ verifier pass? ──yes──► return  ($0)
              │ no
              ▼
      Stage 3: paid model (Groq/Fireworks, Gemma)
              │
              ▼
         return  (only stage that costs tokens)
```

### Stage 0 — Domain classification

Local sentence-embedding similarity (`sentence-transformers`, `all-MiniLM-L6-v2`) against a small set of per-domain exemplar prompts. Sub-millisecond, no GPU required. Ambiguous top-2 scores fall back to a single local Ollama classification call — still $0.

### Domain → tool / verifier / local model table

| Domain | Stage 1 tool (free) | Stage 1 verifier | Stage 2 local model | Stage 2 verifier |
|---|---|---|---|---|
| Factual knowledge | — | — | qwen2.5:7b | self-consistency: 2 samples agree |
| Mathematical reasoning | sympy / guarded expression eval | re-execution matches extracted answer | qwen2.5:7b (generates expression, tool computes) | tool re-verifies computed result |
| Sentiment classification | VADER lexicon | confidence ≥ threshold | phi3.5:3.8b (only if tool confidence low) | label schema check |
| Text summarization | — | — | llama3.2:3b or qwen2.5:7b | heuristic: length bounds + key-entity coverage |
| Named entity recognition | spaCy NER | span/label schema validation | qwen2.5:7b (only if spaCy confidence low) | schema validation |
| Code debugging | sandboxed test execution | tests pass/fail (ground truth) | qwen2.5:7b generates fix, tool re-runs tests | test execution (same verifier) |
| Logical reasoning | z3 solver (solver-shaped problems only) | SAT/proof returned | qwen2.5:7b | self-consistency across 2–3 samples |
| Code generation | sandboxed execution against test cases | tests pass/fail (ground truth) | qwen2.5:7b or code-tuned local model | test execution |

5 of 8 domains (math, sentiment, NER, code debugging, code generation) have ground-truth or near-ground-truth free verification and should resolve at or near $0 in aggregate. The other 3 (factual knowledge, summarization, logical reasoning) have no deterministic check and rely on weaker confidence proxies — these carry most of the expected token cost. The verification gates, confidence thresholds, and self-consistency parameters across all domains must be calibrated systematically to maximize our accuracy-to-cost ratio, as detailed in [evaluation-tuning-strategy.md](file:///c:/Users/Rama%20Bolishetty/OneDrive/Desktop/amd_v2/plan/evaluation-tuning-strategy.md).

### Stage 3 — Paid escalation

Instead of escalating to a single static paid model, the agent employs **Dynamic Fireworks Routing** to select the cheapest model capable of solving the task based on domain and complexity. If the escalated model outputs an answer that fails verification (e.g., test cases fail), the agent initiates an **agentic self-correction loop** to retry with context on the failure. The model tiers, pricing, and routing policy are detailed in [fireworks-model-catalog.md](file:///c:/Users/Rama%20Bolishetty/OneDrive/Desktop/amd_v2/plan/fireworks-model-catalog.md).

### Token cost tracking

Every paid-provider call is wrapped by a cost tracker that records input/output tokens and computed $ cost against a price table, attached to that task's log entry. This is what makes "accuracy ÷ cost" measurable and tunable during development, not just something we hope holds at judging time.

### Dataset logging — the deliverable of Phase 1

Every task appends one structured record to a JSONL log: input, inferred domain, which stage produced the final answer, the answer, verifier and/or critique verdict + reasoning, tokens/cost. This log is the entire training dataset Phase 2 depends on.

### LLM critique — for dataset labeling only, not live routing

The 3 weak-verifier domains (factual knowledge, summarization, logical reasoning) have no ground truth to label outcomes against. A post-hoc critique step supplies one:

- **Skipped entirely** for the 5 domains with deterministic tool verifiers — the tool result already is ground truth, an LLM judge adds no label quality.
- **Required** for factual knowledge, summarization, logical reasoning: primary critic = an OpenRouter free-tier model; fallback critic (when OpenRouter is rate-limited/unavailable) = local `mistral-nemo:12b-instruct`, run via Ollama.
- **Hard rule, enforced in code**: the critic must never be the same model that produced the answer. Mistral-Nemo is a deliberately distinct lineage from every actor model (Qwen, Llama, Phi), so this holds automatically for the local fallback path.
- Critique output = verdict (correct/incorrect or quality score) + reasoning text, both logged. It enriches the dataset record; it never alters or blocks the answer already returned to the harness.

### Provider abstraction & Docker/ROCm

- `LLMProvider` ABC with `GroqProvider`, `FireworksProvider`, `OllamaProvider` implementations, all OpenAI-chat-compatible — one interface for every "generate text" need, local or paid.
- ROCm-enabled Docker base image. **No `--gpus all`** (NVIDIA-only). AMD devices exposed via `--device=/dev/kfd --device=/dev/dri --group-add video --group-add render`, passed at `docker run` time, never baked into the image.
- **Environment Resiliency**: Only Ollama touches ROCm. The system dynamically executes a startup health check to determine if the local GPU environment is active. If local Ollama is unavailable or too slow, the orchestration layer falls back to a cheap-tier cloud model (Llama 3.1 8B on Fireworks) for Stage 2 processing. Safe test-execution sandboxing is used for code domains. See [standardized-env-strategy.md](file:///c:/Users/Rama%20Bolishetty/OneDrive/Desktop/amd_v2/plan/standardized-env-strategy.md) for full environmental and sandbox integration details.

## Phase 2 — Learned hybrid router (conditional: only if time/credits remain)

Train a lightweight LightGBM classifier on Phase 1's logged dataset: features = domain, prompt embedding, task-shape signals (length, code-fence presence, math-token presence); label = the cheapest tier whose outcome was verified/critiqued as correct. Deploy in **hybrid mode**: the router predicts a tier directly, but low-confidence predictions still fall back through the full Phase 1 cascade rather than committing blind — this bounds the misprediction risk that a pure learned router would otherwise carry.

Phase 2 cannot be the starting architecture — it has a hard dependency on Phase 1's logged data. It is scaffolded (`router/phase2/`) but not implemented until Phase 1 is running and producing real logs, and only pursued if time/credits allow.

## Status

Phase 1 is in active implementation. See the engineering implementation plan (module layout, library choices, build order, verification plan) tracked separately for build execution detail.

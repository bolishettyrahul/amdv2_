# Approach C — Learned Predictive Router

## Core idea

Instead of always attempting Stage 1 → Stage 2 → Stage 3 in order and
letting free verifiers decide escalation, train
likely to succeed for a given task, then invoke only that tier directly.

```
task ──► [Stage 0: domain classify]  (free, local — same as Approach B)
              │
              ▼
      [feature extraction]  (domain, prompt embedding, length,
                              code-block present?, math-expr present?,
                              historical difficulty signals...)
              │
              ▼
      [learned router model]  (local, e.g. gradient-boosted tree
                                or logistic regression on embeddings)
              │
              ▼
      predicted tier ∈ {tool, local, paid}
              │
              ▼
      invoke ONLY the predicted tier ──► return answer
```

The router model itself must be cheap to run (CPU inference, milliseconds)
— it is not an LLM call, it's a lightweight classifier trained offline.

## Domain → tool / verifier / local model table

Identical tool/verifier/model bank to Approach B — the execution primitives
don't change, only how the decision to invoke them is made:

| Domain | Tool (free) | Local model | Paid model |
|---|---|---|---|
| Factual knowledge | — | qwen2.5:7b | Gemma (Groq/Fireworks) |
| Mathematical reasoning | sympy / python eval | qwen2.5:7b + tool | Gemma |
| Sentiment classification | VADER / lexicon | phi3.5:3.8b | Gemma |
| Text summarization | — | llama3.2:3b / qwen2.5:7b | Gemma |
| Named entity recognition | spaCy NER | qwen2.5:7b | Gemma |
| Code debugging | sandboxed test execution | qwen2.5:7b + tool | Gemma |
| Logical reasoning | z3 solver (where applicable) | qwen2.5:7b | Gemma |
| Code generation | sandboxed test execution | qwen2.5:7b | Gemma |

## The cold-start problem

A learned router needs labeled training data: for each historical task,
which tier was the *cheapest tier that would have succeeded*. This data
does not exist before the system has run. Two ways to get it:

1. **Bootstrap via Approach B**: run the verification-gated cascade for a
   period, log (features, stage-that-resolved-it, verifier-outcomes) for
   every task, then train the router on those logs. This is the natural
   path — Approach C becomes a later optimization layered on top of B,
   not a day-one alternative.
2. **Deliberate probing**: run all three tiers on a sample of tasks up
   front purely to collect labels. This spends extra tokens (defeats the
   cost objective) just to bootstrap, so it's worse than (1) unless you
   have a labeled dataset already.

Either way, **Approach C cannot be the starting architecture** — it
requires Approach B (or an equivalent logging harness) to run first.

## Router model

- Features: domain (one-hot), sentence-embedding of the prompt (local,
  free), prompt length, presence of code fences, presence of numeric/math
  tokens, and (once available) historical per-domain difficulty stats.
- Model: gradient-boosted trees (e.g. LightGBM) or logistic regression —
  small, fast, interpretable, retrainable in seconds on CPU.
- Output: predicted cheapest-sufficient tier, optionally with a confidence
  score used to fall back to Approach B's cascade when confidence is low
  (a hybrid safety net — see Risks below).

## Provider abstraction & Docker/ROCm

Identical to Approach B: same `LLMProvider` interface (Groq now, Fireworks
later, Gemma model), same ROCm-enabled Docker setup (`/dev/kfd`,
`/dev/dri`, `video`/`render` groups, no `--gpus all`). This layer is
routing-strategy-agnostic.

## Example end-to-end flow (code debugging)

1. Task arrives: buggy Python function + failing test.
2. Stage 0 classifies domain = `code_debugging`.
3. Feature extraction: embedding, code length, cyclomatic-complexity proxy,
   presence of a stack trace in the prompt.
4. Router predicts tier = `local` with 0.82 confidence (based on training
   data showing similar-shaped bugs were fixed locally 80%+ of the time).
5. Invoke qwen2.5:7b directly, apply the fix, run tests. **No Stage-1-tool
   sanity pass, no fallback check unless the hybrid safety net is enabled.**
6. If the router had instead predicted `paid` (e.g., multi-file bug, long
   stack trace, features resembling historically-hard cases), it would go
   straight to Gemma — skipping the local attempt entirely.

## Where this actually saves vs. Approach B

This is the key trade-off to be explicit about, given the actual scoring
function is **accuracy ÷ token cost**, not accuracy ÷ latency:

- Local model calls are **already free** under this project's constraints
  (GPU compute on the AMD Instinct hardware, not API tokens). Approach B
  already escalates to paid *only* when free verification fails — so it
  already achieves the token-optimal outcome whenever a free verifier
  exists.
- Approach C's main structural advantage over B is **skipping the wasted
  local attempt for tasks predicted to fail locally anyway** — i.e.,
  going straight to paid instead of local-then-paid. That is a
  **latency/throughput** win, not a token-cost win, since Stage 2 in B
  never spends tokens either way.
- The one place C can save tokens over B: for domains **without** a
  reliable free verifier (factual knowledge, summarization, open logical
  reasoning), B has no gate to trust and may escalate more conservatively
  than necessary. A well-trained router could, in principle, learn "this
  local answer is probably fine" for a case where B's self-consistency
  gate would have escalated. This is a real but narrow advantage, offset
  by the misprediction risk below.

## Risks

- **Misprediction cost is asymmetric and directly hurts the score**:
  predicting `local` for a task that actually needed `paid` returns a
  wrong answer with no safety net (no verifier ran) — an accuracy loss.
  Predicting `paid` for an easy task wastes tokens directly — a cost loss.
  Both failure modes are the exact thing the scoring function penalizes.
- Requires an ongoing retraining/versioning pipeline as task distribution
  shifts — infra complexity with no counterpart in Approach B.
- The safety-net hybrid (fall back to B's verifiers when router confidence
  is low) mitigates the accuracy risk but reintroduces most of B's
  complexity, eroding C's simplicity advantage.

## Pros

- Can skip a doomed local attempt and go straight to paid for
  historically-hard task shapes — reduces latency/throughput cost.
- Narrow token-saving potential on weak-verifier domains, if trained well.
- Natural evolution once Approach B has been running and logging.

## Cons

- Cannot be the starting architecture — has a hard dependency on
  historical labeled data from Approach B (or an equivalent logging setup).
- Misprediction risk directly hurts the accuracy ÷ cost score in both
  directions (wrong answers or wasted tokens), unlike B where escalation
  is never a guess.
- Added ML infra (training, feature pipeline, retraining cadence,
  versioning) for a benefit that is mostly latency, not token cost, given
  local compute is already free in this environment.

## Recommendation if pursued

Treat this as a **Phase 2** layered on top of Approach B, not a
replacement: run B first, log everything, train the router on B's logs,
then deploy C in hybrid mode (router prediction + fallback to B's
verifier gate on low confidence) rather than as a full replacement of the
cascade.

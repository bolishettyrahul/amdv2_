# Validation & Threshold Tuning Strategy

To optimize the scoring metric of `accuracy ÷ token cost`, we need a systematic way to calibrate the verification gate thresholds, sample counts, and routing decision parameters.

## The Objective Function
The agent is evaluated on a test dataset using the following metric:

$$\text{Score} = \frac{\text{Accuracy}}{\text{Total Paid Token Cost}}$$

Where:
*   $\text{Accuracy}$ is the percentage of tasks resolved correctly.
*   $\text{Total Paid Token Cost}$ is the sum of costs for all requests routed through Fireworks AI.

## The Calibration Problem
*   If verification gates are **too strict** (e.g., high confidence thresholds, high agreement requirements in self-consistency), the agent will reject correct local/free answers and escalate to expensive Fireworks models unnecessarily, blowing up the cost denominator.
*   If verification gates are **too loose** (e.g., low confidence thresholds, single-sample checks), the agent will accept incorrect local/free answers, tanking the accuracy numerator.

## Tuning Strategy & Workflow

```
[Labeled Validation Dataset]
            │
            ▼
┌───────────────────────────────┐
│     Sweep Parameters          │◄── (Grid Search / Bayesian Opt)
│ - NER / Sentiment threshold   │
│ - Math parser strictness      │
│ - Self-consistency count (k)  │
│ - Model routing thresholds    │
└───────────┬───────────────────┘
            │
            ▼
┌───────────────────────────────┐
│    Evaluate Routing Agent     │
│ - Calculate accuracy          │
│ - Calculate total token cost  │
└───────────┬───────────────────┘
            │
            ▼
┌───────────────────────────────┐
│     Map Pareto Frontier       │
│ - Select parameter set        │
│   maximizing: Accuracy / Cost │
└───────────────────────────────┘
```

### 1. Build a Local Validation Set
We must extract/assemble a balanced validation dataset containing:
*   At least 50–100 labeled samples per domain (total 400–800 tasks).
*   Varying difficulty levels (simple factual queries, medium logic, complex coding).
*   Ground-truth outputs or test cases.

### 2. Grid-Search Parameters
We will run batch offline evaluations sweeping the following parameters:

| Parameter | Type | Sweep Range | Affected Domain(s) |
|---|---|---|---|
| `sentiment_confidence_threshold` | Float | `0.4` to `0.9` | Sentiment (VADER) |
| `ner_confidence_threshold` | Float | `0.5` to `0.95` | Named Entity Recognition |
| `factual_self_consistency_k` | Integer | `1` to `5` | Factual Knowledge (Local Qwen) |
| `logical_self_consistency_k` | Integer | `1` to `5` | Logical Reasoning (Local Qwen) |
| `code_retry_limit` | Integer | `1` to `3` | Code Debugging / Generation |
| `router_confidence_cutoff` | Float | `0.5` to `0.95` | Phase 2 Learned Router |

### 3. Pareto Frontier Analysis
We will plot `Accuracy` on the y-axis and `Token Cost` on the x-axis for each parameter combination. The optimal configuration is the point along the Pareto frontier that maximizes the ratio $\frac{\text{Accuracy}}{\text{Token Cost}}$.

> [!IMPORTANT]
> If the competition scoring has a minimum accuracy constraint (e.g. "without falling below the accuracy threshold"), this acts as a hard constraint. The optimization goal becomes:
> $$\text{Maximize: } \frac{\text{Accuracy}}{\text{Token Cost}} \quad \text{subject to: } \text{Accuracy} \ge \text{Threshold}$$

## Verification Pipeline Execution
*   Develop a CLI script `scripts/evaluate_routing.py` that takes a dataset file, runs the routing logic, simulates API costs, and outputs the detailed scoring metric.
*   Log every run's execution path (e.g., `Stage 0 -> Stage 2 (Verified) -> Return`) to identify which domains are the "cost sinks" and where verification is leaking incorrect answers.

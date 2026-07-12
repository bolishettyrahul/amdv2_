# AMD Developer Cloud — LLM Routing Agent

A hybrid, token-efficient LLM routing agent built for the AMD Developer Cloud competition. The objective of this agent is to maximize classification and generation accuracy while minimizing paid token usage.

## 🎯 Scoring Metric
The agent is optimized against the following performance metric:

$$\text{Score} = \frac{\text{Accuracy}}{\text{Total Paid Token Cost}}$$

To maximize this score, the agent uses a **4-stage cascading architecture** where simple tasks are solved using zero-cost methods (rules, libraries, or free local models) and only high-complexity or failed tasks escalate to paid LLMs.

---

## 🏗️ Cascading Architecture (Phase 1)

Every incoming task is processed sequentially through the following pipeline stages:

```
Task ──► Stage 0: Domain Classification (Free local sentence-transformers / keyword fallback)
              │
              ▼
         Stage 1: Deterministic Tools & Sandbox (Free local computation — SymPy, VADER, spaCy, Z3)
              │ ──► [Verifier Pass?] ──yes──► Return ($0)
              ▼
         Stage 2: Local GPU Models via Ollama (Free local ROCm GPU — Qwen 2.5, Phi 3.5, Llama 3.2)
              │ ──► [Verifier Pass?] ──yes──► Return ($0)
              ▼
         Stage 3: Paid Dynamic Routing (Fireworks API / Groq API testing fallback)
              │ ──► [Dynamic model selection based on inferred domain + difficulty]
              ▼
         Return & Log to logs/tasks.jsonl (For telemetry & Phase 2 routing training)
```

---

## 🛠️ What has been Implemented

The initial implementation of **Phase 1** is complete and fully covered by testing:

### 1. Stage 0: Domain Classifier (`router/domain.py`)
* Maps prompts into 8 distinct domains (Factual Knowledge, Math Reasoning, Sentiment Classification, Summarization, NER, Code Debugging, Logical Reasoning, Code Generation).
* Uses `sentence-transformers` (`all-MiniLM-L6-v2`) locally to compute embeddings.
* Falls back gracefully to a regex-based keyword matching classifier if GPU/optional packages are missing.

### 2. Stage 1: Deterministic Solvers (`router/tools/`, `router/stage1.py`)
* **Math Solver (`router/tools/math_tool.py`)**: Uses `SymPy` for expression parsing, evaluation, equation solving, and verification.
* **Sentiment Analyzer (`router/tools/sentiment_tool.py`)**: Employs `vaderSentiment` for rule-based analysis.
* **Named Entity Recognition (`router/tools/ner_tool.py`)**: Integrates `spaCy` (`en_core_web_sm`) for entity extraction.
* **Logic Solver (`router/tools/logic_tool.py`)**: Uses `z3-solver` to check logical satisfiability and constraints.
* **Code Sandbox (`router/tools/sandbox.py`)**: A secure, subprocess-based environment executing Python code snippets under custom memory limits and timeouts.

### 3. Stage 2: Local GPU Inference via Ollama (`router/stage2.py`, `router/health.py`)
* Connects to a local Ollama instance running on ROCm.
* **Health Check & Latency Fallback**: Checks startup status. If the local GPU is unavailable or does not respond within latency limits (`local_max_latency_s`), the orchestrator falls back to running a lightweight model (e.g., Llama 3.1 8B) on Fireworks to maintain SLA.

### 4. Stage 3: Paid Dynamic Routing (`router/stage3.py`, `router/providers/`)
* Dynamically maps tasks to the most cost-effective paid LLM (Groq or Fireworks) based on the inferred domain and complexity.
* Avoids "overpaying" by preventing simple tasks from routing to premium, high-cost models (e.g., Llama 3.1 405B) unless smaller tiers fail validation.

### 5. Verification & Critique Framework (`router/verifiers.py`, `router/critique.py`)
* Domain-specific verifiers validate outputs from Stages 1 & 2 before deciding whether to exit early or escalate.
* For domains without a deterministic parser/verifier (Factual Knowledge, Summarization, Logical Reasoning), an LLM critic evaluates the answer (using OpenRouter/local fallback) for training data labeling.

### 6. Pipeline Orchestrator & Telemetry (`router/pipeline.py`, `router/task_log.py`)
* Sequentially coordinates execution across stages.
* Automatically records detailed metadata (inferred domain, executing stage, latency, output, paid token count, and calculated cost) into `logs/tasks.jsonl` for offline analysis and Phase 2 training.

---

## 📂 Codebase Layout

```
.
├── .claude/                # Claude Code local config and permissions
├── plan/                   # Competition planning and methodology documents
├── router/                 # Pipeline core logic
│   ├── providers/          # LLM API providers (Ollama, Groq, Fireworks)
│   ├── tools/              # Stage 1 deterministic tool wrappers and sandboxing
│   ├── batch.py            # Dataset batch-runner
│   ├── config.py           # Configuration loading & settings
│   ├── pipeline.py         # Main execution pipeline
│   ├── verifiers.py        # Stage exit validation verifiers
│   └── types.py            # Enums, Dataclasses, Type aliases
├── tests/                  # Unit and integration tests for every module
├── pyproject.toml          # Project package definitions
├── requirements.txt        # Hard requirements (CPU-safe libraries)
└── requirements-optional.txt # Soft dependencies (spacy, sentence-transformers, lightgbm)
```

---

## 🚀 Getting Started

### Installation

1. Create and activate a python virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate    # Windows
   source .venv/bin/activate  # macOS/Linux
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. (Optional) Install NLP and embeddings libraries:
   ```bash
   pip install -r requirements-optional.txt
   python -m spacy download en_core_web_sm
   ```

### Configuration
Configure settings via environment variables (or let them fallback to defaults in `router/config.py`):
```bash
# Provider Configuration
export PAID_PROVIDER="fireworks" # or "groq"
export FIREWORKS_API_KEY="your-api-key"
export GROQ_API_KEY="your-api-key"
export OLLAMA_HOST="http://localhost:11434"

# Settings & Thresholds
export LOCAL_MAX_LATENCY_S="3.0"
export TASK_LOG_PATH="logs/tasks.jsonl"
```

### Running Tests
To run the full validation test suite (120 tests):
```bash
python -m pytest
```

### Running the CLI Agent (Bare-Metal)
You can run the routing agent directly over any dataset in JSONL or CSV format:
```bash
python -m router.main --input <path_to_dataset> --output <path_to_results.jsonl>
```
* **Example**:
  ```bash
  python -m router.main --input fixtures/sample_tasks.jsonl --output logs/results.jsonl
  ```

### Running the Offline Routing Evaluator
To run a parameter sweep across all routing thresholds and logic gates to determine the Pareto frontier (Accuracy vs. Cost):
```bash
python scripts/evaluate_routing.py
```

### Running with Docker
For instructions on building the container image and running with GPU/ROCm passthrough, see the [RUNNING.md](file:///c:/Users/Rama%20Bolishetty/OneDrive/Desktop/amd_v2/RUNNING.md) guide.

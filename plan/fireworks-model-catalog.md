# Fireworks AI Model Catalog & Routing Profiles

This catalog documents the candidate models available on Fireworks AI, their token pricing, capabilities, and target domains to guide the routing agent's decision-making.

## Model Catalog & Pricing Structure

| Model ID | Short Name | Parameter Count | Cost per 1M Input Tokens | Cost per 1M Output Tokens | Pricing Tier | Typical Strengths |
|---|---|---|---|---|---|---|
| `accounts/fireworks/models/llama-v3p1-8b-instruct` | Llama 3.1 8B | 8B | ~$0.20 | ~$0.20 | Ultra-Low | NER, Sentiment, basic math verification, short summaries |
| `accounts/fireworks/models/gemma2-9b-it` | Gemma 2 9B | 9B | ~$0.20 | ~$0.20 | Ultra-Low | Logic, general reasoning, factual questions |
| `accounts/fireworks/models/gemma2-27b-it` | Gemma 2 27B | 27B | ~$0.80 | ~$0.80 | Low-Medium | Complex reasoning, math word problems, entity extraction |
| `accounts/fireworks/models/llama-v3p1-70b-instruct` | Llama 3.1 70B | 70B | ~$0.90 | ~$0.90 | Medium | Code generation, complex debugging, agentic planning |
| `accounts/fireworks/models/qwen2p5-72b-instruct` | Qwen 2.5 72B | 72B | ~$0.90 | ~$0.90 | Medium | Advanced coding, mathematical proofs, structured JSON generation |
| `accounts/fireworks/models/llama-v3p1-405b-instruct` | Llama 3.1 405B | 405B | ~$4.00 | ~$4.00 | High | Highly complex logical tasks, fallback verification |

## Routing Matrix (Dynamic Stage 3 Escalation)

When a task fails free local verification (Stage 2) or has low confidence, the routing agent will escalate to Fireworks AI. Instead of using a single "paid model", it chooses based on the inferred domain and estimated task complexity:

### 1. Factual Knowledge
*   **Simple/Direct Questions:** `gemma2-9b-it` (Fast, cheap, good trivia coverage).
*   **Multi-hop / Comparative Questions:** `llama-v3p1-70b-instruct` (Synthesizes facts across contexts better).

### 2. Mathematical Reasoning
*   **Arithmetic / Basic Algebra:** `gemma2-9b-it` (If Stage 1/2 tool eval fails).
*   **Proof-based / Advanced Word Problems:** `qwen2p5-72b-instruct` or `gemma2-27b-it` (Highly optimized math kernels).

### 3. Sentiment Classification
*   *Almost never escalates past Stage 1/2.* If it does (e.g., highly sarcastic or ambiguous text): `llama-v3p1-8b-instruct`.

### 4. Text Summarization
*   **Short Prompts (< 2k tokens):** `llama-v3p1-8b-instruct` or `gemma2-9b-it`.
*   **Long-context / Highly Technical Summarization:** `llama-v3p1-70b-instruct` (Leverages 128k context window).

### 5. Named Entity Recognition
*   *Almost never escalates past spaCy/Local.* If complex/nested extraction is needed: `llama-v3p1-8b-instruct`.

### 6. Code Debugging
*   **Simple Syntax / Standard library errors:** `llama-v3p1-8b-instruct` (Check candidate fix using Stage 1 sandbox).
*   **Logical / Multi-file / Concurrency bugs:** `llama-v3p1-70b-instruct` or `qwen2p5-72b-instruct`.

### 7. Logical Reasoning
*   **Puzzles / Constraint Problems:** `gemma2-27b-it` or `llama-v3p1-70b-instruct`.
*   **Fallback extreme logical validation:** `llama-v3p1-405b-instruct`.

### 8. Code Generation
*   **Boilerplate / Algorithms:** `llama-v3p1-8b-instruct` or local model.
*   **Complex architectural / API orchestration:** `qwen2p5-72b-instruct` or `llama-v3p1-70b-instruct`.

---

## Strategic Implications for the Routing Agent

1.  **Exploit the Llama 3.1 8B vs. 70B/405B Cost Differential:** Llama 3.1 8B is 20x cheaper than Llama 3.1 405B. Under the metric `accuracy ÷ cost`, escalating directly to the 405B model for a simple task is a massive penalty.
2.  **Fallback to Self-Correction:** When using cheaper models like `llama-v3p1-8b-instruct` or `gemma2-9b-it`, if their outputs fail the verification gate (e.g., tests fail), the agent should not give up. Instead, it can escalate to `llama-v3p1-70b-instruct` or `qwen2p5-72b-instruct`. This keeps token costs low for easy-to-medium tasks while protecting overall accuracy on hard tasks.

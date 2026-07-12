# Running the LLM Routing Agent Container

This guide documents how to build and run the LLM Routing Agent inside a Docker container. 

The image uses a lean, CPU-only configuration by default (installing only `requirements.txt`). However, it supports AMD/ROCm GPU device access at run-time for local GPU inference via Ollama.

---

## 🛠️ Building the Image

Build the container image using the following command from the project root:

```bash
docker build -t amd-router .
```

---

## 🚀 Running the Container

### Local GPU / AMD ROCm Passthrough

### Local GPU / AMD ROCm Passthrough

To allow the container's own bundled Ollama service to access AMD ROCm GPU resources for accelerated hardware execution, pass the device mapping and video/render groups at run-time.

> [!WARNING]
> Do **NOT** use `--gpus all`. This flag is NVIDIA-specific and will silently fail or do nothing on AMD systems.

Instead, use:
* `--device=/dev/kfd`
* `--device=/dev/dri`
* `--group-add video`
* `--group-add render`

### Full Example Command

Run a batch job by mounting your input/output directories and specifying the CLI arguments:

```bash
docker run --rm \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  -v "/path/to/local/data:/data" \
  -e PAID_PROVIDER="fireworks" \
  -e FIREWORKS_API_KEY="your-fireworks-api-key" \
  -e GROQ_API_KEY="your-groq-api-key" \
  -e OPENROUTER_API_KEY="your-openrouter-api-key" \
  amd-router --input /data/dataset.jsonl --output /data/results.jsonl
```

---

## ⚙️ Configuration (Environment Variables)

The container accepts the following environment variables to override runtime settings (configured in [router/config.py](file:///app/router/config.py)):

| Environment Variable | Default Value | Description |
| :--- | :--- | :--- |
| `PAID_PROVIDER` | `groq` | Cloud LLM provider for paid routing (`groq` or `fireworks`). |
| `GROQ_API_KEY` | `""` | API key for Groq. |
| `FIREWORKS_API_KEY` | `""` | API key for Fireworks AI. |
| `OPENROUTER_API_KEY` | `""` | API key for OpenRouter (used for critique labeling). |
| `OLLAMA_HOST` | `http://localhost:11434` | Endpoint of the Ollama server. |
| `USE_CLOUD_FALLBACK` | *None* | Set to `true` or `false` to force/bypass cloud fallback mode. If unset, decided dynamically at startup via health check. |
| `SENTIMENT_THRESHOLD`| `0.5` | Threshold for Stage 1 sentiment classification confidence. |
| `FACTUAL_K` | `2` | Parameter for Stage 1 factual knowledge check. |
| `LOGIC_K` | `3` | Parameter for Stage 1 logic reasoning check. |
| `CODE_RETRY_LIMIT` | `2` | Maximum retries for code debugging and generation domains. |
| `SANDBOX_TIMEOUT_S` | `2.0` | Timeout in seconds for sandboxed execution of code. |
| `LOCAL_MAX_LATENCY_S`| `3.0` | Health check latency budget in seconds for local GPU model generation. |
| `TASK_LOG_PATH` | `logs/tasks.jsonl` | Path inside the container where task telemetry is saved. |

#!/bin/bash
# Harness contract: this script must NEVER exit nonzero because of Ollama. (Trigger Build)
# The pipeline's startup health check (router/health.py) flips Stage 2 to the
# cheap cloud fallback whenever local inference is unavailable, so every
# Ollama problem here is survivable — warn and keep going.

echo ">>> Starting Ollama server in background..."
ollama serve &

# Wait for Ollama to respond to API requests
TIMEOUT=20
OLLAMA_READY=0
start_time=$(date +%s)
echo ">>> Waiting for Ollama API to be ready (timeout: ${TIMEOUT}s)..."

while true; do
    if curl -s http://localhost:11434/ >/dev/null; then
        echo ">>> Ollama API is ready!"
        OLLAMA_READY=1
        break
    fi

    current_time=$(date +%s)
    elapsed=$((current_time - start_time))
    if [ "$elapsed" -ge "$TIMEOUT" ]; then
        echo ">>> WARNING: Ollama did not start within ${TIMEOUT}s; continuing without it (cloud fallback covers Stage 2)."
        break
    fi
    sleep 1
done

# The model is baked into the image at build time; this pull is a no-op fast
# path that only downloads if the baked layer is missing. Bounded so a
# blocked-but-hanging registry connection can't eat the grading runtime budget.
if [ "$OLLAMA_READY" -eq 1 ]; then
    echo ">>> Ensuring model llama3.2:3b is present..."
    if ! timeout 120 ollama pull llama3.2:3b; then
        echo ">>> WARNING: model pull failed or timed out; continuing (cloud fallback covers Stage 2)."
    fi
fi

# Pass through all incoming CLI arguments to the main python script
echo ">>> Executing router main command: python -m router.main $@"
exec python -m router.main "$@"

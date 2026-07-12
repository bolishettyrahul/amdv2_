#!/bin/bash
set -e

echo ">>> Starting Ollama server in background..."
ollama serve &

# Wait for Ollama to respond to API requests
TIMEOUT=20
start_time=$(date +%s)
echo ">>> Waiting for Ollama API to be ready (timeout: ${TIMEOUT}s)..."

while true; do
    # Perform a quick curl status check
    if curl -s http://localhost:11434/ >/dev/null; then
        echo ">>> Ollama API is ready!"
        break
    fi
    
    current_time=$(date +%s)
    elapsed=$((current_time - start_time))
    if [ $elapsed -ge $TIMEOUT ]; then
        echo ">>> ERROR: Ollama server failed to start within ${TIMEOUT} seconds."
        exit 1
    fi
    sleep 1
done

# Pass through all incoming CLI arguments to the main python script
echo ">>> Executing router main command: python -m router.main $@"
exec python -m router.main "$@"

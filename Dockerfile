FROM python:3.13-slim

WORKDIR /app

# Copy requirements file first to leverage layer caching
COPY requirements.txt /app/

# Install only the core dependencies (lightweight, CPU-only baseline),
# plus the spaCy NER model Stage 1 uses to resolve NER tasks at $0
RUN pip install --no-cache-dir -r requirements.txt && \
    python -m spacy download en_core_web_sm

# Install Ollama and curl/ca-certificates/zstd, then trim GPU libs we can
# never use: CUDA (AMD-only judging) and every rocBLAS kernel except gfx942
# (MI300X). This is what keeps the image ~3.7GB instead of ~12GB.
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates zstd && \
    curl -fsSL https://ollama.com/install.sh | sh && \
    curl -fL https://ollama.com/download/ollama-linux-amd64-rocm.tar.zst -o /tmp/ollama-rocm.tar.zst && \
    tar -C /usr/local -xf /tmp/ollama-rocm.tar.zst && \
    find /usr/local/lib/ollama -type d -name "*cuda*" -exec rm -rf {} + && \
    find /usr/local/lib/ollama -path "*/rocblas/library/*" -type f ! -name "*gfx942*" -delete && \
    du -sh /usr/local/lib/ollama && \
    rm -rf /tmp/ollama-rocm.tar.zst /var/lib/apt/lists/*

# Bake the Stage 2 model into the image so the grading sandbox never needs
# egress to ollama.com (it may be blocked, and a runtime pull risks the
# per-run time limit). Placed before the source COPY so code edits don't
# invalidate this ~2GB layer.
RUN ollama serve & \
    for i in $(seq 1 30); do curl -s http://localhost:11434/ >/dev/null && break; sleep 1; done && \
    ollama pull llama3.2:3b

# Copy the router source code and helper scripts
COPY router/ /app/router/
COPY scripts/ /app/scripts/

# Normalize line endings (a Windows checkout would inject \r and break bash)
# and make the entrypoint script executable
RUN sed -i 's/\r$//' /app/scripts/entrypoint.sh && chmod +x /app/scripts/entrypoint.sh

# Set the default entrypoint to run the entrypoint script
ENTRYPOINT ["/bin/bash", "/app/scripts/entrypoint.sh"]

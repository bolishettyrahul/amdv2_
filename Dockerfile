FROM python:3.13-slim

WORKDIR /app

# Copy requirements file first to leverage layer caching
COPY requirements.txt /app/

# Install only the core dependencies (lightweight, CPU-only baseline)
RUN pip install --no-cache-dir -r requirements.txt

# Install Ollama and curl/ca-certificates/zstd
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates zstd && \
    curl -fsSL https://ollama.com/install.sh | sh && \
    curl -L https://ollama.com/download/ollama-linux-amd64-rocm.tar.zst -o /tmp/ollama-rocm.tar.zst && \
    tar -C /usr/local -xf /tmp/ollama-rocm.tar.zst && \
    rm -rf /tmp/ollama-rocm.tar.zst /var/lib/apt/lists/*

# Pull Ollama model during the build process (only llama3.2:3b for consolidated Option B)
RUN ollama serve & \
    sleep 5 && \
    ollama pull llama3.2:3b

# Copy the router source code and helper scripts
COPY router/ /app/router/
COPY scripts/ /app/scripts/

# Normalize line endings (a Windows checkout would inject \r and break bash)
# and make the entrypoint script executable
RUN sed -i 's/\r$//' /app/scripts/entrypoint.sh && chmod +x /app/scripts/entrypoint.sh

# Set the default entrypoint to run the entrypoint script
ENTRYPOINT ["/bin/bash", "/app/scripts/entrypoint.sh"]

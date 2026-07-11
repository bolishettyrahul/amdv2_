FROM python:3.13-slim

WORKDIR /app

# Copy requirements file first to leverage layer caching
COPY requirements.txt /app/

# Install only the core dependencies (lightweight, CPU-only baseline)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the router source code
COPY router/ /app/router/

# Set the default entrypoint to run the router main module with arguments passed through
ENTRYPOINT ["python", "-m", "router.main"]

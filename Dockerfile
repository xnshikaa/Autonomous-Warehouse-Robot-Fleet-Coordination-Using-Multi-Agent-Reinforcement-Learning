# Dockerfile — Warehouse MARL Inference Service
# Base: python:3.12-slim
FROM python:3.12-slim

WORKDIR /app

# Install OS dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY src/ ./src/
COPY python_infra/ ./python_infra/

# Copy trained artifacts (optional — override at runtime via volume mount or env vars)
COPY artifacts/ ./artifacts/

# Non-root user for security
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Expose inference service port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/health')"

# Default: run inference service
# Override QMIX_CHECKPOINT or QMIX_ONNX environment variables at runtime
ENV PYTHONPATH=/app
CMD ["python", "-m", "uvicorn", "src.api.inference_service:create_inference_app", "--factory", "--host", "0.0.0.0", "--port", "8001"]

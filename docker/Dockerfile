# ── Stage 1: Builder ──────────────────────────────────────
FROM python:3.12-slim AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Set explicit cache path for fastembed
ENV FASTEMBED_CACHE_PATH=/app/fastembed_cache

# Copy dependency files
COPY pyproject.toml .
COPY uv.lock* .

# Install dependencies into virtual environment
RUN uv sync --frozen --no-dev --no-cache

# Activate venv
ENV PATH="/app/.venv/bin:$PATH"

# Pre-download ONNX model
RUN python -c "from fastembed import TextEmbedding; \
model = TextEmbedding('sentence-transformers/all-MiniLM-L6-v2'); \
list(model.embed(['warmup'])); \
print('Model cached successfully')"


# ── Stage 2: Runtime ──────────────────────────────────────
FROM python:3.12-slim

WORKDIR /app

# Create non-root user
RUN useradd --create-home appuser

# Copy virtual environment
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv

# Copy model cache
COPY --from=builder --chown=appuser:appuser /app/fastembed_cache \
    /home/appuser/.cache/fastembed

# Copy application code
COPY --chown=appuser:appuser ./app ./app

# Create ChromaDB directory with correct permissions
RUN mkdir -p /app/chroma_db && \
    chown appuser:appuser /app/chroma_db

# Environment variables
ENV PATH="/app/.venv/bin:$PATH" \
    FASTEMBED_CACHE_PATH=/home/appuser/.cache/fastembed \
    CHROMA_DB_DIR=/app/chroma_db \
    PYTHONUNBUFFERED=1 \
    PORT=8080

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8080

# Run FastAPI app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
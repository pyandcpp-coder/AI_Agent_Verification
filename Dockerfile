# Production Dockerfile for AI Agent Verification System
# Multi-stage build for optimized image size

FROM nvidia/cuda:12.6.0-cudnn-runtime-ubuntu24.04 AS base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ca-certificates \
    gnupg \
    wget \
    curl \
    netcat-openbsd \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    gcc \
    g++ \
    tesseract-ocr \
    tesseract-ocr-hin \
    tesseract-ocr-tel \
    tesseract-ocr-ben \
    tesseract-ocr-eng \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# Create app directory and user for security
RUN useradd -m -u 1001 appuser && \
    mkdir -p /app /app/models /app/temp /app/logs && \
    chown -R appuser:appuser /app

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir --break-system-packages -r requirements.txt && \
    pip3 install --no-cache-dir --break-system-packages gunicorn

# Copy application code
COPY --chown=appuser:appuser . .

# Make entrypoint executable
RUN chmod +x docker-entrypoint.sh

# Create necessary directories
RUN mkdir -p temp logs debug_output data && \
    chown -R appuser:appuser temp logs debug_output data

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8101/health || exit 1

# Expose port
EXPOSE 8101

# Set entrypoint
ENTRYPOINT ["./docker-entrypoint.sh"]

# Run with gunicorn for production
CMD ["gunicorn", "main:app", \
     "--bind", "0.0.0.0:8101", \
     "--workers", "1", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--timeout", "300", \
     "--graceful-timeout", "120", \
     "--keep-alive", "5", \
     "--access-logfile", "/app/logs/access.log", \
     "--error-logfile", "/app/logs/error.log", \
     "--log-level", "info"]

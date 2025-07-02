# Multi-stage Dockerfile for py-atranscribe
# Automated Audio Transcription with Speaker Diarization

# Build stage
FROM python:3.11-slim AS builder

# Set build arguments
ARG DEBIAN_FRONTEND=noninteractive

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements first for better caching
COPY requirements.txt /tmp/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# Production stage
FROM python:3.11-slim AS production

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user
RUN groupadd -r transcribe && useradd -r -g transcribe transcribe

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create application directories
RUN mkdir -p /app /data/in /data/out /data/backup /tmp/transcribe && \
    chown -R transcribe:transcribe /app /data /tmp/transcribe

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=transcribe:transcribe . .

# Create directories for models cache
RUN mkdir -p /home/transcribe/.cache && \
    chown -R transcribe:transcribe /home/transcribe

# Switch to non-root user
USER transcribe

# Expose health check port
EXPOSE 8000

# Set default environment variables
ENV CONFIG_PATH=/app/config.yaml
ENV INPUT_DIR=/data/in
ENV OUTPUT_DIR=/data/out
ENV BACKUP_DIR=/data/backup
ENV TEMP_DIR=/tmp/transcribe
ENV LOG_LEVEL=INFO

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Volume mounts
VOLUME ["/data/in", "/data/out", "/data/backup"]

# Default command
CMD ["python", "auto_diarize_transcribe.py"]
# Build stage
FROM python:3.11.9-slim AS builder

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    ffmpeg \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create venv and install deps
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11.9-slim AS production

LABEL maintainer="Your Name <your.email@example.com>"
LABEL org.opencontainers.image.source="https://github.com/your/repo"
LABEL org.opencontainers.image.description="Automated Audio Transcription with Speaker Diarization"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    CONFIG_PATH=/app/config.yaml \
    INPUT_DIR=/data/in \
    OUTPUT_DIR=/data/out \
    BACKUP_DIR=/data/backup \
    TEMP_DIR=/tmp/transcribe \
    LOG_LEVEL=INFO

RUN groupadd -r transcribe && useradd -r -g transcribe -m transcribe

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy venv and set permissions
COPY --from=builder /opt/venv /opt/venv
RUN chown -R transcribe:transcribe /opt/venv

# App, data, and cache dirs
RUN mkdir -p /app /data/in /data/out /data/backup /tmp/transcribe /home/transcribe/.cache && \
    chown -R transcribe:transcribe /app /data /tmp/transcribe /home/transcribe

WORKDIR /app
COPY --chown=transcribe:transcribe . .

USER transcribe

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

VOLUME ["/data/in", "/data/out", "/data/backup"]

ENTRYPOINT ["python", "auto_diarize_transcribe.py"]

# ---- Base image with CUDA runtime and cuDNN ----
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04 AS base

ARG DEBIAN_FRONTEND=noninteractive

# Install Python 3.11 and essential dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    build-essential \
    git \
    ffmpeg \
    libsndfile1 \
    curl \
    wget \
    ca-certificates \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Set up Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# ---- Torch layer: Only rebuilt when torch version changes ----
FROM base AS torch

WORKDIR /torch

# Create a virtualenv to isolate the torch install
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install PyTorch with CUDA 12.8 support (matching the base image)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Verify PyTorch CUDA installation
RUN python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'cuDNN version: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else \"Not available\"}')"

# ---- Builder: All other Python deps except torch* ----
FROM torch AS builder

WORKDIR /app

COPY requirements.txt .

# Remove torch/torchvision/torchaudio from requirements.txt to avoid duplication
RUN grep -vE '^(torch|torchvision|torchaudio)' requirements.txt > requirements.notorch.txt || true

RUN pip install --no-cache-dir -r requirements.notorch.txt \
    && pip install --no-cache-dir torchmetrics

# ---- Production: Minimal runtime with cuDNN ----
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04 AS production

LABEL maintainer="Robin Collins <robin.f.collins@outlook.com>"
LABEL org.opencontainers.image.source="https://github.com/robin-collins/py-atranscribe"
LABEL org.opencontainers.image.description="Automated Audio Transcription with Speaker Diarization"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    CONFIG_PATH=/app/config.yaml \
    INPUT_DIR=/data/in \
    OUTPUT_DIR=/data/out \
    BACKUP_DIR=/data/backup \
    TEMP_DIR=/tmp/transcribe \
    LOG_LEVEL=INFO \
    CUDA_LAUNCH_BLOCKING=0 \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 \
    CUDA_MODULE_LOADING=LAZY

ARG DEBIAN_FRONTEND=noninteractive

# Install Python 3.11 and runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    ffmpeg \
    libsndfile1 \
    curl \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Set up Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Create user
RUN groupadd -r transcribe && useradd -r -g transcribe -m transcribe

# Configure sudoers for transcribe user
RUN echo "transcribe    ALL=(ALL)       NOPASSWD:       ALL" > /etc/sudoers.d/transcribe \
    && chmod 0440 /etc/sudoers.d/transcribe

# Copy prebuilt venv with torch from the builder stage
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
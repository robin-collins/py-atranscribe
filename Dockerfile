# Use official PyTorch image with CUDA 12.8 and compatible cuDNN
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel AS base

ARG DEBIAN_FRONTEND=noninteractive

# Install essential dependencies (PyTorch is already included in base image)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    ffmpeg \
    libsndfile1 \
    curl \
    wget \
    ca-certificates \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Verify PyTorch installation from base image
RUN python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'cuDNN version: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else \"Not available\"}')"

# TEMPORARY: List all installed packages for debugging
RUN echo "=== INSTALLED PACKAGES ===" && pip list && echo "=== END PACKAGES ==="

# ---- Builder: Install additional Python dependencies ----
FROM base AS builder

WORKDIR /app

COPY requirements.txt .

# Install additional dependencies (PyTorch packages already in base image)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir torchmetrics

# ---- Production: Use the same PyTorch base for consistency ----
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel AS production

LABEL maintainer="Robin Collins <robin.f.collins@outlook.com>"
LABEL org.opencontainers.image.source="https://github.com/robin-collins/py-atranscribe"
LABEL org.opencontainers.image.description="Automated Audio Transcription with Speaker Diarization"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CONFIG_PATH=/app/config.yaml \
    INPUT_DIR=/data/in \
    OUTPUT_DIR=/data/out \
    BACKUP_DIR=/data/backup \
    TEMP_DIR=/tmp/transcribe \
    LOG_LEVEL=INFO \
    CUDA_LAUNCH_BLOCKING=0 \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True \
    CUDA_MODULE_LOADING=LAZY \
    CUDNN_LOGINFO_DBG=0 \
    CUDNN_LOGERR_DBG=0 \
    CUDNN_LOGWARN_DBG=0 \
    CUDA_CACHE_PATH=/tmp/cuda_cache \
    TORCH_CUDNN_V8_API_ENABLED=1

ARG DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    curl \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Create user
RUN groupadd -r transcribe && useradd -r -g transcribe -m transcribe

# Configure sudoers for transcribe user
RUN echo "transcribe    ALL=(ALL)       NOPASSWD:       ALL" > /etc/sudoers.d/transcribe \
    && chmod 0440 /etc/sudoers.d/transcribe

# Copy Python dependencies from builder stage
COPY --from=builder /opt/conda /opt/conda

# Fix cuDNN library path issue - create symlinks in standard location
RUN mkdir -p /usr/local/cuda/lib64 && \
    ln -sf /opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib/libcudnn*.so* /usr/local/cuda/lib64/ && \
    echo "/usr/local/cuda/lib64" >> /etc/ld.so.conf.d/cuda.conf && \
    echo "/opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib" >> /etc/ld.so.conf.d/cuda.conf && \
    ldconfig

# App, data, and cache dirs
RUN mkdir -p /app /data/in /data/out /data/backup /tmp/transcribe /home/transcribe/.cache && \
    chown -R transcribe:transcribe /app /data /tmp/transcribe /home/transcribe

WORKDIR /app
COPY --chown=transcribe:transcribe . .

# Add entrypoint script (as root, before USER transcribe)
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

USER transcribe

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

VOLUME ["/data/in", "/data/out", "/data/backup"]

ENTRYPOINT ["/entrypoint.sh", "python", "auto_diarize_transcribe.py"]

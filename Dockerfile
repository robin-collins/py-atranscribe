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
# Force reinstall NumPy and SciPy to ensure compatible versions
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    # Remove all potentially conflicting packages
    pip uninstall -y numpy scipy scikit-learn pandas || true && \
    # Clear pip cache to avoid cached incompatible wheels
    pip cache purge && \
    # Install other requirements first (excluding numpy/scipy)
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir torchmetrics && \
    # Force reinstall correct versions AFTER other packages (this ensures they stick)
    pip install --no-cache-dir --force-reinstall "numpy==1.26.4" && \
    pip install --no-cache-dir --force-reinstall "scipy==1.14.1" && \
    # Verify installation - should show NumPy 1.26.4, SciPy 1.14.1
    python -c "import numpy; import scipy; import torch; print(f'NumPy: {numpy.__version__}, SciPy: {scipy.__version__}, PyTorch: {torch.__version__}')" && \
    # Test the problematic import that was failing
    python -c "from scipy import special; print('SciPy special module imported successfully')" && \
    python -c "from pyannote.audio import Pipeline; print('pyannote.audio imported successfully')"

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

# Copy entrypoint script (as root, before USER transcribe).
# This changes less frequently than app code and should be early to be cached.
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /app

# --- Application Setup ---
# Copy files that define the project and its config.
# These layers are cached unless the files change.
COPY --chown=transcribe:transcribe pyproject.toml ./
COPY --chown=transcribe:transcribe config.yaml ./

# Copy the application source code last.
# This ensures that code changes do not invalidate previous, slow-to-build layers,
# leading to much faster rebuilds during development.
COPY --chown=transcribe:transcribe src/ ./src/
COPY --chown=transcribe:transcribe auto_diarize_transcribe.py ./

USER transcribe

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

VOLUME ["/data/in", "/data/out", "/data/backup"]

ENTRYPOINT ["/entrypoint.sh", "python", "auto_diarize_transcribe.py"]

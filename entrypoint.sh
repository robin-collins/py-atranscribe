#!/bin/bash
set -e

# Enhanced cuDNN and CUDA environment setup
export CUDNN_LOGINFO_DBG=${CUDNN_LOGINFO_DBG:-0}
export CUDNN_LOGERR_DBG=${CUDNN_LOGERR_DBG:-1}
export CUDNN_LOGWARN_DBG=${CUDNN_LOGWARN_DBG:-1}
export CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-0}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"max_split_size_mb:512,expandable_segments:True"}
export CUDA_MODULE_LOADING=${CUDA_MODULE_LOADING:-LAZY}
export TORCH_CUDNN_V8_API_ENABLED=${TORCH_CUDNN_V8_API_ENABLED:-1}

# Create CUDA cache directory if it doesn't exist
mkdir -p /tmp/cuda_cache
export CUDA_CACHE_PATH=/tmp/cuda_cache

# Ensure proper permissions for directories
sudo chown -R transcribe:transcribe /data/in /data/out /data/backup /tmp/transcribe /tmp/cuda_cache 2>/dev/null || true

# Create required directories
mkdir -p /data/in /data/out /data/backup /tmp/transcribe

# Set proper permissions
chmod 755 /data/in /data/out /data/backup /tmp/transcribe

echo "🖥️  === Environment Setup entrypoint.sh ==="
echo "🔧  CUDA_LAUNCH_BLOCKING: $CUDA_LAUNCH_BLOCKING"
echo "🔧  PYTORCH_CUDA_ALLOC_CONF: $PYTORCH_CUDA_ALLOC_CONF"
echo "🔧  CUDNN_LOGINFO_DBG: $CUDNN_LOGINFO_DBG"
echo "🔧  TORCH_CUDNN_V8_API_ENABLED: $TORCH_CUDNN_V8_API_ENABLED"
echo "🖥️  ======================================="

# Check if CUDA is available (informational, with unique emoticons)
python -c "
import torch
print('🦄🔸 PyTorch version:', torch.__version__)
print('🦄🔸 CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('🦄🔸 CUDA version:', torch.version.cuda)
    print('🦄🔸 cuDNN enabled:', torch.backends.cudnn.enabled)
    try:
        print('🦄🔸 cuDNN version:', torch.backends.cudnn.version())
    except:
        print('🦄🔸 cuDNN version: Unable to determine')
    print('🦄🔸 GPU count:', torch.cuda.device_count())
    if torch.cuda.device_count() > 0:
        print('🦄🔸 GPU name:', torch.cuda.get_device_name(0))
else:
    print('🦄🔸 CUDA not available - will use CPU')
" 2>/dev/null || echo "🦄🔸 Unable to check CUDA status"

# Execute the main command
exec "$@"

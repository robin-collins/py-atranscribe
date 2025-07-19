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




echo "🖥️  === Environment Setup entrypoint.sh ==="
echo "🔧  CUDA_LAUNCH_BLOCKING: $CUDA_LAUNCH_BLOCKING"
echo "🔧  PYTORCH_CUDA_ALLOC_CONF: $PYTORCH_CUDA_ALLOC_CONF"
echo "🔧  CUDNN_LOGINFO_DBG: $CUDNN_LOGINFO_DBG"
echo "🔧  TORCH_CUDNN_V8_API_ENABLED: $TORCH_CUDNN_V8_API_ENABLED"
echo "🖥️  ======================================="

# Critical dependency test - fail fast if imports don't work
echo "🔍 Testing critical dependencies..."
python -c "
import sys
try:
    import numpy
    print(f'✅ NumPy {numpy.__version__} imported successfully')

    import scipy
    print(f'✅ SciPy {scipy.__version__} imported successfully')

    from scipy import special
    print('✅ SciPy special module imported successfully')

    import torch
    print(f'✅ PyTorch {torch.__version__} imported successfully')

    from pyannote.audio import Pipeline
    print('✅ pyannote.audio imported successfully')

    print('🎉 All critical dependencies imported successfully!')

except Exception as e:
    print(f'❌ CRITICAL IMPORT ERROR: {e}')
    print('🚨 Container will exit to prevent restart loop')
    sys.exit(1)
"

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

# Ensure proper permissions for directories (process one at a time with progress)
echo "🔧 Setting up directory permissions..."

for dir in "/data/in" "/data/out" "/data/backup" "/tmp/transcribe" "/tmp/cuda_cache"; do
    if [ -d "$dir" ]; then
        echo "🔧 Processing directory: $dir"
        file_count=$(find "$dir" -type f 2>/dev/null | wc -l || echo "unknown")
        echo "🔧 Files to process: $file_count"
        sudo chown -R transcribe:transcribe "$dir" 2>/dev/null || echo "⚠️  Failed to change ownership of $dir"
        echo "✅ Completed: $dir"
    else
        echo "⚠️  Directory not found: $dir"
    fi
done

echo "🔧 Directory permissions setup complete"



# Execute the main command
exec "$@"

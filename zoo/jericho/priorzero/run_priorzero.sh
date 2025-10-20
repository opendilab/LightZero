#!/bin/bash
################################################################################
# PriorZero Training Launcher Script
#
# This script handles:
# 1. Proxy settings for model downloads
# 2. vLLM environment configuration
# 3. Error handling and recovery
#
# Usage:
#   bash run_priorzero.sh [--quick_test] [--env_id ENV] [--seed SEED]
#
# Examples:
#   bash run_priorzero.sh --quick_test
#   bash run_priorzero.sh --env_id zork1.z5 --seed 42
################################################################################

set -e  # Exit on error

# ==============================================================================
# 1. Proxy Configuration (for model downloads)
# ==============================================================================
export http_proxy=http://zhangjinouwen:e82NJ6SrPvzUXdwSsgquo88FIGqui2phOEaggXss3w3EFr7Bgu7aIjQhgqT9@10.1.20.50:23128/
export https_proxy=http://zhangjinouwen:e82NJ6SrPvzUXdwSsgquo88FIGqui2phOEaggXss3w3EFr7Bgu7aIjQhgqT9@10.1.20.50:23128/
export HTTP_PROXY=http://zhangjinouwen:e82NJ6SrPvzUXdwSsgquo88FIGqui2phOEaggXss3w3EFr7Bgu7aIjQhgqT9@10.1.20.50:23128/
export HTTPS_PROXY=http://zhangjinouwen:e82NJ6SrPvzUXdwSsgquo88FIGqui2phOEaggXss3w3EFr7Bgu7aIjQhgqT9@10.1.20.50:23128/

echo "✓ Proxy configured for model downloads"

# ==============================================================================
# 2. vLLM Configuration for Shared GPU Environment
# ==============================================================================
# Disable V1 engine to avoid memory profiling issues in shared environments
export VLLM_USE_V1=0

# Optional: Set lower memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Optional: Reduce vLLM logging verbosity
# export VLLM_LOGGING_LEVEL=WARNING

echo "✓ vLLM configured for shared GPU environment"

# ==============================================================================
# 3. CUDA Configuration
# ==============================================================================
# Use a specific GPU if needed (comment out to use all available GPUs)
# export CUDA_VISIBLE_DEVICES=2  # Use GPU 2 (has least memory usage)

# Show GPU status
echo ""
echo "=========================================="
echo "GPU Status:"
echo "=========================================="
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | \
    awk '{printf "GPU %s: %s (%.1f/%.1f GB, %d%% util)\n", $1, $2, $3/1024, $4/1024, $5}'
echo ""

# ==============================================================================
# 4. Run PriorZero Training
# ==============================================================================
echo "=========================================="
echo "Starting PriorZero Training"
echo "=========================================="
echo "Arguments: $@"
echo ""

# Change to script directory
cd "$(dirname "$0")"

# Run with error handling
python priorzero_entry.py "$@"

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Training completed successfully!"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "✗ Training failed with exit code: $exit_code"
    echo "=========================================="
    echo ""
    echo "Common issues and solutions:"
    echo "1. Memory profiling error:"
    echo "   - Check if VLLM_USE_V1=0 is set (should be automatic)"
    echo "   - Try using a dedicated GPU: CUDA_VISIBLE_DEVICES=2 bash run_priorzero.sh"
    echo ""
    echo "2. Out of memory error:"
    echo "   - Reduce batch size in config"
    echo "   - Reduce gpu_memory_utilization in config"
    echo ""
    echo "3. Model download fails:"
    echo "   - Check proxy settings"
    echo "   - Verify network connection"
    echo ""
fi

exit $exit_code

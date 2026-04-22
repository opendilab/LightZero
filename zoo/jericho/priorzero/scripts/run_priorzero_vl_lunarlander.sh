#!/bin/bash
# PriorZero VL Training on LunarLander-v2 (Image Input)
#
# Usage:
#   bash run_priorzero_vl_lunarlander.sh [NUM_GPUS] [VL_MODEL] [SEED] [EXTRA_ARGS...]
#
# Examples:
#   bash run_priorzero_vl_lunarlander.sh 4 Qwen2.5-VL-7b 0
#   bash run_priorzero_vl_lunarlander.sh 2 Qwen3-VL-2b 42
#   bash run_priorzero_vl_lunarlander.sh 1 Qwen3-VL-2b 0 --quick_test
#   bash run_priorzero_vl_lunarlander.sh 1 Qwen3-VL-2b 0 --quick_test --no_cot --no_vl_fixed --mcts_mode wm_logits
#   bash run_priorzero_vl_lunarlander.sh 1 Qwen3-VL-2b 0 --cot_weight 0.05

set -euo pipefail

# ===================== Configurable Parameters =====================
NUM_GPUS=${1:-4}
VL_MODEL=${2:-"Qwen2.5-VL-3b"}
SEED=${3:-0}
EXTRA_ARGS="${@:4}"
# CUDA_DEVICES=${CUDA_DEVICES:-"0,1,2,3"}
CUDA_DEVICES=${CUDA_DEVICES:-"0,1"}
# MASTER_PORT=${MASTER_PORT:-29500}
MASTER_PORT=${MASTER_PORT:-29501}

# ===================================================================

# DDP / NCCL debugging environment variables
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
export PYTHONFAULTHANDLER=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_ID="LunarLander-v2"
TIMESTAMP="$(date +%y%m%d_%H%M%S)"

# ---- Parse key flags from EXTRA_ARGS for log naming ----
COT_TAG="cot"
VL_FIXED_TAG="vlFixed"
MCTS_MODE="llm_plus_wm_logits"
COT_WEIGHT="0.1"
IMG_MODE="current_only"
LOGPROB_MODE="approximate"

for arg in ${EXTRA_ARGS}; do
    case "${prev_arg:-}" in
        --mcts_mode)    MCTS_MODE="$arg" ;;
        --cot_weight)   COT_WEIGHT="$arg" ;;
        --vlm_image_mode) IMG_MODE="$arg" ;;
        --logprob_mode) LOGPROB_MODE="$arg" ;;
    esac
    case "$arg" in
        --no_cot)       COT_TAG="noCot" ;;
        --no_vl_fixed)  VL_FIXED_TAG="vlTrain" ;;
    esac
    prev_arg="$arg"
done

if [ "${COT_TAG}" = "cot" ]; then
    COT_TAG="cot${COT_WEIGHT}"
fi

# Build structured log directory: logs/<env>/<model>/<key_params>/
LOG_DIR="${SCRIPT_DIR}/logs/LunarLander/${VL_MODEL}/${VL_FIXED_TAG}/${COT_TAG}_mcts_${MCTS_MODE}_img_${IMG_MODE}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/seed${SEED}_gpu${NUM_GPUS}_${TIMESTAMP}.log"

echo "========================================"
echo "PriorZero VL - LunarLander-v2 (Image)"
echo "========================================"
echo "GPUs:        ${NUM_GPUS}"
echo "VL Model:    ${VL_MODEL}"
echo "Seed:        ${SEED}"
echo "Extra Args:  ${EXTRA_ARGS}"
echo "CUDA:        ${CUDA_DEVICES}"
echo "Master Port: ${MASTER_PORT}"
echo "Log File:    ${LOG_FILE}"
echo "========================================"

cd "${SCRIPT_DIR}"

torchrun \
    --nproc_per_node "${NUM_GPUS}" \
    --master-port "${MASTER_PORT}" \
    priorzero_entry_unified.py \
    --input_type image \
    --env_id "${ENV_ID}" \
    --vl_model "${VL_MODEL}" \
    --seed "${SEED}" \
    --max_iter 1e6 \
    ${EXTRA_ARGS} \
    2>&1 | tee "${LOG_FILE}"

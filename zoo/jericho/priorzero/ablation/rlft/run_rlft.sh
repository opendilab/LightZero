#!/bin/bash
set -e
set -x
set -o pipefail

PRIORZERO_DIR="/mnt/afs/niuyazhe/workspace/xiongjyu/LightZero/zoo/jericho/priorzero"

CUDA_DEVICES="${CUDA_DEVICES:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-24564}"

ENV_ID="${ENV_ID:-detective.z5}"
LLM_MODEL="${LLM_MODEL:-qwen2.5-3b}"
HIS_LEN="${HIS_LEN:-25}"
SEEDS="${SEEDS:-0 1}"
MAX_ENV_STEPS="${MAX_ENV_STEPS:-100000}"
ROLLOUT_EPISODES_PER_ITER="${ROLLOUT_EPISODES_PER_ITER:-50}"
LOG_DIR="${LOG_DIR:-${PRIORZERO_DIR}/data_ablation/run_logs}"

mkdir -p "${LOG_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
export PYTHONFAULTHANDLER=1
export TOKENIZERS_PARALLELISM=false
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-OFF}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

cd "${PRIORZERO_DIR}"

for SEED in ${SEEDS}; do
    CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
    LOG_FILE="${LOG_DIR}/rlft_${ENV_ID}_${LLM_MODEL}_his${HIS_LEN}_seed${SEED}_${CURRENT_TIME}.txt"
    RUN_MASTER_PORT=$((MASTER_PORT + SEED))

    torchrun \
        --nproc_per_node="${NPROC_PER_NODE}" \
        --master-port="${RUN_MASTER_PORT}" \
        "${PRIORZERO_DIR}/ablation/rlft/run_rlft.py" \
        --env_id "${ENV_ID}" \
        --model "${LLM_MODEL}" \
        --seed "${SEED}" \
        --history_len "${HIS_LEN}" \
        --max_env_steps "${MAX_ENV_STEPS}" \
        --rollout_episodes_per_iter "${ROLLOUT_EPISODES_PER_ITER}" \
        2>&1 | tee "${LOG_FILE}"
done

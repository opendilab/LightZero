#!/bin/bash
set -x
set -o pipefail

PRIORZERO_DIR="/mnt/afs/niuyazhe/workspace/xiongjyu/LightZero/zoo/jericho/priorzero"
PYTHON_BIN="/mnt/afs/niuyazhe/workspace/xiongjyu/envs/rft/bin/python"

CUDA_DEVICES="${CUDA_DEVICES:-0}"
ENV_ID="${ENV_ID:-detective.z5}"
LLM_MODEL="${LLM_MODEL:-qwen2.5-3b}"
HIS_LEN="${HIS_LEN:-25}"
SEEDS="${SEEDS:-0 1}"
LOG_DIR="${LOG_DIR:-${PRIORZERO_DIR}/data_ablation/run_logs}"

mkdir -p "${LOG_DIR}"
CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/llm_as_policy_${ENV_ID}_${LLM_MODEL}_his${HIS_LEN}_${CURRENT_TIME}.txt"

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
export PYTHONFAULTHANDLER=1
export TOKENIZERS_PARALLELISM=false

cd "${PRIORZERO_DIR}"

"${PYTHON_BIN}" \
    "${PRIORZERO_DIR}/ablation/llm_as_policy/run_llm_as_policy.py" \
    --env_id "${ENV_ID}" \
    --model "${LLM_MODEL}" \
    --history_len "${HIS_LEN}" \
    --seeds ${SEEDS} \
    2>&1 | tee "${LOG_FILE}"

#!/bin/bash
set -x

cd /mnt/shared-storage-user/puyuan/code/LightZero/zoo/babyai/priorzero
export PYTHONPATH=/mnt/shared-storage-user/puyuan/code/LightZero:$PYTHONPATH

# ============================================================================
# PREREQUISITE: Start BabyAI server FIRST
#   cd /path/to/AgentGym-RL/AgentGym/agentenv-babyai
#   python -m agentenv_babyai.launch --port 8000
# ============================================================================

# 1. Training environment parameters
CUDA_DEVICES="0,1,2,3"
NPROC_PER_NODE=4
MASTER_PORT=24554

# 2. BabyAI-specific parameters
AGENTGYM_SERVER_ADDR="http://127.0.0.1:8000"
DATA_IDX=0                   # level = data_idx % 40 + 1, seed = data_idx // 40
USE_HIGH_LEVEL=true          # true = server high-level actions, false = 7 atomic actions

# 3. Model parameters
LLM_MODEL="qwen2.5-3b"      # "qwen2.5-0.5b" "qwen2.5-1.5b" "qwen2.5-3b" "qwen2.5-7b"
USE_COT=false
LOG_DIR="./data_priorzero/babyai/run_logs"
mkdir -p "${LOG_DIR}"

CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
LEVEL_ID=$(( DATA_IDX % 40 + 1 ))
LOG_FILE="${LOG_DIR}/log_level${LEVEL_ID}_${LLM_MODEL}_${CURRENT_TIME}.txt"

# 4. Environment variables
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
export PYTHONFAULTHANDLER=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO

# 5. Build command
CMD_ARGS="--env_id babyai --env_addr ${AGENTGYM_SERVER_ADDR} --data_idx ${DATA_IDX} --model ${LLM_MODEL}"

if [ "${USE_COT}" = true ]; then
    CMD_ARGS="${CMD_ARGS} --use_cot"
fi

if [ "${USE_HIGH_LEVEL}" = false ]; then
    CMD_ARGS="${CMD_ARGS} --use_low_level_actions"
fi

torchrun \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master-port="${MASTER_PORT}" \
    ./src/priorzero_entry_sync_ddp.py \
    ${CMD_ARGS} \
    2>&1 | tee "${LOG_FILE}"

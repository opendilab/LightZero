
#!/bin/bash
set -x

# 1. 训练环境参数
CUDA_DEVICES="0,1,2,3"      
NPROC_PER_NODE=4          
MASTER_PORT=24554          

# 2. 程序相关参数
ENV_ID="detective.z5"       # "zork1.z5" "acorncourt.z5" "omniquest.z5"
LOG_DIR="./data_priorzero_0719_try/run_logs"   
LLM_MODEL="qwen2.5-3b"  # "qwen2.5-3b" "qwen2.5-7b"
USE_COT=true                # true / false
mkdir -p "${LOG_DIR}"       
SEED=0

CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/log_${ENV_ID}_${LLM_MODEL}_${CURRENT_TIME}.txt"

# 3. 设置环境变量 
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
export PYTHONFAULTHANDLER=1 
export TORCH_DISTRIBUTED_DEBUG=DETAIL 
export NCCL_DEBUG=INFO 

if [ "${USE_COT}" = true ]; then
    torchrun \
        --nproc_per_node="${NPROC_PER_NODE}" \
        --master-port="${MASTER_PORT}" \
        ./src/priorzero_entry_sync_ddp.py \
        --use_cot \
        --env_id "${ENV_ID}" \
        --model "${LLM_MODEL}" \
        --seed "${SEED}" \
        2>&1 | tee "${LOG_FILE}"
else
    torchrun \
        --nproc_per_node="${NPROC_PER_NODE}" \
        --master-port="${MASTER_PORT}" \
        ./src/priorzero_entry_sync_ddp.py \
        --env_id "${ENV_ID}" \
        --model "${LLM_MODEL}" \
        --seed "${SEED}" \
        2>&1 | tee "${LOG_FILE}"
fi
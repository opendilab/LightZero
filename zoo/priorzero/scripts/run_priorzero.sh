
#!/bin/bash
set -x

# 1. 训练环境参数
CUDA_DEVICES="0"      
NPROC_PER_NODE=1           
MASTER_PORT=24554          

# 2. 程序相关参数
ENV_ID="detective.z5"       # "zork1.z5" "acorncourt.z5" "omniquest.z5"
LOG_DIR="./data_priorzero/run_logs"        
LLM_MODEL="qwen2.5-3b"  # "qwen2.5-3b" "qwen2.5-7b"
mkdir -p "${LOG_DIR}"     

CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/log_${ENV_ID}_${LLM_MODEL}_${CURRENT_TIME}.txt"

# 3. 设置环境变量 
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
export PYTHONFAULTHANDLER=1 
export TORCH_DISTRIBUTED_DEBUG=DETAIL 
export NCCL_DEBUG=INFO 


torchrun \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master-port="${MASTER_PORT}" \
    ./src/priorzero_entry_sync.py \
    --use_cot \
    --env_id "${ENV_ID}" \
    --model  "${LLM_MODEL}" \
    2>&1 | tee "${LOG_FILE}"
#!/bin/bash
set -x

cd /mnt/shared-storage-user/puyuan/code/LightZero/zoo/babyai/priorzero
export PYTHONPATH=/mnt/shared-storage-user/puyuan/code/LightZero:$PYTHONPATH

torchrun --nproc_per_node=1 --master-port=24554 ./src/priorzero_entry_sync_ddp.py --quick_test --env_addr http://127.0.0.1:8000 --data_idx 0 --model qwen2.5-3b

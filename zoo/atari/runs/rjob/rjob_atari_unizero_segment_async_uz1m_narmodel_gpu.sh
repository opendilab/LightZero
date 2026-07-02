#!/bin/bash

################################################################################
# Submit UniZero Atari async uz-1m jobs to narmodel_gpu.
#
# Defaults:
#   - Pong and MsPacman
#   - fast_noaux UniZero variant
#   - 1,000,000 env steps per task
#   - two GPUs requested so both games can run concurrently by default
################################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SCRIPT="${RUN_SCRIPT:-${SCRIPT_DIR}/run_atari_unizero_segment_rjob.sh}"

export MODE="${MODE:-multitask}"
export ATARI_ENVS="${ATARI_ENVS:-ALE/Pong-v5,ALE/MsPacman-v5}"
export SEEDS="${SEEDS:-0}"
export BASELINE_VARIANTS="${BASELINE_VARIANTS:-fast_noaux}"
export MAX_ENV_STEP="${MAX_ENV_STEP:-1000000}"
export RUN_TAG="${RUN_TAG:-uz-atari-segment-async-uz1m-$(date '+%y%m%d_%H%M%S')}"

export ASYNC_PIPELINE="${ASYNC_PIPELINE:-1}"
export NUM_COLLECTOR_ACTORS="${NUM_COLLECTOR_ACTORS:-1}"
export MAX_POLICY_LAG="${MAX_POLICY_LAG:-0}"
export MAX_TRAIN_CHUNK_STEPS="${MAX_TRAIN_CHUNK_STEPS:-2}"
export WEIGHT_SYNC_INTERVAL="${WEIGHT_SYNC_INTERVAL:-1}"
export COLLECTOR_NUM_GPUS="${COLLECTOR_NUM_GPUS:-0}"
export EVALUATOR_NUM_GPUS="${EVALUATOR_NUM_GPUS:-0}"
export USE_NEW_CACHE_MANAGER="${USE_NEW_CACHE_MANAGER:-1}"
export SAVE_CKPT="${SAVE_CKPT:-0}"

export RJOB_NAME="${RJOB_NAME:-uz-atari-unizero-async-uz1m}"
export RJOB_CHARGED_GROUP="${RJOB_CHARGED_GROUP:-narmodel_gpu}"
export RJOB_GPU="${RJOB_GPU:-2}"
export MAX_PARALLEL="${MAX_PARALLEL:-${RJOB_GPU}}"
export RJOB_CPU="${RJOB_CPU:-48}"
export RJOB_MEMORY="${RJOB_MEMORY:-450000}"

exec "${SCRIPT_DIR}/rjob_atari_unizero_segment_1node_8gpu.sh"

#!/bin/bash

################################################################################
# Submit UniZero Atari segment baselines as a detached rjob task.
#
# The paired run script executes inside the rjob worker and launches one
# single-GPU training process per baseline task. The default request is one
# GPU; set RJOB_GPU/MAX_PARALLEL explicitly for multi-task sweeps.
################################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SCRIPT="${RUN_SCRIPT:-${SCRIPT_DIR}/run_atari_unizero_segment_rjob.sh}"
LIGHTZERO_HOME="${LIGHTZERO_HOME:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)}"

MODE="${MODE:-multitask}"
ATARI_ENVS="${ATARI_ENVS:-ALE/Pong-v5,ALE/MsPacman-v5}"
SEEDS="${SEEDS:-0}"
BASELINE_VARIANTS="${BASELINE_VARIANTS:-fast_noaux,compile_noaux,recon_only,recon_lpips}"
RUN_TAG="${RUN_TAG:-uz-atari-segment-baseline-$(date '+%y%m%d_%H%M%S')}"
EXP_ROOT="${EXP_ROOT:-data_unizero/rjob}"
RJOB_LOG_ROOT="${RJOB_LOG_ROOT:-${LIGHTZERO_HOME}/rjob_logs/atari_unizero_segment}"
MAX_ENV_STEP="${MAX_ENV_STEP:-}"
TOKENIZER_PRETRAINED_VGG="${TOKENIZER_PRETRAINED_VGG:-${LIGHTZERO_HOME}/tokenizer_pretrained_vgg}"
TORCH_HOME="${TORCH_HOME:-${TOKENIZER_PRETRAINED_VGG}}"
USE_NEW_CACHE_MANAGER="${USE_NEW_CACHE_MANAGER:-1}"
SAVE_CKPT="${SAVE_CKPT:-0}"
ASYNC_PIPELINE="${ASYNC_PIPELINE:-0}"
NUM_COLLECTOR_ACTORS="${NUM_COLLECTOR_ACTORS:-1}"
MAX_POLICY_LAG="${MAX_POLICY_LAG:-0}"
MAX_TRAIN_CHUNK_STEPS="${MAX_TRAIN_CHUNK_STEPS:-4}"
WEIGHT_SYNC_INTERVAL="${WEIGHT_SYNC_INTERVAL:-1}"
COLLECTOR_NUM_GPUS="${COLLECTOR_NUM_GPUS:-0}"
EVALUATOR_NUM_GPUS="${EVALUATOR_NUM_GPUS:-0}"
SMOKE_TEST="${SMOKE_TEST:-0}"

RJOB_NAME="${RJOB_NAME:-uz-atari-unizero-segment-1n1g}"
RJOB_MEMORY="${RJOB_MEMORY:-300000}"
RJOB_CPU="${RJOB_CPU:-32}"
RJOB_GPU="${RJOB_GPU:-1}"
MAX_PARALLEL="${MAX_PARALLEL:-${RJOB_GPU}}"
RJOB_CHARGED_GROUP="${RJOB_CHARGED_GROUP:-narmodel_gpu}"
RJOB_PRIVATE_MACHINE="${RJOB_PRIVATE_MACHINE:-group}"
RJOB_IMAGE="${RJOB_IMAGE:-registry.h.pjlab.org.cn/ailab-rlinfra-rlinfra_gpu/rft:20260408}"
RJOB_REPLICA="${RJOB_REPLICA:-1}"
RJOB_CUSTOM_RESOURCES="${RJOB_CUSTOM_RESOURCES:-brainpp.cn/fuse=1}"

CONDA_ENV_PATH="${CONDA_ENV_PATH:-/mnt/shared-storage-user/puyuan/conda_envs/lz}"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_ENV_PATH}/bin/python}"
CONFIG_SCRIPT="${CONFIG_SCRIPT:-${LIGHTZERO_HOME}/zoo/atari/config/atari_unizero_segment_config.py}"

if [[ ! -f "${RUN_SCRIPT}" ]]; then
    echo "Run script not found: ${RUN_SCRIPT}" >&2
    exit 1
fi

case "${RJOB_PRIVATE_MACHINE,,}" in
    yes|true|1)
        RJOB_PRIVATE_MACHINE="group"
        ;;
    false|0)
        RJOB_PRIVATE_MACHINE="no"
        ;;
    group|no|project|tenant)
        RJOB_PRIVATE_MACHINE="${RJOB_PRIVATE_MACHINE,,}"
        ;;
    *)
        echo "Invalid RJOB_PRIVATE_MACHINE=${RJOB_PRIVATE_MACHINE}; expected group, no, project, tenant, or legacy yes/true/1/false/0." >&2
        exit 1
        ;;
esac

submit_cmd=(
    rjob submit
    --name="${RJOB_NAME}"
    --gpu="${RJOB_GPU}"
    --memory="${RJOB_MEMORY}"
    --cpu="${RJOB_CPU}"
    --charged-group="${RJOB_CHARGED_GROUP}"
    --private-machine="${RJOB_PRIVATE_MACHINE}"
    -P "${RJOB_REPLICA}"
    --image="${RJOB_IMAGE}"
    --mount=gpfs://gpfs1/puyuan:/mnt/shared-storage-user/puyuan
    --mount=gpfs://gpfs1/luyudong:/mnt/shared-storage-user/luyudong
    --mount=gpfs://gpfs2/gpfs2-shared-public:/mnt/shared-storage-gpfs2/gpfs2-shared-public
    --mount=gpfs://gpfs2/narmodel:/mnt/shared-storage-user/narmodel
    -e INSIDE_RJOB=1
    -e SUBMIT_RJOB=0
    -e LIGHTZERO_HOME="${LIGHTZERO_HOME}"
    -e MODE="${MODE}"
    -e ATARI_ENVS="${ATARI_ENVS}"
    -e SEEDS="${SEEDS}"
    -e BASELINE_VARIANTS="${BASELINE_VARIANTS}"
    -e RUN_TAG="${RUN_TAG}"
    -e EXP_ROOT="${EXP_ROOT}"
    -e RJOB_LOG_ROOT="${RJOB_LOG_ROOT}"
    -e MAX_ENV_STEP="${MAX_ENV_STEP}"
    -e MAX_PARALLEL="${MAX_PARALLEL}"
    -e TOKENIZER_PRETRAINED_VGG="${TOKENIZER_PRETRAINED_VGG}"
    -e TORCH_HOME="${TORCH_HOME}"
    -e USE_NEW_CACHE_MANAGER="${USE_NEW_CACHE_MANAGER}"
    -e SAVE_CKPT="${SAVE_CKPT}"
    -e ASYNC_PIPELINE="${ASYNC_PIPELINE}"
    -e NUM_COLLECTOR_ACTORS="${NUM_COLLECTOR_ACTORS}"
    -e MAX_POLICY_LAG="${MAX_POLICY_LAG}"
    -e MAX_TRAIN_CHUNK_STEPS="${MAX_TRAIN_CHUNK_STEPS}"
    -e WEIGHT_SYNC_INTERVAL="${WEIGHT_SYNC_INTERVAL}"
    -e COLLECTOR_NUM_GPUS="${COLLECTOR_NUM_GPUS}"
    -e EVALUATOR_NUM_GPUS="${EVALUATOR_NUM_GPUS}"
    -e SMOKE_TEST="${SMOKE_TEST}"
    -e RJOB_MEMORY="${RJOB_MEMORY}"
    -e RJOB_CPU="${RJOB_CPU}"
    -e RJOB_GPU="${RJOB_GPU}"
    -e RJOB_CHARGED_GROUP="${RJOB_CHARGED_GROUP}"
    -e RJOB_PRIVATE_MACHINE="${RJOB_PRIVATE_MACHINE}"
    -e RJOB_CUSTOM_RESOURCES="${RJOB_CUSTOM_RESOURCES}"
    -e RJOB_IMAGE="${RJOB_IMAGE}"
    -e CONDA_ENV_PATH="${CONDA_ENV_PATH}"
    -e PYTHON_BIN="${PYTHON_BIN}"
    -e CONFIG_SCRIPT="${CONFIG_SCRIPT}"
    --custom-resources "${RJOB_CUSTOM_RESOURCES}"
    -- bash -exc "${RUN_SCRIPT}"
)

echo "Submitting UniZero Atari rjob:"
echo "  name:       ${RJOB_NAME}"
echo "  run_tag:    ${RUN_TAG}"
echo "  group:      ${RJOB_CHARGED_GROUP}"
echo "  gpu:        ${RJOB_GPU}"
echo "  envs:       ${ATARI_ENVS}"
echo "  seeds:      ${SEEDS}"
echo "  variants:   ${BASELINE_VARIANTS}"
echo "  logs:       ${RJOB_LOG_ROOT}/${RUN_TAG}"
echo "  tensorboard:${LIGHTZERO_HOME}/${EXP_ROOT}/${RUN_TAG}"
echo "  torch_home: ${TORCH_HOME}"
echo "  new_cache:  ${USE_NEW_CACHE_MANAGER}"
echo "  save_ckpt:  ${SAVE_CKPT}"
echo "  async:      ${ASYNC_PIPELINE}"
echo "  smoke:      ${SMOKE_TEST}"

if [[ "${RJOB_DRY_RUN:-0}" == "1" ]]; then
    printf 'Dry-run command:'
    printf ' %q' "${submit_cmd[@]}"
    printf '\n'
    exit 0
fi

exec "${submit_cmd[@]}"

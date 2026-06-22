#!/bin/bash

################################################################################
# Submit UniZero Atari segment baselines as a detached rjob task.
#
# The paired run script executes inside the rjob worker and launches one
# single-GPU training process per baseline task.
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
MAX_PARALLEL="${MAX_PARALLEL:-8}"

RJOB_NAME="${RJOB_NAME:-uz-atari-unizero-segment-1n8g}"
RJOB_MEMORY="${RJOB_MEMORY:-1500000}"
RJOB_CPU="${RJOB_CPU:-150}"
RJOB_GPU="${RJOB_GPU:-8}"
RJOB_CHARGED_GROUP="${RJOB_CHARGED_GROUP:-rlinfra_gpu}"
RJOB_PRIVATE_MACHINE="${RJOB_PRIVATE_MACHINE:-yes}"
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
    --mount=gpfs://gpfs1/luyudong:/mnt/shared-storage-user/tangjia
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

if [[ "${RJOB_DRY_RUN:-0}" == "1" ]]; then
    printf 'Dry-run command:'
    printf ' %q' "${submit_cmd[@]}"
    printf '\n'
    exit 0
fi

exec "${submit_cmd[@]}"

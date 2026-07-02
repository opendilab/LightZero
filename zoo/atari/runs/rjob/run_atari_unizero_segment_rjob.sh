#!/bin/bash

################################################################################
# UniZero Atari segment training worker/local runner
#
# Defaults:
#   - single Pong run on one GPU
#   - structured stdout logs under rjob_logs/atari_unizero_segment/<RUN_TAG>
#   - LightZero experiment/TensorBoard logs under data_unizero/rjob/<RUN_TAG>
#
# For detached cluster submission use:
#   bash zoo/atari/runs/rjob/rjob_atari_unizero_segment_1node_8gpu.sh
#
# Examples:
#   SUBMIT_RJOB=0 DRY_RUN=1 \
#   bash zoo/atari/runs/rjob/run_atari_unizero_segment_rjob.sh
#
#   SUBMIT_RJOB=0 MODE=multitask \
#   bash zoo/atari/runs/rjob/run_atari_unizero_segment_rjob.sh
#
#   SUBMIT_RJOB=0 MODE=multitask ATARI_ENVS="ALE/Pong-v5,ALE/Breakout-v5" SEEDS="0,1" \
#   MAX_PARALLEL=4 bash zoo/atari/runs/rjob/run_atari_unizero_segment_rjob.sh
################################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIGHTZERO_HOME="${LIGHTZERO_HOME:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)}"
SCRIPT_PATH="${SCRIPT_PATH:-${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")}"

MODE="${MODE:-single}"
if [[ -z "${ATARI_ENVS:-}" ]]; then
    if [[ "${MODE}" == "multitask" ]]; then
        ATARI_ENVS="ALE/Pong-v5,ALE/Breakout-v5,ALE/Qbert-v5,ALE/Seaquest-v5,ALE/SpaceInvaders-v5,ALE/MsPacman-v5,ALE/Asterix-v5,ALE/BeamRider-v5"
    else
        ATARI_ENVS="ALE/Pong-v5"
    fi
fi
SEEDS="${SEEDS:-0}"
BASELINE_VARIANTS="${BASELINE_VARIANTS:-fast_noaux}"
RUN_TAG="${RUN_TAG:-atari-unizero-segment-$(date '+%y%m%d_%H%M%S')}"
EXP_ROOT="${EXP_ROOT:-data_unizero/rjob}"
RJOB_LOG_ROOT="${RJOB_LOG_ROOT:-${LIGHTZERO_HOME}/rjob_logs/atari_unizero_segment}"
MAX_ENV_STEP="${MAX_ENV_STEP:-}"
DRY_RUN="${DRY_RUN:-0}"
TOKENIZER_PRETRAINED_VGG="${TOKENIZER_PRETRAINED_VGG:-${LIGHTZERO_HOME}/tokenizer_pretrained_vgg}"
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

RJOB_MEMORY="${RJOB_MEMORY:-300000}"
RJOB_CPU="${RJOB_CPU:-32}"
RJOB_GPU="${RJOB_GPU:-1}"
RJOB_CHARGED_GROUP="${RJOB_CHARGED_GROUP:-narmodel_gpu}"
RJOB_PRIVATE_MACHINE="${RJOB_PRIVATE_MACHINE:-group}"
RJOB_CUSTOM_RESOURCES="${RJOB_CUSTOM_RESOURCES:-brainpp.cn/fuse=1}"
RJOB_IMAGE="${RJOB_IMAGE:-registry.h.pjlab.org.cn/ailab-rlinfra-rlinfra_gpu/rft:20260408}"
RLAUNCH_BIN="${RLAUNCH_BIN:-rlaunch}"

CONDA_ENV_PATH="${CONDA_ENV_PATH:-/mnt/shared-storage-user/puyuan/conda_envs/lz}"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_ENV_PATH}/bin/python}"
CONFIG_SCRIPT="${CONFIG_SCRIPT:-${LIGHTZERO_HOME}/zoo/atari/config/atari_unizero_segment_config.py}"

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

if [[ "${INSIDE_RJOB:-0}" != "1" && "${SUBMIT_RJOB:-1}" == "1" ]]; then
    if ! command -v "${RLAUNCH_BIN}" >/dev/null 2>&1; then
        echo "rlaunch command not found: ${RLAUNCH_BIN}" >&2
        echo "Set SUBMIT_RJOB=0 to run inside the current shell." >&2
        exit 1
    fi

    exec "${RLAUNCH_BIN}" \
        --memory="${RJOB_MEMORY}" \
        --cpu="${RJOB_CPU}" \
        --gpu="${RJOB_GPU}" \
        --charged-group="${RJOB_CHARGED_GROUP}" \
        --private-machine="${RJOB_PRIVATE_MACHINE}" \
        --custom-resources "${RJOB_CUSTOM_RESOURCES}" \
        --image="${RJOB_IMAGE}" \
        --mount=gpfs://gpfs1/puyuan:/mnt/shared-storage-user/puyuan \
        --mount=gpfs://gpfs1/luyudong:/mnt/shared-storage-user/luyudong \
        --mount=gpfs://gpfs2/gpfs2-shared-public:/mnt/shared-storage-gpfs2/gpfs2-shared-public \
        --mount=gpfs://gpfs2/narmodel:/mnt/shared-storage-user/narmodel \
        -- env \
            INSIDE_RJOB=1 \
            LIGHTZERO_HOME="${LIGHTZERO_HOME}" \
            SCRIPT_PATH="${SCRIPT_PATH}" \
            MODE="${MODE}" \
            ATARI_ENVS="${ATARI_ENVS}" \
            SEEDS="${SEEDS}" \
            BASELINE_VARIANTS="${BASELINE_VARIANTS}" \
            RUN_TAG="${RUN_TAG}" \
            EXP_ROOT="${EXP_ROOT}" \
            RJOB_LOG_ROOT="${RJOB_LOG_ROOT}" \
            MAX_ENV_STEP="${MAX_ENV_STEP}" \
            DRY_RUN="${DRY_RUN}" \
            TOKENIZER_PRETRAINED_VGG="${TOKENIZER_PRETRAINED_VGG}" \
            TORCH_HOME="${TORCH_HOME:-${TOKENIZER_PRETRAINED_VGG}}" \
            USE_NEW_CACHE_MANAGER="${USE_NEW_CACHE_MANAGER}" \
            SAVE_CKPT="${SAVE_CKPT}" \
            ASYNC_PIPELINE="${ASYNC_PIPELINE}" \
            NUM_COLLECTOR_ACTORS="${NUM_COLLECTOR_ACTORS}" \
            MAX_POLICY_LAG="${MAX_POLICY_LAG}" \
            MAX_TRAIN_CHUNK_STEPS="${MAX_TRAIN_CHUNK_STEPS}" \
            WEIGHT_SYNC_INTERVAL="${WEIGHT_SYNC_INTERVAL}" \
            COLLECTOR_NUM_GPUS="${COLLECTOR_NUM_GPUS}" \
            EVALUATOR_NUM_GPUS="${EVALUATOR_NUM_GPUS}" \
            SMOKE_TEST="${SMOKE_TEST}" \
            RJOB_MEMORY="${RJOB_MEMORY}" \
            RJOB_CPU="${RJOB_CPU}" \
            RJOB_GPU="${RJOB_GPU}" \
            RJOB_CHARGED_GROUP="${RJOB_CHARGED_GROUP}" \
            RJOB_PRIVATE_MACHINE="${RJOB_PRIVATE_MACHINE}" \
            RJOB_CUSTOM_RESOURCES="${RJOB_CUSTOM_RESOURCES}" \
            RJOB_IMAGE="${RJOB_IMAGE}" \
            CONDA_ENV_PATH="${CONDA_ENV_PATH}" \
            PYTHON_BIN="${PYTHON_BIN}" \
            MAX_PARALLEL="${MAX_PARALLEL:-}" \
            bash "${SCRIPT_PATH}"
fi

RUN_DIR="${RJOB_LOG_ROOT}/${RUN_TAG}"
TASK_ROOT="${RUN_DIR}/tasks"
META_DIR="${RUN_DIR}/meta"
mkdir -p "${TASK_ROOT}" "${META_DIR}"

MAIN_LOG="${RUN_DIR}/launcher.log"
touch "${MAIN_LOG}"

log() {
    local line
    if [[ "$#" -gt 0 ]]; then
        printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*" | tee -a "${MAIN_LOG}"
    else
        while IFS= read -r line; do
            printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "${line}" | tee -a "${MAIN_LOG}"
        done
    fi
}

log_err() {
    local line
    if [[ "$#" -gt 0 ]]; then
        printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*" | tee -a "${MAIN_LOG}" >&2
    else
        while IFS= read -r line; do
            printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "${line}" | tee -a "${MAIN_LOG}" >&2
        done
    fi
}

if [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
    source /root/miniconda3/etc/profile.d/conda.sh
    if ! conda activate "${CONDA_ENV_PATH}"; then
        log_err "Warning: conda activate failed for ${CONDA_ENV_PATH}; continuing with explicit python path."
    fi
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
    log_err "Python not found or not executable: ${PYTHON_BIN}"
    exit 1
fi
if [[ ! -f "${CONFIG_SCRIPT}" ]]; then
    log_err "Config script not found: ${CONFIG_SCRIPT}"
    exit 1
fi

cd "${LIGHTZERO_HOME}"

export PATH="${CONDA_ENV_PATH}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_ENV_PATH}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${LIGHTZERO_HOME}:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${RUN_DIR}/cache}"
export TORCH_HOME="${TORCH_HOME:-${TOKENIZER_PRETRAINED_VGG}}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-garbage_collection_threshold:0.7,max_split_size_mb:256}"
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"

normalize_list() {
    echo "$1" | tr ',' ' '
}

atari_game_name() {
    local env_id="$1"
    local game_name="${env_id##*/}"
    game_name="${game_name%%-*}"
    game_name="${game_name%NoFrameskip}"
    game_name="${game_name%Deterministic}"
    echo "${game_name}"
}

baseline_args_for_variant() {
    local variant="$1"
    case "${variant}" in
        fast_noaux|fast|default)
            printf '%s\n' \
                --baseline-name "${variant}" \
                --latent-recon-loss-weight 0.0 \
                --perceptual-loss-weight 0.0 \
                --torch-compile false \
                --empty-cuda-cache-on-cache-reset false \
                --zero-init-head-names value,reward
            ;;
        compile_noaux|compile)
            printf '%s\n' \
                --baseline-name "${variant}" \
                --latent-recon-loss-weight 0.0 \
                --perceptual-loss-weight 0.0 \
                --torch-compile true \
                --empty-cuda-cache-on-cache-reset false \
                --zero-init-head-names value,reward
            ;;
        recon_only|recon)
            printf '%s\n' \
                --baseline-name "${variant}" \
                --latent-recon-loss-weight 0.1 \
                --perceptual-loss-weight 0.0 \
                --torch-compile false \
                --empty-cuda-cache-on-cache-reset false \
                --zero-init-head-names value,reward
            ;;
        recon_lpips|legacy_aux|aux)
            printf '%s\n' \
                --baseline-name "${variant}" \
                --latent-recon-loss-weight 0.1 \
                --perceptual-loss-weight 0.1 \
                --torch-compile false \
                --empty-cuda-cache-on-cache-reset false \
                --zero-init-head-names value,reward
            ;;
        legacy_full)
            printf '%s\n' \
                --baseline-name "${variant}" \
                --latent-recon-loss-weight 0.1 \
                --perceptual-loss-weight 0.1 \
                --torch-compile true \
                --empty-cuda-cache-on-cache-reset true \
                --zero-init-head-names all
            ;;
        *)
            return 1
            ;;
    esac
}

variant_requires_lpips() {
    local variant="$1"
    case "${variant}" in
        recon_lpips|legacy_aux|aux|legacy_full)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

read -r -a ENV_LIST <<< "$(normalize_list "${ATARI_ENVS}")"
read -r -a SEED_LIST <<< "$(normalize_list "${SEEDS}")"
read -r -a VARIANT_LIST <<< "$(normalize_list "${BASELINE_VARIANTS}")"

for variant in "${VARIANT_LIST[@]}"; do
    if variant_requires_lpips "${variant}"; then
        if [[ ! -f "${TORCH_HOME}/vgg.pth" ]]; then
            log_err "LPIPS variant '${variant}' requires ${TORCH_HOME}/vgg.pth."
            exit 1
        fi
        if [[ ! -f "${TORCH_HOME}/hub/checkpoints/vgg16-397923af.pth" ]]; then
            log_err "LPIPS variant '${variant}' requires ${TORCH_HOME}/hub/checkpoints/vgg16-397923af.pth."
            exit 1
        fi
    fi
done

if [[ "${MODE}" != "single" && "${MODE}" != "multitask" ]]; then
    log_err "Unsupported MODE=${MODE}. Expected single or multitask."
    exit 1
fi

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a GPU_LIST <<< "${CUDA_VISIBLE_DEVICES}"
else
    GPU_LIST=()
    for ((gpu_idx = 0; gpu_idx < RJOB_GPU; gpu_idx++)); do
        GPU_LIST+=("${gpu_idx}")
    done
fi
GPU_COUNT="${#GPU_LIST[@]}"
if [[ "${GPU_COUNT}" -eq 0 ]]; then
    log_err "No GPUs available from CUDA_VISIBLE_DEVICES/RJOB_GPU."
    exit 1
fi

MAX_PARALLEL="${MAX_PARALLEL:-${GPU_COUNT}}"
if [[ "${MAX_PARALLEL}" -lt 1 ]]; then
    log_err "MAX_PARALLEL must be >= 1, got ${MAX_PARALLEL}"
    exit 1
fi

if [[ "${EXP_ROOT}" = /* ]]; then
    TB_ROOT="${EXP_ROOT}/${RUN_TAG}"
else
    TB_ROOT="${LIGHTZERO_HOME}/${EXP_ROOT}/${RUN_TAG}"
fi

cat > "${META_DIR}/run.env" <<EOF
RUN_TAG=${RUN_TAG}
MODE=${MODE}
ATARI_ENVS=${ATARI_ENVS}
SEEDS=${SEEDS}
BASELINE_VARIANTS=${BASELINE_VARIANTS}
EXP_ROOT=${EXP_ROOT}
RJOB_LOG_ROOT=${RJOB_LOG_ROOT}
MAX_ENV_STEP=${MAX_ENV_STEP}
DRY_RUN=${DRY_RUN}
LIGHTZERO_HOME=${LIGHTZERO_HOME}
CONFIG_SCRIPT=${CONFIG_SCRIPT}
CONDA_ENV_PATH=${CONDA_ENV_PATH}
PYTHON_BIN=${PYTHON_BIN}
TOKENIZER_PRETRAINED_VGG=${TOKENIZER_PRETRAINED_VGG}
TORCH_HOME=${TORCH_HOME}
USE_NEW_CACHE_MANAGER=${USE_NEW_CACHE_MANAGER}
SAVE_CKPT=${SAVE_CKPT}
ASYNC_PIPELINE=${ASYNC_PIPELINE}
NUM_COLLECTOR_ACTORS=${NUM_COLLECTOR_ACTORS}
MAX_POLICY_LAG=${MAX_POLICY_LAG}
MAX_TRAIN_CHUNK_STEPS=${MAX_TRAIN_CHUNK_STEPS}
WEIGHT_SYNC_INTERVAL=${WEIGHT_SYNC_INTERVAL}
COLLECTOR_NUM_GPUS=${COLLECTOR_NUM_GPUS}
EVALUATOR_NUM_GPUS=${EVALUATOR_NUM_GPUS}
SMOKE_TEST=${SMOKE_TEST}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}
GPU_LIST=${GPU_LIST[*]}
MAX_PARALLEL=${MAX_PARALLEL}
RJOB_MEMORY=${RJOB_MEMORY}
RJOB_CPU=${RJOB_CPU}
RJOB_GPU=${RJOB_GPU}
RJOB_CHARGED_GROUP=${RJOB_CHARGED_GROUP}
RJOB_IMAGE=${RJOB_IMAGE}
EOF

cp "${SCRIPT_PATH}" "${META_DIR}/run_script.sh" 2>/dev/null || true
cp "${CONFIG_SCRIPT}" "${META_DIR}/atari_unizero_segment_config.py" 2>/dev/null || true
git --git-dir="${LIGHTZERO_HOME}/.git" --work-tree="${LIGHTZERO_HOME}" rev-parse HEAD > "${META_DIR}/git_commit.txt" 2>/dev/null || true
git --git-dir="${LIGHTZERO_HOME}/.git" --work-tree="${LIGHTZERO_HOME}" status --short > "${META_DIR}/git_status.txt" 2>/dev/null || true

log "=== UniZero Atari rjob launcher ==="
log "RUN_TAG:       ${RUN_TAG}"
log "MODE:          ${MODE}"
log "ATARI_ENVS:    ${ATARI_ENVS}"
log "SEEDS:         ${SEEDS}"
log "VARIANTS:      ${BASELINE_VARIANTS}"
log "GPU_LIST:      ${GPU_LIST[*]}"
log "MAX_PARALLEL:  ${MAX_PARALLEL}"
log "LIGHTZERO_HOME:${LIGHTZERO_HOME}"
log "CONFIG_SCRIPT: ${CONFIG_SCRIPT}"
log "PYTHON_BIN:    ${PYTHON_BIN}"
log "RUN_DIR:       ${RUN_DIR}"
log "EXP_ROOT:      ${EXP_ROOT}"
log "TB_ROOT:       ${TB_ROOT}"
log "TORCH_HOME:    ${TORCH_HOME}"
log "NEW_CACHE:     ${USE_NEW_CACHE_MANAGER}"
log "SAVE_CKPT:     ${SAVE_CKPT}"
log "ASYNC_PIPELINE:${ASYNC_PIPELINE}"
log "SMOKE_TEST:    ${SMOKE_TEST}"
log "DRY_RUN:       ${DRY_RUN}"

if [[ "${DRY_RUN}" == "1" ]]; then
    log "=== Dry-run task plan ==="
    plan_index=0
    for env_id in "${ENV_LIST[@]}"; do
        for seed in "${SEED_LIST[@]}"; do
            for variant in "${VARIANT_LIST[@]}"; do
                if ! baseline_args_for_variant "${variant}" >/dev/null; then
                    log_err "Unknown BASELINE_VARIANTS entry: ${variant}"
                    exit 1
                fi
                gpu_id="${GPU_LIST[$((plan_index % GPU_COUNT))]}"
                game_name="$(atari_game_name "${env_id}")"
                log "Task ${plan_index}: env=${env_id} seed=${seed} variant=${variant} gpu=${gpu_id} log=${TASK_ROOT}/${game_name}/seed${seed}/${variant}/train.log"
                plan_index=$((plan_index + 1))
            done
        done
    done
    log "Task count: ${plan_index}"
    log "Run logs:   ${RUN_DIR}"
    log "TB root:    ${TB_ROOT}"
    log "Try:        tensorboard --logdir ${TB_ROOT}"
    exit 0
fi

"${PYTHON_BIN}" - <<'PY_PREFLIGHT' 2>&1 | log
import sys
import torch
import easydict
import ding
print(f"Python preflight ok: {sys.executable}")
print(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()} cuda_count={torch.cuda.device_count()}")
print(f"ding={getattr(ding, '__version__', 'unknown')}")
PY_PREFLIGHT

launch_task() {
    local task_index="$1"
    local variant="$2"
    local env_id="$3"
    local seed="$4"
    local gpu_id="$5"
    local game_name
    game_name="$(atari_game_name "${env_id}")"
    local task_name="${game_name}_${variant}_seed${seed}"
    local task_dir="${TASK_ROOT}/${game_name}/seed${seed}/${variant}"
    local task_log="${task_dir}/train.log"
    local variant_args=()
    mkdir -p "${task_dir}"

    if ! mapfile -t variant_args < <(baseline_args_for_variant "${variant}"); then
        log_err "Unknown BASELINE_VARIANTS entry: ${variant}"
        exit 1
    fi

    log "Launching task ${task_index}: env=${env_id} seed=${seed} variant=${variant} gpu=${gpu_id} log=${task_log}"

    (
        set -euo pipefail
        export CUDA_VISIBLE_DEVICES="${gpu_id}"
        export LIGHTZERO_RJOB_RUN_TAG="${RUN_TAG}"
        export LIGHTZERO_RJOB_TASK_INDEX="${task_index}"
        export LIGHTZERO_RJOB_TASK_NAME="${task_name}"

        echo "=== Task ${task_index}: ${task_name} ==="
        echo "env_id=${env_id}"
        echo "seed=${seed}"
        echo "variant=${variant}"
        echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
        echo "task_dir=${task_dir}"
        echo "task_log=${task_log}"

        cmd=(
            "${PYTHON_BIN}" -u "${CONFIG_SCRIPT}"
            --env "${env_id}"
            --seed "${seed}"
            --exp-root "${EXP_ROOT}"
            --run-tag "${RUN_TAG}"
            --use-new-cache-manager "${USE_NEW_CACHE_MANAGER}"
            --save-ckpt "${SAVE_CKPT}"
            "${variant_args[@]}"
        )
        if [[ "${ASYNC_PIPELINE}" == "1" ]]; then
            cmd+=(
                --async-pipeline
                --num-collector-actors "${NUM_COLLECTOR_ACTORS}"
                --max-policy-lag "${MAX_POLICY_LAG}"
                --max-train-chunk-steps "${MAX_TRAIN_CHUNK_STEPS}"
                --weight-sync-interval "${WEIGHT_SYNC_INTERVAL}"
                --collector-num-gpus "${COLLECTOR_NUM_GPUS}"
                --evaluator-num-gpus "${EVALUATOR_NUM_GPUS}"
            )
        fi
        if [[ "${SMOKE_TEST}" == "1" ]]; then
            cmd+=(--smoke-test)
        fi
        if [[ -n "${MAX_ENV_STEP}" ]]; then
            cmd+=(--max-env-step "${MAX_ENV_STEP}")
        fi

        echo "Command: ${cmd[*]}"
        "${cmd[@]}"
    ) > >(awk -v task="${task_name}" '{ print strftime("[%Y-%m-%d %H:%M:%S]"), "[" task "]", $0; fflush() }' | tee -a "${task_log}") 2>&1 &
    task_pid="$!"
    TASK_PIDS+=("${task_pid}")
    TASK_NAMES+=("${task_name}")
    TASK_LOGS+=("${task_log}")
}

task_index=0
running=0
failed=0
TASK_PIDS=()
TASK_NAMES=()
TASK_LOGS=()

record_finished_task() {
    local pid="$1"
    local status="$2"
    local idx
    for idx in "${!TASK_PIDS[@]}"; do
        if [[ "${TASK_PIDS[$idx]}" == "${pid}" ]]; then
            log "Task finished: ${TASK_NAMES[$idx]} pid=${pid} exit_code=${status} log=${TASK_LOGS[$idx]}"
            unset "TASK_PIDS[$idx]"
            unset "TASK_NAMES[$idx]"
            unset "TASK_LOGS[$idx]"
            break
        fi
    done
    if [[ "${status}" -ne 0 ]]; then
        failed=1
    fi
}

wait_for_one_task() {
    local finished_pid
    local status
    set +e
    wait -n -p finished_pid "${TASK_PIDS[@]}"
    status="$?"
    set -e
    if [[ -n "${finished_pid:-}" ]]; then
        record_finished_task "${finished_pid}" "${status}"
    else
        log_err "wait -n returned status=${status} without a finished pid."
        if [[ "${status}" -ne 0 ]]; then
            failed=1
        fi
    fi
}

for env_id in "${ENV_LIST[@]}"; do
    for seed in "${SEED_LIST[@]}"; do
        for variant in "${VARIANT_LIST[@]}"; do
            gpu_id="${GPU_LIST[$((task_index % GPU_COUNT))]}"
            launch_task "${task_index}" "${variant}" "${env_id}" "${seed}" "${gpu_id}"
            task_index=$((task_index + 1))
            running=$((running + 1))

            if [[ "${running}" -ge "${MAX_PARALLEL}" ]]; then
                wait_for_one_task
                running=$((running - 1))
            fi
        done
    done
done

while [[ "${running}" -gt 0 ]]; do
    wait_for_one_task
    running=$((running - 1))
done

log "=== Run complete ==="
log "Task count: ${task_index}"
log "Run logs:   ${RUN_DIR}"
log "TB root:    ${TB_ROOT}"
log "Try:        tensorboard --logdir ${TB_ROOT}"

if [[ "${failed}" -ne 0 ]]; then
    log_err "At least one task failed. See per-task train.log files under ${TASK_ROOT}."
    exit 1
fi

#!/usr/bin/env bash
set -euo pipefail

REPO=${UZ_REPO:-/mnt/shared-storage-user/puyuan/code/LightZero}
PYTHON=${UZ_PYTHON:-/mnt/shared-storage-user/puyuan/conda_envs/lz/bin/python}
OUTPUT_ROOT=${UZ_OUTPUT_ROOT:-/mnt/shared-storage-gpfs2/trustcyberdata/private/docker-infra/tmp/puyuan/rl/lightzero/unizero_mspacman_3m_20260824}
JOB_NAME=${UZ_JOB_NAME:-uz-mspacman-v3-3m}
PREEMPTIBLE=${UZ_PREEMPTIBLE:-no}
PRIVATE_MACHINE=${UZ_PRIVATE_MACHINE:-group}
RJOB_DRY_RUN=${UZ_RJOB_DRY_RUN:-false}
MODE=${1:-submit}

# The 2026-08 matrix concluded: v3 (historical best recipe + value_loss_weight=0.5) is the
# canonical arm. The module defaults are the baseline arm, so run_seed passes the three v3
# feature flags explicitly. Seeds are assigned to GPUs in batches of four.
# Default: seed 0 only. To queue replication seeds:
# UZ_SEED_QUEUE=0,1,2 bash scripts/launch_unizero_mspacman_3M.sh worker
SEED_QUEUE=${UZ_SEED_QUEUE:-0}

MODULE=zoo.atari.config.atari_unizero_segment_experimental_config
GPUS_PER_BATCH=4

find_checkpoint() {
  "$PYTHON" "$REPO/scripts/find_valid_unizero_checkpoint.py" "$1" 2>/dev/null || true
}

monitor_startup_health() {
  local train_pid=$1
  local log=$2
  local run_dir=$3
  local seed=$4
  local health_log="$OUTPUT_ROOT/controller/health_unizero_mspacman_v3_seed${seed}_3M.log"

  while kill -0 "$train_pid" 2>/dev/null; do
    sleep 60
    local envstep='not-yet-reported'
    local eval_return='not-yet-reported'
    if [ -f "$log" ]; then
      envstep=$(grep -E 'total_envstep_count:[[:space:]]*[0-9]+' "$log" | tail -1 | awk '{print $2}' || true)
      eval_return=$(grep -E 'eval_episode_return_mean:[[:space:]]*[-+0-9.eE]+' "$log" | tail -1 | awk '{print $2}' || true)
      envstep=${envstep:-not-yet-reported}
      eval_return=${eval_return:-not-yet-reported}
      if tail -400 "$log" | grep -Eq 'CUDA out of memory|Traceback \(most recent call last\)|(^|[^[:alpha:]])NaN([^[:alpha:]]|$)|non-finite'; then
        printf '[%s] HEALTH_ALERT seed=%s envstep=%s eval=%s\n' \
          "$(date -Is)" "$seed" "$envstep" "$eval_return" >>"$health_log"
      else
        printf '[%s] HEALTH_OK seed=%s envstep=%s eval=%s\n' \
          "$(date -Is)" "$seed" "$envstep" "$eval_return" >>"$health_log"
      fi
    fi
    if [[ "$envstep" =~ ^[0-9]+$ ]] && [ "$envstep" -ge 50000 ]; then
      touch "$run_dir/.startup_health_50k"
      printf '[%s] STARTUP_HEALTH_COMPLETE seed=%s envstep=%s\n' \
        "$(date -Is)" "$seed" "$envstep" >>"$health_log"
      return 0
    fi
  done
}

run_seed() {
  local gpu=$1
  local seed=$2
  local run_name="unizero_mspacman_v3_seed${seed}_3M"
  local run_dir="$OUTPUT_ROOT/$run_name"
  local log_dir="$OUTPUT_ROOT/controller"
  local log="$log_dir/gpu${gpu}_${run_name}.log"
  local attempt=0
  local emergency_batch=''

  if [ -f "$run_dir/.completed" ]; then
    printf '[%s] SKIP completed run=%s\n' "$(date -Is)" "$run_name" >>"$log"
    return 0
  fi

  while [ "$attempt" -lt 3 ]; do
    attempt=$((attempt + 1))
    local checkpoint
    checkpoint=$(find_checkpoint "$run_dir")
    local resume_args=()
    if [ -n "$checkpoint" ]; then
      resume_args=(--resume-from "$checkpoint" --resume-in-place)
    elif [ -d "$run_dir" ]; then
      local failed_dir="${run_dir}.failed-attempt${attempt}-$(date +%Y%m%d_%H%M%S)"
      printf '[%s] preserving non-resumable run at %s\n' "$(date -Is)" "$failed_dir" >>"$log"
      mv "$run_dir" "$failed_dir"
    fi
    local batch_args=()
    if [ -n "$emergency_batch" ]; then
      batch_args=(--batch-size "$emergency_batch")
    fi
    printf '[%s] gpu=%s seed=%s attempt=%s checkpoint=%s batch_override=%s\n' \
      "$(date -Is)" "$gpu" "$seed" "$attempt" "${checkpoint:-fresh}" \
      "${emergency_batch:-none}" >>"$log"
    set +e
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -m "$MODULE" \
      --env ALE/MsPacman-v5 --seed "$seed" \
      --output-root "$OUTPUT_ROOT" --run-name "$run_name" \
      --rebuild-kv-window-from-tokens --bootstrap-value-context \
      --open-loop-consistency-weight 1.0 \
      "${resume_args[@]}" "${batch_args[@]}" >>"$log" 2>&1 &
    local train_pid=$!
    monitor_startup_health "$train_pid" "$log" "$run_dir" "$seed" &
    local health_pid=$!
    wait "$train_pid"
    local status=$?
    kill "$health_pid" 2>/dev/null || true
    wait "$health_pid" 2>/dev/null || true
    set -e
    if [ "$status" -eq 0 ]; then
      touch "$run_dir/.completed"
      printf '[%s] COMPLETE seed=%s\n' "$(date -Is)" "$seed" >>"$log"
      return 0
    fi
    if tail -400 "$log" | grep -Eqi 'CUDA out of memory|out of memory.*CUDA'; then
      emergency_batch=128
      printf '[%s] OOM detected; retrying with explicit emergency batch_size=128\n' \
        "$(date -Is)" >>"$log"
    fi
    printf '[%s] FAILED status=%s; retry %s/2 follows in 15s\n' \
      "$(date -Is)" "$status" "$attempt" >>"$log"
    sleep 15
  done
  printf '[%s] TERMINAL_FAILED seed=%s after 3 attempts\n' \
    "$(date -Is)" "$seed" >>"$log"
  return 1
}

worker() {
  mkdir -p "$OUTPUT_ROOT/controller"
  cd "$REPO"
  IFS=',' read -r -a seeds <<<"$SEED_QUEUE"
  local overall=0
  local i gpu
  local batch_pids=()
  for i in "${!seeds[@]}"; do
    gpu=$((i % GPUS_PER_BATCH))
    run_seed "$gpu" "${seeds[$i]}" &
    batch_pids+=($!)
    if [ "$gpu" -eq $((GPUS_PER_BATCH - 1)) ] || [ "$i" -eq $((${#seeds[@]} - 1)) ]; then
      for pid in "${batch_pids[@]}"; do
        wait "$pid" || overall=1
      done
      batch_pids=()
    fi
  done
  return "$overall"
}

submit() {
  local private_args=()
  if [ "$PRIVATE_MACHINE" != public ]; then
    private_args=(--private-machine="$PRIVATE_MACHINE")
  fi
  rjob submit \
    --name="$JOB_NAME" \
    --dry-run="$RJOB_DRY_RUN" \
    --gpu=4 --memory=300000 --cpu=32 \
    --charged-group=narmodel_gpu \
    --preemptible="$PREEMPTIBLE" \
    "${private_args[@]}" -P 1 \
    --auto-restart=true \
    --image=registry.h.pjlab.org.cn/ailab-rlinfra-rlinfra_gpu/rft:20260408 \
    --mount=gpfs://gpfs1/puyuan:/mnt/shared-storage-user/puyuan \
    --mount=gpfs://gpfs1/luyudong:/mnt/shared-storage-user/luyudong \
    --mount=gpfs://gpfs2/gpfs2-shared-public:/mnt/shared-storage-gpfs2/gpfs2-shared-public \
    --mount=gpfs://gpfs2/trustcyberdata:/mnt/shared-storage-gpfs2/trustcyberdata \
    --mount=gpfs://gpfs2/narmodel:/mnt/shared-storage-user/narmodel \
    --custom-resources brainpp.cn/fuse=1 \
    -- bash "$REPO/scripts/launch_unizero_mspacman_3M.sh" worker
}

submit_multitask() {
  local private_args=()
  if [ "$PRIVATE_MACHINE" != public ]; then
    private_args=(--private-machine="$PRIVATE_MACHINE")
  fi
  local common=(
    --name="$JOB_NAME"
    --dry-run="$RJOB_DRY_RUN"
    --gpu=1 --memory=80000 --cpu=8
    --charged-group=narmodel_gpu
    --preemptible="$PREEMPTIBLE"
    --gang-start=false
    "${private_args[@]}"
    --auto-restart=true
    --image=registry.h.pjlab.org.cn/ailab-rlinfra-rlinfra_gpu/rft:20260408
    --mount=gpfs://gpfs1/puyuan:/mnt/shared-storage-user/puyuan
    --mount=gpfs://gpfs1/luyudong:/mnt/shared-storage-user/luyudong
    --mount=gpfs://gpfs2/gpfs2-shared-public:/mnt/shared-storage-gpfs2/gpfs2-shared-public
    --mount=gpfs://gpfs2/trustcyberdata:/mnt/shared-storage-gpfs2/trustcyberdata
    --mount=gpfs://gpfs2/narmodel:/mnt/shared-storage-user/narmodel
    --custom-resources brainpp.cn/fuse=1
  )
  rjob submit "${common[@]}" \
    --- --task_name=seed0 -- bash "$REPO/scripts/launch_unizero_mspacman_3M.sh" single 0 \
    --- --task_name=seed1 -- bash "$REPO/scripts/launch_unizero_mspacman_3M.sh" single 1 \
    --- --task_name=seed2 -- bash "$REPO/scripts/launch_unizero_mspacman_3M.sh" single 2
}

submit_sharded() {
  local private_args=()
  if [ "$PRIVATE_MACHINE" != public ]; then
    private_args=(--private-machine="$PRIVATE_MACHINE")
  fi
  local overall=0
  for seed in 0 1 2; do
    local task_job="${JOB_NAME}-seed${seed}"
    local existing
    existing=$(rjob get "$task_job" 2>/dev/null || true)
    if [ -n "$existing" ]; then
      printf 'SKIP existing rjob=%s\n' "$task_job"
      continue
    fi
    rjob submit \
      --name="$task_job" \
      --dry-run="$RJOB_DRY_RUN" \
      --gpu=1 --memory=80000 --cpu=8 \
      --charged-group=narmodel_gpu \
      --preemptible="$PREEMPTIBLE" \
      "${private_args[@]}" -P 1 \
      --auto-restart=true \
      --image=registry.h.pjlab.org.cn/ailab-rlinfra-rlinfra_gpu/rft:20260408 \
      --mount=gpfs://gpfs1/puyuan:/mnt/shared-storage-user/puyuan \
      --mount=gpfs://gpfs1/luyudong:/mnt/shared-storage-user/luyudong \
      --mount=gpfs://gpfs2/gpfs2-shared-public:/mnt/shared-storage-gpfs2/gpfs2-shared-public \
      --mount=gpfs://gpfs2/trustcyberdata:/mnt/shared-storage-gpfs2/trustcyberdata \
      --mount=gpfs://gpfs2/narmodel:/mnt/shared-storage-user/narmodel \
      --custom-resources brainpp.cn/fuse=1 \
      -- bash "$REPO/scripts/launch_unizero_mspacman_3M.sh" single "$seed" || overall=1
  done
  return "$overall"
}

case "$MODE" in
  submit) submit ;;
  submit-multitask) submit_multitask ;;
  submit-sharded) submit_sharded ;;
  worker) worker ;;
  single)
    requested_seed=${2:?single mode requires SEED}
    case "$requested_seed" in
      ''|*[!0-9]*) printf 'single mode SEED must be an integer: %s\n' "$requested_seed" >&2; exit 2 ;;
    esac
    mkdir -p "$OUTPUT_ROOT/controller"
    cd "$REPO"
    run_seed 0 "$requested_seed"
    ;;
  print-matrix)
    IFS=',' read -r -a seeds <<<"$SEED_QUEUE"
    for i in "${!seeds[@]}"; do
      printf 'gpu=%s seed=%s module=%s\n' \
        "$((i % GPUS_PER_BATCH))" "${seeds[$i]}" "$MODULE"
    done
    ;;
  check-kill)
    exec "$PYTHON" "$REPO/scripts/check_unizero_mspacman_3m.py" "$OUTPUT_ROOT" \
      --seed "${UZ_CHECK_SEED:-0}" --margin "${UZ_KILL_MARGIN:-0.35}"
    ;;
  *) printf 'usage: %s [submit|submit-multitask|submit-sharded|worker|single|print-matrix|check-kill]\n' "$0" >&2; exit 2 ;;
esac

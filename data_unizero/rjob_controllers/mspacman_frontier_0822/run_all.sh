#!/usr/bin/env bash
set -euo pipefail

REPO=/mnt/shared-storage-user/puyuan/code/LightZero
PYTHON=/mnt/shared-storage-user/puyuan/conda_envs/lz/bin/python
CTRL="$REPO/data_unizero/rjob_controllers/mspacman_frontier_0822"
OUTPUT_BASE=${UZ_OUTPUT_BASE:-/mnt/shared-storage-gpfs2/trustcyberdata/private/docker-infra/tmp/puyuan/rl/lightzero}
EXPERIMENT_ID=${UZ_EXPERIMENT_ID:-mspacman_frontier_0823_v4_nopong}
RUNROOT=${UZ_RUN_ROOT:-"$OUTPUT_BASE/$EXPERIMENT_ID"}
LOGROOT="$RUNROOT/controller"
mkdir -p "$LOGROOT"
export UZ_RUN_ROOT="$RUNROOT"

available_kb=$(df -Pk "$RUNROOT" | awk 'NR == 2 {print $4}')
if [ -z "$available_kb" ] || [ "$available_kb" -lt 10485760 ]; then
  printf 'Output filesystem requires at least 10 GiB free: path=%s available_kb=%s\n' \
    "$RUNROOT" "${available_kb:-unknown}" >&2
  exit 3
fi

NAMES=(uz_h5_value05 uz_h10_value05 uz_h5_value025 uz_h5_reanalysis02_rbs32)
ARGS=(
  "--num-unroll-steps 5 --value-loss-weight 0.5"
  "--num-unroll-steps 10 --value-loss-weight 0.5"
  "--num-unroll-steps 5 --value-loss-weight 0.25"
  "--num-unroll-steps 5 --value-loss-weight 0.5 --buffer-reanalyze-freq 0.02 --reanalyze-batch-size 32 --contextual-reanalysis"
)
PIDS=()

cleanup() {
  trap - TERM INT EXIT
  for pid in "${PIDS[@]:-}"; do pkill -TERM -P "$pid" 2>/dev/null || true; done
  for pid in "${PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done
  wait 2>/dev/null || true
}
trap cleanup TERM INT EXIT

cd "$REPO"

for gpu in 0 1 2 3; do
  name=${NAMES[$gpu]}
  log="$LOGROOT/gpu${gpu}_${name}.log"
  (
    attempt=0
    while true; do
      attempt=$((attempt + 1))
      printf '\n[%s] gpu=%s variant=%s attempt=%s start\n' "$(date -Is)" "$gpu" "$name" "$attempt" >>"$log"
      run_dir="$RUNROOT/${name}-seed0-3m"
      resume_args=()
      if [ -d "$run_dir" ]; then
        checkpoint=''
        checkpoint=$("$PYTHON" "$CTRL/find_valid_checkpoint.py" "$run_dir" 2>>"$log" || true)
        if [ -z "$checkpoint" ]; then
          printf '[%s] variant=%s cannot resume: run directory exists without a complete checkpoint\n' \
            "$(date -Is)" "$name" >>"$log"
          exit 2
        fi
        resume_args=(--resume-from "$checkpoint" --resume-in-place)
        printf '[%s] variant=%s resume=%s\n' "$(date -Is)" "$name" "$checkpoint" >>"$log"
      fi
      # shellcheck disable=SC2086
      if CUDA_VISIBLE_DEVICES="$gpu" bash "$CTRL/run_variant.sh" "$name" ${ARGS[$gpu]} \
          "${resume_args[@]}" >>"$log" 2>&1; then
        rc=0
      else
        rc=$?
      fi
      printf '[%s] gpu=%s variant=%s exit=%s\n' "$(date -Is)" "$gpu" "$name" "$rc" >>"$log"
      [ "$rc" -eq 0 ] && break
      sleep 15
    done
  ) &
  PIDS[$gpu]=$!
done

status=0
for pid in "${PIDS[@]}"; do wait "$pid" || status=1; done
exit "$status"

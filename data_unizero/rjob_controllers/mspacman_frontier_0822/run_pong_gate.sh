#!/usr/bin/env bash
set -euo pipefail

REPO=/mnt/shared-storage-user/puyuan/code/LightZero
PYTHON=/mnt/shared-storage-user/puyuan/conda_envs/lz/bin/python
OUTPUT_ROOT=${UZ_RUN_ROOT:-/mnt/shared-storage-gpfs2/trustcyberdata/private/docker-infra/tmp/puyuan/rl/lightzero/mspacman_frontier_0823_v3}
RUN_NAME=pong_known_good_seed0_250k
RUN_DIR="$REPO/$OUTPUT_ROOT/$RUN_NAME"
RESUME_ARGS=()

if [ -d "$RUN_DIR" ]; then
  checkpoint=$("$PYTHON" "$REPO/data_unizero/rjob_controllers/mspacman_frontier_0822/find_valid_checkpoint.py" \
    "$RUN_DIR" || true)
  if [ -z "$checkpoint" ]; then
    echo "Pong gate directory exists without a recoverable checkpoint: $RUN_DIR" >&2
    exit 2
  fi
  RESUME_ARGS=(--resume-from "$checkpoint" --resume-in-place)
fi

cd "$REPO"
exec "$PYTHON" zoo/atari/config/atari_unizero_segment_experimental_config.py \
  --env ALE/Pong-v5 \
  --seed 0 \
  --output-root "$OUTPUT_ROOT" \
  --run-name "$RUN_NAME" \
  --max-env-step 250000 \
  --evaluator-env-num 8 \
  --collect-num-simulations 25 \
  --collect-temperature 0.25 \
  --game-segment-length 20 \
  --num-unroll-steps 10 \
  --infer-context-length 4 \
  --enable-encoder-clip \
  --stab-fix \
  --use-adaptive-alpha \
  --use-priority \
  --no-augmentation \
  --obs-loss-weight 10 \
  --value-loss-weight 0.25 \
  --grad-clip-value 5 \
  --replay-buffer-size 500000 \
  --no-empty-cuda-cache-on-cache-reset \
  --save-ckpt-after-iter 10000 \
  --periodic-ckpt-keep-last 1 \
  --no-save-ckpt-in-eval \
  --ignore-checkpoint-save-errors \
  "${RESUME_ARGS[@]}"

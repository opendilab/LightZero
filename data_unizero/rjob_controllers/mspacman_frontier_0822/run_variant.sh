#!/usr/bin/env bash
set -euo pipefail

REPO=/mnt/shared-storage-user/puyuan/code/LightZero
PYTHON=/mnt/shared-storage-user/puyuan/conda_envs/lz/bin/python
VARIANT=${1:?variant name required}
OUTPUT_ROOT=${UZ_RUN_ROOT:-/mnt/shared-storage-gpfs2/trustcyberdata/private/docker-infra/tmp/puyuan/rl/lightzero/mspacman_frontier_0823_v4_nopong}
shift

cd "$REPO"
exec "$PYTHON" zoo/atari/config/atari_unizero_segment_experimental_config.py \
  --env ALE/MsPacman-v5 \
  --seed 0 \
  --output-root "$OUTPUT_ROOT" \
  --run-name "${VARIANT}/seed0-3m" \
  --max-env-step 3000000 \
  --collect-num-simulations 25 \
  --collect-temperature 0.25 \
  --game-segment-length 200 \
  --use-new-cache-manager \
  --root-cache-key-round-decimals 4 \
  --kv-cache-clear-interval 2000 \
  --no-empty-cuda-cache-on-cache-reset \
  --use-priority \
  --use-augmentation \
  --stab-fix \
  --bootstrap-value-context \
  --rebuild-kv-window-from-tokens \
  --obs-loss-weight 10 \
  --replay-ratio 0.1 \
  --batch-size 256 \
  --save-ckpt-after-iter 50000 \
  --periodic-ckpt-keep-last 1 \
  --no-save-ckpt-in-eval \
  --ignore-checkpoint-save-errors \
  "$@"

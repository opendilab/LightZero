#!/usr/bin/env bash
set -euo pipefail

rjob submit \
  --name=uz-mspacman-frontier-0823-gpfs-nopong-c32 \
  --gpu=4 --memory=300000 --cpu=32 \
  --charged-group=narmodel_gpu \
  --private-machine=group -P 1 \
  --auto-restart=true \
  --image=registry.h.pjlab.org.cn/ailab-rlinfra-rlinfra_gpu/rft:20260408 \
  --mount=gpfs://gpfs1/puyuan:/mnt/shared-storage-user/puyuan \
  --mount=gpfs://gpfs1/luyudong:/mnt/shared-storage-user/luyudong \
  --mount=gpfs://gpfs2/gpfs2-shared-public:/mnt/shared-storage-gpfs2/gpfs2-shared-public \
  --mount=gpfs://gpfs2/trustcyberdata:/mnt/shared-storage-gpfs2/trustcyberdata \
  --mount=gpfs://gpfs2/narmodel:/mnt/shared-storage-user/narmodel \
  --custom-resources brainpp.cn/fuse=1 \
  -- bash /mnt/shared-storage-user/puyuan/code/LightZero/data_unizero/rjob_controllers/mspacman_frontier_0822/run_all.sh

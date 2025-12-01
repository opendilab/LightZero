#!/bin/bash

# Batch script to generate loss landscape for checkpoints from 10K to 100K
# Usage: bash run_loss_landscape_batch.sh

# Base checkpoint directory
CKPT_BASE_DIR="/mnt/shared-storage-user/tangjia/unizero/LightZero/mnt/shared-storage-user/tangjia/rftinfra/tangjia/ckpt/MsPacman_uz_brf0.02-rbs160-rp0.75_nlayer2_numsegments-8_gsl20_rr0.25_Htrain10-Hinfer4_bs64_seed0/ckpt"

# Config script path
CONFIG_SCRIPT="/mnt/shared-storage-user/tangjia/unizero/LightZero/zoo/atari/config/atari_unizero_loss_landscape.py"

# Base log directory
BASE_LOG_DIR="data_lz/loss_landscape_batch_mspacman"

# Environment and seed settings
ENV_ID="MsPacmanNoFrameskip-v4"
SEED=0

# Create base log directory if it doesn't exist
mkdir -p ${BASE_LOG_DIR}

# Loop through iterations from 10K to 100K (step of 10K)
for iter in 10000 20000 30000 40000 50000 60000 70000 80000 90000 100000
do
    echo "=================================================="
    echo "Processing checkpoint: iteration_${iter}.pth.tar"
    echo "=================================================="

    # Set checkpoint path
    CKPT_PATH="${CKPT_BASE_DIR}/iteration_${iter}.pth.tar"

    # Check if checkpoint exists
    if [ ! -f "${CKPT_PATH}" ]; then
        echo "WARNING: Checkpoint ${CKPT_PATH} does not exist, skipping..."
        continue
    fi

    # Set log directory for this iteration
    LOG_DIR="${BASE_LOG_DIR}/iteration_${iter}"

    echo "Checkpoint: ${CKPT_PATH}"
    echo "Log directory: ${LOG_DIR}"
    echo ""

    # Run the loss landscape script
    python ${CONFIG_SCRIPT} \
        --env ${ENV_ID} \
        --seed ${SEED} \
        --ckpt ${CKPT_PATH} \
        --log_dir ${LOG_DIR}

    # Check if the command succeeded
    if [ $? -eq 0 ]; then
        echo "SUCCESS: Finished processing iteration ${iter}"
    else
        echo "ERROR: Failed to process iteration ${iter}"
    fi

    echo ""
done

echo "=================================================="
echo "All checkpoints processed!"
echo "Results saved in: ${BASE_LOG_DIR}"
echo "=================================================="

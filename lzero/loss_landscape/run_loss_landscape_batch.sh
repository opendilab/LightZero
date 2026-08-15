#!/bin/bash

# Batch script to generate loss landscape for checkpoints from 10K to 100K
# Usage: bash run_loss_landscape_batch.sh
#
# This script performs batch evaluation of loss landscapes for UniZero models.
# It iterates through multiple checkpoint iterations and generates:
# - HDF5 data files with loss landscape values
# - Visualization images (contour, filled contour, heatmap, 3D surface)
#
# Expected input directory structure:
# CKPT_BASE_DIR/
# ├── iteration_10000.pth.tar
# ├── iteration_20000.pth.tar
# ├── iteration_30000.pth.tar
# ├── ...
# └── iteration_100000.pth.tar
#
# Expected output directory structure:
# BASE_LOG_DIR/
# ├── iteration_10000/
# │   ├── loss_landscape_*.h5
# │   ├── loss_landscape_*_2dcontour.pdf
# │   ├── loss_landscape_*_2dcontourf.pdf
# │   ├── loss_landscape_*_2dheat.pdf
# │   └── loss_landscape_*_3dsurface.pdf
# ├── iteration_20000/
# │   └── ...
# └── ...

# Base checkpoint directory (modify this path to your checkpoint location)
CKPT_BASE_DIR="/path/to/checkpoint/directory"

# Config script path (modify this path to your config location)
CONFIG_SCRIPT="zoo/atari/config/atari_unizero_loss_landscape.py"

# Base log directory (modify this to your desired output location)
BASE_LOG_DIR="/path/to/output/directory"

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

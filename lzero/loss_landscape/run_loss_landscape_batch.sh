#!/bin/bash

################################################################################
# Batch Loss Landscape Generation Script
#
# This script generates loss landscapes for multiple checkpoints in batch mode.
# It iterates through checkpoints and visualizes the loss landscape for each.
#
# Usage:
#     bash run_loss_landscape_batch.sh [OPTIONS]
#
# Options:
#     --ckpt-dir <path>      Directory containing checkpoint files (required)
#     --env <env_id>         Atari environment ID (default: MsPacmanNoFrameskip-v4)
#     --seed <seed>          Random seed (default: 0)
#     --log-dir <path>       Base directory for outputs (default: ./loss_landscape_batch)
#     --iterations <list>    Comma-separated iterations to process (default: 10000,20000,...,100000)
#     --help                 Show this help message
#
# Examples:
#     # Process checkpoints from 10K to 100K with default settings
#     bash run_loss_landscape_batch.sh --ckpt-dir ./checkpoints
#
#     # Custom environment and iterations
#     bash run_loss_landscape_batch.sh \
#         --ckpt-dir ./checkpoints \
#         --env PongNoFrameskip-v4 \
#         --iterations 10000,50000,100000 \
#         --log-dir ./results/loss_landscapes
#
################################################################################

set -e  # Exit on error

# Get script directory for relative paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"  # Go up to project root

# Default values
ENV_ID="MsPacmanNoFrameskip-v4"
SEED=0
BASE_LOG_DIR="./loss_landscape_batch"
ITERATIONS="10000,20000,30000,40000,50000,60000,70000,80000,90000,100000"
CKPT_BASE_DIR=""

# Config script - using relative path from script location
CONFIG_SCRIPT="${SCRIPT_DIR}/../../zoo/atari/config/atari_unizero_loss_landscape.py"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'  # No Color

# Functions
print_help() {
    grep "^#" "$0" | tail -n +3 | head -n 30 | sed 's/^# //' | sed 's/^#!//'
}

print_error() {
    echo -e "${RED}ERROR: $1${NC}" >&2
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --ckpt-dir)
            CKPT_BASE_DIR="$2"
            shift 2
            ;;
        --env)
            ENV_ID="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --log-dir)
            BASE_LOG_DIR="$2"
            shift 2
            ;;
        --iterations)
            ITERATIONS="$2"
            shift 2
            ;;
        --help)
            print_help
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$CKPT_BASE_DIR" ]; then
    print_error "Checkpoint directory is required"
    echo "Use --help for usage information"
    exit 1
fi

# Create directories
mkdir -p "$BASE_LOG_DIR"

# Verify config script exists
if [ ! -f "$CONFIG_SCRIPT" ]; then
    print_error "Config script not found: $CONFIG_SCRIPT"
    exit 1
fi

# Verify checkpoint directory exists
if [ ! -d "$CKPT_BASE_DIR" ]; then
    print_error "Checkpoint directory not found: $CKPT_BASE_DIR"
    exit 1
fi

# Convert comma-separated iterations to array
IFS=',' read -ra ITER_ARRAY <<< "$ITERATIONS"

# Summary
echo "================================================================================"
print_info "Loss Landscape Batch Processing"
echo "================================================================================"
echo "Environment:           $ENV_ID"
echo "Random seed:           $SEED"
echo "Checkpoint directory:  $CKPT_BASE_DIR"
echo "Output directory:      $BASE_LOG_DIR"
echo "Iterations to process: ${#ITER_ARRAY[@]}"
echo "================================================================================"
echo ""

# Process each checkpoint
processed=0
success=0
failed=0

for iter in "${ITER_ARRAY[@]}"; do
    iter=$(echo "$iter" | xargs)  # Trim whitespace

    echo "=================================================="
    print_info "Processing checkpoint: iteration_${iter}.pth.tar"
    echo "=================================================="

    # Set checkpoint path
    CKPT_PATH="${CKPT_BASE_DIR}/iteration_${iter}.pth.tar"

    # Check if checkpoint exists
    if [ ! -f "$CKPT_PATH" ]; then
        print_warning "Checkpoint not found: $CKPT_PATH"
        echo "Skipping iteration ${iter}..."
        ((failed++))
        echo ""
        continue
    fi

    # Set log directory for this iteration
    LOG_DIR="${BASE_LOG_DIR}/iteration_${iter}"

    echo "Checkpoint path:   $CKPT_PATH"
    echo "Output directory:  $LOG_DIR"
    echo ""

    # Run the loss landscape script
    if python "$CONFIG_SCRIPT" \
        --env "$ENV_ID" \
        --seed "$SEED" \
        --ckpt "$CKPT_PATH" \
        --log_dir "$LOG_DIR"; then
        print_success "Completed iteration ${iter}"
        ((success++))
    else
        print_error "Failed to process iteration ${iter}"
        ((failed++))
    fi

    ((processed++))
    echo ""
done

# Final summary
echo "================================================================================"
print_info "Batch Processing Complete"
echo "================================================================================"
echo "Total processed:  $processed"
print_success "Successful:       $success"
if [ $failed -gt 0 ]; then
    print_error "Failed:           $failed"
else
    print_success "Failed:           0"
fi
echo "Results saved in: $BASE_LOG_DIR"
echo "================================================================================"

# Exit with error code if any failed
if [ $failed -gt 0 ]; then
    exit 1
fi

exit 0

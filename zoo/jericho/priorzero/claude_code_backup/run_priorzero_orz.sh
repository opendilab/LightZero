#!/bin/bash
# PriorZero-ORZ Quick Start Script
#
# Usage:
#   bash run_priorzero_orz.sh [mode]
#
# Modes:
#   debug    - Debug mode with small settings
#   single   - Single node training
#   multi    - Multi-node training (run on master node)
#   stop     - Stop all Ray processes

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
LIGHTZER_PATH="/mnt/nfs/zhangjinouwen/puyuan/LightZero"
ORZ_PATH="/mnt/nfs/zhangjinouwen/puyuan/Open-Reasoner-Zero"
MODE="${1:-single}"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}PriorZero-ORZ Training Pipeline${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Function: Check dependencies
check_dependencies() {
    echo -e "${YELLOW}[1/4] Checking dependencies...${NC}"

    # Check Ray
    if ! python -c "import ray" 2>/dev/null; then
        echo -e "${RED}Error: Ray not installed${NC}"
        echo "Install: pip install ray"
        exit 1
    fi

    # Check ORZ path
    if [ ! -d "$ORZ_PATH" ]; then
        echo -e "${RED}Error: ORZ not found at $ORZ_PATH${NC}"
        exit 1
    fi

    # Check vLLM
    if ! python -c "import vllm" 2>/dev/null; then
        echo -e "${YELLOW}Warning: vLLM not installed (optional for inference)${NC}"
    fi

    echo -e "${GREEN}✓ Dependencies OK${NC}"
}

# Function: Setup environment
setup_environment() {
    echo -e "${YELLOW}[2/4] Setting up environment...${NC}"

    # Add ORZ to PYTHONPATH
    export PYTHONPATH="$ORZ_PATH:$PYTHONPATH"

    # Set working directory
    cd "$LIGHTZER_PATH"

    echo -e "${GREEN}✓ Environment ready${NC}"
}

# Function: Start training
start_training() {
    local mode=$1

    echo -e "${YELLOW}[3/4] Starting training (mode: $mode)...${NC}"

    case $mode in
        debug)
            echo -e "${YELLOW}Running in DEBUG mode...${NC}"
            DEBUG_MODE=True python -m zoo.jericho.priorzero.priorzero_orz_entry
            ;;

        single)
            echo -e "${YELLOW}Running single-node training...${NC}"
            python -m zoo.jericho.priorzero.priorzero_orz_entry
            ;;

        multi)
            echo -e "${YELLOW}Running multi-node training...${NC}"
            echo -e "${YELLOW}Make sure you've started Ray on all nodes!${NC}"
            echo ""
            echo "Commands for other nodes:"
            echo "  ray start --address='<this-node-ip>:6379'"
            echo ""
            read -p "Press Enter to continue..."

            # Start Ray head
            ray start --head --port=6379 --num-cpus=0

            # Run training
            python -m zoo.jericho.priorzero.priorzero_orz_entry
            ;;

        stop)
            echo -e "${YELLOW}Stopping Ray...${NC}"
            ray stop
            echo -e "${GREEN}✓ Ray stopped${NC}"
            exit 0
            ;;

        *)
            echo -e "${RED}Error: Unknown mode '$mode'${NC}"
            echo "Valid modes: debug, single, multi, stop"
            exit 1
            ;;
    esac
}

# Function: Monitor training
monitor_training() {
    echo ""
    echo -e "${YELLOW}[4/4] Training started!${NC}"
    echo ""
    echo -e "${GREEN}Monitor options:${NC}"
    echo "  TensorBoard:  tensorboard --logdir=priorzero_orz_logs/ --port=6006"
    echo "  Logs:         tail -f priorzero_orz_logs/*/log/*.log"
    echo "  Ray status:   ray status"
    echo ""
}

# Main execution
main() {
    echo -e "${YELLOW}Mode: $MODE${NC}"
    echo ""

    check_dependencies
    setup_environment
    start_training "$MODE"
    monitor_training
}

# Run main
main

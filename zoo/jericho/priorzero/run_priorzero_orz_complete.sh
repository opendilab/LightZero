#!/bin/bash
# PriorZero-ORZ Complete Integration - Quick Start Script
#
# This script provides easy commands to run the PriorZero-ORZ hybrid pipeline
# with different configurations.
#
# Author: PriorZero Team
# Date: 2025-10-21
# Version: v2.0 - Complete ORZ Integration

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Base directory
BASE_DIR="/mnt/nfs/zhangjinouwen/puyuan/LightZero"
ORZ_DIR="/mnt/nfs/zhangjinouwen/puyuan/Open-Reasoner-Zero"

echo -e "${BLUE}================================================================${NC}"
echo -e "${BLUE}PriorZero-ORZ Complete Integration - Quick Start${NC}"
echo -e "${BLUE}================================================================${NC}"

# Check if we're in the right directory
if [ ! -d "$BASE_DIR" ]; then
    echo -e "${RED}Error: LightZero directory not found: $BASE_DIR${NC}"
    exit 1
fi

cd "$BASE_DIR"
echo -e "${GREEN}✓ Changed to directory: $BASE_DIR${NC}"

# Check ORZ availability
if [ -d "$ORZ_DIR" ]; then
    echo -e "${GREEN}✓ ORZ directory found: $ORZ_DIR${NC}"
    export PYTHONPATH="${ORZ_DIR}:${PYTHONPATH}"
    echo -e "${GREEN}✓ Added ORZ to PYTHONPATH${NC}"
else
    echo -e "${YELLOW}⚠ ORZ directory not found: $ORZ_DIR${NC}"
    echo -e "${YELLOW}  Will use PriorZero's built-in LLM training${NC}"
fi

# Check Python
if ! command -v python &> /dev/null; then
    echo -e "${RED}Error: Python not found${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Python found: $(python --version)${NC}"

# Check CUDA
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓ CUDA available:${NC}"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -3
else
    echo -e "${YELLOW}⚠ nvidia-smi not found - GPU may not be available${NC}"
fi

echo -e "${BLUE}================================================================${NC}"
echo ""

# Show menu
echo -e "${BLUE}Select training mode:${NC}"
echo ""
echo -e "  ${GREEN}1)${NC} Debug mode (quick test, ~30-60 min)"
echo -e "  ${GREEN}2)${NC} Normal training (full training, several hours)"
echo -e "  ${GREEN}3)${NC} Debug mode without ORZ (PriorZero only)"
echo -e "  ${GREEN}4)${NC} Check dependencies and environment"
echo -e "  ${GREEN}5)${NC} View logs (real-time)"
echo -e "  ${GREEN}6)${NC} View TensorBoard"
echo -e "  ${GREEN}7)${NC} Clean up old runs"
echo -e "  ${GREEN}q)${NC} Quit"
echo ""

read -p "Enter your choice [1-7 or q]: " choice

case $choice in
    1)
        echo -e "${GREEN}Starting Debug mode...${NC}"
        echo -e "${YELLOW}This will run for ~30-60 minutes (100 iterations)${NC}"
        echo ""
        read -p "Press Enter to continue or Ctrl+C to cancel..."
        DEBUG_MODE=True python -m zoo.jericho.priorzero.priorzero_orz_complete
        ;;

    2)
        echo -e "${GREEN}Starting Normal training...${NC}"
        echo -e "${YELLOW}This will run for several hours (10000 iterations)${NC}"
        echo ""
        read -p "Press Enter to continue or Ctrl+C to cancel..."
        python -m zoo.jericho.priorzero.priorzero_orz_complete
        ;;

    3)
        echo -e "${GREEN}Starting Debug mode (PriorZero only)...${NC}"
        echo -e "${YELLOW}ORZ will be disabled${NC}"
        echo ""
        read -p "Press Enter to continue or Ctrl+C to cancel..."
        # Note: User needs to modify code to set use_orz_trainer = False
        echo -e "${RED}TODO: Modify priorzero_orz_complete.py:${NC}"
        echo -e "  Set: ${YELLOW}self.use_orz_trainer = False${NC}"
        echo -e "  Then run: ${YELLOW}DEBUG_MODE=True python -m zoo.jericho.priorzero.priorzero_orz_complete${NC}"
        ;;

    4)
        echo -e "${GREEN}Checking dependencies...${NC}"
        echo ""

        # Check vLLM
        python -c "
try:
    from vllm import AsyncLLMEngine
    print('${GREEN}✓ vLLM available${NC}')
except ImportError:
    print('${RED}✗ vLLM not available${NC}')
"

        # Check ORZ
        python -c "
import sys
sys.path.insert(0, '${ORZ_DIR}')
try:
    from orz.ppo import RayPPOTrainer
    print('${GREEN}✓ ORZ available${NC}')
except ImportError as e:
    print('${RED}✗ ORZ not available:${NC}', e)
"

        # Check Ray
        python -c "
try:
    import ray
    print('${GREEN}✓ Ray available${NC}')
except ImportError:
    print('${RED}✗ Ray not available${NC}')
"

        # Check transformers
        python -c "
try:
    from transformers import AutoTokenizer
    print('${GREEN}✓ Transformers available${NC}')
except ImportError:
    print('${RED}✗ Transformers not available${NC}')
"

        echo ""
        echo -e "${BLUE}GPU Status:${NC}"
        nvidia-smi
        ;;

    5)
        echo -e "${GREEN}Viewing logs (real-time)...${NC}"
        echo -e "${YELLOW}Press Ctrl+C to exit${NC}"
        echo ""

        # Find latest log directory
        LOG_DIR=$(ls -td data_priorzero_*/log/*.log 2>/dev/null | head -1)

        if [ -z "$LOG_DIR" ]; then
            echo -e "${RED}No log files found${NC}"
            exit 1
        fi

        echo -e "${BLUE}Tailing: $LOG_DIR${NC}"
        echo ""
        tail -f "$LOG_DIR"
        ;;

    6)
        echo -e "${GREEN}Starting TensorBoard...${NC}"
        echo ""

        # Find log directories
        if [ ! -d "data_priorzero"* ]; then
            echo -e "${RED}No training runs found${NC}"
            exit 1
        fi

        echo -e "${BLUE}TensorBoard will be available at: http://localhost:6006${NC}"
        echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
        echo ""

        tensorboard --logdir=./data_priorzero --port=6006
        ;;

    7)
        echo -e "${GREEN}Clean up old runs...${NC}"
        echo ""

        # List existing runs
        echo -e "${BLUE}Existing runs:${NC}"
        du -sh data_priorzero_* 2>/dev/null || echo "  No runs found"
        echo ""

        read -p "Delete ALL old runs? [y/N]: " confirm

        if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
            rm -rf data_priorzero_*
            echo -e "${GREEN}✓ All old runs deleted${NC}"
        else
            echo -e "${YELLOW}Cancelled${NC}"
        fi
        ;;

    q|Q)
        echo -e "${BLUE}Goodbye!${NC}"
        exit 0
        ;;

    *)
        echo -e "${RED}Invalid choice${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${BLUE}================================================================${NC}"
echo -e "${GREEN}Done!${NC}"
echo -e "${BLUE}================================================================${NC}"

#!/bin/bash
# fix_environment.sh
# Fix numpy version conflicts and other dependency issues

echo "=========================================="
echo "Fixing PriorZero Environment Dependencies"
echo "=========================================="

# 1. Fix numpy version (downgrade to 1.26.4 for compatibility)
echo ""
echo "1. Fixing numpy version..."
pip install "numpy<2,>=1.24.1" --force-reinstall --no-deps

# 2. Reinstall conflicting packages
echo ""
echo "2. Reinstalling di-engine and lightzero..."
pip install di-engine==0.5.3 --no-deps
pip install lightzero==0.2.0 --no-deps

# 3. Verify installations
echo ""
echo "3. Verifying installations..."
python -c "import numpy; print(f'numpy version: {numpy.__version__}')"
python -c "import torch; print(f'torch version: {torch.__version__}')"
python -c "import vllm; print(f'vllm version: {vllm.__version__}')"

echo ""
echo "=========================================="
echo "Environment fix complete!"
echo "=========================================="
echo ""
echo "Now you can run:"
echo "  python priorzero_config.py"
echo "  python game_segment_priorzero.py"
echo "  python priorzero_entry.py --quick_test"

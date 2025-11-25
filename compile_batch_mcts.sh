#!/bin/bash

# AlphaZero Batch MCTS 编译脚本
# 此脚本自动编译batch MCTS C++模块

set -e  # 遇到错误立即退出

echo "========================================================================"
echo "AlphaZero Batch MCTS Compilation Script"
echo "========================================================================"

# 0. 检查当前Python路径
CURRENT_PYTHON=$(which python)
echo "Target Python: ${CURRENT_PYTHON}"
echo "Python Version: $(python --version)"

# 进入目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}/lzero/mcts/ctree/ctree_alphazero"

echo ""
echo "[Step 1/4] Preparing CMakeLists.txt..."
# 备份原CMakeLists.txt
if [ ! -f "CMakeLists.txt.backup" ]; then
    cp CMakeLists.txt CMakeLists.txt.backup
    echo "  ✓ Backed up original CMakeLists.txt"
else
    echo "  ✓ Backup already exists"
fi

# 使用batch版本
cp CMakeLists_batch.txt CMakeLists.txt
echo "  ✓ Using CMakeLists_batch.txt"

echo ""
echo "[Step 2/4] Creating build directory..."
# 强制清理旧的 build 目录以确保重新检测 Python 版本
if [ -d "build_batch" ]; then
    rm -rf build_batch
    echo "  ✓ Cleaned old build directory"
fi
mkdir -p build_batch
cd build_batch
echo "  ✓ Directory ready: $(pwd)"

echo ""
echo "[Step 3/4] Running CMake..."
# 修改点：添加 -DPYTHON_EXECUTABLE=$(which python) 强制使用当前环境的Python
cmake -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE="${CURRENT_PYTHON}" .. || {
    echo "  ❌ CMake failed"
    cd ..
    mv CMakeLists.txt.backup CMakeLists.txt
    exit 1
}
echo "  ✓ CMake configuration successful"

echo ""
echo "[Step 4/4] Compiling..."
make -j$(nproc) || {
    echo "  ❌ Compilation failed"
    cd ..
    mv CMakeLists.txt.backup CMakeLists.txt
    exit 1
}
echo "  ✓ Compilation successful"

# 恢复原CMakeLists.txt
cd ..
mv CMakeLists.txt.backup CMakeLists.txt
echo ""
echo "  ✓ Restored original CMakeLists.txt"

# 检查输出
echo ""
echo "========================================================================"
echo "Compilation Complete!"
echo "========================================================================"
OUTPUT_FILE="build/mcts_alphazero_batch.cpython-*.so"
if ls $OUTPUT_FILE 1> /dev/null 2>&1; then
    echo "Module location: $(ls $OUTPUT_FILE)"
    echo "Module size: $(du -h $OUTPUT_FILE | cut -f1)"
else
    echo "⚠ Warning: Output file not found"
fi

echo ""
echo "Next steps:"
echo "  1. Test: python test_batch_mcts_simple.py"
echo "  2. Run: python test_performance_comparison.py"
echo "  3. Use alphazero_batch in your config"
echo ""
echo "Documentation: ALPHAZERO_BATCH_IMPLEMENTATION_GUIDE.md"
echo "========================================================================"
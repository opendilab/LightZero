"""
智能导入模块 - 自动处理路径和Python版本问题
这个模块提供鲁棒的导入机制
"""
import sys
import os
import glob
import importlib.util

def get_batch_mcts_module():
    """
    智能导入 mcts_alphazero_batch 模块

    Returns:
        module: 导入的模块

    Raises:
        ImportError: 如果无法导入
    """
    # 1. 确定模块路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(script_dir, 'lzero', 'mcts', 'ctree', 'ctree_alphazero', 'build')

    if not os.path.exists(build_dir):
        raise ImportError(f"Build directory not found: {build_dir}")

    # 2. 查找.so文件
    so_pattern = os.path.join(build_dir, "mcts_alphazero_batch*.so")
    so_files = glob.glob(so_pattern)

    if not so_files:
        raise ImportError(
            f"No .so file found in {build_dir}\n"
            f"Please compile first: ./compile_batch_mcts.sh"
        )

    # 3. 检查Python版本匹配
    current_py_ver = f"{sys.version_info.major}{sys.version_info.minor}"
    matching_files = [f for f in so_files if f"cpython-{current_py_ver}" in f]

    if not matching_files:
        # 没有匹配的版本,列出可用的版本
        available_versions = []
        for f in so_files:
            if 'cpython-' in f:
                ver = f.split('cpython-')[1].split('-')[0]
                available_versions.append(f"Python 3.{ver[1:]}")

        raise ImportError(
            f"No .so file for Python {sys.version_info.major}.{sys.version_info.minor}\n"
            f"Found .so files for: {', '.join(available_versions)}\n"
            f"Please recompile with current Python: ./compile_batch_mcts.sh"
        )

    # 4. 尝试导入
    if build_dir not in sys.path:
        sys.path.insert(0, build_dir)

    try:
        import mcts_alphazero_batch
        return mcts_alphazero_batch
    except ImportError as e:
        # 提供详细错误信息
        raise ImportError(
            f"Failed to import mcts_alphazero_batch: {e}\n"
            f"Module file: {matching_files[0]}\n"
            f"Build dir: {build_dir}\n"
            f"Python: {sys.executable}\n"
            f"Solution: Try recompiling with ./compile_batch_mcts.sh"
        )

# 使用示例
if __name__ == "__main__":
    try:
        module = get_batch_mcts_module()
        print("✓ Module imported successfully!")
        print(f"  Location: {module.__file__}")
        print(f"  Has Roots: {hasattr(module, 'Roots')}")
        print(f"  Has batch_traverse: {hasattr(module, 'batch_traverse')}")
        print(f"  Has batch_backpropagate: {hasattr(module, 'batch_backpropagate')}")
    except ImportError as e:
        print(f"❌ Import failed:\n{e}")
        sys.exit(1)

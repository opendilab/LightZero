#!/usr/bin/env python3
"""
快速验证 Batch MCTS 模块是否正常工作 (增强版)
使用智能导入机制,自动处理路径和版本问题
"""
import sys
import os

# 添加当前目录到路径
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

# 使用智能导入
from smart_import import get_batch_mcts_module

print("="*70)
print("Batch MCTS 模块快速验证 (增强版)")
print("="*70)

# 1. 显示Python信息
print(f"\nPython 信息:")
print(f"  版本: {sys.version.split()[0]}")
print(f"  路径: {sys.executable}")

# 2. 尝试导入模块
print(f"\n导入模块:")
try:
    mcts_alphazero_batch = get_batch_mcts_module()
    print(f"  ✓ 导入成功")
    print(f"  位置: {mcts_alphazero_batch.__file__}")
except ImportError as e:
    print(f"  ❌ 导入失败")
    print(f"\n错误详情:")
    print(f"  {e}")
    sys.exit(1)

# 3. 检查核心功能
print(f"\n检查核心功能:")
checks = [
    ("Roots 类", hasattr(mcts_alphazero_batch, 'Roots')),
    ("SearchResults 类", hasattr(mcts_alphazero_batch, 'SearchResults')),
    ("batch_traverse 函数", hasattr(mcts_alphazero_batch, 'batch_traverse')),
    ("batch_backpropagate 函数", hasattr(mcts_alphazero_batch, 'batch_backpropagate')),
]

all_passed = True
for name, result in checks:
    status = "✓" if result else "❌"
    print(f"  {status} {name}")
    if not result:
        all_passed = False

if not all_passed:
    print("\n❌ 部分功能缺失")
    sys.exit(1)

# 4. 简单功能测试
print("\n执行简单功能测试:")
try:
    import numpy as np

    # 创建roots
    batch_size = 4
    legal_actions = [[0, 1, 2] for _ in range(batch_size)]
    roots = mcts_alphazero_batch.Roots(batch_size, legal_actions)
    print(f"  ✓ 创建 {batch_size} 个 roots")

    # 准备
    noises = [np.random.dirichlet([0.3] * 3).tolist() for _ in range(batch_size)]
    values = [0.0] * batch_size
    policies = [[0.33, 0.33, 0.34] for _ in range(batch_size)]
    roots.prepare(0.25, noises, values, policies)
    print(f"  ✓ Roots 准备完成")

    # Traverse
    current_legal = [[0, 1, 2] for _ in range(batch_size)]
    results = mcts_alphazero_batch.batch_traverse(roots, 19652, 1.25, current_legal)
    print(f"  ✓ Batch traverse 成功")

    # Backpropagate
    values = [0.5, -0.3, 0.8, 0.1]
    policies = [[0.33, 0.33, 0.34] for _ in range(batch_size)]
    mcts_alphazero_batch.batch_backpropagate(results, values, policies, current_legal, "play_with_bot_mode")
    print(f"  ✓ Batch backpropagate 成功")

    # 获取结果
    distributions = roots.get_distributions()
    print(f"  ✓ 获取 distributions 成功")

except Exception as e:
    print(f"\n❌ 功能测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("✅ 所有检查通过! Batch MCTS 模块工作正常")
print("="*70)
print("\n下一步:")
print("  1. 运行完整测试: python test_batch_mcts_simple.py")
print("  2. 性能对比: python test_performance_comparison.py")
print("  3. 在训练中使用: 修改config使用 alphazero_batch")
print("\n详细文档: QUICK_START.md")

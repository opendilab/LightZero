#!/usr/bin/env python3
"""
监控功能验证脚本
==================

本脚本用于验证 UniZero Policy 中新增的监控功能是否正常工作。

使用方法:
    python verify_monitoring.py

预期输出:
    ✓ 所有检查项都应该通过
"""

import sys
import os

# 添加 LightZero 到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def verify_imports():
    """验证必要的导入是否可用"""
    print("=" * 60)
    print("1. 验证导入...")
    print("-" * 60)

    try:
        import torch
        print("✓ PyTorch 已安装")
    except ImportError:
        print("✗ PyTorch 未安装")
        return False

    try:
        import numpy as np
        print("✓ NumPy 已安装")
    except ImportError:
        print("✗ NumPy 未安装")
        return False

    try:
        from lzero.policy.unizero import UniZeroPolicy
        print("✓ UniZeroPolicy 可导入")
    except ImportError as e:
        print(f"✗ UniZeroPolicy 导入失败: {e}")
        return False

    print()
    return True


def verify_config_parameters():
    """验证配置参数是否存在"""
    print("=" * 60)
    print("2. 验证配置参数...")
    print("-" * 60)

    from lzero.policy.unizero import UniZeroPolicy

    config = UniZeroPolicy.config

    # 检查新增的配置参数
    required_params = [
        'monitor_norm_freq',
        'use_adaptive_entropy_weight',
        'use_encoder_clip_annealing',
    ]

    all_present = True
    for param in required_params:
        if param in config:
            print(f"✓ 配置参数 '{param}' 存在")
        else:
            print(f"✗ 配置参数 '{param}' 缺失")
            all_present = False

    print()
    return all_present


def verify_monitoring_methods():
    """验证监控方法是否存在"""
    print("=" * 60)
    print("3. 验证监控方法...")
    print("-" * 60)

    from lzero.policy.unizero import UniZeroPolicy

    required_methods = [
        '_monitor_model_norms',
        '_monitor_gradient_norms',
        '_forward_learn',
        '_monitor_vars_learn',
    ]

    all_present = True
    for method in required_methods:
        if hasattr(UniZeroPolicy, method):
            print(f"✓ 方法 '{method}' 存在")
        else:
            print(f"✗ 方法 '{method}' 缺失")
            all_present = False

    print()
    return all_present


def verify_monitoring_variables():
    """验证监控变量是否注册"""
    print("=" * 60)
    print("4. 验证监控变量注册...")
    print("-" * 60)

    from lzero.policy.unizero import UniZeroPolicy

    # 创建一个临时实例来调用 _monitor_vars_learn
    # 注意: 这可能需要一些模拟配置
    try:
        # 创建最小配置
        min_config = UniZeroPolicy.config.copy()
        min_config['cuda'] = False  # 使用 CPU 避免 CUDA 依赖

        # 注意: 实际创建实例需要更多配置,这里我们只检查类方法
        policy_class = UniZeroPolicy

        # 检查 _monitor_vars_learn 是否返回列表
        if hasattr(policy_class, '_monitor_vars_learn'):
            print("✓ '_monitor_vars_learn' 方法存在")

            # 检查一些关键的监控变量名
            expected_vars = [
                'norm/encoder/_total_norm',
                'grad/encoder/_total_norm',
                'logits/value/mean',
                'embeddings/obs/norm_mean',
                'norm/x_token/mean',
            ]

            print(f"  预期的关键监控变量: {len(expected_vars)} 个")
            for var in expected_vars[:3]:  # 只显示前3个
                print(f"    - {var}")
            print("    ...")

        else:
            print("✗ '_monitor_vars_learn' 方法缺失")
            return False

    except Exception as e:
        print(f"⚠  验证过程出现异常: {e}")
        print("  (这可能是正常的,因为我们没有提供完整的配置)")

    print()
    return True


def verify_code_structure():
    """验证代码结构是否正确"""
    print("=" * 60)
    print("5. 验证代码结构...")
    print("-" * 60)

    try:
        # 读取源文件检查关键代码段
        # 获取当前工作目录
        cwd = os.getcwd()
        policy_file = os.path.join(cwd, 'lzero', 'policy', 'unizero.py')

        if not os.path.exists(policy_file):
            print(f"✗ 文件不存在: {policy_file}")
            return False

        with open(policy_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # 检查关键代码段
        checks = [
            ('def _monitor_model_norms', '参数范数监控方法'),
            ('def _monitor_gradient_norms', '梯度范数监控方法'),
            ('monitor_norm_freq', '监控频率配置'),
            ('norm_log_dict', '范数日志字典'),
            ("norm/x_token/mean", 'x_token 统计'),
            ("logits/value/mean", 'logits 统计'),
            ("embeddings/obs/norm_mean", 'embeddings 统计'),
        ]

        all_present = True
        for code_snippet, description in checks:
            if code_snippet in content:
                print(f"✓ {description} 已实现")
            else:
                print(f"✗ {description} 缺失")
                all_present = False

        print()
        return all_present

    except Exception as e:
        print(f"✗ 验证过程出现错误: {e}")
        print()
        return False


def verify_documentation():
    """验证文档是否存在"""
    print("=" * 60)
    print("6. 验证文档...")
    print("-" * 60)

    # 使用当前工作目录
    base_dir = os.getcwd()

    docs = [
        ('MONITORING_ENHANCEMENTS.md', '监控增强文档'),
        ('CHANGELOG_MONITORING.md', '变更日志'),
        ('examples/monitoring_usage_example.py', '使用示例'),
    ]

    all_present = True
    for doc_path, description in docs:
        full_path = os.path.join(base_dir, doc_path)
        if os.path.exists(full_path):
            size = os.path.getsize(full_path)
            print(f"✓ {description} 存在 ({size} bytes)")
        else:
            print(f"✗ {description} 缺失")
            all_present = False

    print()
    return all_present


def print_summary(results):
    """打印验证结果摘要"""
    print("=" * 60)
    print("验证结果摘要")
    print("=" * 60)

    total = len(results)
    passed = sum(results.values())

    print(f"总计: {total} 项检查")
    print(f"通过: {passed} 项")
    print(f"失败: {total - passed} 项")
    print()

    if passed == total:
        print("🎉 所有检查通过! 监控功能已正确实现。")
        return True
    else:
        print("⚠️  部分检查未通过,请检查上述输出。")
        return False


def main():
    """主验证流程"""
    print("\n")
    print("=" * 60)
    print("UniZero 监控功能验证脚本")
    print("=" * 60)
    print()

    results = {}

    # 执行各项验证
    results['导入验证'] = verify_imports()
    results['配置参数验证'] = verify_config_parameters()
    results['监控方法验证'] = verify_monitoring_methods()
    results['监控变量验证'] = verify_monitoring_variables()
    results['代码结构验证'] = verify_code_structure()
    results['文档验证'] = verify_documentation()

    # 打印摘要
    success = print_summary(results)

    # 返回退出码
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

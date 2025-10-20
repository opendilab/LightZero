#!/usr/bin/env python
"""
vLLM Compatibility Checker and Auto-Fixer

This script checks for common vLLM issues and applies fixes automatically.

Usage:
    python check_vllm_compatibility.py

Author: PriorZero Team
Date: 2025-10-20
"""

import os
import sys
import subprocess
from typing import Dict, List, Tuple


def check_gpu_status() -> List[Dict]:
    """Check GPU memory and usage status."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,memory.used,memory.free,utilization.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            check=True
        )

        gpus = []
        for line in result.stdout.strip().split('\n'):
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 5:
                gpus.append({
                    'index': int(parts[0]),
                    'name': parts[1],
                    'used_mb': float(parts[2]),
                    'free_mb': float(parts[3]),
                    'utilization': int(parts[4])
                })
        return gpus
    except Exception as e:
        print(f"⚠️  Warning: Could not check GPU status: {e}")
        return []


def find_best_gpu(gpus: List[Dict]) -> int:
    """Find the GPU with most free memory and lowest utilization."""
    if not gpus:
        return 0

    # Score based on free memory (70%) and low utilization (30%)
    def score(gpu):
        return gpu['free_mb'] * 0.7 + (100 - gpu['utilization']) * 1000 * 0.3

    best_gpu = max(gpus, key=score)
    return best_gpu['index']


def check_vllm_version() -> str:
    """Check installed vLLM version."""
    try:
        import vllm
        return vllm.__version__
    except Exception:
        return "unknown"


def check_ray_status() -> Tuple[bool, str]:
    """Check if Ray is already initialized."""
    try:
        import ray
        if ray.is_initialized():
            try:
                # Try to get Ray version
                return True, f"Ray {ray.__version__} (already initialized)"
            except:
                return True, "Ray (already initialized, version unknown)"
        return False, f"Ray {ray.__version__} (not initialized)"
    except Exception as e:
        return False, f"Ray not available: {e}"


def apply_fixes() -> Dict[str, str]:
    """Apply recommended environment fixes."""
    fixes = {}

    # Fix 1: Disable V1 engine
    if 'VLLM_USE_V1' not in os.environ or os.environ['VLLM_USE_V1'] != '0':
        os.environ['VLLM_USE_V1'] = '0'
        fixes['VLLM_USE_V1'] = '0 (disabled V1 engine for stability)'

    # Fix 2: Set CUDA memory allocation strategy
    if 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ:
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
        fixes['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512 (reduce fragmentation)'

    # Fix 3: Suggest best GPU if multiple available
    gpus = check_gpu_status()
    if len(gpus) > 1 and 'CUDA_VISIBLE_DEVICES' not in os.environ:
        best_gpu = find_best_gpu(gpus)
        suggestion = f"Suggested: export CUDA_VISIBLE_DEVICES={best_gpu} (GPU {best_gpu} has most free memory)"
        fixes['CUDA_VISIBLE_DEVICES'] = suggestion

    return fixes


def main():
    print("=" * 80)
    print("vLLM Compatibility Checker")
    print("=" * 80)
    print()

    # Check 1: GPU Status
    print("📊 GPU Status:")
    print("-" * 80)
    gpus = check_gpu_status()
    if gpus:
        for gpu in gpus:
            status = "🟢" if gpu['utilization'] < 50 else "🟡" if gpu['utilization'] < 80 else "🔴"
            print(f"{status} GPU {gpu['index']}: {gpu['name']}")
            print(f"   Memory: {gpu['used_mb']/1024:.1f} GB used, {gpu['free_mb']/1024:.1f} GB free")
            print(f"   Utilization: {gpu['utilization']}%")

        # Find and recommend best GPU
        best_gpu = find_best_gpu(gpus)
        print()
        print(f"✨ Recommended GPU: {best_gpu} (most available resources)")
    else:
        print("⚠️  Could not detect GPUs")

    print()

    # Check 2: vLLM Version
    print("🔧 vLLM Configuration:")
    print("-" * 80)
    vllm_version = check_vllm_version()
    print(f"vLLM Version: {vllm_version}")

    # Check 3: Ray Status
    ray_initialized, ray_info = check_ray_status()
    ray_status = "🟢" if not ray_initialized else "🟡"
    print(f"{ray_status} {ray_info}")
    if ray_initialized:
        print("   ⚠️  Ray cluster detected - may cause version conflicts")

    print()

    # Check 4: Current Environment Variables
    print("🌍 Current Environment:")
    print("-" * 80)
    env_vars = {
        'VLLM_USE_V1': os.environ.get('VLLM_USE_V1', 'not set'),
        'PYTORCH_CUDA_ALLOC_CONF': os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'not set'),
        'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES', 'not set (all GPUs)'),
    }
    for key, value in env_vars.items():
        status = "✓" if value != 'not set' else "✗"
        print(f"{status} {key}: {value}")

    print()

    # Apply Fixes
    print("🔨 Applying Recommended Fixes:")
    print("-" * 80)
    fixes = apply_fixes()
    if fixes:
        for key, value in fixes.items():
            print(f"✓ {key}: {value}")
    else:
        print("✓ All recommended settings already applied")

    print()

    # Final Recommendations
    print("💡 Recommendations:")
    print("-" * 80)
    print("1. If running in shared GPU environment:")
    print("   - Use dedicated GPU: export CUDA_VISIBLE_DEVICES=<gpu_id>")
    print("   - Lower memory utilization: gpu_memory_utilization=0.75")
    print()
    print("2. If encountering memory profiling errors:")
    print("   - Ensure VLLM_USE_V1=0 (✓ already set by this script)")
    print()
    print("3. If model download is slow:")
    print("   - Set proxy environment variables (http_proxy, https_proxy)")
    print()
    print("4. Launch script:")
    print("   bash run_priorzero.sh --quick_test")
    print()

    print("=" * 80)
    print("✓ Compatibility check complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

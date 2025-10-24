"""
KV Cache 系统切换简化测试
==========================

快速验证新旧系统切换功能。
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

print("\n" + "="*70)
print("KV Cache 系统切换测试 (简化版)")
print("="*70 + "\n")

# Test 1: Import check
print("测试 1: 导入检查")
try:
    from lzero.model.unizero_world_models.kv_cache_manager import (
        KVCacheManager,
        KVCachePool,
        EvictionStrategy,
        CacheStats
    )
    print("✓ KVCacheManager 模块导入成功")
    print(f"  - KVCacheManager: {KVCacheManager}")
    print(f"  - KVCachePool: {KVCachePool}")
    print(f"  - EvictionStrategy: {EvictionStrategy}")
    print(f"  - CacheStats: {CacheStats}")
except Exception as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

# Test 2: Configuration flag test
print("\n测试 2: 配置标志测试")

# Use a simple class instead of Mock
class ConfigOld:
    env_num = 4
    game_segment_length = 100
    num_simulations = 50
    # use_new_cache_manager not defined, should default to False

class ConfigNew:
    env_num = 4
    game_segment_length = 100
    num_simulations = 50
    use_new_cache_manager = True

config_old = ConfigOld()
config_new = ConfigNew()

# Test old config (no flag)
use_new = getattr(config_old, 'use_new_cache_manager', False)
print(f"✓ 默认配置 (未设置标志): use_new_cache_manager = {use_new}")
assert use_new == False, f"Should default to False, got {use_new}"

# Test new config (flag=True)
use_new = getattr(config_new, 'use_new_cache_manager', False)
print(f"✓ 新系统配置 (设置标志=True): use_new_cache_manager = {use_new}")
assert use_new == True, f"Should be True, got {use_new}"

# Test 3: KVCacheManager creation
print("\n测试 3: KVCacheManager 创建")
try:
    manager = KVCacheManager(
        config=config_new,
        env_num=4,
        enable_stats=True
    )
    print(f"✓ KVCacheManager 创建成功")
    print(f"  - env_num: {manager.env_num}")
    print(f"  - init_pools: {len(manager.init_pools)} pools")
    print(f"  - recur_pool: {manager.recur_pool}")
    print(f"  - wm_pool: {manager.wm_pool}")
    print(f"  - enable_stats: {manager.enable_stats}")
except Exception as e:
    print(f"❌ 创建失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Stats
print("\n测试 4: 统计信息")
try:
    stats = manager.get_stats_summary()
    print(f"✓ 统计信息获取成功")
    print(f"  - stats_enabled: {stats['stats_enabled']}")
    print(f"  - init_pools keys: {list(stats['init_pools'].keys())}")
    assert stats['stats_enabled'] == True
except Exception as e:
    print(f"❌ 统计失败: {e}")
    sys.exit(1)

# Test 5: Cache operations
print("\n测试 5: Cache 操作")
try:
    from lzero.model.unizero_world_models.kv_caching import KeysValues
    import torch

    # Create a simple KeysValues
    test_kv = KeysValues(
        num_samples=2,
        num_heads=4,
        max_tokens=20,
        embed_dim=256,
        num_layers=2,
        device=torch.device('cpu')
    )
    print(f"✓ KeysValues 创建成功")

    # Test set and get
    cache_key = 12345
    env_id = 0

    index = manager.set_init_cache(env_id=env_id, cache_key=cache_key, kv_cache=test_kv)
    print(f"✓ Set cache: env_id={env_id}, cache_key={cache_key}, index={index}")

    retrieved = manager.get_init_cache(env_id=env_id, cache_key=cache_key)
    assert retrieved is not None
    assert retrieved is test_kv
    print(f"✓ Get cache: 检索成功")

    # Test cache miss
    missing = manager.get_init_cache(env_id=env_id, cache_key=99999)
    assert missing is None
    print(f"✓ Cache miss: 正确返回 None")

    # Check stats
    stats = manager.get_stats_summary()
    print(f"✓ 操作后统计: {stats['init_pools']['env_0']}")

except Exception as e:
    print(f"❌ Cache 操作失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Clear operation
print("\n测试 6: Clear 操作")
try:
    manager.clear_all()
    print(f"✓ clear_all() 执行成功")

    # Verify cache is cleared
    retrieved = manager.get_init_cache(env_id=0, cache_key=12345)
    assert retrieved is None
    print(f"✓ 清除后验证: cache 已清空")

except Exception as e:
    print(f"❌ Clear 失败: {e}")
    sys.exit(1)

# Summary
print("\n" + "="*70)
print("🎉 所有测试通过!")
print("="*70)
print("\n✅ Phase 1 集成验证成功:")
print("  1. ✓ KVCacheManager 模块可以正常导入")
print("  2. ✓ 配置标志工作正常 (默认False, 可设置True)")
print("  3. ✓ KVCacheManager 可以成功创建")
print("  4. ✓ 统计信息功能正常")
print("  5. ✓ Cache set/get/miss 操作正常")
print("  6. ✓ Clear 操作正常")
print("\n下一步: 在 world_model.py 中测试实际集成")
print("="*70 + "\n")

sys.exit(0)

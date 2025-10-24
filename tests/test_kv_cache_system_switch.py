"""
KV Cache 系统切换测试脚本
=============================

测试新旧 KV Cache 系统的配置切换功能。

运行方式:
    python tests/test_kv_cache_system_switch.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from unittest.mock import Mock
from lzero.model.unizero_world_models.world_model import WorldModel
from lzero.model.unizero_world_models.transformer import TransformerConfig
from lzero.model.unizero_world_models.tokenizer import Tokenizer


def create_test_config(use_new_cache=False):
    """创建测试配置"""
    config = TransformerConfig(
        tokens_per_block=2,
        max_blocks=10,
        max_tokens=20,
        context_length=8,
        num_layers=2,
        num_heads=4,
        embed_dim=256,
        embed_pdrop=0.1,
        resid_pdrop=0.1,
        attn_pdrop=0.1,
        action_space_size=6,
        group_size=8,
        task_num=1,
        env_num=4,
        support_size=101,
        num_simulations=25,
        game_segment_length=100,
        obs_type='image',
        observation_shape=(3, 64, 64),
        image_channel=3,
        device='cpu',
        continuous_action_space=False,
        use_new_cache_manager=use_new_cache,  # ✨ 关键配置
    )
    return config


def create_test_tokenizer(config):
    """创建测试用的 tokenizer"""
    tokenizer = Tokenizer(
        obs_type=config.obs_type,
        obs_shape=config.observation_shape,
        embedding_dim=config.embed_dim,
        use_norm=False,
        group_size=config.group_size,
        device=config.device,
    )
    return tokenizer


def test_old_cache_system():
    """测试旧的 cache 系统"""
    print("\n" + "="*70)
    print("测试 1: 旧 KV Cache 系统")
    print("="*70)

    config = create_test_config(use_new_cache=False)
    tokenizer = create_test_tokenizer(config)
    model = WorldModel(config, tokenizer)

    # 验证
    assert hasattr(model, 'use_new_cache_manager')
    assert model.use_new_cache_manager == False
    assert hasattr(model, 'past_kv_cache_init_infer_envs')
    assert hasattr(model, 'past_kv_cache_recurrent_infer')
    assert hasattr(model, 'keys_values_wm_list')
    assert not hasattr(model, 'kv_cache_manager')

    print("✓ 旧系统初始化成功")
    print(f"  - use_new_cache_manager: {model.use_new_cache_manager}")
    print(f"  - past_kv_cache_init_infer_envs: {len(model.past_kv_cache_init_infer_envs)} envs")
    print(f"  - past_kv_cache_recurrent_infer: initialized")

    # 测试 clear_caches
    model.clear_caches()
    print("✓ 旧系统 clear_caches() 执行成功")

    print("\n✅ 测试 1 通过: 旧系统工作正常\n")


def test_new_cache_system():
    """测试新的 cache 系统"""
    print("\n" + "="*70)
    print("测试 2: 新 KV Cache 系统")
    print("="*70)

    config = create_test_config(use_new_cache=True)
    tokenizer = create_test_tokenizer(config)
    model = WorldModel(config, tokenizer)

    # 验证
    assert hasattr(model, 'use_new_cache_manager')
    assert model.use_new_cache_manager == True
    assert hasattr(model, 'kv_cache_manager')
    assert hasattr(model.kv_cache_manager, 'init_pools')
    assert hasattr(model.kv_cache_manager, 'recur_pool')
    assert hasattr(model.kv_cache_manager, 'wm_pool')

    print("✓ 新系统初始化成功")
    print(f"  - use_new_cache_manager: {model.use_new_cache_manager}")
    print(f"  - kv_cache_manager: {model.kv_cache_manager}")
    print(f"  - init_pools: {len(model.kv_cache_manager.init_pools)} pools")

    # 获取统计信息
    stats = model.kv_cache_manager.get_stats_summary()
    print(f"✓ 统计信息获取成功:")
    print(f"  - stats_enabled: {stats['stats_enabled']}")
    print(f"  - init_pools: {len(stats['init_pools'])} envs")

    # 测试 clear_caches
    model.clear_caches()
    print("✓ 新系统 clear_caches() 执行成功")

    print("\n✅ 测试 2 通过: 新系统工作正常\n")


def test_cache_operations():
    """测试基本的 cache 操作"""
    print("\n" + "="*70)
    print("测试 3: 新系统 Cache 操作")
    print("="*70)

    config = create_test_config(use_new_cache=True)
    tokenizer = create_test_tokenizer(config)
    model = WorldModel(config, tokenizer)

    manager = model.kv_cache_manager

    # 创建测试用的 KeysValues 对象
    from lzero.model.unizero_world_models.kv_caching import KeysValues

    test_kv = KeysValues(
        num_samples=1,
        num_heads=4,
        max_tokens=20,
        embed_dim=256,
        num_layers=2,
        device=torch.device('cpu')
    )

    # 测试 set 和 get
    cache_key = 12345
    index = manager.set_init_cache(env_id=0, cache_key=cache_key, kv_cache=test_kv)
    print(f"✓ Set cache: env_id=0, cache_key={cache_key}, index={index}")

    retrieved = manager.get_init_cache(env_id=0, cache_key=cache_key)
    assert retrieved is not None
    assert retrieved is test_kv
    print(f"✓ Get cache: retrieved successfully")

    # 测试 cache miss
    missing = manager.get_init_cache(env_id=0, cache_key=99999)
    assert missing is None
    print(f"✓ Cache miss: returned None as expected")

    # 测试统计
    stats = manager.get_stats_summary()
    print(f"✓ Stats after operations:")
    print(f"  - {stats['init_pools']['env_0']}")

    print("\n✅ 测试 3 通过: Cache 操作正常\n")


def test_backward_compatibility():
    """测试向后兼容性"""
    print("\n" + "="*70)
    print("测试 4: 向后兼容性")
    print("="*70)

    config = create_test_config(use_new_cache=True)
    tokenizer = create_test_tokenizer(config)
    model = WorldModel(config, tokenizer)

    # 验证向后兼容的引用
    assert hasattr(model, 'keys_values_wm_list')
    assert hasattr(model, 'keys_values_wm_size_list')
    assert model.keys_values_wm_list is model.kv_cache_manager.keys_values_wm_list
    assert model.keys_values_wm_size_list is model.kv_cache_manager.keys_values_wm_size_list

    print("✓ 向后兼容引用验证通过")
    print(f"  - keys_values_wm_list: {type(model.keys_values_wm_list)}")
    print(f"  - keys_values_wm_size_list: {type(model.keys_values_wm_size_list)}")

    print("\n✅ 测试 4 通过: 向后兼容性正常\n")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*70)
    print("KV Cache 系统切换测试")
    print("="*70)

    try:
        test_old_cache_system()
        test_new_cache_system()
        test_cache_operations()
        test_backward_compatibility()

        print("\n" + "="*70)
        print("🎉 所有测试通过!")
        print("="*70)
        print("\n✅ Phase 1 集成成功:")
        print("  - 旧系统仍正常工作")
        print("  - 新系统可以通过配置启用")
        print("  - 新旧系统可以无缝切换")
        print("  - Cache 操作功能正常")
        print("  - 向后兼容性保持")
        print("\n下一步: 可以开始在实际训练中测试新系统")
        print("="*70 + "\n")

        return True

    except Exception as e:
        print("\n" + "="*70)
        print(f"❌ 测试失败: {e}")
        print("="*70)
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

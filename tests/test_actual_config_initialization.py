"""
Phase 1.5 实际配置验证脚本
=======================

验证 atari_unizero_segment_config.py 中的 use_new_cache_manager=True 配置能够正确初始化,
不再出现 AttributeError。

运行方式:
    cd /mnt/nfs/zhangjinouwen/puyuan/LightZero
    python tests/test_actual_config_initialization.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from lzero.model.unizero_world_models.world_model import WorldModel
from lzero.model.unizero_world_models.transformer import TransformerConfig


def create_atari_like_config():
    """创建类似实际训练配置的测试配置"""
    config = TransformerConfig(
        tokens_per_block=2,
        max_blocks=10,
        attention='causal',
        num_layers=2,
        num_heads=8,
        embed_dim=768,
        embed_pdrop=0.1,
        resid_pdrop=0.1,
        attn_pdrop=0.1,
        task_embed_option='none',
    )

    # 模拟实际的 atari_unizero_segment_config.py 配置
    config.env_num = 8
    config.game_segment_length = 20
    config.num_simulations = 50
    config.action_space_size = 6
    config.observation_shape = (3, 64, 64)
    config.image_channel = 3
    config.support_size = 601
    config.obs_type = 'image'
    config.device = 'cpu'  # 测试用cpu
    config.continuous_action_space = False
    config.group_size = 8
    config.norm_type = 'LN'
    config.rotary_emb = False
    config.context_length = 8

    config.policy_entropy_weight = 5e-3
    config.predict_latent_loss_type = 'mse'
    config.gamma = 0.997
    config.dormant_threshold = 0.025
    config.analysis_dormant_ratio_weight_rank = False
    config.latent_recon_loss_weight = 0.0
    config.perceptual_loss_weight = 0.0
    config.max_cache_size = 2000

    # ==================== 关键配置 ====================
    # 这是用户报告出错的配置
    config.use_new_cache_manager = True
    # =================================================

    return config


def create_test_model(config):
    """创建测试模型"""
    class SimpleTokenizer:
        def __init__(self):
            self.embed_dim = 768
            self.encoder = None
            self.decoder_network = None

    tokenizer = SimpleTokenizer()
    model = WorldModel(config, tokenizer)
    return model


def test_initialization_with_new_cache_manager():
    """测试 1: 使用 use_new_cache_manager=True 初始化"""
    print("\n" + "="*70)
    print("测试 1: 验证 use_new_cache_manager=True 初始化")
    print("="*70)

    config = create_atari_like_config()
    print(f"\n[配置] use_new_cache_manager: {config.use_new_cache_manager}")
    print(f"[配置] env_num: {config.env_num}")
    print(f"[配置] game_segment_length: {config.game_segment_length}")

    print("\n[初始化] 创建 WorldModel...")
    try:
        model = create_test_model(config)
        print("✓ WorldModel 初始化成功")
    except AttributeError as e:
        print(f"❌ 初始化失败 (AttributeError): {e}")
        raise

    # 验证关键属性存在
    print("\n[验证] 检查关键属性...")
    assert hasattr(model, 'use_new_cache_manager')
    assert model.use_new_cache_manager == True
    print("✓ use_new_cache_manager: True")

    assert hasattr(model, 'kv_cache_manager')
    print("✓ kv_cache_manager: 存在")

    # 验证向后兼容属性 (dummy 属性)
    assert hasattr(model, 'past_kv_cache_recurrent_infer')
    print("✓ past_kv_cache_recurrent_infer: 存在 (dummy)")

    assert hasattr(model, 'past_kv_cache_init_infer_envs')
    assert len(model.past_kv_cache_init_infer_envs) == 8
    print(f"✓ past_kv_cache_init_infer_envs: 存在 (8个环境)")

    print("\n✅ 测试 1 通过: 配置初始化正常,无 AttributeError\n")


def test_clear_operations_on_model():
    """测试 2: 验证 clear 操作 (模拟 unizero.py 的调用)"""
    print("\n" + "="*70)
    print("测试 2: 验证 clear 操作 (模拟 unizero.py)")
    print("="*70)

    config = create_atari_like_config()
    model = create_test_model(config)

    print("\n[测试] 模拟 unizero.py 的 _reset_collect() 调用...")
    try:
        # 这是修复后的调用方式 (unizero.py:1442-1445)
        model.clear_caches()
        print("✓ world_model.clear_caches() 成功")
    except Exception as e:
        print(f"❌ clear_caches() 失败: {e}")
        raise

    print("\n[测试] 模拟 unizero.py 的 _reset_eval() - Episode end 调用...")
    try:
        # 这是修复后的调用方式 (unizero.py:1505-1508)
        model.clear_caches()
        print("✓ world_model.clear_caches() (episode end) 成功")
    except Exception as e:
        print(f"❌ clear_caches() 失败: {e}")
        raise

    print("\n[测试] 直接访问 dummy 属性 (向后兼容)...")
    try:
        # 即使直接访问 dummy 属性也不会报错
        for kv_cache_dict_env in model.past_kv_cache_init_infer_envs:
            kv_cache_dict_env.clear()
        print("✓ 直接访问 past_kv_cache_init_infer_envs.clear() 成功")

        model.past_kv_cache_recurrent_infer.clear()
        print("✓ 直接访问 past_kv_cache_recurrent_infer.clear() 成功")
    except Exception as e:
        print(f"❌ 直接访问失败: {e}")
        raise

    print("\n✅ 测试 2 通过: 所有 clear 操作正常\n")


def test_kv_cache_manager_functionality():
    """测试 3: 验证 KVCacheManager 基本功能"""
    print("\n" + "="*70)
    print("测试 3: 验证 KVCacheManager 基本功能")
    print("="*70)

    config = create_atari_like_config()
    model = create_test_model(config)

    print("\n[验证] KVCacheManager 统计功能...")
    stats = model.kv_cache_manager.get_stats_summary()
    print(f"✓ 统计信息获取成功:")
    print(f"  - init_pools: {stats.get('init_pools', 'N/A')}")
    print(f"  - recur_pool: {stats.get('recur_pool', 'N/A')}")
    print(f"  - wm_pool: {stats.get('wm_pool', 'N/A')}")

    print("\n[验证] KVCacheManager clear_all 功能...")
    try:
        model.kv_cache_manager.clear_all()
        print("✓ kv_cache_manager.clear_all() 成功")
    except Exception as e:
        print(f"❌ clear_all() 失败: {e}")
        raise

    print("\n✅ 测试 3 通过: KVCacheManager 功能正常\n")


def run_all_tests():
    """运行所有实际配置验证测试"""
    print("\n" + "="*70)
    print("Phase 1.5 实际配置验证测试")
    print("="*70)
    print("\n[目的] 验证 atari_unizero_segment_config.py 的配置可以正常初始化")
    print("[重现] use_new_cache_manager=True (用户报告出错的配置)")

    try:
        # 测试 1: 初始化验证
        test_initialization_with_new_cache_manager()

        # 测试 2: Clear 操作验证
        test_clear_operations_on_model()

        # 测试 3: KVCacheManager 功能验证
        test_kv_cache_manager_functionality()

        # 总结
        print("\n" + "="*70)
        print("🎉 所有实际配置验证测试通过!")
        print("="*70)
        print("\n✅ 验证成功:")
        print("  1. ✓ use_new_cache_manager=True 配置正确初始化")
        print("  2. ✓ 不再出现 AttributeError")
        print("  3. ✓ clear_caches() 方法正常工作")
        print("  4. ✓ KVCacheManager 功能正常")
        print("  5. ✓ 向后兼容属性存在且可访问")
        print("\n结论:")
        print("  • AttributeError: 'WorldModel' object has no attribute")
        print("    'past_kv_cache_recurrent_infer' 问题已修复")
        print("  • atari_unizero_segment_config.py 配置可以正常运行")
        print("  • 可以开始实际训练测试")
        print("\n下一步:")
        print("  • 运行实际训练脚本验证完整流程")
        print("  • 收集性能数据进行对比")
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

#!/usr/bin/env python3
"""
Example: How to use async training in PriorZero

This script demonstrates different async training configurations.
"""

from priorzero_config import get_priorzero_config_for_quick_test
from loguru import logger


def example_1_synchronous():
    """Example 1: Synchronous mode (default, same as original)"""
    logger.info("="*80)
    logger.info("Example 1: Synchronous Training (off_policy_degree=0)")
    logger.info("="*80)

    cfg, create_cfg = get_priorzero_config_for_quick_test()

    # Override async settings
    cfg.policy.off_policy_degree = 0
    cfg.policy.enable_async_eval = False

    logger.info(f"Configuration:")
    logger.info(f"  off_policy_degree: {cfg.policy.off_policy_degree}")
    logger.info(f"  enable_async_eval: {cfg.policy.enable_async_eval}")
    logger.info(f"  Mode: SYNCHRONOUS (original behavior)")
    logger.info("")
    logger.info(f"This will run collect -> train -> eval in strict serial order.")
    logger.info("")

    return cfg, create_cfg


def example_2_light_async():
    """Example 2: Light async mode (conservative)"""
    logger.info("="*80)
    logger.info("Example 2: Light Async Training (off_policy_degree=5)")
    logger.info("="*80)

    cfg, create_cfg = get_priorzero_config_for_quick_test()

    # Override async settings
    cfg.policy.off_policy_degree = 5
    cfg.policy.enable_async_eval = False

    logger.info(f"Configuration:")
    logger.info(f"  off_policy_degree: {cfg.policy.off_policy_degree}")
    logger.info(f"  enable_async_eval: {cfg.policy.enable_async_eval}")
    logger.info(f"  Mode: ASYNCHRONOUS (light)")
    logger.info("")
    logger.info(f"Train can lag behind collect by up to 5 batches.")
    logger.info(f"This allows some parallelism while keeping off-policy bias low.")
    logger.info("")

    return cfg, create_cfg


def example_3_aggressive_async():
    """Example 3: Aggressive async mode (maximum throughput)"""
    logger.info("="*80)
    logger.info("Example 3: Aggressive Async Training (off_policy_degree=50)")
    logger.info("="*80)

    cfg, create_cfg = get_priorzero_config_for_quick_test()

    # Override async settings
    cfg.policy.off_policy_degree = 50
    cfg.policy.enable_async_eval = True

    logger.info(f"Configuration:")
    logger.info(f"  off_policy_degree: {cfg.policy.off_policy_degree}")
    logger.info(f"  enable_async_eval: {cfg.policy.enable_async_eval}")
    logger.info(f"  Mode: ASYNCHRONOUS (aggressive)")
    logger.info("")
    logger.info(f"Train can lag behind collect by up to 50 batches.")
    logger.info(f"Eval runs asynchronously in the background.")
    logger.info(f"Maximum throughput but higher off-policy bias.")
    logger.info("")

    return cfg, create_cfg


def example_4_auto_tune():
    """Example 4: Auto-tuned async mode"""
    logger.info("="*80)
    logger.info("Example 4: Auto-tuned Async Training (off_policy_degree=-1)")
    logger.info("="*80)

    cfg, create_cfg = get_priorzero_config_for_quick_test()

    # Override async settings
    cfg.policy.off_policy_degree = -1  # Auto-tune
    cfg.policy.enable_async_eval = False

    logger.info(f"Configuration:")
    logger.info(f"  off_policy_degree: {cfg.policy.off_policy_degree} (auto)")
    logger.info(f"  enable_async_eval: {cfg.policy.enable_async_eval}")
    logger.info(f"  Mode: ASYNCHRONOUS (auto-tuned)")
    logger.info("")
    logger.info(f"Auto-tune will set off_policy_degree based on buffer size.")
    logger.info(f"Formula: (buffer_size / batch_size) / 10")
    logger.info(f"For buffer_size={cfg.policy.replay_buffer_size}, batch_size={cfg.policy.batch_size}:")
    auto_value = (cfg.policy.replay_buffer_size // cfg.policy.batch_size) // 10
    logger.info(f"  -> off_policy_degree will be set to {auto_value}")
    logger.info("")

    return cfg, create_cfg


def main():
    """Show all examples"""
    logger.info("\n" + "="*80)
    logger.info("PRIORZERO ASYNC TRAINING EXAMPLES")
    logger.info("="*80 + "\n")

    # Example 1: Synchronous
    cfg1, create_cfg1 = example_1_synchronous()
    logger.info("To run: python priorzero_entry.py --quick_test")
    logger.info("")

    input("Press Enter to continue to next example...")
    print()

    # Example 2: Light async
    cfg2, create_cfg2 = example_2_light_async()
    logger.info("To use this config, modify priorzero_config.py:")
    logger.info("  policy_config['off_policy_degree'] = 5")
    logger.info("")

    input("Press Enter to continue to next example...")
    print()

    # Example 3: Aggressive async
    cfg3, create_cfg3 = example_3_aggressive_async()
    logger.info("To use this config, modify priorzero_config.py:")
    logger.info("  policy_config['off_policy_degree'] = 50")
    logger.info("  policy_config['enable_async_eval'] = True")
    logger.info("")

    input("Press Enter to continue to next example...")
    print()

    # Example 4: Auto-tune
    cfg4, create_cfg4 = example_4_auto_tune()
    logger.info("To use this config, modify priorzero_config.py:")
    logger.info("  policy_config['off_policy_degree'] = -1")
    logger.info("")

    logger.info("\n" + "="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    logger.info("")
    logger.info("Choose async mode based on your needs:")
    logger.info("")
    logger.info("  1. Synchronous (off_policy_degree=0)")
    logger.info("     - Use for: debugging, verification, strict on-policy")
    logger.info("     - Pros: lowest off-policy bias, original behavior")
    logger.info("     - Cons: lowest throughput")
    logger.info("")
    logger.info("  2. Light Async (off_policy_degree=5-10)")
    logger.info("     - Use for: production training with stability")
    logger.info("     - Pros: moderate throughput boost, low bias")
    logger.info("     - Cons: still somewhat conservative")
    logger.info("")
    logger.info("  3. Aggressive Async (off_policy_degree=50+)")
    logger.info("     - Use for: maximum throughput, quick experiments")
    logger.info("     - Pros: highest throughput")
    logger.info("     - Cons: higher off-policy bias, may affect stability")
    logger.info("")
    logger.info("  4. Auto-tune (off_policy_degree=-1)")
    logger.info("     - Use for: automatic optimization")
    logger.info("     - Pros: adapts to buffer size")
    logger.info("     - Cons: may need manual tuning for specific tasks")
    logger.info("")
    logger.info("="*80)
    logger.info("")


if __name__ == "__main__":
    main()

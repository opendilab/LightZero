# priorzero_entry.py
"""
[PRIORZERO] Main Training Entry Point

This module provides the main async training loop for PriorZero.

Key Features:
- Async training with vLLM integration
- Checkpoint management and recovery
- Comprehensive logging (TensorBoard + file logs)
- Graceful error handling

Author: PriorZero Team
Date: 2025-01-20
"""

import asyncio
import os
import sys
from functools import partial
from pathlib import Path

# ==============================================================================
# [CRITICAL] Ensure local LightZero is used for PriorZero-specific adaptations
# ==============================================================================
from ensure_local_lightzero import ensure_local_lightzero
ensure_local_lightzero()


import ray
import torch
import wandb
from ding.config import compile_config
from ding.envs import create_env_manager, get_vec_env_setting
from ding.policy import create_policy
from ding.utils import set_pkg_seed, get_rank
from ding.worker import create_buffer, BaseLearner
from tensorboardX import SummaryWriter
from loguru import logger
from vllm import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs

# Import PriorZero components
from priorzero_config import get_priorzero_config, get_priorzero_config_for_quick_test
from priorzero_collector import PriorZeroCollector
from priorzero_evaluator import PriorZeroEvaluator
# Import policy to ensure registration happens
import priorzero_policy  # noqa: F401


async def train_priorzero(
    cfg: dict,
    create_cfg: dict,
    seed: int = 0,
    max_train_iter: int = int(1e6),
    enable_save: bool = True
):
    """
    [PRIORZERO-MODIFIED]
    Main async training function for PriorZero.

    Args:
        cfg: Main configuration dictionary
        create_cfg: Creation configuration for DI-engine components
        seed: Random seed
        max_train_iter: Maximum training iterations
        enable_save: Whether to save checkpoints
    """
    # ==================================================================
    # 1. Compile Configuration
    # ==================================================================
    cfg = compile_config(cfg, seed=seed, auto=True, create_cfg=create_cfg)

    # ==================================================================
    # 2. Initialize Ray (for distributed vLLM)
    # ==================================================================
    # Note: vLLM will initialize Ray internally if needed.
    # We skip manual Ray initialization to avoid conflicts with existing clusters.
    if ray.is_initialized():
        logger.info(f"✓ Ray already initialized (connected to existing cluster)")
    else:
        logger.info(f"✓ Ray not initialized - vLLM will handle initialization if needed")

    # ==================================================================
    # 3. Create vLLM Engine
    # ==================================================================
    logger.info("Creating vLLM engine...")

    # [ROBUST FIX] Handle shared GPU environment
    # Issue: vLLM V1 engine fails when other processes release GPU memory during init
    # Solution: Use alternative initialization method that bypasses V1 checks
    import os

    # Note: In vLLM>=0.3.0, worker_use_ray is replaced by distributed_executor_backend
    # For single GPU: use "mp" (multiprocessing)
    # For multi-GPU: use "ray" if available
    tensor_parallel = cfg.policy.llm_policy_cfg.vllm_tensor_parallel_size
    distributed_backend = "ray" if tensor_parallel > 1 and ray.is_initialized() else None

    # [ROBUST FIX] Lower GPU memory utilization in shared environment
    # This leaves more headroom for memory fluctuations
    gpu_mem_util = cfg.policy.llm_policy_cfg.gpu_memory_utilization
    if gpu_mem_util > 0.85:
        gpu_mem_util = 0.75  # More conservative in shared environment
        logger.info(f"✓ Adjusted GPU memory utilization to {gpu_mem_util} for stability")

    # [ROBUST FIX] Use alternative initialization to avoid V1 engine issues
    # Set env var BEFORE importing to ensure it takes effect
    use_v1_env = os.environ.get('VLLM_USE_V1', None)
    if use_v1_env is None:
        # Only set if not already set by user
        os.environ['VLLM_USE_V1'] = '0'
        logger.info("✓ Using vLLM V0 engine for stability in shared GPU environment")

    try:
        engine_args = AsyncEngineArgs(
            model=cfg.policy.llm_policy_cfg.pretrain_llm_path,
            tensor_parallel_size=tensor_parallel,
            gpu_memory_utilization=gpu_mem_util,
            distributed_executor_backend=distributed_backend,
            trust_remote_code=True,
            # [ROBUST FIX] Disable prefix caching in shared environment to reduce memory complexity
            enable_prefix_caching=False,
            # [ROBUST FIX] Disable enforce_eager to avoid memory profiling issues
            enforce_eager=False,
        )
        vllm_engine = AsyncLLMEngine.from_engine_args(engine_args)
        logger.info(f"✓ vLLM Engine created (backend: {distributed_backend or 'default'})")
    except (ValueError, RuntimeError) as e:
        if "VLLM_USE_V1" in str(e) or "memory profiling" in str(e):
            # Fallback: Try without V1 env var
            logger.warning(f"⚠️  Initial vLLM initialization failed: {e}")
            logger.info("Retrying with alternative configuration...")
            if 'VLLM_USE_V1' in os.environ:
                del os.environ['VLLM_USE_V1']

            engine_args = AsyncEngineArgs(
                model=cfg.policy.llm_policy_cfg.pretrain_llm_path,
                tensor_parallel_size=tensor_parallel,
                gpu_memory_utilization=gpu_mem_util * 0.9,  # Even more conservative
                distributed_executor_backend=distributed_backend,
                trust_remote_code=True,
                enable_prefix_caching=False,
                enforce_eager=True,  # Force eager mode as fallback
            )
            vllm_engine = AsyncLLMEngine.from_engine_args(engine_args)
            logger.info(f"✓ vLLM Engine created with fallback configuration")
        else:
            raise

    # ==================================================================
    # 4. Create Environments
    # ==================================================================
    logger.info("Creating environments...")
    env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)
    collector_env = create_env_manager(
        cfg.env.manager,
        [partial(env_fn, cfg=c) for c in collector_env_cfg]
    )
    evaluator_env = create_env_manager(
        cfg.env.manager,
        [partial(env_fn, cfg=c) for c in evaluator_env_cfg]
    )

    # Seed environments
    collector_env.seed(seed)
    evaluator_env.seed(seed, dynamic_seed=False)
    set_pkg_seed(seed, use_cuda=True)
    logger.info(f"✓ Environments created and seeded (seed={seed})")

    # ==================================================================
    # 5. Create Policy, Buffer, and Components
    # ==================================================================
    logger.info("Creating policy, buffer, and components...")

    # Create policy (align with UniZero)
    policy = create_policy(
        cfg.policy,
        enable_field=['learn', 'collect', 'eval']
    )
    logger.info("✓ Policy created")

    # Create TensorBoard logger (align with UniZero)
    os.makedirs(f'./{cfg.exp_name}/log/', exist_ok=True)
    tb_logger = SummaryWriter(os.path.join(f'./{cfg.exp_name}/log/', 'serial')) if get_rank() == 0 else None
    logger.info(f"✓ TensorBoard logger: ./{cfg.exp_name}/log/")

    # Create learner (align with UniZero - this sets up policy._logger)
    learner = BaseLearner(
        cfg.policy.learn.learner,
        policy.learn_mode,
        tb_logger,
        exp_name=cfg.exp_name
    )
    logger.info("✓ BaseLearner created")

    # Create replay buffer (align with UniZero - use GameBuffer from policy type)
    from lzero.mcts import UniZeroGameBuffer
    replay_buffer = UniZeroGameBuffer(cfg.policy)
    logger.info("✓ Replay buffer created")

    # Create collector
    collector = PriorZeroCollector(
        env=collector_env,
        policy=policy.collect_mode,
        tb_logger=tb_logger,
        exp_name=cfg.exp_name,
        vllm_engine=vllm_engine,
        policy_config=cfg.policy,
    )
    logger.info("✓ Collector created")

    # Create evaluator
    evaluator = PriorZeroEvaluator(
        eval_freq=cfg.policy.eval_freq,
        n_evaluator_episode=cfg.env.n_evaluator_episode,
        stop_value=cfg.env.stop_value,
        env=evaluator_env,
        policy=policy.eval_mode,
        tb_logger=tb_logger,
        exp_name=cfg.exp_name,
        vllm_engine=vllm_engine,
    )
    logger.info("✓ Evaluator created")

    # Initialize WandB if enabled (PriorZero enhancement)
    if cfg.policy.get('use_wandb', False):
        if get_rank() == 0:
            wandb.init(
                project=cfg.policy.get('wandb_project', 'priorzero'),
                name=cfg.exp_name,
                config=cfg,
                tags=['priorzero', 'unizero', 'llm-policy'],
            )
            logger.info("✓ WandB initialized")
        # Set train iter and env step for policy wandb logging
        policy.set_train_iter_env_step(learner.train_iter, collector.envstep)

    # Call learner's before_run hook (align with UniZero)
    learner.call_hook('before_run')

    # ==================================================================
    # 6. Main Training Loop
    # ==================================================================
    logger.info("="*80)
    logger.info("Starting PriorZero Training")
    logger.info("="*80)
    logger.info(f"Experiment: {cfg.exp_name}")
    logger.info(f"Max iterations: {max_train_iter}")
    logger.info(f"Batch size: {cfg.policy.batch_size}")
    logger.info(f"LLM model: {cfg.policy.llm_policy_cfg.pretrain_llm_path}")
    logger.info(f"World model layers: {cfg.policy.model.world_model_cfg.num_layers}")
    logger.info("="*80)

    train_iter = 0
    best_eval_reward = -float('inf')

    try:
        while train_iter < max_train_iter:
            # ============================================================
            # Collect Data
            # ============================================================
            logger.info(f"\n[Iter {train_iter}] Collecting data...")
            collect_kwargs = {}  # Can add temperature, epsilon, etc.

            new_data = await collector.collect(
                train_iter=train_iter,
                policy_kwargs=collect_kwargs
            )

            # Push to replay buffer
            replay_buffer.push(new_data, cur_collector_envstep=collector.envstep)
            logger.info(f"  ✓ Data collected, buffer size: {len(replay_buffer)}")

            # ============================================================
            # Training
            # ============================================================
            if train_iter >= cfg.policy.train_start_after_envsteps:
                logger.info(f"[Iter {train_iter}] Training...")

                # Determine number of updates
                if cfg.policy.update_per_collect is None:
                    # Auto-calculate based on collected data
                    num_updates = max(
                        1,
                        int(collector.envstep * cfg.policy.replay_ratio)
                    )
                else:
                    num_updates = cfg.policy.update_per_collect

                for update_idx in range(num_updates):
                    # Sample batch
                    train_data = replay_buffer.sample(cfg.policy.batch_size)
                    if train_data is None:
                        logger.warning("  ⚠ No data to sample, skipping update")
                        break

                    # Training step (includes game_segments for LLM training)
                    if hasattr(train_data, 'game_segments'):
                        train_data_with_segments = (
                            *train_data,
                            train_iter,
                            train_data.game_segments
                        )
                    else:
                        logger.warning("  ⚠ No game_segments in train_data")
                        train_data_with_segments = (*train_data, train_iter, [])

                    log_vars = policy.learn(data=train_data_with_segments)

                    # Log to TensorBoard
                    for k, v in log_vars.items():
                        tb_logger.add_scalar(f'train/{k}', v, train_iter)

                    # Periodic logging
                    if update_idx % 10 == 0:
                        logger.info(
                            f"  Update {update_idx}/{num_updates}: "
                            f"total_loss={log_vars.get('total_loss', 0):.4f}, "
                            f"wm_loss={log_vars.get('wm_total_loss', 0):.4f}, "
                            f"llm_loss={log_vars.get('llm_total_loss', 0):.4f}"
                        )

                    train_iter += 1

                    # Check max iterations
                    if train_iter >= max_train_iter:
                        break

            # ============================================================
            # Evaluation
            # ============================================================
            if evaluator.should_eval(train_iter):
                logger.info(f"\n[Iter {train_iter}] Evaluating...")

                stop, eval_reward_dict = await evaluator.eval(
                    save_ckpt_fn=policy.save if enable_save else None,
                    train_iter=train_iter,
                    envstep=collector.envstep
                )

                mean_reward = eval_reward_dict.get('reward_mean', 0)
                logger.info(f"  ✓ Evaluation done: reward_mean={mean_reward:.2f}")

                # Save best model
                if mean_reward > best_eval_reward and enable_save:
                    best_eval_reward = mean_reward
                    ckpt_path = os.path.join(cfg.exp_name, 'ckpt_best.pth.tar')
                    policy.save(ckpt_path)
                    logger.info(f"  🏆 New best model saved: {ckpt_path}")

                # Check stop condition
                if stop:
                    logger.info(f"  🎉 Training converged! (reward >= {cfg.env.stop_value})")
                    break

    except KeyboardInterrupt:
        logger.warning("\n⚠ Training interrupted by user (Ctrl+C)")

    except Exception as e:
        logger.error(f"\n✗ Training error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # ============================================================
        # Cleanup
        # ============================================================
        logger.info("\nCleaning up...")
        collector_env.close()
        evaluator_env.close()
        tb_logger.close()

        logger.info("="*80)
        logger.info("Training Complete!")
        logger.info(f"Total iterations: {train_iter}")
        logger.info(f"Best eval reward: {best_eval_reward:.2f}")
        logger.info("="*80)


def main():
    """
    Main entry point with argument parsing.
    """
    import argparse

    parser = argparse.ArgumentParser(description='PriorZero Training')
    parser.add_argument('--env_id', type=str, default='zork1.z5', help='Jericho game ID')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--max_iter', type=int, default=int(1e6), help='Max training iterations')
    parser.add_argument('--quick_test', action='store_true', help='Use quick test config')
    parser.add_argument('--no_save', action='store_true', help='Disable checkpoint saving')

    args = parser.parse_args()

    # Get configuration
    if args.quick_test:
        logger.info("Using quick test configuration")
        main_cfg, create_cfg = get_priorzero_config_for_quick_test(args.env_id, args.seed)
    else:
        main_cfg, create_cfg = get_priorzero_config(args.env_id, args.seed)

    # Run training
    asyncio.run(train_priorzero(
        main_cfg,
        create_cfg,
        seed=args.seed,
        max_train_iter=args.max_iter,
        enable_save=not args.no_save
    ))


if __name__ == "__main__":
    import os
    # Disable tokenizer parallelism to prevent multi-process conflicts
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    main()

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
from typing import Tuple, Optional
# from lzero.entry.utils import log_buffer_memory_usage
# from lzero.policy import visit_count_temperature
# from ding.rl_utils import get_epsilon_greedy_fn

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
    max_env_step: Optional[int] = int(1e10),
    enable_save: bool = True,
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

    # [PRIORZERO-MODIFIED] Create PriorZero-specific replay buffer
    # This buffer returns game_segments for LLM training (SFT/RFT)
    from lzero.mcts.buffer.game_buffer_priorzero import PriorZeroGameBufferOptimized
    replay_buffer = PriorZeroGameBufferOptimized(cfg.policy)
    logger.info("✓ PriorZero replay buffer created (with game_segments support)")

    # Create collector
    collector = PriorZeroCollector(
        env=collector_env,
        policy=policy.collect_mode,
        tb_logger=tb_logger,
        exp_name=cfg.exp_name,
        vllm_engine=vllm_engine,
        policy_config=cfg.policy,
        debug_mode=cfg.get('debug_mode', False),
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
    if cfg.policy.get('use_wandb', True):
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

    # [ALIGN WITH UNIZERO] Initialize reanalyze-related counters (train_unizero_segment.py line 119-121)
    buffer_reanalyze_count = 0
    train_epoch = 0
    reanalyze_batch_size = cfg.policy.reanalyze_batch_size
    batch_size = cfg.policy.batch_size
    best_eval_reward = -float('inf')
    policy_config = cfg.policy

    try:
        while True:
            # # Log buffer memory usage
            # log_buffer_memory_usage(learner.train_iter, replay_buffer, tb_logger)

            # # Set temperature for visit count distributions
            # collect_kwargs = {
            #     'temperature': visit_count_temperature(
            #         policy_config.manual_temperature_decay,
            #         policy_config.fixed_temperature_value,
            #         policy_config.threshold_training_steps_for_final_temperature,
            #         trained_steps=learner.train_iter
            #     ),
            #     'epsilon': 0.0  # Default epsilon value
            # }

            # # Configure epsilon for epsilon-greedy exploration
            # if policy_config.eps.eps_greedy_exploration_in_collect:
            #     epsilon_greedy_fn = get_epsilon_greedy_fn(
            #         start=policy_config.eps.start,
            #         end=policy_config.eps.end,
            #         decay=policy_config.eps.decay,
            #         type_=policy_config.eps.type
            #     )
            #     collect_kwargs['epsilon'] = epsilon_greedy_fn(collector.envstep)

            collect_kwargs = {
                'temperature': 0.25,
                'epsilon': 0.0  # Default epsilon value
            }

            # ============================================================
            # Evaluation (align with train_unizero_segment.py line 158-162)
            # ============================================================
            if learner.train_iter > 0 and evaluator.should_eval(learner.train_iter):
                logger.info(f"\n[Iter {learner.train_iter}] Evaluating...")

                stop, eval_reward_dict = await evaluator.eval(
                    save_ckpt_fn=learner.save_checkpoint if enable_save else None,
                    train_iter=learner.train_iter,
                    envstep=collector.envstep
                )

                mean_reward = eval_reward_dict.get('reward_mean', 0)
                logger.info(f"  ✓ Evaluation done: reward_mean={mean_reward:.2f}")

                if stop:
                    logger.info(f"  🎉 Training converged! (reward >= {cfg.env.stop_value})")
                    break

            # ============================================================
            # Collect Data (align with train_unizero_segment.py line 165)
            # ============================================================
            logger.info(f"\n[Iter {learner.train_iter}] Collecting data...")

            new_data = await collector.collect(
                train_iter=learner.train_iter,
                policy_kwargs=collect_kwargs
            )

            # Determine updates per collection (align with train_unizero_segment.py line 168)
            from lzero.entry.utils import calculate_update_per_collect
            update_per_collect = calculate_update_per_collect(cfg, new_data, world_size=1)

            # Update replay buffer (align with train_unizero_segment.py line 171-172)
            replay_buffer.push_game_segments(new_data)
            replay_buffer.remove_oldest_data_to_fit()
            # [FIX] Use get_num_of_transitions() instead of len()
            buffer_size = replay_buffer.get_num_of_transitions() if hasattr(replay_buffer, 'get_num_of_transitions') else 0
            logger.info(f"  ✓ Data collected, buffer size: {buffer_size} transitions")

            # ============================================================
            # Periodically reanalyze buffer (align with train_unizero_segment.py line 175-186)
            # ============================================================
            if cfg.policy.buffer_reanalyze_freq >= 1:
                # Reanalyze buffer <buffer_reanalyze_freq> times in one train_epoch
                reanalyze_interval = update_per_collect // cfg.policy.buffer_reanalyze_freq
            else:
                # Reanalyze buffer each <1/buffer_reanalyze_freq> train_epoch
                if train_epoch > 0 and train_epoch % int(1/cfg.policy.buffer_reanalyze_freq) == 0 and replay_buffer.get_num_of_transitions()//cfg.policy.num_unroll_steps > int(reanalyze_batch_size/cfg.policy.reanalyze_partition):
                    logger.info(f"[Reanalyze] Starting buffer reanalysis...")
                    replay_buffer.reanalyze_buffer(reanalyze_batch_size, policy)
                    buffer_reanalyze_count += 1
                    logger.info(f"  ✓ Buffer reanalyze count: {buffer_reanalyze_count}")

            # ============================================================
            # Training (align with train_unizero_segment.py line 189-221)
            # ============================================================
            if collector.envstep > cfg.policy.train_start_after_envsteps:
                # Check if there is sufficient data for training
                if cfg.policy.sample_type == 'episode':
                    data_sufficient = replay_buffer.get_num_of_game_segments() > batch_size
                else:
                    data_sufficient = replay_buffer.get_num_of_transitions() > batch_size

                if not data_sufficient:
                    logger.warning(
                        f'  ⚠ Data in replay_buffer is not sufficient: '
                        f'batch_size: {batch_size}, replay_buffer: {replay_buffer}. Continue to collect...'
                    )
                    continue

                logger.info(f"[Iter {learner.train_iter}] Training...")

                for i in range(update_per_collect):
                    # Reanalyze buffer during training (align with train_unizero_segment.py line 202-210)
                    if cfg.policy.buffer_reanalyze_freq >= 1:
                        if i % reanalyze_interval == 0 and replay_buffer.get_num_of_transitions()//cfg.policy.num_unroll_steps > int(reanalyze_batch_size/cfg.policy.reanalyze_partition):
                            logger.info(f"[Reanalyze] Starting buffer reanalysis (update {i}/{update_per_collect})...")
                            replay_buffer.reanalyze_buffer(reanalyze_batch_size, policy)
                            buffer_reanalyze_count += 1
                            logger.info(f"  ✓ Buffer reanalyze count: {buffer_reanalyze_count}")

                    # Sample batch (align with train_unizero_segment.py line 212)
                    # [PRIORZERO-KEY] PriorZeroGameBuffer returns [current_batch, target_batch, game_segments]
                    train_data = replay_buffer.sample(batch_size, policy)

                    # [PRIORZERO-KEY] Insert train_iter at index 2 (before game_segments)
                    # Policy expects: (current_batch, target_batch, train_iter, game_segments)
                    # Buffer returns: [current_batch, target_batch, game_segments]
                    # After insert(2, train_iter): [current_batch, target_batch, train_iter, game_segments]
                    train_data.insert(2, learner.train_iter)

                    # Train (align with train_unizero_segment.py line 217)
                    # Policy will receive: (current_batch, target_batch, train_iter, game_segments)
                    log_vars = learner.train(train_data, collector.envstep)

                    # Update priority if enabled (align with train_unizero_segment.py line 219-220)
                    if cfg.policy.use_priority:
                        replay_buffer.update_priority(train_data, log_vars[0]['value_priority_orig'])

            # Increment epoch counter (align with train_unizero_segment.py line 222)
            train_epoch += 1
            # Note: KV cache is cleared BEFORE collection (see line 298), not after epoch

            # [FIX] Clear KV cache BEFORE collection to prevent index overflow during MCTS
            policy.recompute_pos_emb_diff_and_clear_cache()

            # ============================================================
            # Check stopping criteria (align with train_unizero_segment.py line 226-227)
            # ============================================================
            if collector.envstep >= max_env_step or learner.train_iter >= max_train_iter:
                logger.info("Stopping condition met, training ends!")
                break

    except KeyboardInterrupt:
        logger.warning("\n⚠ Training interrupted by user (Ctrl+C)")

    except Exception as e:
        logger.error(f"\n✗ Training error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # ============================================================
        # Cleanup (align with train_unizero_segment.py line 229)
        # ============================================================
        learner.call_hook('after_run')

        logger.info("\nCleaning up...")
        collector_env.close()
        evaluator_env.close()
        tb_logger.close()

        logger.info("="*80)
        logger.info("Training Complete!")
        logger.info(f"Total iterations: {learner.train_iter}")
        logger.info(f"Best eval reward: {best_eval_reward:.2f}")
        logger.info("="*80)

    return policy


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
    parser.add_argument('--debug', action='store_true', help='Enable detailed debug logging (obs, action, LLM output)')

    args = parser.parse_args()

    # args.quick_test = True # ONLY FOR DEBUG

    # Get configuration
    if args.quick_test:
        logger.info("Using quick test configuration")
        main_cfg, create_cfg = get_priorzero_config_for_quick_test(args.env_id, args.seed, debug_mode=args.debug)
    else:
        main_cfg, create_cfg = get_priorzero_config(args.env_id, args.seed, debug_mode=args.debug)

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

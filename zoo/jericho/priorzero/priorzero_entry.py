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
from priorzero_config import get_priorzero_config, get_priorzero_debug_config
from priorzero_collector import PriorZeroCollector
from priorzero_evaluator import PriorZeroEvaluator
# Import policy to ensure registration happens
import priorzero_policy  # noqa: F401
from lzero.mcts.buffer.game_buffer_priorzero import PriorZeroGameBufferOptimized


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
    cfg = compile_config(cfg, seed=seed, auto=True, create_cfg=create_cfg)
    if ray.is_initialized():
        logger.info(f"✓ Ray already initialized (connected to existing cluster)")
    else:
        logger.info(f"✓ Ray not initialized - vLLM will handle initialization if needed")

    logger.info("Creating vLLM engine...")
    tensor_parallel = cfg.policy.llm_policy_cfg.vllm_tensor_parallel_size
    distributed_backend = "ray" if tensor_parallel > 1 and ray.is_initialized() else None

    gpu_mem_util = cfg.policy.llm_policy_cfg.gpu_memory_utilization
    if gpu_mem_util > 0.85:
        gpu_mem_util = 0.75  
        logger.info(f"✓ Adjusted GPU memory utilization to {gpu_mem_util} for stability")

    use_v1_env = os.environ.get('VLLM_USE_V1', None)
    if use_v1_env is None:
        os.environ['VLLM_USE_V1'] = '0'
        logger.info("✓ Using vLLM V0 engine for stability in shared GPU environment")

    try:
        engine_args = AsyncEngineArgs(
            model=cfg.policy.llm_policy_cfg.pretrain_llm_path,
            tensor_parallel_size=tensor_parallel,
            gpu_memory_utilization=gpu_mem_util,
            distributed_executor_backend=distributed_backend,
            trust_remote_code=True,
            enable_prefix_caching=False,
            enforce_eager=False,
        )
        vllm_engine = AsyncLLMEngine.from_engine_args(engine_args)
        logger.info(f"✓ vLLM Engine created (backend: {distributed_backend or 'default'})")
    except (ValueError, RuntimeError) as e:
        if "VLLM_USE_V1" in str(e) or "memory profiling" in str(e):
            logger.warning(f"⚠️  Initial vLLM initialization failed: {e}")
            logger.info("Retrying with alternative configuration...")
            if 'VLLM_USE_V1' in os.environ:
                del os.environ['VLLM_USE_V1']

            engine_args = AsyncEngineArgs(
                model=cfg.policy.llm_policy_cfg.pretrain_llm_path,
                tensor_parallel_size=tensor_parallel,
                gpu_memory_utilization=gpu_mem_util * 0.7,  # Even more conservative
                distributed_executor_backend=distributed_backend,
                trust_remote_code=True,
                enable_prefix_caching=False, 
                enforce_eager=True,  # Force eager mode as fallback
            )
            vllm_engine = AsyncLLMEngine.from_engine_args(engine_args)
            logger.info(f"✓ vLLM Engine created with fallback configuration")
        else:
            raise

    logger.info("Creating environments...")
    env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)
    collector_env = create_env_manager( cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_env_cfg])
    evaluator_env = create_env_manager( cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])

    collector_env.seed(seed)
    evaluator_env.seed(seed, dynamic_seed=False)
    set_pkg_seed(seed, use_cuda=True)

    logger.info("Creating policy, buffer, and components...")
    policy = create_policy( cfg.policy, enable_field=['learn', 'collect', 'eval'], exp_name=cfg.exp_name)
    logger.info("✓ Policy created")

    os.makedirs(f'./{cfg.exp_name}/log/', exist_ok=True)
    tb_logger = SummaryWriter(os.path.join(f'./{cfg.exp_name}/log/', 'serial')) if get_rank() == 0 else None
    logger.info(f"✓ TensorBoard logger: ./{cfg.exp_name}/log/")

    learner = BaseLearner(
        cfg.policy.learn.learner,
        policy.learn_mode,
        tb_logger,
        exp_name=cfg.exp_name
    )
    logger.info("✓ BaseLearner created")

    
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
        policy_config=cfg.policy,
    )
    logger.info("✓ Evaluator created")
    learner.call_hook('before_run')

    from async_training_coordinator import AsyncTrainingCoordinator

    coordinator = AsyncTrainingCoordinator(
        off_policy_degree=cfg.policy.off_policy_degree,
        enable_async_eval=cfg.policy.enable_async_eval,
        buffer_size=cfg.policy.replay_buffer_size,
        batch_size=cfg.policy.batch_size,
    )

    # ==================================================================
    # Main Training Loop
    # ==================================================================
    logger.info("="*80)
    logger.info("Starting PriorZero Training")
    logger.info("="*80)
    logger.info(f"Experiment: {cfg.exp_name}")
    logger.info(f"Max iterations: {max_train_iter}")
    logger.info(f"Batch size: {cfg.policy.batch_size}")
    logger.info(f"LLM model: {cfg.policy.llm_policy_cfg.pretrain_llm_path}")
    logger.info(f"World model layers: {cfg.policy.model.world_model_cfg.num_layers}")
    logger.info(f"Off-policy degree: {cfg.policy.off_policy_degree} ({'SYNC' if cfg.policy.off_policy_degree == 0 else 'ASYNC'})")
    logger.info(f"Async eval: {cfg.policy.enable_async_eval}")
    logger.info("="*80)

    # [ALIGN WITH UNIZERO] Initialize reanalyze-related counters (train_unizero_segment.py line 119-121)
    buffer_reanalyze_count = 0
    train_epoch = 0
    reanalyze_batch_size = cfg.policy.reanalyze_batch_size
    batch_size = cfg.policy.batch_size
    best_eval_reward = -float('inf')
    policy_config = cfg.policy

    # Async control variables
    collect_task = None
    train_task = None
    pending_new_data = None  # Store collected data waiting to be added to buffer

    try:
        while True:
            is_sync_mode = coordinator.is_synchronous
            if learner.train_iter > 0 and evaluator.should_eval(learner.train_iter):
                logger.info(f"\n[Iter {learner.train_iter}] Evaluating...")

                async def eval_fn():
                    return evaluator.eval(
                        save_ckpt_fn=learner.save_checkpoint if enable_save else None,
                        train_iter=learner.train_iter,
                        envstep=collector.envstep
                    )
                eval_result = await coordinator.run_eval(eval_fn)
                if not cfg.policy.enable_async_eval and eval_result is not None:
                    stop, eval_reward_dict = eval_result
                    mean_reward = eval_reward_dict.get('reward_mean', 0)
                    logger.info(f"  ✓ Evaluation done: reward_mean={mean_reward:.2f}")

                    if mean_reward > best_eval_reward:
                        best_eval_reward = mean_reward

                    if stop:
                        logger.info(f"  🎉 Training converged! (reward >= {cfg.env.stop_value})")
                        break
                else:
                    logger.info(f"  ✓ Async evaluation started in background")

            collect_kwargs = {
                'temperature': 0.25,
                'epsilon': 0.0
            }

            if is_sync_mode:
                logger.info(f"\n[Iter {learner.train_iter}] Collecting data...")

                new_data = await collector.collect(
                    train_iter=learner.train_iter,
                    policy_kwargs=collect_kwargs
                )
                from lzero.entry.utils import calculate_update_per_collect
                update_per_collect = calculate_update_per_collect(cfg, new_data, world_size=1)

                replay_buffer.push_game_segments(new_data)
                replay_buffer.remove_oldest_data_to_fit()
                buffer_size = replay_buffer.get_num_of_transitions() if hasattr(replay_buffer, 'get_num_of_transitions') else 0
                logger.info(f"  ✓ Data collected, buffer size: {buffer_size} transitions")

            else:
                if collect_task is None or collect_task.done():
                    if coordinator.can_collect():
                        logger.info(f"\n[Iter {learner.train_iter}] Starting async collect...")

                        async def collect_fn():
                            return await collector.collect(
                                train_iter=learner.train_iter,
                                policy_kwargs=collect_kwargs
                            )

                        collect_task = asyncio.create_task(coordinator.run_collect(collect_fn))
                    else:
                        logger.debug(f"Collect blocked (lag={coordinator.collect_train_lag}/{coordinator.off_policy_degree})")

                if collect_task is not None and collect_task.done():
                    new_data = await collect_task
                    collect_task = None

                    pending_new_data = new_data
                    logger.info(f"  ✓ Async collect completed, data pending buffer update")

                if pending_new_data is not None:
                    from lzero.entry.utils import calculate_update_per_collect
                    update_per_collect = calculate_update_per_collect(cfg, pending_new_data, world_size=1)

                    replay_buffer.push_game_segments(pending_new_data)
                    replay_buffer.remove_oldest_data_to_fit()
                    buffer_size = replay_buffer.get_num_of_transitions() if hasattr(replay_buffer, 'get_num_of_transitions') else 0
                    logger.info(f"  ✓ Buffer updated, size: {buffer_size} transitions")

                    pending_new_data = None
                else:
                    update_per_collect = cfg.policy.get('update_per_collect', 10)

            if cfg.policy.buffer_reanalyze_freq >= 1:
                reanalyze_interval = update_per_collect // cfg.policy.buffer_reanalyze_freq
            else:
                if train_epoch > 0 and train_epoch % int(1/cfg.policy.buffer_reanalyze_freq) == 0 and replay_buffer.get_num_of_transitions()//cfg.policy.num_unroll_steps > int(reanalyze_batch_size/cfg.policy.reanalyze_partition):
                    logger.info(f"[Reanalyze] Starting buffer reanalysis...")
                    replay_buffer.reanalyze_buffer(reanalyze_batch_size, policy)
                    buffer_reanalyze_count += 1
                    logger.info(f"  ✓ Buffer reanalyze count: {buffer_reanalyze_count}")

            if collector.envstep > cfg.policy.train_start_after_envsteps:
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

                async def train_one_batch():
                    train_data = replay_buffer.sample(batch_size, policy)
                    train_data.append(learner.train_iter)

                    log_vars = learner.train(train_data, collector.envstep)
                    if cfg.policy.use_priority:
                        replay_buffer.update_priority(train_data, log_vars[0]['value_priority_orig'])

                    return log_vars

                if is_sync_mode:
                    for i in range(update_per_collect):
                        await train_one_batch()
                else:
                    if coordinator.can_train():
                        await coordinator.run_train(train_one_batch)
                    else:
                        logger.debug(f"Train waiting for collect...")
            train_epoch += 1
            policy.recompute_pos_emb_diff_and_clear_cache()

            if collector.envstep >= max_env_step or learner.train_iter >= max_train_iter:
                logger.info("Stopping condition met, training ends!")
                break

            if not is_sync_mode:
                await asyncio.sleep(0.001)

    except KeyboardInterrupt:
        logger.warning("\n⚠ Training interrupted by user (Ctrl+C)")

    except Exception as e:
        logger.error(f"\n✗ Training error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        learner.call_hook('after_run')

        if cfg.policy.enable_async_eval:
            logger.info("Waiting for async eval to complete...")
            await coordinator.wait_for_eval()

        # Print async training statistics
        async_stats = coordinator.get_statistics()
        logger.info("\n" + "="*80)
        logger.info("Async Training Statistics:")
        logger.info(f"  Mode: {async_stats['mode'].upper()}")
        logger.info(f"  Collect iterations: {async_stats['collect_count']}")
        logger.info(f"  Train iterations: {async_stats['train_count']}")
        logger.info(f"  Final lag: {async_stats['collect_train_lag']}")
        if 'collect_avg_time' in async_stats:
            logger.info(f"  Avg collect time: {async_stats['collect_avg_time']:.2f}s")
        if 'train_avg_time' in async_stats:
            logger.info(f"  Avg train time: {async_stats['train_avg_time']:.2f}s")
        if 'eval_avg_time' in async_stats:
            logger.info(f"  Avg eval time: {async_stats['eval_avg_time']:.2f}s")
        logger.info("="*80)

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

    
    # args.quick_test = True
    if args.quick_test:
        logger.info("Using quick test configuration")
        main_cfg, create_cfg = get_priorzero_debug_config(args.env_id, args.seed, exp_name=f'data_priorzero/priorzero_debug_cprofile_no_sft_no_rft_{args.env_id}_seed0')
    else:
        main_cfg, create_cfg = get_priorzero_config(args.env_id, args.seed, exp_name=f'data_priorzero/priorzero_cprofile_no_sft_no_rft_{args.env_id}_seed0')

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
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    main()

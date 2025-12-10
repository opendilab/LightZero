import asyncio
import os
import sys
from functools import partial
from pathlib import Path
from typing import Tuple, Optional

import ray
import torch
import wandb
from ding.config import compile_config
from ding.envs import create_env_manager, get_vec_env_setting
from ding.policy import create_policy
from ding.utils import set_pkg_seed, get_rank, get_world_size
from ding.worker import create_buffer, BaseLearner
from tensorboardX import SummaryWriter
from loguru import logger
from ding.utils import DDPContext
from lzero.config.utils import lz_to_ddp_config

os.environ.setdefault("VLLM_USE_V1", "1")
from vllm import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs

from priorzero_config import get_priorzero_config, get_priorzero_debug_config
from priorzero_collector import PriorZeroCollector
from priorzero_evaluator import PriorZeroEvaluator
import priorzero_policy  
from lzero.mcts.buffer.game_buffer_priorzero import PriorZeroGameBufferOptimized
from lzero.entry.utils import calculate_update_per_collect

def train_priorzero(
    cfg: dict,
    create_cfg: dict,
    seed: int = 0,
    max_train_iter: int = int(1e6),
    max_env_step: Optional[int] = int(1e10),
):
    """
    [PRIORZERO-MODIFIED]
    Main async training function for PriorZero.

    Args:
        cfg: Main configuration dictionary
        create_cfg: Creation configuration for DI-engine components
        seed: Random seed
        max_train_iter: Maximum training iterations
    """
    cfg = compile_config(cfg, seed=seed, auto=True, create_cfg=create_cfg)
    if ray.is_initialized():
        logger.info(f"✓ Ray already initialized (connected to existing cluster)")
    else:
        logger.info(f"✓ Ray not initialized - vLLM will handle initialization if needed")

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

    if cfg.policy.llm_policy_cfg.enable_llm:
        policy._init_llm_learn(tb_logger=tb_logger, exp_name=cfg.exp_name)
        
        logger.info("Creating vLLM engine...")
        tensor_parallel = cfg.policy.llm_policy_cfg.vllm_tensor_parallel_size
        distributed_backend = "ray" if tensor_parallel > 1 else None

        gpu_mem_util = cfg.policy.llm_policy_cfg.gpu_memory_utilization

        engine_args = AsyncEngineArgs(
            model=policy.llm_ckpt_dir,
            tensor_parallel_size=tensor_parallel,
            gpu_memory_utilization=gpu_mem_util,
            distributed_executor_backend=distributed_backend,
            trust_remote_code=True,
            enable_prefix_caching=False,
            enforce_eager=False,
        )
        vllm_engine = AsyncLLMEngine.from_engine_args(engine_args)
        logger.info(f"✓ vLLM Engine created (backend: {distributed_backend or 'default'})")
    
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

    buffer_reanalyze_count = 0
    train_epoch = 0
    reanalyze_batch_size = cfg.policy.reanalyze_batch_size
    batch_size = cfg.policy.batch_size
    
    if cfg.policy.multi_gpu:
        world_size = get_world_size()
        rank = get_rank()
    else:
        world_size = 1
        rank = 0
    
    while True:
        if learner.train_iter > 0 and evaluator.should_eval(learner.train_iter):
            logger.info(f"\n[Iter {learner.train_iter}] Evaluating...")
            stop, reward = evaluator.eval(
                    save_ckpt_fn=learner.save_checkpoint,
                    train_iter=learner.train_iter,
                    envstep=collector.envstep
                )
            if stop:
                break

        collect_kwargs = {
            'temperature': 0.25,
            'epsilon': 0.0
        }

        new_data = collector.collect(
            train_iter=learner.train_iter,
            policy_kwargs=collect_kwargs
        )
        update_per_collect = calculate_update_per_collect(cfg, new_data, world_size=world_size)

        replay_buffer.push_game_segments(new_data)
        num_of_transitions = replay_buffer.get_num_of_transitions() 
        logger.info(f"  ✓ Data collected, num_of_transitions: {num_of_transitions} transitions")

        if cfg.policy.buffer_reanalyze_freq >= 1:
            reanalyze_interval = update_per_collect // cfg.policy.buffer_reanalyze_freq
        else:
            if train_epoch > 0 and train_epoch % int(1/cfg.policy.buffer_reanalyze_freq) == 0:
                logger.info(f"[Reanalyze] Starting buffer reanalysis...")
                replay_buffer.reanalyze_buffer(reanalyze_batch_size, policy)
                buffer_reanalyze_count += 1
                logger.info(f"  ✓ Buffer reanalyze count: {buffer_reanalyze_count}")

        if collector.envstep <= cfg.policy.train_start_after_envsteps:
            continue
        
        if cfg.policy.sample_type == 'episode':
            data_sufficient = num_of_transitions > batch_size
        else:
            data_sufficient = num_of_transitions > batch_size

        if not data_sufficient:
            logger.warning(
                f'  ⚠ Data in replay_buffer is not sufficient: '
                f'batch_size: {batch_size}, replay_buffer: {replay_buffer}. Continue to collect...'
            )
            continue

        logger.info(f"[Iter {learner.train_iter}] Training...")
        for i in range(update_per_collect):
            train_data = replay_buffer.sample(batch_size, policy)
            train_data.append(learner.train_iter)

            log_vars = learner.train(train_data, collector.envstep)
            if cfg.policy.use_priority:
                replay_buffer.update_priority(train_data, log_vars[0]['value_priority_orig'])

        if num_of_transitions >= replay_buffer.replay_buffer_size:
            all_data = replay_buffer.sample(batch_size=cfg.policy.llm_policy_cfg.llm_learn_num_samples, policy=policy)
            replay_buffer._clear()
            policy._forward_llm_learn(all_data)

        train_epoch += 1
        policy.recompute_pos_emb_diff_and_clear_cache()

        if collector.envstep >= max_env_step or learner.train_iter >= max_train_iter:
            logger.info("Stopping condition met, training ends!")
            break

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

    
    args.quick_test = True
    if args.quick_test:
        logger.info("Using quick test configuration")
        main_cfg, create_cfg = get_priorzero_debug_config(args.env_id, args.seed, exp_name=f'data_priorzero/priorzero_sync_debug_{args.env_id}_seed0')
    else:
        main_cfg, create_cfg = get_priorzero_config(args.env_id, args.seed, exp_name=f'data_priorzero/priorzero_sync_rft_reinforce++_{args.env_id}_seed0')

    if main_cfg.policy.multi_gpu:
        with DDPContext():
            main_cfg = lz_to_ddp_config(main_cfg)
            asyncio.run(train_priorzero(
                main_cfg,
                create_cfg,
                seed=args.seed,
                max_train_iter=args.max_iter,
            ))
        
    else:
        # Run training
        asyncio.run(train_priorzero(
            main_cfg,
            create_cfg,
            seed=args.seed,
            max_train_iter=args.max_iter,
        ))


if __name__ == "__main__":
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    main()

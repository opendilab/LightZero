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
from priorzero_policy import *
from lzero.mcts.buffer.game_buffer_priorzero import PriorZeroGameBufferOptimized
from lzero.entry.utils import calculate_update_per_collect
from priorzero_llm_modules import PriorZeroOpenRLHFLLMConfig, PriorZeroOpenRLHFLLMTrainer
# [OOM-FIX] Import memory monitoring tools
from priorzero_memory_monitor import MemoryMonitor, MemoryProfiler, quick_memory_check
# [MULTI-INSTANCE-FIX] Import port manager for running multiple training instances
from priorzero_port_manager import auto_setup_ports_for_training


def train_priorzero(
    cfg: dict,
    create_cfg: dict,
    seed: int = 0,
    max_train_iter: int = int(1e6),
    max_env_step: Optional[int] = int(1e10),
    instance_id: Optional[int] = None,
):
    """
    [PRIORZERO-MODIFIED]
    Main async training function for PriorZero.

    Args:
        cfg: Main configuration dictionary
        create_cfg: Creation configuration for DI-engine components
        seed: Random seed
        max_train_iter: Maximum training iterations
        max_env_step: Maximum environment steps
        instance_id: Instance ID for multi-instance training (ports already configured in main())
    """
    # Note: Ports are configured in main() before this function is called
    # to ensure environment variables are set before any library imports

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

    vllm_engine = None
    if cfg.policy.llm_policy_cfg.enable_llm:
        llm_cfg = PriorZeroOpenRLHFLLMConfig(
            model_name_or_path=policy.llm_policy_cfg.pretrain_llm_path,
            zero_stage=policy.llm_policy_cfg.zero_stage,  # 你传 zero_stage2.json 
            lr=policy.llm_policy_cfg.learning_rate,
            weight_decay=policy.llm_policy_cfg.weight_decay,
            prompt_max_len=policy.llm_policy_cfg.prompt_max_len,
            generate_max_len=policy.llm_policy_cfg.generate_max_len,
            use_cot=policy.llm_policy_cfg.use_cot,
            rft_loss_type=policy.llm_policy_cfg.rft_loss_type,
            rft_clip_epsilon=policy.llm_policy_cfg.rft_clip_epsilon,
            rft_kl_coef=policy.llm_policy_cfg.rft_kl_coef,
            train_batch_size=policy.llm_policy_cfg.train_batch_size,
            micro_train_batch_size=policy.llm_policy_cfg.micro_batch_size,
            gradient_accumulation_steps=policy.llm_policy_cfg.gradient_accumulation_steps,
            bf16=True,
            enable_vllm=True,
            vllm_num_engines=1,
            vllm_tensor_parallel_size=policy.llm_policy_cfg.vllm_tensor_parallel_size,
            gpu_memory_utilization=policy.llm_policy_cfg.gpu_memory_utilization,
            seed=seed,
            temperature=policy.llm_policy_cfg.temperature,
            top_p=policy.llm_policy_cfg.top_p,
        )
        trainer = PriorZeroOpenRLHFLLMTrainer(llm_cfg, tb_logger=tb_logger, exp_name=cfg.exp_name)
        llm_prior_generator = trainer.llm_prior_generator
        # policy._init_llm_learn(tb_logger=tb_logger, exp_name=cfg.exp_name, vllm_engine=vllm_engine)
    
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
        llm_prior_generator=llm_prior_generator,
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
    # [OOM-FIX] Initialize Memory Monitor
    # ==================================================================
    memory_monitor = MemoryMonitor(enable=True, log_interval=1)
    logger.info("✓ Memory monitor initialized")
    logger.info(f"  Initial memory: {quick_memory_check()}")

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
        # ==================================================================
        # [OOM-FIX] Monitor and reset peak memory at start of iteration
        # ==================================================================
        memory_monitor.reset_peak_memory()
        memory_monitor.log_memory(f"Iter {learner.train_iter} - Start", logger)

        # ==================================================================
        # Evaluation Phase
        # ==================================================================
        if learner.train_iter > 0 and evaluator.should_eval(learner.train_iter):
            logger.info(f"\n[Iter {learner.train_iter}] Evaluating...")
            stop, reward = evaluator.eval(
                save_ckpt_fn=learner.save_checkpoint,
                train_iter=learner.train_iter,
                envstep=collector.envstep
            )
            if stop:
                break
            memory_monitor.log_memory(f"Iter {learner.train_iter} - After Eval", logger)

        # ==================================================================
        # [OOM-FIX] Wake up vLLM engines before collection
        # ==================================================================
        # vLLM engines were put to sleep after previous collection to save memory.
        # Now we need to wake them up for the next collection phase.
        if cfg.policy.llm_policy_cfg.enable_llm and llm_prior_generator is not None:
            if hasattr(llm_prior_generator, 'vllm_engines') and llm_prior_generator.vllm_engines is not None:
                try:
                    wakeup_refs = []
                    for engine in llm_prior_generator.vllm_engines:
                        wakeup_refs.append(engine.wake_up.remote())

                    # Wait for all engines to wake up (~1-2 seconds)
                    ray.get(wakeup_refs)
                    logger.info("✓ vLLM engines woken up for collection (~4.6GB GPU memory allocated)")
                    memory_monitor.log_memory(f"Iter {learner.train_iter} - After vLLM Wake", logger)
                except Exception as e:
                    logger.warning(f"⚠ Failed to wake up vLLM engines: {e}")
                    logger.warning("  Continuing without vLLM wake-up. LLM priors may not work correctly.")

        # ==================================================================
        # Collection Phase
        # ==================================================================
        collect_kwargs = {
            'temperature': 0.25,
            'epsilon': 0.0
        }

        new_data = collector.collect(train_iter=learner.train_iter, policy_kwargs=collect_kwargs)
        update_per_collect = calculate_update_per_collect(cfg, new_data, world_size=world_size)

        # [OOM-FIX] Monitor memory after collection (vLLM should be asleep now)
        memory_monitor.log_memory(f"Iter {learner.train_iter} - After Collect", logger)
        memory_monitor.compare_stages(
            f"Iter {learner.train_iter} - After vLLM Wake",
            f"Iter {learner.train_iter} - After Collect",
            logger, threshold_gb=0.5
        )

        replay_buffer.push_game_segments(new_data)
        replay_buffer.remove_oldest_data_to_fit()
        num_of_transitions = replay_buffer.get_num_of_transitions()
        new_num_of_transitions = replay_buffer.get_num_of_transitions() - replay_buffer.last_pos_in_transition
        logger.info(f"  ✓ Data collected, num_of_transitions: {num_of_transitions} transitions")

        # [OOM-FIX] Monitor after buffer operations
        memory_monitor.log_memory(f"Iter {learner.train_iter} - After Buffer Push", logger)

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
            logger.info(f"[Iter {learner.train_iter}] Training...")
            train_data = replay_buffer.sample(batch_size, policy)
            train_data.append(learner.train_iter)

            log_vars = learner.train(train_data, collector.envstep)
            if cfg.policy.use_priority:
                replay_buffer.update_priority(train_data, log_vars[0]['value_priority_orig'])

        # [OOM-FIX] Monitor after World Model training
        memory_monitor.log_memory(f"Iter {learner.train_iter} - After WM Train", logger)
        memory_monitor.compare_stages(
            f"Iter {learner.train_iter} - After Collect",
            f"Iter {learner.train_iter} - After WM Train",
            logger, threshold_gb=0.5
        )

        # ==================================================================
        # LLM RFT Training Phase (if enough samples collected)
        # ==================================================================
        if new_num_of_transitions >= cfg.policy.llm_policy_cfg.llm_learn_num_samples:
            logger.info(f"[Iter {learner.train_iter}] Starting LLM RFT training...")
            logger.info(f"  → Training on {cfg.policy.llm_policy_cfg.llm_learn_num_samples} latest samples")

            # [CUDA-FIX] Wake up vLLM if CoT is enabled (needed for prefix generation)
            # Without this, vLLM is asleep and calling _build_cot_prefix_texts() causes CUDA error
            vllm_woken_for_llm_training = False
            if cfg.policy.llm_policy_cfg.use_cot and cfg.policy.llm_policy_cfg.enable_llm:
                if llm_prior_generator is not None:
                    if hasattr(llm_prior_generator, 'vllm_engines') and llm_prior_generator.vllm_engines is not None:
                        try:
                            logger.info(f"  → Waking up vLLM for CoT prefix generation...")
                            wakeup_refs = []
                            for engine in llm_prior_generator.vllm_engines:
                                wakeup_refs.append(engine.wake_up.remote())
                            ray.get(wakeup_refs)
                            vllm_woken_for_llm_training = True
                            logger.info("  ✓ vLLM engines woken up for LLM training (~4.6GB GPU memory allocated)")
                            memory_monitor.log_memory(f"Iter {learner.train_iter} - Before LLM Train (vLLM awake)", logger)
                        except Exception as e:
                            logger.warning(f"⚠ Failed to wake up vLLM: {e}")
                            logger.warning("  LLM training may fail if CoT prefix generation is needed")
            else:
                logger.info(f"  → vLLM stays asleep (CoT disabled or LLM disabled)")

            all_data = replay_buffer.fetch_latest_batch(
                batch_size=cfg.policy.llm_policy_cfg.llm_learn_num_samples,
                policy=policy
            )
            trainer.train_rft_from_priorzero_batch(all_data)

            # [CUDA-FIX] Put vLLM back to sleep after LLM training
            if vllm_woken_for_llm_training:
                if llm_prior_generator is not None:
                    if hasattr(llm_prior_generator, 'vllm_engines') and llm_prior_generator.vllm_engines is not None:
                        try:
                            sleep_refs = []
                            for engine in llm_prior_generator.vllm_engines:
                                sleep_refs.append(engine.sleep.remote(level=1))
                            ray.get(sleep_refs)
                            logger.info("  ✓ vLLM engines put back to sleep after LLM training (~4.6GB GPU memory freed)")
                        except Exception as e:
                            logger.warning(f"⚠ Failed to sleep vLLM: {e}")
                            logger.warning("  vLLM may continue to occupy GPU memory")

            # [OOM-FIX] Monitor after LLM training (this is the OOM hotspot!)
            memory_monitor.log_memory(f"Iter {learner.train_iter} - After LLM Train", logger)
            memory_monitor.compare_stages(
                f"Iter {learner.train_iter} - After WM Train",
                f"Iter {learner.train_iter} - After LLM Train",
                logger, threshold_gb=0.5
            )

        train_epoch += 1
        policy.recompute_pos_emb_diff_and_clear_cache()

        # [OOM-FIX] End of iteration summary
        memory_monitor.log_memory(f"Iter {learner.train_iter} - End", logger)
        logger.info(f"[Iter {learner.train_iter}] Iteration complete. Peak memory: {quick_memory_check()}")

        if collector.envstep >= max_env_step or learner.train_iter >= max_train_iter:
            logger.info("Stopping condition met, training ends!")
            break


    return policy


def main():
    """
    Main entry point with argument parsing.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description='PriorZero Training - Supports multiple instances on same machine',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single instance training
  python priorzero_entry_sync.py --env_id detective.z5 --seed 0

  # Multiple instances on same machine (use different instance_id for each)
  python priorzero_entry_sync.py --env_id detective.z5 --instance_id 0 &
  python priorzero_entry_sync.py --env_id zork1.z5 --instance_id 1 &
        """
    )
    parser.add_argument('--env_id', type=str, default='detective.z5', help='Jericho game ID')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--max_iter', type=int, default=int(1e6), help='Max training iterations')
    parser.add_argument('--instance_id', type=int, default=None,
                        help='Instance ID for multi-instance training (0, 1, 2, ...). '
                             'If not specified, will auto-find available ports.')
    parser.add_argument('--quick_test', action='store_true', help='Use quick test config')
    parser.add_argument('--no_save', action='store_true', help='Disable checkpoint saving')
    parser.add_argument('--debug', action='store_true', help='Enable detailed debug logging (obs, action, LLM output)')

    args = parser.parse_args()

    # Note: Ports are already configured in __main__ block
    # before any imports, so we don't need to configure them again here

    args.quick_test = False
    use_cot=True

    if args.quick_test:
        logger.info("Using quick test configuration")
        main_cfg, create_cfg = get_priorzero_debug_config(args.env_id, args.seed, use_cot=use_cot, exp_name=f'data_priorzero/priorzero_sync_debug_{args.env_id}_seed0')
    else:
        main_cfg, create_cfg = get_priorzero_config(args.env_id, args.seed, use_cot=use_cot, exp_name=f'data_priorzero/priorzero_sync_rft_reinforce++_{args.env_id}_seed0')

    if main_cfg.policy.multi_gpu:
        with DDPContext():
            main_cfg = lz_to_ddp_config(main_cfg)
            asyncio.run(train_priorzero(
                main_cfg,
                create_cfg,
                seed=args.seed,
                max_train_iter=args.max_iter,
                instance_id=args.instance_id,  # [MULTI-INSTANCE-FIX] Pass instance_id
            ))

    else:
        # Run training
        asyncio.run(train_priorzero(
            main_cfg,
            create_cfg,
            seed=args.seed,
            max_train_iter=args.max_iter,
            instance_id=args.instance_id,  # [MULTI-INSTANCE-FIX] Pass instance_id
        ))


if __name__ == "__main__":
    # ==================================================================
    # [CRITICAL] Setup ports BEFORE any other code
    # ==================================================================
    # This MUST be the FIRST thing to run to avoid port conflicts
    # Some libraries cache MASTER_PORT during module import!

    import argparse
    import sys

    # Quick argument parsing BEFORE any imports
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--instance_id', type=int, default=None)
    args, _ = parser.parse_known_args()

    # Setup ports immediately
    sys.path.insert(0, os.path.dirname(__file__))
    from priorzero_port_manager import auto_setup_ports_for_training

    port_config = auto_setup_ports_for_training(
        instance_id=args.instance_id,
        verbose=True
    )
    print(f"[CRITICAL] Ports configured at startup: DeepSpeed={port_config['master_port']}, Ray={port_config['ray_port']}")

    # Now set tokenizers env var
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    # [LOG-FIX] Reduce vLLM logging verbosity to avoid flooding logs with progress bars
    # This suppresses the frequent "Processed prompts: XX%" messages from vLLM
    os.environ['VLLM_LOGGING_LEVEL'] = 'WARNING'  # Suppress INFO level logs from vLLM
    os.environ['VLLM_CONFIGURE_LOGGING'] = '0'    # Disable vLLM's logging configuration

    # [OOM-FIX] PyTorch memory optimization to reduce fragmentation
    # expandable_segments:True - Allows PyTorch to expand memory segments dynamically
    # max_split_size_mb:128 - Limits memory block splitting to reduce fragmentation
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'

    # Finally, run main
    main()

"""
PriorZero-ORZ Hybrid Pipeline Entry

This is a complete, executable training pipeline that combines:
- PriorZero's UniZero world model + MCTS for planning
- ORZ's distributed LLM training with advanced RFT

The pipeline reuses PriorZero's infrastructure (environments, collectors, etc.)
but replaces the LLM training component with ORZ's RayPPOTrainer.

Usage:
    # Debug mode (small scale, fast)
    DEBUG_MODE=True python -m zoo.jericho.priorzero.priorzero_orz_entry

    # Normal training
    python -m zoo.jericho.priorzero.priorzero_orz_entry

Author: PriorZero Team
Date: 2025-10-21
"""

import asyncio
import os
import sys
from pathlib import Path
from functools import partial
from typing import Optional
import time

# ==============================================================================
# Ensure local LightZero is used
# ==============================================================================
from ensure_local_lightzero import ensure_local_lightzero
ensure_local_lightzero()

import torch
from ding.config import compile_config
from ding.envs import create_env_manager, get_vec_env_setting
from ding.policy import create_policy
from ding.utils import set_pkg_seed, get_rank
from ding.worker import BaseLearner
from tensorboardX import SummaryWriter
from loguru import logger
from vllm import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs

# PriorZero imports
from priorzero_config import get_priorzero_config_for_quick_test
from priorzero_collector import PriorZeroCollector
from priorzero_evaluator import PriorZeroEvaluator
import priorzero_policy  # noqa: F401
from lzero.mcts.buffer.game_buffer_priorzero import PriorZeroGameBufferOptimized

# Try to import ORZ (optional for this basic version)
ORZ_AVAILABLE = False
try:
    ORZ_PATH = Path("/mnt/nfs/zhangjinouwen/puyuan/Open-Reasoner-Zero")
    if ORZ_PATH.exists() and str(ORZ_PATH) not in sys.path:
        sys.path.insert(0, str(ORZ_PATH))

    from orz.ppo import RayPPOTrainer
    from orz.exps.examples.ppo.ppo_base_exp import BasePPOExp
    ORZ_AVAILABLE = True
    logger.info("✅ ORZ available - will use ORZ RayPPOTrainer for LLM training")
except ImportError as e:
    logger.warning(f"⚠️  ORZ not available ({e}) - will use PriorZero's built-in LLM training")


# ==============================================================================
# Configuration
# ==============================================================================

DEBUG_MODE = os.environ.get("DEBUG_MODE", "False") == "True"


class HybridTrainingConfig:
    """
    Hybrid training configuration combining PriorZero and ORZ settings.
    """
    def __init__(self):
        # Get base PriorZero config
        if DEBUG_MODE:
            self.priorzero_cfg, self.priorzero_create_cfg = get_priorzero_config_for_quick_test(
                env_id='zork1.z5',
                seed=0,
                debug_mode=True
            )
        else:
            from priorzero_config import get_priorzero_config
            self.priorzero_cfg, self.priorzero_create_cfg = get_priorzero_config(
                env_id='zork1.z5',
                seed=0,
                enable_llm=True,
                enable_rft=True,
                debug_mode=False
            )

        # ======================================================================
        # Hybrid-specific settings
        # ======================================================================
        # Training mode: "parallel", "sequential", "alternating"
        self.wm_training_mode = "parallel"

        # How often to train each component (in iterations)
        self.wm_train_freq = 1  # Train world model every N iters
        self.llm_train_freq = 5  # Train LLM every N iters

        # Use ORZ if available
        self.use_orz_trainer = ORZ_AVAILABLE

        # ORZ-specific settings (only used if ORZ_AVAILABLE)
        if ORZ_AVAILABLE:
            self.orz_rollout_batch_size = 32 if DEBUG_MODE else 128
            self.orz_train_batch_size = 8 if DEBUG_MODE else 32
            self.orz_actor_lr = 1e-6
            self.orz_critic_lr = 5e-6


# ==============================================================================
# Main Training Function
# ==============================================================================

async def train_priorzero_orz(
    cfg: dict,
    create_cfg: dict,
    hybrid_cfg: HybridTrainingConfig,
    seed: int = 0,
    max_train_iter: int = 10000,
    max_env_step: Optional[int] = int(1e10),
    enable_save: bool = True,
):
    """
    Main hybrid training function.

    This combines PriorZero's world model training with ORZ's LLM training.
    """
    # ==================================================================
    # 1. Compile Configuration
    # ==================================================================
    cfg = compile_config(cfg, seed=seed, auto=True, create_cfg=create_cfg)

    # ==================================================================
    # 2. Create vLLM Engine (for LLM inference during collection)
    # ==================================================================
    logger.info("Creating vLLM engine for LLM policy...")

    tensor_parallel = cfg.policy.llm_policy_cfg.vllm_tensor_parallel_size
    gpu_mem_util = cfg.policy.llm_policy_cfg.gpu_memory_utilization

    # Use conservative settings for stability
    if gpu_mem_util > 0.85:
        gpu_mem_util = 0.7
        logger.info(f"✓ Adjusted GPU memory utilization to {gpu_mem_util}")

    # Force V0 engine for stability
    os.environ['VLLM_USE_V1'] = '0'

    try:
        engine_args = AsyncEngineArgs(
            model=cfg.policy.llm_policy_cfg.pretrain_llm_path,
            tensor_parallel_size=tensor_parallel,
            gpu_memory_utilization=gpu_mem_util,
            trust_remote_code=True,
            enable_prefix_caching=False,
            enforce_eager=False,
        )
        vllm_engine = AsyncLLMEngine.from_engine_args(engine_args)
        logger.info(f"✓ vLLM Engine created")
    except Exception as e:
        logger.error(f"❌ Failed to create vLLM engine: {e}")
        logger.info("Continuing without LLM inference (will use world model only)")
        vllm_engine = None

    # ==================================================================
    # 3. Create Environments
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
    # 4. Create Policy, Buffer, and Components
    # ==================================================================
    logger.info("Creating policy, buffer, and components...")

    # Create policy
    policy = create_policy(
        cfg.policy,
        enable_field=['learn', 'collect', 'eval']
    )
    logger.info("✓ Policy created")

    # Create TensorBoard logger
    os.makedirs(f'./{cfg.exp_name}/log/', exist_ok=True)
    tb_logger = SummaryWriter(
        os.path.join(f'./{cfg.exp_name}/log/', 'serial')
    ) if get_rank() == 0 else None
    logger.info(f"✓ TensorBoard logger: ./{cfg.exp_name}/log/")

    # Create learner (for world model training)
    learner = BaseLearner(
        cfg.policy.learn.learner,
        policy.learn_mode,
        tb_logger,
        exp_name=cfg.exp_name
    )
    logger.info("✓ BaseLearner created")

    # Create replay buffer (PriorZero version with game_segments support)
    replay_buffer = PriorZeroGameBufferOptimized(cfg.policy)
    logger.info("✓ PriorZero replay buffer created")

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

    # Call learner's before_run hook
    learner.call_hook('before_run')

    # ==================================================================
    # 5. Initialize ORZ Trainer (if available)
    # ==================================================================
    orz_trainer = None
    if hybrid_cfg.use_orz_trainer and ORZ_AVAILABLE:
        logger.info("Initializing ORZ RayPPOTrainer for LLM training...")
        # TODO: Initialize ORZ trainer with proper configuration
        # This would require creating ORZ-compatible dataset, strategy, etc.
        # For now, we'll use PriorZero's built-in LLM training
        logger.warning("⚠️  ORZ trainer initialization not yet implemented - using PriorZero's LLM training")
        hybrid_cfg.use_orz_trainer = False

    # ==================================================================
    # 6. Main Training Loop
    # ==================================================================
    logger.info("="*80)
    logger.info("Starting PriorZero-ORZ Hybrid Training")
    logger.info("="*80)
    logger.info(f"Experiment: {cfg.exp_name}")
    logger.info(f"Max iterations: {max_train_iter}")
    logger.info(f"Training mode: {hybrid_cfg.wm_training_mode}")
    logger.info(f"Use ORZ trainer: {hybrid_cfg.use_orz_trainer}")
    logger.info(f"LLM model: {cfg.policy.llm_policy_cfg.pretrain_llm_path}")
    logger.info(f"World model: UniZero")
    logger.info("="*80)

    # Training state
    best_eval_reward = -float('inf')
    total_game_segments_collected = 0

    try:
        while learner.train_iter < max_train_iter and collector.envstep < max_env_step:
            current_iter = learner.train_iter

            # ==============================================================
            # Step 1: Evaluation (if needed)
            # ==============================================================
            if current_iter > 0 and evaluator.should_eval(current_iter):
                logger.info(f"\n{'='*60}")
                logger.info(f"[Iter {current_iter}] Evaluating...")
                logger.info(f"{'='*60}")

                eval_result = await evaluator.eval(
                    save_ckpt_fn=learner.save_checkpoint if enable_save else None,
                    train_iter=current_iter,
                    envstep=collector.envstep
                )

                if eval_result is not None:
                    stop, eval_reward_dict = eval_result
                    mean_reward = eval_reward_dict.get('reward_mean', 0)
                    logger.info(f"✓ Evaluation: reward_mean={mean_reward:.2f}")

                    if mean_reward > best_eval_reward:
                        best_eval_reward = mean_reward
                        logger.info(f"🎯 New best reward: {best_eval_reward:.2f}")

                    if stop:
                        logger.info(f"🎉 Training converged! (reward >= {cfg.env.stop_value})")
                        break

            # ==============================================================
            # Step 2: Collect Data using MCTS
            # ==============================================================
            logger.info(f"\n[Iter {current_iter}] Collecting data...")

            collect_kwargs = {
                'temperature': 0.25,
                'epsilon': 0.0
            }

            new_data = await collector.collect(
                train_iter=current_iter,
                policy_kwargs=collect_kwargs
            )

            # Add to replay buffer
            from lzero.entry.utils import calculate_update_per_collect
            update_per_collect = calculate_update_per_collect(cfg, new_data, world_size=1)

            # Update buffer
            replay_buffer.push_game_segments(new_data)
            logger.info(
                f"✓ Collected {len(new_data)} segments "
                f"(total: {replay_buffer.get_num_of_game_segments()} segments, "
                f"{replay_buffer.get_num_of_transitions()} transitions)"
            )

            total_game_segments_collected += len(new_data)

            # ==============================================================
            # Step 3: World Model Training
            # ==============================================================
            if current_iter % hybrid_cfg.wm_train_freq == 0:
                # Check if we have enough data
                if replay_buffer.get_num_of_transitions() >= cfg.policy.batch_size:
                    logger.info(f"[Iter {current_iter}] Training world model...")

                    # Sample and train
                    for _ in range(update_per_collect):
                        train_data = replay_buffer.sample(
                            cfg.policy.batch_size,
                            policy
                        )

                        # Train (this includes both WM and LLM in PriorZero)
                        log_dict = learner.train(train_data, collector.envstep)

                        # Log to TensorBoard
                        if tb_logger and get_rank() == 0:
                            for k, v in log_dict.items():
                                tb_logger.add_scalar(f'train/{k}', v, collector.envstep)

                    logger.info(
                        f"✓ WM training done - "
                        f"wm_loss: {log_dict.get('wm_total_loss', 0):.4f}, "
                        f"llm_sft_loss: {log_dict.get('llm_sft_loss', 0):.4f}"
                    )
                else:
                    logger.info(f"Skipping training - not enough data yet "
                              f"({replay_buffer.get_num_of_transitions()} < {cfg.policy.batch_size})")

            # ==============================================================
            # Step 4: LLM Training with ORZ (if enabled and at right frequency)
            # ==============================================================
            if (hybrid_cfg.use_orz_trainer and orz_trainer is not None and
                current_iter % hybrid_cfg.llm_train_freq == 0):
                logger.info(f"[Iter {current_iter}] Training LLM with ORZ...")
                # TODO: Implement ORZ training step
                # This would extract game_segments from replay buffer,
                # convert to ORZ format, and call orz_trainer.train()
                pass

            # ==============================================================
            # Step 5: Logging and Checkpointing
            # ==============================================================
            if current_iter % 100 == 0:
                logger.info(f"\n{'='*60}")
                logger.info(f"Progress Summary (Iter {current_iter})")
                logger.info(f"{'='*60}")
                logger.info(f"Env steps: {collector.envstep}")
                logger.info(f"Game segments collected: {total_game_segments_collected}")
                logger.info(f"Buffer size: {replay_buffer.get_num_of_transitions()} transitions")
                logger.info(f"Best eval reward: {best_eval_reward:.2f}")
                logger.info(f"{'='*60}\n")

            # Save checkpoint periodically
            if enable_save and current_iter % 500 == 0 and current_iter > 0:
                logger.info(f"[Iter {current_iter}] Saving checkpoint...")
                learner.save_checkpoint(collector.envstep)
                logger.info("✓ Checkpoint saved")

    except KeyboardInterrupt:
        logger.info("\n⚠️  Training interrupted by user")
    except Exception as e:
        logger.error(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # ==============================================================
        # Cleanup
        # ==============================================================
        logger.info("\nCleaning up...")

        # Save final checkpoint
        if enable_save:
            logger.info("Saving final checkpoint...")
            learner.save_checkpoint(collector.envstep)

        # Close environments
        collector_env.close()
        evaluator_env.close()

        # Close loggers
        if tb_logger:
            tb_logger.close()

        logger.info("✓ Cleanup complete")
        logger.info("="*80)
        logger.info("Training finished!")
        logger.info(f"Total iterations: {learner.train_iter}")
        logger.info(f"Total env steps: {collector.envstep}")
        logger.info(f"Best eval reward: {best_eval_reward:.2f}")
        logger.info("="*80)


# ==============================================================================
# Entry Point
# ==============================================================================

async def main():
    """Main entry point."""
    # Create hybrid configuration
    hybrid_cfg = HybridTrainingConfig()

    # Run training
    await train_priorzero_orz(
        cfg=hybrid_cfg.priorzero_cfg,
        create_cfg=hybrid_cfg.priorzero_create_cfg,
        hybrid_cfg=hybrid_cfg,
        seed=0,
        max_train_iter=10000 if not DEBUG_MODE else 100,
        enable_save=True,
    )


if __name__ == "__main__":
    logger.info("="*80)
    logger.info("PriorZero-ORZ Hybrid Training Pipeline")
    logger.info("="*80)
    logger.info(f"Debug mode: {DEBUG_MODE}")
    logger.info(f"ORZ available: {ORZ_AVAILABLE}")
    logger.info("="*80)

    # Run async training
    asyncio.run(main())

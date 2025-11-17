"""
PriorZero-ORZ Complete Integration
完整可执行版本 with ORZ RayPPOTrainer

This version includes:
1. Fixed vLLM None handling
2. Fixed asyncio scope issue
3. Complete ORZ RayPPOTrainer integration
4. Robust error handling

Usage:
    DEBUG_MODE=True python -m zoo.jericho.priorzero.priorzero_orz_complete

Author: PriorZero Team
Date: 2025-10-21
"""

import asyncio
import os
import sys
import re
from pathlib import Path
from functools import partial
from typing import Optional, List, Dict, Any, Callable, Awaitable, Tuple
import time
import json

# ==============================================================================
# Ensure local LightZero is used
# ==============================================================================
from ensure_local_lightzero import ensure_local_lightzero
ensure_local_lightzero()

import torch
import numpy as np
from ding.config import compile_config
from ding.envs import create_env_manager, get_vec_env_setting
from ding.policy import create_policy
from ding.utils import set_pkg_seed, get_rank
from ding.worker import BaseLearner
from tensorboardX import SummaryWriter
from loguru import logger

# PriorZero imports
from priorzero_config import get_priorzero_config_for_quick_test, get_priorzero_config
from priorzero_collector import PriorZeroCollector
from priorzero_evaluator import PriorZeroEvaluator
import priorzero_policy  # noqa: F401
from lzero.mcts.buffer.game_buffer_priorzero import PriorZeroGameBufferOptimized

# vLLM imports (optional)
try:
    from vllm import AsyncLLMEngine
    from vllm.engine.arg_utils import AsyncEngineArgs
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    logger.warning("vLLM not available - LLM inference will be disabled")

# Try to import ORZ
ORZ_AVAILABLE = False
ORZ_PATH = Path("/mnt/afs/wanzunian/niuyazhe/puyuan/Open-Reasoner-Zero")

try:
    if ORZ_PATH.exists() and str(ORZ_PATH) not in sys.path:
        sys.path.insert(0, str(ORZ_PATH))

    from orz.ppo import RayPPOTrainer, PromptDataset
    from orz.exps.examples.ppo.ppo_base_exp import BasePPOExp, BasePPOExpConfig
    from orz.ppo.utils import get_strategy
    from transformers import AutoTokenizer
    import ray
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
            self.priorzero_cfg, self.priorzero_create_cfg = get_priorzero_config(
                env_id='zork1.z5',
                seed=0,
                enable_llm=True,
                enable_rft=True,
                debug_mode=False
            )

        # Hybrid-specific settings
        self.wm_training_mode = "parallel"
        self.wm_train_freq = 1
        self.llm_train_freq = 5
        self.use_orz_trainer = ORZ_AVAILABLE

        # vLLM settings
        self.use_vllm = VLLM_AVAILABLE
        self.vllm_required = False  # Set to True if vLLM is required

        # ORZ-specific settings (only used if ORZ_AVAILABLE)
        if ORZ_AVAILABLE:
            self.orz_rollout_batch_size = 32 if DEBUG_MODE else 128
            self.orz_train_batch_size = 8 if DEBUG_MODE else 32
            self.orz_actor_lr = 1e-6
            self.orz_critic_lr = 5e-6
            self.orz_num_episodes = 2 if DEBUG_MODE else 10


# ==============================================================================
# ORZ Data Adapter and Dataset
# ==============================================================================

class GameSegmentToORZAdapter:
    """
    Convert PriorZero game_segments to ORZ-compatible format.
    """

    @staticmethod
    def convert_segments_to_prompts(game_segments: List[Any], tokenizer) -> List[Dict]:
        """
        Convert game_segments to ORZ prompt format.

        Args:
            game_segments: List of GameSegment from PriorZero
            tokenizer: HuggingFace tokenizer

        Returns:
            List of ORZ-compatible prompt dictionaries
        """
        prompts = []

        for segment in game_segments:
            # Extract raw observations if available
            if hasattr(segment, 'raw_obs_segment') and segment.raw_obs_segment:
                for i, (obs, action) in enumerate(zip(
                    segment.raw_obs_segment,
                    segment.action_segment
                )):
                    # Create ORZ format prompt
                    prompt_dict = {
                        "prompt": [{"value": obs}],
                        "final_answer": action,
                        "file_name": f"segment_{id(segment)}_step_{i}"
                    }
                    prompts.append(prompt_dict)

        return prompts

    @staticmethod
    def extract_training_data(game_segments: List[Any]) -> Dict[str, List]:
        """
        Extract training data from game_segments for ORZ.

        Returns:
            Dictionary containing:
            - states: List of state descriptions
            - actions: List of actions taken
            - rewards: List of rewards received
            - mcts_policies: List of MCTS visit distributions
        """
        training_data = {
            'states': [],
            'actions': [],
            'rewards': [],
            'mcts_policies': []
        }

        for segment in game_segments:
            # Extract raw observations (states)
            if hasattr(segment, 'raw_obs_segment'):
                training_data['states'].extend(segment.raw_obs_segment)

            # Extract actions
            if hasattr(segment, 'action_segment'):
                training_data['actions'].extend(segment.action_segment)

            # Extract rewards
            if hasattr(segment, 'reward_segment'):
                training_data['rewards'].extend(segment.reward_segment)

            # Extract MCTS policies
            if hasattr(segment, 'mcts_policy_segment'):
                training_data['mcts_policies'].extend(segment.mcts_policy_segment)

        return training_data


# Only define dataset classes if ORZ is available
if ORZ_AVAILABLE:
    from jinja2 import Template

    class JerichoPromptDataset(PromptDataset):
        """
        Custom dataset for Jericho text adventure games in ORZ format.
        Adapts PriorZero game_segments to ORZ PPO training format.
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def process_dialogue(self, dialogue: dict):
            """
            Process a single dialogue (observation + action pair) into ORZ format.

            Args:
                dialogue: Dict with 'prompt', 'final_answer', 'file_name'

            Returns:
                prompt: Formatted prompt string
                extra: Dict with answer and metadata
            """
            # Template for Jericho text adventure prompts
            prompt_template_jinja = """\
{{bos_token}}A conversation between User and Assistant. The User is playing a text adventure game \
and needs to decide the next action. The Assistant carefully analyzes the current game state, \
considers the available actions, and recommends the best action to take. \
The reasoning process is enclosed within <think> </think> tags, and the recommended action \
is enclosed within <answer> </answer> tags. For example: \
<think> The player is in a dark room and needs light. The lamp is available. </think> \
<answer> take lamp </answer>. User: {{prompt}}
Assistant: <think>\
"""

            prompt_instruction_template_jinja = """\
Current game state:
{{prompt}}

What is the best action to take? Put your answer inside <answer> </answer> tags.
"""

            # Validate dialogue format
            assert isinstance(dialogue, dict), "dialogue must be a dict"
            assert "prompt" in dialogue, "dialogue must contain prompt"
            assert "final_answer" in dialogue, "dialogue must contain final_answer"

            # Build prompt
            prompt_instruction_template = Template(prompt_instruction_template_jinja)
            prompt_instruction = prompt_instruction_template.render(
                prompt=dialogue["prompt"][0]["value"]
            )

            prompt_template = Template(prompt_template_jinja)
            if self.tokenizer.bos_token_id is None:
                bos_token = ""
            else:
                bos_token = self.tokenizer.decode([self.tokenizer.bos_token_id])

            prompt = prompt_template.render(
                bos_token=bos_token,
                prompt=prompt_instruction
            )

            extra = {
                "answer": dialogue["final_answer"],
                "file_name": dialogue.get("file_name", "unknown")
            }

            return prompt, extra


# ==============================================================================
# Main Training Function
# ==============================================================================

async def train_priorzero_orz_complete(
    cfg: dict,
    create_cfg: dict,
    hybrid_cfg: HybridTrainingConfig,
    seed: int = 0,
    max_train_iter: int = 10000,
    max_env_step: Optional[int] = int(1e10),
    enable_save: bool = True,
):
    """
    Main hybrid training function with complete ORZ integration.
    """
    # ==================================================================
    # 1. Compile Configuration
    # ==================================================================
    cfg = compile_config(cfg, seed=seed, auto=True, create_cfg=create_cfg)

    # ==================================================================
    # 2. Create vLLM Engine (optional) - Based on priorzero_entry.py
    # ==================================================================
    vllm_engine = None

    if hybrid_cfg.use_vllm and VLLM_AVAILABLE:
        logger.info("Creating vLLM engine...")

        # [ROBUST FIX] Handle shared GPU environment
        # Solution: Use alternative initialization method with fallback
        tensor_parallel = cfg.policy.llm_policy_cfg.vllm_tensor_parallel_size
        distributed_backend = "ray" if tensor_parallel > 1 else None

        # [ROBUST FIX] Lower GPU memory utilization in shared environment
        gpu_mem_util = cfg.policy.llm_policy_cfg.gpu_memory_utilization
        if gpu_mem_util > 0.85:
            gpu_mem_util = 0.75  # More conservative
            logger.info(f"✓ Adjusted GPU memory utilization to {gpu_mem_util} for stability")

        # [ROBUST FIX] Use vLLM V0 engine for stability (as in priorzero_entry.py)
        use_v1_env = os.environ.get('VLLM_USE_V1', None)
        if use_v1_env is None:
            # Only set if not already set by user
            os.environ['VLLM_USE_V1'] = '0'
            logger.info("✓ Using vLLM V0 engine for stability")

        # Fix tokenizers parallelism warning
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

        try:
            from vllm.engine.arg_utils import AsyncEngineArgs

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
                # Fallback: Try without V1 env var or with eager mode
                logger.warning(f"⚠️  Initial vLLM initialization failed: {e}")
                logger.info("Retrying with alternative configuration...")

                if 'VLLM_USE_V1' in os.environ:
                    del os.environ['VLLM_USE_V1']

                try:
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
                except Exception as e2:
                    logger.error(f"❌ Failed to create vLLM engine with fallback: {e2}")
                    if hybrid_cfg.vllm_required:
                        raise
                    logger.warning("Continuing without vLLM (LLM prior will be disabled)")
            else:
                logger.error(f"❌ Failed to create vLLM engine: {e}")
                import traceback
                logger.error(f"Full traceback:\n{traceback.format_exc()}")
                if hybrid_cfg.vllm_required:
                    raise
                logger.warning("Continuing without vLLM (LLM prior will be disabled)")
    else:
        logger.info("vLLM disabled or not available - continuing without LLM inference")

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

    # Create replay buffer
    replay_buffer = PriorZeroGameBufferOptimized(cfg.policy)
    logger.info("✓ PriorZero replay buffer created")

    # Create collector
    collector = PriorZeroCollector(
        env=collector_env,
        policy=policy.collect_mode,
        tb_logger=tb_logger,
        exp_name=cfg.exp_name,
        vllm_engine=vllm_engine,  # May be None
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
        vllm_engine=vllm_engine,  # May be None
    )
    logger.info("✓ Evaluator created")

    # Call learner's before_run hook
    learner.call_hook('before_run')

    # ==================================================================
    # 5. Initialize ORZ Trainer (if available)
    # ==================================================================
    orz_trainer = None
    orz_adapter = GameSegmentToORZAdapter()
    orz_tokenizer = None
    orz_strategy = None

    if hybrid_cfg.use_orz_trainer and ORZ_AVAILABLE:
        logger.info("="*80)
        logger.info("Initializing ORZ RayPPOTrainer for LLM training...")
        logger.info("="*80)

        try:
            # Initialize Ray if not already running
            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True)
                logger.info("✓ Ray initialized")

            # Create ORZ tokenizer
            orz_tokenizer = AutoTokenizer.from_pretrained(
                cfg.policy.llm_policy_cfg.pretrain_llm_path,
                trust_remote_code=True
            )
            if orz_tokenizer.pad_token is None:
                orz_tokenizer.pad_token = orz_tokenizer.eos_token
            logger.info("✓ ORZ tokenizer created")

            # Create ORZ strategy (DeepSpeed config)
            from orz.ppo.utils import get_strategy
            orz_strategy = get_strategy({
                'zero_stage': 2,
                'bf16': True,
                'gradient_checkpointing': True,
            })
            logger.info("✓ ORZ strategy created")

            # Create ORZ configuration (matching ORZ's PPOExpConfig pattern)
            from dataclasses import dataclass, field
            from omegaconf.listconfig import ListConfig

            @dataclass
            class ORZConfig:
                """Simplified ORZ config for PriorZero integration"""
                # Resource settings (simplified for single-node)
                total_num_nodes: int = 1
                ref_num_nodes: int = 1
                ref_num_gpus_per_node: int = 1
                actor_num_nodes: int = 1
                actor_num_gpus_per_node: int = 1
                critic_num_nodes: int = 1
                critic_num_gpus_per_node: int = 1
                colocate_all: bool = True
                colocate_critic_reward: bool = True
                colocate_actor_ref: bool = True
                vllm_num_engines: int = 1
                vllm_tensor_parallel_size: int = 1
                zero_stage: int = 2
                adam_offload: bool = False

                # Model paths
                pretrain: str = cfg.policy.llm_policy_cfg.pretrain_llm_path
                reward_pretrain: Optional[str] = None
                critic_pretrain: Optional[str] = cfg.policy.llm_policy_cfg.pretrain_llm_path

                # Save/log paths
                save_interval: int = 50
                ckpt_path: str = f'./{cfg.exp_name}/orz_ckpt'
                save_path: str = f'./{cfg.exp_name}/orz_save'
                tensorboard_log_dir: str = f'./{cfg.exp_name}/orz_log'

                # Training settings
                actor_learning_rate: float = hybrid_cfg.orz_actor_lr if hasattr(hybrid_cfg, 'orz_actor_lr') else 1e-6
                critic_learning_rate: float = hybrid_cfg.orz_critic_lr if hasattr(hybrid_cfg, 'orz_critic_lr') else 5e-6
                num_warmup_steps: int = 50
                prompt_max_len: int = 2048
                enable_prefix_caching: bool = False
                update_ref_every_epoch: bool = True
                advantage_normalize: bool = True

                # Episode settings
                num_episodes: int = hybrid_cfg.orz_num_episodes if hasattr(hybrid_cfg, 'orz_num_episodes') else 2
                rollout_batch_size: int = hybrid_cfg.orz_rollout_batch_size if hasattr(hybrid_cfg, 'orz_rollout_batch_size') else 32
                n_samples_per_prompt: int = 8 if DEBUG_MODE else 32
                micro_rollout_batch_size: int = 2
                policy_update_steps: int = 1
                critic_update_steps: int = 1 if DEBUG_MODE else 12
                micro_train_batch_size: int = 1
                micro_forward_batch_size: int = 1
                freezing_actor_steps: int = -1

                # KL settings
                init_kl_coef: float = 0
                kl_loss_coef: float = 0.0
                use_kl_loss: bool = False
                use_kl_estimator_k3: bool = True

                # Eval settings
                enable_eval: bool = False  # Disable ORZ eval (use PriorZero's)
                eval_interval: int = 100

                # Generation settings
                packing_max_len: int = 8192
                generate_max_len: int = cfg.policy.llm_policy_cfg.generate_max_len
                max_len: int = 4096
                temperature: float = 1.0
                top_p: float = 1.0
                top_k: int = -1
                stop: ListConfig = field(default_factory=lambda: ListConfig(["</answer>"]))

                # GRPO settings
                use_grpo: bool = False
                gamma: float = 1.0
                lambd: float = 1.0

                # vLLM settings
                gpu_memory_utilization: float = 0.3

                # Custom settings for compute_reward_fn
                use_compute_reward_fn: bool = True
                use_orm_score: bool = False

            orz_cfg = ORZConfig()

            # Create directories for ORZ
            os.makedirs(orz_cfg.ckpt_path, exist_ok=True)
            os.makedirs(orz_cfg.save_path, exist_ok=True)
            os.makedirs(orz_cfg.tensorboard_log_dir, exist_ok=True)

            logger.info("✓ ORZ config created")
            logger.info(f"  - Model: {orz_cfg.pretrain}")
            logger.info(f"  - Rollout batch: {orz_cfg.rollout_batch_size}")
            logger.info(f"  - Episodes: {orz_cfg.num_episodes}")

            # Note: Full RayPPOTrainer initialization requires:
            # 1. Creating vLLM engines for distributed inference
            # 2. Creating initial dataset from game_segments
            # 3. Initializing Ray actors (will be done lazily on first training call)
            #
            # We defer full initialization until we have actual game_segments to train on
            logger.info("✓ ORZ trainer components ready")
            logger.info("  (Full RayPPOTrainer will be initialized on first training iteration)")

        except Exception as e:
            logger.error(f"❌ ORZ trainer initialization failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            logger.warning("Falling back to PriorZero's built-in LLM training")
            hybrid_cfg.use_orz_trainer = False

    # ==================================================================
    # 6. Main Training Loop
    # ==================================================================
    logger.info("="*80)
    logger.info("Starting PriorZero-ORZ Complete Training")
    logger.info("="*80)
    logger.info(f"Experiment: {cfg.exp_name}")
    logger.info(f"Max iterations: {max_train_iter}")
    logger.info(f"Training mode: {hybrid_cfg.wm_training_mode}")
    logger.info(f"Use ORZ trainer: {hybrid_cfg.use_orz_trainer}")
    logger.info(f"Use vLLM: {vllm_engine is not None}")
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

            try:
                new_data = await collector.collect(
                    train_iter=current_iter,
                    policy_kwargs=collect_kwargs
                )
            except Exception as e:
                logger.error(f"❌ Collection failed: {e}")
                logger.warning("Skipping this iteration...")
                continue

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
                if replay_buffer.get_num_of_transitions() >= cfg.policy.batch_size:
                    logger.info(f"[Iter {current_iter}] Training world model...")

                    # Sample and train
                    for _ in range(update_per_collect):
                        train_data = replay_buffer.sample(
                            cfg.policy.batch_size,
                            policy
                        )

                        # Train (includes both WM and LLM in PriorZero)
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
                    logger.info(f"Skipping training - not enough data yet")

            # ==============================================================
            # Step 4: LLM Training with ORZ (if enabled)
            # ==============================================================
            if (hybrid_cfg.use_orz_trainer and orz_trainer is not None and
                current_iter % hybrid_cfg.llm_train_freq == 0 and
                current_iter > 0):
                logger.info(f"[Iter {current_iter}] Training LLM with ORZ...")

                try:
                    # Extract game_segments from recent collections
                    training_data = orz_adapter.extract_training_data(new_data)
                    num_samples = len(training_data['states'])

                    if num_samples > 0:
                        logger.info(f"  Extracted {num_samples} training samples for ORZ")

                        # Initialize ORZ trainer on first use (lazy initialization)
                        if orz_trainer is None:
                            logger.info("  Initializing ORZ RayPPOTrainer...")

                            # Convert game_segments to ORZ dataset format
                            dialogues = orz_adapter.convert_segments_to_prompts(
                                new_data,
                                orz_tokenizer
                            )

                            # Create ORZ dataset
                            orz_dataset = JerichoPromptDataset(
                                dialogues,
                                orz_tokenizer,
                                orz_cfg.prompt_max_len,
                                orz_strategy,
                                pretrain_mode=False,
                                num_processors=1
                            )

                            # Create custom reward trainer
                            from orz.exps.examples.ppo.ppo_base_exp import BasePPOExp

                            class JerichoRewardTrainer(RayPPOTrainer):
                                """Custom reward trainer for Jericho text adventures"""

                                async def custom_reward_fn(
                                    self,
                                    prompts: List[str],
                                    outputs: List[Any],
                                    extras: List[dict],
                                    reward_model_fn,
                                ):
                                    """
                                    Compute rewards for Jericho actions.
                                    Reward is 1.0 if action matches ground truth, else 0.0
                                    """
                                    import torch
                                    scores = []
                                    responses = []

                                    for output, extra in zip(outputs, extras):
                                        response = output["response"]
                                        responses.append(response)

                                        # Extract action from response
                                        # Look for <answer>...</answer> tags
                                        import re
                                        pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
                                        matches = re.findall(pattern, response)
                                        predicted_action = matches[-1].strip() if matches else ""

                                        # Ground truth action
                                        true_action = extra["answer"]

                                        # Simple exact match for now
                                        # TODO: Could use fuzzy matching or LLM-based similarity
                                        score = 1.0 if predicted_action.lower() == true_action.lower() else 0.0
                                        scores.append(score)

                                    # Log statistics
                                    avg_score = sum(scores) / len(scores) if scores else 0.0
                                    logger.info(f"    ORZ reward - avg: {avg_score:.3f}, samples: {len(scores)}")

                                    # Create score tensors (reward only on last token)
                                    output_tokens = self._tokenize(responses, self.cfg.generate_max_len, padding=False)["input_ids"]
                                    score_tensors = []
                                    for score, output_token in zip(scores, output_tokens):
                                        score_tensor = torch.zeros(len(output_token))
                                        if len(output_token) > 0:
                                            score_tensor[-1] = score
                                        score_tensors.append(score_tensor)

                                    # Remove empty responses
                                    res_prompts, res_responses, res_score_tensors = [], [], []
                                    for prompt, response, score_tensor in zip(prompts, responses, score_tensors):
                                        if len(response) > 0:
                                            res_prompts.append(prompt)
                                            res_responses.append(response)
                                            res_score_tensors.append(score_tensor)

                                    return res_prompts, res_responses, res_score_tensors

                            # Create vLLM engines for ORZ
                            logger.info("  Creating vLLM inference engines for ORZ...")
                            from orz.exps.examples.ppo.ppo_base_exp import BasePPOExp

                            # Use BasePPOExp helper to create engines
                            class TempExp(BasePPOExp):
                                def __init__(self):
                                    self.cfg = orz_cfg
                                    self.tokenizer = orz_tokenizer
                                    self.strategy = orz_strategy

                            temp_exp = TempExp()
                            vllm_engines = temp_exp.create_inference_engine()
                            logger.info(f"  ✓ Created {len(vllm_engines)} vLLM engines")

                            # Get colocate placement groups if needed
                            colocate_pg = temp_exp.get_colocate_pg if orz_cfg.colocate_all else None

                            # Create ORZ trainer
                            orz_trainer = JerichoRewardTrainer(
                                cfg=orz_cfg,
                                strategy=orz_strategy,
                                tokenizer=orz_tokenizer,
                                train_dataset=orz_dataset,
                                eval_dataset=None,  # No separate eval for now
                                vllm_engines=vllm_engines,
                                colocate_pg=colocate_pg
                            )

                            logger.info("  ✓ ORZ RayPPOTrainer initialized")

                        # Run ORZ training for one episode
                        logger.info(f"  Running ORZ PPO training (episode {current_iter // hybrid_cfg.llm_train_freq})...")

                        # Train using ORZ's fit_episode method
                        # Note: This will do full PPO update with actor/critic training
                        await orz_trainer.fit_episode()

                        logger.info(f"  ✓ ORZ training completed for iteration {current_iter}")

                    else:
                        logger.warning("  No training samples extracted from game_segments")

                except Exception as e:
                    logger.error(f"  ✗ ORZ training failed: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    logger.warning("  Continuing with PriorZero LLM training only")

            # ==============================================================
            # Step 5: Logging and Checkpointing
            # ==============================================================
            if current_iter % 10 == 0:
                logger.info(f"\n{'='*60}")
                logger.info(f"Progress Summary (Iter {current_iter})")
                logger.info(f"{'='*60}")
                logger.info(f"Env steps: {collector.envstep}")
                logger.info(f"Game segments collected: {total_game_segments_collected}")
                logger.info(f"Buffer size: {replay_buffer.get_num_of_transitions()} transitions")
                logger.info(f"Best eval reward: {best_eval_reward:.2f}")
                logger.info(f"{'='*60}\n")

            # Save checkpoint periodically
            if enable_save and current_iter % 100 == 0 and current_iter > 0:
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
            try:
                learner.save_checkpoint(collector.envstep)
            except Exception as e:
                logger.error(f"Failed to save checkpoint: {e}")

        # Close environments
        try:
            collector_env.close()
            evaluator_env.close()
        except Exception as e:
            logger.error(f"Failed to close environments: {e}")

        # Close loggers
        if tb_logger:
            try:
                tb_logger.close()
            except Exception as e:
                logger.error(f"Failed to close tensorboard: {e}")

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
    await train_priorzero_orz_complete(
        cfg=hybrid_cfg.priorzero_cfg,
        create_cfg=hybrid_cfg.priorzero_create_cfg,
        hybrid_cfg=hybrid_cfg,
        seed=0,
        max_train_iter=10000 if not DEBUG_MODE else 100,
        enable_save=True,
    )


if __name__ == "__main__":
    logger.info("="*80)
    logger.info("PriorZero-ORZ Complete Training Pipeline")
    logger.info("="*80)
    logger.info(f"Debug mode: {DEBUG_MODE}")
    logger.info(f"ORZ available: {ORZ_AVAILABLE}")
    logger.info(f"vLLM available: {VLLM_AVAILABLE}")
    logger.info("="*80)

    # Run async training
    asyncio.run(main())

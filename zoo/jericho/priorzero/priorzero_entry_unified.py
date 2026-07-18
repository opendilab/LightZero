"""
Complete PriorZero Entry with VL (Vision-Language) Support

This is the COMPLETE implementation with full training loop.
Supports both text (LLM) and image (VL) inputs.
"""
import sys
import os
from pathlib import Path

# Add project root to path
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parents[3]
if str(project_root) not in sys.path:
    print(f"[SYSTEM] Inserting project root to sys.path: {project_root}")
    sys.path.insert(0, str(project_root))

# Add src/ directory to path so that modules like strategy, models, vllm_utils, utils can be found
src_dir = current_file_path.parent / 'src'
if str(src_dir) not in sys.path:
    print(f"[SYSTEM] Inserting src dir to sys.path: {src_dir}")
    sys.path.insert(0, str(src_dir))

# Add priorzero/ directory itself so sibling modules (prior_generator, vl_config, etc.) can be found
priorzero_dir = str(current_file_path.parent)
if priorzero_dir not in sys.path:
    print(f"[SYSTEM] Inserting priorzero dir to sys.path: {priorzero_dir}")
    sys.path.insert(0, priorzero_dir)

import argparse
from functools import partial
from typing import Tuple, Optional, List

import torch
import torch.distributed as dist
from ding.config import compile_config
from ding.envs import create_env_manager, get_vec_env_setting
from ding.policy import create_policy
from ding.utils import set_pkg_seed, get_rank, get_world_size
from ding.worker import BaseLearner
from tensorboardX import SummaryWriter
from loguru import logger

from lzero.mcts.buffer.game_buffer_priorzero import PriorZeroGameBufferOptimized
from lzero.entry.utils import calculate_update_per_collect


def all_gather_cmd(world_size, obj) -> List:
    """Gather command from all ranks."""
    if world_size <= 1:
        return [obj]
    lst = [None] * dist.get_world_size()
    dist.all_gather_object(lst, obj)
    return lst


def prepare_common_components(rank, cfg, create_cfg, seed):
    """Prepare components common to both LLM and VL."""
    cfg = compile_config(cfg, seed=seed, auto=True, create_cfg=create_cfg)

    # Create environments
    env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)
    collector_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_env_cfg])
    evaluator_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])

    collector_env.seed(seed)
    evaluator_env.seed(seed, dynamic_seed=False)

    # Create policy
    policy = create_policy(cfg.policy, enable_field=['learn', 'collect', 'eval'], exp_name=cfg.exp_name)
    logger.info(f"[Rank {rank}] Policy created")

    # Create logger and learner
    os.makedirs(f'./{cfg.exp_name}/log/', exist_ok=True)
    tb_logger = SummaryWriter(os.path.join(f'./{cfg.exp_name}/log/', 'serial')) if rank == 0 else None
    logger.info(f"[Rank {rank}] TensorBoard logger: ./{cfg.exp_name}/log/")

    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)
    logger.info(f"[Rank {rank}] BaseLearner created")

    # Create replay buffer
    replay_buffer = PriorZeroGameBufferOptimized(cfg.policy)
    logger.info(f"[Rank {rank}] PriorZero replay buffer created")

    return cfg, collector_env, evaluator_env, policy, learner, replay_buffer, tb_logger


def prepare_llm_components(rank, cfg, llm_cfg, strategy, collector_env, evaluator_env, policy, tb_logger, seed):
    """Prepare LLM-specific components for text input."""
    from utils import Profiler, dump_dataclass_cfg_py
    from models.actor import PolicyModel, ReferenceModel
    from vllm_utils.vllm_engine import create_vllm_engine
    from priorzero_datafactory_unified import UnifiedDataProcessor
    from priorzero_trainer import PriorZeroLLMTrainer
    from priorzero_collector_unified import PriorZeroCollector
    from priorzero_evaluator import PriorZeroEvaluator
    from prior_generator import LLMPriorGenerator

    prof = Profiler(log_interval=10, stats_file=f'./{cfg.exp_name}/log/profiler.txt', enable_profile=False)

    if rank == 0:
        dump_dataclass_cfg_py(llm_cfg, path=f"{cfg.exp_name}/llm_cfg.py")
        llm_cfg.save_path = f'./{cfg.exp_name}/llm_ckpt/'

    logger.info(f"[Rank {rank}] Initializing LLM components...")
    set_pkg_seed(seed + rank, use_cuda=True)

    # Reference model
    ref_model = ReferenceModel(strategy=strategy, pretrain=llm_cfg.model_name_or_path) if llm_cfg.rft_kl_coef > 0 else None

    # vLLM engine
    vllm_engine = create_vllm_engine(
        tensor_parallel_size=llm_cfg.vllm_tensor_parallel_size,
        pretrain=llm_cfg.model_name_or_path,
        enable_prefix_caching=llm_cfg.enable_prefix_caching,
        max_model_len=llm_cfg.prompt_max_len + llm_cfg.generate_max_len,
        gpu_memory_utilization=llm_cfg.gpu_memory_utilization,
        vllm_enable_sleep=llm_cfg.vllm_enable_sleep,
    )
    logger.info(f'[Rank {rank}] vLLM engine created')

    # Data processor
    world_size = getattr(strategy, "world_size", 1)
    data_processor = UnifiedDataProcessor(
        rank=rank,
        world_size=world_size,
        vllm_engine=vllm_engine,
        strategy=strategy,
        model_path=llm_cfg.model_name_or_path,
        exp_name=cfg.exp_name if rank == 0 else None,
        obs_type='text',
    )

    # Policy model
    policy_model = PolicyModel(
        strategy=strategy,
        pretrain=llm_cfg.model_name_or_path,
        vllm_engine=vllm_engine,
        max_steps=llm_cfg.max_steps
    )

    # Trainer
    trainer = PriorZeroLLMTrainer(
        cfg=llm_cfg,
        pretrain=llm_cfg.model_name_or_path,
        strategy=strategy,
        vllm_engine=vllm_engine,
        policy_model=policy_model,
        reference_model=ref_model,
        exp_name=cfg.exp_name if rank == 0 else None,
        tb_logger=tb_logger if rank == 0 else None,
        llm_save_freq=llm_cfg.llm_save_freq
    )

    # Prior generator
    prior_generator = LLMPriorGenerator(
        vllm_engine=vllm_engine,
        data_processor=data_processor,
        model_name=llm_cfg.model_name_or_path,
        use_cot=llm_cfg.use_cot,
    )

    # Collector
    collector = PriorZeroCollector(
        env=collector_env,
        policy=policy.collect_mode,
        llm_config=llm_cfg,
        tb_logger=tb_logger,
        exp_name=cfg.exp_name,
        policy_config=cfg.policy,
        data_processor=data_processor,
        prior_generator=prior_generator,
        obs_type='text',
    )
    collector.prof = prof

    # Evaluator
    evaluator = PriorZeroEvaluator(
        eval_freq=cfg.policy.eval_freq,
        n_evaluator_episode=cfg.env.n_evaluator_episode,
        stop_value=cfg.env.stop_value,
        env=evaluator_env,
        policy=policy.eval_mode,
        tb_logger=tb_logger,
        exp_name=cfg.exp_name,
        policy_config=cfg.policy,
        llm_config=llm_cfg,
        data_processor=data_processor,
    )

    logger.info(f"[Rank {rank}] ✓ LLM components initialized")

    return {
        'prior_generator': prior_generator,
        'vllm_engine': vllm_engine,
        'policy_model': policy_model,
        'ref_model': ref_model,
        'trainer': trainer,
        'data_processor': data_processor,
        'collector': collector,
        'evaluator': evaluator,
        'prof': prof,
    }


def prepare_vl_components(rank, cfg, vl_cfg, strategy, collector_env, evaluator_env, policy, tb_logger, seed):
    """Prepare VL-specific components for image input."""
    from utils import Profiler, dump_dataclass_cfg_py
    from models.actor import PolicyModel, ReferenceModel
    from vl_engine import create_vl_engine
    from priorzero_datafactory_unified import UnifiedDataProcessor
    from priorzero_trainer import PriorZeroLLMTrainer  # Can reuse for VL
    from priorzero_collector_unified import PriorZeroCollector
    from priorzero_evaluator import PriorZeroEvaluator
    from prior_generator import VLPriorGenerator

    prof = Profiler(log_interval=10, stats_file=f'./{cfg.exp_name}/log/profiler.txt', enable_profile=False)

    if rank == 0:
        dump_dataclass_cfg_py(vl_cfg, path=f"{cfg.exp_name}/vl_cfg.py")
        vl_cfg.save_path = f'./{cfg.exp_name}/vl_ckpt/'

    logger.info(f"[Rank {rank}] Initializing VL components...")
    set_pkg_seed(seed + rank, use_cuda=True)

    # Reference model
    ref_model = ReferenceModel(strategy=strategy, pretrain=vl_cfg.model_name_or_path) if vl_cfg.rft_kl_coef > 0 else None

    # VL engine
    # Determine limit_mm_per_prompt based on vlm_image_mode
    vlm_image_mode = getattr(vl_cfg, 'vlm_image_mode', 'current_only')
    if vlm_image_mode == "current_only":
        limit_mm_per_prompt = {"image": 1}
    else:
        # first_and_current or all_history: need up to history_length + 1 images
        limit_mm_per_prompt = {"image": vl_cfg.history_length + 1}

    vl_engine = create_vl_engine(
        model_name=vl_cfg.vl_model_type,
        model_path=vl_cfg.model_name_or_path,
        tensor_parallel_size=vl_cfg.tensor_parallel_size,
        gpu_memory_utilization=vl_cfg.gpu_memory_utilization,
        max_model_len=vl_cfg.prompt_max_len + vl_cfg.generate_max_len,
        limit_mm_per_prompt=limit_mm_per_prompt,
    )
    logger.info(f'[Rank {rank}] VL engine created: {vl_cfg.vl_model_type}')

    # Data processor
    world_size = getattr(strategy, "world_size", 1)
    data_processor = UnifiedDataProcessor(
        rank=rank,
        world_size=world_size,
        vllm_engine=vl_engine,
        strategy=strategy,
        model_path=vl_cfg.model_name_or_path,
        exp_name=cfg.exp_name if rank == 0 else None,
        obs_type='image',
    )

    # Policy model
    policy_model = PolicyModel(
        strategy=strategy,
        pretrain=vl_cfg.model_name_or_path,
        vllm_engine=vl_engine,
        max_steps=vl_cfg.max_steps
    )

    # Trainer
    trainer = PriorZeroLLMTrainer(
        cfg=vl_cfg,
        pretrain=vl_cfg.model_name_or_path,
        strategy=strategy,
        vllm_engine=vl_engine,
        policy_model=policy_model,
        reference_model=ref_model,
        exp_name=cfg.exp_name if rank == 0 else None,
        tb_logger=tb_logger if rank == 0 else None,
        instance_name="vl_ppo",
        llm_save_freq=vl_cfg.vl_save_freq
    )

    # Prior generator
    prior_generator = VLPriorGenerator(
        vl_engine=vl_engine,
        model_name=vl_cfg.model_name_or_path,
        use_cot=vl_cfg.use_cot,
        game_description=getattr(vl_cfg, 'game_description', ''),
        vlm_image_mode=vlm_image_mode,
        prompt_style=getattr(vl_cfg, 'prompt_style', 'concise'),
        logprob_extraction_mode=getattr(vl_cfg, 'logprob_extraction_mode', 'approximate'),
    )

    # Collector
    collector = PriorZeroCollector(
        env=collector_env,
        policy=policy.collect_mode,
        llm_config=vl_cfg,
        tb_logger=tb_logger,
        exp_name=cfg.exp_name,
        policy_config=cfg.policy,
        data_processor=data_processor,
        prior_generator=prior_generator,
        obs_type='image',
        env_id=cfg.env.env_id,  # Pass env_id for action mapping
    )
    collector.prof = prof

    # Evaluator
    evaluator = PriorZeroEvaluator(
        eval_freq=cfg.policy.eval_freq,
        n_evaluator_episode=cfg.env.n_evaluator_episode,
        stop_value=cfg.env.stop_value,
        env=evaluator_env,
        policy=policy.eval_mode,
        tb_logger=tb_logger,
        exp_name=cfg.exp_name,
        policy_config=cfg.policy,
        llm_config=vl_cfg,
        data_processor=data_processor,
        prior_generator=prior_generator,
        obs_type='image',
        env_id=cfg.env.env_id,
    )

    logger.info(f"[Rank {rank}] ✓ VL components initialized")

    return {
        'prior_generator': prior_generator,
        'vl_engine': vl_engine,
        'policy_model': policy_model,
        'ref_model': ref_model,
        'trainer': trainer,
        'data_processor': data_processor,
        'collector': collector,
        'evaluator': evaluator,
        'prof': prof,
    }


def train_unified(
    cfg: dict,
    create_cfg: dict,
    prior_cfg,  # LLM or VL config
    seed: int = 0,
    max_train_iter: int = int(1e6),
    max_env_step: Optional[int] = int(1e10),
    enable_profile: bool = False,
    is_text_input: bool = True,
):
    """
    Unified training function supporting both LLM and VL.

    Args:
        cfg: Main configuration
        create_cfg: Creation configuration
        prior_cfg: LLM or VL configuration
        seed: Random seed
        max_train_iter: Maximum training iterations
        max_env_step: Maximum environment steps
        enable_profile: Whether to enable profiling
        is_text_input: Whether using text input (True) or image input (False)
    """
    rank = int(os.environ.get("RANK", "0"))

    # Initialize strategy
    from strategy.deepspeed import get_strategy, torch_dist_barrier_and_cuda_sync
    strategy = get_strategy(prior_cfg)
    strategy.print(prior_cfg)
    strategy.setup_distributed()
    world_size = getattr(strategy, "world_size", 1)

    # Prepare common components
    cfg, collector_env, evaluator_env, policy, learner, replay_buffer, tb_logger = prepare_common_components(
        rank, cfg, create_cfg, seed
    )
    batch_size = cfg.policy.batch_size

    # Prepare input-specific components
    if is_text_input:
        components = prepare_llm_components(
            rank, cfg, prior_cfg, strategy, collector_env, evaluator_env, policy, tb_logger, seed
        )
        engine_name = "vLLM"
    else:
        components = prepare_vl_components(
            rank, cfg, prior_cfg, strategy, collector_env, evaluator_env, policy, tb_logger, seed
        )
        engine_name = "VL"

    # Extract components
    prior_engine = components['vllm_engine'] if is_text_input else components['vl_engine']
    policy_model = components['policy_model']
    trainer = components['trainer']
    data_processor = components['data_processor']
    collector = components['collector']
    evaluator = components['evaluator']
    prof = components['prof']

    # Set llm_cfg on policy so _forward_eval/_forward_collect can access it
    policy.llm_cfg = prior_cfg

    torch_dist_barrier_and_cuda_sync()
    learner.call_hook('before_run')

    logger.info(f"[Rank {rank}] Starting training loop with {engine_name} prior...")

    # Validate VL config consistency (e.g. enable_rft + vl_fixed conflict)
    if not is_text_input and hasattr(prior_cfg, 'validate'):
        prior_cfg.validate()

    # =========================================================================
    # Alternating Training Schedule Setup (aligned with sync_ddp)
    # =========================================================================
    train_schedule = prior_cfg.train_schedule
    train_alternate = train_schedule["alternate"]
    enable_world_model = prior_cfg.enable_world_model
    enable_rft = prior_cfg.enable_rft

    if train_alternate:
        current_phase = train_schedule["start_phase"]
        last_wm_train_iter = 0
        last_llm_train_iter = 0
    else:
        current_phase = None

    # =========================================================================
    # Main Training Loop
    # =========================================================================
    while True:
        if collector.envstep >= max_env_step or learner.train_iter >= max_train_iter:
            break

        # Periodic loop status log (every 500 envsteps, rank 0 only)
        if rank == 0 and collector.envstep % 500 < 10:
            phase_str = current_phase if train_alternate else "joint"
            logger.info(
                f"[Loop Status] envstep={collector.envstep}, wm_iter={learner.train_iter}, "
                f"llm_iter={trainer.global_step if hasattr(trainer, 'global_step') else 'N/A'}, "
                f"phase={phase_str}, enable_rft={enable_rft}, enable_wm={enable_world_model}"
            )

        cmd = 0
        priorzero_batch = None

        # Evaluation
        # if learner.train_iter == 0 or evaluator.should_eval(learner.train_iter):
        if learner.train_iter > 0 and evaluator.should_eval(learner.train_iter):
            logger.info(f"\n[Rank {rank}: Iter {learner.train_iter}] Evaluating...")
            if prior_cfg.vllm_enable_sleep and prior_engine is not None:
                prior_engine.wake_up()
            stop, reward = evaluator.eval(
                train_iter=learner.train_iter,
                envstep=collector.envstep
            )
            if prior_cfg.vllm_enable_sleep and prior_engine is not None:
                prior_engine.sleep()
            torch.cuda.empty_cache()

        # Wake up engine
        if prior_cfg.vllm_enable_sleep and prior_engine is not None:
            prior_engine.wake_up()

        # Data collection
        with prof.block("collect", rank=rank):
            new_data = collector.collect(
                train_iter=learner.train_iter,
                policy_kwargs={'temperature': 0.25, 'epsilon': 0.0}
            )

        # Log output based on input type
        if is_text_input:
            data_processor.get_llm_output_log(
                wm_train_iter=learner.train_iter,
                llm_train_iter=policy_model.train_iter
            )
        else:
            # VL: use prior_generator's log method
            prior_generator = components.get('prior_generator')
            if prior_generator and hasattr(prior_generator, 'get_vl_output_log'):
                prior_generator.get_vl_output_log(
                    wm_train_iter=learner.train_iter,
                    vl_train_iter=policy_model.train_iter
                )

        # Sleep engine
        if prior_cfg.vllm_enable_sleep and prior_engine is not None:
            prior_engine.sleep()

        # Push to replay buffer
        replay_buffer.push_game_segments(new_data)
        replay_buffer.remove_oldest_data_to_fit()

        num_of_transitions = replay_buffer.get_num_of_transitions()
        new_num_of_transitions = num_of_transitions - replay_buffer.last_pos_in_transition

        logger.info(
            f"[Data Collection] Rank {rank} | "
            f"Total transitions: {num_of_transitions} | "
            f"New transitions: {new_num_of_transitions}"
        )

        # TB logging for collect metrics
        if tb_logger is not None:
            tb_logger.add_scalar('collect/num_transitions', num_of_transitions, collector.envstep)
            tb_logger.add_scalar('collect/new_transitions', new_num_of_transitions, collector.envstep)

        torch_dist_barrier_and_cuda_sync()

        # =====================================================================
        # World Model Training (gated by schedule)
        # =====================================================================
        if enable_world_model and (not train_alternate or current_phase == "wm"):
            if not (num_of_transitions > batch_size):
                logger.warning(
                    f'[WM Training] Data insufficient: batch_size={batch_size}, buffer={num_of_transitions}'
                )
                cmd = 0
            else:
                cmd = 1

            if min(all_gather_cmd(world_size=world_size, obj=cmd)) == 0:
                continue

            update_per_collect = calculate_update_per_collect(cfg, new_data, world_size=world_size)
            logger.info(
                f"[WM Training] Rank {rank} | Iter {learner.train_iter} | "
                f"Updates: {update_per_collect}"
            )

            for i in range(update_per_collect):
                with prof.block("train_world_model", rank=rank):
                    train_data = replay_buffer.sample(batch_size, policy)
                    train_data.append(learner.train_iter)
                    log_vars = learner.train(train_data, collector.envstep)
                    if cfg.policy.use_priority:
                        replay_buffer.update_priority(train_data, log_vars[0]['value_priority_orig'])

            policy.recompute_pos_emb_diff_and_clear_cache()

            # TB logging for WM training
            # NOTE: DI-engine's BaseLearner already logs all _monitor_vars_learn() metrics
            # under "learner_iter/" prefix (averaged over log_show_after_iter).
            # We only log the phase-tracking scalar here; per-metric logging is handled by the learner.
            if tb_logger is not None:
                tb_logger.add_scalar('train/wm_train_iter', learner.train_iter, collector.envstep)

            # Phase switching: WM -> LLM/VL
            if train_alternate and learner.train_iter - last_wm_train_iter >= train_schedule["wm_update_iters"]:
                current_phase = "llm"
                last_wm_train_iter = learner.train_iter
                replay_buffer.mark_latest_transitions_consumed()
                logger.info(f"[WM Training][Rank {rank}] Switching to {'VL' if not is_text_input else 'LLM'} training phase at wm iter: {learner.train_iter}")
                continue

        # =====================================================================
        # LLM/VL Training (gated by schedule)
        # =====================================================================
        if enable_rft and (not train_alternate or current_phase == "llm"):
            new_num_of_transitions = replay_buffer.get_num_of_transitions() - replay_buffer.last_pos_in_transition
            logger.info(
                f"[{engine_name} Training] Rank {rank} | "
                f"Total transitions: {num_of_transitions} | "
                f"New transitions: {new_num_of_transitions}"
            )

            with prof.block("fetch_latest_batch", rank=rank):
                priorzero_batch = replay_buffer.fetch_latest_batch(batch_size=-1, policy=policy)
                torch.cuda.empty_cache()

            with prof.block("train_prior_model", rank=rank):
                llm_need_sample_cnt = prior_cfg.train_batch_size * prior_cfg.max_rollout_staleness // world_size
                flag, train_samples = data_processor.make_llm_train_samples(
                    priorzero_batch, ddp=True, max_samples=llm_need_sample_cnt,
                    prior_generator=components.get('prior_generator') if not is_text_input else None,
                )

                if not flag:
                    local_llm_ready = 0
                else:
                    local_llm_ready = 1

                gathered_llm_ready = all_gather_cmd(world_size=world_size, obj=local_llm_ready)

                if min(gathered_llm_ready) == 0:
                    logger.info(
                        f"[Rank {rank}] Skip {engine_name} training: not all ranks ready. "
                        f"ready_flags={gathered_llm_ready}, local={local_llm_ready}, required={llm_need_sample_cnt}, got={len(train_samples)}"
                    )
                    continue

                trainer.train_batch(train_samples, collect_env_steps=collector.envstep)
                replay_buffer.mark_latest_transitions_consumed()

                torch_dist_barrier_and_cuda_sync()

                # Phase switching: LLM/VL -> WM
                if train_alternate and trainer.global_step - last_llm_train_iter >= train_schedule["llm_update_iters"]:
                    current_phase = "wm"
                    last_llm_train_iter = trainer.global_step
                    if data_processor.value_normalizer is not None:
                        data_processor.value_normalizer.clear()
                    logger.info(f"[Rank {rank}] Switching to World Model training phase at {engine_name} iter: {trainer.global_step}")

        # Safety fallback: if RFT is disabled but the alternating scheduler
        # switched to "llm" phase, immediately fall back to WM so the loop
        # does not spin forever doing only data collection.
        if not enable_rft and train_alternate and current_phase == "llm":
            current_phase = "wm"
            logger.info(
                f"[Rank {rank}] enable_rft=False, auto-switching from '{engine_name}' phase back to WM phase"
            )

    logger.info(f"[Rank {rank}] Training completed!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='PriorZero with VL Support')

    # Common arguments
    parser.add_argument('--input_type', type=str, required=True, choices=['text', 'image'])
    parser.add_argument('--env_id', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max_iter', type=int, default=int(1e6))
    parser.add_argument('--quick_test', action='store_true', default=False)
    parser.add_argument('--enable_profile', action='store_true', default=False)

    # Text-specific
    parser.add_argument('--llm_model', type=str, default='qwen2.5-1.5b')

    # Shared LLM/VL arguments
    parser.add_argument('--use_cot', action='store_true', default=True,
                        help='Enable Chain-of-Thought reasoning (default: True)')
    parser.add_argument('--no_cot', action='store_true', default=False,
                        help='Disable Chain-of-Thought reasoning')
    parser.add_argument('--cot_weight', type=float, default=0.1,
                        help='Weight for CoT prefix tokens in loss (default: 0.1)')
    parser.add_argument('--vl_fixed', action='store_true', default=True,
                        help='Freeze VL model (inference only, no VL training) (default: True)')
    parser.add_argument('--no_vl_fixed', action='store_true', default=False,
                        help='Enable VL training (unfreeze)')
    parser.add_argument('--mcts_mode', type=str, default='llm_plus_wm_logits',
                        choices=['llm_logits', 'wm_logits', 'llm_plus_wm_logits'],
                        help='MCTS root logits mode (default: llm_plus_wm_logits)')

    # Image-specific
    parser.add_argument('--vl_model', type=str, default='Qwen2.5-VL-7b')
    parser.add_argument('--use_prior', action='store_true', default=True)
    parser.add_argument('--vlm_image_mode', type=str, default='current_only',
                        choices=['current_only', 'first_and_current', 'all_history'],
                        help='VLM image mode: how many images to send to VL model (default: current_only)')
    parser.add_argument('--prompt_style', type=str, default='legacy',
                        choices=['concise', 'legacy'],
                        help='Prompt style: concise (shorter, better for small VLMs) or legacy (verbose)')

    args = parser.parse_args()

    # Resolve --no_xxx flags (explicit --no_cot / --no_vl_fixed override defaults)
    args.use_cot = not args.no_cot
    args.vl_fixed = not args.no_vl_fixed

    print(f"\n{'='*80}")
    print(f"PriorZero Training with {'LLM' if args.input_type == 'text' else 'VL'} Prior")
    print(f"{'='*80}")
    print(f"Input Type: {args.input_type}")
    print(f"Environment: {args.env_id}")
    print(f"Seed: {args.seed}")
    print(f"Quick Test: {args.quick_test}")
    print(f"Use CoT: {args.use_cot}")
    if args.input_type == 'image':
        print(f"VL Fixed: {args.vl_fixed}")
        print(f"MCTS Mode: {args.mcts_mode}")
        print(f"VLM Image Mode: {args.vlm_image_mode}")
    print(f"{'='*80}\n")

    if args.input_type == 'text':
        from priorzero_config import get_priorzero_config, get_priorzero_debug_config

        if args.quick_test:
            main_cfg, create_cfg, llm_cfg = get_priorzero_debug_config(
                args.env_id, args.seed, use_cot=args.use_cot,
                exp_name=f'data_priorzero_complete/text_{args.env_id}_seed{args.seed}',
                model_key=args.llm_model,
            )
        else:
            main_cfg, create_cfg, llm_cfg = get_priorzero_config(
                args.env_id, args.seed, use_cot=args.use_cot,
                exp_name=f'data_priorzero_complete/text_{args.env_id}_seed{args.seed}',
                model_key=args.llm_model,
                multi_gpu=True
            )

        train_unified(
            main_cfg, create_cfg, llm_cfg,
            seed=args.seed,
            max_train_iter=args.max_iter,
            enable_profile=args.enable_profile,
            is_text_input=True,
        )

    else:
        from vl_config import get_priorzero_vl_config

        # Build a clean env short name: strip common suffixes
        env_short = args.env_id
        for suffix in ['NoFrameskip-v4', '-v2', '-v1', '-v0', '-v5']:
            if env_short.endswith(suffix):
                env_short = env_short[:-len(suffix)]
                break

        from datetime import datetime
        timestamp = datetime.now().strftime('%y%m%d_%H%M%S')
        cot_tag = f"cot{args.cot_weight}" if args.use_cot else "noCot"
        fixed_tag = "vlFixed" if args.vl_fixed else "vlTrain"
        exp_name = (
            f'data_priorzero_complete/'
            f'{env_short}_{args.vl_model}_{fixed_tag}/'
            f'{cot_tag}_mcts_{args.mcts_mode}_img_{args.vlm_image_mode}/'
            f'seed{args.seed}_{timestamp}'
        )

        main_cfg, create_cfg, vl_cfg = get_priorzero_vl_config(
            args.env_id, args.seed,
            exp_name=exp_name,
            vl_model_key=args.vl_model,
            use_prior=args.use_prior,
            multi_gpu=int(os.environ.get('WORLD_SIZE', '1')) > 1,
            quick_test=args.quick_test,
        )

        # Apply CLI overrides to vl_cfg
        if vl_cfg is not None:
            vl_cfg.use_cot = args.use_cot
            vl_cfg.cot_weight = args.cot_weight
            vl_cfg.vl_fixed = args.vl_fixed
            vl_cfg.mcts_root_logits_dict.mode = args.mcts_mode
            vl_cfg.vlm_image_mode = args.vlm_image_mode
            vl_cfg.prompt_style = args.prompt_style
            # Ensure consistency: vl_fixed=True → disable PPO training
            if vl_cfg.vl_fixed:
                vl_cfg.enable_rft = False

        train_unified(
            main_cfg, create_cfg, vl_cfg,
            seed=args.seed,
            max_train_iter=args.max_iter,
            enable_profile=args.enable_profile,
            is_text_input=False,
        )


if __name__ == "__main__":
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    main()

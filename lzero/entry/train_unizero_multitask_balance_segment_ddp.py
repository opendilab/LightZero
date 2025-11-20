import logging
import os
from functools import partial
from typing import Tuple, Optional, List, Dict, Any

import torch
import numpy as np
from ding.config import compile_config
from ding.envs import create_env_manager, get_vec_env_setting
from ding.policy import create_policy
from ding.rl_utils import get_epsilon_greedy_fn
from ding.utils import set_pkg_seed, get_rank, get_world_size
from ding.worker import BaseLearner
from tensorboardX import SummaryWriter

from lzero.entry.utils import log_buffer_memory_usage, TemperatureScheduler
from lzero.policy import visit_count_temperature
from lzero.worker import MuZeroEvaluator as Evaluator
from lzero.worker import MuZeroSegmentCollector as Collector
from ding.utils import EasyTimer
import torch.nn.functional as F
import torch.distributed as dist
import concurrent.futures
from lzero.model.unizero_world_models.transformer import set_curriculum_stage, CurriculumLoRALinear

from collections import defaultdict
import math
from .utils import (
    freeze_non_lora_parameters,
    compute_task_weights,
    log_module_trainable_status,
    log_param_statistics,
    tasks_per_stage,
    compute_unizero_mt_normalized_stats,
    allocate_batch_size
)

# A global dictionary to store the most recent evaluation return for each task.
# Format: {task_id: eval_episode_return_mean}
GLOBAL_EVAL_RETURNS: Dict[int, float] = defaultdict(lambda: None)

# Timeout for the evaluation process in seconds.
EVALUATION_TIMEOUT = 12000  # 200 minutes


class CurriculumController:
    """
    Overview:
        Manages the curriculum learning stages for a multi-task policy.
        It tracks the number of solved tasks and training iterations to decide when to transition
        to the next curriculum stage, which typically involves freezing parts of the model
        and activating new LoRA adapters.
    """

    def __init__(self, cfg: 'EasyDict', policy: 'Policy') -> None:
        """
        Overview:
            Initializes the CurriculumController.
        Arguments:
            - cfg (:obj:`EasyDict`): The experiment configuration.
            - policy (:obj:`Policy`): The policy being trained.
        """
        world_model_cfg = cfg.policy.model.world_model_cfg
        self.stage_num: int = world_model_cfg.curriculum_stage_num
        self.min_stage0_iters: int = world_model_cfg.min_stage0_iters
        self.max_stage_iters: int = world_model_cfg.max_stage_iters
        self.policy: 'Policy' = policy

        # Flag to determine if curriculum learning should also be applied to the encoder.
        # Defaults to False for backward compatibility.
        self.apply_curriculum_to_encoder: bool = getattr(world_model_cfg, 'apply_curriculum_to_encoder', False)
        logging.info(f"[CurriculumController] Initialized. Curriculum will be applied to Encoder: {self.apply_curriculum_to_encoder}")

        self.stage: int = 0
        self.last_switch_iter: int = 0
        self.last_solved_count: int = 0  # Snapshot of the last count of solved tasks

    def step(self, solved_count: int, unsolved_count: int, train_iter: int) -> bool:
        """
        Overview:
            Checks if the curriculum should transition to the next stage and performs the switch if needed.
            This method should be called at the end of each training loop.
        Arguments:
            - solved_count (:obj:`int`): The current total number of solved tasks.
            - unsolved_count (:obj:`int`): The current number of tasks yet to be solved.
            - train_iter (:obj:`int`): The current training iteration.
        Returns:
            - bool: True if a stage switch occurred, False otherwise.
        """
        # --- Stage 0 is a mandatory training phase for a minimum number of iterations ---
        if self.stage == 0 and train_iter < self.min_stage0_iters:
            return False

        # --- Determine if a stage switch is necessary ---
        should_switch = False

        # 1. Trigger based on task progress
        newly_solved = solved_count - self.last_solved_count
        remaining_lora_stages = self.stage_num - 1 - self.stage  # Stage 0 doesn't use LoRA
        if remaining_lora_stages > 0:
            # Calculate tasks per stage (tps) for the remaining unsolved tasks
            tps = tasks_per_stage(unsolved_count, remaining_lora_stages)
            if newly_solved >= tps:
                should_switch = True

        # 2. Trigger based on maximum iterations per stage
        if train_iter - self.last_switch_iter >= self.max_stage_iters:
            should_switch = True

        # --- Execute the stage switch ---
        if should_switch and self.stage < self.stage_num - 1:
            is_entering_stage1 = (self.stage == 0)
            self.stage += 1
            
            world_model = self.policy._learn_model.world_model
            vit_encoder = world_model.tokenizer.encoder
            transformer_backbone = world_model.transformer

            # --- Apply curriculum stage update and freeze parameters accordingly ---

            # 1. Conditionally apply to ViT Encoder based on configuration
            if self.apply_curriculum_to_encoder:
                logging.info(f"[Curriculum] Applying curriculum stage {self.stage} to ViT Encoder.")
                set_curriculum_stage(vit_encoder, self.stage)
                if is_entering_stage1:
                    logging.info("[Curriculum] Entering Stage 1. Freezing non-LoRA parameters in ViT Encoder.")
                    freeze_non_lora_parameters(vit_encoder, freeze=True, verbose=True)
                log_module_trainable_status(vit_encoder, "ViT Encoder")
            else:
                logging.info("[Curriculum] Skipping curriculum stage update for ViT Encoder as per configuration.")
                log_module_trainable_status(vit_encoder, "ViT Encoder (Curriculum Not Applied)")

            # 2. Always apply to Transformer Decoder
            logging.info(f"[Curriculum] Applying curriculum stage {self.stage} to Transformer Backbone.")
            set_curriculum_stage(transformer_backbone, self.stage)
            if is_entering_stage1:
                logging.info("[Curriculum] Entering Stage 1. Freezing non-LoRA parameters in Transformer Backbone.")
                freeze_non_lora_parameters(transformer_backbone, freeze=True, verbose=True)
            log_module_trainable_status(transformer_backbone, "Transformer Backbone")

            logging.info(
                f'[Curriculum] Switched to stage {self.stage} '
                f'(solved={solved_count}, unsolved={unsolved_count}, iter={train_iter})'
            )

            # Log parameter statistics after the switch
            updated_params = sum(p.requires_grad for p in self.policy._learn_model.world_model.parameters())
            total_params = sum(1 for _ in self.policy._learn_model.world_model.parameters())
            logging.info(f'{updated_params}/{total_params} parameters in the world model will be optimized.')
            log_param_statistics(self.policy._learn_model.world_model)

            self.last_solved_count = solved_count
            self.last_switch_iter = train_iter
            return True

        return False


def safe_eval(
        evaluator: Evaluator,
        learner: BaseLearner,
        collector: Collector,
        rank: int,
        world_size: int
) -> Tuple[Optional[bool], Optional[Dict[str, Any]]]:
    """
    Overview:
        Executes the evaluation process with a timeout to prevent the training from stalling.
    Arguments:
        - evaluator (:obj:`Evaluator`): The evaluator instance.
        - learner (:obj:`BaseLearner`): The learner instance, used to save checkpoints.
        - collector (:obj:`Collector`): The collector instance, used to get the current envstep.
        - rank (:obj:`int`): The rank of the current process.
        - world_size (:obj:`int`): The total number of processes.
    Returns:
        - Tuple[Optional[bool], Optional[Dict[str, Any]]]: A tuple containing the stop flag and the reward dictionary
          if evaluation succeeds. Returns (None, None) on timeout or error.
    """
    try:
        logging.info(f"========= Evaluation starting on Rank {rank}/{world_size} =========")
        # Ensure the stop_event is clear before starting a new evaluation.
        evaluator.stop_event.clear()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit the evaluation task.
            future = executor.submit(evaluator.eval, learner.save_checkpoint, learner.train_iter, collector.envstep)
            try:
                stop_flag, reward_dict = future.result(timeout=EVALUATION_TIMEOUT)
            except concurrent.futures.TimeoutError:
                # Set the stop_event to terminate the stuck evaluation thread.
                evaluator.stop_event.set()
                logging.error(f"Evaluation timed out on Rank {rank}/{world_size} after {EVALUATION_TIMEOUT} seconds.")
                return None, None

        logging.info(f"====== Evaluation finished on Rank {rank}/{world_size} ======")
        return stop_flag, reward_dict
    except Exception as e:
        logging.error(f"An error occurred during evaluation on Rank {rank}/{world_size}: {e}", exc_info=True)
        return None, None


def train_unizero_multitask_balance_segment_ddp(
        input_cfg_list: List[Tuple[int, Tuple[dict, dict]]],
        seed: int = 0,
        model: Optional[torch.nn.Module] = None,
        model_path: Optional[str] = None,
        max_train_iter: Optional[int] = int(1e10),
        max_env_step: Optional[int] = int(1e10),
        benchmark_name: str = "atari"
) -> 'Policy':
    """
    Overview:
        The main training entry point for UniZero in a multi-task, curriculum-based setting using DDP.
        This function orchestrates distributed data collection, training, and evaluation across multiple tasks.
        The curriculum learning strategy involves:
          - Defining a `target_return` for each task.
          - Moving tasks to a `solved_task_pool` once they achieve their target return, excluding them from
            further training and collection.
          - Progressing through curriculum stages where the model's backbone is frozen, and only specialized
            modules (like LoRA) are trained on harder, unsolved tasks.
        This allows the model to first learn general features and then specialize on difficult tasks without
        catastrophic forgetting.
    Arguments:
        - input_cfg_list (:obj:`List[Tuple[int, Tuple[dict, dict]]]`): A list of configurations for each task.
        - seed (:obj:`int`): The random seed.
        - model (:obj:`Optional[torch.nn.Module]`): An optional pre-existing model instance.
        - model_path (:obj:`Optional[str]`): Path to a pre-trained model checkpoint file.
        - max_train_iter (:obj:`Optional[int]`): The maximum number of training iterations.
        - max_env_step (:obj:`Optional[int]`): The maximum number of environment steps.
        - benchmark_name (:obj:`str`): The name of the benchmark (e.g., "atari", "dmc") to load normalization scores.
    Returns:
        - Policy: The trained policy.
    """
    # --- Initialization and DDP Setup ---
    logging.basicConfig(level=logging.INFO)
    rank = get_rank()
    world_size = get_world_size()
    timer = EasyTimer()

    # --- Benchmark Score Initialization ---
    if benchmark_name == "atari":
        RANDOM_SCORES = np.array([
            227.8, 5.8, 222.4, 210.0, 14.2, 2360.0, 0.1, 1.7, 811.0, 10780.5,
            152.1, 0.0, 65.2, 257.6, 1027.0, 29.0, 52.0, 1598.0, 258.5, 307.3,
            -20.7, 24.9, 163.9, 11.5, 68.4, 533.4
        ])
        HUMAN_SCORES = np.array([
            7127.7, 1719.5, 742.0, 8503.3, 753.1, 37187.5, 12.1, 30.5, 7387.8, 35829.4,
            1971.0, 29.6, 4334.7, 2412.5, 30826.4, 302.8, 3035.0, 2665.5, 22736.3, 6951.6,
            14.6, 69571.3, 13455.0, 7845.0, 42054.7, 11693.2
        ])
        new_order = [
            20, 19, 24, 6, 0, 8, 14, 23, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 15, 16, 17, 18, 21, 25, 22, 7
        ]
        new_RANDOM_SCORES = RANDOM_SCORES[new_order]
        new_HUMAN_SCORES = HUMAN_SCORES[new_order]
    elif benchmark_name == "dmc":
        new_RANDOM_SCORES = np.zeros(26)
        new_HUMAN_SCORES = np.ones(26) * 1000
    else:
        raise ValueError(f"Unsupported benchmark_name: {benchmark_name}")

    # --- Task Distribution Across Ranks ---
    total_tasks = len(input_cfg_list)
    tasks_per_rank = total_tasks // world_size
    remainder = total_tasks % world_size
    start_idx = rank * tasks_per_rank + min(rank, remainder)
    end_idx = start_idx + tasks_per_rank + (1 if rank < remainder else 0)
    tasks_for_this_rank = input_cfg_list[start_idx:end_idx]

    if not tasks_for_this_rank:
        logging.warning(f"Rank {rank}: No tasks assigned. Process will idle but maintain DDP communication.")
        # An idle process must still participate in collective communications.
        # The main loop handles this by waiting at barriers.
        while True:
            dist.barrier()  # Wait for other processes
            dist.barrier()  # Sync after potential training step
            # A mechanism to terminate idle processes would be needed here,
            # for now, they sync and wait.
            # This part requires a robust termination signal from active processes.
    
    logging.info(f"Rank {rank}/{world_size} is handling tasks from index {start_idx} to {end_idx - 1}.")

    # --- Environment, Policy, and Worker Initialization ---
    task_configs, replay_buffers, collectors, evaluators = [], [], [], []

    # Use the first task's config to create the shared policy and learner
    _, [main_cfg, main_create_cfg] = tasks_for_this_rank[0]
    for _, [cfg, _] in tasks_for_this_rank:
        cfg.policy.task_num = len(tasks_for_this_rank)

    # ==================== START: Robust exp_name Fix ====================
    # Ensure main_cfg has a valid exp_name before calling compile_config.
    # If exp_name is missing, None, or too long, set a safe default.
    if not hasattr(main_cfg, 'exp_name') or main_cfg.exp_name is None or len(str(main_cfg.exp_name)) > 200:
        # Use a simplified experiment name for the main config
        safe_exp_name = f'data_unizero_mt_balance/dmc_multitask_seed{seed}'
        logging.warning(
            f"Rank {rank}: main_cfg.exp_name is missing, None, or too long. "
            f"Setting to safe default: {safe_exp_name}"
        )
        main_cfg.exp_name = safe_exp_name
    else:
        logging.info(f"Rank {rank}: Using exp_name from config: {main_cfg.exp_name}")
    # ==================== END: Robust exp_name Fix ====================

    assert main_create_cfg.policy.type in ['unizero_multitask', 'sampled_unizero_multitask'], \
        "This entry only supports 'unizero_multitask' or 'sampled_unizero_multitask' policies."

    GameBuffer = None
    if main_create_cfg.policy.type == 'unizero_multitask':
        from lzero.mcts import UniZeroGameBuffer as GameBuffer
    elif main_create_cfg.policy.type == 'sampled_unizero_multitask':
        from lzero.mcts import SampledUniZeroGameBuffer as GameBuffer

    main_cfg.policy.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    compiled_cfg = compile_config(main_cfg, seed=seed, auto=True, create_cfg=main_create_cfg, save_cfg=True)

    policy = create_policy(compiled_cfg.policy, model=model, enable_field=['learn', 'collect', 'eval'])

    # Log initial model architecture info BEFORE loading checkpoint
    if rank == 0:
        num_layers_config = compiled_cfg.policy.model.world_model_cfg.num_layers
        initial_params = sum(p.numel() for p in policy._learn_model.world_model.parameters())
        initial_trainable = sum(p.numel() for p in policy._learn_model.world_model.parameters() if p.requires_grad)
        logging.info(f"=" * 80)
        logging.info(f"Model Architecture Configuration:")
        logging.info(f"  - num_layers from config: {num_layers_config}")
        logging.info(f"  - Total parameters (before checkpoint load): {initial_params:,}")
        logging.info(f"  - Trainable parameters (before checkpoint load): {initial_trainable:,}")
        logging.info(f"=" * 80)

    if model_path:
        logging.info(f'Loading pre-trained model from: {model_path}')
        policy.learn_mode.load_state_dict(torch.load(model_path, map_location=compiled_cfg.policy.device))
        logging.info('Model loading complete.')
        if rank == 0:
            loaded_params = sum(p.numel() for p in policy._learn_model.world_model.parameters())
            loaded_trainable = sum(p.numel() for p in policy._learn_model.world_model.parameters() if p.requires_grad)
            logging.info(f"Model Parameters After Loading Checkpoint:")
            logging.info(f"  - Total parameters (after checkpoint load): {loaded_params:,}")
            logging.info(f"  - Trainable parameters (after checkpoint load): {loaded_trainable:,}")
            if initial_params != loaded_params:
                logging.warning(f"⚠️ WARNING: Parameter count mismatch!")
                logging.warning(f"  Config specifies {initial_params:,} params, but loaded model has {loaded_params:,} params")
                logging.warning(f"  This usually means the checkpoint was trained with different num_layers!")
                logging.warning(f"  The loaded checkpoint architecture will override your config settings.")


    tb_logger = SummaryWriter(os.path.join(f'./{compiled_cfg.exp_name}/log', f'rank_{rank}'))
    learner = BaseLearner(compiled_cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=compiled_cfg.exp_name)
    learner.call_hook('before_run')

    # Initialize components for each assigned task
    for local_task_id, (task_id, [cfg, create_cfg]) in enumerate(tasks_for_this_rank):
        task_seed = seed + task_id
        cfg.policy.device = 'cuda' if cfg.policy.cuda and torch.cuda.is_available() else 'cpu'

        # ==================== START: Robust exp_name Fix for Task Config ====================
        # Ensure each task config has a valid exp_name before calling compile_config
        if not hasattr(cfg, 'exp_name') or cfg.exp_name is None:
            # Extract env_id from config if available, otherwise use task_id
            env_id = getattr(cfg.env, 'env_id', f'task{task_id}')
            cfg.exp_name = f'data_unizero_mt_balance/task_{env_id}_seed{task_seed}'
            logging.warning(
                f"Rank {rank}: Task {task_id} config missing exp_name. "
                f"Setting to: {cfg.exp_name}"
            )
        # ==================== END: Robust exp_name Fix for Task Config ====================

        compiled_task_cfg = compile_config(cfg, seed=task_seed, auto=True, create_cfg=create_cfg, save_cfg=True)
        
        env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(compiled_task_cfg.env)
        collector_env = create_env_manager(compiled_task_cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_env_cfg])
        evaluator_env = create_env_manager(compiled_task_cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])
        collector_env.seed(task_seed)
        evaluator_env.seed(task_seed, dynamic_seed=False)
        set_pkg_seed(task_seed, use_cuda=compiled_task_cfg.policy.cuda)

        replay_buffers.append(GameBuffer(compiled_task_cfg.policy))
        collectors.append(Collector(
            collect_print_freq=100,
            env=collector_env,
            policy=policy.collect_mode,
            tb_logger=tb_logger,
            exp_name=compiled_task_cfg.exp_name,
            instance_name=f'collector_task{task_id}',
            policy_config=compiled_task_cfg.policy,
            task_id=task_id
        ))
        evaluators.append(Evaluator(
            eval_freq=compiled_task_cfg.policy.eval_freq,
            n_evaluator_episode=compiled_task_cfg.env.n_evaluator_episode,
            stop_value=compiled_task_cfg.env.stop_value,
            env=evaluator_env,
            policy=policy.eval_mode,
            tb_logger=tb_logger,
            exp_name=compiled_task_cfg.exp_name,
            instance_name=f'evaluator_task{task_id}',
            policy_config=compiled_task_cfg.policy,
            task_id=task_id
        ))
        task_configs.append(compiled_task_cfg)

    # --- Curriculum and Training Loop Initialization ---
    solved_task_pool = set()
    curriculum_controller = CurriculumController(compiled_cfg, policy)
    temperature_scheduler = TemperatureScheduler(initial_temp=10.0, final_temp=1.0, threshold_steps=int(1e4), mode='linear')
    
    train_epoch = 0
    buffer_reanalyze_count = 0

    logging.info(f"Rank {rank}: Initial trainable parameters in world model: {sum(p.requires_grad for p in policy._learn_model.world_model.parameters())}/{sum(1 for _ in policy._learn_model.world_model.parameters())}")

    # ============================================================================================
    # Main Training Loop
    # ============================================================================================
    while True:
        # --- 1. Dynamic Batch Size Allocation (Optional) ---
        if compiled_cfg.policy.allocated_batch_sizes:
            clip_scale = np.clip(1 + (3 * train_epoch / 1000), 1, 4)
            allocated_batch_sizes = allocate_batch_size(task_configs, replay_buffers, alpha=1.0, clip_scale=clip_scale)
            if rank == 0:
                logging.info(f"Dynamically allocated batch sizes: {allocated_batch_sizes}")
            # Assign the corresponding batch size to each task config
            for i, cfg in enumerate(task_configs):
                task_id = cfg.policy.task_id
                if isinstance(allocated_batch_sizes, dict):
                    cfg.policy.batch_size = allocated_batch_sizes.get(task_id, cfg.policy.batch_size)
                elif isinstance(allocated_batch_sizes, list):
                    # Use the index in the list or task_id as fallback
                    cfg.policy.batch_size = allocated_batch_sizes[i] if i < len(allocated_batch_sizes) else cfg.policy.batch_size
                else:
                    logging.warning(f"Unexpected type for allocated_batch_sizes: {type(allocated_batch_sizes)}")
            # Also update the policy config (use the full list for compatibility)
            policy._cfg.batch_size = allocated_batch_sizes

        # --- 2. Data Collection and Evaluation for each task on this rank ---
        local_task_returns = {}
        for i, (cfg, collector, evaluator, replay_buffer) in enumerate(zip(task_configs, collectors, evaluators, replay_buffers)):
            task_id = cfg.policy.task_id
            if task_id in solved_task_pool:
                continue

            # Evaluate policy if it's time
            if learner.train_iter > 10 and evaluator.should_eval(learner.train_iter):
                logging.info(f'Rank {rank} evaluating task_id: {task_id}...')
                evaluator._policy.reset(reset_init_data=True, task_id=task_id)
                stop_flag, reward_dict = safe_eval(evaluator, learner, collector, rank, world_size)

                if reward_dict is not None:
                    eval_mean_reward = reward_dict.get('eval_episode_return_mean', float('-inf'))
                    logging.info(f"Task {task_id} evaluation reward: {eval_mean_reward}")
                    local_task_returns[task_id] = eval_mean_reward
                    if eval_mean_reward >= cfg.policy.target_return:
                        logging.info(f"Task {task_id} has reached its target return of {cfg.policy.target_return}. Adding to solved pool.")
                        solved_task_pool.add(task_id)
                else:
                    logging.warning(f"Evaluation failed or timed out for task {task_id}. Assigning a low score.")
                    local_task_returns[task_id] = float('-inf')

            # Collect new data
            logging.info(f'Rank {rank} collecting data for task_id: {task_id}...')
            collect_kwargs = {'temperature': visit_count_temperature(cfg.policy.manual_temperature_decay, cfg.policy.fixed_temperature_value, cfg.policy.threshold_training_steps_for_final_temperature, learner.train_iter)}
            if cfg.policy.eps.eps_greedy_exploration_in_collect:
                epsilon_fn = get_epsilon_greedy_fn(cfg.policy.eps.start, cfg.policy.eps.end, cfg.policy.eps.decay, cfg.policy.eps.type)
                collect_kwargs['epsilon'] = epsilon_fn(collector.envstep)
            
            collector._policy.reset(reset_init_data=True, task_id=task_id)
            new_data = collector.collect(train_iter=learner.train_iter, policy_kwargs=collect_kwargs)
            replay_buffer.push_game_segments(new_data)
            replay_buffer.remove_oldest_data_to_fit()
            logging.info(f'Rank {rank}: Data collection finished for task {task_id}.')

        # --- 3. DDP Synchronization of Task Status and Weights ---
        dist.barrier()
        # Gather solved tasks from all ranks
        all_solved_pools = [None for _ in range(world_size)]
        dist.all_gather_object(all_solved_pools, solved_task_pool)
        global_solved_task_pool = set().union(*[pool for pool in all_solved_pools if pool is not None])
        solved_task_pool = global_solved_task_pool  # Sync local pool with global
        global_solved_count = len(solved_task_pool)

        # Gather evaluation returns and compute task weights
        task_weights = None
        if learner.train_iter > 10 and learner.train_iter % compiled_cfg.policy.eval_freq == 0:
            all_task_returns = [None for _ in range(world_size)]
            dist.all_gather_object(all_task_returns, local_task_returns)
            
            merged_task_returns = {k: v for d in all_task_returns if d for k, v in d.items()}
            for tid, ret in merged_task_returns.items():
                GLOBAL_EVAL_RETURNS[tid] = ret # Update global tracker

            unsolved_task_returns = {tid: ret for tid, ret in merged_task_returns.items() if tid not in solved_task_pool}

            if rank == 0:
                logging.info(f"Global unsolved task returns for weight calculation: {unsolved_task_returns}")
                if compiled_cfg.policy.task_complexity_weight and unsolved_task_returns:
                    temp = temperature_scheduler.get_temperature(learner.train_iter)
                    task_weights = compute_task_weights(unsolved_task_returns, option="rank", temperature=temp)
                    logging.info(f"Computed task weights: {task_weights}")
                
                # Log UniZero-MT normalized stats
                mean_norm, median_norm = compute_unizero_mt_normalized_stats(GLOBAL_EVAL_RETURNS)
                if mean_norm is not None:
                    tb_logger.add_scalar('UniZero-MT/NormalizedMean', mean_norm, learner.train_iter)
                    tb_logger.add_scalar('UniZero-MT/NormalizedMedian', median_norm, learner.train_iter)
                    logging.info(f"UniZero-MT Normalized Mean={mean_norm:.4f}, Median={median_norm:.4f}")

            # Broadcast weights from rank 0 to all other ranks
            broadcast_objects = [task_weights]
            dist.broadcast_object_list(broadcast_objects, src=0)
            task_weights = broadcast_objects[0]

        # --- 4. Curriculum Stage Update ---
        unsolved_count = total_tasks - global_solved_count
        switched = curriculum_controller.step(global_solved_count, unsolved_count, learner.train_iter)
        
        if rank == 0:
            tb_logger.add_scalar('Curriculum/Stage', curriculum_controller.stage, learner.train_iter)
            tb_logger.add_scalar('Curriculum/GlobalSolvedTasks', global_solved_count, learner.train_iter)

            # TODO 遍历 transformer 中所有子模块，根据其名称查找 CurriculumLoRALinear 模块
            # transformer = policy._learn_model.world_model.transformer
            # for module_name, module in transformer.named_modules():
            #     if isinstance(module, CurriculumLoRALinear) and module.adapters is not None:
            #         for adapter_idx, scale_param in enumerate(module.adapter_scales):
            #             tb_logger.add_scalar(
            #                 f'Curriculum/adapter_scales/{module_name}/adapter_{adapter_idx}',
            #                 scale_param().item(),
            #                 global_step=learner.train_iter
            #             )
            
            # 新增的 alpha 缩放因子日志记录
            try:
                transformer = policy._learn_model.world_model.transformer
                for module_name, module in transformer.named_modules():
                    if isinstance(module, CurriculumLoRALinear):
                        # 检查模块是否有 base_weight_scale 属性
                        if hasattr(module, 'base_weight_scale') and module.base_weight_scale is not None:
                            # 1. 记录基座权重的缩放因子 (alpha_0)
                            tb_logger.add_scalar(
                                f'Curriculum/alpha_scales/{module_name}/alpha_0_base_weight',
                                module.base_weight_scale().item(),
                                global_step=learner.train_iter
                            )

                        # 检查模块是否有 adapter_scales 属性
                        if hasattr(module, 'adapter_scales') and module.adapter_scales is not None:
                            # 2. 遍历并记录所有适配器的缩放因子 (alpha_1, alpha_2, ...)
                            for adapter_idx, scale_param in enumerate(module.adapter_scales):
                                # adapter_idx 是从 0 开始的，对应 alpha_{idx+1}
                                tb_logger.add_scalar(
                                    f'Curriculum/alpha_scales/{module_name}/alpha_{adapter_idx + 1}',
                                    scale_param().item(),
                                    global_step=learner.train_iter
                                )
            except Exception as e:
                logging.warning(f"Failed to log alpha scales: {e}")
                        

        # Ensure all processes are aware of a potential stage switch
        dist.barrier()

        # --- 5. Training Step ---
        unsolved_buffers = [rb for cfg, rb in zip(task_configs, replay_buffers) if cfg.policy.task_id not in solved_task_pool]
        unsolved_cfgs = [cfg for cfg in task_configs if cfg.policy.task_id not in solved_task_pool]

        if not unsolved_buffers:
            logging.info(f"Rank {rank}: All assigned tasks are solved. Performing dummy training to maintain DDP sync.")
            # When all local tasks are solved, we must still participate in DDP.
            # A dummy forward/backward pass with zeroed gradients can ensure this.
            # The current implementation uses a minimal batch from solved tasks with `ignore_grad=True`.
            for _ in range(compiled_cfg.policy.update_per_collect):
                train_data_list = []
                for cfg, replay_buffer in zip(task_configs, replay_buffers): # Use original buffers
                    batch_size = 2 # Minimal batch size for sync
                    if replay_buffer.get_num_of_transitions() >= batch_size:
                        train_data = replay_buffer.sample(batch_size, policy)
                        train_data.append(cfg.policy.task_id)
                        train_data_list.append(train_data)
                
                if train_data_list:
                    learner.train(train_data_list, collector.envstep, policy_kwargs={'task_weights': None, "ignore_grad": True})

        else:
            for _ in range(compiled_cfg.policy.update_per_collect):
                train_data_list = []
                total_envstep = sum(c.envstep for c in collectors)
                for cfg, replay_buffer in zip(unsolved_cfgs, unsolved_buffers):
                    # Handle batch_size whether it's an int, list, or dict
                    if isinstance(cfg.policy.batch_size, (list, tuple)):
                        batch_size = cfg.policy.batch_size[cfg.policy.task_id]
                    elif isinstance(cfg.policy.batch_size, dict):
                        batch_size = cfg.policy.batch_size[cfg.policy.task_id]
                    else:
                        # batch_size is already an integer
                        batch_size = cfg.policy.batch_size

                    if replay_buffer.get_num_of_transitions() >= batch_size:
                        train_data = replay_buffer.sample(batch_size, policy)
                        train_data.append(cfg.policy.task_id)
                        train_data_list.append(train_data)
                    else:
                        logging.warning(f"Skipping training for task {cfg.policy.task_id}: not enough data in buffer.")
                
                if train_data_list:
                    learn_kwargs = {'task_weights': task_weights, "ignore_grad": False}
                    learner.train(train_data_list, total_envstep, policy_kwargs=learn_kwargs)

        train_epoch += 1
        policy.recompute_pos_emb_diff_and_clear_cache()

        # --- 6. Synchronization and Termination Check ---
        dist.barrier() # Ensure all ranks complete the training step
        
        # Check for termination conditions
        max_iter_reached = torch.tensor([learner.train_iter >= max_train_iter], dtype=torch.bool, device=compiled_cfg.policy.device)
        dist.all_reduce(max_iter_reached, op=dist.ReduceOp.SUM)
        
        # For env_step, gather from all collectors on all ranks
        local_env_steps = torch.tensor([c.envstep for c in collectors], dtype=torch.long, device=compiled_cfg.policy.device)
        all_env_steps = [torch.zeros_like(local_env_steps) for _ in range(world_size)]
        # Note: all_gather requires all tensors to be the same size. This assumes each rank has the same number of collectors.
        # If not, a more complex gathering method (e.g., all_gather_object) is needed.
        try:
            dist.all_gather(all_env_steps, local_env_steps)
            max_step_reached = (torch.cat(all_env_steps).min() >= max_env_step) if all_env_steps else False
        except RuntimeError: # If tensor sizes mismatch
            max_step_reached = False # Fallback, consider logging an error
            logging.warning("Could not gather env_steps due to tensor size mismatch across ranks. Termination check may be inaccurate.")

        if max_iter_reached.item() or max_step_reached:
            logging.info(f"Rank {rank}: Termination condition met. Stopping training.")
            break

    # --- Finalization ---
    learner.call_hook('after_run')
    return policy
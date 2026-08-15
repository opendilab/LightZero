import concurrent.futures
import os
from collections import defaultdict
from functools import partial
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from ding.config import compile_config
from ding.envs import create_env_manager, get_vec_env_setting
from ding.policy import Policy, create_policy
from ding.rl_utils import get_epsilon_greedy_fn
from ding.utils import EasyTimer, get_rank, get_world_size, set_pkg_seed
from ding.worker import BaseLearner
from ditk import logging
from lzero.entry.utils import (
    EVALUATION_TIMEOUT,
    TemperatureScheduler,
    allocate_batch_size,
    compute_task_weights,
    compute_unizero_mt_normalized_stats,
    log_buffer_memory_usage,
    safe_eval,
    symlog,
    inv_symlog,
)
# NOTE: The following imports are for type hinting purposes.
# The actual GameBuffer is selected dynamically based on the policy type.
from lzero.mcts import UniZeroGameBuffer
from lzero.policy import visit_count_temperature
from lzero.worker import MuZeroEvaluator as Evaluator
from lzero.worker import MuZeroSegmentCollector as Collector
from tensorboardX import SummaryWriter

# ====================================================================================================================
# Note: Benchmark score definitions are initialized dynamically within the `train_unizero_multitask_segment_ddp`
# function based on the `benchmark_name` argument to ensure correct score assignment.
# ====================================================================================================================

# Stores the latest evaluation returns: {task_id: eval_episode_return_mean}
GLOBAL_EVAL_RETURNS: Dict[int, float] = defaultdict(lambda: None)

timer = EasyTimer()


def train_unizero_multitask_segment_ddp(
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
        The training entry point for UniZero, designed to enhance the planning capabilities of reinforcement learning agents
        by addressing the limitations of MuZero-like algorithms in environments requiring long-term dependency capture.
        For more details, please refer to https://arxiv.org/abs/2406.10667.

    Arguments:
        - input_cfg_list (:obj:`List[Tuple[int, Tuple[dict, dict]]]`): A list of configurations for different tasks.
        - seed (:obj:`int`): The random seed.
        - model (:obj:`Optional[torch.nn.Module]`): An instance of torch.nn.Module.
        - model_path (:obj:`Optional[str]`): The path to a pre-trained model checkpoint file.
        - max_train_iter (:obj:`Optional[int]`): The maximum number of policy update iterations during training.
        - max_env_step (:obj:`Optional[int]`): The maximum number of environment interaction steps to collect.
        - benchmark_name (:obj:`str`): The name of the benchmark, e.g., "atari" or "dmc".

    Returns:
        - policy (:obj:`Policy`): The converged policy.
    """
    # ------------------------------------------------------------------------------------
    # ====== UniZero-MT Benchmark Scores (corresponding to 26 Atari100k task IDs) ======
    # Original RANDOM_SCORES and HUMAN_SCORES
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
    elif benchmark_name == "dmc":
        RANDOM_SCORES = np.zeros(26)
        HUMAN_SCORES = np.ones(26) * 1000
    else:
        raise ValueError(f"Unsupported BENCHMARK_NAME: {benchmark_name}")

    # New order to original index mapping
    # New order: [Pong, MsPacman, Seaquest, Boxing, Alien, ChopperCommand, Hero, RoadRunner,
    #            Amidar, Assault, Asterix, BankHeist, BattleZone, CrazyClimber, DemonAttack,
    #            Freeway, Frostbite, Gopher, Jamesbond, Kangaroo, Krull, KungFuMaster,
    #            PrivateEye, UpNDown, Qbert, Breakout]
    # Mapping to indices in the original array (0-based)
    new_order = [
        20, 19, 24, 6, 0, 8, 14, 23, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 15, 16, 17, 18, 21, 25, 22, 7
    ]
    global new_RANDOM_SCORES, new_HUMAN_SCORES
    # Generate new arrays based on new_order
    new_RANDOM_SCORES = RANDOM_SCORES[new_order]
    new_HUMAN_SCORES = HUMAN_SCORES[new_order]
    # Log the reordered results
    logging.info("Reordered RANDOM_SCORES:")
    logging.info(new_RANDOM_SCORES)
    logging.info("\nReordered HUMAN_SCORES:")
    logging.info(new_HUMAN_SCORES)
    # ------------------------------------------------------------------------------------

    # Initialize the temperature scheduler for task weighting.
    initial_temperature = 10.0
    final_temperature = 1.0
    threshold_steps = int(1e4)  # Temperature drops to 1.0 after 10k training steps.
    temperature_scheduler = TemperatureScheduler(
        initial_temp=initial_temperature,
        final_temp=final_temperature,
        threshold_steps=threshold_steps,
        mode='linear'  # or 'exponential'
    )

    # Get the current process rank and total world size.
    rank = get_rank()
    world_size = get_world_size()

    # Task partitioning among ranks.
    total_tasks = len(input_cfg_list)
    tasks_per_rank = total_tasks // world_size
    remainder = total_tasks % world_size

    # 1. Precisely calculate the number of tasks assigned to the current rank.
    if rank < remainder:
        start_idx = rank * (tasks_per_rank + 1)
        end_idx = start_idx + tasks_per_rank + 1
        num_tasks_for_this_rank = tasks_per_rank + 1
    else:
        start_idx = rank * tasks_per_rank + remainder
        end_idx = start_idx + tasks_per_rank
        num_tasks_for_this_rank = tasks_per_rank

    tasks_for_this_rank = input_cfg_list[start_idx:end_idx]

    # Ensure at least one task is assigned.
    if len(tasks_for_this_rank) == 0:
        logging.warning(f"Rank {rank}: No tasks assigned, continuing execution.")
        # Initialize empty lists to avoid errors later.
        cfgs, game_buffers, collector_envs, evaluator_envs, collectors, evaluators = [], [], [], [], [], []
    else:
        logging.info(f"Rank {rank}/{world_size} processing tasks {start_idx} to {end_idx - 1}")

    cfgs = []
    game_buffers = []
    collector_envs = []
    evaluator_envs = []
    collectors = []
    evaluators = []

    if tasks_for_this_rank:
        # Use the config of the first task to create a shared policy.
        task_id, [cfg, create_cfg] = tasks_for_this_rank[0]

        # ==================== START: Critical Fix ====================
        # 2. Set the correct task count to *all* related configurations.
        #    Configuration must be correct before creating the Policy instance.
        for config_tuple in tasks_for_this_rank:
            # config_tuple is (task_id, [cfg_obj, create_cfg_obj])
            config_tuple[1][0].policy.task_num = num_tasks_for_this_rank

        # 3. Ensure the cfg object used to create the Policy also has the correct task_num.
        cfg.policy.task_num = num_tasks_for_this_rank
        # ==================== END: Critical Fix ====================

        # Ensure the specified policy type is supported.
        assert create_cfg.policy.type in ['unizero_multitask', 'sampled_unizero_multitask'], \
            "train_unizero entry currently only supports 'unizero_multitask' or 'sampled_unizero_multitask'"

        if create_cfg.policy.type == 'unizero_multitask':
            from lzero.mcts import UniZeroGameBuffer as GameBuffer
        if create_cfg.policy.type == 'sampled_unizero_multitask':
            from lzero.mcts import SampledUniZeroGameBuffer as GameBuffer

        # Set device based on CUDA availability.
        cfg.policy.device = cfg.policy.model.world_model_cfg.device if torch.cuda.is_available() else 'cpu'
        logging.info(f'Configured device: {cfg.policy.device}')

        # Compile the configuration.
        cfg = compile_config(cfg, seed=seed, env=None, auto=True, create_cfg=create_cfg, save_cfg=True)
        # Create the shared policy.
        policy = create_policy(cfg.policy, model=model, enable_field=['learn', 'collect', 'eval'])

        # Load a pre-trained model if a path is provided.
        if model_path is not None:
            logging.info(f'Starting to load model: {model_path}')
            policy.learn_mode.load_state_dict(torch.load(model_path, map_location=cfg.policy.device))
            logging.info(f'Finished loading model: {model_path}')

        # Create a TensorBoard logger.
        log_dir = os.path.join('./{}/log'.format(cfg.exp_name), f'serial_rank_{rank}')
        tb_logger = SummaryWriter(log_dir)

        # Create the shared learner.
        learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)

        policy_config = cfg.policy

        # Process each task assigned to the current rank.
        for local_task_id, (task_id, [cfg, create_cfg]) in enumerate(tasks_for_this_rank):
            # Set a unique random seed for each task.
            cfg.policy.device = 'cuda' if cfg.policy.device == 'cuda' and torch.cuda.is_available() else 'cpu'
            cfg = compile_config(cfg, seed=seed + task_id, env=None, auto=True, create_cfg=create_cfg, save_cfg=True)
            policy_config = cfg.policy
            policy.collect_mode.get_attribute('cfg').n_episode = policy_config.n_episode
            policy.eval_mode.get_attribute('cfg').n_episode = policy_config.n_episode

            # Create environments.
            env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)
            collector_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_env_cfg])
            evaluator_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])
            collector_env.seed(cfg.seed + task_id)
            evaluator_env.seed(cfg.seed + task_id, dynamic_seed=False)
            set_pkg_seed(cfg.seed + task_id, use_cuda=cfg.policy.cuda)

            # Create task-specific game buffers, collectors, and evaluators.
            replay_buffer = GameBuffer(policy_config)
            collector = Collector(
                env=collector_env,
                policy=policy.collect_mode,
                tb_logger=tb_logger,
                exp_name=cfg.exp_name,
                policy_config=policy_config,
                task_id=task_id
            )
            evaluator = Evaluator(
                eval_freq=cfg.policy.eval_freq,
                n_evaluator_episode=cfg.env.n_evaluator_episode,
                stop_value=cfg.env.stop_value,
                env=evaluator_env,
                policy=policy.eval_mode,
                tb_logger=tb_logger,
                exp_name=cfg.exp_name,
                policy_config=policy_config,
                task_id=task_id
            )

            cfgs.append(cfg)
            # Handle batch_size robustly - it might be a list or already an integer
            if isinstance(cfg.policy.batch_size, (list, tuple)):
                replay_buffer.batch_size = cfg.policy.batch_size[task_id]
            elif isinstance(cfg.policy.batch_size, dict):
                replay_buffer.batch_size = cfg.policy.batch_size[task_id]
            else:
                replay_buffer.batch_size = cfg.policy.batch_size

            game_buffers.append(replay_buffer)
            collector_envs.append(collector_env)
            evaluator_envs.append(evaluator_env)
            collectors.append(collector)
            evaluators.append(evaluator)

    # Call the learner's before_run hook.
    learner.call_hook('before_run')
    value_priority_tasks = {}

    buffer_reanalyze_count = 0
    train_epoch = 0
    reanalyze_batch_size = cfg.policy.reanalyze_batch_size
    update_per_collect = cfg.policy.update_per_collect

    task_exploitation_weight = None

    # Dictionary to store task rewards.
    task_returns = {}  # {task_id: reward}

    while True:
        # Dynamically adjust batch sizes.
        if cfg.policy.allocated_batch_sizes:
            clip_scale = np.clip(1 + (3 * train_epoch / 1000), 1, 4)
            allocated_batch_sizes = allocate_batch_size(cfgs, game_buffers, alpha=1.0, clip_scale=clip_scale)
            if rank == 0:
                logging.info("Allocated batch_sizes: ", allocated_batch_sizes)
            # Assign the corresponding batch size to each task config
            for idx, (cfg, collector, evaluator, replay_buffer) in enumerate(
                    zip(cfgs, collectors, evaluators, game_buffers)):
                task_id = cfg.policy.task_id
                if isinstance(allocated_batch_sizes, dict):
                    cfg.policy.batch_size = allocated_batch_sizes.get(task_id, cfg.policy.batch_size)
                elif isinstance(allocated_batch_sizes, list):
                    # Use the index in the list or task_id as fallback
                    cfg.policy.batch_size = allocated_batch_sizes[idx] if idx < len(allocated_batch_sizes) else cfg.policy.batch_size
                else:
                    logging.warning(f"Unexpected type for allocated_batch_sizes: {type(allocated_batch_sizes)}")
            # Also update the policy config (use the full list for compatibility)
            policy._cfg.batch_size = allocated_batch_sizes

        # For each task on the current rank, perform data collection and evaluation.
        for idx, (cfg, collector, evaluator, replay_buffer) in enumerate(
                zip(cfgs, collectors, evaluators, game_buffers)):

            # Log buffer memory usage.
            log_buffer_memory_usage(learner.train_iter, replay_buffer, tb_logger, cfg.policy.task_id)

            collect_kwargs = {
                'temperature': visit_count_temperature(
                    policy_config.manual_temperature_decay,
                    policy_config.fixed_temperature_value,
                    policy_config.threshold_training_steps_for_final_temperature,
                    trained_steps=learner.train_iter
                ),
                'epsilon': 0.0  # Default epsilon value.
            }

            if policy_config.eps.eps_greedy_exploration_in_collect:
                epsilon_greedy_fn = get_epsilon_greedy_fn(
                    start=policy_config.eps.start,
                    end=policy_config.eps.end,
                    decay=policy_config.eps.decay,
                    type_=policy_config.eps.type
                )
                collect_kwargs['epsilon'] = epsilon_greedy_fn(collector.envstep)

            # Check if it's time for evaluation.
            if learner.train_iter > 10 and learner.train_iter % cfg.policy.eval_freq == 0:
            # if learner.train_iter == 0 or learner.train_iter % cfg.policy.eval_freq == 0: # TODO: Only for debug

                logging.info('=' * 20)
                logging.info(f'Rank {rank} evaluating task_id: {cfg.policy.task_id}...')

                # TODO: Ensure policy reset logic is optimal for multi-task settings.
                evaluator._policy.reset(reset_init_data=True, task_id=cfg.policy.task_id)

                # Perform safe evaluation.
                stop, reward = safe_eval(evaluator, learner, collector, rank, world_size)
                # Check if evaluation was successful.
                if stop is None or reward is None:
                    logging.warning(f"Rank {rank} encountered issues during evaluation, continuing training...")
                    task_returns[cfg.policy.task_id] = float('inf')  # Set task difficulty to max if evaluation fails.
                else:
                    # Extract 'eval_episode_return_mean' from the reward dictionary.
                    try:
                        eval_mean_reward = reward.get('eval_episode_return_mean', float('inf'))
                        logging.info(f"Task {cfg.policy.task_id} evaluation reward: {eval_mean_reward}")
                        task_returns[cfg.policy.task_id] = eval_mean_reward
                    except Exception as e:
                        logging.error(f"Error extracting evaluation reward: {e}")
                        task_returns[cfg.policy.task_id] = float('inf')  # Set reward to max on error.

            logging.info('=' * 20)
            logging.info(f'Starting collection for Rank {rank} task_id: {cfg.policy.task_id}...')
            logging.info(f'Rank {rank}: cfg.policy.task_id={cfg.policy.task_id} ')
            logging.info(f'Rank {rank}: Starting data collection for task {cfg.policy.task_id} at train_iter {learner.train_iter}')

            # Reset initial data before each collection, crucial for multi-task settings.
            collector._policy.reset(reset_init_data=True, task_id=cfg.policy.task_id)
            # Collect data.
            new_data = collector.collect(train_iter=learner.train_iter, policy_kwargs=collect_kwargs)
            logging.info(f'Rank {rank}: Finished data collection for task {cfg.policy.task_id}, collected {len(new_data[0]) if new_data else 0} segments')

            # Update the replay buffer.
            replay_buffer.push_game_segments(new_data)
            replay_buffer.remove_oldest_data_to_fit()

            # Periodically reanalyze the buffer.
            if cfg.policy.buffer_reanalyze_freq >= 1:
                reanalyze_interval = update_per_collect // cfg.policy.buffer_reanalyze_freq
            else:
                if train_epoch > 0 and train_epoch % int(1 / cfg.policy.buffer_reanalyze_freq) == 0 and \
                        replay_buffer.get_num_of_transitions() // cfg.policy.num_unroll_steps > int(
                    reanalyze_batch_size / cfg.policy.reanalyze_partition):
                    with timer:
                        replay_buffer.reanalyze_buffer(reanalyze_batch_size, policy)
                    buffer_reanalyze_count += 1
                    logging.info(f'Buffer reanalysis count: {buffer_reanalyze_count}')
                    logging.info(f'Buffer reanalysis time: {timer.value}')

            # Log after data collection.
            logging.info(f'Rank {rank}: Completed data collection for task {cfg.policy.task_id}')

        # ========== Synchronize all ranks after data collection ==========
        # Wait for all ranks to complete their data collection before proceeding.
        # This prevents fast-collecting ranks from reaching barriers/all_gather calls
        # while slow-collecting ranks are still in the collection loop.
        try:
            logging.info(f'Rank {rank}: Waiting at post-collection barrier...')
            dist.barrier()
            logging.info(f'Rank {rank}: All ranks completed data collection, proceeding...')
        except Exception as e:
            logging.error(f'Rank {rank}: Post-collection barrier failed, error: {e}')
            raise e
        # ===============================================================================

        # Check if there is enough data for training.
        local_not_enough_data = any(
            replay_buffer.get_num_of_transitions() < cfgs[0].policy.total_batch_size / world_size
            for replay_buffer in game_buffers
        )
        logging.info(f"Rank {rank} local_not_enough_data:{local_not_enough_data}")
        flag_tensor = torch.tensor(1.0 if local_not_enough_data else 0.0, device=cfg.policy.device)
        dist.all_reduce(flag_tensor, op=dist.ReduceOp.MAX)
        not_enough_data = (flag_tensor.item() > 0.5)
        if rank == 0:
            logging.info(f"Global not_enough_data status: {not_enough_data}")
        
        # Get the current temperature for task weighting.
        current_temperature_task_weight = temperature_scheduler.get_temperature(learner.train_iter)

        if learner.train_iter > 10 and learner.train_iter % cfg.policy.eval_freq == 0:
            # Calculate task weights.
            try:
                # Gather task rewards.
                logging.info(f'Rank {rank}: Entering evaluation synchronization barrier at train_iter {learner.train_iter}')
                dist.barrier()
                logging.info(f'Rank {rank}: Passed evaluation barrier, gathering task returns')
                all_task_returns = [None for _ in range(world_size)]
                dist.all_gather_object(all_task_returns, task_returns)
                # Merge task rewards.
                merged_task_returns = {}
                for returns in all_task_returns:
                    if returns:
                        merged_task_returns.update(returns)

                logging.warning(f"Rank {rank}: merged_task_returns: {merged_task_returns}")

                # Calculate global task weights.
                task_weights = compute_task_weights(merged_task_returns, temperature=current_temperature_task_weight)

                # ---------- Maintain UniZero-MT global evaluation results ----------
                for tid, ret in merged_task_returns.items():
                    GLOBAL_EVAL_RETURNS[tid] = ret  # Update even for solved tasks.

                # Calculate Human-Normalized Mean / Median.
                # Convert arrays to dictionaries with task_id as keys
                human_scores_dict = {i: new_HUMAN_SCORES[i] for i in range(len(new_HUMAN_SCORES))}
                random_scores_dict = {i: new_RANDOM_SCORES[i] for i in range(len(new_RANDOM_SCORES))}
                uni_mean, uni_median = compute_unizero_mt_normalized_stats(
                    GLOBAL_EVAL_RETURNS, human_scores_dict, random_scores_dict
                )

                if uni_mean is not None:  # At least one task has been evaluated.
                    if rank == 0:  # Only write to TensorBoard on rank 0 to avoid duplication.
                        tb_logger.add_scalar('UniZero-MT/NormalizedMean', uni_mean, global_step=learner.train_iter)
                        tb_logger.add_scalar('UniZero-MT/NormalizedMedian', uni_median, global_step=learner.train_iter)
                    logging.info(f"Rank {rank}: UniZero-MT Norm Mean={uni_mean:.4f}, Median={uni_median:.4f}")
                else:
                    logging.info(f"Rank {rank}: No data available to compute UniZero-MT normalized metrics")

                # Synchronize task weights.
                dist.broadcast_object_list([task_weights], src=0)
            except Exception as e:
                logging.error(f'Rank {rank}: Failed to synchronize task weights, error: {e}')
                break

        # ---------------- Sampling done, preparing for backward pass ----------------
        # dist.barrier()  # ★★★ Critical synchronization point ★★★

        # Learn policy.
        if not not_enough_data:
            for i in range(update_per_collect):
                train_data_multi_task = []
                envstep_multi_task = 0
                for idx, (cfg, collector, replay_buffer) in enumerate(zip(cfgs, collectors, game_buffers)):
                    envstep_multi_task += collector.envstep
                    # Handle batch_size robustly - it might be a list or already an integer
                    if isinstance(cfg.policy.batch_size, (list, tuple)):
                        batch_size = cfg.policy.batch_size[cfg.policy.task_id]
                    elif isinstance(cfg.policy.batch_size, dict):
                        batch_size = cfg.policy.batch_size[cfg.policy.task_id]
                    else:
                        batch_size = cfg.policy.batch_size

                    if replay_buffer.get_num_of_transitions() > batch_size:
                        if cfg.policy.buffer_reanalyze_freq >= 1:
                            if i % reanalyze_interval == 0 and \
                                    replay_buffer.get_num_of_transitions() // cfg.policy.num_unroll_steps > int(
                                reanalyze_batch_size / cfg.policy.reanalyze_partition):
                                with timer:
                                    replay_buffer.reanalyze_buffer(reanalyze_batch_size, policy)
                                buffer_reanalyze_count += 1
                                logging.info(f'Buffer reanalysis count: {buffer_reanalyze_count}')
                                logging.info(f'Buffer reanalysis time: {timer.value}')

                        train_data = replay_buffer.sample(batch_size, policy)
                        train_data.append(cfg.policy.task_id)  # Append task_id to differentiate tasks.
                        train_data_multi_task.append(train_data)
                    else:
                        logging.warning(
                            f'Insufficient data in replay buffer to sample mini-batch: '
                            f'batch_size: {batch_size}, replay_buffer: {replay_buffer}'
                        )
                        break

                if train_data_multi_task:
                    learn_kwargs = {'task_weights': None,"train_iter":learner.train_iter}
                    
                    # DDP automatically synchronizes gradients and parameters during training.
                    log_vars = learner.train(train_data_multi_task, envstep_multi_task, policy_kwargs=learn_kwargs)

                    # Check if task_exploitation_weight needs to be calculated.
                    if i == 0:
                        # Calculate task weights.
                        try:
                            dist.barrier()  # Wait for all processes to synchronize.
                            if cfg.policy.use_task_exploitation_weight:  # Use obs loss now, new polish.
                                # Gather obs_loss from all tasks.
                                all_obs_loss = [None for _ in range(world_size)]
                                # Build obs_loss data for the current process's tasks.
                                merged_obs_loss_task = {}
                                for cfg, replay_buffer in zip(cfgs, game_buffers):
                                    task_id = cfg.policy.task_id
                                    if f'noreduce_obs_loss_task{task_id}' in log_vars[0]:
                                        merged_obs_loss_task[task_id] = log_vars[0][
                                            f'noreduce_obs_loss_task{task_id}']
                                # Gather obs_loss data from all processes.
                                dist.all_gather_object(all_obs_loss, merged_obs_loss_task)
                                # Merge obs_loss data from all processes.
                                global_obs_loss_task = {}
                                for obs_loss_task in all_obs_loss:
                                    if obs_loss_task:
                                        global_obs_loss_task.update(obs_loss_task)
                                # Calculate global task weights.
                                if global_obs_loss_task:
                                    task_exploitation_weight = compute_task_weights(
                                        global_obs_loss_task,
                                        option="rank",
                                        # TODO: Decide whether to use the temperature scheduler here.
                                        temperature=1,
                                    )
                                    # Broadcast task weights to all processes.
                                    dist.broadcast_object_list([task_exploitation_weight], src=0)
                                    logging.info(
                                        f"rank{rank}, task_exploitation_weight (sorted by task_id): {task_exploitation_weight}")
                                else:
                                    logging.warning(f"Rank {rank}: Unable to compute global obs_loss task weights, obs_loss data is empty.")
                                    task_exploitation_weight = None
                            else:
                                task_exploitation_weight = None
                            # Update training parameters to include the calculated task weights.
                            learn_kwargs['task_weight'] = task_exploitation_weight
                        except Exception as e:
                            logging.error(f'Rank {rank}: Failed to synchronize task weights, error: {e}')
                            raise e  # Re-raise the exception for external capture and analysis.

                    if cfg.policy.use_priority:
                        for idx, (cfg, replay_buffer) in enumerate(zip(cfgs, game_buffers)):
                            # Update task-specific replay buffer priorities.
                            task_id = cfg.policy.task_id
                            replay_buffer.update_priority(
                                train_data_multi_task[idx],
                                log_vars[0][f'noreduce_value_priority_task{task_id}']
                            )

        train_epoch += 1
        policy.recompute_pos_emb_diff_and_clear_cache()

        # Synchronize all ranks to ensure they have completed training.
        try:
            dist.barrier()
            logging.info(f'Rank {rank}: Passed synchronization barrier after training')
        except Exception as e:
            logging.error(f'Rank {rank}: Synchronization barrier failed, error: {e}')
            break

        # Check for termination conditions.
        try:
            local_envsteps = [collector.envstep for collector in collectors]
            total_envsteps = [None for _ in range(world_size)]
            dist.all_gather_object(total_envsteps, local_envsteps)

            all_envsteps = torch.cat([torch.tensor(envsteps, device=cfg.policy.device) for envsteps in total_envsteps])
            max_envstep_reached = torch.all(all_envsteps >= max_env_step)

            # Gather train_iter from all processes.
            global_train_iter = torch.tensor([learner.train_iter], device=cfg.policy.device)
            all_train_iters = [torch.zeros_like(global_train_iter) for _ in range(world_size)]
            dist.all_gather(all_train_iters, global_train_iter)

            max_train_iter_reached = torch.any(torch.stack(all_train_iters) >= max_train_iter)

            if max_envstep_reached.item() or max_train_iter_reached.item():
                logging.info(f'Rank {rank}: Termination condition reached')
                dist.barrier()  # Ensure all processes synchronize before exiting.
                break
        except Exception as e:
            logging.error(f'Rank {rank}: Termination check failed, error: {e}')
            break

    # Call the learner's after_run hook.
    learner.call_hook('after_run')
    return policy

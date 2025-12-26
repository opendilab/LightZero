import logging
import os
import concurrent.futures
from functools import partial
from typing import Tuple, Optional, List, Dict, Any, Type

import torch
import torch.distributed as dist
import numpy as np
from ding.config import compile_config
from ding.envs import create_env_manager, get_vec_env_setting
from ding.policy import create_policy, Policy
from ding.rl_utils import get_epsilon_greedy_fn
from ding.utils import set_pkg_seed, get_rank, get_world_size, EasyTimer
from ding.worker import BaseLearner
from tensorboardX import SummaryWriter

from lzero.entry.utils import log_buffer_memory_usage
from lzero.policy import visit_count_temperature
from lzero.mcts import UniZeroGameBuffer as GameBuffer
from lzero.worker import MuZeroEvaluator as Evaluator
from lzero.worker import MuZeroSegmentCollector as Collector

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)


def safe_eval(
        evaluator: Evaluator,
        learner: BaseLearner,
        collector: Collector,
        rank: int,
        world_size: int,
        timeout: int = 12000
) -> Tuple[Optional[bool], Optional[float]]:
    """
    Overview:
        Safely evaluates the policy using the evaluator with a specified timeout. This wrapper prevents
        the entire training process from crashing due to evaluation-related issues like deadlocks.
    Arguments:
        - evaluator (:obj:`Evaluator`): The evaluator instance to run.
        - learner (:obj:`BaseLearner`): The learner instance, used to access checkpoint saving and training iteration.
        - collector (:obj:`Collector`): The collector instance, used to access the environment step count.
        - rank (:obj:`int`): The rank of the current process in distributed training.
        - world_size (:obj:`int`): The total number of processes.
        - timeout (:obj:`int`): The maximum time in seconds to wait for the evaluation to complete.
    Returns:
        - (:obj:`Tuple[Optional[bool], Optional[float]]`): A tuple containing the stop flag and the reward.
          Returns (None, None) if evaluation times out or an exception occurs.
    """
    try:
        logging.info(f"Rank {rank}/{world_size}: Starting evaluation.")
        # Ensure the stop_event is clear before starting a new evaluation.
        evaluator.stop_event.clear()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(
                evaluator.eval,
                learner.save_checkpoint,
                learner.train_iter,
                collector.envstep
            )
            try:
                stop, reward = future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                # If evaluation exceeds the timeout, set the evaluator's stop event to terminate it gracefully.
                evaluator.stop_event.set()
                logging.warning(f"Rank {rank}/{world_size}: Evaluation timed out after {timeout} seconds.")
                return None, None

        logging.info(f"Rank {rank}/{world_size}: Evaluation finished successfully.")
        return stop, reward
    except Exception as e:
        logging.error(f"Rank {rank}/{world_size}: An error occurred during evaluation: {e}", exc_info=True)
        return None, None


def allocate_batch_size(
        cfgs: List[Any],
        game_buffers: List[GameBuffer],
        alpha: float = 1.0,
        clip_scale: int = 1
) -> List[int]:
    """
    Overview:
        Allocates batch sizes inversely proportional to the number of collected episodes for each task.
        This dynamic adjustment helps balance training focus across multiple tasks, prioritizing those
        with less data. The batch sizes are clipped to a dynamic range to maintain stability.
    Arguments:
        - cfgs (:obj:`List[Any]`): List of configuration objects for each task.
        - game_buffers (:obj:`List[GameBuffer]`): List of replay buffer instances for each task.
        - alpha (:obj:`float`): A hyperparameter controlling the degree of inverse proportionality. Defaults to 1.0.
        - clip_scale (:obj:`int`): A scaling factor to define the clipping range for the batch size. Defaults to 1.
    Returns:
        - (:obj:`List[int]`): A list of allocated batch sizes for each task.
    """
    # Extract the number of collected episodes from each task's buffer.
    buffer_num_of_collected_episodes = [buffer.num_of_collected_episodes for buffer in game_buffers]

    world_size = get_world_size()
    rank = get_rank()

    # Gather the episode counts from all ranks.
    all_task_num_of_collected_episodes_obj = [None for _ in range(world_size)]
    dist.all_gather_object(all_task_num_of_collected_episodes_obj, buffer_num_of_collected_episodes)

    # Concatenate the lists from all ranks into a single flat list.
    all_task_num_of_collected_episodes = [item for sublist in all_task_num_of_collected_episodes_obj for item in sublist]
    if rank == 0:
        logging.info(f'All task collected episodes: {all_task_num_of_collected_episodes}')

    # Calculate the inverse weight for each task. Adding 1 to avoid division by zero.
    inv_episodes = np.array([1.0 / (episodes + 1) for episodes in all_task_num_of_collected_episodes])
    inv_sum = np.sum(inv_episodes)

    # The total batch size is defined in the config of the first task.
    total_batch_size = cfgs[0].policy.total_batch_size

    # Define a dynamic range for batch sizes to prevent extreme values.
    avg_batch_size = total_batch_size / world_size
    min_batch_size = avg_batch_size / clip_scale
    max_batch_size = avg_batch_size * clip_scale

    # Calculate task weights based on inverse proportionality, smoothed by alpha.
    task_weights = (inv_episodes / inv_sum) ** alpha
    batch_sizes = total_batch_size * task_weights

    # Clip the batch sizes to the calculated dynamic range.
    batch_sizes = np.clip(batch_sizes, min_batch_size, max_batch_size)

    # Ensure batch sizes are integers.
    batch_sizes = [int(size) for size in batch_sizes]

    return batch_sizes


def train_unizero_multitask_segment_eval(
        input_cfg_list: List[Tuple[int, Tuple[Dict[str, Any], Dict[str, Any]]]],
        seed: int = 0,
        model: Optional[torch.nn.Module] = None,
        model_path: Optional[str] = None,
        max_train_iter: Optional[int] = int(1e10),
        max_env_step: Optional[int] = int(1e10),
) -> 'Policy':
    """
    Overview:
        The main training entry point for UniZero, as proposed in the paper "UniZero: Generalized and Efficient Planning
        with Scalable Latent World Models" (https://arxiv.org/abs/2406.10667). This function sets up a distributed
        multi-task training environment where multiple reinforcement learning tasks are trained in parallel using a
        single shared model. It handles task distribution, component initialization (policy, learner, buffers, etc.),
        and the main training loop orchestration.
    Arguments:
        - input_cfg_list (:obj:`List[Tuple[int, Tuple[Dict, Dict]]]`): A list of configurations for each task. Each
          element is a tuple containing the task ID and its corresponding configuration dictionaries.
        - seed (:obj:`int`): The master random seed for reproducibility.
        - model (:obj:`Optional[torch.nn.Module]`): An optional pre-existing model instance. If None, a new model is
          created based on the config.
        - model_path (:obj:`Optional[str]`): An optional path to a pre-trained model checkpoint.
        - max_train_iter (:obj:`Optional[int]`): The maximum number of training iterations before termination.
        - max_env_step (:obj:`Optional[int]`): The maximum number of environment steps before termination.
    Returns:
        - (:obj:`'Policy'`): The trained policy instance after the training loop has converged or terminated.
    """
    # ==============================================================
    # 1. Initialization
    # ==============================================================

    # 1.1. Distributed Setup & Task Partitioning
    rank = get_rank()
    world_size = get_world_size()

    total_tasks = len(input_cfg_list)
    tasks_per_rank = total_tasks // world_size
    remainder = total_tasks % world_size

    if rank < remainder:
        start_idx = rank * (tasks_per_rank + 1)
        end_idx = start_idx + tasks_per_rank + 1
    else:
        start_idx = rank * tasks_per_rank + remainder
        end_idx = start_idx + tasks_per_rank

    tasks_for_this_rank = input_cfg_list[start_idx:end_idx]

    if not tasks_for_this_rank:
        logging.warning(f"Rank {rank}: No tasks assigned. This rank will be idle.")
        # Keep the process alive to participate in collective communications.
        dist.barrier()
        return

    logging.info(f"Rank {rank}/{world_size}: Handling tasks from index {start_idx} to {end_idx - 1}.")

    # 1.2. Shared Policy, Learner, and Logger Initialization
    # Use the configuration of the first task on this rank to create the shared components.
    _, (first_cfg, first_create_cfg) = tasks_for_this_rank[0]

    # Set task_num for learner logging purposes.
    for _, (cfg, _) in tasks_for_this_rank:
        cfg.policy.task_num = tasks_per_rank

    assert first_create_cfg.policy.type in ['unizero_multitask'], \
        "This entry point currently only supports 'unizero_multitask' policy type."

    first_cfg.policy.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f'Shared policy device: {first_cfg.policy.device}')

    # Compile the main configuration.
    cfg = compile_config(first_cfg, seed=seed, auto=True, create_cfg=first_create_cfg, save_cfg=True)

    # Create the shared policy.
    policy = create_policy(cfg.policy, model=model, enable_field=['learn', 'collect', 'eval'])

    # Load a pre-trained model if a path is provided.
    if model_path is not None:
        logging.info(f'Loading pre-trained model from: {model_path}')
        policy.learn_mode.load_state_dict(torch.load(model_path, map_location=cfg.policy.device))
        logging.info('Model loading complete.')

    # Create a TensorBoard logger for this rank.
    log_dir = os.path.join(f'./{cfg.exp_name}/log', f'serial_rank_{rank}')
    tb_logger = SummaryWriter(log_dir)

    # Create the shared learner instance.
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)

    # 1.3. Task-Specific Components Initialization
    cfgs, game_buffers, collectors, evaluators = [], [], [], []
    for task_id, (task_cfg, task_create_cfg) in tasks_for_this_rank:
        # Set a unique seed for each task to ensure diversity in data collection.
        task_seed = seed + task_id
        task_cfg.policy.device = 'cuda' if task_cfg.policy.cuda and torch.cuda.is_available() else 'cpu'
        task_cfg = compile_config(task_cfg, seed=task_seed, auto=True, create_cfg=task_create_cfg, save_cfg=True)

        policy.collect_mode.get_attribute('cfg').n_episode = task_cfg.policy.n_episode
        policy.eval_mode.get_attribute('cfg').n_episode = task_cfg.policy.n_episode

        # Create environment managers for collection and evaluation.
        env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(task_cfg.env)
        collector_env = create_env_manager(task_cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_env_cfg])
        evaluator_env = create_env_manager(task_cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])
        collector_env.seed(task_seed)
        evaluator_env.seed(task_seed, dynamic_seed=False)
        set_pkg_seed(task_seed, use_cuda=task_cfg.policy.cuda)

        # Create task-specific buffers, collectors, and evaluators.
        replay_buffer = GameBuffer(task_cfg.policy)
        replay_buffer.batch_size = task_cfg.policy.batch_size[task_id]

        collector = Collector(
            env=collector_env, policy=policy.collect_mode, tb_logger=tb_logger, exp_name=task_cfg.exp_name,
            policy_config=task_cfg.policy, task_id=task_id
        )
        evaluator = Evaluator(
            eval_freq=task_cfg.policy.eval_freq, n_evaluator_episode=task_cfg.env.n_evaluator_episode,
            stop_value=task_cfg.env.stop_value, env=evaluator_env, policy=policy.eval_mode,
            tb_logger=tb_logger, exp_name=task_cfg.exp_name, policy_config=task_cfg.policy, task_id=task_id
        )

        cfgs.append(task_cfg)
        game_buffers.append(replay_buffer)
        collectors.append(collector)
        evaluators.append(evaluator)

    learner.call_hook('before_run')

    # ==============================================================
    # 2. Main Training Loop
    # ==============================================================
    buffer_reanalyze_count = 0
    train_epoch = 0
    while True:
        if learner.train_iter >= max_train_iter or collector.envstep >= max_env_step:
            break

        # 2.1. Dynamic Batch Size Allocation (Optional)
        if cfg.policy.allocated_batch_sizes:
            # As training progresses, allow for a larger divergence in batch sizes.
            clip_scale = np.clip(1 + (3 * train_epoch / 1000), 1, 4)
            allocated_batch_sizes = allocate_batch_size(cfgs, game_buffers, alpha=1.0, clip_scale=clip_scale)
            if rank == 0:
                logging.info(f"Allocated batch sizes: {allocated_batch_sizes}")
            for task_cfg, replay_buffer in zip(cfgs, game_buffers):
                task_cfg.policy.batch_size = allocated_batch_sizes
                policy._cfg.batch_size = allocated_batch_sizes

        # 2.2. Collection and Evaluation Phase
        for task_cfg, collector, evaluator, replay_buffer in zip(cfgs, collectors, evaluators, game_buffers):
            log_buffer_memory_usage(learner.train_iter, replay_buffer, tb_logger, task_cfg.policy.task_id)

            # Determine exploration parameters for collection.
            collect_kwargs = {
                'temperature': visit_count_temperature(
                    task_cfg.policy.manual_temperature_decay, task_cfg.policy.fixed_temperature_value,
                    task_cfg.policy.threshold_training_steps_for_final_temperature, trained_steps=learner.train_iter
                ),
                'epsilon': 0.0
            }
            if task_cfg.policy.eps.eps_greedy_exploration_in_collect:
                epsilon_fn = get_epsilon_greedy_fn(
                    start=task_cfg.policy.eps.start, end=task_cfg.policy.eps.end,
                    decay=task_cfg.policy.eps.decay, type_=task_cfg.policy.eps.type
                )
                collect_kwargs['epsilon'] = epsilon_fn(collector.envstep)

            # Evaluate the policy periodically.
            if evaluator.should_eval(learner.train_iter):
                logging.info(f'Rank {rank} evaluating task_id: {task_cfg.policy.task_id}...')
                stop, reward = safe_eval(evaluator, learner, collector, rank, world_size)
                if stop is None or reward is None:
                    logging.warning(f"Rank {rank} evaluation for task {task_cfg.policy.task_id} failed or timed out.")
                else:
                    logging.info(f"Evaluation successful for task {task_cfg.policy.task_id}: stop={stop}, reward={reward}")

            # Collect new data.
            logging.info(f'Rank {rank} collecting for task_id: {task_cfg.policy.task_id}...')
            # NOTE: Resetting initial data is crucial in multi-task settings to avoid state leakage.
            collector._policy.reset(reset_init_data=True)
            new_data = collector.collect(train_iter=learner.train_iter, policy_kwargs=collect_kwargs)

            # Update the replay buffer.
            replay_buffer.push_game_segments(new_data)
            replay_buffer.remove_oldest_data_to_fit()

            # Periodically reanalyze the buffer to update value/policy targets with a more recent model.
            # This logic handles two cases for `buffer_reanalyze_freq`:
            # Case 1: freq < 1 (e.g., 0.5) -> Reanalyze every `1/freq` training epochs.
            if 0 < task_cfg.policy.buffer_reanalyze_freq < 1:
                if (train_epoch % int(1 / task_cfg.policy.buffer_reanalyze_freq) == 0 and
                        replay_buffer.get_num_of_transitions() // task_cfg.policy.num_unroll_steps >
                        int(task_cfg.policy.reanalyze_batch_size / task_cfg.policy.reanalyze_partition)):
                    with EasyTimer() as timer:
                        replay_buffer.reanalyze_buffer(task_cfg.policy.reanalyze_batch_size, policy)
                    buffer_reanalyze_count += 1
                    logging.info(f'Buffer reanalyze count: {buffer_reanalyze_count}, Time: {timer.value:.2f}s')

            logging.info(f'Rank {rank}: Data collection complete for task {task_cfg.policy.task_id}')

        # 2.3. Pre-Training Synchronization and Data Check
        # Check if any buffer has insufficient data for training.
        not_enough_data = any(
            rb.get_num_of_transitions() < cfg.policy.total_batch_size / world_size for rb in game_buffers
        )

        try:
            dist.barrier()
        except Exception as e:
            logging.error(f'Rank {rank}: Barrier failed before training with error {e}', exc_info=True)
            break

        # 2.4. Training Phase
        if not not_enough_data:
            update_per_collect = cfg.policy.update_per_collect
            for i in range(update_per_collect):
                train_data_multi_task = []
                envstep_multi_task = sum(c.envstep for c in collectors)

                for task_cfg, replay_buffer in zip(cfgs, game_buffers):
                    batch_size = task_cfg.policy.batch_size[task_cfg.policy.task_id]
                    if replay_buffer.get_num_of_transitions() > batch_size:
                        # Case 2: freq >= 1 -> Reanalyze `freq` times per collection cycle (spread across updates).
                        if task_cfg.policy.buffer_reanalyze_freq >= 1:
                            reanalyze_interval = update_per_collect // task_cfg.policy.buffer_reanalyze_freq
                            if (i % reanalyze_interval == 0 and
                                    replay_buffer.get_num_of_transitions() // task_cfg.policy.num_unroll_steps >
                                    int(task_cfg.policy.reanalyze_batch_size / task_cfg.policy.reanalyze_partition)):
                                with EasyTimer() as timer:
                                    replay_buffer.reanalyze_buffer(task_cfg.policy.reanalyze_batch_size, policy)
                                buffer_reanalyze_count += 1
                                logging.info(f'Buffer reanalyze count: {buffer_reanalyze_count}, Time: {timer.value:.2f}s')

                        # Sample data and append task_id for multi-task learning.
                        train_data = replay_buffer.sample(batch_size, policy)
                        train_data.append(task_cfg.policy.task_id)
                        train_data_multi_task.append(train_data)
                    else:
                        logging.warning(
                            f"Skipping training for task {task_cfg.policy.task_id}: insufficient data. "
                            f"Required: {batch_size}, Available: {replay_buffer.get_num_of_transitions()}"
                        )

                if train_data_multi_task:
                    # DDP handles gradient synchronization automatically.
                    learner.train(train_data_multi_task, envstep_multi_task)

                # Synchronize after each training step to maintain consistency.
                try:
                    dist.barrier()
                except Exception as e:
                    logging.error(f'Rank {rank}: Barrier failed during training step with error {e}', exc_info=True)
                    break
        else:
            logging.warning(f"Rank {rank}: Skipping training cycle due to insufficient data in one or more buffers.")

        train_epoch += 1
        policy.recompute_pos_emb_diff_and_clear_cache()

        # 2.5. Post-Training Synchronization and Termination Check
        try:
            dist.barrier()
        except Exception as e:
            logging.error(f'Rank {rank}: Barrier failed after training cycle with error {e}', exc_info=True)
            break

    learner.call_hook('after_run')
    logging.info(f"Rank {rank}: Training finished.")
    return policy
# -*- coding: utf-8 -*-
"""
Main entry point for training UniZero in a multi-task setting using Distributed Data Parallel (DDP).
This script is designed to handle the complexities of multi-task reinforcement learning,
including dynamic resource allocation, task-specific data handling, and synchronized training across multiple processes.
For more details on the UniZero algorithm, please refer to the paper: https://arxiv.org/abs/2406.10667.
"""
import concurrent.futures
import logging
import os
from collections import defaultdict
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from ding.config import compile_config
from ding.envs import create_env_manager, get_vec_env_setting
from ding.policy import create_policy, Policy
from ding.rl_utils import get_epsilon_greedy_fn
from ding.utils import EasyTimer, get_rank, get_world_size, set_pkg_seed
from ding.worker import BaseLearner
from tensorboardX import SummaryWriter

from lzero.entry.utils import log_buffer_memory_usage, TemperatureScheduler
from lzero.policy import visit_count_temperature
from lzero.worker import MuZeroEvaluator
from lzero.worker import MuZeroSegmentCollector

# ==============================================================
# 1. Global Constants and Configurations
# ==============================================================

# Timeout for the evaluation process in seconds.
EVAL_TIMEOUT_SECONDS = 12000

# Define benchmark scores for Atari 100k.
ATARI_RANDOM_SCORES = np.array([
    227.8, 5.8, 222.4, 210.0, 14.2, 2360.0, 0.1, 1.7, 811.0, 10780.5,
    152.1, 0.0, 65.2, 257.6, 1027.0, 29.0, 52.0, 1598.0, 258.5, 307.3,
    -20.7, 24.9, 163.9, 11.5, 68.4, 533.4
])
ATARI_HUMAN_SCORES = np.array([
    7127.7, 1719.5, 742.0, 8503.3, 753.1, 37187.5, 12.1, 30.5, 7387.8, 35829.4,
    1971.0, 29.6, 4334.7, 2412.5, 30826.4, 302.8, 3035.0, 2665.5, 22736.3, 6951.6,
    14.6, 69571.3, 13455.0, 7845.0, 42054.7, 11693.2
])

# Define benchmark scores for DeepMind Control Suite (DMC).
DMC_RANDOM_SCORES = np.zeros(26)
DMC_HUMAN_SCORES = np.ones(26) * 1000

# The new order of tasks corresponds to the original indices.
# New order: [Pong, MsPacman, Seaquest, Boxing, Alien, ChopperCommand, Hero, RoadRunner,
#            Amidar, Assault, Asterix, BankHeist, BattleZone, CrazyClimber, DemonAttack,
#            Freeway, Frostbite, Gopher, Jamesbond, Kangaroo, Krull, KungFuMaster,
#            PrivateEye, UpNDown, Qbert, Breakout]
TASK_REORDER_INDICES = [
    20, 19, 24, 6, 0, 8, 14, 23, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 15, 16, 17, 18, 21, 25, 22, 7
]


# ==============================================================
# 2. Utility Functions
# ==============================================================

def get_reordered_benchmark_scores(benchmark_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Overview:
        Get the reordered random and human benchmark scores based on the benchmark name.
    Arguments:
        - benchmark_name (:obj:`str`): The name of the benchmark, e.g., "atari" or "dmc".
    Returns:
        - Tuple[np.ndarray, np.ndarray]: A tuple containing the reordered random scores and human scores.
    """
    if benchmark_name == "atari":
        random_scores, human_scores = ATARI_RANDOM_SCORES, ATARI_HUMAN_SCORES
    elif benchmark_name == "dmc":
        random_scores, human_scores = DMC_RANDOM_SCORES, DMC_HUMAN_SCORES
    else:
        raise ValueError(f"Unsupported benchmark_name: {benchmark_name}")

    reordered_random_scores = random_scores[TASK_REORDER_INDICES]
    reordered_human_scores = human_scores[TASK_REORDER_INDICES]
    return reordered_random_scores, reordered_human_scores


def compute_unizero_mt_normalized_stats(
        eval_returns: Dict[int, float],
        random_scores: np.ndarray,
        human_scores: np.ndarray
) -> Tuple[Optional[float], Optional[float]]:
    """
    Overview:
        Compute the Human-Normalized Mean and Median from evaluation returns.
    Arguments:
        - eval_returns (:obj:`Dict[int, float]`): A dictionary mapping task_id to its evaluation return.
        - random_scores (:obj:`np.ndarray`): An array of random scores for each task.
        - human_scores (:obj:`np.ndarray`): An array of human scores for each task.
    Returns:
        - Tuple[Optional[float], Optional[float]]: A tuple of (mean, median). Returns (None, None) if no valid data.
    """
    normalized = []
    for tid, ret in eval_returns.items():
        if ret is None:
            continue
        # Denominator for normalization.
        denom = human_scores[tid] - random_scores[tid]
        if denom == 0:
            continue
        normalized.append((ret - random_scores[tid]) / denom)

    if not normalized:
        return None, None

    arr = np.asarray(normalized, dtype=np.float32)
    return float(arr.mean()), float(np.median(arr))


def safe_eval(
        evaluator: MuZeroEvaluator,
        learner: BaseLearner,
        collector: MuZeroSegmentCollector,
        rank: int,
        world_size: int
) -> Tuple[Optional[bool], Optional[Dict[str, Any]]]:
    """
    Overview:
        Execute the evaluation process with a timeout to prevent hanging.
    Arguments:
        - evaluator (:obj:`MuZeroEvaluator`): The evaluator instance.
        - learner (:obj:`BaseLearner`): The learner instance.
        - collector (:obj:`MuZeroSegmentCollector`): The collector instance.
        - rank (:obj:`int`): The rank of the current process.
        - world_size (:obj:`int`): The total number of processes.
    Returns:
        - Tuple[Optional[bool], Optional[Dict[str, Any]]]: A tuple of (stop_flag, reward_dict).
          Returns (None, None) on timeout or error.
    """
    try:
        logging.info(f"========= Evaluation Start: Rank {rank}/{world_size} =========")
        # Ensure the stop_event is clear before starting evaluation.
        evaluator.stop_event.clear()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(evaluator.eval, learner.save_checkpoint, learner.train_iter, collector.envstep)
            try:
                stop, reward = future.result(timeout=EVAL_TIMEOUT_SECONDS)
            except concurrent.futures.TimeoutError:
                evaluator.stop_event.set()
                logging.error(
                    f"Evaluation timed out on Rank {rank}/{world_size} after {EVAL_TIMEOUT_SECONDS} seconds."
                )
                return None, None

        logging.info(f"====== Evaluation End: Rank {rank}/{world_size} ======")
        return stop, reward
    except Exception as e:
        logging.error(f"An error occurred during evaluation on Rank {rank}/{world_size}: {e}")
        return None, None


def allocate_batch_size(
        cfgs: List[Dict],
        game_buffers: List[Any],
        total_batch_size: int,
        alpha: float = 1.0,
        clip_scale: int = 1
) -> List[int]:
    """
    Overview:
        Dynamically allocate batch sizes for different tasks based on the inverse of collected episodes.
        This helps to balance training focus across tasks.
    Arguments:
        - cfgs (:obj:`List[Dict]`): List of configurations for each task.
        - game_buffers (:obj:`List[Any]`): List of replay buffer instances for each task.
        - total_batch_size (:obj:`int`): The total batch size to be distributed among all tasks.
        - alpha (:obj:`float`): Hyperparameter to control the inverse proportion. Defaults to 1.0.
        - clip_scale (:obj:`int`): Scale factor for dynamic clipping of batch sizes. Defaults to 1.
    Returns:
        - List[int]: A list of allocated batch sizes for each task.
    """
    # Extract the number of collected episodes for each task on the current rank.
    local_episodes = [buffer.num_of_collected_episodes for buffer in game_buffers]

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # Gather the number of episodes from all ranks.
    all_task_episodes = [None for _ in range(world_size)]
    dist.all_gather_object(all_task_episodes, local_episodes)

    # Flatten the list of lists into a single list of episode counts for all tasks.
    flat_task_episodes = [episode for sublist in all_task_episodes for episode in sublist]
    if rank == 0:
        logging.info(f'Number of collected episodes for all tasks: {flat_task_episodes}')

    # Calculate weights inversely proportional to the number of episodes.
    inv_episodes = np.array([1.0 / (episodes + 1) for episodes in flat_task_episodes])
    inv_sum = np.sum(inv_episodes)

    # Define dynamic min/max batch size range.
    avg_batch_size = total_batch_size / world_size
    min_batch_size = avg_batch_size / clip_scale
    max_batch_size = avg_batch_size * clip_scale

    # Calculate batch sizes based on task weights.
    task_weights = (inv_episodes / inv_sum) ** alpha
    batch_sizes = total_batch_size * task_weights

    # Clip batch sizes to be within the dynamic range.
    batch_sizes = np.clip(batch_sizes, min_batch_size, max_batch_size)

    return [int(size) for size in batch_sizes]


def symlog(x: torch.Tensor) -> torch.Tensor:
    """
    Overview:
        Apply the symlog transformation: sign(x) * log(|x| + 1).
        This helps in normalizing target values with large magnitudes.
    Arguments:
        - x (:obj:`torch.Tensor`): The input tensor.
    Returns:
        - torch.Tensor: The transformed tensor.
    """
    return torch.sign(x) * torch.log(torch.abs(x) + 1)


def inv_symlog(x: torch.Tensor) -> torch.Tensor:
    """
    Overview:
        Apply the inverse of the symlog transformation: sign(x) * (exp(|x|) - 1).
    Arguments:
        - x (:obj:`torch.Tensor`): The input tensor.
    Returns:
        - torch.Tensor: The inverse-transformed tensor.
    """
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


# ==============================================================
# 3. Main Trainer Class
# ==============================================================

class UniZeroMultiTaskTrainer:
    """
    Overview:
        The main trainer class for UniZero in a multi-task setting.
        It encapsulates the entire training pipeline, including setup, data collection,
        evaluation, and learning steps.
    """

    def __init__(
            self,
            input_cfg_list: List[Tuple[int, Tuple[dict, dict]]],
            seed: int = 0,
            model: Optional[nn.Module] = None,
            model_path: Optional[str] = None,
            max_train_iter: int = int(1e10),
            max_env_step: int = int(1e10),
            benchmark_name: str = "atari"
    ) -> None:
        """
        Overview:
            Initialize the UniZeroMultiTaskTrainer.
        Arguments:
            - input_cfg_list (:obj:`List[Tuple[int, Tuple[dict, dict]]]`): List of task configurations.
            - seed (:obj:`int`): The random seed.
            - model (:obj:`Optional[nn.Module]`): An optional pre-existing model instance.
            - model_path (:obj:`Optional[str]`): Path to a pre-trained model checkpoint.
            - max_train_iter (:obj:`int`): Maximum number of training iterations.
            - max_env_step (:obj:`int`): Maximum number of environment steps.
            - benchmark_name (:obj:`str`): Name of the benchmark ("atari" or "dmc").
        """
        self.input_cfg_list = input_cfg_list
        self.seed = seed
        self.model = model
        self.model_path = model_path
        self.max_train_iter = max_train_iter
        self.max_env_step = max_env_step
        self.benchmark_name = benchmark_name

        self._setup_distributed()
        self._initialize_components()

    def _setup_distributed(self) -> None:
        """
        Overview:
            Set up the distributed environment, including rank, world size, and task allocation.
        """
        self.rank = get_rank()
        self.world_size = get_world_size()

        total_tasks = len(self.input_cfg_list)
        tasks_per_rank = total_tasks // self.world_size
        remainder = total_tasks % self.world_size

        if self.rank < remainder:
            start_idx = self.rank * (tasks_per_rank + 1)
            end_idx = start_idx + tasks_per_rank + 1
        else:
            start_idx = self.rank * tasks_per_rank + remainder
            end_idx = start_idx + tasks_per_rank

        self.tasks_for_this_rank = self.input_cfg_list[start_idx:end_idx]
        if not self.tasks_for_this_rank:
            logging.warning(f"Rank {self.rank}: No tasks assigned, will proceed without action.")
        else:
            logging.info(f"Rank {self.rank}/{self.world_size} is handling tasks from index {start_idx} to {end_idx - 1}.")

    def _initialize_components(self) -> None:
        """
        Overview:
            Initialize all core components, including policy, learner, collectors, evaluators,
            and replay buffers for the assigned tasks.
        """
        self.cfgs, self.game_buffers, self.collectors, self.evaluators = [], [], [], []
        self.collector_envs, self.evaluator_envs = [], []
        self.policy = None
        self.learner = None
        self.tb_logger = None

        if not self.tasks_for_this_rank:
            return

        # Use the first task's config to create a shared policy and learner.
        _, [main_cfg, main_create_cfg] = self.tasks_for_this_rank[0]

        # Ensure the policy type is supported.
        policy_type = main_create_cfg.policy.type
        assert policy_type in ['unizero_multitask', 'sampled_unizero_multitask'], \
            f"Policy type '{policy_type}' is not supported. Use 'unizero_multitask' or 'sampled_unizero_multitask'."

        if policy_type == 'unizero_multitask':
            from lzero.mcts import UniZeroGameBuffer as GameBuffer
        else:  # sampled_unizero_multitask
            from lzero.mcts import SampledUniZeroGameBuffer as GameBuffer

        # Set device and compile the main config.
        main_cfg.policy.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.cfg = compile_config(main_cfg, seed=self.seed, auto=True, create_cfg=main_create_cfg, save_cfg=True)

        # Create shared policy and learner.
        self.policy = create_policy(self.cfg.policy, model=self.model, enable_field=['learn', 'collect', 'eval'])
        if self.model_path:
            logging.info(f'Loading pre-trained model from: {self.model_path}')
            self.policy.learn_mode.load_state_dict(torch.load(self.model_path, map_location=self.cfg.policy.device))
            logging.info('Model loading complete.')

        log_dir = os.path.join(f'./{self.cfg.exp_name}/log', f'serial_rank_{self.rank}')
        self.tb_logger = SummaryWriter(log_dir)
        self.learner = BaseLearner(self.cfg.policy.learn.learner, self.policy.learn_mode, self.tb_logger,
                                   exp_name=self.cfg.exp_name)
        self.learner.call_hook('before_run')

        # Initialize components for each assigned task.
        for task_id, [cfg, create_cfg] in self.tasks_for_this_rank:
            cfg.policy.device = 'cuda' if cfg.policy.cuda and torch.cuda.is_available() else 'cpu'
            compiled_cfg = compile_config(cfg, seed=self.seed + task_id, auto=True, create_cfg=create_cfg,
                                          save_cfg=True)
            self.cfgs.append(compiled_cfg)

            env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(compiled_cfg.env)
            collector_env = create_env_manager(compiled_cfg.env.manager,
                                               [partial(env_fn, cfg=c) for c in collector_env_cfg])
            evaluator_env = create_env_manager(compiled_cfg.env.manager,
                                               [partial(env_fn, cfg=c) for c in evaluator_env_cfg])
            collector_env.seed(self.seed + task_id)
            evaluator_env.seed(self.seed + task_id, dynamic_seed=False)
            set_pkg_seed(self.seed + task_id, use_cuda=compiled_cfg.policy.cuda)

            replay_buffer = GameBuffer(compiled_cfg.policy)
            replay_buffer.batch_size = compiled_cfg.policy.batch_size[task_id]
            self.game_buffers.append(replay_buffer)

            self.collectors.append(
                MuZeroSegmentCollector(
                    env=collector_env,
                    policy=self.policy.collect_mode,
                    tb_logger=self.tb_logger,
                    exp_name=compiled_cfg.exp_name,
                    policy_config=compiled_cfg.policy,
                    task_id=task_id
                )
            )
            self.evaluators.append(
                MuZeroEvaluator(
                    eval_freq=compiled_cfg.policy.eval_freq,
                    n_evaluator_episode=compiled_cfg.env.n_evaluator_episode,
                    stop_value=compiled_cfg.env.stop_value,
                    env=evaluator_env,
                    policy=self.policy.eval_mode,
                    tb_logger=self.tb_logger,
                    exp_name=compiled_cfg.exp_name,
                    policy_config=compiled_cfg.policy,
                    task_id=task_id
                )
            )

        # Initialize benchmark scores and other training-related states.
        self.random_scores, self.human_scores = get_reordered_benchmark_scores(self.benchmark_name)
        self.global_eval_returns = defaultdict(lambda: None)
        self.task_returns = {}
        self.train_epoch = 0
        self.timer = EasyTimer()

        self.temperature_scheduler = TemperatureScheduler(
            initial_temp=10.0, final_temp=1.0, threshold_steps=int(1e4), mode='linear'
        )

    def run(self) -> Optional[Policy]:
        """
        Overview:
            The main training loop. It orchestrates data collection, evaluation, and model updates.
        Returns:
            - Optional[Policy]: The trained policy, or None if training was not initialized.
        """
        if not self.tasks_for_this_rank:
            return None

        while not self._check_termination():
            self._update_dynamic_batch_sizes()
            self._collect_step()
            self._evaluation_step()

            if not self._is_data_sufficient():
                continue

            self._train_loop()
            self.train_epoch += 1
            self.policy.recompute_pos_emb_diff_and_clear_cache()

            try:
                dist.barrier()
                logging.info(f'Rank {self.rank}: Passed post-training synchronization barrier.')
            except Exception as e:
                logging.error(f'Rank {self.rank}: Synchronization barrier failed: {e}')
                break

        self._shutdown()
        return self.policy

    def _collect_step(self) -> None:
        """
        Overview:
            Perform one step of data collection for all assigned tasks.
        """
        for i, (cfg, collector, replay_buffer) in enumerate(zip(self.cfgs, self.collectors, self.game_buffers)):
            task_id = cfg.policy.task_id
            log_buffer_memory_usage(self.learner.train_iter, replay_buffer, self.tb_logger, task_id)

            collect_kwargs = {
                'temperature': visit_count_temperature(
                    cfg.policy.manual_temperature_decay,
                    cfg.policy.fixed_temperature_value,
                    cfg.policy.threshold_training_steps_for_final_temperature,
                    trained_steps=self.learner.train_iter
                ),
                'epsilon': 0.0
            }
            if cfg.policy.eps.eps_greedy_exploration_in_collect:
                eps_fn = get_epsilon_greedy_fn(
                    start=cfg.policy.eps.start, end=cfg.policy.eps.end,
                    decay=cfg.policy.eps.decay, type_=cfg.policy.eps.type
                )
                collect_kwargs['epsilon'] = eps_fn(collector.envstep)

            logging.info(f'Starting collection for task_id: {task_id} on Rank {self.rank}...')
            collector._policy.reset(reset_init_data=True, task_id=task_id)
            new_data = collector.collect(train_iter=self.learner.train_iter, policy_kwargs=collect_kwargs)
            replay_buffer.push_game_segments(new_data)
            replay_buffer.remove_oldest_data_to_fit()
            logging.info(f'Finished collection for task_id: {task_id} on Rank {self.rank}.')

    def _evaluation_step(self) -> None:
        """
        Overview:
            Perform evaluation if the current iteration is an evaluation step.
            It also computes and syncs task weights based on evaluation results.
        """
        if not (self.learner.train_iter > 10 and self.learner.train_iter % self.cfg.policy.eval_freq == 0):
            return

        for i, (cfg, collector, evaluator) in enumerate(zip(self.cfgs, self.collectors, self.evaluators)):
            task_id = cfg.policy.task_id
            logging.info(f'Evaluating task_id: {task_id} on Rank {self.rank}...')
            evaluator._policy.reset(reset_init_data=True, task_id=task_id)
            stop, reward_dict = safe_eval(evaluator, self.learner, collector, self.rank, self.world_size)

            if reward_dict is None:
                logging.warning(f"Evaluation failed for task {task_id} on Rank {self.rank}. Setting reward to infinity.")
                self.task_returns[task_id] = float('inf')
            else:
                eval_mean_reward = reward_dict.get('eval_episode_return_mean', float('inf'))
                logging.info(f"Task {task_id} evaluation reward: {eval_mean_reward}")
                self.task_returns[task_id] = eval_mean_reward

        self._sync_and_log_evaluation_metrics()

    def _sync_and_log_evaluation_metrics(self) -> None:
        """
        Overview:
            Synchronize evaluation results across all ranks and log normalized statistics.
        """
        try:
            dist.barrier()
            all_task_returns = [None for _ in range(self.world_size)]
            dist.all_gather_object(all_task_returns, self.task_returns)

            merged_task_returns = {}
            for returns in all_task_returns:
                if returns:
                    merged_task_returns.update(returns)
            logging.warning(f"Rank {self.rank}: Merged task returns: {merged_task_returns}")

            for tid, ret in merged_task_returns.items():
                self.global_eval_returns[tid] = ret

            uni_mean, uni_median = compute_unizero_mt_normalized_stats(
                self.global_eval_returns, self.random_scores, self.human_scores
            )

            if uni_mean is not None and self.rank == 0:
                self.tb_logger.add_scalar('UniZero-MT/NormalizedMean', uni_mean, global_step=self.learner.train_iter)
                self.tb_logger.add_scalar('UniZero-MT/NormalizedMedian', uni_median, global_step=self.learner.train_iter)
                logging.info(f"UniZero-MT Norm Mean={uni_mean:.4f}, Median={uni_median:.4f}")

        except Exception as e:
            logging.error(f'Rank {self.rank}: Failed to sync evaluation metrics: {e}')

    def _train_loop(self) -> None:
        """
        Overview:
            Execute the main training loop for a fixed number of updates per collection cycle.
        """
        update_per_collect = self.cfg.policy.update_per_collect
        task_exploitation_weight = None

        for i in range(update_per_collect):
            train_data_multi_task = []
            envstep_multi_task = 0
            for cfg, collector, replay_buffer in zip(self.cfgs, self.collectors, self.game_buffers):
                envstep_multi_task += collector.envstep
                batch_size = cfg.policy.batch_size[cfg.policy.task_id]
                if replay_buffer.get_num_of_transitions() > batch_size:
                    train_data = replay_buffer.sample(batch_size, self.policy)
                    train_data.append(cfg.policy.task_id)  # Append task_id for differentiation
                    train_data_multi_task.append(train_data)
                else:
                    logging.warning(f"Not enough data in replay buffer for task {cfg.policy.task_id} to sample a mini-batch.")
                    break
            
            if not train_data_multi_task:
                continue

            learn_kwargs = {'task_weights': task_exploitation_weight}
            log_vars = self.learner.train(train_data_multi_task, envstep_multi_task, policy_kwargs=learn_kwargs)

            # On the first update, calculate and sync exploitation weights if enabled.
            if i == 0 and self.cfg.policy.use_task_exploitation_weight:
                task_exploitation_weight = self._calculate_and_sync_exploitation_weights(log_vars)

            # Update priorities if priority sampling is enabled.
            if self.cfg.policy.use_priority:
                self._update_priorities(train_data_multi_task, log_vars)

    def _calculate_and_sync_exploitation_weights(self, log_vars: List[Dict]) -> Optional[Dict]:
        """
        Overview:
            Calculate task exploitation weights based on observation loss and synchronize them across all ranks.
        Arguments:
            - log_vars (:obj:`List[Dict]`): A list of log variables from the learner.
        Returns:
            - Optional[Dict]: A dictionary of task exploitation weights.
        """
        try:
            dist.barrier()
            local_obs_loss = {}
            for cfg in self.cfgs:
                task_id = cfg.policy.task_id
                key = f'noreduce_obs_loss_task{task_id}'
                if key in log_vars[0]:
                    local_obs_loss[task_id] = log_vars[0][key]

            all_obs_loss = [None for _ in range(self.world_size)]
            dist.all_gather_object(all_obs_loss, local_obs_loss)

            global_obs_loss = {}
            for obs_loss_part in all_obs_loss:
                if obs_loss_part:
                    global_obs_loss.update(obs_loss_part)

            if global_obs_loss:
                # This function is not provided in the original code, assuming a placeholder.
                # Replace `compute_task_weights` with the actual implementation.
                task_weights = {} # compute_task_weights(global_obs_loss, option="rank", temperature=1)
                dist.broadcast_object_list([task_weights], src=0)
                logging.info(f"Rank {self.rank}, task_exploitation_weight: {task_weights}")
                return task_weights
            else:
                logging.warning("Cannot compute exploitation weights; observation loss data is empty.")
                return None
        except Exception as e:
            logging.error(f'Rank {self.rank}: Failed to sync task exploitation weights: {e}')
            raise e

    def _update_priorities(self, train_data_multi_task: List, log_vars: List[Dict]) -> None:
        """
        Overview:
            Update the priorities in the replay buffer if priority sampling is used.
        Arguments:
            - train_data_multi_task (:obj:`List`): The training data sampled from buffers.
            - log_vars (:obj:`List[Dict]`): A list of log variables from the learner.
        """
        for idx, (cfg, replay_buffer) in enumerate(zip(self.cfgs, self.game_buffers)):
            task_id = cfg.policy.task_id
            priority_key = f'value_priority_task{task_id}'
            if priority_key in log_vars[0]:
                priorities = log_vars[0][priority_key]
                replay_buffer.update_priority(train_data_multi_task[idx], priorities)

    def _update_dynamic_batch_sizes(self) -> None:
        """
        Overview:
            Update batch sizes dynamically if the feature is enabled in the config.
        """
        if self.cfg.policy.allocated_batch_sizes:
            clip_scale = np.clip(1 + (3 * self.train_epoch / 1000), 1, 4)
            allocated_sizes = allocate_batch_size(
                self.cfgs, self.game_buffers, self.cfg.policy.total_batch_size, alpha=1.0, clip_scale=clip_scale
            )
            if self.rank == 0:
                logging.info(f"Allocated batch sizes: {allocated_sizes}")
            for cfg in self.cfgs:
                cfg.policy.batch_size = allocated_sizes
            self.policy._cfg.batch_size = allocated_sizes

    def _is_data_sufficient(self) -> bool:
        """
        Overview:
            Check if there is enough data in the replay buffers to start training.
        Returns:
            - bool: True if data is sufficient, False otherwise.
        """
        min_transitions_needed = self.cfg.policy.total_batch_size / self.world_size
        is_insufficient = any(
            rb.get_num_of_transitions() < min_transitions_needed for rb in self.game_buffers
        )
        if is_insufficient:
            logging.warning("Not enough data across all task buffers to start training.")
        return not is_insufficient

    def _check_termination(self) -> bool:
        """
        Overview:
            Check if the training should be terminated based on max iterations or environment steps.
        Returns:
            - bool: True if termination conditions are met, False otherwise.
        """
        try:
            local_envsteps = [c.envstep for c in self.collectors]
            all_envsteps_obj = [None for _ in range(self.world_size)]
            dist.all_gather_object(all_envsteps_obj, local_envsteps)
            
            flat_envsteps = [step for sublist in all_envsteps_obj for step in sublist]
            if not flat_envsteps:
                 return False
                 
            min_envstep = min(flat_envsteps)
            if min_envstep >= self.max_env_step:
                logging.info(f"All tasks reached max_env_step ({self.max_env_step}). Terminating.")
                return True

            if self.learner.train_iter >= self.max_train_iter:
                logging.info(f"Reached max_train_iter ({self.max_train_iter}). Terminating.")
                return True

        except Exception as e:
            logging.error(f'Rank {self.rank}: Termination check failed: {e}')
            return True  # Terminate on error to prevent hanging.
        return False

    def _shutdown(self) -> None:
        """
        Overview:
            Perform cleanup operations at the end of training.
        """
        if self.learner:
            self.learner.call_hook('after_run')
        logging.info(f"Trainer on Rank {self.rank} is shutting down.")


def train_unizero_multitask_segment_ddp(
        input_cfg_list: List[Tuple[int, Tuple[dict, dict]]],
        seed: int = 0,
        model: Optional[nn.Module] = None,
        model_path: Optional[str] = None,
        max_train_iter: Optional[int] = int(1e10),
        max_env_step: Optional[int] = int(1e10),
        benchmark_name: str = "atari"
) -> Optional[Policy]:
    """
    Overview:
        The main entry point for training UniZero. This function sets up and runs the
        UniZeroMultiTaskTrainer, which encapsulates the training logic. UniZero aims to
        enhance the planning capabilities of reinforcement learning agents by addressing
        limitations in MuZero-like algorithms, particularly in environments requiring
        long-term dependency modeling. For more details, see https://arxiv.org/abs/2406.10667.
    Arguments:
        - input_cfg_list (:obj:`List[Tuple[int, Tuple[dict, dict]]]`): A list of configurations for different tasks.
        - seed (:obj:`int`): The random seed.
        - model (:obj:`Optional[torch.nn.Module]`): An optional pre-existing torch.nn.Module instance.
        - model_path (:obj:`Optional[str]`): Path to a pre-trained model checkpoint.
        - max_train_iter (:obj:`Optional[int]`): The maximum number of policy update iterations.
        - max_env_step (:obj:`Optional[int]`): The maximum number of environment interaction steps.
        - benchmark_name (:obj:`str`): The name of the benchmark, e.g., "atari" or "dmc".
    Returns:
        - Optional[Policy]: The converged policy, or None if training did not complete successfully.
    """
    trainer = UniZeroMultiTaskTrainer(
        input_cfg_list=input_cfg_list,
        seed=seed,
        model=model,
        model_path=model_path,
        max_train_iter=max_train_iter,
        max_env_step=max_env_step,
        benchmark_name=benchmark_name,
    )
    return trainer.run()
# -*- coding: utf-8 -*-
"""
### 🛠️ Utility Modules

- **`utils.py`** - Common utility functions library
  - **Math & Tensor Utilities**:
    - `symlog`, `inv_symlog` - Symmetric logarithm transformations
    - `initialize_zeros_batch`, `initialize_pad_batch` - Batch initialization

  - **LoRA Utilities**:
    - `freeze_non_lora_parameters` - Freeze non-LoRA parameters

  - **Task & Curriculum Learning Utilities**:
    - `compute_task_weights` - Compute task weights
    - `TemperatureScheduler` - Temperature scheduler
    - `tasks_per_stage` - Calculate tasks per stage
    - `compute_unizero_mt_normalized_stats` - Compute normalized statistics
    - `allocate_batch_size` - Dynamically allocate batch sizes

  - **Distributed Training Utilities (DDP)**:
    - `is_ddp_enabled` - Check if DDP is enabled
    - `ddp_synchronize` - DDP synchronization
    - `ddp_all_reduce_sum` - DDP all-reduce sum

  - **RL Workflow Utilities**:
    - `calculate_update_per_collect` - Calculate updates per collection
    - `random_collect` - Random policy data collection
    - `convert_to_batch_for_unizero` - UniZero batch data conversion
    - `create_unizero_loss_metrics` - Create loss metrics function
    - `UniZeroDataLoader` - UniZero data loader

  - **Logging Utilities**:
    - `log_module_trainable_status` - Log module trainable status
    - `log_param_statistics` - Log parameter statistics
    - `log_buffer_memory_usage` - Log buffer memory usage
    - `log_buffer_run_time` - Log buffer runtime

  - **MoE Statistics Utilities**:
    - `merge_expert_stats_across_ranks` - Merge expert selection stats from distributed ranks
    - `create_heatmap_with_values_fast` - Fast Task-Expert heatmap generation
    - `collect_and_log_moe_statistics` - End-to-end MoE stats collection and TensorBoard logging
    - `jensen_shannon_divergence_batch_gpu` - GPU batch JS divergence for task distributions
    - `wasserstein_distance_batch_gpu` - GPU batch Wasserstein distance
    - `compute_distribution_divergences_optimized` - Optimized inter-task distribution divergence

- **`__init__.py`** - Package initialization file
  - Exports all training and evaluation entry functions
  - Exports commonly used functions from utility modules

"""

# ==============================================================================
# Imports
# ==============================================================================
from __future__ import annotations

import concurrent.futures
from ditk import logging
import math
import os
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import psutil
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from pympler.asizeof import asizeof
from tensorboardX import SummaryWriter
import time
from typing import Optional, Callable, Union, List, Tuple, Dict
from io import BytesIO
import concurrent.futures
import seaborn as sns
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt
# ==============================================================================
# Placeholder Types for External Dependencies
#
# To ensure type hints work without having the full definitions of these complex
# external classes, we define them as `Any`.
# ==============================================================================
EasyDict = Any
Policy = Any
RandomPolicy = Any
ISerialCollector = Any
BaseEnvManager = Any
IBuffer = Any
GameBuffer = Any
BaseLearner = Any
Evaluator = Any
Collector = Any


# ==============================================================================
# Global Constants
# ==============================================================================

# Timeout for evaluation process in seconds (200 minutes)
EVALUATION_TIMEOUT = 12000


# ==============================================================================
# Mathematical & Tensor Utilities
# ==============================================================================

def symlog(x: torch.Tensor) -> torch.Tensor:
    """
    Overview:
        Applies the symlog transformation to a tensor, which is useful for
        normalizing target values with large magnitude differences.
        The transformation is defined as: symlog(x) = sign(x) * log(|x| + 1).

    Arguments:
        - x (:obj:`torch.Tensor`): The input tensor.

    Returns:
        - torch.Tensor: The tensor after applying the symlog transformation.
    """
    return torch.sign(x) * torch.log(torch.abs(x) + 1)


def inv_symlog(x: torch.Tensor) -> torch.Tensor:
    """
    Overview:
        Applies the inverse of the symlog transformation to a tensor, restoring
        the original scale of the values.
        The transformation is defined as: inv_symlog(x) = sign(x) * (exp(|x|) - 1).

    Arguments:
        - x (:obj:`torch.Tensor`): The input tensor in symlog space.

    Returns:
        - torch.Tensor: The tensor restored to its original scale.
    """
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


# ==============================================================================
# LoRA (Low-Rank Adaptation) Utilities
# ==============================================================================

# A compiled regex pattern to efficiently detect LoRA-related parameters.
# It matches parameter names ending with:
# - .lora_A or .lora_B (for LoRA weights)
# - .adapter_scales.{digit}.logit (for learnable scale parameters)
_LORA_PAT = re.compile(r"\.(?:lora_[AB]|adapter_scales\.\d+\.logit)$")


def _is_lora_param(name: str) -> bool:
    """A helper function to check if a parameter name matches the LoRA pattern."""
    return bool(_LORA_PAT.search(name))


def freeze_non_lora_parameters(
    module: nn.Module,
    freeze: bool = True,
    *,
    verbose: bool = False,
) -> Tuple[int, int]:
    """
    Overview:
        Freezes or un-freezes all parameters in a module that are not identified
        as LoRA-related parameters. This is useful for curriculum learning stages
        where the backbone model is frozen and only LoRA adapters are trained.

    Arguments:
        - module (:obj:`nn.Module`): The PyTorch module to process (e.g., a transformer).
        - freeze (:obj:`bool`): If True, sets `requires_grad=False` for non-LoRA parameters.
                                If False, sets `requires_grad=True` for non-LoRA parameters.
        - verbose (:obj:`bool`): If True, prints a summary of trainable and frozen parameters.

    Returns:
        - Tuple[int, int]: A tuple containing the number of frozen parameters and trainable parameters.
    """
    n_frozen = 0
    n_trainable = 0

    for name, param in module.named_parameters():
        if _is_lora_param(name):
            # LoRA-related parameters should always be trainable.
            param.requires_grad = True
            n_trainable += 1
        else:
            # All other parameters are frozen or unfrozen based on the `freeze` flag.
            param.requires_grad = not freeze
            if param.requires_grad:
                n_trainable += 1
            else:
                n_frozen += 1

    if verbose:
        total = n_frozen + n_trainable
        # Ensure total is not zero to avoid division by zero error.
        percentage_trainable = (n_trainable / total * 100) if total > 0 else 0
        print(
            f"[freeze_non_lora] Trainable: {n_trainable}/{total} ({percentage_trainable:.1f}%), "
            f"Frozen: {n_frozen}"
        )
    return n_frozen, n_trainable


# ==============================================================================
# Task & Curriculum Learning Utilities
# ==============================================================================

def compute_task_weights(
    task_returns: Dict[str, float],
    option: str = "symlog",
    epsilon: float = 1e-6,
    temperature: float = 1.0,
    use_softmax: bool = False,
    reverse: bool = False,
    clip_min: float = 1e-2,
    clip_max: float = 1.0,
) -> Dict[str, float]:
    """
    Overview:
        Calculates sampling weights for different tasks based on their returns (e.g., rewards or losses).
        This function supports various normalization methods, softmax-based distribution,
        proportional/inverse weighting, and weight clipping.

    Arguments:
        - task_returns (:obj:`Dict[str, float]`): A dictionary mapping task IDs to their return values.
        - option (:obj:`str`): Normalization method. One of ["symlog", "max-min", "run-max-min", "rank", "none"].
        - epsilon (:obj:`float`): A small value to prevent division by zero.
        - temperature (:obj:`float`): A temperature parameter to control the sharpness of the weight distribution.
        - use_softmax (:obj:`bool`): If True, use softmax to compute weights; otherwise, use direct normalization.
        - reverse (:obj:`bool`): If True, weights are inversely proportional to returns; otherwise, directly proportional.
        - clip_min (:obj:`float`): The minimum value to clip the final weights to.
        - clip_max (:obj:`float`): The maximum value to clip the final weights to.

    Returns:
        - Dict[str, float]: A dictionary mapping task IDs to their computed weights.
    """
    if not task_returns:
        return {}

    task_ids = list(task_returns.keys())
    returns_tensor = torch.tensor(list(task_returns.values()), dtype=torch.float32)

    # Step 1: Normalize the returns based on the chosen option.
    scaled_returns: torch.Tensor
    if option == "symlog":
        scaled_returns = symlog(returns_tensor)
    elif option == "max-min":
        min_val, max_val = returns_tensor.min(), returns_tensor.max()
        scaled_returns = (returns_tensor - min_val) / (max_val - min_val + epsilon)
    elif option == "run-max-min":
        # Use function attributes to maintain state across calls, avoiding global variables.
        compute_task_weights.RUNNING_MAX = max(compute_task_weights.RUNNING_MAX, returns_tensor.max().item())
        compute_task_weights.RUNNING_MIN = min(compute_task_weights.RUNNING_MIN, returns_tensor.min().item())
        scaled_returns = (returns_tensor - compute_task_weights.RUNNING_MIN) / \
                         (compute_task_weights.RUNNING_MAX - compute_task_weights.RUNNING_MIN + epsilon)
    elif option == "rank":
        sorted_indices = torch.argsort(returns_tensor)
        ranks = torch.empty_like(returns_tensor)
        # Ranks are from 1 to N.
        ranks[sorted_indices] = torch.arange(1, len(returns_tensor) + 1, dtype=torch.float32)
        scaled_returns = ranks
    elif option == "none":
        scaled_returns = returns_tensor
    else:
        raise ValueError(f"Unsupported normalization option: {option}")

    # Step 2: Determine if weights should be proportional or inversely proportional to returns.
    if reverse:
        # Inverse proportion: smaller return -> higher weight.
        raw_weights = 1.0 / (scaled_returns + epsilon)
    else:
        # Direct proportion: higher return -> higher weight.
        raw_weights = scaled_returns

    # Step 3: Calculate final weights using either softmax or direct normalization.
    final_weights: np.ndarray
    safe_temperature = max(temperature, epsilon)
    if use_softmax:
        # Softmax provides a smooth distribution, often used with inverse weights.
        # A higher beta (lower temperature) makes the distribution sharper.
        beta = 1.0 / safe_temperature
        # The sign depends on whether we want to favor high or low raw_weights.
        # If reverse=True, raw_weights are high for low returns. We want to sample these more.
        # Softmax(logits) gives higher probability to higher logits.
        # So, logits should be proportional to the desired sampling probability.
        logits = raw_weights if reverse else -raw_weights
        final_weights = F.softmax(logits * beta, dim=0).numpy()
    else:
        # Direct normalization with temperature scaling.
        scaled_weights = raw_weights**(1 / safe_temperature)
        total_weight = scaled_weights.sum()
        normalized_weights = scaled_weights / (total_weight + epsilon)
        final_weights = normalized_weights.numpy()

    # Step 4: Clip weights to the desired range and create the result dictionary.
    weights_dict = {
        task_id: np.clip(weight, clip_min, clip_max)
        for task_id, weight in zip(task_ids, final_weights)
    }

    return weights_dict

# Initialize state for the 'run-max-min' option as function attributes.
compute_task_weights.RUNNING_MAX = -float('inf')
compute_task_weights.RUNNING_MIN = float('inf')


class TemperatureScheduler:
    """
    Overview:
        A scheduler to gradually adjust a temperature value over a specified number
        of training steps. This can be used for exploration or weighting schemes.

    Arguments:
        - initial_temp (:obj:`float`): The starting temperature.
        - final_temp (:obj:`float`): The target temperature to be reached after `threshold_steps`.
        - threshold_steps (:obj:`int`): The number of steps over which the temperature will anneal.
        - mode (:obj:`str`): The annealing mode, either 'linear' or 'exponential'.
    """

    def __init__(self, initial_temp: float, final_temp: float, threshold_steps: int, mode: str = 'linear'):
        if mode not in ['linear', 'exponential']:
            raise ValueError("Mode must be 'linear' or 'exponential'.")
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.threshold_steps = max(1, threshold_steps)  # Avoid division by zero
        self.mode = mode

    def get_temperature(self, current_step: int) -> float:
        """
        Overview:
            Calculates the temperature for the given training step.

        Arguments:
            - current_step (:obj:`int`): The current training step.

        Returns:
            - float: The calculated temperature for the current step.
        """
        if current_step >= self.threshold_steps:
            return self.final_temp

        progress = current_step / self.threshold_steps

        if self.mode == 'linear':
            return self.initial_temp - (self.initial_temp - self.final_temp) * progress
        else:  # 'exponential'
            # Exponential decay from initial_temp to final_temp
            # T(t) = T_initial * (T_final / T_initial)^(t / N)
            if self.initial_temp <= 0:
                 raise ValueError("Initial temperature must be positive for exponential decay.")
            scale = self.final_temp / self.initial_temp
            return self.initial_temp * (scale**progress)


def tasks_per_stage(unsolved: int, remain_lora: int) -> int:
    """
    Overview:
        Calculates the number of tasks to assign per LoRA adapter stage.
        It's the ceiling of the division of unsolved tasks by remaining adapters.

    Arguments:
        - unsolved (:obj:`int`): The number of tasks yet to be solved.
        - remain_lora (:obj:`int`): The number of available LoRA adapters.

    Returns:
        - int: The number of tasks to be handled in the current stage, at least 1.
    """
    return max(1, math.ceil(unsolved / max(remain_lora, 1)))


def compute_unizero_mt_normalized_stats(
    eval_returns: Dict[int, float],
    human_scores: Dict[int, float],
    random_scores: Dict[int, float]
) -> Tuple[Optional[float], Optional[float]]:
    """
    Overview:
        Calculates the Human-Normalized Mean and Median for a set of evaluation returns.
        If no valid returns are provided, it returns (None, None).

    Arguments:
        - eval_returns (:obj:`Dict[int, float]`): A dictionary of evaluation returns per task ID.
        - human_scores (:obj:`Dict[int, float]`): A dictionary of human expert scores per task ID.
        - random_scores (:obj:`Dict[int, float]`): A dictionary of random policy scores per task ID.

    Returns:
        - Tuple[Optional[float], Optional[float]]: A tuple containing the human-normalized mean and median.
    """
    normalized = []
    for tid, ret in eval_returns.items():
        if ret is None or tid not in human_scores or tid not in random_scores:
            continue
        denom = human_scores[tid] - random_scores[tid]
        if denom == 0:
            continue
        normalized.append((ret - random_scores[tid]) / denom)

    if not normalized:
        return None, None

    arr = np.asarray(normalized, dtype=np.float32)
    return float(arr.mean()), float(np.median(arr))


def allocate_batch_size(
    cfgs: List[EasyDict],
    game_buffers: List[GameBuffer],
    alpha: float = 1.0,
    clip_scale: int = 1
) -> List[int]:
    """
    Overview:
        Allocates batch sizes for different tasks inversely proportional to the
        number of collected episodes for each task. It also dynamically clips
        the batch size range to improve training stability.

    Arguments:
        - cfgs (:obj:`List[EasyDict]`): A list of configuration objects for each task.
        - game_buffers (:obj:`List[GameBuffer]`): A list of replay buffer instances for each task.
        - alpha (:obj:`float`): A hyperparameter to control the degree of inverse proportionality.
        - clip_scale (:obj:`int`): A scaling factor to determine the min/max batch size clip range.

    Returns:
        - List[int]: A list of allocated batch sizes for each task.
    """
    # This function assumes a DDP environment.
    if not dist.is_available() or not dist.is_initialized():
        # Fallback for non-DDP environment if needed, though the logic is DDP-centric.
        logging.warning("allocate_batch_size is designed for DDP and may not work as expected.")
        world_size = 1
        rank = 0
    else:
        world_size = dist.get_world_size()
        rank = dist.get_rank()

    # Extract the number of collected episodes from each local buffer.
    local_episodes = [buffer.num_of_collected_episodes for buffer in game_buffers]

    # Gather episode counts from all ranks.
    all_task_episodes_list = [None for _ in range(world_size)]
    dist.all_gather_object(all_task_episodes_list, local_episodes)

    # Flatten the list of lists into a single list of episode counts for all tasks.
    all_task_episodes = [ep for sublist in all_task_episodes_list for ep in sublist]

    if rank == 0:
        logging.info(f'All task collected episodes: {all_task_episodes}')

    # Calculate weights inversely proportional to episode counts.
    # Add 1 to avoid division by zero for new tasks.
    inv_episodes = np.array([1.0 / (episodes + 1) for episodes in all_task_episodes])
    inv_sum = np.sum(inv_episodes)

    # Total batch size is assumed to be consistent across configs.
    total_batch_size = cfgs[0].policy.total_batch_size

    # Define dynamic clipping range for batch sizes.
    avg_batch_size = total_batch_size / len(all_task_episodes)
    min_batch_size = avg_batch_size / clip_scale
    max_batch_size = avg_batch_size * clip_scale

    # Calculate batch sizes based on weights, apply alpha for smoothing.
    task_weights = (inv_episodes / inv_sum)**alpha
    batch_sizes = total_batch_size * task_weights

    # Clip and convert to integers.
    batch_sizes = np.clip(batch_sizes, min_batch_size, max_batch_size)
    batch_sizes = [int(size) for size in batch_sizes]

    return batch_sizes


# ==============================================================================
# Distributed Data Parallel (DDP) Utilities
# ==============================================================================

def is_ddp_enabled() -> bool:
    """
    Overview:
        Checks if the environment is set up for Distributed Data Parallel (DDP) training.

    Returns:
        - bool: True if `torch.distributed` is available and initialized, False otherwise.
    """
    return dist.is_available() and dist.is_initialized()


def ddp_synchronize() -> None:
    """
    Overview:
        Performs a barrier synchronization across all processes in a DDP group.
        This ensures that all processes reach this point before any of them proceed.
    """
    if is_ddp_enabled():
        dist.barrier()


def ddp_all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    """
    Overview:
        Performs an all-reduce operation (sum) on a given tensor across all
        processes in the DDP group.

    Arguments:
        - tensor (:obj:`torch.Tensor`): The tensor to be reduced.

    Returns:
        - torch.Tensor: The reduced tensor, with values summed across all processes.
    """
    if is_ddp_enabled():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


# ==============================================================================
# Reinforcement Learning Workflow Utilities
# ==============================================================================

def calculate_update_per_collect(
    cfg: EasyDict,
    new_data: List[List[torch.Tensor]],
    world_size: int = 1
) -> int:
    """
    Overview:
        Calculates the number of training updates to perform per data collection cycle.
        In a DDP setting, it synchronizes transition counts across all GPUs to ensure
        a consistent `update_per_collect` value.

    Arguments:
        - cfg (:obj:`EasyDict`): The configuration object containing policy settings.
                                 It's expected to have `cfg.policy.update_per_collect`,
                                 `cfg.policy.replay_ratio`, etc.
        - new_data (:obj:`List[List[torch.Tensor]]`): The newly collected data segments.
        - world_size (:obj:`int`): The total number of DDP processes.

    Returns:
        - int: The number of updates to perform.
    """
    update_per_collect = cfg.policy.get('update_per_collect')

    if update_per_collect is not None:
        return update_per_collect

    # If not explicitly set, calculate based on replay ratio.
    # Note: A game segment's length can be less than `game_segment_length` if it's the
    # final segment of an episode.
    collected_transitions_num = sum(
        min(len(game_segment), cfg.policy.game_segment_length)
        for game_segment in new_data[0]
    )

    if torch.cuda.is_available() and world_size > 1:
        # In DDP, synchronize the transition count across all GPUs.
        collected_transitions_tensor = torch.tensor(
            collected_transitions_num, dtype=torch.int64, device='cuda'
        )
        total_collected_transitions = ddp_all_reduce_sum(
            collected_transitions_tensor
        ).item()
        updates = int(total_collected_transitions * cfg.policy.replay_ratio)
    else:
        # In a single-process setup.
        updates = int(collected_transitions_num * cfg.policy.replay_ratio)

    return max(1, updates) # Ensure at least one update.



def random_collect(
    policy_cfg: EasyDict,
    policy: Policy,
    RandomPolicy: Callable,
    collector: ISerialCollector,
    collector_env: BaseEnvManager,
    replay_buffer: IBuffer,
    postprocess_data_fn: Optional[Callable] = None
) -> None:
    """
    Overview:
        Performs an initial data collection phase using a random policy to populate
        the replay buffer before training begins.

    Arguments:
        - policy_cfg (:obj:`EasyDict`): Configuration for the policy.
        - policy (:obj:`Policy`): The main training policy instance.
        - RandomPolicy (:obj:`Callable`): A constructor or class for creating a random policy.
        - collector (:obj:`ISerialCollector`): The data collector instance.
        - collector_env (:obj:`BaseEnvManager`): The environment manager.
        - replay_buffer (:obj:`IBuffer`): The replay buffer to store collected data.
        - postprocess_data_fn (:obj:`Optional[Callable]`): An optional function to process data after collection.
    """
    random_collect_episode_num = policy_cfg.get('random_collect_episode_num', 0)
    if random_collect_episode_num <= 0:
        return

    random_policy = RandomPolicy(cfg=policy_cfg, action_space=collector_env.env_ref.action_space)
    collector.reset_policy(random_policy.collect_mode)

    # Use neutral MCTS parameters for random collection.
    collect_kwargs = {'temperature': 1.0, 'epsilon': 0.0}

    new_data = collector.collect(
        n_episode=random_collect_episode_num,
        train_iter=0,
        policy_kwargs=collect_kwargs
    )

    if postprocess_data_fn:
        new_data = postprocess_data_fn(new_data)

    replay_buffer.push_game_segments(new_data)
    replay_buffer.remove_oldest_data_to_fit()

    # Restore the original policy to the collector.
    collector.reset_policy(policy.collect_mode)


def safe_eval(
    evaluator: Evaluator,
    learner: BaseLearner,
    collector: Collector,
    rank: int,
    world_size: int,
    timeout: int = EVALUATION_TIMEOUT
) -> Tuple[Optional[bool], Optional[Any]]:
    """
    Overview:
        Safely executes an evaluation task with a timeout to prevent hangs.
        This function runs the evaluation in a separate thread and enforces a timeout
        to ensure the training process doesn't get stuck during evaluation.

    Arguments:
        - evaluator (:obj:`Evaluator`): The evaluator instance.
        - learner (:obj:`BaseLearner`): The learner instance, used for saving checkpoints.
        - collector (:obj:`Collector`): The data collector instance, used to get current envstep.
        - rank (:obj:`int`): The rank of the current process in distributed training.
        - world_size (:obj:`int`): The total number of processes in distributed training.
        - timeout (:obj:`int`): The maximum time (in seconds) to wait for evaluation. Defaults to EVALUATION_TIMEOUT.

    Returns:
        - Tuple[Optional[bool], Optional[Any]]: A tuple containing:
            - stop_flag: Boolean indicating if training should stop (None if timeout/error)
            - reward_dict: Dictionary containing evaluation metrics (None if timeout/error)
    """
    try:
        logging.info(f"========= Evaluation starting on Rank {rank}/{world_size} =========")
        # Reset the stop_event to ensure it is not set before each evaluation.
        evaluator.stop_event.clear()

        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit the evaluation task to run in a separate thread.
            future = executor.submit(
                evaluator.eval,
                learner.save_checkpoint,
                learner.train_iter,
                collector.envstep
            )
            try:
                stop_flag, reward = future.result(timeout=timeout)
                logging.info(f"====== Evaluation finished on Rank {rank}/{world_size} ======")
                return stop_flag, reward
            except concurrent.futures.TimeoutError:
                # If a timeout occurs, set the stop_event to signal the evaluation thread to stop.
                evaluator.stop_event.set()
                logging.error(
                    f"Evaluation timed out on Rank {rank}/{world_size} after {timeout} seconds. "
                    f"Continuing training."
                )
                return None, None

    except Exception as e:
        logging.error(
            f"An error occurred during evaluation on Rank {rank}/{world_size}: {e}",
            exc_info=True
        )
        return None, None


def convert_to_batch_for_unizero(batch_data, policy_cfg, reward_support, value_support):
    """
    Overview:
        Convert replay buffer sample data to batch_for_unizero format for world_model.compute_loss.
        This function transforms the raw data from the replay buffer into the format expected
        by the UniZero world model's compute_loss method.

    Arguments:
        - batch_data: Data sampled from replay buffer (current_batch, target_batch)
        - policy_cfg: Policy configuration object
        - reward_support: Reward support tensor for categorical distribution
        - value_support: Value support tensor for categorical distribution

    Returns:
        - batch_for_unizero (:obj:`dict`): Dictionary containing formatted data for world model
    """
    from lzero.policy.utils import to_torch_float_tensor, prepare_obs, prepare_obs_stack_for_unizero
    from lzero.policy import scalar_transform, phi_transform

    # Unpack batch data
    current_batch, target_batch = batch_data[:2]
    obs_batch_ori, action_batch, target_action_batch, mask_batch, indices, weights, make_time, timestep_batch = current_batch
    target_reward, target_value, target_policy = target_batch

    # Prepare observations
    if policy_cfg.model.frame_stack_num > 1:
        obs_batch, obs_target_batch = prepare_obs_stack_for_unizero(obs_batch_ori, policy_cfg)
    else:
        obs_batch, obs_target_batch = prepare_obs(obs_batch_ori, policy_cfg)

    # Convert to tensors
    action_batch = torch.from_numpy(action_batch).to(policy_cfg.device).unsqueeze(-1).long()
    timestep_batch = torch.from_numpy(timestep_batch).to(policy_cfg.device).unsqueeze(-1).long()
    data_list = [mask_batch, target_reward, target_value, target_policy, weights]
    mask_batch, target_reward, target_value, target_policy, weights = to_torch_float_tensor(
        data_list, policy_cfg.device
    )
    target_reward = target_reward.view(policy_cfg.batch_size, -1)
    target_value = target_value.view(policy_cfg.batch_size, -1)

    # Transform rewards and values
    transformed_target_reward = scalar_transform(target_reward)
    transformed_target_value = scalar_transform(target_value)

    # Convert to categorical distributions
    target_reward_categorical = phi_transform(reward_support, transformed_target_reward)
    target_value_categorical = phi_transform(value_support, transformed_target_value)

    # Prepare batch_for_unizero
    batch_for_unizero = {}
    if isinstance(policy_cfg.model.observation_shape, int) or len(policy_cfg.model.observation_shape) == 1:
        batch_for_unizero['observations'] = torch.cat((obs_batch, obs_target_batch), dim=1).reshape(
            policy_cfg.batch_size, -1, policy_cfg.model.observation_shape)
    elif len(policy_cfg.model.observation_shape) == 3:
        batch_for_unizero['observations'] = torch.cat((obs_batch, obs_target_batch), dim=1).reshape(
            policy_cfg.batch_size, -1, *policy_cfg.model.observation_shape)

    batch_for_unizero['actions'] = action_batch.squeeze(-1)
    batch_for_unizero['timestep'] = timestep_batch.squeeze(-1)
    batch_for_unizero['rewards'] = target_reward_categorical[:, :-1]
    batch_for_unizero['mask_padding'] = mask_batch == 1.0
    batch_for_unizero['mask_padding'] = batch_for_unizero['mask_padding'][:, :-1]
    batch_for_unizero['observations'] = batch_for_unizero['observations'][:, :-1]
    batch_for_unizero['ends'] = torch.zeros(batch_for_unizero['mask_padding'].shape, dtype=torch.long, device=policy_cfg.device)
    batch_for_unizero['target_value'] = target_value_categorical[:, :-1]
    batch_for_unizero['target_policy'] = target_policy[:, :-1]

    return batch_for_unizero


def create_unizero_loss_metrics(policy):
    """
    Overview:
        Create a metrics function for computing UniZero losses without gradient updates.
        This is used for loss landscape visualization where we need to compute losses
        at different parameter values without actually updating the model.

    Arguments:
        - policy: The policy instance containing model, configuration, and all necessary attributes

    Returns:
        - compute_metrics (:obj:`Callable`): Function that computes losses for a batch of data
    """
    from ditk import logging

    # Get reward_support and value_support from policy
    reward_support = policy.reward_support
    value_support = policy.value_support

    def compute_metrics(net, dataloader, use_cuda):
        """
        Compute losses for loss landscape visualization.

        Arguments:
            - net: The neural network model
            - dataloader: DataLoader providing batches of data
            - use_cuda: Whether to use CUDA

        Returns:
            - dict: Dictionary containing averaged losses (policy_loss, value_loss, reward_loss, total_loss)
        """
        net.eval()
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_reward_loss = 0.0
        total_batches = 0

        with torch.no_grad():
            for batch_data in dataloader:
                try:
                    # Convert replay buffer sample to batch_for_unizero format
                    batch_for_unizero = convert_to_batch_for_unizero(
                        batch_data,
                        policy._cfg,
                        reward_support,
                        value_support
                    )

                    # Call world_model.compute_loss (no backward, no optimizer.step)
                    losses = net.world_model.compute_loss(
                        batch_for_unizero,
                        policy._target_model.world_model.tokenizer,
                        policy.value_inverse_scalar_transform_handle
                    )

                    # Extract individual losses from intermediate_losses
                    total_policy_loss += losses.intermediate_losses['loss_policy'].item()
                    total_value_loss += losses.intermediate_losses['loss_value'].item()
                    total_reward_loss += losses.intermediate_losses['loss_rewards'].item()
                    total_batches += 1
                except Exception as e:
                    logging.warning(f"Error processing batch in compute_metrics: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

        if total_batches > 0:
            return {
                'policy_loss': total_policy_loss / total_batches,
                'value_loss': total_value_loss / total_batches,
                'reward_loss': total_reward_loss / total_batches,
                'total_loss': (total_policy_loss + total_value_loss + total_reward_loss) / total_batches
            }
        else:
            return {'policy_loss': 0.0, 'value_loss': 0.0, 'reward_loss': 0.0, 'total_loss': 0.0}

    return compute_metrics


class UniZeroDataLoader:
    """
    Overview:
        DataLoader wrapper for UniZero replay buffer sampling.
        This provides an iterator interface for sampling batches from the replay buffer,
        compatible with loss landscape visualization tools.

    Arguments:
        - replay_buffer: The game buffer containing collected episodes
        - policy: The policy instance for sampling
        - batch_size (:obj:`int`): Number of samples per batch
        - num_batches (:obj:`int`): Total number of batches to sample
    """
    def __init__(self, replay_buffer, policy, batch_size, num_batches):
        self.buffer = replay_buffer
        self.policy = policy
        self.batch_size = batch_size
        self.num_batches = num_batches

    def __iter__(self):
        """Iterator that yields batches from the replay buffer"""
        for _ in range(self.num_batches):
            batch = self.buffer.sample(self.batch_size, self.policy)
            yield batch

    def __len__(self):
        """Return the total number of batches"""
        return self.num_batches

# ==============================================================================
# Logging Utilities
# ==============================================================================

def log_module_trainable_status(
    module: nn.Module,
    module_name: str,
    logger: logging.Logger
) -> None:
    """
    Overview:
        Logs the detailed trainable/frozen status of all parameters within a given module.

    Arguments:
        - module (:obj:`nn.Module`): The module to inspect (e.g., a ViT Encoder).
        - module_name (:obj:`str`): The name of the module for logging purposes.
        - logger (:obj:`logging.Logger`): The logger instance to use for output.
    """
    logger.info(f"--- Parameter Status Details for Module: '{module_name}' ---")

    total_params = 0
    trainable_params = 0

    param_list = list(module.named_parameters())
    if not param_list:
        logger.info("  - No parameters found in this module.")
        return

    for name, param in param_list:
        total_params += param.numel()
        status = "Trainable" if param.requires_grad else "Frozen"
        logger.info(f"  - {name:<60} | Shape: {str(param.shape):<25} | Status: {status}")
        if param.requires_grad:
            trainable_params += param.numel()

    logger.info(f"--- Summary for Module: '{module_name}' ---")
    logger.info(f"  - Total Parameters: {total_params:,}")
    logger.info(f"  - Trainable Parameters: {trainable_params:,}")
    if total_params > 0:
        percentage = 100 * trainable_params / total_params
        logger.info(f"  - Trainable Percentage: {percentage:.4f}%")
    logger.info("-" * (len(module_name) + 40))


def log_param_statistics(model: nn.Module, logger: logging.Logger) -> None:
    """
    Overview:
        Logs a concise summary of the number and size of trainable versus total
        parameters in a model.

    Arguments:
        - model (:obj:`nn.Module`): The model to analyze.
        - logger (:obj:`logging.Logger`): The logger instance for output.
    """
    n_tensors_total = sum(1 for _ in model.parameters())
    n_tensors_train = sum(1 for p in model.parameters() if p.requires_grad)

    n_elems_total = sum(p.numel() for p in model.parameters())
    n_elems_train = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(
        f'Trainable Parameters: '
        f'{n_tensors_train}/{n_tensors_total} tensors | '
        f'{n_elems_train:,}/{n_elems_total:,} elements '
        f'({n_elems_train/1e6:.2f}M / {n_elems_total/1e6:.2f}M)'
    )


def log_buffer_memory_usage(
    train_iter: int,
    buffer: GameBuffer,
    writer: SummaryWriter,
    task_id: int = 0
) -> None:
    """
    Overview:
        Logs the memory usage of the replay buffer and the current process to TensorBoard.

    Arguments:
        - train_iter (:obj:`int`): The current training iteration.
        - buffer (:obj:`GameBuffer`): The replay buffer instance.
        - writer (:obj:`SummaryWriter`): The TensorBoard writer.
        - task_id (:obj:`int`): An optional ID to distinguish logs for different tasks.
    """
    # In DDP, only the main process should write to TensorBoard.
    if writer is None:
        return

    prefix = f"Buffer/Task_{task_id}"
    writer.add_scalar(f'{prefix}/num_collected_episodes', buffer.num_of_collected_episodes, train_iter)
    writer.add_scalar(f'{prefix}/num_game_segments', len(buffer.game_segment_buffer), train_iter)
    writer.add_scalar(f'{prefix}/num_transitions', len(buffer.game_segment_game_pos_look_up), train_iter)

    # Calculate and log memory usage of the main buffer component.
    buffer_memory_bytes = asizeof(buffer.game_segment_buffer)
    buffer_memory_mb = buffer_memory_bytes / (1024 * 1024)
    writer.add_scalar(f'{prefix}/memory_usage_mb/game_segment_buffer', buffer_memory_mb, train_iter)

    # Get and log total memory usage of the current process.
    process = psutil.Process(os.getpid())
    process_memory_bytes = process.memory_info().rss
    process_memory_mb = process_memory_bytes / (1024 * 1024)
    writer.add_scalar(f'{prefix}/memory_usage_mb/process', process_memory_mb, train_iter)


def log_buffer_run_time(train_iter: int, buffer: GameBuffer, writer: SummaryWriter) -> None:
    """
    Overview:
        Logs average runtime metrics related to buffer operations (e.g., sampling, search)
        to TensorBoard.

    Arguments:
        - train_iter (:obj:`int`): The current training iteration.
        - buffer (:obj:`GameBuffer`): The buffer instance containing runtime metrics.
        - writer (:obj:`SummaryWriter`): The TensorBoard writer.
    """
    if writer is None or buffer.sample_times == 0:
        return

    sample_times = buffer.sample_times
    writer.add_scalar('Buffer/avg_reanalyze_time_ms', (buffer.compute_target_re_time / sample_times) * 1000, train_iter)
    writer.add_scalar('Buffer/avg_origin_search_time_ms', (buffer.origin_search_time / sample_times) * 1000, train_iter)
    writer.add_scalar('Buffer/avg_reuse_search_time_ms', (buffer.reuse_search_time / sample_times) * 1000, train_iter)
    writer.add_scalar('Buffer/avg_active_root_num', buffer.active_root_num / sample_times, train_iter)

    # Reset metrics after logging to prepare for the next interval.
    buffer.reset_runtime_metrics()


# ==============================================================================
# Example Usage
# ==============================================================================
if __name__ == '__main__':
    # Configure a basic logger to see output from functions with `verbose=True`
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    print("\n--- Example for `compute_task_weights` ---")
    task_rewards_list = [
        {"task1": 10, "task2": 100, "task3": 1000, "task4": 500, "task5": 300},
        {"task1": 1, "task2": 10, "task3": 100, "task4": 1000, "task5": 10000},
        {"task1": 0.1, "task2": 0.5, "task3": 0.9, "task4": 5, "task5": 10},
    ]

    for i, task_rewards in enumerate(task_rewards_list, start=1):
        print(f"\n--- Case {i} ---")
        print(f"Original Rewards: {task_rewards}")

        # Example 1: Using 'none' normalization (proportional to raw values)
        weights_none = compute_task_weights(task_rewards, option="none", use_softmax=False)
        print(f"Weights (proportional to raw values): {weights_none}")

        # Example 2: Using 'symlog' normalization
        weights_symlog = compute_task_weights(task_rewards, option="symlog", use_softmax=False)
        print(f"Weights (with symlog normalization): {weights_symlog}")

        # Example 3: Using 'rank' normalization and softmax with inverse proportion
        weights_rank_softmax = compute_task_weights(task_rewards, option="rank", use_softmax=True, reverse=True)
        print(f"Weights (inverse rank with softmax): {weights_rank_softmax}")

    print("\n--- Example for `freeze_non_lora` ---")

    # ==========================================================================
    # FIX: The nn.Parameter must be wrapped in an nn.Module subclass to be
    #      placed inside an nn.ModuleDict.
    # ==========================================================================
    class AdapterScale(nn.Module):
        """A simple nn.Module wrapper for a single learnable parameter."""
        def __init__(self):
            super().__init__()
            self.logit = nn.Parameter(torch.randn(1))

    # Create a dummy model to demonstrate freezing
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Linear(10, 10)
            self.layer1 = nn.Linear(10, 10)
            # Simulate LoRA parameters with correct naming
            self.layer1.lora_A = nn.Parameter(torch.randn(10, 2))
            self.layer1.lora_B = nn.Parameter(torch.randn(2, 10))
            
            # Correctly structure the adapter_scales using the wrapper module.
            # This ensures that the value associated with key '0' is a valid nn.Module.
            self.adapter_scales = nn.ModuleDict({
                '0': AdapterScale()
            })

    model = DummyModel()
    print("Initial parameter status:")
    log_module_trainable_status(model, "DummyModel", logging.getLogger())

    print("\nFreezing non-LoRA parameters...")
    freeze_non_lora(model, freeze=True, verbose=True)
    print("\nParameter status after freezing:")
    log_module_trainable_status(model, "DummyModel", logging.getLogger())

    print("\nUn-freezing non-LoRA parameters...")
    freeze_non_lora(model, freeze=False, verbose=True)
    print("\nParameter status after un-freezing:")
    log_module_trainable_status(model, "DummyModel", logging.getLogger())
    
_GLOBAL_HEATMAP_FIG = None
_GLOBAL_HEATMAP_AX = None


def merge_expert_stats_across_ranks(all_expert_stats):
    """
    Overview:
        Merge expert selection statistics data from all distributed training ranks.
        Combines statistics from different GPU processes for comprehensive analysis.

    Arguments:
        - all_expert_stats (:obj:`list`): List of expert statistics from all ranks.
            Each element is a dict: {task_id: {window_type: {frequencies, total_selections, data_points}}}.

    Returns:
        - merged_stats (:obj:`dict`): Merged statistics dictionary with structure
            {task_id: {window_type: {frequencies, total_selections, data_points}}}.
            Frequencies are converted to numpy arrays for serialization.

    Notes:
        Only processes statistics with total_selections > 0.

    Examples:
        >>> stats_list = [rank0_stats, rank1_stats, rank2_stats]
        >>> merged = merge_expert_stats_across_ranks(stats_list)
        >>> print(f"Merged {len(merged)} tasks")
    """
    merged_stats = {}  # {task_id: {window_type: stats}}

    for rank_expert_stats in all_expert_stats:
        if rank_expert_stats:
            for task_id, task_stats in rank_expert_stats.items():
                if task_id not in merged_stats:
                    merged_stats[task_id] = {}

                for window_type, stats in task_stats.items():
                    # Only process statistics with actual data (tasks handled by current GPU)
                    if stats and stats.get('total_selections', 0) > 0:
                        merged_stats[task_id][window_type] = {
                            'frequencies': np.array(stats['frequencies']),
                            'total_selections': stats['total_selections'],
                            'data_points': stats['data_points']
                        }
    return merged_stats


def _get_or_create_heatmap_figure(figsize):
    """
    Overview:
        Get or create a reusable heatmap figure for memory efficiency.
        Maintains global figure cache to reduce memory allocation overhead.
    Arguments:
        - figsize (:obj:`tuple`): Figure size as (width, height).
    Returns:
        - fig (:obj:`matplotlib.figure.Figure`): Matplotlib figure object.
        - ax (:obj:`matplotlib.axes.Axes`): Matplotlib axes object.
    Examples:
        >>> fig, ax = _get_or_create_heatmap_figure((10, 8))
        >>> ax.plot([1, 2, 3], [4, 5, 6])
    """
    global _GLOBAL_HEATMAP_FIG, _GLOBAL_HEATMAP_AX
    if _GLOBAL_HEATMAP_FIG is None:
        _GLOBAL_HEATMAP_FIG, _GLOBAL_HEATMAP_AX = plt.subplots(figsize=figsize)
    else:
        # Clear previous content
        _GLOBAL_HEATMAP_AX.clear()
        # Adjust image size
        _GLOBAL_HEATMAP_FIG.set_size_inches(figsize)
    return _GLOBAL_HEATMAP_FIG, _GLOBAL_HEATMAP_AX


def create_heatmap_with_values_fast(matrix, task_ids, title="Task-Expert Selection Frequencies"):
    """
    Overview:
        Efficiently create annotated blue-themed heatmap with performance optimizations.
        Optimizations include matplotlib figure reuse, selective value annotations,
        optimized image conversion pipeline, and reduced DPI for faster computation.
    Arguments:
        - matrix (:obj:`numpy.ndarray`): Input matrix for heatmap visualization.
        - task_ids (:obj:`list`): List of task identifiers for y-axis labels.
        - title (:obj:`str`, optional): Heatmap title. Default is "Task-Expert Selection Frequencies".
    Returns:
        - img_array (:obj:`numpy.ndarray`): Image array in CHW format for TensorBoard logging.
    Shapes:
        - matrix: :math:`(N_{tasks}, N_{experts})` where N_tasks and N_experts are dimensions.
        - img_array: :math:`(3, H, W)` where H and W are image height and width.
    Examples:
        >>> import numpy as np
        >>> matrix = np.random.rand(5, 8)
        >>> task_ids = [0, 1, 2, 3, 4]
        >>> heatmap = create_heatmap_with_values_fast(matrix, task_ids)
        >>> print(f"Heatmap shape: {heatmap.shape}")  # (3, height, width)
    """
    try:
        figsize = (max(6, matrix.shape[1]), max(4, matrix.shape[0]))
        fig, ax = _get_or_create_heatmap_figure(figsize)

        # Intelligently choose whether to display value annotations
        show_annot = matrix.size <= 64  # Only display values for 8x8 or smaller matrices

        # Use matplotlib directly to avoid seaborn overhead
        im = ax.imshow(matrix, cmap='Blues', aspect='auto')

        # Selectively add value annotations
        if show_annot:
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    value = matrix[i, j]
                    color = 'white' if value > 0.5 else 'black'
                    ax.text(j, i, f'{value:.3f}', ha='center', va='center',
                           color=color, fontsize=8)

        # Set labels and title
        ax.set_xticks(range(matrix.shape[1]))
        ax.set_yticks(range(matrix.shape[0]))
        ax.set_xticklabels([f'E{i}' for i in range(matrix.shape[1])], fontsize=10)
        ax.set_yticklabels([f'T{tid}' for tid in task_ids], fontsize=10)
        ax.set_title(title, fontsize=12, pad=15)
        ax.set_xlabel('Experts', fontsize=10)
        ax.set_ylabel('Tasks', fontsize=10)

        # Simplified colorbar
        if not hasattr(fig, '_colorbar_created'):
            plt.colorbar(im, ax=ax, label='Frequency')
            fig._colorbar_created = True

        # Optimized image conversion: using lower DPI and simplified pipeline
        fig.canvas.draw()
        try:
            # Get RGB data directly from canvas
            if hasattr(fig.canvas, 'buffer_rgba'):
                buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
                buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
                img_array = buf[:, :, :3]  # Remove alpha channel
            else:
                buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                img_array = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            # Convert to CHW format
            img_array = img_array.transpose(2, 0, 1)

        except Exception:
            # Fallback: create simple blue gradient matrix
            h, w = matrix.shape
            img_array = np.zeros((3, h*20, w*20), dtype=np.uint8)
            # Simple matrix upscaling and mapping to blue channel
            matrix_resized = np.repeat(np.repeat(matrix, 20, axis=0), 20, axis=1)
            img_array[2] = (matrix_resized * 255).astype(np.uint8)

        return img_array

    except Exception as e:
        print(f"Warning: Heatmap generation failed: {e}, using fallback")
        # Ultimate fallback: return blank image
        return np.zeros((3, 100, 100), dtype=np.uint8)


def create_heatmap_with_values(matrix, task_ids, title="Task-Expert Selection Frequencies"):
    """
    Overview:
        Create annotated blue-themed heatmap using seaborn - original version for fallback.
        This function serves as a backup when the optimized version encounters issues.
    Arguments:
        - matrix (:obj:`numpy.ndarray`): Input matrix for heatmap visualization.
        - task_ids (:obj:`list`): List of task identifiers for y-axis labels.
        - title (:obj:`str`, optional): Heatmap title. Default is "Task-Expert Selection Frequencies".
    Returns:
        - img_array (:obj:`numpy.ndarray`): Image array in CHW format for TensorBoard logging.
    Shapes:
        - matrix: :math:`(N_{tasks}, N_{experts})` where N_tasks and N_experts are dimensions.
        - img_array: :math:`(3, H, W)` where H and W are image height and width.
    Examples:
        >>> import numpy as np
        >>> matrix = np.random.rand(5, 8)
        >>> task_ids = [0, 1, 2, 3, 4]
        >>> heatmap = create_heatmap_with_values(matrix, task_ids)
        >>> print(f"Heatmap shape: {heatmap.shape}")  # (3, height, width)
    """
    fig, ax = plt.subplots(figsize=(max(8, matrix.shape[1]), max(6, matrix.shape[0])))

    # Use blue color scheme
    sns.heatmap(matrix,
                annot=True,  # Display values
                fmt='.3f',   # Value format
                cmap='Blues',  # Blue theme
                ax=ax,
                cbar_kws={'label': 'Selection Frequency'},
                xticklabels=[f'Expert{i}' for i in range(matrix.shape[1])],
                yticklabels=[f'Task{tid}' for tid in task_ids])

    ax.set_title(title, fontsize=14, pad=20)
    ax.set_xlabel('Experts', fontsize=12)
    ax.set_ylabel('Tasks', fontsize=12)

    plt.tight_layout()

    # Save to BytesIO
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)

    # Convert to numpy array for tensorboard
    img = Image.open(buf)
    img_array = np.array(img)
    buf.close()
    plt.close(fig)

    # Convert to CHW format (Channel, Height, Width)
    if len(img_array.shape) == 3:
        img_array = img_array.transpose(2, 0, 1)

    return img_array


def log_expert_selection_details(tb_logger, merged_stats, valid_task_ids, matrix, window_type, train_iter):
    """
    Overview:
        Log detailed expert selection statistics for each task.
        Records frequency entropy, variance, and total selections for analysis.
    Arguments:
        - tb_logger (:obj:`SummaryWriter`): TensorBoard logger for metric recording.
        - merged_stats (:obj:`dict`): Merged expert selection statistics across ranks.
        - valid_task_ids (:obj:`list`): List of valid task identifiers.
        - matrix (:obj:`numpy.ndarray`): Expert selection frequency matrix.
        - window_type (:obj:`str`): Time window type (immediate, short, medium, long).
        - train_iter (:obj:`int`): Current training iteration for logging.
    Examples:
        >>> log_expert_selection_details(tb_logger, stats, [0,1,2], matrix, 'immediate', 1000)
    """
    for i, task_id in enumerate(valid_task_ids):
        frequencies = matrix[i]
        stats = merged_stats[task_id][window_type]

        # Calculate and record task expert selection entropy (uniformity metric)
        task_frequencies = np.array(frequencies)
        task_frequencies = task_frequencies + 1e-8  # Avoid log(0)
        task_entropy = -np.sum(task_frequencies * np.log(task_frequencies))
        tb_logger.add_scalar(
            f'MOE_Details/Task{task_id}_{window_type}/ExpertSelectionEntropy',
            task_entropy, global_step=train_iter
        )

        # Record task expert selection variance (dispersion)
        expert_variance = np.var(task_frequencies)
        tb_logger.add_scalar(
            f'MOE_Details/Task{task_id}_{window_type}/ExpertSelectionVariance',
            expert_variance, global_step=train_iter
        )

        # Record task-level summary statistics
        tb_logger.add_scalar(
            f'MOE_Details/Task{task_id}_{window_type}/TotalSelections',
            stats['total_selections'], global_step=train_iter
        )
        tb_logger.add_scalar(
            f'MOE_Details/Task{task_id}_{window_type}/DataPoints',
            stats['data_points'], global_step=train_iter
        )


def log_global_moe_statistics(tb_logger, matrix, window_type, valid_task_ids, train_iter):
    """
    Overview:
        Log global MOE statistics including expert usage uniformity and extremes.
        Provides system-wide view of expert utilization patterns.
    Arguments:
        - tb_logger (:obj:`SummaryWriter`): TensorBoard logger for metric recording.
        - matrix (:obj:`numpy.ndarray`): Expert selection frequency matrix.
        - window_type (:obj:`str`): Time window type (immediate, short, medium, long).
        - valid_task_ids (:obj:`list`): List of valid task identifiers.
        - train_iter (:obj:`int`): Current training iteration for logging.
    Examples:
        >>> log_global_moe_statistics(tb_logger, matrix, 'immediate', [0,1,2], 1000)
    """
    # Record basic information
    tb_logger.add_scalar(
        f'MOE_Global/{window_type}/NumActiveTasks',
        len(valid_task_ids), global_step=train_iter
    )
    tb_logger.add_scalar(
        f'MOE_Global/{window_type}/NumExperts',
        matrix.shape[1], global_step=train_iter
    )

    # Calculate expert usage uniformity
    expert_avg_usage = np.mean(matrix, axis=0)  # Average usage frequency per expert
    usage_entropy = -np.sum(expert_avg_usage * np.log(expert_avg_usage + 1e-8))
    tb_logger.add_scalar(
        f'MOE_Global/{window_type}/ExpertUsageEntropy',
        usage_entropy, global_step=train_iter
    )

    # Record most and least used experts
    most_used_expert = np.argmax(expert_avg_usage)
    least_used_expert = np.argmin(expert_avg_usage)
    tb_logger.add_scalar(
        f'MOE_Global/{window_type}/MostUsedExpert',
        most_used_expert, global_step=train_iter
    )
    tb_logger.add_scalar(
        f'MOE_Global/{window_type}/LeastUsedExpert',
        least_used_expert, global_step=train_iter
    )


def process_and_log_moe_heatmaps_fast(tb_logger, merged_stats, window_type, train_iter):
    """
    Overview:
        Efficiently process and log MOE heatmaps with performance optimizations.
        Includes vectorized data processing, conditional heatmap generation,
        and batch statistical processing.
    Arguments:
        - tb_logger (:obj:`SummaryWriter`): TensorBoard logger for metric recording.
        - merged_stats (:obj:`dict`): Merged expert selection statistics across ranks.
        - window_type (:obj:`str`): Time window type (immediate, short, medium, long).
        - train_iter (:obj:`int`): Current training iteration for logging.
    Examples:
        >>> process_and_log_moe_heatmaps_fast(tb_logger, stats, 'immediate', 1000)
    """
    # Quick filtering of valid tasks
    valid_task_data = [(tid, stats[window_type]['frequencies'])
                      for tid, stats in merged_stats.items()
                      if window_type in stats]

    if not valid_task_data:
        return

    # Vectorized matrix construction
    valid_task_ids, frequencies_list = zip(*valid_task_data)
    matrix = np.array(frequencies_list)

    # Conditional heatmap generation: only for small matrices
    if matrix.size <= 200:  # Only generate heatmap when tasks*experts <= 200
        try:
            heatmap_img = create_heatmap_with_values_fast(
                matrix, valid_task_ids,
                f'MOE {window_type} Task-Expert Selection'
            )

            # Log heatmap to tensorboard
            tb_logger.add_image(
                f'MOE_Heatmap/{window_type}_TaskExpert_Heatmap',
                heatmap_img,
                global_step=train_iter,
                dataformats='CHW'
            )
        except Exception as e:
            print(f"Warning: Heatmap generation failed: {e}")

    # Always log statistical data (lightweight operation)
    log_expert_selection_details(tb_logger, merged_stats, valid_task_ids, matrix, window_type, train_iter)
    log_global_moe_statistics(tb_logger, matrix, window_type, valid_task_ids, train_iter)


def process_and_log_moe_heatmaps(tb_logger, merged_stats, window_type, train_iter):
    """
    Overview:
        Process and log MOE heatmaps - original version for fallback.
        This function serves as a backup when the optimized version encounters issues.
    Arguments:
        - tb_logger (:obj:`SummaryWriter`): TensorBoard logger for metric recording.
        - merged_stats (:obj:`dict`): Merged expert selection statistics across ranks.
        - window_type (:obj:`str`): Time window type (immediate, short, medium, long).
        - train_iter (:obj:`int`): Current training iteration for logging.
    Examples:
        >>> process_and_log_moe_heatmaps(tb_logger, stats, 'immediate', 1000)
    """
    all_task_ids = sorted(merged_stats.keys())
    task_expert_matrix = []
    valid_task_ids = []

    # Collect frequency data from valid tasks
    for task_id in all_task_ids:
        if window_type in merged_stats[task_id]:
            frequencies = merged_stats[task_id][window_type]['frequencies']
            task_expert_matrix.append(frequencies)
            valid_task_ids.append(task_id)

    if not task_expert_matrix:
        return

    # Convert to numpy matrix (num_tasks, num_experts)
    matrix = np.array(task_expert_matrix)

    # Create annotated blue-themed heatmap
    heatmap_img = create_heatmap_with_values(
        matrix, valid_task_ids,
        f'MOE {window_type} Task-Expert Selection Frequencies'
    )

    # Log heatmap to tensorboard
    tb_logger.add_image(
        f'MOE_Heatmap/{window_type}_TaskExpert_Heatmap',
        heatmap_img,
        global_step=train_iter,
        dataformats='CHW'
    )

    # Log detailed and global statistics
    log_expert_selection_details(tb_logger, merged_stats, valid_task_ids, matrix, window_type, train_iter)


def convert_stats_to_serializable(moe_stats):
    """
    Overview:
        Convert tensor data in MOE statistics to serializable numpy format.
        Ensures compatibility with distributed communication protocols.
    Arguments:
        - moe_stats (:obj:`dict`): MOE statistics containing tensor data.
    Returns:
        - converted (:obj:`dict`): Converted statistics with numpy arrays.
    Examples:
        >>> tensor_stats = {'task_0': {'immediate': {'frequencies': torch.tensor([0.1, 0.9])}}}
        >>> numpy_stats = convert_stats_to_serializable(tensor_stats)
        >>> type(numpy_stats['task_0']['immediate']['frequencies'])  # <class 'list'>
    """
    if not moe_stats:
        return {}

    converted = {}
    for task_id, task_stats in moe_stats.items():
        converted[task_id] = {}
        for window_type, stats in task_stats.items():
            if stats and 'frequencies' in stats:
                converted[task_id][window_type] = {
                    'frequencies': stats['frequencies'].cpu().numpy().tolist(),
                    'total_selections': stats['total_selections'],
                    'data_points': stats['data_points']
                }
    return converted


def gather_distributed_moe_stats(local_stats, world_size):
    """
    Overview:
        Gather MOE statistics from all GPUs in distributed training environment.
        Handles communication failures gracefully with fallback to local statistics.
    Arguments:
        - local_stats (:obj:`dict`): Local GPU's MOE statistics.
        - world_size (:obj:`int`): Total number of distributed training processes.
    Returns:
        - all_stats (:obj:`list`): List of statistics from all ranks.
    Examples:
        >>> local_data = {'task_0': {'immediate': {'frequencies': [0.1, 0.9]}}}
        >>> all_data = gather_distributed_moe_stats(local_data, 4)
        >>> len(all_data)  # 4 (from 4 GPUs)
    """
    all_stats = [None for _ in range(world_size)]
    try:
        dist.all_gather_object(all_stats, local_stats)
        return all_stats
    except Exception as e:
        print(f"Distributed MOE statistics gathering failed: {e}")
        return [local_stats]  # fallback to local statistics


def collect_and_log_moe_statistics(policy, tb_logger, train_iter, world_size, rank):
    """
    Overview:
        Collect and log MoE expert selection statistics including heatmaps and distribution analysis.
        Handles distributed data collection, merging, and TensorBoard visualization.

    Arguments:
        - policy (:obj:`Policy`): Training policy with world_model.transformer supporting get_expert_selection_stats.
        - tb_logger (:obj:`SummaryWriter`): TensorBoard logger for metric recording.
        - train_iter (:obj:`int`): Current training iteration number.
        - world_size (:obj:`int`): Total number of GPUs in distributed training.
        - rank (:obj:`int`): Current GPU rank identifier.

    Returns:
        - None: No return value; performs logging only.

    Notes:
        Logs heatmaps for immediate/short/medium/long windows and JS/Wasserstein divergence between task distributions.

    Examples:
        >>> collect_and_log_moe_statistics(policy, tb_logger, 1000, 8, 0)
    """
    try:
        # Step 1: Get MOE statistics from policy's transformer model
        moe_stats = None

        transformer = policy._model.world_model.transformer
        if hasattr(transformer, 'get_expert_selection_stats'):
            moe_stats = transformer.get_expert_selection_stats()

        if moe_stats is None:
            print(f"Rank {rank}: Warning: Unable to get MOE statistics, train_iter={train_iter}")
            return

        # Step 2: Convert tensor data to serializable format
        serializable_stats = convert_stats_to_serializable(moe_stats)

        print(f"Rank {rank}: Local MOE statistics - tasks: {len(serializable_stats)}, train_iter={train_iter}")

        # Step 3: Gather statistics from all GPUs in distributed setting
        all_expert_stats = gather_distributed_moe_stats(serializable_stats, world_size)

        # Step 4: Merge statistics data
        merged_stats = merge_expert_stats_across_ranks(all_expert_stats)

        if not merged_stats:
            print(f"Rank {rank}: Warning: Merged MOE statistics empty, train_iter={train_iter}")
            return

        # Step 5: All GPUs log MOE statistics
        print(f"Rank {rank}: Starting MOE statistics logging - merged tasks: {len(merged_stats)}, train_iter={train_iter}")

        # Generate heatmaps and statistics for each time window
        for window_type in ['immediate', 'short', 'medium', 'long']:
            if any(window_type in task_stats for task_stats in merged_stats.values()):
                process_and_log_moe_heatmaps_fast(tb_logger, merged_stats, window_type, train_iter)

        # Log overall MOE usage
        tb_logger.add_scalar('MOE_Global/ActiveTasks', len(merged_stats), global_step=train_iter)

        # Step 6: Add distribution difference computation and logging
        if any('immediate' in task_stats for task_stats in merged_stats.values()):
            print(f"Rank {rank}: Starting inter-task distribution difference calculation...")
            collect_and_log_divergences_with_heatmaps(tb_logger, merged_stats, train_iter)

        print(f"Rank {rank}: MOE statistics logging completed, train_iter={train_iter}")

    except Exception as e:
        print(f"Rank {rank}: MOE statistics collection failed - {e}, train_iter={train_iter}")
        import traceback
        traceback.print_exc()


# ====== GPU-Optimized Distribution Divergence Calculation and Visualization Functions ======
def jensen_shannon_divergence_batch_gpu(distributions_tensor):
    """
    Overview:
        GPU batch computation of JS divergence matrix - fully vectorized, no loops.
        Efficiently computes Jensen-Shannon divergence between all pairs of distributions.

    Arguments:
        - distributions_tensor (:obj:`torch.Tensor`): Shape (n_tasks, n_experts), GPU tensor.

    Returns:
        - js_matrix (:obj:`torch.Tensor`): Shape (n_tasks, n_tasks), symmetric matrix.

    Shapes:
        - distributions_tensor: :math:`(N_{tasks}, N_{experts})`
        - js_matrix: :math:`(N_{tasks}, N_{tasks})`

    Notes:
        Input is normalized to a probability distribution before computation.
    Examples:
        >>> dist_tensor = torch.rand(5, 8).cuda()
        >>> js_matrix = jensen_shannon_divergence_batch_gpu(dist_tensor)
        >>> print(js_matrix.shape)  # torch.Size([5, 5])
    """
    device = distributions_tensor.device
    n_tasks, n_experts = distributions_tensor.shape

    # 1. Normalize to probability distributions
    eps = 1e-8
    distributions_tensor = distributions_tensor / (distributions_tensor.sum(dim=1, keepdim=True) + eps)

    # 2. Use broadcasting to compute average distributions for all task pairs
    # P_i: (n_tasks, 1, n_experts), P_j: (1, n_tasks, n_experts)
    P_i = distributions_tensor.unsqueeze(1)
    P_j = distributions_tensor.unsqueeze(0)
    M = 0.5 * (P_i + P_j)  # shape: (n_tasks, n_tasks, n_experts)

    # 3. Batch compute KL divergences - fully vectorized
    # KL(P_i || M) for all pairs
    log_ratio_i = torch.log((P_i + eps) / (M + eps))
    kl_i_m = torch.sum(P_i * log_ratio_i, dim=2)  # (n_tasks, n_tasks)

    # KL(P_j || M) for all pairs
    log_ratio_j = torch.log((P_j + eps) / (M + eps))
    kl_j_m = torch.sum(P_j * log_ratio_j, dim=2)  # (n_tasks, n_tasks)

    # 4. JS divergence matrix
    js_matrix = 0.5 * (kl_i_m + kl_j_m)

    return js_matrix


def wasserstein_distance_batch_gpu(distributions_tensor):
    """
    Overview:
        GPU batch computation of Wasserstein distance matrix - efficient 1D distribution implementation.
        Computes Earth Mover's Distance between all pairs of discrete distributions.

    Arguments:
        - distributions_tensor (:obj:`torch.Tensor`): Shape (n_tasks, n_experts), GPU tensor.

    Returns:
        - wasserstein_matrix (:obj:`torch.Tensor`): Shape (n_tasks, n_tasks), symmetric matrix.

    Shapes:
        - distributions_tensor: :math:`(N_{tasks}, N_{experts})`
        - wasserstein_matrix: :math:`(N_{tasks}, N_{tasks})`

    Notes:
        Uses L1 norm of CDF differences for discrete distributions.
    Examples:
        >>> dist_tensor = torch.rand(5, 8).cuda()
        >>> wass_matrix = wasserstein_distance_batch_gpu(dist_tensor)
        >>> print(wass_matrix.shape)  # torch.Size([5, 5])
    """
    device = distributions_tensor.device
    n_tasks, n_experts = distributions_tensor.shape
    eps = 1e-8

    # 1. Normalize to probability distributions
    distributions_tensor = distributions_tensor / (distributions_tensor.sum(dim=1, keepdim=True) + eps)

    # 2. Compute cumulative distribution functions (CDF)
    cdf_tensor = torch.cumsum(distributions_tensor, dim=1)  # (n_tasks, n_experts)

    # 3. Use broadcasting to compute L1 distances between all CDF pairs
    cdf_i = cdf_tensor.unsqueeze(1)  # (n_tasks, 1, n_experts)
    cdf_j = cdf_tensor.unsqueeze(0)  # (1, n_tasks, n_experts)

    # Wasserstein distance = L1 norm of cumulative distribution differences
    wasserstein_matrix = torch.sum(torch.abs(cdf_i - cdf_j), dim=2)

    return wasserstein_matrix


def compute_distribution_divergences_optimized(merged_stats, window_type='immediate'):
    """
    Overview:
        GPU-optimized version for efficient distribution divergence computation.
        Leverages GPU acceleration for batch processing of divergence metrics.

    Arguments:
        - merged_stats (:obj:`dict`): Merged MoE statistics from all distributed ranks.
        - window_type (:obj:`str`, optional): Time window type. Default is 'immediate'.

    Returns:
        - divergence_data (:obj:`dict`): Dictionary containing:
            - task_ids: List of task IDs
            - n_tasks, n_experts: Dimensions
            - device, gpu_accelerated: Device info
            - js_matrix, wasserstein_matrix: Divergence matrices (numpy)
            - js_stats, wasserstein_stats: avg, max, min, std for each metric

    Examples:
        >>> stats = {'task_0': {'immediate': {'frequencies': [0.1, 0.9]}}}
        >>> result = compute_distribution_divergences_optimized(stats)
        >>> print(f"GPU accelerated: {result['gpu_accelerated']}")
    """
    # 1. Data preprocessing
    valid_tasks = [(tid, stats[window_type]['frequencies'])
                  for tid, stats in merged_stats.items()
                  if window_type in stats]

    if len(valid_tasks) < 2:
        return {}

    task_ids, frequencies_list = zip(*valid_tasks)

    # 2. Efficient tensor conversion
    try:
        if isinstance(frequencies_list[0], torch.Tensor):
            frequencies_tensor = torch.stack(frequencies_list)
        else:
            frequencies_tensor = torch.tensor(
                np.array(frequencies_list),
                dtype=torch.float32
            )

        # Automatic GPU acceleration
        if torch.cuda.is_available():
            frequencies_tensor = frequencies_tensor.cuda()

    except Exception as e:
        print(f"GPU conversion failed, using CPU: {e}")
        frequencies_tensor = torch.tensor(np.array(frequencies_list), dtype=torch.float32)

    device = frequencies_tensor.device
    n_tasks, n_experts = frequencies_tensor.shape

    # 3. GPU batch computation (no loops)
    with torch.no_grad():
        # Batch compute JS divergence and Wasserstein distance
        js_matrix = jensen_shannon_divergence_batch_gpu(frequencies_tensor)
        wasserstein_matrix = wasserstein_distance_batch_gpu(frequencies_tensor)

        # Efficiently extract upper triangular values (avoid duplicate computation)
        triu_indices = torch.triu_indices(n_tasks, n_tasks, offset=1, device=device)
        js_values = js_matrix[triu_indices[0], triu_indices[1]]
        wasserstein_values = wasserstein_matrix[triu_indices[0], triu_indices[1]]

        # Statistical computation (vectorized)
        js_stats = {
            'avg': torch.mean(js_values).item(),
            'max': torch.max(js_values).item(),
            'min': torch.min(js_values).item(),
            'std': torch.std(js_values).item()
        }

        wasserstein_stats = {
            'avg': torch.mean(wasserstein_values).item(),
            'max': torch.max(wasserstein_values).item(),
            'min': torch.min(wasserstein_values).item(),
            'std': torch.std(wasserstein_values).item()
        }

    return {
        'task_ids': task_ids,
        'n_tasks': n_tasks,
        'n_experts': n_experts,
        'device': str(device),
        'gpu_accelerated': 'cuda' in str(device),

        # Return CPU versions for logging
        'js_matrix': js_matrix.cpu().numpy(),
        'wasserstein_matrix': wasserstein_matrix.cpu().numpy(),
        'js_stats': js_stats,
        'wasserstein_stats': wasserstein_stats
    }


def create_similarity_heatmap_no_diagonal(similarity_matrix, task_ids, metric_name, title_suffix=""):
    """
    Overview:
        Create task similarity heatmap with diagonal elements removed.
        Provides clear visualization of inter-task relationships without self-similarity noise.
    Arguments:
        - similarity_matrix (:obj:`numpy.ndarray`): Similarity matrix (n_tasks, n_tasks).
        - task_ids (:obj:`list`): Task identifier list for axis labels.
        - metric_name (:obj:`str`): Metric name ('js_divergence', 'wasserstein_distance').
        - title_suffix (:obj:`str`, optional): Additional title suffix. Default is "".
    Returns:
        - img_array (:obj:`numpy.ndarray`): Image array in CHW format for TensorBoard.
    Shapes:
        - similarity_matrix: :math:`(N_{tasks}, N_{tasks})`
        - img_array: :math:`(3, H, W)` where H and W are image dimensions.
    Examples:
        >>> matrix = np.random.rand(5, 5)
        >>> task_ids = [0, 1, 2, 3, 4]
        >>> heatmap = create_similarity_heatmap_no_diagonal(matrix, task_ids, 'js_divergence')
        >>> print(f"Output shape: {heatmap.shape}")  # (3, height, width)
    """
    try:
        # Copy matrix to avoid modifying original data
        matrix = similarity_matrix.copy()

        # Set diagonal to NaN so matplotlib displays as blank
        np.fill_diagonal(matrix, np.nan)

        figsize = (max(6, len(task_ids)), max(4, len(task_ids)))
        fig, ax = plt.subplots(figsize=figsize)  # Create new figure to avoid reuse issues

        # Choose color mapping based on metric type
        if 'js' in metric_name.lower():
            cmap = 'Reds'
            title_name = 'JS Divergence'
            vmin, vmax = 0, 1.0
        else:  # wasserstein
            cmap = 'Blues'
            title_name = 'Wasserstein Distance'
            vmin, vmax = None, None  # Adaptive

        # Use masked array to handle NaN values, diagonal displays as white
        masked_matrix = np.ma.masked_invalid(matrix)
        im = ax.imshow(masked_matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')

        # Add value annotations (skip diagonal)
        if len(task_ids) <= 15:  # Only add annotations for smaller task counts
            for i in range(len(task_ids)):
                for j in range(len(task_ids)):
                    if i != j:  # Skip diagonal
                        value = matrix[i, j]
                        if not np.isnan(value):
                            threshold = (vmax or np.nanmax(matrix)) * 0.5 if vmax else np.nanmax(matrix) * 0.5
                            color = 'white' if value > threshold else 'black'
                            ax.text(j, i, f'{value:.3f}', ha='center', va='center',
                                   color=color, fontsize=8)

        # Set labels
        ax.set_xticks(range(len(task_ids)))
        ax.set_yticks(range(len(task_ids)))
        ax.set_xticklabels([f'T{tid}' for tid in task_ids], fontsize=9)
        ax.set_yticklabels([f'T{tid}' for tid in task_ids], fontsize=9)
        ax.set_title(f'Task {title_name} Matrix {title_suffix} (No Diagonal)', fontsize=12)
        ax.set_xlabel('Tasks', fontsize=10)
        ax.set_ylabel('Tasks', fontsize=10)

        # Add colorbar
        plt.colorbar(im, ax=ax, label=title_name, shrink=0.8)

        # Convert to image array - fix matplotlib version compatibility
        fig.canvas.draw()

        try:
            # New matplotlib uses buffer_rgba
            if hasattr(fig.canvas, 'buffer_rgba'):
                buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
                h, w = fig.canvas.get_width_height()
                img_array = buf.reshape(h, w, 4)[:, :, :3]  # Remove alpha channel
            else:
                # Old matplotlib fallback
                buf = fig.canvas.print_to_string()
                img_array = np.frombuffer(buf, dtype=np.uint8)
                h, w = fig.canvas.get_width_height()
                img_array = img_array.reshape(h, w, 3)
        except Exception as conv_e:
            print(f"Image conversion method failed: {conv_e}, trying PIL approach")
            # Final fallback: convert through PIL
            buf = BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            img = Image.open(buf)
            img_array = np.array(img)[:, :, :3]  # Remove alpha channel
            buf.close()

        img_array = img_array.transpose(2, 0, 1)  # CHW format
        plt.close(fig)  # Close figure to avoid memory leak

        return img_array

    except Exception as e:
        print(f"Warning: No-diagonal heatmap generation failed: {e}")
        return np.zeros((3, 100, 100), dtype=np.uint8)


def log_pairwise_optimized(tb_logger, divergence_data, train_iter):
    """
    Overview:
        Optimized task pair logging with batch processing.
        Efficiently logs pairwise divergence metrics for all task combinations.
    Arguments:
        - tb_logger (:obj:`SummaryWriter`): TensorBoard logger for metric recording.
        - divergence_data (:obj:`dict`): Divergence computation results.
        - train_iter (:obj:`int`): Current training iteration for logging.
    Examples:
        >>> log_pairwise_optimized(tb_logger, divergence_data, 1000)
    """
    task_ids = divergence_data['task_ids']
    js_matrix = divergence_data['js_matrix']
    wasserstein_matrix = divergence_data['wasserstein_matrix']

    # Batch construct task pair metric dictionary
    pairwise_scalars = {}

    for i, task_i in enumerate(task_ids):
        for j, task_j in enumerate(task_ids):
            if i < j:  # Only log upper triangle
                # Construct metric names
                js_key = f'TaskPairwise/Immediate_Task{task_i}_Task{task_j}_JS_Divergence'
                wass_key = f'TaskPairwise/Immediate_Task{task_i}_Task{task_j}_Wasserstein_Distance'

                pairwise_scalars[js_key] = js_matrix[i, j]
                pairwise_scalars[wass_key] = wasserstein_matrix[i, j]

    # Batch write to TensorBoard
    for key, value in pairwise_scalars.items():
        tb_logger.add_scalar(key, float(value), global_step=train_iter)


def log_divergences_with_heatmaps(tb_logger, divergence_data, train_iter):
    """
    Overview:
        Log distribution divergence metrics and heatmaps (with diagonal removed).
        Comprehensive logging of inter-task distribution analysis results.
    Arguments:
        - tb_logger (:obj:`SummaryWriter`): TensorBoard logger for metric recording.
        - divergence_data (:obj:`dict`): Divergence computation results.
        - train_iter (:obj:`int`): Current training iteration for logging.
    Examples:
        >>> log_divergences_with_heatmaps(tb_logger, divergence_data, 1000)
    """
    if not divergence_data:
        return

    js_stats = divergence_data['js_stats']
    wasserstein_stats = divergence_data['wasserstein_stats']
    task_ids = divergence_data['task_ids']
    n_tasks = divergence_data['n_tasks']

    # Debug: Check matrix data
    js_matrix = divergence_data['js_matrix']
    wasserstein_matrix = divergence_data['wasserstein_matrix']
    print(f"DEBUG: JS matrix shape={js_matrix.shape}, range=[{np.min(js_matrix):.6f}, {np.max(js_matrix):.6f}]")
    print(f"DEBUG: Wasserstein matrix shape={wasserstein_matrix.shape}, range=[{np.min(wasserstein_matrix):.6f}, {np.max(wasserstein_matrix):.6f}]")

    # 1. Log scalar metrics
    scalar_dict = {
        'MOE_Divergence/Immediate_AvgJS_Divergence': js_stats['avg'],
        'MOE_Divergence/Immediate_MaxJS_Divergence': js_stats['max'],
        'MOE_Divergence/Immediate_AvgWasserstein_Distance': wasserstein_stats['avg'],
        'MOE_Divergence/Immediate_MaxWasserstein_Distance': wasserstein_stats['max'],
    }

    for key, value in scalar_dict.items():
        tb_logger.add_scalar(key, value, global_step=train_iter)

    # 1.1 Print core metrics to console
    print("=" * 65)
    print(f" Inter-Task Distribution Divergence Statistics (Iteration: {train_iter})")
    print("=" * 65)
    print(f"Participating tasks: {n_tasks} | Task IDs: {list(task_ids)}")
    print(f"Computing device: {divergence_data.get('device', 'Unknown')} | GPU acceleration: {'Enabled' if divergence_data.get('gpu_accelerated', False) else 'Disabled'}")
    print("-" * 65)
    print("JS Divergence (Jensen-Shannon Divergence):")
    print(f"  Average: {js_stats['avg']:.6f}  |  Maximum: {js_stats['max']:.6f}")
    print(f"  Minimum: {js_stats['min']:.6f}  |  Std Dev: {js_stats['std']:.6f}")
    print("-" * 65)
    print("Wasserstein Distance:")
    print(f"  Average: {wasserstein_stats['avg']:.6f}  |  Maximum: {wasserstein_stats['max']:.6f}")
    print(f"  Minimum: {wasserstein_stats['min']:.6f}  |  Std Dev: {wasserstein_stats['std']:.6f}")
    print("=" * 65)

    # 2. Log similarity matrix heatmaps with diagonal removed
    task_ids = divergence_data['task_ids']
    n_tasks = divergence_data['n_tasks']

    if n_tasks <= 25:  # Limit matrix size to avoid oversized heatmaps
        try:
            # JS divergence matrix heatmap (no diagonal)
            js_heatmap = create_similarity_heatmap_no_diagonal(
                divergence_data['js_matrix'],
                task_ids,
                'js_divergence',
                f'(Immediate-{n_tasks} tasks)'
            )
            tb_logger.add_image(
                'TaskSimilarity/Immediate_JS_Matrix_NoDiagonal',
                js_heatmap,
                global_step=train_iter,
                dataformats='CHW'
            )

            # Wasserstein distance matrix heatmap (no diagonal)
            wass_heatmap = create_similarity_heatmap_no_diagonal(
                divergence_data['wasserstein_matrix'],
                task_ids,
                'wasserstein_distance',
                f'(Immediate-{n_tasks} tasks)'
            )
            tb_logger.add_image(
                'TaskSimilarity/Immediate_Wasserstein_Matrix_NoDiagonal',
                wass_heatmap,
                global_step=train_iter,
                dataformats='CHW'
            )

        except Exception as e:
            print(f"Warning: Similarity matrix heatmap generation failed: {e}")

    # 3. Log task pair metrics (optional)
    if n_tasks <= 20:
        log_pairwise_optimized(tb_logger, divergence_data, train_iter)


def collect_and_log_divergences_with_heatmaps(tb_logger, merged_stats, train_iter):
    """
    Overview:
        Complete distribution divergence computation and logging (including no-diagonal heatmaps).
        End-to-end pipeline for analyzing and visualizing inter-task distribution differences.
    Arguments:
        - tb_logger (:obj:`SummaryWriter`): TensorBoard logger for metric recording.
        - merged_stats (:obj:`dict`): Merged MOE statistics from distributed training.
        - train_iter (:obj:`int`): Current training iteration for logging.
    Examples:
        >>> collect_and_log_divergences_with_heatmaps(tb_logger, merged_stats, 1000)
    """
    try:
        # GPU-optimized computation
        divergence_data = compute_distribution_divergences_optimized(merged_stats, 'immediate')

        if not divergence_data:
            print(f"Skipping distribution divergence computation - insufficient tasks (need >=2 tasks)")
            return

        # Log metrics and heatmaps
        log_divergences_with_heatmaps(tb_logger, divergence_data, train_iter)

        # Summary print
        print(f">> Distribution divergence statistics completed and logged to TensorBoard")
        if divergence_data.get('n_tasks', 0) <= 25:
            print(f">> Similarity matrix heatmaps generated (diagonal removed)")
        if divergence_data.get('n_tasks', 0) <= 20:
            print(f">> Task pair detailed metrics logged")
        print()  # Blank line separator

    except Exception as e:
        print(f"ERROR: Distribution divergence computation failed - {e}")
        import traceback
        traceback.print_exc()
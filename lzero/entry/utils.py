# -*- coding: utf-8 -*-
"""
Optimized and refactored utility code for reinforcement learning models,
focusing on clarity, professionalism, efficiency, and extensibility.
"""

# ==============================================================================
# Imports
# ==============================================================================
from __future__ import annotations

import logging
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


def initialize_zeros_batch(
    observation_shape: Union[int, List[int], Tuple[int, ...]],
    batch_size: int,
    device: str
) -> torch.Tensor:
    """
    Overview:
        Initializes a zeros tensor for a batch of observations based on the
        provided shape. This is commonly used to prepare initial input for models
        like UniZero.

    Arguments:
        - observation_shape (:obj:`Union[int, List[int], Tuple[int, ...]]`): The shape of a single observation.
        - batch_size (:obj:`int`): The number of observations in the batch.
        - device (:obj:`str`): The device to store the tensor on (e.g., 'cpu', 'cuda').

    Returns:
        - torch.Tensor: A zeros tensor with the shape [batch_size, *observation_shape].
    """
    if isinstance(observation_shape, (list, tuple)):
        shape = (batch_size, *observation_shape)
    elif isinstance(observation_shape, int):
        shape = (batch_size, observation_shape)
    else:
        raise TypeError(
            f"observation_shape must be an int, list, or tuple, but got {type(observation_shape).__name__}"
        )
    return torch.zeros(shape, device=device)


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


def initialize_pad_batch(observation_shape: Union[int, List[int], Tuple[int]], batch_size: int, device: str, pad_token_id: int = 0) -> torch.Tensor:
    """
    Overview:
        Initialize a tensor filled with `pad_token_id` for batch observations. 
        This function is designed to be flexible and can handle both textual 
        and non-textual observations:
        
        - For textual observations: it initializes `input_ids` with padding tokens, 
        ensuring consistent sequence lengths within a batch.
        - For non-textual observations: it provides a convenient way to fill 
        observation tensors with a default of 0, 
        ensuring shape compatibility and preventing uninitialized values.
    Arguments:
        - observation_shape (:obj:`Union[int, List[int], Tuple[int]]`): The shape of the observation tensor.
        - batch_size (:obj:`int`): The batch size.
        - device (:obj:`str`): The device to store the tensor.
        - pad_token_id (:obj:`int`): The token ID (or placeholder value) used for padding.
    Returns:
        - padded_tensor (:obj:`torch.Tensor`): A tensor of the given shape, 
        filled with `pad_token_id`.
    """
    if isinstance(observation_shape, (list, tuple)):
        shape = [batch_size, *observation_shape]
    elif isinstance(observation_shape, int):
        shape = [batch_size, observation_shape]
    else:
        raise TypeError(f"observation_shape must be int, list, or tuple, but got {type(observation_shape).__name__}")

    return torch.full(shape, fill_value=pad_token_id, dtype=torch.float32, device=device) if pad_token_id == -1 else torch.full(shape, fill_value=pad_token_id, dtype=torch.long, device=device)

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

<<<<<<< HEAD
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
=======
        # Reset the time records in the buffer.
        buffer.reset_runtime_metrics()


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
    import logging

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
>>>>>>> origin/main

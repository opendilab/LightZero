import os
import time
from typing import Optional, Callable, Union, List, Tuple, Dict
from io import BytesIO
import concurrent.futures

import psutil
import torch
import torch.distributed as dist
from pympler.asizeof import asizeof
from tensorboardX import SummaryWriter

import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# ============================================================
#  freeze_non_lora.py
# ------------------------------------------------------------
#  A tiny utility that (un)freezes **all** parameters except
#  those belonging to LoRA adapters / LearnableScale objects.
#
#  • Works with both CurriculumLoRALinear  &  ordinary LoRALinear
#  • O(#parameters) – just one linear scan, no recursion
#  • Can be called repeatedly; idempotent
#  • Returns (n_frozen, n_trainable) for quick logging
# ============================================================

import re
from typing import Iterable, Tuple

import torch.nn as nn


# -----------------------------------------------------------------
# helper: detect LoRA / LearnableScale parameters by their names
# -----------------------------------------------------------------
#   CurriculumLoRALinear parameters have canonical names like
#     "...adapters.3.lora_A"  (weight)
#     "...adapters.3.lora_B"  (weight)
#     "...adapter_scales.3.logit" (learnable scalar)
#   Ordinary LoRALinear (if you ever use it) typically carries
#     ".lora_A", ".lora_B" in their names as well.
#
#   So a simple regexp matching is sufficient and cheap.
# -----------------------------------------------------------------
_LORA_PAT = re.compile(r"\.(?:lora_[AB]|adapter_scales\.\d+\.logit)$")


def _is_lora_param(name: str) -> bool:
    return bool(_LORA_PAT.search(name))


# -----------------------------------------------------------------
# main API
# -----------------------------------------------------------------
def freeze_non_lora(
    module: nn.Module,
    freeze: bool = True,
    *,
    verbose: bool = False,
) -> Tuple[int, int]:
    """
    Freeze (or un-freeze) every parameter **except** LoRA / LearnableScale.

    Args:
        module   : the transformer (or any nn.Module tree)
        freeze   : True  -> set requires_grad=False for non-LoRA params
                   False -> set requires_grad=True  for non-LoRA params
        verbose  : print a short summary if True

    Returns:
        (n_frozen, n_trainable) – number of tensors in each group
    """
    n_frozen = 0
    n_train  = 0

    for name, param in module.named_parameters():
        if _is_lora_param(name):           # LoRA / scale param – keep opposite state
            param.requires_grad = True
            n_train += 1
        else:                              # everything else
            param.requires_grad = (not freeze)
            if param.requires_grad:
                n_train += 1
            else:
                n_frozen += 1

    if verbose:
        total = n_frozen + n_train
        print(
            f"[freeze_non_lora] trainable={n_train}/{total} "
            f"({n_train/total:.1%}), frozen={n_frozen}"
        )
    return n_frozen, n_train


# -----------------------------------------------------------------
# example usage inside CurriculumController.switch()
# -----------------------------------------------------------------
#
#   ...
#   if need_switch and self.stage < self.stage_num - 1:
#       self.stage += 1
#       set_curriculum_stage_for_transformer(
#           self.policy._learn_model.world_model.transformer, self.stage
#       )
#
#       # NEW : freeze all non-LoRA weights from stage-1 onwards
#       freeze_non_lora(
#           self.policy._learn_model.world_model.transformer,
#           freeze=(self.stage >= 1),
#           verbose=True,
#       )
#   ...


class TemperatureScheduler:
    def __init__(self, initial_temp: float, final_temp: float, threshold_steps: int, mode: str = 'linear'):
        """
        温度调度器，用于根据当前训练步数逐渐调整温度。

        Args:
            initial_temp (float): 初始温度值。
            final_temp (float): 最终温度值。
            threshold_steps (int): 温度衰减到最终温度所需的训练步数。
            mode (str): 衰减方式，可选 'linear' 或 'exponential'。默认 'linear'。
        """
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.threshold_steps = threshold_steps
        assert mode in ['linear', 'exponential'], "Mode must be 'linear' or 'exponential'."
        self.mode = mode

    def get_temperature(self, current_step: int) -> float:
        """
        根据当前步数计算温度。

        Args:
            current_step (int): 当前的训练步数。

        Returns:
            float: 当前温度值。
        """
        if current_step >= self.threshold_steps:
            return self.final_temp
        progress = current_step / self.threshold_steps
        if self.mode == 'linear':
            temp = self.initial_temp - (self.initial_temp - self.final_temp) * progress
        elif self.mode == 'exponential':
            # 指数衰减，确保温度逐渐接近 final_temp
            decay_rate = np.log(self.final_temp / self.initial_temp) / self.threshold_steps
            temp = self.initial_temp * np.exp(decay_rate * current_step)
            temp = max(temp, self.final_temp)
        return temp

def is_ddp_enabled():
    """
    Check if Distributed Data Parallel (DDP) is enabled by verifying if
    PyTorch's distributed package is available and initialized.
    """
    return dist.is_available() and dist.is_initialized()

def ddp_synchronize():
    """
    Perform a barrier synchronization across all processes in DDP mode.
    Ensures all processes reach this point before continuing.
    """
    if is_ddp_enabled():
        dist.barrier()

def ddp_all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    """
    Perform an all-reduce operation (sum) on the given tensor across
    all processes in DDP mode. Returns the reduced tensor.

    Arguments:
        - tensor (:obj:`torch.Tensor`): The input tensor to be reduced.

    Returns:
        - torch.Tensor: The reduced tensor, summed across all processes.
    """
    if is_ddp_enabled():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor

def calculate_update_per_collect(cfg: 'EasyDict', new_data: List[List[torch.Tensor]], world_size: int = 1) -> int:
    """
    Calculate the number of updates to perform per data collection in a
    Distributed Data Parallel (DDP) setting. This ensures that all GPUs
    compute the same `update_per_collect` value, synchronized across processes.

    Arguments:
        - cfg: Configuration object containing policy settings.
        - new_data (List[List[torch.Tensor]]): The newly collected data segments.
        - world_size (int): The total number of processes.

    Returns:
        - int: The number of updates to perform per collection.
    """
    # Retrieve the update_per_collect setting from the configuration
    update_per_collect = cfg.policy.update_per_collect

    if update_per_collect is None:
        # If update_per_collect is not explicitly set, calculate it based on
        # the number of collected transitions and the replay ratio.

        # The length of game_segment (i.e., len(game_segment.action_segment)) can be smaller than cfg.policy.game_segment_length if it represents the final segment of the game.
        # On the other hand, its length will be less than cfg.policy.game_segment_length + padding_length when it is not the last game segment. Typically, padding_length is the sum of unroll_steps and td_steps.
        collected_transitions_num = sum(
            min(len(game_segment), cfg.policy.game_segment_length)
            for game_segment in new_data[0]
        )

        if torch.cuda.is_available() and world_size > 1:
            # Convert the collected transitions count to a GPU tensor for DDP operations.
            collected_transitions_tensor = torch.tensor(
                collected_transitions_num, dtype=torch.int64, device='cuda'
            )

            # Synchronize the collected transitions count across all GPUs using all-reduce.
            total_collected_transitions = ddp_all_reduce_sum(
                collected_transitions_tensor
            ).item()

            # Calculate update_per_collect based on the total synchronized transitions count.
            update_per_collect = int(total_collected_transitions * cfg.policy.replay_ratio)

            # Ensure the computed update_per_collect is positive.
            assert update_per_collect > 0, "update_per_collect must be positive"
        else:
            # If not using DDP, calculate update_per_collect directly from the local count.
            update_per_collect = int(collected_transitions_num * cfg.policy.replay_ratio)

    return update_per_collect

def initialize_zeros_batch(observation_shape: Union[int, List[int], Tuple[int]], batch_size: int, device: str) -> torch.Tensor:
    """
    Overview:
        Initialize a zeros tensor for batch observations based on the shape. This function is used to initialize the UniZero model input.
    Arguments:
        - observation_shape (:obj:`Union[int, List[int], Tuple[int]]`): The shape of the observation tensor.
        - batch_size (:obj:`int`): The batch size.
        - device (:obj:`str`): The device to store the tensor.
    Returns:
        - zeros (:obj:`torch.Tensor`): The zeros tensor.
    """
    if isinstance(observation_shape, (list,tuple)):
        shape = [batch_size, *observation_shape]
    elif isinstance(observation_shape, int):
        shape = [batch_size, observation_shape]
    else:
        raise TypeError(f"observation_shape must be either an int, a list, or a tuple, but got {type(observation_shape).__name__}")

    return torch.zeros(shape).to(device)

def random_collect(
        policy_cfg: 'EasyDict',  # noqa
        policy: 'Policy',  # noqa
        RandomPolicy: 'Policy',  # noqa
        collector: 'ISerialCollector',  # noqa
        collector_env: 'BaseEnvManager',  # noqa
        replay_buffer: 'IBuffer',  # noqa
        postprocess_data_fn: Optional[Callable] = None
) -> None:  # noqa
    assert policy_cfg.random_collect_episode_num > 0

    random_policy = RandomPolicy(cfg=policy_cfg, action_space=collector_env.env_ref.action_space)
    # set the policy to random policy
    collector.reset_policy(random_policy.collect_mode)

    # set temperature for visit count distributions according to the train_iter,
    # please refer to Appendix D in MuZero paper for details.
    collect_kwargs = {'temperature': 1, 'epsilon': 0.0}

    # Collect data by default config n_sample/n_episode.
    new_data = collector.collect(n_episode=policy_cfg.random_collect_episode_num, train_iter=0,
                                 policy_kwargs=collect_kwargs)

    if postprocess_data_fn is not None:
        new_data = postprocess_data_fn(new_data)

    # save returned new_data collected by the collector
    replay_buffer.push_game_segments(new_data)
    # remove the oldest data if the replay buffer is full.
    replay_buffer.remove_oldest_data_to_fit()

    # restore the policy
    collector.reset_policy(policy.collect_mode)


def log_buffer_memory_usage(train_iter: int, buffer: "GameBuffer", writer: SummaryWriter, task_id=0) -> None:
    """
    Overview:
        Log the memory usage of the buffer and the current process to TensorBoard.
    Arguments:
        - train_iter (:obj:`int`): The current training iteration.
        - buffer (:obj:`GameBuffer`): The game buffer.
        - writer (:obj:`SummaryWriter`): The TensorBoard writer.
    """
    # "writer is None" means we are in a slave process in the DDP setup.
    if writer is not None:
        writer.add_scalar(f'Buffer/num_of_all_collected_episodes_{task_id}', buffer.num_of_collected_episodes, train_iter)
        writer.add_scalar(f'Buffer/num_of_game_segments_{task_id}', len(buffer.game_segment_buffer), train_iter)
        writer.add_scalar(f'Buffer/num_of_transitions_{task_id}', len(buffer.game_segment_game_pos_look_up), train_iter)

        game_segment_buffer = buffer.game_segment_buffer

        # Calculate the amount of memory occupied by self.game_segment_buffer (in bytes).
        buffer_memory_usage = asizeof(game_segment_buffer)

        # Convert buffer_memory_usage to megabytes (MB).
        buffer_memory_usage_mb = buffer_memory_usage / (1024 * 1024)

        # Record the memory usage of self.game_segment_buffer to TensorBoard.
        writer.add_scalar(f'Buffer/memory_usage/game_segment_buffer_{task_id}', buffer_memory_usage_mb, train_iter)

        # Get the amount of memory currently used by the process (in bytes).
        process = psutil.Process(os.getpid())
        process_memory_usage = process.memory_info().rss

        # Convert process_memory_usage to megabytes (MB).
        process_memory_usage_mb = process_memory_usage / (1024 * 1024)

        # Record the memory usage of the process to TensorBoard.
        writer.add_scalar(f'Buffer/memory_usage/process_{task_id}', process_memory_usage_mb, train_iter)


def log_buffer_run_time(train_iter: int, buffer: "GameBuffer", writer: SummaryWriter) -> None:
    """
    Overview:
        Log the average runtime metrics of the buffer to TensorBoard.
    Arguments:
        - train_iter (:obj:`int`): The current training iteration.
        - buffer (:obj:`GameBuffer`): The game buffer containing runtime metrics.
        - writer (:obj:`SummaryWriter`): The TensorBoard writer for logging metrics.

    .. note::
        "writer is None" indicates that the function is being called in a slave process in the DDP setup.
    """
    if writer is not None:
        sample_times = buffer.sample_times

        if sample_times == 0:
            return

        # Calculate and log average reanalyze time.
        average_reanalyze_time = buffer.compute_target_re_time / sample_times
        writer.add_scalar('Buffer/average_reanalyze_time', average_reanalyze_time, train_iter)

        # Calculate and log average origin search time.
        average_origin_search_time = buffer.origin_search_time / sample_times
        writer.add_scalar('Buffer/average_origin_search_time', average_origin_search_time, train_iter)

        # Calculate and log average reuse search time.
        average_reuse_search_time = buffer.reuse_search_time / sample_times
        writer.add_scalar('Buffer/average_reuse_search_time', average_reuse_search_time, train_iter)

        # Calculate and log average active root number.
        average_active_root_num = buffer.active_root_num / sample_times
        writer.add_scalar('Buffer/average_active_root_num', average_active_root_num, train_iter)

        # Reset the time records in the buffer.
        buffer.reset_runtime_metrics()


# ============================================================
# MOE Expert Selection Statistics Functions
# ============================================================

# Global heatmap figure cache to avoid repeated creation
_GLOBAL_HEATMAP_FIG = None
_GLOBAL_HEATMAP_AX = None


def merge_expert_stats_across_ranks(all_expert_stats):
    """
    Overview:
        Merge expert selection statistics data from all distributed training ranks.
        Combines statistics from different GPU processes for comprehensive analysis.
    Arguments:
        - all_expert_stats (:obj:`list`): List of expert statistics from all ranks.
    Returns:
        - merged_stats (:obj:`dict`): Merged statistics dictionary with structure
                                    {task_id: {window_type: stats}}.
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
        Collect and log MOE expert selection statistics including heatmaps and distribution analysis.
        Comprehensive function that handles distributed data collection, merging, and visualization.
    Arguments:
        - policy (:obj:`Policy`): Training policy object containing world model.
        - tb_logger (:obj:`SummaryWriter`): TensorBoard logger for metric recording.
        - train_iter (:obj:`int`): Current training iteration number.
        - world_size (:obj:`int`): Total number of GPUs in distributed training.
        - rank (:obj:`int`): Current GPU rank identifier.
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
        - merged_stats (:obj:`dict`): Merged MOE statistics from all distributed ranks.
        - window_type (:obj:`str`, optional): Time window type. Default is 'immediate'.
    Returns:
        - divergence_data (:obj:`dict`): Comprehensive divergence analysis results including
                                        matrices, statistics, and metadata.
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

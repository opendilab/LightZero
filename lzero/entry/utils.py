import os
import time
from typing import Optional, Callable, Union, List, Tuple

import psutil
import torch
import torch.distributed as dist
from pympler.asizeof import asizeof
from tensorboardX import SummaryWriter


import torch
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

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


# ==================== MoE TensorBoard 记录模块 =============================
# 导入必要的模块
import seaborn as sns
from io import BytesIO
from PIL import Image
import concurrent.futures

# 全局图像缓存，避免重复创建 figure
_GLOBAL_HEATMAP_FIG = None
_GLOBAL_HEATMAP_AX = None

def _get_or_create_heatmap_figure(figsize):
    """获取或创建复用的 heatmap figure"""
    global _GLOBAL_HEATMAP_FIG, _GLOBAL_HEATMAP_AX
    if _GLOBAL_HEATMAP_FIG is None:
        _GLOBAL_HEATMAP_FIG, _GLOBAL_HEATMAP_AX = plt.subplots(figsize=figsize)
    else:
        # 清除之前的内容
        _GLOBAL_HEATMAP_AX.clear()
        # 调整图像大小
        _GLOBAL_HEATMAP_FIG.set_size_inches(figsize)
    return _GLOBAL_HEATMAP_FIG, _GLOBAL_HEATMAP_AX

def create_heatmap_with_values_fast(matrix, task_ids, title="Task-Expert Selection Frequencies"):
    """
    高效创建带数值标注的蓝色系热力图 - 优化版本
    
    优化点:
    1. 复用 matplotlib figure，减少内存分配
    2. 大矩阵跳过数值标注，避免性能损失
    3. 优化图像转换流程
    4. 使用更低的 DPI 减少计算量
    """
    try:
        figsize = (max(6, matrix.shape[1]), max(4, matrix.shape[0]))
        fig, ax = _get_or_create_heatmap_figure(figsize)
        
        # 智能选择是否显示数值标注
        show_annot = matrix.size <= 64  # 只在 8x8 或更小时显示数值
        
        # 使用 matplotlib 直接绘制，避免 seaborn 的额外开销
        im = ax.imshow(matrix, cmap='Blues', aspect='auto')
        
        # 有选择性地添加数值标注
        if show_annot:
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    value = matrix[i, j]
                    color = 'white' if value > 0.5 else 'black'
                    ax.text(j, i, f'{value:.3f}', ha='center', va='center', 
                           color=color, fontsize=8)
        
        # 设置标签和标题
        ax.set_xticks(range(matrix.shape[1]))
        ax.set_yticks(range(matrix.shape[0]))
        ax.set_xticklabels([f'E{i}' for i in range(matrix.shape[1])], fontsize=10)
        ax.set_yticklabels([f'T{tid}' for tid in task_ids], fontsize=10)
        ax.set_title(title, fontsize=12, pad=15)
        ax.set_xlabel('Experts', fontsize=10)
        ax.set_ylabel('Tasks', fontsize=10)
        
        # 简化的 colorbar
        if not hasattr(fig, '_colorbar_created'):
            plt.colorbar(im, ax=ax, label='Frequency')
            fig._colorbar_created = True
        
        # 优化的图像转换：使用更低 DPI 和简化流程
        fig.canvas.draw()
        try:
            # 直接从 canvas 获取 RGB 数据
            if hasattr(fig.canvas, 'buffer_rgba'):
                buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
                buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
                img_array = buf[:, :, :3]  # 去掉 alpha 通道
            else:
                buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                img_array = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            
            # 转换为 CHW 格式
            img_array = img_array.transpose(2, 0, 1)
            
        except Exception:
            # 回退方案：创建简单的蓝色渠度矩阵
            h, w = matrix.shape
            img_array = np.zeros((3, h*20, w*20), dtype=np.uint8)
            # 简单放大矩阵并映射到蓝色通道
            matrix_resized = np.repeat(np.repeat(matrix, 20, axis=0), 20, axis=1)
            img_array[2] = (matrix_resized * 255).astype(np.uint8)
        
        return img_array
        
    except Exception as e:
        print(f"Warning: 热力图生成失败: {e}, 使用回退方案")
        # 终极回退：返回空白图像
        return np.zeros((3, 100, 100), dtype=np.uint8)

def create_heatmap_with_values(matrix, task_ids, title="Task-Expert Selection Frequencies"):
    """创建带数值标注的蓝色系热力图 - 原始版本（回退用）"""
    fig, ax = plt.subplots(figsize=(max(8, matrix.shape[1]), max(6, matrix.shape[0])))
    
    # 使用蓝色系颜色映射
    sns.heatmap(matrix, 
                annot=True,  # 显示数值
                fmt='.3f',   # 数值格式
                cmap='Blues',  # 蓝色系
                ax=ax,
                cbar_kws={'label': 'Selection Frequency'},
                xticklabels=[f'Expert{i}' for i in range(matrix.shape[1])],
                yticklabels=[f'Task{tid}' for tid in task_ids])
    
    ax.set_title(title, fontsize=14, pad=20)
    ax.set_xlabel('Experts', fontsize=12)
    ax.set_ylabel('Tasks', fontsize=12)
    
    plt.tight_layout()
    
    # 保存到BytesIO
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    
    # 转换为numpy数组用于tensorboard
    img = Image.open(buf)
    img_array = np.array(img)
    buf.close()
    plt.close(fig)
    
    # 转换为CHW格式 (Channel, Height, Width)
    if len(img_array.shape) == 3:
        img_array = img_array.transpose(2, 0, 1)
    
    return img_array

def log_expert_selection_details(tb_logger, merged_stats, valid_task_ids, matrix, window_type, train_iter):
    """记录每个任务的详细专家选择统计"""
    for i, task_id in enumerate(valid_task_ids):
        frequencies = matrix[i]
        stats = merged_stats[task_id][window_type]
        
        # 计算并记录该任务选择专家的熵（均匀性指标）
        task_frequencies = np.array(frequencies)
        task_frequencies = task_frequencies + 1e-8  # 避免log(0)
        task_entropy = -np.sum(task_frequencies * np.log(task_frequencies))
        tb_logger.add_scalar(
            f'MOE_Details/Task{task_id}_{window_type}/ExpertSelectionEntropy',
            task_entropy, global_step=train_iter
        )
        
        # 记录该任务专家选择的方差（分散程度）
        expert_variance = np.var(task_frequencies)
        tb_logger.add_scalar(
            f'MOE_Details/Task{task_id}_{window_type}/ExpertSelectionVariance',
            expert_variance, global_step=train_iter
        )
        
        # 记录任务级别的汇总统计
        tb_logger.add_scalar(
            f'MOE_Details/Task{task_id}_{window_type}/TotalSelections',
            stats['total_selections'], global_step=train_iter
        )
        tb_logger.add_scalar(
            f'MOE_Details/Task{task_id}_{window_type}/DataPoints',
            stats['data_points'], global_step=train_iter
        )

def log_global_moe_statistics(tb_logger, matrix, window_type, valid_task_ids, train_iter):
    """记录全局MOE统计信息"""
    # 记录基本信息
    tb_logger.add_scalar(
        f'MOE_Global/{window_type}/NumActiveTasks',
        len(valid_task_ids), global_step=train_iter
    )
    tb_logger.add_scalar(
        f'MOE_Global/{window_type}/NumExperts', 
        matrix.shape[1], global_step=train_iter
    )
    
    # 计算专家使用均匀性
    expert_avg_usage = np.mean(matrix, axis=0)  # 每个专家的平均使用频率
    usage_entropy = -np.sum(expert_avg_usage * np.log(expert_avg_usage + 1e-8))
    tb_logger.add_scalar(
        f'MOE_Global/{window_type}/ExpertUsageEntropy',
        usage_entropy, global_step=train_iter
    )
    
    # 记录最常用和最少用的专家
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
    高效处理和记录MOE热力图 - 优化版本
    
    优化点:
    1. 向量化数据处理，减少循环
    2. 使用高效的热力图生成函数
    3. 条件性热力图生成
    4. 批量处理统计数据
    """
    # 快速筛选有效任务
    valid_task_data = [(tid, stats[window_type]['frequencies']) 
                      for tid, stats in merged_stats.items() 
                      if window_type in stats]
    
    if not valid_task_data:
        return
    
    # 向量化构建矩阵
    valid_task_ids, frequencies_list = zip(*valid_task_data)
    matrix = np.array(frequencies_list)
    
    # 条件性热力图生成：小矩阵才生成热力图
    if matrix.size <= 200:  # 只有在任务数*专家数 <= 200时才生成热力图
        try:
            heatmap_img = create_heatmap_with_values_fast(
                matrix, valid_task_ids, 
                f'MOE {window_type} Task-Expert Selection'
            )
            
            # 记录热力图到tensorboard
            tb_logger.add_image(
                f'MOE_Heatmap/{window_type}_TaskExpert_Heatmap',
                heatmap_img,
                global_step=train_iter,
                dataformats='CHW'
            )
        except Exception as e:
            print(f"Warning: 热力图生成失败: {e}")
    
    # 始终记录统计数据（轻量级操作）
    log_expert_selection_details(tb_logger, merged_stats, valid_task_ids, matrix, window_type, train_iter)
    log_global_moe_statistics(tb_logger, matrix, window_type, valid_task_ids, train_iter)

def process_and_log_moe_heatmaps(tb_logger, merged_stats, window_type, train_iter):
    """处理和记录MOE热力图 - 原始版本（回退用）"""
    all_task_ids = sorted(merged_stats.keys())
    task_expert_matrix = []
    valid_task_ids = []
    
    # 收集有效任务的频率数据
    for task_id in all_task_ids:
        if window_type in merged_stats[task_id]:
            frequencies = merged_stats[task_id][window_type]['frequencies']
            task_expert_matrix.append(frequencies)
            valid_task_ids.append(task_id)
    
    if not task_expert_matrix:
        return
    
    # 转换为numpy矩阵 (num_tasks, num_experts)
    matrix = np.array(task_expert_matrix)
    
    # 创建带数值标注的蓝色系热力图
    heatmap_img = create_heatmap_with_values(
        matrix, valid_task_ids, 
        f'MOE {window_type} Task-Expert Selection Frequencies'
    )
    
    # 记录热力图到tensorboard
    tb_logger.add_image(
        f'MOE_Heatmap/{window_type}_TaskExpert_Heatmap',
        heatmap_img,
        global_step=train_iter,
        dataformats='CHW'
    )
    
    # 记录详细统计和全局统计
    log_expert_selection_details(tb_logger, merged_stats, valid_task_ids, matrix, window_type, train_iter)

def convert_stats_to_serializable(moe_stats):
    """将MOE统计数据中的tensor转换为可序列化的numpy格式"""
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
    """在分布式训练中收集和合并MOE统计数据"""
    if world_size == 1:
        return local_stats
    
    # 将本地统计转换为可序列化格式后进行分布式收集
    serializable_stats = convert_stats_to_serializable(local_stats)
    return serializable_stats

def collect_and_log_moe_statistics(policy, tb_logger, train_iter, world_size, rank):
    """
    收集和记录MOE统计信息 - 主要入口函数
    
    优化版本，增加了异常处理和性能监控
    """
    try:
        # 从policy收集本地MOE统计
        local_stats = {}
        if hasattr(policy, '_learn_model') and hasattr(policy._learn_model, 'world_model'):
            world_model = policy._learn_model.world_model
            
            # 检查是否有transformer和MoE层
            if hasattr(world_model, 'transformer'):
                transformer = world_model.transformer
                if hasattr(transformer, 'moe_layers') and transformer.moe_layers:
                    # 只从最后一个MoE层收集统计（性能优化）
                    last_moe_layer = transformer.moe_layers[-1]
                    if hasattr(last_moe_layer, 'get_expert_selection_stats'):
                        local_stats = last_moe_layer.get_expert_selection_stats()
        
        # 分布式收集统计（简化版本）
        merged_stats = gather_distributed_moe_stats(local_stats, world_size)
        
        # 只在rank 0记录到TensorBoard
        if rank == 0 and tb_logger and merged_stats:
            # 处理不同时间窗口的统计
            for window_type in ['immediate', 'short', 'medium', 'long']:
                # 检查是否有有效数据
                has_data = any(window_type in task_stats for task_stats in merged_stats.values())
                if has_data:
                    # 使用优化版本的热力图处理
                    process_and_log_moe_heatmaps_fast(tb_logger, merged_stats, window_type, train_iter)
    
    except Exception as e:
        print(f"Rank {rank}: MOE统计收集失败 - {e}, train_iter={train_iter}")
        import traceback
        traceback.print_exc()

# ====== GPU优化的分布差异计算和可视化函数 ======
def jensen_shannon_divergence_batch_gpu(distributions_tensor):
    """
    GPU批量计算JS散度矩阵 - 使用GPU优化器的内存池
    
    Args:
        distributions_tensor: shape (n_tasks, n_experts), GPU张量
    
    Returns:
        js_matrix: shape (n_tasks, n_tasks), 对称矩阵
    """
    # 使用GPU优化器提升性能
    return get_gpu_optimizer().optimized_js_divergence(distributions_tensor)

def wasserstein_distance_batch_gpu(distributions_tensor):
    """
    GPU批量计算Wasserstein距离矩阵 - 使用GPU优化器的内存池
    
    Args:
        distributions_tensor: shape (n_tasks, n_experts), GPU张量
    
    Returns:
        wasserstein_matrix: shape (n_tasks, n_tasks), 对称矩阵
    """
    # 使用GPU优化器提升性能
    return get_gpu_optimizer().optimized_wasserstein(distributions_tensor)

def compute_distribution_divergences_optimized(merged_stats, window_type='immediate'):
    """
    GPU优化版本 - 高效分布差异计算
    """
    # 1. 数据预处理
    valid_tasks = [(tid, stats[window_type]['frequencies']) 
                  for tid, stats in merged_stats.items() 
                  if window_type in stats]
    
    if len(valid_tasks) < 2:
        return {}
    
    task_ids, frequencies_list = zip(*valid_tasks)
    
    # 2. 高效张量转换
    try:
        if isinstance(frequencies_list[0], torch.Tensor):
            frequencies_tensor = torch.stack(frequencies_list)
        else:
            frequencies_tensor = torch.tensor(
                np.array(frequencies_list), 
                dtype=torch.float32
            )
        
        # 自动GPU加速
        if torch.cuda.is_available():
            frequencies_tensor = frequencies_tensor.cuda()
            
    except Exception as e:
        print(f"GPU转换失败，使用CPU: {e}")
        frequencies_tensor = torch.tensor(np.array(frequencies_list), dtype=torch.float32)
    
    device = frequencies_tensor.device
    n_tasks, n_experts = frequencies_tensor.shape
    
    # 3. GPU批量计算（无循环）
    with torch.no_grad():
        # 批量计算JS散度和Wasserstein距离
        js_matrix = jensen_shannon_divergence_batch_gpu(frequencies_tensor)
        wasserstein_matrix = wasserstein_distance_batch_gpu(frequencies_tensor)
        
        # 高效提取上三角值（避免重复计算）
        triu_indices = torch.triu_indices(n_tasks, n_tasks, offset=1, device=device)
        js_values = js_matrix[triu_indices[0], triu_indices[1]]
        wasserstein_values = wasserstein_matrix[triu_indices[0], triu_indices[1]]
        
        # 统计计算（向量化）
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
        
        # 返回CPU版本用于记录
        'js_matrix': js_matrix.cpu().numpy(),
        'wasserstein_matrix': wasserstein_matrix.cpu().numpy(),
        'js_stats': js_stats,
        'wasserstein_stats': wasserstein_stats
    }

def create_similarity_heatmap_no_diagonal(similarity_matrix, task_ids, metric_name, title_suffix=""):
    """
    创建任务相似度热力图 - 去掉对角线部分
    
    Args:
        similarity_matrix: 相似度矩阵 (n_tasks, n_tasks)
        task_ids: 任务ID列表
        metric_name: 指标名称 ('js_divergence', 'wasserstein_distance')
        title_suffix: 标题后缀
    """
    try:
        # 复制矩阵避免修改原数据
        matrix = similarity_matrix.copy()
        
        # 将对角线设置为NaN，这样matplotlib会显示为空白
        np.fill_diagonal(matrix, np.nan)
        
        figsize = (max(6, len(task_ids)), max(4, len(task_ids)))
        fig, ax = plt.subplots(figsize=figsize)  # 创建新figure避免复用问题
        
        # 根据指标类型选择颜色映射
        if 'js' in metric_name.lower():
            cmap = 'Reds'
            title_name = 'JS Divergence'
            vmin, vmax = 0, 1.0
        else:  # wasserstein
            cmap = 'Blues'  
            title_name = 'Wasserstein Distance'
            vmin, vmax = None, None  # 自适应
        
        # 使用masked数组处理NaN值，对角线显示为白色
        masked_matrix = np.ma.masked_invalid(matrix)
        im = ax.imshow(masked_matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
        
        # 添加数值标注（跳过对角线）
        if len(task_ids) <= 15:  # 只在任务数较少时添加标注
            for i in range(len(task_ids)):
                for j in range(len(task_ids)):
                    if i != j:  # 跳过对角线
                        value = matrix[i, j]
                        if not np.isnan(value):
                            threshold = (vmax or np.nanmax(matrix)) * 0.5 if vmax else np.nanmax(matrix) * 0.5
                            color = 'white' if value > threshold else 'black'
                            ax.text(j, i, f'{value:.3f}', ha='center', va='center', 
                                   color=color, fontsize=8)
        
        # 设置标签
        ax.set_xticks(range(len(task_ids)))
        ax.set_yticks(range(len(task_ids)))
        ax.set_xticklabels([f'T{tid}' for tid in task_ids], fontsize=9)
        ax.set_yticklabels([f'T{tid}' for tid in task_ids], fontsize=9)
        ax.set_title(f'Task {title_name} Matrix {title_suffix} (No Diagonal)', fontsize=12)
        ax.set_xlabel('Tasks', fontsize=10)
        ax.set_ylabel('Tasks', fontsize=10)
        
        # 添加colorbar
        plt.colorbar(im, ax=ax, label=title_name, shrink=0.8)
        
        # 转换为图像数组 - 修复matplotlib版本兼容性
        fig.canvas.draw()
        
        try:
            # 新版matplotlib使用buffer_rgba
            if hasattr(fig.canvas, 'buffer_rgba'):
                buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
                h, w = fig.canvas.get_width_height()
                img_array = buf.reshape(h, w, 4)[:, :, :3]  # 去掉alpha通道
            else:
                # 旧版matplotlib回退方案
                buf = fig.canvas.print_to_string()
                img_array = np.frombuffer(buf, dtype=np.uint8)
                h, w = fig.canvas.get_width_height()
                img_array = img_array.reshape(h, w, 3)
        except Exception as conv_e:
            print(f"图像转换方法失败: {conv_e}, 尝试PIL方案")
            # 最终回退：通过PIL转换
            from io import BytesIO
            buf = BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            from PIL import Image
            img = Image.open(buf)
            img_array = np.array(img)[:, :, :3]  # 去掉alpha通道
            buf.close()
        
        img_array = img_array.transpose(2, 0, 1)  # CHW格式
        plt.close(fig)  # 关闭figure避免内存泄漏
        
        return img_array
        
    except Exception as e:
        print(f"Warning: 无对角线热力图生成失败: {e}")
        return np.zeros((3, 100, 100), dtype=np.uint8)

def log_pairwise_optimized(tb_logger, divergence_data, train_iter):
    """
    优化的任务对记录 - 批量处理
    """
    task_ids = divergence_data['task_ids']
    js_matrix = divergence_data['js_matrix']
    wasserstein_matrix = divergence_data['wasserstein_matrix']
    
    # 批量构建任务对指标字典
    pairwise_scalars = {}
    
    for i, task_i in enumerate(task_ids):
        for j, task_j in enumerate(task_ids):
            if i < j:  # 只记录上三角
                # 构建指标名称
                js_key = f'TaskPairwise/Immediate_Task{task_i}_Task{task_j}_JS_Divergence'
                wass_key = f'TaskPairwise/Immediate_Task{task_i}_Task{task_j}_Wasserstein_Distance'
                
                pairwise_scalars[js_key] = js_matrix[i, j]
                pairwise_scalars[wass_key] = wasserstein_matrix[i, j]
    
    # 批量写入TensorBoard
    for key, value in pairwise_scalars.items():
        tb_logger.add_scalar(key, float(value), global_step=train_iter)

def log_divergences_with_heatmaps(tb_logger, divergence_data, train_iter):
    """
    记录分布差异指标和热力图（去掉对角线）
    """
    if not divergence_data:
        return
    
    js_stats = divergence_data['js_stats']
    wasserstein_stats = divergence_data['wasserstein_stats']
    task_ids = divergence_data['task_ids']
    n_tasks = divergence_data['n_tasks']
    
    # 调试：检查矩阵数据
    js_matrix = divergence_data['js_matrix']
    wasserstein_matrix = divergence_data['wasserstein_matrix']
    print(f"DEBUG: JS矩阵形状={js_matrix.shape}, 范围=[{np.min(js_matrix):.6f}, {np.max(js_matrix):.6f}]")
    print(f"DEBUG: Wasserstein矩阵形状={wasserstein_matrix.shape}, 范围=[{np.min(wasserstein_matrix):.6f}, {np.max(wasserstein_matrix):.6f}]")
    
    # 1. 记录标量指标
    scalar_dict = {
        'MOE_Divergence/Immediate_AvgJS_Divergence': js_stats['avg'],
        'MOE_Divergence/Immediate_MaxJS_Divergence': js_stats['max'],
        'MOE_Divergence/Immediate_AvgWasserstein_Distance': wasserstein_stats['avg'],
        'MOE_Divergence/Immediate_MaxWasserstein_Distance': wasserstein_stats['max'],
    }
    
    for key, value in scalar_dict.items():
        tb_logger.add_scalar(key, value, global_step=train_iter)
    
    # 1.1 打印核心指标到控制台
    print("=" * 65)
    print(f" 任务间分布差异统计 (Iteration: {train_iter})")
    print("=" * 65)
    print(f"参与任务数量: {n_tasks} | 任务ID: {list(task_ids)}")
    print(f"计算设备: {divergence_data.get('device', 'Unknown')} | GPU加速: {'启用' if divergence_data.get('gpu_accelerated', False) else '禁用'}")
    print("-" * 65)
    print("JS散度 (Jensen-Shannon Divergence):")
    print(f"  平均值: {js_stats['avg']:.6f}  |  最大值: {js_stats['max']:.6f}")
    print(f"  最小值: {js_stats['min']:.6f}  |  标准差: {js_stats['std']:.6f}")
    print("-" * 65)
    print("Wasserstein距离:")
    print(f"  平均值: {wasserstein_stats['avg']:.6f}  |  最大值: {wasserstein_stats['max']:.6f}")
    print(f"  最小值: {wasserstein_stats['min']:.6f}  |  标准差: {wasserstein_stats['std']:.6f}")
    print("=" * 65)
    
    # 2. 记录去掉对角线的相似度矩阵热力图
    task_ids = divergence_data['task_ids']
    n_tasks = divergence_data['n_tasks']
    
    if n_tasks <= 25:  # 限制矩阵大小避免过大热力图
        try:
            # JS散度矩阵热力图（无对角线）
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
            
            # Wasserstein距离矩阵热力图（无对角线）
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
            print(f"Warning: 相似度矩阵热力图生成失败: {e}")
    
    # 3. 记录任务对指标（可选）
    if n_tasks <= 20:
        log_pairwise_optimized(tb_logger, divergence_data, train_iter)

def collect_and_log_divergences_with_heatmaps(tb_logger, merged_stats, train_iter):
    """
    完整的分布差异计算和记录（包含无对角线热力图）
    """
    try:
        # GPU优化计算
        divergence_data = compute_distribution_divergences_optimized(merged_stats, 'immediate')
        
        if not divergence_data:
            print(f"跳过分布差异计算 - 任务数不足 (需要>=2个任务)")
            return
        
        # 记录指标和热力图
        log_divergences_with_heatmaps(tb_logger, divergence_data, train_iter)
        
        # 汇总打印
        print(f">> 分布差异统计已完成并记录到TensorBoard")
        if divergence_data.get('n_tasks', 0) <= 25:
            print(f">> 相似度矩阵热力图已生成 (去除对角线)")
        if divergence_data.get('n_tasks', 0) <= 20:
            print(f">> 任务对详细指标已记录")
        print()  # 空行分隔
        
    except Exception as e:
        print(f"ERROR: 分布差异计算失败 - {e}")
        import traceback
        traceback.print_exc()


# ==================== GPU内存池优化模块 =============================
class GPUTensorPool:
    """
    轻量级GPU张量池 - 针对8x8矩阵优化
    
    只缓存最常用的张量：
    - 频率矩阵 (8, 8)
    - JS散度矩阵 (8, 8) 
    - Wasserstein矩阵 (8, 8)
    - 临时计算缓冲区
    """
    def __init__(self, device):
        self.device = device
        self.tensor_cache = {}
        self.max_cache_size = 20  # 限制缓存大小
        self.hit_count = 0
        self.miss_count = 0
    
    def get_tensor(self, shape, dtype=torch.float32, key="default"):
        """获取缓存的张量或创建新的"""
        cache_key = (tuple(shape), dtype, key)
        
        if cache_key in self.tensor_cache:
            tensor = self.tensor_cache[cache_key]
            if tensor.shape == shape and tensor.device == self.device:
                self.hit_count += 1
                return tensor.zero_()  # 复用并清零
        
        # 创建新张量并缓存
        tensor = torch.zeros(shape, dtype=dtype, device=self.device)
        if len(self.tensor_cache) < self.max_cache_size:
            self.tensor_cache[cache_key] = tensor
        
        self.miss_count += 1
        return tensor
    
    def get_cache_stats(self):
        """获取缓存命中率统计"""
        total = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total if total > 0 else 0
        return {
            'hit_count': self.hit_count,
            'miss_count': self.miss_count, 
            'hit_rate': hit_rate,
            'cache_size': len(self.tensor_cache)
        }
    
    def clear_cache(self):
        """清理缓存"""
        self.tensor_cache.clear()
        self.hit_count = 0
        self.miss_count = 0


class BatchComputeOptimizer:
    """
    批量计算优化器 - GPU向量化处理
    
    优化目标：
    - JS散度计算向量化
    - Wasserstein距离计算向量化  
    - 减少GPU内存分配
    """
    def __init__(self, tensor_pool):
        self.pool = tensor_pool
        self.compute_count = 0
        self.total_compute_time = 0.0
    
    def optimized_js_divergence(self, distributions_tensor):
        """优化的JS散度计算 - 复用内存"""
        start_time = time.time() if hasattr(time, 'time') else 0
        
        n_tasks, n_experts = distributions_tensor.shape
        device = distributions_tensor.device
        
        # 复用缓存的张量
        js_matrix = self.pool.get_tensor((n_tasks, n_tasks), key="js_matrix")
        
        # 向量化计算（原有算法保持不变）
        eps = 1e-8
        distributions_tensor = distributions_tensor / (distributions_tensor.sum(dim=1, keepdim=True) + eps)
        
        P_i = distributions_tensor.unsqueeze(1)  
        P_j = distributions_tensor.unsqueeze(0)  
        M = 0.5 * (P_i + P_j)
        
        log_ratio_i = torch.log((P_i + eps) / (M + eps))
        kl_i_m = torch.sum(P_i * log_ratio_i, dim=2)
        
        log_ratio_j = torch.log((P_j + eps) / (M + eps))
        kl_j_m = torch.sum(P_j * log_ratio_j, dim=2)
        
        js_matrix.copy_(0.5 * (kl_i_m + kl_j_m))
        
        # 统计计算时间
        if hasattr(time, 'time'):
            self.total_compute_time += time.time() - start_time
            self.compute_count += 1
            
        return js_matrix
        
    def optimized_wasserstein(self, distributions_tensor):
        """优化的Wasserstein距离计算 - 复用内存"""
        start_time = time.time() if hasattr(time, 'time') else 0
        
        n_tasks, n_experts = distributions_tensor.shape
        
        # 复用缓存的张量
        wass_matrix = self.pool.get_tensor((n_tasks, n_tasks), key="wass_matrix")
        cdf_buffer = self.pool.get_tensor((n_tasks, n_experts), key="cdf_buffer")
        
        # 向量化计算
        eps = 1e-8
        distributions_tensor = distributions_tensor / (distributions_tensor.sum(dim=1, keepdim=True) + eps)
        
        # 复用缓冲区计算CDF
        torch.cumsum(distributions_tensor, dim=1, out=cdf_buffer)
        
        cdf_i = cdf_buffer.unsqueeze(1)
        cdf_j = cdf_buffer.unsqueeze(0)
        
        wass_matrix.copy_(torch.sum(torch.abs(cdf_i - cdf_j), dim=2))
        
        # 统计计算时间
        if hasattr(time, 'time'):
            self.total_compute_time += time.time() - start_time
            self.compute_count += 1
            
        return wass_matrix
    
    def get_performance_stats(self):
        """获取性能统计"""
        avg_time = self.total_compute_time / self.compute_count if self.compute_count > 0 else 0
        return {
            'compute_count': self.compute_count,
            'total_time': self.total_compute_time,
            'avg_time_per_compute': avg_time,
            'cache_stats': self.pool.get_cache_stats()
        }


# 全局优化器实例
_gpu_optimizer = None

def get_gpu_optimizer():
    """获取全局GPU优化器实例"""
    global _gpu_optimizer
    if _gpu_optimizer is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tensor_pool = GPUTensorPool(device)
        _gpu_optimizer = BatchComputeOptimizer(tensor_pool)
    return _gpu_optimizer

def get_optimization_stats():
    """获取优化性能统计（调试用）"""
    if _gpu_optimizer is not None:
        return _gpu_optimizer.get_performance_stats()
    return None

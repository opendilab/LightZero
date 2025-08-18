import logging
import os
from functools import partial
from typing import Tuple, Optional, List

import torch
import numpy as np
from ding.config import compile_config
from ding.envs import create_env_manager, get_vec_env_setting
from ding.policy import create_policy
from ding.rl_utils import get_epsilon_greedy_fn
from ding.utils import set_pkg_seed, get_rank, get_world_size
from ding.worker import BaseLearner
from tensorboardX import SummaryWriter

# 添加性能监控相关导入
try:
    from line_profiler import LineProfiler
except ImportError:
    LineProfiler = None

from lzero.entry.utils import (
    log_buffer_memory_usage, TemperatureScheduler,
    collect_and_log_moe_statistics, collect_and_log_divergences_with_heatmaps
)
from lzero.policy import visit_count_temperature
from lzero.worker import MuZeroEvaluator as Evaluator
from lzero.worker import MuZeroSegmentCollector as Collector
from ding.utils import EasyTimer
import torch.nn.functional as F
import sys
import os
PROJECT_ROOT = os.path.abspath("/fs-computility/niuyazhe/tangjia/github/LightZero") # 或者直接写死路径
sys.path.insert(0, PROJECT_ROOT)
import torch.distributed as dist
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from PIL import Image
# tb_logger = None
# ------------------------------------------------------------
# 1. 额外增加 learner 专用 process-group 
#    (在 main / learner 初始化时调用一次)
# ------------------------------------------------------------
def build_learner_group(learner_ranks: list[int]) -> dist.ProcessGroup:
    """
    learner_ranks 里只放 **真正执行 backward** 的那些 rank
      例：CUDA_VISIBLE_DEVICES=0,1  →  learner_ranks=[0,1]
    返回一个新的 ProcessGroup，后续给 GenericMoCo 使用
    """
    world_pg = dist.group.WORLD
    pg = dist.new_group(ranks=learner_ranks, backend='nccl')
    if dist.get_rank() in learner_ranks:
        torch.cuda.set_device(learner_ranks.index(dist.get_rank()))
    return pg


# ------------------------------------------------------------
# MOE专家选择统计相关函数
# ------------------------------------------------------------
def merge_expert_stats_across_ranks(all_expert_stats):
    """合并所有rank的专家选择统计数据"""
    merged_stats = {}  # {task_id: {window_type: stats}}
    
    for rank_expert_stats in all_expert_stats:
        if rank_expert_stats:
            for task_id, task_stats in rank_expert_stats.items():
                if task_id not in merged_stats:
                    merged_stats[task_id] = {}
                
                for window_type, stats in task_stats.items():
                    # 只处理有实际数据的统计（当前GPU负责的任务）
                    if stats and stats.get('total_selections', 0) > 0:
                        merged_stats[task_id][window_type] = {
                            'frequencies': np.array(stats['frequencies']),
                            'total_selections': stats['total_selections'],
                            'data_points': stats['data_points']
                        }
    return merged_stats


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

# 保留原始函数作为回退
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
        
        # 记录每个专家的选择频率
        # for expert_id, freq in enumerate(frequencies):
        #     tb_logger.add_scalar(
        #         f'MOE_Details/Task{task_id}_{window_type}/Expert{expert_id}_Frequency',
        #         float(freq), global_step=train_iter
        #     )
        
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

# 保留原始函数作为回退
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
    """分布式环境下汇总所有GPU的MOE统计数据"""
    all_stats = [None for _ in range(world_size)]
    try:
        dist.all_gather_object(all_stats, local_stats)
        return all_stats
    except Exception as e:
        print(f"分布式MOE统计汇总失败: {e}")
        return [local_stats]  # fallback到本地统计


def collect_and_log_moe_statistics(policy, tb_logger, train_iter, world_size, rank):
    """
    收集并记录MOE专家选择统计信息，包括热力图和分布分析
    
    Args:
        policy: 训练策略对象，包含世界模型
        tb_logger: TensorBoard日志记录器
        train_iter: 当前训练迭代次数
        world_size: 分布式训练的总GPU数量
        rank: 当前GPU的rank
    """
    try:
        # Step 1: 从policy的transformer模型中获取MOE统计
        moe_stats = None
        
        transformer = policy._model.world_model.transformer
        if hasattr(transformer, 'get_expert_selection_stats'):
            moe_stats = transformer.get_expert_selection_stats()
        
        if moe_stats is None:
            print(f"Rank {rank}: 警告: 无法获取MOE统计数据，train_iter={train_iter}")
            return
        
        # Step 2: 转换tensor数据为可序列化格式
        serializable_stats = convert_stats_to_serializable(moe_stats)
        
        print(f"Rank {rank}: 本地MOE统计 - 任务数: {len(serializable_stats)}, train_iter={train_iter}")
        
        # Step 3: 分布式汇总所有GPU的统计数据
        all_expert_stats = gather_distributed_moe_stats(serializable_stats, world_size)
        
        # Step 4: 合并统计数据
        merged_stats = merge_expert_stats_across_ranks(all_expert_stats)
        
        if not merged_stats:
            print(f"Rank {rank}: 警告: 合并后的MOE统计为空，train_iter={train_iter}")
            return
        
        # Step 5: 所有GPU都记录MOE统计，每个GPU记录自己的日志
        print(f"Rank {rank}: 开始记录MOE统计 - 合并任务数: {len(merged_stats)}, train_iter={train_iter}")
        
        # 为每个时间窗口生成热力图和统计
        for window_type in ['immediate', 'short', 'medium', 'long']:
            if any(window_type in task_stats for task_stats in merged_stats.values()):
                process_and_log_moe_heatmaps_fast(tb_logger, merged_stats, window_type, train_iter)
        
        # 记录总体MOE使用情况
        tb_logger.add_scalar('MOE_Global/ActiveTasks', len(merged_stats), global_step=train_iter)
        
        # Step 6: 新增分布差异计算和记录（包含去对角线热力图）
        if any('immediate' in task_stats for task_stats in merged_stats.values()):
            print(f"Rank {rank}: 开始计算任务间分布差异...")
            collect_and_log_divergences_with_heatmaps(tb_logger, merged_stats, train_iter)
        
        print(f"Rank {rank}: MOE统计记录完成，train_iter={train_iter}")
    
    except Exception as e:
        print(f"Rank {rank}: MOE统计收集失败 - {e}, train_iter={train_iter}")
        import traceback
        traceback.print_exc()

import concurrent.futures

# ====== GPU优化的分布差异计算和可视化函数 ======
def jensen_shannon_divergence_batch_gpu(distributions_tensor):
    """
    GPU批量计算JS散度矩阵 - 完全向量化，无循环
    
    Args:
        distributions_tensor: shape (n_tasks, n_experts), GPU张量
    
    Returns:
        js_matrix: shape (n_tasks, n_tasks), 对称矩阵
    """
    device = distributions_tensor.device
    n_tasks, n_experts = distributions_tensor.shape
    
    # 1. 归一化为概率分布
    eps = 1e-8
    distributions_tensor = distributions_tensor / (distributions_tensor.sum(dim=1, keepdim=True) + eps)
    
    # 2. 使用广播计算所有任务对的平均分布
    # P_i: (n_tasks, 1, n_experts), P_j: (1, n_tasks, n_experts)
    P_i = distributions_tensor.unsqueeze(1)  
    P_j = distributions_tensor.unsqueeze(0)  
    M = 0.5 * (P_i + P_j)  # shape: (n_tasks, n_tasks, n_experts)
    
    # 3. 批量计算KL散度 - 完全向量化
    # KL(P_i || M) for all pairs
    log_ratio_i = torch.log((P_i + eps) / (M + eps))
    kl_i_m = torch.sum(P_i * log_ratio_i, dim=2)  # (n_tasks, n_tasks)
    
    # KL(P_j || M) for all pairs  
    log_ratio_j = torch.log((P_j + eps) / (M + eps))
    kl_j_m = torch.sum(P_j * log_ratio_j, dim=2)  # (n_tasks, n_tasks)
    
    # 4. JS散度矩阵
    js_matrix = 0.5 * (kl_i_m + kl_j_m)
    
    return js_matrix


def wasserstein_distance_batch_gpu(distributions_tensor):
    """
    GPU批量计算Wasserstein距离矩阵 - 1D分布的高效实现
    
    Args:
        distributions_tensor: shape (n_tasks, n_experts), GPU张量
    
    Returns:
        wasserstein_matrix: shape (n_tasks, n_tasks), 对称矩阵
    """
    device = distributions_tensor.device
    n_tasks, n_experts = distributions_tensor.shape
    eps = 1e-8
    
    # 1. 归一化为概率分布
    distributions_tensor = distributions_tensor / (distributions_tensor.sum(dim=1, keepdim=True) + eps)
    
    # 2. 计算累积分布函数 (CDF)
    cdf_tensor = torch.cumsum(distributions_tensor, dim=1)  # (n_tasks, n_experts)
    
    # 3. 使用广播计算所有CDF对之间的L1距离
    cdf_i = cdf_tensor.unsqueeze(1)  # (n_tasks, 1, n_experts)
    cdf_j = cdf_tensor.unsqueeze(0)  # (1, n_tasks, n_experts)
    
    # Wasserstein距离 = 累积分布差异的L1范数
    wasserstein_matrix = torch.sum(torch.abs(cdf_i - cdf_j), dim=2)
    
    return wasserstein_matrix


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

# ====== UniZero-MT 归一化所需基准分数 (26 Atari100k task_id 对应索引) ======
# 原始的 RANDOM_SCORES 和 HUMAN_SCORES


# global BENCHMARK_NAME
# # BENCHMARK_NAME = "atari"
# BENCHMARK_NAME = "dmc" # TODO
# if BENCHMARK_NAME == "atari":
#     RANDOM_SCORES = np.array([
#         227.8, 5.8, 222.4, 210.0, 14.2, 2360.0, 0.1, 1.7, 811.0, 10780.5,
#         152.1, 0.0, 65.2, 257.6, 1027.0, 29.0, 52.0, 1598.0, 258.5, 307.3,
#         -20.7, 24.9, 163.9, 11.5, 68.4, 533.4
#     ])
#     HUMAN_SCORES = np.array([
#         7127.7, 1719.5, 742.0, 8503.3, 753.1, 37187.5, 12.1, 30.5, 7387.8, 35829.4,
#         1971.0, 29.6, 4334.7, 2412.5, 30826.4, 302.8, 3035.0, 2665.5, 22736.3, 6951.6,
#         14.6, 69571.3, 13455.0, 7845.0, 42054.7, 11693.2
#     ])
# elif BENCHMARK_NAME == "dmc":
#     RANDOM_SCORES = np.array([0]*26)
#     HUMAN_SCORES = np.array([1000]*26)


# # 新顺序对应的原始索引列表
# # 新顺序： [Pong, MsPacman, Seaquest, Boxing, Alien, ChopperCommand, Hero, RoadRunner,
# #            Amidar, Assault, Asterix, BankHeist, BattleZone, CrazyClimber, DemonAttack,
# #            Freeway, Frostbite, Gopher, Jamesbond, Kangaroo, Krull, KungFuMaster,
# #            PrivateEye, UpNDown, Qbert, Breakout]
# # 映射为原始数组中的索引（注意：索引均从0开始）
# new_order = [
#     20,  # Pong
#     19,  # MsPacman
#     24,  # Seaquest
#     6,   # Boxing
#     0,   # Alien
#     8,   # ChopperCommand
#     14,  # Hero
#     23,  # RoadRunner
#     1,   # Amidar
#     2,   # Assault
#     3,   # Asterix
#     4,   # BankHeist
#     5,   # BattleZone
#     9,   # CrazyClimber
#     10,  # DemonAttack
#     11,  # Freeway
#     12,  # Frostbite
#     13,  # Gopher
#     15,  # Jamesbond
#     16,  # Kangaroo
#     17,  # Krull
#     18,  # KungFuMaster
#     21,  # PrivateEye
#     25,  # UpNDown
#     22,  # Qbert
#     7    # Breakout
# ]

# # 根据 new_order 生成新的数组
# new_RANDOM_SCORES = RANDOM_SCORES[new_order]
# new_HUMAN_SCORES = HUMAN_SCORES[new_order]

# # 查看重排后的结果
# print("重排后的 RANDOM_SCORES:")
# print(new_RANDOM_SCORES)
# print("\n重排后的 HUMAN_SCORES:")
# print(new_HUMAN_SCORES)

# 保存最近一次评估回报：{task_id: eval_episode_return_mean}
from collections import defaultdict
GLOBAL_EVAL_RETURNS: dict[int, float] = defaultdict(lambda: None)
def compute_unizero_mt_normalized_stats(
        eval_returns: dict[int, float]
) -> tuple[Optional[float], Optional[float]]:
    """
    由 eval_returns 计算 Human-Normalized Mean 和 Median。
    若暂无样本，返回 (None, None)。
    """
    normalized = []
    for tid, ret in eval_returns.items():
        if ret is None:
            continue
        denom = new_HUMAN_SCORES[tid] - new_RANDOM_SCORES[tid]
        if denom == 0:
            continue
        normalized.append((ret - new_RANDOM_SCORES[tid]) / denom)

    if not normalized:
        return None, None
    arr = np.asarray(normalized, dtype=np.float32)
    return float(arr.mean()), float(np.median(arr))

# 设置超时时间 (秒)
TIMEOUT = 12000  # 例如200分钟

timer = EasyTimer()


def safe_eval(
        evaluator: Evaluator,
        learner: BaseLearner,
        collector: Collector,
        rank: int,
        world_size: int
) -> Tuple[Optional[bool], Optional[float]]:
    """
    Safely执行评估任务，避免超时。

    Args:
        evaluator (Evaluator): 评估器实例。
        learner (BaseLearner): 学习器实例。
        collector (Collector): 数据收集器实例。
        rank (int): 当前进程的rank。
        world_size (int): 总进程数。

    Returns:
        Tuple[Optional[bool], Optional[float]]: 如果评估成功，返回停止标志和奖励，否则返回（None, None）。
    """
    try:
        print(f"=========评估开始 Rank {rank}/{world_size}===========")
        # 重置 stop_event，确保每次评估前都处于未设置状态
        evaluator.stop_event.clear()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # 提交评估任务
            future = executor.submit(evaluator.eval, learner.save_checkpoint, learner.train_iter, collector.envstep)
            try:
                stop, reward = future.result(timeout=TIMEOUT)
            except concurrent.futures.TimeoutError:
                # 超时，设置 stop_event
                evaluator.stop_event.set()
                print(f"评估操作在 Rank {rank}/{world_size} 上超时，耗时 {TIMEOUT} 秒。")
                return None, None

        print(f"======评估结束 Rank {rank}/{world_size}======")
        return stop, reward
    except Exception as e:
        print(f"Rank {rank}/{world_size} 评估过程中发生错误: {e}")
        return None, None


def allocate_batch_size(
        cfgs: List[dict],
        game_buffers,
        alpha: float = 1.0,
        clip_scale: int = 1
) -> List[int]:
    """
    根据不同任务的收集剧集数反比分配batch_size，并动态调整batch_size范围以提高训练稳定性和效率。

    Args:
        cfgs (List[dict]): 每个任务的配置列表。
        game_buffers (List[GameBuffer]): 每个任务的重放缓冲区实例列表。
        alpha (float, optional): 控制反比程度的超参数。默认为1.0。
        clip_scale (int, optional): 动态调整的clip比例。默认为1。

    Returns:
        List[int]: 分配后的batch_size列表。
    """
    # 提取每个任务的 collected episodes 数量
    buffer_num_of_collected_episodes = [buffer.num_of_collected_episodes for buffer in game_buffers]

    # 获取当前的 world_size 和 rank
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()

    # 收集所有 rank 的 collected episodes 列表
    all_task_num_of_collected_episodes = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(all_task_num_of_collected_episodes, buffer_num_of_collected_episodes)

    # 将所有 rank 的 collected episodes 合并为一个大列表
    all_task_num_of_collected_episodes = [
        episode for sublist in all_task_num_of_collected_episodes for episode in sublist
    ]
    if rank == 0:
        print(f'所有任务的 collected episodes: {all_task_num_of_collected_episodes}')

    # 计算每个任务的反比权重
    inv_episodes = np.array([1.0 / (episodes + 1) for episodes in all_task_num_of_collected_episodes])
    inv_sum = np.sum(inv_episodes)

    # 计算总的batch_size (所有任务 cfg.policy.batch_size 的和)
    total_batch_size = cfgs[0].policy.total_batch_size

    # 动态调整的部分：最小和最大的 batch_size 范围
    avg_batch_size = total_batch_size / world_size
    min_batch_size = avg_batch_size / clip_scale
    max_batch_size = avg_batch_size * clip_scale

    # 动态调整 alpha，让 batch_size 的变化更加平滑
    task_weights = (inv_episodes / inv_sum) ** alpha
    batch_sizes = total_batch_size * task_weights

    # 控制 batch_size 在 [min_batch_size, max_batch_size] 之间
    batch_sizes = np.clip(batch_sizes, min_batch_size, max_batch_size)

    # 确保 batch_size 是整数
    batch_sizes = [int(size) for size in batch_sizes]

    return batch_sizes

import numpy as np


def symlog(x: torch.Tensor) -> torch.Tensor:
    """
    Symlog 归一化，减少目标值的幅度差异。
    symlog(x) = sign(x) * log(|x| + 1)
    """
    return torch.sign(x) * torch.log(torch.abs(x) + 1)

def inv_symlog(x: torch.Tensor) -> torch.Tensor:
    """
    Symlog 的逆操作，用于恢复原始值。
    inv_symlog(x) = sign(x) * (exp(|x|) - 1)
    """
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)

# 全局最大值和最小值（用于 "run-max-min"）
GLOBAL_MAX = -float('inf')
GLOBAL_MIN = float('inf')

def compute_task_weights(
    task_returns: dict,
    option: str = "symlog",
    epsilon: float = 1e-6,
    temperature: float = 1.0,
    use_softmax: bool = False,  # 是否使用 Softmax
    reverse: bool = False,  # 正比 (False) 或反比 (True)
    clip_min: float = 1e-2,  # 权重的最小值
    clip_max: float = 1.0,  # 权重的最大值
) -> dict:
    """
    改进后的任务权重计算函数，支持多种标准化方式、Softmax 和正反比权重计算，并增加权重范围裁剪功能。

    Args:
        task_returns (dict): 每个任务的字典，键为 task_id，值为评估奖励或损失。
        option (str): 标准化方式，可选值为 "symlog", "max-min", "run-max-min", "rank", "none"。
        epsilon (float): 避免分母为零的小值。
        temperature (float): 控制权重分布的温度系数。
        use_softmax (bool): 是否使用 Softmax 进行权重分配。
        reverse (bool): 若为 True，权重与值反比；若为 False，权重与值正比。
        clip_min (float): 权重的最小值，用于裁剪。
        clip_max (float): 权重的最大值，用于裁剪。

    Returns:
        dict: 每个任务的权重，键为 task_id，值为归一化后的权重。
    """
    import torch
    import torch.nn.functional as F

    global GLOBAL_MAX, GLOBAL_MIN

    # 如果输入为空字典，直接返回空结果
    if not task_returns:
        return {}

    # Step 1: 对 task_returns 的值构造张量
    task_ids = list(task_returns.keys())
    returns_tensor = torch.tensor(list(task_returns.values()), dtype=torch.float32)

    if option == "symlog":
        # 使用 symlog 标准化
        scaled_returns = symlog(returns_tensor)
    elif option == "max-min":
        # 使用最大最小值归一化
        max_reward = returns_tensor.max().item()
        min_reward = returns_tensor.min().item()
        scaled_returns = (returns_tensor - min_reward) / (max_reward - min_reward + epsilon)
    elif option == "run-max-min":
        # 使用全局最大最小值归一化
        GLOBAL_MAX = max(GLOBAL_MAX, returns_tensor.max().item())
        GLOBAL_MIN = min(GLOBAL_MIN, returns_tensor.min().item())
        scaled_returns = (returns_tensor - GLOBAL_MIN) / (GLOBAL_MAX - GLOBAL_MIN + epsilon)
    elif option == "rank":
        # 使用 rank 标准化
        # Rank 是基于值大小的排名，1 表示最小值，越大排名越高
        sorted_indices = torch.argsort(returns_tensor)
        scaled_returns = torch.empty_like(returns_tensor)
        rank_values = torch.arange(1, len(returns_tensor) + 1, dtype=torch.float32)  # 1 到 N
        scaled_returns[sorted_indices] = rank_values
    elif option == "none":
        # 不进行标准化
        scaled_returns = returns_tensor
    else:
        raise ValueError(f"Unsupported option: {option}")

    # Step 2: 根据 reverse 确定权重是正比还是反比
    if not reverse:
        # 正比：权重与值正相关
        raw_weights = scaled_returns
    else:
        # 反比：权重与值负相关
        # 避免 scaled_returns 为负数或零
        scaled_returns = torch.clamp(scaled_returns, min=epsilon)
        raw_weights = 1.0 / scaled_returns

    # Step 3: 根据是否使用 Softmax 进行权重计算
    if use_softmax:
        # 使用 Softmax 进行权重分配
        beta = 1.0 / max(temperature, epsilon)  # 确保 temperature 不为零
        logits = -beta * raw_weights
        softmax_weights = F.softmax(logits, dim=0).numpy()
        weights = dict(zip(task_ids, softmax_weights))
    else:
        # 不使用 Softmax，直接计算权重
        # 温度缩放
        scaled_weights = raw_weights ** (1 / max(temperature, epsilon))  # 确保温度不为零

        # 归一化权重
        total_weight = scaled_weights.sum()
        normalized_weights = scaled_weights / total_weight

        # 转换为字典
        weights = dict(zip(task_ids, normalized_weights.numpy()))

    # Step 4: Clip 权重范围
    for task_id in weights:
        weights[task_id] = max(min(weights[task_id], clip_max), clip_min)

    return weights

a=1
def train_unizero_multitask_segment_ddp(
        input_cfg_list: List[Tuple[int, Tuple[dict, dict]]],
        seed: int = 0,
        model: Optional[torch.nn.Module] = None,
        model_path: Optional[str] = None,
        max_train_iter: Optional[int] = int(1e10),
        max_env_step: Optional[int] = int(1e10),
        benchmark_name: str = "atari",
        finetune_components=[],
        cal_moe_profile: bool = False  # 新增：控制MOE性能监控的开关
) -> 'Policy':
    """
    Overview:
        UniZero的训练入口，旨在通过解决MuZero类算法在需要捕捉长期依赖环境中的局限性，提高强化学习代理的规划能力。
        详细信息请参阅 https://arxiv.org/abs/2406.10667。

    Args:
        - input_cfg_list (:obj:`List[Tuple[int, Tuple[dict, dict]]]`): 不同任务的配置列表。
        - seed (:obj:`int`): 随机种子。
        - model (:obj:`Optional[torch.nn.Module]`): torch.nn.Module实例。
        - model_path (:obj:`Optional[str]`): 预训练模型路径，应指向预训练模型的ckpt文件。
        - max_train_iter (:obj:`Optional[int]`): 训练中的最大策略更新迭代次数。
        - max_env_step (:obj:`Optional[int]`): 最大收集环境交互步数。

    Returns:
        - policy (:obj:`Policy`): 收敛的策略。
    """

     # ---------------------------------------------------------------
    # ====== UniZero-MT 需要用到的基准分数（与 26 个 Atari100k 任务 id 一一对应）======
    #   原始的 RANDOM_SCORES 和 HUMAN_SCORES
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
        # RANDOM_SCORES = np.array([0]*26)
        # HUMAN_SCORES = np.array([1000]*26)
        RANDOM_SCORES = np.zeros(26)
        HUMAN_SCORES  = np.ones(26) * 1000
    else:
        raise ValueError(f"Unsupported BENCHMARK_NAME: {BENCHMARK_NAME}")

    # 新顺序对应的原始索引列表
    # 新顺序： [Pong, MsPacman, Seaquest, Boxing, Alien, ChopperCommand, Hero, RoadRunner,
    #            Amidar, Assault, Asterix, BankHeist, BattleZone, CrazyClimber, DemonAttack,
    #            Freeway, Frostbite, Gopher, Jamesbond, Kangaroo, Krull, KungFuMaster,
    #            PrivateEye, UpNDown, Qbert, Breakout]
    # 映射为原始数组中的索引（注意：索引均从0开始）
    new_order = [
        20,  # Pong
        19,  # MsPacman
        24,  # Seaquest
        6,   # Boxing
        0,   # Alien
        8,   # ChopperCommand
        14,  # Hero
        23,  # RoadRunner
        1,   # Amidar
        2,   # Assault
        3,   # Asterix
        4,   # BankHeist
        5,   # BattleZone
        9,   # CrazyClimber
        10,  # DemonAttack
        11,  # Freeway
        12,  # Frostbite
        13,  # Gopher
        15,  # Jamesbond
        16,  # Kangaroo
        17,  # Krull
        18,  # KungFuMaster
        21,  # PrivateEye
        25,  # UpNDown
        22,  # Qbert
        7    # Breakout
    ]
    global new_RANDOM_SCORES, new_HUMAN_SCORES
    # 根据 new_order 生成新的数组
    new_RANDOM_SCORES = RANDOM_SCORES[new_order]
    new_HUMAN_SCORES = HUMAN_SCORES[new_order]
    # 查看重排后的结果
    print("重排后的 RANDOM_SCORES:")
    print(new_RANDOM_SCORES)
    print("\n重排后的 HUMAN_SCORES:")
    print(new_HUMAN_SCORES)
    # ---------------------------------------------------------------

    # 初始化温度调度器
    initial_temperature = 10.0
    final_temperature = 1.0
    threshold_steps = int(1e4)  # 训练步数达到 10k 时，温度降至 1.0
    temperature_scheduler = TemperatureScheduler(
        initial_temp=initial_temperature,
        final_temp=final_temperature,
        threshold_steps=threshold_steps,
        mode='linear'  # 或 'exponential'
    )

    # 获取当前进程的rank和总进程数
    rank = get_rank()
    world_size = get_world_size()
    
    # 初始化MOE统计性能监控
    moe_profiler = None
    if cal_moe_profile and LineProfiler is not None:
        moe_profiler = LineProfiler()
        moe_profiler.add_function(collect_and_log_moe_statistics)
        moe_profiler.enable_by_count()
        print(f"Rank {rank}: MOE统计性能监控已启用")
    elif cal_moe_profile and LineProfiler is None:
        print(f"Rank {rank}: 警告: line_profiler未安装，无法启用MOE性能监控")

    # 任务划分
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

    # 确保至少有一个任务
    if len(tasks_for_this_rank) == 0:
        logging.warning(f"Rank {rank}: 未分配任务，继续执行。")
        # 初始化空列表以避免后续代码报错
        cfgs, game_buffers, collector_envs, evaluator_envs, collectors, evaluators = [], [], [], [], [], []
    else:
        print(f"Rank {rank}/{world_size}, 处理任务 {start_idx} 到 {end_idx - 1}")

    cfgs = []
    game_buffers = []
    collector_envs = []
    evaluator_envs = []
    collectors = []
    evaluators = []

    if tasks_for_this_rank:
        # 使用第一个任务的配置创建共享的policy
        task_id, [cfg, create_cfg] = tasks_for_this_rank[0]

        for config in tasks_for_this_rank:
            config[1][0].policy.task_num = tasks_per_rank

        # 确保指定的策略类型受支持
        assert create_cfg.policy.type in ['unizero_multitask',
                                          'sampled_unizero_multitask'], "train_unizero entry 目前仅支持 'unizero_multitask'"

        if create_cfg.policy.type == 'unizero_multitask':
            from lzero.mcts import UniZeroGameBuffer as GameBuffer
        if create_cfg.policy.type == 'sampled_unizero_multitask':
            from lzero.mcts import SampledUniZeroGameBuffer as GameBuffer


        # 根据CUDA可用性设置设备
        cfg.policy.device = cfg.policy.model.world_model_cfg.device if torch.cuda.is_available() else 'cpu'
        logging.info(f'配置的设备: {cfg.policy.device}')

        # 编译配置
        cfg = compile_config(cfg, seed=seed, env=None, auto=True, create_cfg=create_cfg, save_cfg=True)
        # 创建共享的policy
        
        # print("===============================")
        # exit()
        # 创建TensorBoard日志记录器
        log_dir = os.path.join('./{}/log'.format(cfg.exp_name), f'serial_rank_{rank}')
        # global tb_logger
        tb_logger = SummaryWriter(log_dir)
        
        cfg.policy.logger=tb_logger
        
        policy = create_policy(cfg.policy, model=model, enable_field=['learn', 'collect', 'eval']) # MOE

        # 加载预训练模型（如果提供）
        if model_path is not None:
            logging.info(f'开始加载模型: {model_path}')
            policy.learn_mode.load_state_dict(torch.load(model_path, map_location=cfg.policy.device),finetune_components=finetune_components)
            logging.info(f'完成加载模型: {model_path}')
       

        # 创建共享的learner  #todo: cfg.policy.learn.learner.hook.log_show_after_iter
        cfg.policy.learn.learner.hook.log_show_after_iter=1
        learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)

        policy_config = cfg.policy

        # 处理当前进程分配到的每个任务
        for local_task_id, (task_id, [cfg, create_cfg]) in enumerate(tasks_for_this_rank):
            # 设置每个任务的随机种子
            cfg.policy.device = 'cuda' if cfg.policy.cuda and torch.cuda.is_available() else 'cpu'
            cfg = compile_config(cfg, seed=seed + task_id, env=None, auto=True, create_cfg=create_cfg, save_cfg=True)
            policy_config = cfg.policy
            policy.collect_mode.get_attribute('cfg').n_episode = policy_config.n_episode
            policy.eval_mode.get_attribute('cfg').n_episode = policy_config.n_episode

            # 创建环境
            env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)
            collector_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_env_cfg])
            evaluator_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])
            collector_env.seed(cfg.seed + task_id)
            evaluator_env.seed(cfg.seed + task_id, dynamic_seed=False)
            set_pkg_seed(cfg.seed + task_id, use_cuda=cfg.policy.cuda)

            # 创建不同的game buffer、collector和evaluator
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
            replay_buffer.batch_size = cfg.policy.batch_size[task_id]

            game_buffers.append(replay_buffer)
            collector_envs.append(collector_env)
            evaluator_envs.append(evaluator_env)
            collectors.append(collector)
            evaluators.append(evaluator)

    # 调用learner的before_run钩子
    learner.call_hook('before_run')
    value_priority_tasks = {}
    
    buffer_reanalyze_count = 0
    train_epoch = 0
    reanalyze_batch_size = cfg.policy.reanalyze_batch_size
    update_per_collect = cfg.policy.update_per_collect

    # use_task_exploitation_weight = cfg.policy.use_task_exploitation_weight
    task_exploitation_weight = None

    # 创建任务奖励字典
    task_returns = {}  # {task_id: reward}

    while True:
        # 动态调整batch_size
        if cfg.policy.allocated_batch_sizes:
            clip_scale = np.clip(1 + (3 * train_epoch / 1000), 1, 4)
            allocated_batch_sizes = allocate_batch_size(cfgs, game_buffers, alpha=1.0, clip_scale=clip_scale)
            if rank == 0:
                print("分配后的 batch_sizes: ", allocated_batch_sizes)
            for idx, (cfg, collector, evaluator, replay_buffer) in enumerate(
                    zip(cfgs, collectors, evaluators, game_buffers)):
                cfg.policy.batch_size = allocated_batch_sizes
                policy._cfg.batch_size = allocated_batch_sizes

        # 对于当前进程的每个任务，进行数据收集和评估
        for idx, (cfg, collector, evaluator, replay_buffer) in enumerate(
                zip(cfgs, collectors, evaluators, game_buffers)):

            # 记录缓冲区内存使用情况
            log_buffer_memory_usage(learner.train_iter, replay_buffer, tb_logger, cfg.policy.task_id)

            collect_kwargs = {
                'temperature': visit_count_temperature(
                    policy_config.manual_temperature_decay,
                    policy_config.fixed_temperature_value,
                    policy_config.threshold_training_steps_for_final_temperature,
                    trained_steps=learner.train_iter
                ),
                'epsilon': 0.0  # 默认的epsilon值
            }

            if policy_config.eps.eps_greedy_exploration_in_collect:
                epsilon_greedy_fn = get_epsilon_greedy_fn(
                    start=policy_config.eps.start,
                    end=policy_config.eps.end,
                    decay=policy_config.eps.decay,
                    type_=policy_config.eps.type
                )
                collect_kwargs['epsilon'] = epsilon_greedy_fn(collector.envstep)

            # 判断是否需要进行评估
            # if learner.train_iter == 0 or evaluator.should_eval(learner.train_iter):
            if learner.train_iter > 10 and learner.train_iter % cfg.policy.eval_freq == 0 :
            # if learner.train_iter > 10 and evaluator.should_eval(learner.train_iter): # only for debug
            # if evaluator.should_eval(learner.train_iter):
                print('=' * 20)
                
                print(f'Rank {rank} 评估任务_id: {cfg.policy.task_id}...')

                # =========TODO=========
                evaluator._policy.reset(reset_init_data=True, task_id=cfg.policy.task_id)

                # 执行安全评估
                stop, reward = safe_eval(evaluator, learner, collector, rank, world_size)
                # 判断评估是否成功
                if stop is None or reward is None:
                    print(f"Rank {rank} 在评估过程中遇到问题，继续训练...")
                    task_returns[cfg.policy.task_id] = float('inf')  # 如果评估失败，将任务难度设为最大值
                else:
                    # 确保从评估结果中提取 `eval_episode_return_mean` 作为奖励值
                    try:
                        eval_mean_reward = reward.get('eval_episode_return_mean', float('inf'))
                        print(f"任务 {cfg.policy.task_id} 的评估奖励: {eval_mean_reward}")
                        task_returns[cfg.policy.task_id] = eval_mean_reward
                    except Exception as e:
                        print(f"提取评估奖励时发生错误: {e}")
                        task_returns[cfg.policy.task_id] = float('inf')  # 出现问题时，将奖励设为最大值


            print('=' * 20)
            print(f'开始收集 Rank {rank} 的任务_id: {cfg.policy.task_id}...')
            print(f'Rank {rank}: cfg.policy.task_id={cfg.policy.task_id} ')


            # while replay_buffer.get_num_of_transitions() < cfg.policy.batch_size[cfg.policy.task_id]:
            # for ddp training, 避免后面 train 时replay buffer中样本小于batch size 导致ddp hangs

            # 在每次收集之前重置初始数据，这对于多任务设置非常重要
            collector._policy.reset(reset_init_data=True, task_id=cfg.policy.task_id)
            # 收集数据
            new_data = collector.collect(train_iter=learner.train_iter, policy_kwargs=collect_kwargs)

            # 更新重放缓冲区
            replay_buffer.push_game_segments(new_data)
            replay_buffer.remove_oldest_data_to_fit()


            # # ===== only for debug =====
            # if train_epoch > 2:
            #     with timer:
            #         replay_buffer.reanalyze_buffer(2, policy)
            #     buffer_reanalyze_count += 1
            #     logging.info(f'缓冲区重新分析次数: {buffer_reanalyze_count}')
            #     logging.info(f'缓冲区重新分析耗时: {timer.value}') 
            # # ===== only for debug =====


            # 周期性地重新分析缓冲区
            if cfg.policy.buffer_reanalyze_freq >= 1:
                reanalyze_interval = update_per_collect // cfg.policy.buffer_reanalyze_freq
            else:
                if train_epoch > 0 and train_epoch % int(1 / cfg.policy.buffer_reanalyze_freq) == 0 and \
                        replay_buffer.get_num_of_transitions() // cfg.policy.num_unroll_steps > int(
                    reanalyze_batch_size / cfg.policy.reanalyze_partition):
                    with timer:
                        replay_buffer.reanalyze_buffer(reanalyze_batch_size, policy)
                    buffer_reanalyze_count += 1
                    logging.info(f'缓冲区重新分析次数: {buffer_reanalyze_count}')
                    logging.info(f'缓冲区重新分析耗时: {timer.value}')

            # 数据收集结束后添加日志
            logging.info(f'Rank {rank}: 完成任务 {cfg.policy.task_id} 的数据收集')

        # 检查是否有足够的数据进行训练
        not_enough_data = any(
            replay_buffer.get_num_of_transitions() < cfgs[0].policy.total_batch_size / world_size
            for replay_buffer in game_buffers
        )

        print(f"not_enough_data:{not_enough_data}")
        # 获取当前温度
        current_temperature_task_weight = temperature_scheduler.get_temperature(learner.train_iter)
        
        # if learner.train_iter == 0 or learner.train_iter % cfg.policy.eval_freq == 0 :
        if learner.train_iter > 10 and learner.train_iter % cfg.policy.eval_freq == 0 :
        
            # 计算任务权重
            try:
                # 汇聚任务奖励
                dist.barrier()
                # if cfg.policy.task_complexity_weight:
                all_task_returns = [None for _ in range(world_size)]
                dist.all_gather_object(all_task_returns, task_returns)
                # 合并任务奖励
                merged_task_returns = {}
                for returns in all_task_returns:
                    if returns:
                        merged_task_returns.update(returns)
                
                logging.warning(f"Rank {rank}: merged_task_returns: {merged_task_returns}")

                # 计算全局任务权重
                task_weights = compute_task_weights(merged_task_returns, temperature=current_temperature_task_weight)
                
                # ---------- 维护 UniZero-MT 全局评估结果 ----------
                for tid, ret in merged_task_returns.items():
                    GLOBAL_EVAL_RETURNS[tid] = ret   # solved 的任务同样更新

                # 计算 Human-Normalized Mean / Median
                uni_mean, uni_median = compute_unizero_mt_normalized_stats(GLOBAL_EVAL_RETURNS)

                if uni_mean is not None:            # 至少评估过 1 个任务
                    if rank == 0:                   # 仅在 rank0 写 TensorBoard，防止重复
                        tb_logger.add_scalar('UniZero-MT/NormalizedMean',   uni_mean,   global_step=learner.train_iter)
                        tb_logger.add_scalar('UniZero-MT/NormalizedMedian', uni_median, global_step=learner.train_iter)
                    logging.info(f"Rank {rank}: UniZero-MT Norm Mean={uni_mean:.4f}, Median={uni_median:.4f}")
                else:
                    logging.info(f"Rank {rank}: 暂无数据计算 UniZero-MT 归一化指标")

                # 同步任务权重
                dist.broadcast_object_list([task_weights], src=0)
                # print(f"rank{rank}, 全局任务权重 (按 task_id 排列): {task_weights}")
                # else:
                #     task_weights = None
            except Exception as e:
                logging.error(f'Rank {rank}: 同步任务权重失败，错误: {e}')
                break


        # ---------------- 采样完成，准备进入反向 ----------------
        # if dist.is_available() and dist.is_initialized():
        #     dist.barrier()                 # ★★★ 关键同步 ★★★

        # 学习策略
        if not not_enough_data:
            for i in range(update_per_collect):
                train_data_multi_task = []
                envstep_multi_task = 0
                for idx, (cfg, collector, replay_buffer) in enumerate(zip(cfgs, collectors, game_buffers)):
                    envstep_multi_task += collector.envstep
                    batch_size = cfg.policy.batch_size[cfg.policy.task_id]
                    if replay_buffer.get_num_of_transitions() > batch_size:
                        if cfg.policy.buffer_reanalyze_freq >= 1:
                            if i % reanalyze_interval == 0 and \
                                    replay_buffer.get_num_of_transitions() // cfg.policy.num_unroll_steps > int(
                                reanalyze_batch_size / cfg.policy.reanalyze_partition):
                                with timer:
                                    replay_buffer.reanalyze_buffer(reanalyze_batch_size, policy)
                                buffer_reanalyze_count += 1
                                logging.info(f'缓冲区重新分析次数: {buffer_reanalyze_count}')
                                logging.info(f'缓冲区重新分析耗时: {timer.value}')

                        train_data = replay_buffer.sample(batch_size, policy)
                        train_data.append(cfg.policy.task_id)  # 追加task_id以区分任务
                        train_data_multi_task.append(train_data)
                    else:
                        logging.warning(
                            f'重放缓冲区中的数据不足以采样mini-batch: '
                            f'batch_size: {batch_size}, replay_buffer: {replay_buffer}'
                        )
                        break

                if train_data_multi_task:
                    # learn_kwargs = {'task_exploitation_weight':task_exploitation_weight, 'task_weights':task_weights, }
                    # learn_kwargs = {'task_weights': task_weights, }
                    # learn_kwargs = {'task_weights':task_exploitation_weight}

                    learn_kwargs = {'task_weights': None,}
                    # logging.info(f'Rank {rank}: iter {i} one learn step start')

                    # 在训练时，DDP会自动同步梯度和参数
                    log_vars = learner.train(train_data_multi_task, envstep_multi_task, policy_kwargs=learn_kwargs)
                    
                    print("训练结束！！！")
                    
                    # +++++++++++++++++++++++++++++++++ MOE专家选择统计记录 +++++++++++++++++++++++++++++++++
                    if cfg.policy.model.world_model_cfg.multiplication_moe_in_transformer:
                        # 控制MoE统计记录频率
                        moe_log_interval = getattr(cfg.policy, 'moe_log_interval', 500)  # 默认每500个iter记录一次
                        
                        if learner.train_iter % moe_log_interval == 0:
                            # # 性能监控开始
                            # if cal_moe_profile:
                            #     import time
                            #     moe_start_time = time.perf_counter()
                            
                            collect_and_log_moe_statistics(policy, tb_logger, learner.train_iter, world_size, rank)
                            
                            if rank == 0:  # 只在rank 0打印日志
                                print(f"MoE统计已记录 (train_iter={learner.train_iter})")
                        
                        # global a
                        # a+=1
                        # 性能监控结束
                        if cal_moe_profile :
                            
                            if moe_profiler is not None:
                                try:
                                    # 禁用profiler
                                    moe_profiler.disable_by_count()
                                    
                                    # 生成性能分析报告文件名
                                    profile_filename = f'moe_profile_rank{rank}_train{learner.train_iter}.txt'
                                    profile_path = os.path.join(cfg.exp_name, 'profile', profile_filename)
                                    
                                    # 确保目录存在
                                    os.makedirs(os.path.dirname(profile_path), exist_ok=True)
                                    
                                    # 保存性能分析结果到文件
                                    with open(profile_path, 'w') as f:
                                        moe_profiler.print_stats(stream=f)
                                    
                                    print(f"Rank {rank}: MOE性能分析结果已保存到 {profile_path}")
                                    
                                    # 也输出到控制台（可选，用于调试）
                                    if rank == 0:  # 只在rank 0输出到控制台，避免混乱
                                        print(f"\n=== Rank {rank}: MOE性能分析摘要 ===")
                                        moe_profiler.print_stats()
                                        print("=" * 50)
                                        
                                except Exception as e:
                                    print(f"Rank {rank}: 保存MOE性能分析失败: {e}")
                            
                            
                            
                            # moe_end_time = time.perf_counter()
                            # moe_elapsed = (moe_end_time - moe_start_time) * 1000  # 转换为毫秒
                            
                            # 记录性能指标
                            # tb_logger.add_scalar('Performance/MOE_Statistics_Time_ms', moe_elapsed, global_step=learner.train_iter)
                            
                            # 打印性能信息（每10次迭代打印一次，避免日志过多）
                            # if learner.train_iter % 10 == 0:
                            #     print(f"Rank {rank}: MOE统计耗时 {moe_elapsed:.2f}ms (train_iter={learner.train_iter})")
                        
                    # +++++++++++++++++++++++++++++++++ MOE专家选择统计记录结束 +++++++++++++++++++++++++++++++++
                    
                    # logging.error(f'Rank {rank}: one learn step done')

                    # 判断是否需要计算task_exploitation_weight
                    if i == 0:
                        # 计算任务权重
                        try:
                            dist.barrier()  # 等待所有进程同步
                            if cfg.policy.use_task_exploitation_weight: # use obs loss now, new polish
                                # 收集所有任务的 obs_loss
                                all_obs_loss = [None for _ in range(world_size)]
                                # 构建当前进程的任务 obs_loss 数据
                                merged_obs_loss_task = {}
                                for cfg, replay_buffer in zip(cfgs, game_buffers):
                                    task_id = cfg.policy.task_id
                                    if f'noreduce_obs_loss_task{task_id}' in log_vars[0]:
                                        merged_obs_loss_task[task_id] = log_vars[0][f'noreduce_obs_loss_task{task_id}']
                                # 汇聚所有进程的 obs_loss 数据
                                dist.all_gather_object(all_obs_loss, merged_obs_loss_task)
                                # 合并所有进程的 obs_loss 数据
                                global_obs_loss_task = {}
                                for obs_loss_task in all_obs_loss:
                                    if obs_loss_task:
                                        global_obs_loss_task.update(obs_loss_task)
                                # 计算全局任务权重
                                if global_obs_loss_task:
                                    task_exploitation_weight = compute_task_weights(
                                        global_obs_loss_task,
                                        option="rank",
                                        # temperature=current_temperature_task_weight # TODO
                                        temperature=1,
                                    )
                                    # 广播任务权重到所有进程
                                    dist.broadcast_object_list([task_exploitation_weight], src=0)
                                    print(f"rank{rank}, task_exploitation_weight (按 task_id 排列): {task_exploitation_weight}")
                                else:
                                    logging.warning(f"Rank {rank}: 未能计算全局 obs_loss 任务权重，obs_loss 数据为空。")
                                    task_exploitation_weight = None
                            else:
                                task_exploitation_weight = None
                            # 更新训练参数，使其包含计算后的任务权重
                            learn_kwargs['task_weight'] = task_exploitation_weight
                        except Exception as e:
                            logging.error(f'Rank {rank}: 同步任务权重失败，错误: {e}')
                            raise e  # 保留异常抛出，便于外部捕获和分析



                    if cfg.policy.use_priority:
                        for idx, (cfg, replay_buffer) in enumerate(zip(cfgs, game_buffers)):
                            # 更新任务特定的重放缓冲区优先级
                            task_id = cfg.policy.task_id
                            replay_buffer.update_priority(
                                train_data_multi_task[idx],
                                log_vars[0][f'value_priority_task{task_id}']
                            )

                            current_priorities = log_vars[0][f'value_priority_task{task_id}']
                            mean_priority = np.mean(current_priorities)
                            std_priority = np.std(current_priorities)

                            alpha = 0.1  # 平滑因子
                            if f'running_mean_priority_task{task_id}' not in value_priority_tasks:
                                value_priority_tasks[f'running_mean_priority_task{task_id}'] = mean_priority
                            else:
                                value_priority_tasks[f'running_mean_priority_task{task_id}'] = (
                                        alpha * mean_priority +
                                        (1 - alpha) * value_priority_tasks[f'running_mean_priority_task{task_id}']
                                )

                            # 使用运行均值计算归一化的优先级
                            running_mean_priority = value_priority_tasks[f'running_mean_priority_task{task_id}']
                            normalized_priorities = (current_priorities - running_mean_priority) / (std_priority + 1e-6)

                            # 如果需要，可以将归一化的优先级存储回重放缓冲区
                            # replay_buffer.update_priority(train_data_multi_task[idx], normalized_priorities)

                            # 记录优先级统计信息
                            if cfg.policy.print_task_priority_logs:
                                print(f"任务 {task_id} - 平均优先级: {mean_priority:.8f}, "
                                      f"运行平均优先级: {running_mean_priority:.8f}, "
                                      f"标准差: {std_priority:.8f}")

        train_epoch += 1
        policy.recompute_pos_emb_diff_and_clear_cache()

        # 同步所有Rank，确保所有Rank完成训练
        try:
            dist.barrier()
            logging.info(f'Rank {rank}: 通过训练后的同步障碍')
        except Exception as e:
            logging.error(f'Rank {rank}: 同步障碍失败，错误: {e}')
            break

        # 检查是否需要终止训练
        try:
            local_envsteps = [collector.envstep for collector in collectors]
            total_envsteps = [None for _ in range(world_size)]
            dist.all_gather_object(total_envsteps, local_envsteps)

            all_envsteps = torch.cat([torch.tensor(envsteps, device=cfg.policy.device) for envsteps in total_envsteps])
            max_envstep_reached = torch.all(all_envsteps >= max_env_step)

            # 收集所有进程的train_iter
            global_train_iter = torch.tensor([learner.train_iter], device=cfg.policy.device)
            all_train_iters = [torch.zeros_like(global_train_iter) for _ in range(world_size)]
            dist.all_gather(all_train_iters, global_train_iter)

            max_train_iter_reached = torch.any(torch.stack(all_train_iters) >= max_train_iter)

            if max_envstep_reached.item() or max_train_iter_reached.item():
                logging.info(f'Rank {rank}: 达到终止条件')
                dist.barrier()  # 确保所有进程同步
                break
        except Exception as e:
            logging.error(f'Rank {rank}: 终止检查失败，错误: {e}')
            break

    # 调用learner的after_run钩子
    learner.call_hook('after_run')
    
    # 保存MOE性能监控结果
    
    return policy
"""
Head Clip Manager - 与 Encoder-Clip 原理一致的动态 Head Clipping 实现

该模块提供了类似 Encoder-Clip 的动态 Head Clipping 功能：
1. 监控 head 输出（logits）的范围
2. 当超过阈值时，缩放整个 head 模块的所有权重
3. 支持 annealing（阈值从宽松逐渐变严格）
4. 支持多个 head 独立配置

与之前 Head Weight Scaling 的区别：
- 之前：在初始化时静态缩放一次
- 现在：在训练过程中动态监控和缩放（与 Encoder-Clip 一致）

"""

import logging
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from torch.nn.utils.convert_parameters import parameters_to_vector, vector_to_parameters

logging.getLogger().setLevel(logging.INFO)


@dataclass
class HeadClipConfig:
    """单个 Head 的 Clip 配置"""
    # 固定阈值（如果不启用 annealing）
    clip_threshold: float = 15.0

    # 是否启用 annealing
    use_annealing: bool = True

    # Annealing 配置
    anneal_type: str = 'cosine'  # 'cosine' 或 'linear'
    start_value: float = 30.0    # 初期宽松
    end_value: float = 10.0      # 后期严格
    anneal_steps: int = 500000

    def __post_init__(self):
        """验证配置"""
        if self.clip_threshold <= 0:
            raise ValueError(f"clip_threshold must be positive, got {self.clip_threshold}")
        if self.use_annealing:
            if self.start_value <= 0 or self.end_value <= 0:
                raise ValueError("start_value and end_value must be positive")
            if self.anneal_steps <= 0:
                raise ValueError("anneal_steps must be positive")
            if self.anneal_type not in ['cosine', 'linear']:
                raise ValueError(f"anneal_type must be 'cosine' or 'linear', got {self.anneal_type}")


class HeadClipManager:
    """
    Head Clip Manager - 动态监控和裁剪 Head 输出

    工作原理（与 Encoder-Clip 一致）：
    1. 每次训练迭代后，监控各个 head 的输出（logits）
    2. 计算 max(|logits|)
    3. 如果超过当前阈值，缩放整个 head 模块的权重
    4. 阈值支持 annealing（从宽松到严格）
    """

    def __init__(
        self,
        enabled: bool = True,
        enabled_heads: Optional[List[str]] = None,
        head_configs: Optional[Dict[str, HeadClipConfig]] = None,
        monitor_freq: int = 1,
        log_freq: int = 1000,
    ):
        """
        初始化 Head Clip Manager

        Args:
            enabled (bool): 是否启用 Head Clip
            enabled_heads (List[str], optional): 需要 clip 的 head 列表
                例如: ['policy', 'value', 'rewards']
                如果为 None，则不启用任何 head
            head_configs (Dict[str, HeadClipConfig], optional): 每个 head 的配置
                例如: {'policy': HeadClipConfig(...), 'value': HeadClipConfig(...)}
                如果某个 head 不在此字典中，使用默认配置
            monitor_freq (int): 监控频率（每隔多少个 iter 检查一次）
            log_freq (int): 日志打印频率
        """
        self.enabled = enabled
        self.enabled_heads = enabled_heads or []
        self.head_configs = head_configs or {}
        self.monitor_freq = monitor_freq
        self.log_freq = log_freq

        # 统计信息
        self.scaling_history = {head: [] for head in self.enabled_heads}
        self.iteration_count = 0

        # 日志映射
        self.logits_key_mapping = {
            'policy': 'logits_policy',
            'value': 'logits_value',
            'reward': 'logits_reward',
            'rewards': 'logits_reward',  # 兼容两种命名
            'observations': 'logits_observations',
        }

        self.head_module_mapping = {
            'policy': 'head_policy',
            'value': 'head_value',
            'reward': 'head_rewards',
            'rewards': 'head_rewards',
            'observations': 'head_observations',
        }

        if self.enabled and self.enabled_heads:
            logging.info("=" * 60)
            logging.info(">>> Head Clip Manager 已启用 <<<")
            logging.info(f"    Enabled heads: {self.enabled_heads}")
            logging.info(f"    Monitor freq: {self.monitor_freq}")
            logging.info(f"    Log freq: {self.log_freq}")
            for head_name in self.enabled_heads:
                config = self.get_head_config(head_name)
                if config.use_annealing:
                    logging.info(
                        f"    {head_name}: annealing {config.start_value:.1f} → {config.end_value:.1f} "
                        f"over {config.anneal_steps} steps ({config.anneal_type})"
                    )
                else:
                    logging.info(f"    {head_name}: fixed threshold = {config.clip_threshold:.1f}")
            logging.info("=" * 60)

    def get_head_config(self, head_name: str) -> HeadClipConfig:
        """
        获取指定 head 的配置，如果不存在则返回默认配置

        Args:
            head_name (str): Head 名称

        Returns:
            HeadClipConfig: 配置对象
        """
        if head_name in self.head_configs:
            return self.head_configs[head_name]
        else:
            # 返回默认配置
            return HeadClipConfig()

    def compute_current_threshold(
        self,
        head_name: str,
        train_iter: int
    ) -> float:
        """
        计算当前训练步的阈值（考虑 annealing）

        Args:
            head_name (str): Head 名称
            train_iter (int): 当前训练迭代次数

        Returns:
            float: 当前阈值
        """
        config = self.get_head_config(head_name)

        if not config.use_annealing:
            return config.clip_threshold

        # 计算 annealing 进度
        progress = min(1.0, train_iter / config.anneal_steps)

        if config.anneal_type == 'cosine':
            # 余弦调度: 从1平滑过渡到0
            cosine_progress = 0.5 * (1.0 + np.cos(np.pi * progress))
            current_value = config.end_value + \
                          (config.start_value - config.end_value) * cosine_progress
        else:  # 'linear'
            current_value = config.start_value * (1 - progress) + \
                          config.end_value * progress

        return current_value

    def apply_head_clip(
        self,
        world_model: nn.Module,
        losses: Any,  # LossWithIntermediateLosses
        train_iter: int
    ) -> Dict[str, Dict]:
        """
        应用 Head Clip（主函数）

        工作流程：
        1. 遍历所有启用的 head
        2. 获取 head 的输出（logits）
        3. 计算 max(|logits|)
        4. 如果超过当前阈值，缩放整个 head 模块

        Args:
            world_model (nn.Module): WorldModel 实例
            losses (LossWithIntermediateLosses): 包含中间输出的损失对象
            train_iter (int): 当前训练迭代次数

        Returns:
            Dict[str, Dict]: 每个 head 的缩放信息
                例如: {
                    'policy': {
                        'max_logits': 25.5,
                        'threshold': 15.0,
                        'scale_factor': 0.588,
                        'scaled': True
                    }
                }
        """
        if not self.enabled:
            return {}

        # 只在指定频率检查
        if train_iter % self.monitor_freq != 0:
            return {}

        self.iteration_count = train_iter
        results = {}

        for head_name in self.enabled_heads:
            # 1. 获取 logits
            logits = self._get_head_logits(losses, head_name)
            if logits is None:
                continue

            # 2. 计算当前阈值
            current_threshold = self.compute_current_threshold(head_name, train_iter)

            # 3. 计算 logits 的最大绝对值
            max_logits = logits.abs().max().item()

            # 4. 判断是否需要缩放
            scaled = False
            scale_factor = 1.0

            if max_logits > current_threshold:
                scale_factor = current_threshold / max_logits

                # 获取 head 模块
                head_module = self._get_head_module(world_model, head_name)
                if head_module is not None:
                    # 缩放整个 head 模块的所有权重
                    success = self._scale_module_weights(head_module, scale_factor)
                    scaled = success

                    if success:
                        # 记录历史
                        self.scaling_history[head_name].append({
                            'iteration': train_iter,
                            'max_logits': max_logits,
                            'threshold': current_threshold,
                            'scale_factor': scale_factor,
                        })

            # 5. 记录结果
            results[head_name] = {
                'max_logits': max_logits,
                'threshold': current_threshold,
                'scale_factor': scale_factor,
                'scaled': scaled,
            }

            # 6. 打印日志
            if scaled and train_iter % self.log_freq == 0:
                logging.info(
                    f"[Head-Clip] Iter {train_iter}: {head_name} head - "
                    f"max_logits={max_logits:.2f} > threshold={current_threshold:.2f}, "
                    f"scaling by {scale_factor:.4f}"
                )

        return results

    def _get_head_logits(
        self,
        losses: Any,
        head_name: str
    ) -> Optional[torch.Tensor]:
        """
        从 losses 对象中获取指定 head 的 logits

        Args:
            losses (LossWithIntermediateLosses): 损失对象
            head_name (str): Head 名称

        Returns:
            Optional[torch.Tensor]: logits 张量，如果未找到则返回 None
        """
        if not hasattr(losses, 'intermediate_losses'):
            return None

        logits_key = self.logits_key_mapping.get(head_name)
        if logits_key is None:
            return None

        return losses.intermediate_losses.get(logits_key)

    def _get_head_module(
        self,
        world_model: nn.Module,
        head_name: str
    ) -> Optional[nn.Module]:
        """
        获取指定 head 的模块

        Args:
            world_model (nn.Module): WorldModel 实例
            head_name (str): Head 名称

        Returns:
            Optional[nn.Module]: Head 模块，如果未找到则返回 None
        """
        module_name = self.head_module_mapping.get(head_name)
        if module_name is None:
            return None

        if hasattr(world_model, module_name):
            return getattr(world_model, module_name)
        else:
            return None

    def _scale_module_weights(
        self,
        module: nn.Module,
        scale_factor: float
    ) -> bool:
        """
        缩放模块的所有权重（与 scale_module_weights_vectorized 一致）

        Args:
            module (nn.Module): 要缩放的模块
            scale_factor (float): 缩放因子

        Returns:
            bool: 是否成功
        """
        if not (0.0 < scale_factor < 1.0):
            return False

        try:
            # 1. 将模块的所有参数展平成一个单一向量
            params_vec = parameters_to_vector(module.parameters())

            # 2. 在这个向量上执行一次乘法操作
            params_vec.data.mul_(scale_factor)

            # 3. 将缩放后的向量复制回模块的各个参数
            vector_to_parameters(params_vec, module.parameters())

            return True
        except Exception as e:
            logging.error(f"Error scaling module weights: {e}")
            return False

    def get_statistics(self) -> Dict:
        """
        获取统计信息

        Returns:
            Dict: 统计信息
        """
        stats = {
            'enabled': self.enabled,
            'total_iterations': self.iteration_count,
            'scaling_history': {},
        }

        for head_name in self.enabled_heads:
            history = self.scaling_history.get(head_name, [])
            if history:
                stats['scaling_history'][head_name] = {
                    'total_scalings': len(history),
                    'last_scaling': history[-1],
                    'average_scale_factor': sum(h['scale_factor'] for h in history) / len(history),
                    'total_cumulative_scaling': np.prod([h['scale_factor'] for h in history]),
                }

        return stats


def create_head_clip_manager_from_dict(config_dict: Dict) -> HeadClipManager:
    """
    从配置字典创建 HeadClipManager

    Args:
        config_dict (Dict): 配置字典

    Returns:
        HeadClipManager: 管理器实例

    示例:
        config_dict = {
            'enabled': True,
            'enabled_heads': ['policy', 'value'],
            'head_configs': {
                'policy': {
                    'use_annealing': True,
                    'start_value': 30.0,
                    'end_value': 10.0,
                    'anneal_steps': 500000,
                    'anneal_type': 'cosine',
                },
                'value': {
                    'clip_threshold': 20.0,
                    'use_annealing': False,
                },
            },
            'monitor_freq': 1,
            'log_freq': 1000,
        }
    """
    enabled = config_dict.get('enabled', True)
    enabled_heads = config_dict.get('enabled_heads', [])
    monitor_freq = config_dict.get('monitor_freq', 1)
    log_freq = config_dict.get('log_freq', 1000)

    # 解析 head_configs
    head_configs = {}
    head_configs_dict = config_dict.get('head_configs', {})
    for head_name, head_config_dict in head_configs_dict.items():
        head_configs[head_name] = HeadClipConfig(**head_config_dict)

    return HeadClipManager(
        enabled=enabled,
        enabled_heads=enabled_heads,
        head_configs=head_configs,
        monitor_freq=monitor_freq,
        log_freq=log_freq,
    )


if __name__ == "__main__":
    # 使用示例
    print("=" * 60)
    print("Head Clip Manager 使用示例")
    print("=" * 60)

    # 示例 1: 基本配置
    print("\n示例 1: 基本配置")
    config_dict = {
        'enabled': True,
        'enabled_heads': ['policy'],
        'head_configs': {
            'policy': {
                'use_annealing': True,
                'start_value': 30.0,
                'end_value': 10.0,
                'anneal_steps': 500000,
                'anneal_type': 'cosine',
            },
        },
        'monitor_freq': 1,
        'log_freq': 1000,
    }

    manager = create_head_clip_manager_from_dict(config_dict)
    print(f"Manager 创建成功，启用的 head: {manager.enabled_heads}")

    # 示例 2: 计算当前阈值
    print("\n示例 2: 计算当前阈值")
    for iter in [0, 100000, 250000, 500000]:
        threshold = manager.compute_current_threshold('policy', iter)
        print(f"  Iter {iter}: threshold = {threshold:.2f}")

    print("\n" + "=" * 60)
    print("所有示例运行成功！")
    print("=" * 60)

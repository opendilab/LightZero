"""
Head Clip Manager - Dynamic Head Clipping implementation consistent with Encoder-Clip principles

This module provides dynamic Head Clipping functionality similar to Encoder-Clip:
1. Monitor the range of head outputs (logits)
2. Scale all weights of the entire head module when exceeding the threshold
3. Support annealing (threshold gradually becomes stricter from loose)
4. Support independent configuration for multiple heads

Differences from previous Head Weight Scaling:
- Before: Static scaling once during initialization
- Now: Dynamic monitoring and scaling during training (consistent with Encoder-Clip)

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
    """Clip configuration for a single Head"""
    # Fixed threshold (if annealing is not enabled)
    clip_threshold: float = 15.0

    # Whether to enable annealing
    use_annealing: bool = True

    # Annealing configuration
    anneal_type: str = 'cosine'  # 'cosine' or 'linear'
    start_value: float = 30.0    # Loose in the early phase
    end_value: float = 10.0      # Strict in the later phase
    anneal_steps: int = 500000

    def __post_init__(self):
        """Validate configuration"""
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
    Head Clip Manager - Dynamically monitor and clip Head outputs

    Working principle (consistent with Encoder-Clip):
    1. After each training iteration, monitor the outputs (logits) of each head
    2. Calculate max(|logits|)
    3. If exceeding the current threshold, scale the weights of the entire head module
    4. Threshold supports annealing (from loose to strict)
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
        Initialize Head Clip Manager

        Args:
            enabled (bool): Whether to enable Head Clip
            enabled_heads (List[str], optional): List of heads that need clipping
                Example: ['policy', 'value', 'rewards']
                If None, no head will be enabled
            head_configs (Dict[str, HeadClipConfig], optional): Configuration for each head
                Example: {'policy': HeadClipConfig(...), 'value': HeadClipConfig(...)}
                If a head is not in this dictionary, use default configuration
            monitor_freq (int): Monitoring frequency (check every N iterations)
            log_freq (int): Log printing frequency
        """
        self.enabled = enabled
        self.enabled_heads = enabled_heads or []
        self.head_configs = head_configs or {}
        self.monitor_freq = monitor_freq
        self.log_freq = log_freq

        # Statistical information
        self.scaling_history = {head: [] for head in self.enabled_heads}
        self.iteration_count = 0

        # Log mapping
        self.logits_key_mapping = {
            'policy': 'logits_policy',
            'value': 'logits_value',
            'reward': 'logits_reward',
            'rewards': 'logits_reward',  # Compatible with both naming conventions
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
            logging.info(">>> Head Clip Manager Enabled <<<")
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
        Get the configuration for the specified head. If it doesn't exist, return the default configuration

        Args:
            head_name (str): Name of the head

        Returns:
            HeadClipConfig: Configuration object
        """
        if head_name in self.head_configs:
            return self.head_configs[head_name]
        else:
            # Return default configuration
            return HeadClipConfig()

    def compute_current_threshold(
        self,
        head_name: str,
        train_iter: int
    ) -> float:
        """
        Compute the threshold for the current training step (considering annealing)

        Args:
            head_name (str): Name of the head
            train_iter (int): Current training iteration count

        Returns:
            float: Current threshold
        """
        config = self.get_head_config(head_name)

        if not config.use_annealing:
            return config.clip_threshold

        # Calculate annealing progress
        progress = min(1.0, train_iter / config.anneal_steps)

        if config.anneal_type == 'cosine':
            # Cosine schedule: smooth transition from 1 to 0
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
        Apply Head Clip (main function)

        Workflow:
        1. Iterate through all enabled heads
        2. Get the output (logits) of each head
        3. Calculate max(|logits|)
        4. If exceeding the current threshold, scale the entire head module

        Args:
            world_model (nn.Module): WorldModel instance
            losses (LossWithIntermediateLosses): Loss object containing intermediate outputs
            train_iter (int): Current training iteration count

        Returns:
            Dict[str, Dict]: Scaling information for each head
                Example: {
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

        # Only check at specified frequency
        if train_iter % self.monitor_freq != 0:
            return {}

        self.iteration_count = train_iter
        results = {}

        for head_name in self.enabled_heads:
            # 1. Get logits
            logits = self._get_head_logits(losses, head_name)
            if logits is None:
                continue

            # 2. Calculate current threshold
            current_threshold = self.compute_current_threshold(head_name, train_iter)

            # 3. Calculate maximum absolute value of logits
            max_logits = logits.abs().max().item()

            # 4. Determine if scaling is needed
            scaled = False
            scale_factor = 1.0

            if max_logits > current_threshold:
                scale_factor = current_threshold / max_logits

                # Get head module
                head_module = self._get_head_module(world_model, head_name)
                if head_module is not None:
                    # Scale all weights of the entire head module
                    success = self._scale_module_weights(head_module, scale_factor)
                    scaled = success

                    if success:
                        # Record history
                        self.scaling_history[head_name].append({
                            'iteration': train_iter,
                            'max_logits': max_logits,
                            'threshold': current_threshold,
                            'scale_factor': scale_factor,
                        })

            # 5. Record results
            results[head_name] = {
                'max_logits': max_logits,
                'threshold': current_threshold,
                'scale_factor': scale_factor,
                'scaled': scaled,
            }

            # 6. Print log
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
        Get the logits of the specified head from the losses object

        Args:
            losses (LossWithIntermediateLosses): Loss object
            head_name (str): Name of the head

        Returns:
            Optional[torch.Tensor]: Logits tensor, returns None if not found
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
        Get the module of the specified head

        Args:
            world_model (nn.Module): WorldModel instance
            head_name (str): Name of the head

        Returns:
            Optional[nn.Module]: Head module, returns None if not found
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
        Scale all weights of the module (consistent with scale_module_weights_vectorized)

        Args:
            module (nn.Module): Module to be scaled
            scale_factor (float): Scaling factor

        Returns:
            bool: Whether the operation was successful
        """
        if not (0.0 < scale_factor < 1.0):
            return False

        try:
            # 1. Flatten all parameters of the module into a single vector
            params_vec = parameters_to_vector(module.parameters())

            # 2. Perform multiplication operation on this vector
            params_vec.data.mul_(scale_factor)

            # 3. Copy the scaled vector back to the individual parameters of the module
            vector_to_parameters(params_vec, module.parameters())

            return True
        except Exception as e:
            logging.error(f"Error scaling module weights: {e}")
            return False

    def get_statistics(self) -> Dict:
        """
        Get statistical information

        Returns:
            Dict: Statistical information
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
    Create HeadClipManager from a configuration dictionary

    Args:
        config_dict (Dict): Configuration dictionary

    Returns:
        HeadClipManager: Manager instance

    Example:
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

    # Parse head_configs
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
    # Usage example
    print("=" * 60)
    print("Head Clip Manager Usage Example")
    print("=" * 60)

    # Example 1: Basic configuration
    print("\nExample 1: Basic configuration")
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
    print(f"Manager created successfully, enabled heads: {manager.enabled_heads}")

    # Example 2: Compute current threshold
    print("\nExample 2: Compute current threshold")
    for iter in [0, 100000, 250000, 500000]:
        threshold = manager.compute_current_threshold('policy', iter)
        print(f"  Iter {iter}: threshold = {threshold:.2f}")

    print("\n" + "=" * 60)
    print("All examples ran successfully!")
    print("=" * 60)

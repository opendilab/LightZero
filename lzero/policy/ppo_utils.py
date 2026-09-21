from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch.distributions import Categorical


def masked_categorical(logits: torch.Tensor, action_mask: torch.Tensor) -> Categorical:
    """Build a categorical policy while rejecting rows without a legal action."""
    action_mask = action_mask.to(device=logits.device, dtype=torch.bool)
    if action_mask.shape != logits.shape:
        raise ValueError(
            f'action_mask must match logits, got {tuple(action_mask.shape)} and {tuple(logits.shape)}'
        )
    if not action_mask.any(dim=-1).all():
        raise ValueError('Every policy row must contain at least one legal action')
    return Categorical(logits=logits.masked_fill(~action_mask, torch.finfo(logits.dtype).min))


def ppo_policy_loss(
        policy_logits: torch.Tensor,
        action_mask: torch.Tensor,
        actions: torch.Tensor,
        old_log_prob: torch.Tensor,
        advantages: torch.Tensor,
        valid_mask: torch.Tensor,
        clip_ratio: float,
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """Return per-step PPO loss/entropy and masked diagnostics.

    ``old_log_prob`` is stored at collection time for the action that was actually
    executed. Keeping this scalar avoids reconstructing a behavior distribution
    after the shared latent model has already changed.
    """
    if not 0.0 < clip_ratio < 1.0:
        raise ValueError(f'clip_ratio must be in (0, 1), got {clip_ratio}')

    dist = masked_categorical(policy_logits, action_mask)
    log_prob = dist.log_prob(actions.long())
    log_ratio = log_prob - old_log_prob
    ratio = log_ratio.exp()
    unclipped = ratio * advantages
    clipped = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
    per_step_policy_loss = -torch.minimum(unclipped, clipped)
    per_step_entropy = dist.entropy()

    valid = valid_mask.to(dtype=torch.bool)
    valid_count = valid.sum().clamp_min(1)
    with torch.no_grad():
        approx_kl = ((ratio - 1.0) - log_ratio)[valid].sum() / valid_count
        clip_fraction = ((ratio - 1.0).abs() > clip_ratio)[valid].float().sum() / valid_count
        ratio_valid = ratio[valid]
        diagnostics = {
            'ppo_approx_kl': approx_kl,
            'ppo_clip_fraction': clip_fraction,
            'ppo_ratio_mean': ratio_valid.mean() if ratio_valid.numel() else ratio.new_tensor(1.0),
            'ppo_ratio_min': ratio_valid.min() if ratio_valid.numel() else ratio.new_tensor(1.0),
            'ppo_ratio_max': ratio_valid.max() if ratio_valid.numel() else ratio.new_tensor(1.0),
        }
    return per_step_policy_loss, per_step_entropy, diagnostics


def compute_gae(
        rewards: Sequence[float],
        values: Sequence[float],
        gamma: float,
        gae_lambda: float,
        bootstrap_value: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute unnormalized GAE and returns for one chronological trajectory."""
    rewards_np = np.asarray(rewards, dtype=np.float32).reshape(-1)
    values_np = np.asarray(values, dtype=np.float32).reshape(-1)
    if rewards_np.shape != values_np.shape:
        raise ValueError(f'rewards and values must align, got {rewards_np.shape} and {values_np.shape}')
    if not 0.0 <= gamma <= 1.0 or not 0.0 <= gae_lambda <= 1.0:
        raise ValueError(f'gamma and gae_lambda must be in [0, 1], got {gamma}, {gae_lambda}')

    advantages = np.zeros_like(rewards_np)
    gae_value = 0.0
    next_value = float(bootstrap_value)
    for index in range(len(rewards_np) - 1, -1, -1):
        delta = rewards_np[index] + gamma * next_value - values_np[index]
        gae_value = delta + gamma * gae_lambda * gae_value
        advantages[index] = gae_value
        next_value = float(values_np[index])
    return advantages, advantages + values_np


def normalize_advantages(advantages: Iterable[np.ndarray], eps: float = 1e-8) -> Tuple[List[np.ndarray], float, float]:
    """Normalize all valid transitions in one freshly collected rollout together."""
    arrays = [np.asarray(advantage, dtype=np.float32) for advantage in advantages]
    if not arrays:
        return [], 0.0, 1.0
    flat = np.concatenate([array.reshape(-1) for array in arrays])
    if flat.size == 0:
        return arrays, 0.0, 1.0
    mean = float(flat.mean())
    std = float(flat.std())
    scale = max(std, eps)
    return [((array - mean) / scale).astype(np.float32) for array in arrays], mean, std

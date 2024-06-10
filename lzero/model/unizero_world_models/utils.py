import random
import shutil
from collections import OrderedDict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

from lzero.model.common import RepresentationNetwork

# 使用LRU缓存替换原有的字典缓存
from functools import lru_cache
import hashlib

@lru_cache(maxsize=5000)
def quantize_state_with_lru_cache(state, num_buckets=15):
    quantized_state = np.digitize(state, bins=np.linspace(0, 1, num=num_buckets))
    return tuple(quantized_state)


def quantize_state(state, num_buckets=100):
    """
    量化状态向量。
    参数:
        state: 要量化的状态向量。
        num_buckets: 量化的桶数。
    返回:
        量化后的状态向量的哈希值。
    """
    # 使用np.digitize将状态向量的每个维度值映射到num_buckets个桶中
    quantized_state = np.digitize(state, bins=np.linspace(0, 1, num=num_buckets))
    # 使用更稳定的哈希函数
    quantized_state_bytes = quantized_state.tobytes()
    hash_object = hashlib.sha256(quantized_state_bytes)
    return hash_object.hexdigest()

@dataclass
class WorldModelOutput:
    output_sequence: torch.FloatTensor
    logits_observations: torch.FloatTensor
    logits_rewards: torch.FloatTensor
    logits_ends: torch.FloatTensor
    logits_policy: torch.FloatTensor
    logits_value: torch.FloatTensor

class SimNorm(nn.Module):
    """
    简单单位向量归一化。
    改编自 https://arxiv.org/abs/2204.00616.
    """

    def __init__(self, simnorm_dim):
        super().__init__()
        self.dim = simnorm_dim

    def forward(self, x):
        shp = x.shape
        # 确保至少有一个单纯形用于归一化。
        if shp[1] != 0:
            x = x.view(*shp[:-1], -1, self.dim)
            x = F.softmax(x, dim=-1)
            return x.view(*shp)
        else:
            return x

    def __repr__(self):
        return f"SimNorm(dim={self.dim})"


def init_weights(module):
    if not isinstance(module, RepresentationNetwork):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


def extract_state_dict(state_dict, module_name):
    return OrderedDict({k.split('.', 1)[1]: v for k, v in state_dict.items() if k.startswith(module_name)})


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)


def remove_dir(path, should_ask=False):
    assert path.is_dir()
    if (not should_ask) or input(f"Remove directory : {path} ? [Y/n] ").lower() != 'n':
        shutil.rmtree(path)


def compute_lambda_returns(rewards, values, ends, gamma, lambda_):
    assert rewards.ndim == 2 or (rewards.ndim == 3 and rewards.size(2) == 1)
    assert rewards.shape == ends.shape == values.shape, f"{rewards.shape}, {values.shape}, {ends.shape}"  # (B, T, 1)
    t = rewards.size(1)
    lambda_returns = torch.empty_like(values)
    lambda_returns[:, -1] = values[:, -1]
    lambda_returns[:, :-1] = rewards[:, :-1] + ends[:, :-1].logical_not() * gamma * (1 - lambda_) * values[:, 1:]

    last = values[:, -1]
    for i in list(range(t - 1))[::-1]:
        lambda_returns[:, i] += ends[:, i].logical_not() * gamma * lambda_ * last
        last = lambda_returns[:, i]

    return lambda_returns


class LossWithIntermediateLosses:
    def __init__(self, latent_recon_loss_weight=0, perceptual_loss_weight=0,
                 #  first_step_losses=None, middle_step_losses=None, last_step_losses=None,
                 **kwargs):
        # self.first_step_losses = first_step_losses
        # self.middle_step_losses = middle_step_losses
        # self.last_step_losses = last_step_losses
        # self.loss_total = sum(kwargs.values())

        # Ensure that kwargs is not empty
        if not kwargs:
            raise ValueError("At least one loss must be provided")

        # Get a reference device from one of the provided losses
        device = next(iter(kwargs.values())).device

        # similar with ssl_loss in EZ
        # self.obs_loss_weight = 2.
        # self.reward_loss_weight = 1.
        # self.value_loss_weight = 0.25
        # self.policy_loss_weight = 1.
        # # self.ends_loss_weight = 1.
        # self.ends_loss_weight = 0.

        self.obs_loss_weight = 10
        # self.obs_loss_weight = 20
        self.reward_loss_weight = 1.
        self.value_loss_weight = 0.25
        self.policy_loss_weight = 1.
        # self.ends_loss_weight = 1.
        self.ends_loss_weight = 0.

        # self.obs_loss_weight = 20
        # self.reward_loss_weight = 0.1
        # self.value_loss_weight = 0.1
        # self.policy_loss_weight = 0.1

        self.latent_recon_loss_weight = latent_recon_loss_weight
        self.perceptual_loss_weight = perceptual_loss_weight
        # self.latent_recon_loss_weight = 0.1

        # Initialize the total loss tensor on the correct device
        self.loss_total = torch.tensor(0., device=device)
        for k, v in kwargs.items():
            if k == 'loss_obs':
                self.loss_total += self.obs_loss_weight * v
            elif k == 'loss_rewards':
                self.loss_total += self.reward_loss_weight * v
            elif k == 'loss_policy':
                self.loss_total += self.policy_loss_weight * v
            elif k == 'loss_value':
                self.loss_total += self.value_loss_weight * v
            elif k == 'loss_ends':
                self.loss_total += self.ends_loss_weight * v
            elif k == 'latent_recon_loss':
                self.loss_total += self.latent_recon_loss_weight * v
            elif k == 'perceptual_loss':
                self.loss_total += self.perceptual_loss_weight * v
            # else:
            #     raise ValueError(f"Unknown loss type : {k}")

        # self.intermediate_losses = {k: v.item() for k, v in kwargs.items()}
        # self.intermediate_losses = {k: v if isinstance(v, dict) elif isinstance(v, float) v else v.item() for k, v in kwargs.items()}
        self.intermediate_losses = {
            k: v if isinstance(v, dict) else (v if isinstance(v, float) else v.item())
            for k, v in kwargs.items()
        }

    def __truediv__(self, value):
        for k, v in self.intermediate_losses.items():
            self.intermediate_losses[k] = v / value
        self.loss_total = self.loss_total / value
        return self


class RandomHeuristic:
    def __init__(self, num_actions):
        self.num_actions = num_actions

    def act(self, obs):
        assert obs.ndim == 4  # (N, H, W, C)
        n = obs.size(0)
        return torch.randint(low=0, high=self.num_actions, size=(n,))


def make_video(fname, fps, frames):
    assert frames.ndim == 4  # (t, h, w, c)
    t, h, w, c = frames.shape
    assert c == 3

    video = cv2.VideoWriter(str(fname), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for frame in frames:
        video.write(frame[:, :, ::-1])
    video.release()

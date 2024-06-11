import hashlib
import random
import shutil
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lzero.model.common import RepresentationNetwork
from .kv_caching import KeysValues


def to_device_for_kvcache(keys_values: KeysValues, device: str) -> KeysValues:
    """
    Transfer all KVCache objects within the KeysValues object to a certain device.

    Arguments:
        - keys_values (KeysValues): The KeysValues object to be transferred.
        - device (str): The device to transfer to.

    Returns:
        - keys_values (KeysValues): The KeysValues object with its caches transferred to the specified device.
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    for kv_cache in keys_values:
        kv_cache._k_cache._cache = kv_cache._k_cache._cache.to(device)
        kv_cache._v_cache._cache = kv_cache._v_cache._cache.to(device)
    return keys_values


def convert_to_depth(search_path, depth_map, last_depth):
    # Get the newly added element
    new_index = search_path[-1]

    # If the depth of the newly added element has not been calculated, compute it based on the depth of the parent node
    if new_index not in depth_map:
        if search_path[new_index] not in depth_map:
            depth_map[search_path[new_index]] = max(list(depth_map.values())) + 1
        else:
            depth_map[new_index] = depth_map[search_path[new_index]] + 1

    # Append the depth of the newly added element to the end of last_depth
    last_depth.append(depth_map[new_index])

    return last_depth


# Function to calculate CUDA memory usage in gigabytes
def calculate_cuda_memory_gb(past_keys_values_cache, num_layers: int):
    total_memory_bytes = 0

    # Iterate over all KeysValues instances in the OrderedDict
    for kv_instance in past_keys_values_cache.values():
        num_layers = len(kv_instance)  # Get the number of layers
        for layer in range(num_layers):
            kv_cache = kv_instance[layer]
            k_shape = kv_cache._k_cache.shape  # Get the shape of the keys cache
            v_shape = kv_cache._v_cache.shape  # Get the shape of the values cache

            # Calculate the number of elements and multiply by the number of bytes per element
            k_memory = torch.prod(torch.tensor(k_shape)) * 4
            v_memory = torch.prod(torch.tensor(v_shape)) * 4

            # Accumulate the memory used by the keys and values cache
            layer_memory = k_memory + v_memory
            total_memory_bytes += layer_memory.item()  # .item() ensures conversion to a standard Python number

    # Convert total memory from bytes to gigabytes
    total_memory_gb = total_memory_bytes / (1024 ** 3)
    return total_memory_gb


@lru_cache(maxsize=5000)
def quantize_state_with_lru_cache(state, num_buckets=15):
    quantized_state = np.digitize(state, bins=np.linspace(0, 1, num=num_buckets))
    return tuple(quantized_state)


def quantize_state(state, num_buckets=100):
    """
    Quantize the state vector.
    Args:
        state: The state vector to be quantized.
        num_buckets: The number of quantization buckets.
    Returns:
        The hash value of the quantized state vector.
    """
    # Use np.digitize to map each dimension value of the state vector into num_buckets
    quantized_state = np.digitize(state, bins=np.linspace(0, 1, num=num_buckets))
    # Use a more stable hash function
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
    Simplified normalization. Adapted from https://arxiv.org/abs/2204.00616.
    """

    def __init__(self, simnorm_dim):
        super().__init__()
        self.dim = simnorm_dim

    def forward(self, x):
        shp = x.shape
        # Ensure there is at least one simplex for normalization.
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
    def __init__(self, latent_recon_loss_weight=0, perceptual_loss_weight=0, **kwargs):
        # Ensure that kwargs is not empty
        if not kwargs:
            raise ValueError("At least one loss must be provided")

        # Get a reference device from one of the provided losses
        device = next(iter(kwargs.values())).device

        # Define the weights for each loss type
        self.obs_loss_weight = 10
        self.reward_loss_weight = 1.
        self.value_loss_weight = 0.25
        self.policy_loss_weight = 1.
        self.ends_loss_weight = 0.

        self.latent_recon_loss_weight = latent_recon_loss_weight
        self.perceptual_loss_weight = perceptual_loss_weight

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

        self.intermediate_losses = {
            k: v if isinstance(v, dict) else (v if isinstance(v, float) else v.item())
            for k, v in kwargs.items()
        }

    def __truediv__(self, value):
        for k, v in self.intermediate_losses.items():
            self.intermediate_losses[k] = v / value
        self.loss_total = self.loss_total / value
        return self


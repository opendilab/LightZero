import hashlib
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

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


def quantize_state(state, num_buckets=100):
    """
    Quantize the state vector.

    Arguments:
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


def init_weights(module, norm_type='BN'):
    """
    Initialize the weights of the module based on the specified normalization type.

    Arguments:
        module (nn.Module): The module to initialize.
        norm_type (str): The type of normalization to use ('BN' for BatchNorm, 'LN' for LayerNorm).
    """
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
        print(f"Init {module} using zero bias, 1 weight")
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    elif isinstance(module, nn.BatchNorm2d):
        print(f"Init nn.BatchNorm2d using zero bias, 1 weight")
        module.weight.data.fill_(1.0)
        module.bias.data.zero_()
    elif isinstance(module, nn.Conv2d):
        if norm_type == 'BN':
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            print(f"Init nn.Conv2d using kaiming normal for BN")
        elif norm_type == 'LN':
            nn.init.xavier_uniform_(module.weight)
            print(f"Init nn.Conv2d using xavier uniform for LN")
    elif isinstance(module, nn.Linear):
        if norm_type == 'BN':
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            print("Init Linear using kaiming normal for BN")
        elif norm_type == 'LN':
            nn.init.xavier_uniform_(module.weight)
            print("Init Linear using xavier uniform for LN")


class LossWithIntermediateLosses:
    """
    Overview:
        A class to store the total loss and intermediate losses for a model.
    Arguments:
        - latent_recon_loss_weight (float): The weight for the latent reconstruction loss.
        - perceptual_loss_weight (float): The weight for the perceptual loss.
        - **kwargs: The intermediate losses to store.
    Returns:
        - None
    """
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
            k: v if isinstance(v, dict) or isinstance(v, torch.Tensor) else (v if isinstance(v, float) else v.item())
            for k, v in kwargs.items()
        }

    def __truediv__(self, value):
        for k, v in self.intermediate_losses.items():
            self.intermediate_losses[k] = v / value
        self.loss_total = self.loss_total / value
        return self

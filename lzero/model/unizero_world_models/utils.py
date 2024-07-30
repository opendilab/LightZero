import hashlib
from typing import Optional, List, Tuple, Union
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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


@dataclass
class WorldModelOutputSoftModulization:
    output_sequence: torch.FloatTensor
    logits_observations: torch.FloatTensor
    logits_rewards: torch.FloatTensor
    logits_ends: torch.FloatTensor
    logits_policy: torch.FloatTensor
    logits_value: torch.FloatTensor
    task_id: int
    observation_weights_list: List[torch.FloatTensor] = None
    reward_weights_list: List[torch.FloatTensor] = None
    policy_weights_list: List[torch.FloatTensor] = None
    value_weights_list: List[torch.FloatTensor] = None

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
        try:
            module.bias.data.zero_()
        except Exception as e:
            print(e)
        try:
             module.weight.data.fill_(1.0)
        except Exception as e:
            print(e)
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
        
        # updated from soft modulization 
        self.task_id = kwargs.get("task_id", None)
        self.obs_soft_module_route_weights = kwargs.get("observation_weights_list", None)
        
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
            k: v if isinstance(v, dict) or isinstance(v, np.ndarray) or isinstance(v, torch.Tensor) else (v if isinstance(v, float) else v.item())
            for k, v in kwargs.items() if k not in ["task_id", "observation_weights_list"]
        }

    def __truediv__(self, value):
        for k, v in self.intermediate_losses.items():
            self.intermediate_losses[k] = v / value
        self.loss_total = self.loss_total / value
        return self


class SoftModulizationHead(nn.Module):
    """
    Overview:
        SoftModulizationHead is an nn.Module class that implements soft modulization for multi-task reinforcement learning.
    Arguments:
        - task_num (:obj:`int`): The number of tasks.
        - embed_dim (:obj:`int`): The embedding dimension.
        - base_layers_num (:obj:`int`): The number of base layers in the model.
        - device (:obj:`torch.device`): The device to run computations on.
    """

    def __init__(self, 
                 task_num: int, 
                 embed_dim: int, 
                 gating_embed_mlp_num: int,
                 base_model_modulelists, 
                 base_layers_num: int,
                 base_modules_num: int, 
                 device: torch.device
        ) -> None:
        super(SoftModulizationHead, self).__init__()
        self.task_num = task_num
        self.embed_dim = embed_dim
        self.gating_embed_mlp_num = gating_embed_mlp_num
        self.base_layers_num = base_layers_num
        self.base_modules_num = base_modules_num
        self.base_model_modulelists = base_model_modulelists
        self.device = device

        # Task embedding layer
        self.task_embed_layer = nn.Linear(task_num, embed_dim)
        # Ex. (.., task_num) -> (.., 768)
        
        # Gating fully connected layers
        gating_fc_layer_module = [nn.ReLU(), nn.Linear(embed_dim, embed_dim)] * (self.gating_embed_mlp_num - 1)
        self.gating_fcs = nn.Sequential(*gating_fc_layer_module)

        # Initial gating weight layer
        self.gating_weight_fc_0 = nn.Linear(embed_dim, task_num * task_num)
        # (.., 768) -> (.., 16)
        
        # Conditional gating weight layers
        self.gating_weight_fcs = nn.ModuleList()
        self.gating_weight_cond_fcs = nn.ModuleList()
        for k in range(self.base_layers_num - 2):
            self.gating_weight_cond_fcs.append(nn.Linear((k + 1) * task_num * task_num, embed_dim))
            self.gating_weight_fcs.append(nn.Linear(embed_dim, task_num * task_num))
        
        # Cond_weight_fcs [Linear(16, 768), Linear(32, 768), Linear(48, 768)]
        # weight_fcs: [Linear]
        
        # Final gating weight layers
        self.gating_weight_cond_last = nn.Linear((self.base_layers_num - 1) * task_num * task_num, embed_dim)
        self.gating_weight_last_fc = nn.Linear(embed_dim, task_num)

    def forward(self, x: torch.Tensor, task_id: int, 
                final_norm: Optional[nn.Module]=None, return_weight: bool=False
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Overview:
            Forward pass for soft modulization.
        Arguments:
            - x (:obj:`torch.Tensor`): Input tensor.
            - task_id (:obj:`int`): ID of the task.
            - base_model (:obj:`nn.Module`): Base model containing the layers to be modularized.
        Returns:
            - torch.Tensor: Output tensor after soft modulization.
        """
        # print(f"x.shape:  {x.shape}")
        task_id_vector = torch.zeros(self.task_num).to(self.device)
        task_id_vector[task_id] = 1

        # Process task embedding
        task_embedding = self.task_embed_layer(task_id_vector).to(self.device)
        # print(f"task_embedding.shape before * x :  {task_embedding.shape}")
        task_embedding = F.relu(task_embedding * x)
        task_embedding = self.gating_fcs(task_embedding)
        # print(f"task_embedding.shape after * x:  {task_embedding.shape}")
        
        weights = []
        flatten_weights = []
        base_shape = task_embedding.shape[:-1]
        weight_shape = base_shape + torch.Size([self.task_num, self.task_num])
        flatten_shape = base_shape + torch.Size([self.task_num * self.task_num])

        # Calculate weights between layers
        raw_weight = self.gating_weight_fc_0(F.relu(task_embedding))
        raw_weight = raw_weight.view(weight_shape)
        
        
        softmax_weight = F.softmax(raw_weight, dim=-1)
        # print(f"softmax_weight:  {softmax_weight.shape}")
        flatten_weight = softmax_weight.view(flatten_shape)
        # print(f"flatten_weight:  {flatten_weight.shape}")
        weights.append(softmax_weight)
        flatten_weights.append(flatten_weight)

        for i, (gating_weight_fc, gating_weight_cond_fc) in enumerate(zip(self.gating_weight_fcs, self.gating_weight_cond_fcs)):
            cond = F.relu(torch.cat(flatten_weights, dim=-1))
            # print(f"cond_weight:  {cond.shape}")
            cond = gating_weight_cond_fc(cond)
            # print(f"cond_weight:  {cond.shape}")
            cond = F.relu(cond * task_embedding)
            # print(f"cond_weight:  {cond.shape}")
            
            raw_weight = gating_weight_fc(cond)
            raw_weight = raw_weight.view(weight_shape)
            softmax_weight = F.softmax(raw_weight, dim=-1)
            # print(f"softmax_weight:  {softmax_weight.shape}")
            weights.append(softmax_weight)
            flatten_weights.append(raw_weight.view(flatten_shape))

        cond = F.relu(torch.cat(flatten_weights, dim=-1))
        # print(f"cond_weight:  {cond.shape}")
        cond = self.gating_weight_cond_last(cond)
        cond = F.relu(cond * task_embedding)
        # print(f"cond_weight:  {cond.shape}")
        raw_last_weight = self.gating_weight_last_fc(cond)
        # print(f"cond_weight:  {cond.shape}")
        last_weight = F.softmax(raw_last_weight, dim=-1)

        # Forward calculation
        # print(f"self.base_modules_num = {self.base_modules_num}")
        # print(f"len(self.base_model) = {len(self.base_model_modulelists)}")
        # print(f"len(self.base_model[0]) = {len(self.base_model_modulelists[0])}")
        obs_mid_layers = [self.base_model_modulelists[0][i] for i in range(self.base_modules_num)]
        obs_mid_outputs = [obs_mid_layer(x).unsqueeze(-2) for obs_mid_layer in obs_mid_layers]
        obs_mid_outputs = torch.cat(obs_mid_outputs, dim=-2)

        for i in range(self.base_layers_num - 1):
            new_module_outputs = []
            obs_next_mid_layers = [self.base_model_modulelists[i+1][j] for j in range(self.base_modules_num)]

            for j, next_layer_module in enumerate(obs_next_mid_layers):
                
                # print(f"obs_mid_outputs.shape: {obs_mid_outputs.shape}")
                # print(f"weights[{i}][..., {j}, :].unsqueeze(-1).shape: {weights[i][..., j, :].unsqueeze(-1).shape}")
                next_module_input = F.relu((obs_mid_outputs * weights[i][..., j, :].unsqueeze(-1)).sum(dim=-2))
                new_module_outputs.append((next_layer_module(next_module_input)).unsqueeze(-2))
            # print([x.shape for x in new_module_outputs])
            obs_mid_outputs = torch.cat(new_module_outputs, dim=-2)

        obs_module_output = obs_mid_outputs
        obs_output = (obs_module_output * last_weight.unsqueeze(-1)).sum(-2)
        
        if final_norm is not None:
            obs_output = final_norm(obs_output)
            
        if return_weight:
            return obs_output, flatten_weights
        return obs_output
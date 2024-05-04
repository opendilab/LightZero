"""
Overview:
    In this file, we provide a set of utility functions for probing network parameters and gradients,
    which can be helpful in analyzing and debugging the inner workings of various models.
"""
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimNorm(nn.Module):
    """
    Simplicial normalization.
    Adapted from https://arxiv.org/abs/2204.00616.
    """

    def __init__(self, simnorm_dim):
        super().__init__()
        self.dim = simnorm_dim

    def forward(self, x):
        shp = x.shape
        # Ensure that there is at least one simplex to normalize across.
        if shp[1] != 0:
            x = x.view(*shp[:-1], -1, self.dim)
            x = F.softmax(x, dim=-1)
            return x.view(*shp)
        else:
            return x

    def __repr__(self):
        return f"SimNorm(dim={self.dim})"




class LinearOutputHook:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, input, output):
        self.outputs.append(output)

def cal_dormant_ratio(model, *inputs, percentage=0.025):
    hooks = []
    hook_handlers = []
    total_neurons = 0
    dormant_neurons = 0

    for _, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.LSTM)):
            hook = LinearOutputHook()
            hooks.append(hook)
            hook_handlers.append(module.register_forward_hook(hook))

    with torch.no_grad():
        model(*inputs)

    for module, hook in zip(
        (module for module in model.modules() if isinstance(module, (nn.Linear, nn.Conv2d, nn.LSTM))), hooks):

        with torch.no_grad():
            for output_data in hook.outputs:
                mean_output = output_data.abs().mean(0)
                avg_neuron_output = mean_output.mean()
                dormant_indices = (mean_output < avg_neuron_output * percentage).nonzero(as_tuple=True)[0]
                if isinstance(module, nn.Linear):
                    total_neurons += module.weight.shape[0] * output_data.shape[0]
                    dormant_neurons += len(dormant_indices)
                elif isinstance(module, nn.Conv2d):
                    total_neurons += module.weight.shape[0] * output_data.shape[0] * output_data.shape[2] * output_data.shape[3]
                    dormant_neurons += len(dormant_indices)
                elif isinstance(module, nn.LSTM):
                    total_neurons += module.hidden_size * module.num_layers * output_data.shape[0] * output_data.shape[1]
                    dormant_neurons += len(dormant_indices)

    for hook in hooks:
        hook.outputs.clear()
        del hook.outputs

    for hook_handler in hook_handlers:
        hook_handler.remove()
        del hook_handler

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return dormant_neurons / total_neurons

def renormalize(inputs: torch.Tensor, first_dim: int = 1) -> torch.Tensor:
    """
    Overview:
        Normalize the input data using the max-min-normalization.
    Arguments:
        - inputs (:obj:`torch.Tensor`): The input data needs to be normalized.
        - first_dim (:obj:`int`): The first dimension of flattening the input data.
    Returns:
        - output (:obj:`torch.Tensor`): The normalized data.
    """
    if first_dim < 0:
        first_dim = len(inputs.shape) + first_dim
    flat_input = inputs.view(*inputs.shape[:first_dim], -1)
    max_val = torch.max(flat_input, first_dim, keepdim=True).values
    min_val = torch.min(flat_input, first_dim, keepdim=True).values
    flat_input = (flat_input - min_val) / (max_val - min_val)

    return flat_input.view(*input.shape)


def get_dynamic_mean(model: nn.Module) -> float:
    dynamic_mean = np.abs(model.conv.weight.detach().cpu().numpy().reshape(-1)).tolist()

    for block in model.resblocks:
        for name, param in block.named_parameters():
            dynamic_mean += np.abs(param.detach().cpu().numpy().reshape(-1)).tolist()
    dynamic_mean = sum(dynamic_mean) / len(dynamic_mean)
    return dynamic_mean


def get_reward_mean(model: nn.Module) -> Tuple[np.ndarray, float]:
    reward_w_dist = model.conv1x1_reward.weight.detach().cpu().numpy().reshape(-1)

    for name, param in model.fc.named_parameters():
        temp_weights = param.detach().cpu().numpy().reshape(-1)
        reward_w_dist = np.concatenate((reward_w_dist, temp_weights))
    reward_mean = np.abs(reward_w_dist).mean()
    return reward_w_dist, reward_mean


def get_params_mean(model: nn.Module) -> Tuple[np.ndarray, float, float, float]:
    representation_mean = model.representation_network.get_param_mean()
    dynamic_mean = model.dynamics_network.get_dynamic_mean()
    reward_w_dist, reward_mean = model.dynamics_network.get_reward_mean()

    return reward_w_dist, representation_mean, dynamic_mean, reward_mean


def get_gradients(model: nn.Module) -> List[torch.Tensor]:
    grads = []
    for p in model.parameters():
        grad = None if p.grad is None else p.grad.detach()
        grads.append(grad)
    return grads


def set_gradients(model: nn.Module, gradients: List[torch.Tensor]) -> None:
    # TODO due to the drawback of zip operation, we have to check whether gradients match model's parameters
    for g, p in zip(gradients, model.parameters()):
        if g is not None:
            p.grad = g

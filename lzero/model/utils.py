"""
Overview:
    In this file, we provide a set of utility functions for probing network parameters and gradients,
    which can be helpful in analyzing and debugging the inner workings of various models.
"""
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn


class LinearOutputHook:
    """
    Overview:
        Hook to capture the output of linear layers.
    """

    def __init__(self):
        """
        Overview:
            Initialize the hook.
        """
        self.outputs: List[torch.Tensor] = []

    def __call__(self, module: nn.Module, input: Tuple[torch.Tensor], output: torch.Tensor) -> None:
        """
        Overview:
            Capture the output of the module.
        Arguments:
            - module: The module being hooked.
            - input: The input to the module (unused in this hook).
            - output: The output from the module.
        """
        self.outputs.append(output)


def cal_dormant_ratio(model: nn.Module, *inputs: torch.Tensor, percentage: float = 0.025) -> float:
    """
    Overview:
        Calculate the dormant neuron ratio in the model. A neuron is considered dormant if its output is less than a
        specified percentage of the average output of the layer. This function is useful for analyzing the sparsity of the model.
        More details can be found in the paper https://arxiv.org/abs/2302.12902.
    Arguments:
        - model: The model to evaluate.
        - inputs: The inputs to the model.
        - percentage: The threshold percentage to consider a neuron dormant, defaults to 0.025.
    Returns:
        - float: The ratio of dormant neurons in the model.
    """
    # List to store hooks and their handlers
    hooks: List[LinearOutputHook] = []
    hook_handlers: List[torch.utils.hooks.RemovableHandle] = []
    total_neurons: int = 0
    dormant_neurons: int = 0

    # Register hooks to capture outputs of specific layers
    for _, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.LSTM)):
            hook = LinearOutputHook()
            hooks.append(hook)
            hook_handlers.append(module.register_forward_hook(hook))

    with torch.no_grad():
        # Forward pass to capture outputs
        model(*inputs)

    # Analyze the captured outputs
    for module, hook in zip((module for module in model.modules() if isinstance(module, (nn.Linear, nn.Conv2d, nn.LSTM))), hooks):
        with torch.no_grad():
            for output_data in hook.outputs:
                mean_output = output_data.abs().mean(0)
                avg_neuron_output = mean_output.mean()
                dormant_indices = (mean_output < avg_neuron_output * percentage).nonzero(as_tuple=True)[0]

                if isinstance(module, nn.Linear):
                    # Calculate total and dormant neurons for Linear layers
                    total_neurons += module.weight.shape[0] * output_data.shape[0]
                    dormant_neurons += len(dormant_indices)
                elif isinstance(module, nn.Conv2d):
                    # Calculate total and dormant neurons for Conv2D layers
                    total_neurons += module.weight.shape[0] * output_data.shape[0] * output_data.shape[2] * output_data.shape[3]
                    dormant_neurons += len(dormant_indices)
                elif isinstance(module, nn.LSTM):
                    # Calculate total and dormant neurons for LSTM layers
                    total_neurons += module.hidden_size * module.num_layers * output_data.shape[0] * output_data.shape[1]
                    dormant_neurons += len(dormant_indices)

    # Clean up hooks
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

    return flat_input.view(*inputs.shape)


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

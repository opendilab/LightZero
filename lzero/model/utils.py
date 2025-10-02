"""
Overview:
    This file provides a set of utility functions for probing network parameters and gradients.
    These tools are helpful for analyzing and debugging the inner workings of various models.
"""
from typing import List, Tuple, Union, Dict, Type, Optional

import numpy as np
import torch
import torch.nn as nn


def compute_average_weight_magnitude(model: nn.Module) -> float:
    """
    Overview:
        Calculates the average absolute magnitude of all parameters in a given model.

    Arguments:
        - model (:obj:`nn.Module`): The model to be evaluated.

    Returns:
        - float: The average absolute magnitude of the model's weights.
    """
    num_weights = 0
    # Use the device of the model's first parameter to ensure consistency.
    device = next(model.parameters()).device
    sum_weight_magnitude = torch.tensor(0.0, device=device)

    for p in model.parameters():
        num_weights += p.numel()
        sum_weight_magnitude += torch.sum(torch.abs(p))

    if num_weights == 0:
        return 0.0
    return sum_weight_magnitude.cpu().item() / num_weights


def compute_effective_rank(singular_values: np.ndarray) -> float:
    """
    Overview:
        Computes the effective rank from an array of singular values. The formula is:
        effective_rank = exp(-sum_i [p_i * log(p_i)]), where p_i is the normalized singular value.

    Arguments:
        - singular_values (:obj:`np.ndarray`): An array of singular values.

    Returns:
        - float: The calculated effective rank.
    """
    # Normalize singular values to form a probability distribution.
    norm_sv = singular_values / np.sum(np.abs(singular_values))
    entropy = 0.0
    for p in norm_sv:
        if p > 1e-8:  # Avoid log(0)
            entropy -= p * np.log(p)
    return np.exp(entropy)


class IntermediateOutputHook:
    """
    Overview:
        A hook class to capture and store the output tensors from a specific nn.Module during a forward pass.
    """
    def __init__(self):
        self.outputs: List[torch.Tensor] = []

    def __call__(self, module: nn.Module, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        """
        Overview:
            This method is called by PyTorch when the hooked module completes its forward pass.
        """
        # Detach the tensor from the computation graph and move to CPU to save memory.
        self.outputs.append(output.detach().cpu())

    def clear(self) -> None:
        """
        Overview:
            Clears the list of captured outputs.
        """
        self.outputs.clear()


def calculate_effective_rank(
    model: nn.Module,
    inputs: Union[torch.Tensor, List[torch.Tensor]],
    representation_layer_name: str,
) -> float:
    """
    Overview:
        Calculates the effective rank of a specified intermediate layer's output (representation)
        by using a forward hook to capture the activations.

    Arguments:
        - model (:obj:`nn.Module`): The model to be evaluated.
        - inputs (:obj:`Union[torch.Tensor, List[torch.Tensor]]`): The inputs for the model's forward pass.
        - representation_layer_name (:obj:`str`): The name of the representation layer, which must be
                                                  findable within `model.named_modules()`.

    Returns:
        - float: The effective rank of the representation layer's output.
    """
    module_dict = dict(model.named_modules())
    if representation_layer_name not in module_dict:
        raise KeyError(f"Representation layer '{representation_layer_name}' not found in model.named_modules().")
    representation_module = module_dict[representation_layer_name]

    hook = IntermediateOutputHook()
    handle = representation_module.register_forward_hook(hook)

    model.eval()
    with torch.no_grad():
        if isinstance(inputs, (list, tuple)):
            _ = model(*inputs)
        else:
            _ = model(inputs)

    # Always remove the hook to prevent memory leaks.
    handle.remove()

    if not hook.outputs:
        raise RuntimeError("No outputs were captured from the representation layer.")

    # Concatenate all captured outputs along the batch dimension.
    rep_tensor = torch.cat(hook.outputs, dim=0) if len(hook.outputs) > 1 else hook.outputs[0]

    # Reshape the representation to a 2D matrix (samples, features).
    rep_tensor = rep_tensor.view(rep_tensor.size(0), -1)

    # Compute singular values using SVD.
    singular_values = np.linalg.svd(rep_tensor.cpu().numpy(), full_matrices=False, compute_uv=False)

    # Calculate the effective rank.
    e_rank = compute_effective_rank(singular_values)

    hook.clear()
    return e_rank


def compute_dormant_stats(outputs: List[torch.Tensor], threshold: float) -> Tuple[int, int]:
    """
    Overview:
        Computes element-wise statistics for a list of output tensors from a layer.

    Arguments:
        - outputs (:obj:`List[torch.Tensor]`): A list of tensors, each representing an output from a forward pass.
        - threshold (:obj:`float`): The activation threshold below which a neuron is considered dormant.

    Returns:
        - Tuple[int, int]: A tuple containing the total number of elements and the number of dormant elements.
    """
    layer_total = 0
    layer_dormant = 0
    for out in outputs:
        flattened = out.view(-1)
        layer_total += flattened.numel()
        layer_dormant += torch.sum(flattened <= threshold).item()
    return layer_total, layer_dormant


def calculate_dormant_ratio(
    model: nn.Module,
    inputs: Union[torch.Tensor, List[torch.Tensor]],
    dormant_threshold: float = 1e-2,
    target_modules: Tuple[Type[nn.Module], ...] = (nn.Conv2d, nn.Linear),
) -> Dict[str, float]:
    """
    Overview:
        Calculates the dormant ratio (percentage of neurons with activation below a threshold) for
        different parts of a model (e.g., encoder, transformer, head). It assumes the model has
        attributes like `encoder`, `transformer`, or `head_dict`.

    Arguments:
        - model (:obj:`nn.Module`): The model to evaluate, expected to have `encoder`, `transformer`, or `head_dict` attributes.
        - inputs (:obj:`Union[torch.Tensor, List[torch.Tensor]]`): The inputs for the model's forward pass.
        - dormant_threshold (:obj:`float`): The activation threshold for defining a dormant neuron. Defaults to 1e-2.
        - target_modules (:obj:`Tuple[Type[nn.Module], ...]`): A tuple of module types to attach hooks to.

    Returns:
        - Dict[str, float]: A dictionary containing the dormant ratios for each model part and a global ratio.
    """
    parts = {}
    if hasattr(model, "encoder"):
        parts["encoder"] = model.encoder
    if hasattr(model, "transformer"):
        parts["transformer"] = model.transformer
    if hasattr(model, "head_dict"):
        parts["head"] = model.head_dict

    # Fallback for models that don't have the standard part attributes.
    if not parts:
        parts["model"] = model

    hooks_dict = {part: [] for part in parts}
    hook_handles = []

    # Register a forward hook for each target module in each part.
    for part_name, submodule in parts.items():
        for name, module in submodule.named_modules():
            if isinstance(module, target_modules):
                hook = IntermediateOutputHook()
                full_name = f"{part_name}/{name}"
                hooks_dict[part_name].append((full_name, hook))
                handle = module.register_forward_hook(hook)
                hook_handles.append(handle)

    model.eval()
    with torch.no_grad():
        if isinstance(inputs, (list, tuple)):
            _ = model(*inputs)
        else:
            _ = model(inputs)

    results = {}
    total_global = 0
    dormant_global = 0

    # Calculate dormant stats from captured outputs.
    for part, hooks in hooks_dict.items():
        part_total = 0
        part_dormant = 0
        for full_name, hook in hooks:
            layer_total, layer_dormant = compute_dormant_stats(hook.outputs, dormant_threshold)
            part_total += layer_total
            part_dormant += layer_dormant
        
        results[part] = (part_dormant / part_total) * 100.0 if part_total > 0 else 0.0
        total_global += part_total
        dormant_global += part_dormant

    results["global"] = (dormant_global / total_global) * 100.0 if total_global > 0 else 0.0

    # Clean up all hooks.
    for handle in hook_handles:
        handle.remove()
    for hooks in hooks_dict.values():
        for _, hook in hooks:
            hook.clear()

    return results


def renormalize(inputs: torch.Tensor, first_dim: int = 1) -> torch.Tensor:
    """
    Overview:
        Normalizes the input tensor using min-max scaling. The normalization is applied
        over all dimensions starting from `first_dim`.

    Arguments:
        - inputs (:obj:`torch.Tensor`): The input tensor to be normalized.
        - first_dim (:obj:`int`): The first dimension from which to flatten the tensor for normalization.

    Returns:
        - torch.Tensor: The min-max normalized tensor.
    """
    if first_dim < 0:
        first_dim = inputs.dim() + first_dim
    
    shape = inputs.shape
    flat_input = inputs.view(*shape[:first_dim], -1)
    
    max_val, _ = torch.max(flat_input, dim=first_dim, keepdim=True)
    min_val, _ = torch.min(flat_input, dim=first_dim, keepdim=True)
    
    # Add a small epsilon to avoid division by zero.
    denominator = max_val - min_val
    denominator[denominator < 1e-8] = 1e-8
    
    normalized_flat = (flat_input - min_val) / denominator
    
    return normalized_flat.view(*shape)


def get_params_mean(model: nn.Module) -> float:
    """
    Overview:
        Calculates the mean of the absolute values of all parameters in a model. This is an alias
        for `compute_average_weight_magnitude`.

    Arguments:
        - model (:obj:`nn.Module`): The model to be evaluated.

    Returns:
        - float: The mean of the absolute parameter values.
    """
    return compute_average_weight_magnitude(model)


def get_gradients(model: nn.Module) -> List[Optional[torch.Tensor]]:
    """
    Overview:
        Retrieves the gradients of all parameters in a model.

    Arguments:
        - model (:obj:`nn.Module`): The model from which to get gradients.

    Returns:
        - List[Optional[torch.Tensor]]: A list of gradient tensors. If a parameter has no gradient,
                                         the corresponding list entry is None.
    """
    return [p.grad.detach() if p.grad is not None else None for p in model.parameters()]


def set_gradients(model: nn.Module, gradients: List[Optional[torch.Tensor]]) -> None:
    """
    Overview:
        Sets the gradients for all parameters in a model.

    Arguments:
        - model (:obj:`nn.Module`): The model whose gradients are to be set.
        - gradients (:obj:`List[Optional[torch.Tensor]]`): A list of gradients to assign to the model's parameters.
    """
    params = list(model.parameters())
    if len(gradients) != len(params):
        raise ValueError(f"Number of gradients ({len(gradients)}) does not match number of model parameters ({len(params)}).")

    for g, p in zip(gradients, params):
        if g is not None:
            # Ensure the gradient is on the same device as the parameter.
            p.grad = g.to(p.device)
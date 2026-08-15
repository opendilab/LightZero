"""
Overview:
    Direction generation and normalization utilities for loss landscape exploration.
    Provides methods to create random directions and compute directions between models,
    with support for various normalization schemes to ensure meaningful landscape analysis.

This module provides:
    - Random direction generation with customizable normalization
    - Target direction computation for model comparison
    - Multiple normalization methods: filter, layer, weight, and unit-norm variants
    - Weight and state_dict extraction from neural networks
    - Direction manipulation and filtering utilities

Key Functions:
    - create_random_direction: Generate normalized random direction in parameter space
    - create_target_direction: Compute direction vector between two models
    - normalize_direction: Apply normalization to ensure comparable loss landscape scaling
    - get_weights/get_states: Extract model parameters for landscape perturbation

Notes:
    - Based on "Visualizing the Loss Landscape of Neural Nets" (Li et al., 2018)
    - Normalization is crucial: unnormalized directions can scale loss by orders of magnitude
    - Filter-wise normalization preserves network architecture and is most interpretable
    - Ignoring bias and batch norm parameters recommended for cleaner landscapes
"""

import torch
import torch.nn as nn
import copy
from typing import List, Dict


def get_weights(net: nn.Module) -> List[torch.Tensor]:
    """
    Overview:
        Extract all trainable parameters from a PyTorch network as a list of tensors.

    Arguments:
        - net (:obj:`torch.nn.Module`): PyTorch model to extract weights from

    Returns:
        - weights (:obj:`List[torch.Tensor]`): List of parameter tensors in order of net.parameters()

    Examples::
        >>> model = torch.nn.Linear(10, 5)
        >>> weights = get_weights(model)
        >>> print(len(weights))  # 2 (weight and bias)
    """
    return [p.data for p in net.parameters()]


def get_states(net: nn.Module) -> Dict:
    """
    Overview:
        Get state_dict from network, including both parameters and buffers (e.g., batch norm statistics).

    Arguments:
        - net (:obj:`torch.nn.Module`): PyTorch model to extract state from

    Returns:
        - state_dict (:obj:`dict`): State dictionary containing all model parameters and buffers

    Notes:
        - Includes batch norm running mean/variance which are important for loss landscape
        - Uses deepcopy to avoid modifying original model state
    """
    return copy.deepcopy(net.state_dict())


def get_random_weights(weights: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    Overview:
        Generate random direction vector matching the shapes and device of given weights.

    Arguments:
        - weights (:obj:`List[torch.Tensor]`): List of weight tensors to match shapes

    Returns:
        - direction (:obj:`List[torch.Tensor]`): List of random direction tensors with standard normal distribution

    Notes:
        - Maintains device and dtype of input weights
        - Each tensor independently sampled from N(0,1)
    """
    return [torch.randn(w.size(), device=w.device, dtype=w.dtype) for w in weights]


def get_random_states(states: Dict) -> List[torch.Tensor]:
    """
    Overview:
        Generate random direction vector matching state_dict structure.

    Arguments:
        - states (:obj:`dict`): State dictionary from model.state_dict()

    Returns:
        - direction (:obj:`List[torch.Tensor]`): List of random tensors matching state_dict values

    Notes:
        - Used for directions that include batch norm statistics
    """
    return [torch.randn(w.size(), device=w.device, dtype=w.dtype) for k, w in states.items()]


def get_diff_weights(weights: List[torch.Tensor], weights2: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    Overview:
        Compute direction (difference) from source weights to target weights.
        This defines the direction for linear interpolation between two models.

    Arguments:
        - weights (:obj:`List[torch.Tensor]`): Source weight tensors
        - weights2 (:obj:`List[torch.Tensor]`): Target weight tensors (same structure as weights)

    Returns:
        - direction (:obj:`List[torch.Tensor]`): Direction vectors (weights2 - weights)

    Notes:
        - Used for comparing two trained models or tracking training dynamics
        - Unnormalized: scales are preserved from original weights
    """
    return [w2 - w for (w, w2) in zip(weights, weights2)]


def get_diff_states(states: Dict, states2: Dict) -> List[torch.Tensor]:
    """
    Overview:
        Compute direction (difference) from source state_dict to target state_dict.

    Arguments:
        - states (:obj:`dict`): Source state dictionary
        - states2 (:obj:`dict`): Target state dictionary (same keys as states)

    Returns:
        - direction (:obj:`List[torch.Tensor]`): Direction vectors extracted from state dicts

    Notes:
        - Handles both parameters and buffers from state_dict
    """
    return [v2 - v for (k, v), (k2, v2) in zip(states.items(), states2.items())]


def normalize_direction(direction: List[torch.Tensor], weights: List[torch.Tensor], norm: str = 'filter') -> None:
    """
    Overview:
        Rescale direction vector to match the norm scale of weights.
        Crucial for meaningful loss landscape visualization since loss sensitivity varies by layer.

    Arguments:
        - direction (:obj:`List[torch.Tensor]`): Direction tensors to normalize (modified in-place)
        - weights (:obj:`List[torch.Tensor]`): Reference weight tensors for scaling
        - norm (:obj:`str`, optional): Normalization scheme. Default is 'filter'.
            - 'filter': Normalize each filter independently (best for CNNs with conv layers)
            - 'layer': Normalize entire layer to unity (best for MLPs)
            - 'weight': Scale direction by weight magnitude (rarely used)
            - 'dfilter': Unit norm per filter (assumes orthogonal directions)
            - 'dlayer': Unit norm per layer (assumes orthogonal directions)

    Notes:
        - Modifies direction tensors in-place
        - Filter-wise normalization preserves layer structure and is most interpretable
        - Critical for fair landscape comparison across layers with different parameter scales
        - 1D tensors (biases, BN params) usually ignored in preprocessing step

    Examples::
        >>> direction = [torch.randn(3, 3, 3, 3) for _ in range(2)]  # Conv layer + something else
        >>> weights = [torch.randn(3, 3, 3, 3) for _ in range(2)]
        >>> normalize_direction(direction, weights, norm='filter')
        >>> # Now direction magnitude matches weights at filter level
    """
    if norm == 'filter':
        for d, w in zip(direction, weights):
            d.mul_(w.norm() / (d.norm() + 1e-10))
    elif norm == 'layer':
        direction.mul_(weights.norm() / direction.norm())
    elif norm == 'weight':
        direction.mul_(weights)
    elif norm == 'dfilter':
        for d in direction:
            d.div_(d.norm() + 1e-10)
    elif norm == 'dlayer':
        direction.div_(direction.norm())


def normalize_directions_for_weights(direction: List[torch.Tensor], weights: List[torch.Tensor], norm: str = 'filter', ignore: str = 'biasbn') -> None:
    """
    Overview:
        Normalize all direction vectors for weight-based landscape exploration.
        Handles both high-dimensional weight layers and 1D biases/batch norm parameters.

    Arguments:
        - direction (:obj:`List[torch.Tensor]`): List of direction tensors to normalize
        - weights (:obj:`List[torch.Tensor]`): List of weight tensors for reference scaling
        - norm (:obj:`str`, optional): Normalization method. Default is 'filter'.
        - ignore (:obj:`str`, optional): Parameters to ignore. Default is 'biasbn'.
            - 'biasbn': Zero out 1D tensors (biases and batch norm parameters)
            - Other: Copy structure from weights (unusual, not recommended)

    Notes:
        - 1D tensors (1x1 convolutions, biases) are typically zeroed for cleaner landscapes
        - High-dimensional tensors normalized using specified norm scheme
        - Preprocessing step before landscape computation
    """
    assert len(direction) == len(weights)
    for d, w in zip(direction, weights):
        if d.dim() <= 1:
            if ignore == 'biasbn':
                d.fill_(0)  # Ignore 1D weights (bias, BN)
            else:
                d.copy_(w)
        else:
            normalize_direction(d, w, norm)


def normalize_directions_for_states(direction: List[torch.Tensor], states: Dict, norm: str = 'filter', ignore: str = 'biasbn') -> None:
    """
    Overview:
        Normalize all direction vectors for state_dict-based landscape exploration.
        Handles model parameters, batch norm buffers, and other state elements.

    Arguments:
        - direction (:obj:`List[torch.Tensor]`): List of direction tensors to normalize
        - states (:obj:`dict`): State dictionary for reference scaling
        - norm (:obj:`str`, optional): Normalization method. Default is 'filter'.
        - ignore (:obj:`str`, optional): Parameters to ignore. Default is 'biasbn'.

    Notes:
        - Used when perturbing both weights and batch norm running statistics
        - Includes batch norm buffers (running_mean, running_var) which affect loss
    """
    assert len(direction) == len(states)
    for d, (k, w) in zip(direction, states.items()):
        if d.dim() <= 1:
            if ignore == 'biasbn':
                d.fill_(0)
            else:
                d.copy_(w)
        else:
            normalize_direction(d, w, norm)


def create_random_direction(net: nn.Module, dir_type: str = 'weights', ignore: str = 'biasbn', norm: str = 'filter') -> List[torch.Tensor]:
    """
    Overview:
        Create a single random direction for 1D landscape exploration.
        Direction is normalized to ensure comparable loss scaling across layers.

    Arguments:
        - net (:obj:`torch.nn.Module`): PyTorch model to create direction for
        - dir_type (:obj:`str`, optional): Parameter type to perturb. Default is 'weights'.
            - 'weights': Perturb only trainable parameters
            - 'states': Perturb parameters + batch norm buffers
        - ignore (:obj:`str`, optional): Parameters to ignore. Default is 'biasbn'.
        - norm (:obj:`str`, optional): Normalization method. Default is 'filter'.

    Returns:
        - direction (:obj:`List[torch.Tensor]`): Random direction tensors, normalized and possibly filtered

    Notes:
        - Random directions sampled from standard normal distribution
        - Essential for fair landscape exploration across different layer scales
        - Recommended for quick landscape exploration (less expensive than target directions)
        - Filter-wise normalization (default) is most interpretable and stable

    Examples::
        >>> model = torchvision.models.resnet18()
        >>> direction = create_random_direction(model, norm='filter', ignore='biasbn')
        >>> # Now can use this direction for 1D landscape: w(t) = w_0 + t*direction
    """
    if dir_type == 'weights':
        weights = get_weights(net)
        direction = get_random_weights(weights)
        normalize_directions_for_weights(direction, weights, norm, ignore)
    elif dir_type == 'states':
        states = get_states(net)
        direction = get_random_states(states)
        normalize_directions_for_states(direction, states, norm, ignore)
    else:
        raise ValueError(f"Unknown dir_type: {dir_type}")

    return direction


def create_target_direction(net: nn.Module, net2: nn.Module, dir_type: str = 'states') -> List[torch.Tensor]:
    """
    Overview:
        Create direction vector from source model to target model.
        Enables analysis of paths in weight space between different solutions.

    Arguments:
        - net (:obj:`torch.nn.Module`): Source model (typically the one we're analyzing)
        - net2 (:obj:`torch.nn.Module`): Target model (same architecture as net)
        - dir_type (:obj:`str`, optional): Parameter type for direction. Default is 'states'.

    Returns:
        - direction (:obj:`List[torch.Tensor]`): Direction vectors from net to net2 (net2 - net)

    Notes:
        - Direction is NOT normalized (preserves parameter-space distance)
        - Useful for studying:
            * Path from initialization to trained model
            * Connection between different trained solutions
            * Effect of various training procedures
        - Use case: Check if different models lie on connected loss valleys
        - Computation cost: O(1) after models are loaded

    Examples::
        >>> model1 = load_model('checkpoint_epoch_50.pt')
        >>> model2 = load_model('checkpoint_epoch_100.pt')
        >>> # Examine loss landscape along path from epoch 50 to 100
        >>> direction = create_target_direction(model1, model2)
        >>> # Use as single direction for 1D landscape
    """
    assert net2 is not None

    if dir_type == 'weights':
        w = get_weights(net)
        w2 = get_weights(net2)
        direction = get_diff_weights(w, w2)
    elif dir_type == 'states':
        s = get_states(net)
        s2 = get_states(net2)
        direction = get_diff_states(s, s2)
    else:
        raise ValueError(f"Unknown dir_type: {dir_type}")

    return direction


def ignore_biasbn(directions: List[torch.Tensor]) -> None:
    """
    Overview:
        Zero out all 1D tensors in direction list (biases and batch norm parameters).
        Used to simplify landscape when not interested in bias/BN effects.

    Arguments:
        - directions (:obj:`List[torch.Tensor]`): Direction tensors to filter (modified in-place)

    Notes:
        - Permanently removes bias and batch norm contributions from direction
        - Useful for cleaner visualizations and to focus on weight layer interactions
        - Applied to direction list, not individual directions
    """
    for d in directions:
        if d.dim() <= 1:
            d.fill_(0)

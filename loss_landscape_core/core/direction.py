"""
Direction generation and normalization utilities.

Supports:
- Random direction generation
- Target direction (between two models)
- Various normalization methods (filter, layer, weight)
"""

import torch
import copy


def get_weights(net):
    """Extract parameters from network as list of tensors.

    Args:
        net: PyTorch model

    Returns:
        List of parameter tensors
    """
    return [p.data for p in net.parameters()]


def get_states(net):
    """Get state_dict from network.

    Args:
        net: PyTorch model

    Returns:
        state_dict of the network
    """
    return copy.deepcopy(net.state_dict())


def get_random_weights(weights):
    """Generate random direction with same shape as weights.

    Args:
        weights: List of weight tensors

    Returns:
        List of random direction tensors
    """
    return [torch.randn(w.size(), device=w.device, dtype=w.dtype) for w in weights]


def get_random_states(states):
    """Generate random direction with same shape as state_dict.

    Args:
        states: state_dict of network

    Returns:
        List of random direction tensors
    """
    return [torch.randn(w.size(), device=w.device, dtype=w.dtype) for k, w in states.items()]


def get_diff_weights(weights, weights2):
    """Compute direction from weights to weights2.

    Args:
        weights: Source weights
        weights2: Target weights

    Returns:
        Direction (difference) between two weight sets
    """
    return [w2 - w for (w, w2) in zip(weights, weights2)]


def get_diff_states(states, states2):
    """Compute direction from states to states2.

    Args:
        states: Source state_dict
        states2: Target state_dict

    Returns:
        List of direction tensors
    """
    return [v2 - v for (k, v), (k2, v2) in zip(states.items(), states2.items())]


def normalize_direction(direction, weights, norm='filter'):
    """Rescale direction to have similar norm as weights.

    Args:
        direction: List of direction tensors for one layer
        weights: List of weight tensors for one layer
        norm: Normalization method
              - 'filter': normalize at filter level
              - 'layer': normalize at layer level
              - 'weight': scale by weight magnitude
              - 'dfilter': unit norm at filter level
              - 'dlayer': unit norm at layer level
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


def normalize_directions_for_weights(direction, weights, norm='filter', ignore='biasbn'):
    """Normalize direction for weight-based approach.

    Args:
        direction: List of direction tensors
        weights: List of weight tensors
        norm: Normalization method
        ignore: 'biasbn' to ignore bias and batch norm parameters
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


def normalize_directions_for_states(direction, states, norm='filter', ignore='biasbn'):
    """Normalize direction for state_dict-based approach.

    Args:
        direction: List of direction tensors
        states: state_dict of network
        norm: Normalization method
        ignore: 'biasbn' to ignore bias and batch norm parameters
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


def create_random_direction(net, dir_type='weights', ignore='biasbn', norm='filter'):
    """Create a random normalized direction.

    Args:
        net: PyTorch model
        dir_type: 'weights' or 'states' (includes BN running_mean/var)
        ignore: 'biasbn' to ignore bias and BN parameters
        norm: Normalization method (filter|layer|weight|dfilter|dlayer)

    Returns:
        List of direction tensors
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


def create_target_direction(net, net2, dir_type='states'):
    """Create direction from one model to another.

    Args:
        net: Source model
        net2: Target model (same architecture as net)
        dir_type: 'weights' or 'states'

    Returns:
        List of direction tensors
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


def ignore_biasbn(directions):
    """Set bias and BN parameters in directions to zero."""
    for d in directions:
        if d.dim() <= 1:
            d.fill_(0)

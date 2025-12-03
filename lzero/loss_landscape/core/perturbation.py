"""
Overview:
    Weight perturbation utilities for loss landscape exploration.
    Provides methods to move model parameters along specified directions in parameter space.

This module provides:
    - Weight perturbation along 1D directions: w(t) = w_0 + t*d
    - Weight perturbation along 2D directions: w(x,y) = w_0 + x*d_x + y*d_y
    - Batch perturbation for efficient landscape evaluation
    - Both parameter-based and state_dict-based perturbation

Key Functions:
    - set_weights: Modify weights along direction(s) in parameter space
    - set_states: Modify full state_dict along direction(s)

Notes:
    - Essential for evaluating loss at grid points during landscape computation
    - Handles device and dtype conversions automatically
    - Supports both single and 2D direction perturbations for grid evaluation
    - In-place modifications to model parameters
"""

import torch
import torch.nn as nn
import copy
from typing import List, Dict, Optional, Union


def set_weights(net: nn.Module, weights: List[torch.Tensor], directions: Optional[List[List[torch.Tensor]]] = None, step: Optional[Union[float, List[float]]] = None) -> None:
    """
    Overview:
        Modify network weights along given direction(s), or restore original weights.
        Supports both 1D and 2D landscape exploration via parameter space perturbation.

    Arguments:
        - net (:obj:`torch.nn.Module`): PyTorch model whose weights to modify (modified in-place)
        - weights (:obj:`List[torch.Tensor]`): Original weight tensors (reference point in parameter space)
        - directions (:obj:`List[List[torch.Tensor]]`, optional): Direction(s) for perturbation
            - Single direction: [[d1, d2, ...]] for 1D: w' = w + step*d
            - Two directions: [[dx1, dx2, ...], [dy1, dy2, ...]] for 2D: w' = w + step[0]*dx + step[1]*dy
        - step (:obj:`float` or :obj:`List[float]`, optional): Step size(s) along direction(s)
            - For 1D: scalar value
            - For 2D: [step_x, step_y]

    Returns:
        None. Modifies net.parameters() in-place.

    Notes:
        - If directions=None: Restores network to original weights (useful after landscape eval)
        - Handles automatic device and dtype conversion
        - Critical for efficient landscape grid evaluation
        - 2D perturbation formula: w' = w_0 + step[0]*d_x + step[1]*d_y

    Shapes:
        - weights (:obj:`List[torch.Tensor]`): List of shape (param_shape_1, param_shape_2, ...)
        - directions: Single = [List[Tensor]], Two = [List[Tensor], List[Tensor]]

    Examples::
        >>> model = torch.nn.Linear(10, 5)
        >>> original_weights = get_weights(model)
        >>> direction = create_random_direction(model)
        >>> # Evaluate at multiple points
        >>> for x in np.linspace(-1, 1, 51):
        ...     set_weights(model, original_weights, [direction], x)
        ...     loss = criterion(model(data), target)
        >>> # Restore original weights
        >>> set_weights(model, original_weights)
    """
    if directions is None:
        # Simply restore original weights
        for (p, w) in zip(net.parameters(), weights):
            p.data.copy_(w.type(type(p.data)))
    else:
        assert step is not None, 'step must be specified if directions are given'

        if len(directions) == 2:
            # 2D perturbation: w' = w + step[0]*d1 + step[1]*d2
            dx = directions[0]
            dy = directions[1]
            changes = [d0 * step[0] + d1 * step[1] for (d0, d1) in zip(dx, dy)]
        else:
            # 1D perturbation: w' = w + step*d
            changes = [d * step for d in directions[0]]

        for (p, w, d) in zip(net.parameters(), weights, changes):
            # Ensure d is on the same device and dtype as w
            d_tensor = torch.tensor(d, device=w.device, dtype=w.dtype)
            p.data = w + d_tensor


def set_states(net: nn.Module, states: Dict, directions: Optional[List[List[torch.Tensor]]] = None, step: Optional[Union[float, List[float]]] = None) -> None:
    """
    Overview:
        Modify network state_dict along direction(s), or restore original state.
        State includes parameters AND buffers (batch norm statistics, etc.).

    Arguments:
        - net (:obj:`torch.nn.Module`): PyTorch model whose state to modify (modified in-place)
        - states (:obj:`dict`): Original state_dict from model.state_dict() (reference point)
        - directions (:obj:`List[List[torch.Tensor]]`, optional): Direction(s) for perturbation
            - Single direction: [[d1, d2, ...]] for 1D
            - Two directions: [[dx1, dx2, ...], [dy1, dy2, ...]] for 2D
        - step (:obj:`float` or :obj:`List[float]`, optional): Step size(s) along direction(s)
            - For 1D: scalar value
            - For 2D: [step_x, step_y]

    Returns:
        None. Modifies net via load_state_dict() in-place.

    Notes:
        - Includes batch norm running statistics (running_mean, running_var)
        - These buffers affect loss landscape shape significantly
        - If directions=None: Restores to original state
        - Uses state_dict interface for safe state restoration
        - Deep copy prevents corruption of original state reference

    Shapes:
        - states: Dictionary with same structure as model.state_dict()

    Examples::
        >>> model = torch.nn.Sequential(
        ...     torch.nn.Conv2d(3, 64, 3),
        ...     torch.nn.BatchNorm2d(64),
        ... )
        >>> original_states = copy.deepcopy(model.state_dict())
        >>> direction = create_random_direction(model, dir_type='states')
        >>> for x in np.linspace(-1, 1, 21):
        ...     set_states(model, original_states, [direction], x)
        ...     loss = criterion(model(data), target)
    """
    if directions is None:
        # Simply restore original state_dict
        net.load_state_dict(states)
    else:
        assert step is not None, 'step must be specified if directions are given'

        if len(directions) == 2:
            # 2D perturbation
            dx = directions[0]
            dy = directions[1]
            changes = [d0 * step[0] + d1 * step[1] for (d0, d1) in zip(dx, dy)]
        else:
            # 1D perturbation
            changes = [d * step for d in directions[0]]

        new_states = copy.deepcopy(states)
        assert len(new_states) == len(changes)

        for (k, v), d in zip(new_states.items(), changes):
            # Ensure d is on the same device and dtype as v
            d_tensor = torch.tensor(d, device=v.device, dtype=v.dtype)
            v.add_(d_tensor)

        net.load_state_dict(new_states)

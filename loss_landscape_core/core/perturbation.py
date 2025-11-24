"""
Weight perturbation along directions.

Provides functions to modify network weights along given directions.
"""

import torch
import copy


def set_weights(net, weights, directions=None, step=None):
    """Overwrite network weights or perturb them along directions.

    Args:
        net: PyTorch model
        weights: List of weight tensors (original weights)
        directions: List of direction tensors (can be single direction or 2D)
        step: Step size along direction(s)
              - If single direction: scalar step
              - If two directions: [step_x, step_y]
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


def set_states(net, states, directions=None, step=None):
    """Overwrite network state_dict or perturb it along directions.

    Args:
        net: PyTorch model
        states: Original state_dict
        directions: List of direction tensors
        step: Step size along direction(s)
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

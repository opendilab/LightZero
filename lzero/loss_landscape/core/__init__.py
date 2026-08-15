"""
Overview:
    Core functionality for loss landscape computation.
    Provides essential utilities for direction generation, model evaluation, and weight perturbation.

This module provides:
    - Direction generation (random and target) for landscape exploration
    - Direction normalization with multiple schemes (filter, layer, weight)
    - Model evaluation (loss and accuracy computation)
    - Weight perturbation along directions for grid-based landscape sampling

Key Functions:
    - create_random_direction: Generate normalized random direction
    - create_target_direction: Compute direction between two models
    - get_weights/get_states: Extract model parameters
    - eval_loss: Evaluate loss and accuracy on dataset
    - set_weights/set_states: Perturb model parameters along directions

Typical Workflow:
    1. Extract original weights: w0 = get_weights(model)
    2. Create direction: d = create_random_direction(model)
    3. For each point (x, y):
       - Perturb: set_weights(model, w0, [d], x)
       - Evaluate: loss = eval_loss(model, criterion, dataloader)
    4. Restore: set_weights(model, w0)
"""

from .direction import (
    create_random_direction,
    create_target_direction,
    get_weights,
    get_states,
)

from .evaluator import eval_loss

from .perturbation import (
    set_weights,
    set_states,
)

__all__ = [
    'create_random_direction',
    'create_target_direction',
    'get_weights',
    'get_states',
    'eval_loss',
    'set_weights',
    'set_states',
]

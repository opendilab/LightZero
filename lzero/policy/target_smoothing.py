"""Resolve categorical target regularization without conflating policy and scalar heads."""

import math


def scalar_smoothing_eps(config, head: str) -> float:
    """Explicit per-head settings override the legacy shared reward/value epsilon."""
    if head not in ('reward', 'value'):
        raise ValueError(f'Unknown scalar head: {head}')
    value = config.get(f'{head}_label_smoothing_eps')
    if value is None:
        value = config.get('label_smoothing_eps', 0.1)
    value = float(value)
    if not math.isfinite(value) or not 0 <= value < 1:
        raise ValueError(f'{head}_label_smoothing_eps must be finite and in [0, 1), got {value}')
    return value

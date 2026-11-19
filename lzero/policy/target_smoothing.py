"""Resolve categorical target regularization without conflating policy and scalar heads."""

import math


def scalar_smoothing_mode(config, head: str) -> str:
    if head not in ('reward', 'value'):
        raise ValueError(f'Unknown scalar head: {head}')
    config = config if hasattr(config, 'get') else vars(config)
    mode = config.get(f'{head}_label_smoothing_mode', 'uniform')
    if mode not in ('uniform', 'local'):
        raise ValueError(f'Unknown {head} label smoothing mode: {mode}')
    return mode


def scalar_decode_smoothing_eps(config, head: str) -> float:
    """Opt in to decoding the unsmoothed target mean while retaining smooth CE."""
    enabled = (config.get('compensate_scalar_label_smoothing', False) if hasattr(config, 'get')
               else getattr(config, 'compensate_scalar_label_smoothing', False))
    # Local smoothing already preserves its input mean. Compensating it again
    # would create exactly the scale error this switch is intended to remove.
    return (scalar_smoothing_eps(config if hasattr(config, 'get') else vars(config), head)
            if enabled and scalar_smoothing_mode(config, head) == 'uniform' else 0.0)


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

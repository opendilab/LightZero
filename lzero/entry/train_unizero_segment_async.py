from typing import Any, Optional, Tuple

import torch

from .train_muzero_segment_async import train_muzero_segment_async


def train_unizero_segment_async(
        input_cfg: Tuple[dict, dict],
        seed: int = 0,
        model: Optional[torch.nn.Module] = None,
        model_path: Optional[str] = None,
        max_train_iter: Optional[int] = int(1e10),
        max_env_step: Optional[int] = int(1e10),
) -> Any:
    """
    Ray-based asynchronous UniZero segment training entry.

    UniZero uses the same segment collector/evaluator contract as MuZero, while
    the generic async driver handles UniZero-specific train data shape and
    world-model cache cleanup.
    """
    return train_muzero_segment_async(
        input_cfg=input_cfg,
        seed=seed,
        model=model,
        model_path=model_path,
        max_train_iter=max_train_iter,
        max_env_step=max_env_step,
    )

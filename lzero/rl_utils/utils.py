import numpy as np


def get_max_entropy(action_shape: int) -> None:
    p = 1.0 / action_shape
    return -action_shape * p * np.log2(p)

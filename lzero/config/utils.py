import numpy as np
from ding.utils import get_world_size
from easydict import EasyDict


def lz_to_ddp_config(cfg: EasyDict) -> EasyDict:
    r"""
    Overview:
        Convert the LightZero-style config to ddp config
    Arguments:
        - cfg (:obj:`EasyDict`): The config to be converted
    Returns:
        - cfg (:obj:`EasyDict`): The converted config
    """
    w = get_world_size()
    # Generalized handling for multiple keys
    keys_to_scale = ['batch_size', 'n_episode', 'num_segments']
    for key in keys_to_scale:
        if key in cfg.policy:
            cfg.policy[key] = int(np.ceil(cfg.policy[key] / w))
    return cfg

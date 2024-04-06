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
    cfg.policy.batch_size = int(np.ceil(cfg.policy.batch_size / w))
    cfg.policy.n_episode = int(np.ceil(cfg.policy.n_episode) / w)
    return cfg

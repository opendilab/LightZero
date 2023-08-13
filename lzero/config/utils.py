import numpy as np
import torch.distributed as dist
from ding.utils.default_helper import error_wrapper
from easydict import EasyDict


def get_world_size() -> int:
    r"""
    Overview:
        Get the world_size(total process number in data parallel training)
    """
    # return int(os.environ.get('SLURM_NTASKS', 1))
    return error_wrapper(dist.get_world_size, 1)()

def to_ddp_config(cfg: EasyDict) -> EasyDict:
    w = get_world_size()
    cfg.policy.batch_size = int(np.ceil(cfg.policy.batch_size / w))
    cfg.policy.n_episode = int(np.ceil(cfg.policy.n_episode) / w)
    return cfg

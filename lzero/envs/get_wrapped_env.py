import gym
from easydict import EasyDict

from ding.envs import DingEnvWrapper
from lzero.envs.wrappers import ActionDiscretizationEnvWrapper, LightZeroEnvWrapper


def get_wrappered_env(wrapper_cfg: EasyDict, env_id: str):
    """
    Overview:
        Returns a new environment with one or more wrappers applied to it.
    Arguments:
        - wrapper_cfg (:obj:`EasyDict`): A dictionary containing configuration settings for the wrappers.
       -  env_id (:obj:`str`): The name of the environment to create.
    Returns:
        A callable that creates the wrapped environment.
    """
    if wrapper_cfg.manually_discretization:
        return lambda: DingEnvWrapper(
            gym.make(env_id),
            cfg={
                'env_wrapper': [
                    lambda env: ActionDiscretizationEnvWrapper(env, wrapper_cfg), lambda env:
                    LightZeroEnvWrapper(env, wrapper_cfg)
                ]
            }
        )
    else:
        return lambda: DingEnvWrapper(
            gym.make(env_id), cfg={'env_wrapper': [lambda env: LightZeroEnvWrapper(env, wrapper_cfg)]}
        )

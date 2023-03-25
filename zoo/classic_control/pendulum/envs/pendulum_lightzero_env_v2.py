import copy
from typing import Optional

import gym
import numpy as np
import torch
from ding.envs import BaseEnv, BaseEnvTimestep
from ding.envs.common.common_function import affine_transform
from ding.torch_utils import to_ndarray
from ding.utils import ENV_REGISTRY
from easydict import EasyDict

from zoo.classic_control.pendulum.envs.lightzero_env_wrapper import ObsActionMaskToPlayWrapper
from ding.envs import DingEnvWrapper

ding_env = DingEnvWrapper(
    gym.make('Pendulum-v1'),
    cfg={
        'env_wrapper': [
            lambda env: ObsActionMaskToPlayWrapper(env),
        ]
    }
)


@ENV_REGISTRY.register('pendulum_lightzero')
class PendulumLightZeroEnv():

    ding_env = DingEnvWrapper(
        gym.make('Pendulum-v1'),
        cfg={
            'env_wrapper': [
                lambda env: ObsActionMaskToPlayWrapper(env),
            ]
        }
    )


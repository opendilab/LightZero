

import gym
from ding.utils import ENV_REGISTRY

from zoo.classic_control.pendulum.envs.lightzero_env_wrapper import ObsActionMaskToPlayWrapper
from ding.envs import DingEnvWrapper

lightzero_env = DingEnvWrapper(
    gym.make('Pendulum-v1'),
    cfg={
        'env_wrapper': [
            lambda env: ObsActionMaskToPlayWrapper(env),
        ]
    }
)


@ENV_REGISTRY.register('pendulum_lightzero')
class PendulumLightZeroEnv():

    lightzero_env = DingEnvWrapper(
        gym.make('Pendulum-v1'),
        cfg={
            'env_wrapper': [
                lambda env: ObsActionMaskToPlayWrapper(env),
            ]
        }
    )


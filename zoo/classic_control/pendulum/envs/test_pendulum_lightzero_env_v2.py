import gym
import numpy as np
from ding.envs import DingEnvWrapper
from easydict import EasyDict

from zoo.classic_control.pendulum.envs.lightzero_env_wrapper import ObsActionMaskToPlayWrapper

lightzero_env = DingEnvWrapper(
    gym.make('Pendulum-v1'),
    cfg={
        'env_wrapper': [
            lambda env: ObsActionMaskToPlayWrapper(env),
        ]
    }
)

print(lightzero_env.observation_space, lightzero_env.action_space, lightzero_env.reward_space)
cfg = EasyDict(dict(
    collector_env_num=16,
    evaluator_env_num=3,
    is_train=True,
))
l1 = lightzero_env.create_collector_env_cfg(cfg)
assert isinstance(l1, list)
l1 = lightzero_env.create_evaluator_env_cfg(cfg)
assert isinstance(l1, list)

obs = lightzero_env.reset()
print("obs: ", obs)

assert isinstance(obs, dict)
assert isinstance(obs['observation'], np.ndarray) and obs['observation'].shape == (3, 1, 1)
assert obs['action_mask'] is None and obs['to_play'] == -1

action = lightzero_env.random_action()
print('random_action: {}, action_space: {}'.format(action.shape, lightzero_env.action_space))

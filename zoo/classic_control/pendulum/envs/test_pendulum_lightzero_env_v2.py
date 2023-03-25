import gym
import numpy as np
from ding.envs import DingEnvWrapper
from easydict import EasyDict

from zoo.classic_control.pendulum.envs.lightzero_env_wrapper import ObsActionMaskToPlayWrapper

ding_env = DingEnvWrapper(
    gym.make('Pendulum-v1'),
    cfg={
        'env_wrapper': [
            lambda env: ObsActionMaskToPlayWrapper(env),
        ]
    }
)

print(ding_env.observation_space, ding_env.action_space, ding_env.reward_space)
cfg = EasyDict(dict(
    collector_env_num=16,
    evaluator_env_num=3,
    is_train=True,
))
l1 = ding_env.create_collector_env_cfg(cfg)
assert isinstance(l1, list)
l1 = ding_env.create_evaluator_env_cfg(cfg)
assert isinstance(l1, list)

obs = ding_env.reset()
print("obs: ", obs)

assert isinstance(obs, dict)
assert isinstance(obs['observation'], np.ndarray) and obs['observation'].shape == (3, 1, 1)
assert obs['action_mask'] is None and obs['to_play'] == -1

action = ding_env.random_action()
print('random_action: {}, action_space: {}'.format(action.shape, ding_env.action_space))

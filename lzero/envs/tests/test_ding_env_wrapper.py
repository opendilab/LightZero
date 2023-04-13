from easydict import EasyDict
import pytest
import gym
import numpy as np

from ding.envs import DingEnvWrapper


@pytest.mark.unittest
class TestDingEnvWrapper:

    def test(self):
        env_id = 'Pendulum-v1'
        env = gym.make(env_id)
        ding_env = DingEnvWrapper(env=env)
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

        assert isinstance(obs, np.ndarray)
        action = ding_env.random_action()
        print('random_action: {}, action_space: {}'.format(action.shape, ding_env.action_space))

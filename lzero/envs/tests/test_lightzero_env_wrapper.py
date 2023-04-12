import pytest

from ding.envs import DingEnvWrapper
from lzero.envs.wrappers import ActionDiscretizationEnvWrapper, LightZeroEnvWrapper
from easydict import EasyDict
import gym
import numpy as np


@pytest.mark.unittest
class TestLightZeroEnvWrapper:

    def test_continuous_pendulum(self):
        env_cfg = EasyDict(
            dict(
                env_name='Pendulum-v1',
                manually_discretization=False,
                continuous=True,
                each_dim_disc_size=None,
                is_train=True,
            )
        )

        lightzero_env = DingEnvWrapper(
            gym.make(env_cfg.env_name), cfg={'env_wrapper': [
                lambda env: LightZeroEnvWrapper(env, env_cfg),
            ]}
        )

        obs = lightzero_env.reset()
        print("obs: ", obs)

        print(lightzero_env.observation_space, lightzero_env.action_space, lightzero_env.reward_space)

        assert isinstance(obs, dict)
        assert isinstance(obs['observation'], np.ndarray) and obs['observation'].shape == (3, )
        assert obs['action_mask'] is None and obs['to_play'] == -1

        action = lightzero_env.random_action()

        print('random_action: {}, action_space: {}'.format(action.shape, lightzero_env.action_space))

    def test_discretization_pendulum(self):
        env_cfg = EasyDict(
            dict(
                env_name='Pendulum-v1',
                manually_discretization=True,
                continuous=False,
                each_dim_disc_size=11,
                is_train=True,
            )
        )

        lightzero_env = DingEnvWrapper(
            gym.make(env_cfg.env_name),
            cfg={
                'env_wrapper': [
                    lambda env: ActionDiscretizationEnvWrapper(env, env_cfg),
                    lambda env: LightZeroEnvWrapper(env, env_cfg),
                ]
            }
        )

        obs = lightzero_env.reset()
        print("obs: ", obs)

        print(lightzero_env.observation_space, lightzero_env.action_space, lightzero_env.reward_space)

        assert isinstance(obs, dict)
        assert isinstance(obs['observation'], np.ndarray) and obs['observation'].shape == (3, )
        assert obs['action_mask'].sum() == 11 and obs['to_play'] == -1

        action = lightzero_env.random_action()

        print('random_action: {}, action_space: {}'.format(action.shape, lightzero_env.action_space))

    def test_continuous_bipedalwalker(self):
        env_cfg = EasyDict(
            dict(
                env_name='BipedalWalker-v3',
                manually_discretization=False,
                continuous=True,
                each_dim_disc_size=4,
                is_train=True,
            )
        )

        lightzero_env = DingEnvWrapper(
            gym.make(env_cfg.env_name), cfg={'env_wrapper': [
                lambda env: LightZeroEnvWrapper(env, env_cfg),
            ]}
        )

        obs = lightzero_env.reset()
        print("obs: ", obs)

        print(lightzero_env.observation_space, lightzero_env.action_space, lightzero_env.reward_space)

        assert isinstance(obs, dict)
        assert isinstance(obs['observation'], np.ndarray) and obs['observation'].shape == (24, )
        assert obs['action_mask'] is None and obs['to_play'] == -1

        action = lightzero_env.random_action()

        print('random_action: {}, action_space: {}'.format(action.shape, lightzero_env.action_space))

    def test_discretization_bipedalwalker(self):
        env_cfg = EasyDict(
            dict(
                env_name='BipedalWalker-v3',
                manually_discretization=True,
                continuous=False,
                each_dim_disc_size=4,
                is_train=True,
            )
        )

        lightzero_env = DingEnvWrapper(
            gym.make(env_cfg.env_name),
            cfg={
                'env_wrapper': [
                    lambda env: ActionDiscretizationEnvWrapper(env, env_cfg),
                    lambda env: LightZeroEnvWrapper(env, env_cfg),
                ]
            }
        )

        obs = lightzero_env.reset()
        print("obs: ", obs)

        print(lightzero_env.observation_space, lightzero_env.action_space, lightzero_env.reward_space)

        assert isinstance(obs, dict)
        assert isinstance(obs['observation'], np.ndarray) and obs['observation'].shape == (24, )
        assert obs['action_mask'].sum() == 256 and obs['to_play'] == -1

        action = lightzero_env.random_action()

        print('random_action: {}, action_space: {}'.format(action.shape, lightzero_env.action_space))

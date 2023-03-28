import pytest

from ding.envs import DingEnvWrapper
from ding.envs.env_wrappers import *
from lzero.envs.lightzero_env_wrapper import LightZeroEnvWrapper


@pytest.mark.unittest
class TestLightZeroEnvWrapper:

    def test(self):
        env_cfg = EasyDict(dict(
            env_name='Pendulum-v1',
            continuous=True,
            discretization=False,
            is_train=True,
        ))

        lightzero_env = DingEnvWrapper(
            gym.make(env_cfg.env_name),
            cfg={
                'env_wrapper': [
                    lambda env: LightZeroEnvWrapper(env, env_cfg)
                ]
            }
        )

        print(lightzero_env.observation_space, lightzero_env.action_space, lightzero_env.reward_space)

        obs = lightzero_env.reset()
        print("obs: ", obs)

        assert isinstance(obs, dict)
        assert isinstance(obs['observation'], np.ndarray) and obs['observation'].shape == (3, 1, 1)
        assert obs['action_mask'] is None and obs['to_play'] == -1

        action = lightzero_env.random_action()

        print('random_action: {}, action_space: {}'.format(action.shape, lightzero_env.action_space))

    def test_discretization(self):
        env_cfg = EasyDict(dict(
            env_name='Pendulum-v1',
            continuous=False,
            discretization=True,
            is_train=True,
        ))

        lightzero_env = DingEnvWrapper(
            gym.make(env_cfg.env_name),
            cfg={
                'env_wrapper': [
                    lambda env: LightZeroEnvWrapper(env, env_cfg)
                ]
            }
        )

        print(lightzero_env.observation_space, lightzero_env.action_space, lightzero_env.reward_space)

        obs = lightzero_env.reset()
        print("obs: ", obs)

        assert isinstance(obs, dict)
        assert isinstance(obs['observation'], np.ndarray) and obs['observation'].shape == (3, 1, 1)
        assert obs['action_mask'].sum() == 11 and obs['to_play'] == -1

        action = lightzero_env.random_action()

        print('random_action: {}, action_space: {}'.format(action.shape, lightzero_env.action_space))

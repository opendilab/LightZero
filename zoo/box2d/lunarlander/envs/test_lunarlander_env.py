import pytest
import numpy as np
from easydict import EasyDict
from zoo.box2d.lunarlander.envs import LunarLanderEnv


@pytest.mark.envtest
@pytest.mark.parametrize(
    'cfg', [
        EasyDict({
            'env_name': 'LunarLander-v2',
            'act_scale': False,
            'replay_path': None,
            'replay_path_gif': None,
            'save_replay_gif': False,
        }),
        EasyDict({
            'env_name': 'LunarLanderContinuous-v2',
            'act_scale': True,
            'replay_path': None,
            'replay_path_gif': None,
            'save_replay_gif': False,
        })
    ]
)
class TestLunarLanderEnvEnv:
    """
        Overview:
            The env created for testing the LunarLander environment.
            It is used to check information such as observation space, action space and reward space.
    """

    def test_naive(self, cfg):
        env = LunarLanderEnv(cfg)
        env.seed(314)
        assert env._seed == 314
        obs = env.reset()
        assert obs['observation'].shape == (8, )
        for i in range(10):
            random_action = env.random_action()
            timestep = env.step(random_action)
            print(timestep)
            assert isinstance(timestep.obs['observation'], np.ndarray)
            assert isinstance(timestep.done, bool)
            assert timestep.obs['observation'].shape == (8, )
            assert timestep.reward.shape == (1, )
            assert timestep.reward >= env.reward_space.low
            assert timestep.reward <= env.reward_space.high
        print(env.observation_space, env.action_space, env.reward_space)
        env.close()

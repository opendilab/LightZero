import numpy as np
import pytest
from easydict import EasyDict
from zoo.CrowdSim.envs.crowdsim_lightzero_env import CrowdSimEnv

mcfg=EasyDict(
        env_name='CrowdSim-v0',
        dataset = 'purdue',
        robot_num = 2,
        human_num = 59, # purdue
        one_uav_action_space = [[0, 0], [30, 0], [-30, 0], [0, 30], [0, -30]]
        )

@ pytest.mark.envtest

class TestCrowdSimEnv:
    def test_naive(self):
        env = CrowdSimEnv(mcfg)
        env.seed(314)
        assert env._seed == 314
        obs = env.reset()
        assert isinstance(obs['observation'], dict)
        assert obs['observation']['robot_state'].shape == (2, 4)
        assert obs['observation']['human_state'].shape == (59, 4)
        for i in range(10):
            random_action = env.random_action()
            timestep = env.step(random_action)
            print(timestep)
            assert isinstance(timestep.obs['observation'], dict)
            assert timestep.obs['observation']['robot_state'].shape == (2, 4)
            assert timestep.obs['observation']['human_state'].shape == (59, 4)
            assert isinstance(timestep.done, bool)
            assert timestep.reward.shape == (1, )
        print(env.observation_space, env.action_space, env.reward_space)
        env.close()

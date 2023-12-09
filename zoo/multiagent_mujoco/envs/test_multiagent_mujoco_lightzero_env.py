from time import time
import pytest
import numpy as np
from easydict import EasyDict
from zoo.multiagent_mujoco.envs import MAMujocoEnvLZ


@pytest.mark.envtest
@pytest.mark.parametrize(
    'cfg', [
        EasyDict({
            'env_name': 'mujoco_lightzero',
            'scenario': 'Ant-v2',
            'agent_conf': "2x4d",
            'agent_obsk': 2,
            'add_agent_id': False,
            'episode_limit': 1000,
        },) 
    ]
)

class TestMAMujocoEnvLZ:
    def test_naive(self, cfg):
        env = MAMujocoEnvLZ(cfg)
        env.seed(314)
        assert env._seed == 314
        obs = env.reset()
        assert isinstance(obs, dict)
        for i in range(10):
            random_action = env.random_action()
            timestep = env.step(random_action[0])
            print(timestep)
            assert isinstance(timestep.obs, dict)
            assert isinstance(timestep.done, bool)
            assert timestep.obs['observation']['global_state'].shape == (2, 111)
            assert timestep.obs['observation']['agent_state'].shape == (2, 54)
            assert timestep.reward.shape == (1, )
            assert isinstance(timestep, tuple)
        print(env.observation_space, env.action_space, env.reward_space)
        env.close()

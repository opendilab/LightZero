import numpy as np
from easydict import EasyDict
from zoo.CrowdSim.envs.CrowdSim_env import CrowdSimEnv

mcfg = EasyDict(
    env_name='CrowdSim-v0',
    dataset='purdue',
    robot_num=2,
    human_num=59,  # purdue
    one_uav_action_space=[[0, 0], [30, 0], [-30, 0], [0, 30], [0, -30]]
)


def test_naive(cfg):
    env = CrowdSimEnv(cfg)
    env.seed(314)
    assert env._seed == 314
    obs = env.reset()
    assert obs['observation'].shape == (244, )
    for i in range(10):
        random_action = env.random_action()
        timestep = env.step(random_action)
        print(timestep)
        assert isinstance(timestep.obs['observation'], np.ndarray)
        assert isinstance(timestep.done, bool)
        assert timestep.obs['observation'].shape == (244, )
        assert timestep.reward.shape == (1, )
    print(env.observation_space, env.action_space, env.reward_space)
    env.close()


test_naive(mcfg)

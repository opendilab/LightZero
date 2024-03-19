import pytest
import numpy as np
import gym
import sys
from easydict import EasyDict
from zoo.smac.envs.smac_env_lz import SMACLZEnv

agent_num = 3

# @pytest.mark.envtest
# class TestAtariEnv:

def test_samc():
    cfg = dict(
    map_name='3s_vs_5z',
    difficulty=7,
    reward_type='original',
    agent_num=agent_num,
    )
    cfg = EasyDict(cfg)
    samc_env = SMACLZEnv(cfg)
    samc_env.seed(0)
    obs = samc_env.reset()
    assert isinstance(obs, dict)
    while True:
        random_action = np.random.randint(0, 6, size=(agent_num, ))
        timestep = samc_env.step(random_action)
        assert isinstance(obs, dict)
        assert set(['agent_state', 'global_state', 'agent_specifig_global_state']).issubset(timestep[0]['states'])
        assert timestep.reward.shape == (1, )
        if timestep.done:
            assert 'eval_episode_return' in timestep.info, timestep.info
            break
    print(samc_env.observation_space, samc_env.action_space, samc_env.reward_space)
    print('eval_episode_return: {}'.format(timestep.info['eval_episode_return']))
    samc_env.close()

test_samc()
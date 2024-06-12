import pytest
import numpy as np
from easydict import EasyDict
from zoo.smac.envs.smac_env_lightzero_env import SMACLZEnv

agent_num = 3


@pytest.mark.envtest
class TestSmacEnv:

    def test_smac(self):
        cfg = dict(
            map_name='3s_vs_5z',
            difficulty=7,
            reward_type='original',
            agent_num=agent_num,
        )
        cfg = EasyDict(cfg)
        smac_env = SMACLZEnv(cfg)
        smac_env.seed(0)
        obs = smac_env.reset()
        assert isinstance(obs, dict)
        while True:
            random_action = np.random.randint(0, 6, size=(agent_num,))
            timestep = smac_env.step(random_action)
            assert isinstance(obs, dict)
            assert set(['agent_state', 'global_state', 'agent_specific_global_state']).issubset(
                timestep.obs['observation'])
            assert timestep.reward.shape == (1,)
            if timestep.done:
                assert 'eval_episode_return' in timestep.info, timestep.info
                break
        print(smac_env.observation_space, smac_env.action_space, smac_env.reward_space)
        print('eval_episode_return: {}'.format(timestep.info['eval_episode_return']))
        smac_env.close()


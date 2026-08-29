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
            # access random action according to samc_env.action_mask (shape: (agent_num, 14), bool)
            random_actions = np.zeros(agent_num, dtype=int)
            for agent in range(agent_num):
                available_actions = np.where(smac_env.action_mask[agent])[0]
                if available_actions.size > 0:
                    random_actions[agent] = np.random.choice(available_actions)
                else:
                    # Handle case where no actions are available (all False)
                    random_actions[agent] = -1  # or any other placeholder value
            # random_action = np.random.randint(1, 6, size=(agent_num, ))

            timestep = smac_env.step(random_actions)
            assert isinstance(obs, dict)
            assert set(['agent_state', 'global_state', 'agent_specific_global_state']).issubset(timestep.obs['observation']['states'])
            assert timestep.reward.shape == (1, )
            if timestep.done:
                assert 'eval_episode_return' in timestep.info, timestep.info
                break
        print('eval_episode_return: {}'.format(timestep.info['eval_episode_return']))
        smac_env.close()

# test_smac()
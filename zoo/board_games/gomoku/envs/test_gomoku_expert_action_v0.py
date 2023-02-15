import pytest
from easydict import EasyDict

from zoo.board_games.gomoku.envs.gomoku_env import GomokuEnv


@pytest.mark.envtest
class TestExpertActionV0:

    def test_naive(self):
        cfg = EasyDict(
            board_size=6,
            battle_mode='self_play_mode',
            prob_random_agent=0,
            channel_last=True,
            agent_vs_human=False,
            expert_action_type='v0'
        )
        env = GomokuEnv(cfg)
        test_episodes = 5
        for i in range(test_episodes):
            obs = env.reset()
            # print('init board state: ', obs)
            env.render()
            while True:
                action = env.expert_action()
                # action = env.random_action()
                # action = env.human_to_action()
                print('action index of player 1 is:', action)
                print('player 1: ' + env.action_to_string(action))
                obs, reward, done, info = env.step(action)
                env.render()
                if done:
                    if reward > 0:
                        print('player 1 win')
                    else:
                        print('draw')
                    break

                action = env.expert_action()
                # action = env.random_action()
                print('action index of player 2 is:', action)
                print('player 2: ' + env.action_to_string(action))
                obs, reward, done, info = env.step(action)
                env.render()
                if done:
                    if reward > 0:
                        print('player 2 win')
                    else:
                        print('draw')
                    break


test = TestExpertActionV0()
test.test_naive()

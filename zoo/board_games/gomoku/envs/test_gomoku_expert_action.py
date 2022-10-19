import pytest
from easydict import EasyDict

from zoo.board_games.gomoku.envs.gomoku_env import GomokuEnv


@pytest.mark.envtest
class TestExpertAction:

    def test_naive(self):
        cfg = EasyDict(board_size=18, battle_mode='two_player_mode', prob_random_agent=0)
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
                print('original player 1: ', action)
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
                print('original player 2: ', action)
                print('player 2: ' + env.action_to_string(action))
                obs, reward, done, info = env.step(action)
                env.render()
                if done:
                    if reward > 0:
                        print('player 2 win')
                    else:
                        print('draw')
                    break

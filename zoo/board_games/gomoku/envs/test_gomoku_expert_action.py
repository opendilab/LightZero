import pytest
from easydict import EasyDict

from zoo.board_games.gomoku.envs.gomoku_env import GomokuEnv


@pytest.mark.envtest
class TestExpertAction:

    def test_naive(self):
        cfg = EasyDict(board_size=18, battle_mode='two_player_mode', prob_random_agent=0)
        env = GomokuEnv(cfg)
        obs = env.reset()
        print('init board state: ', obs)
        env.render()
        while True:
            action = env.random_action()
            # action = env.human_to_action()
            # action = env.expert_action()
            print('original player 1: ', action)
            print('player 1: ' + env.action_to_string(action))
            obs, reward, done, info = env.step(action)
            env.render()
            if done:
                if reward > 0:
                    print('player 1 (human player) win')
                else:
                    print('draw')
                break

            action = env.expert_action()
            print('player 2 (computer player): ' + env.action_to_string(action))
            obs, reward, done, info = env.step(action)
            env.render()
            if done:
                if reward > 0:
                    print('player 2 (computer player) win')
                else:
                    print('draw')
                break

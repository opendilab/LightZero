import pytest
from ding.utils import EasyTimer
from easydict import EasyDict

from zoo.board_games.gomoku.envs.gomoku_env import GomokuEnv

timer = EasyTimer(cuda=True)

@pytest.mark.envtest
class TestGomokuEnv:

    def test_self_play_mode(self):
        cfg = EasyDict(
            board_size=15,
            battle_mode='self_play_mode',
            prob_random_agent=0,
            channel_last=False,
            scale=True,
            agent_vs_human=False,
            bot_action_type='v0',
            prob_random_action_in_bot=0.,
            check_action_to_connect4_in_bot_v0=False,
        )
        env = GomokuEnv(cfg)
        obs = env.reset()
        print('init board state: ')
        env.render()
        gomoku_env_legal_actions_cython = 0
        gomoku_env_legal_actions_cython_lru = 0
        while True:
            action = env.random_action()
            # action = env.human_to_action()
            print('player 1: ' + env.action_to_string(action))

            with timer:
                legal_actions = env.legal_actions_cython
            gomoku_env_legal_actions_cython += timer.value
            with timer:
                legal_actions = env.legal_actions_cython_lru
            gomoku_env_legal_actions_cython_lru += timer.value
            
            obs, reward, done, info = env.step(action)
            env.render()
            if done:
                if reward > 0:
                    print('player 1 (human player) win')
                else:
                    print('draw')
                break

            # action = env.bot_action()
            action = env.random_action()
            # action = env.human_to_action()
            print('player 2 (computer player): ' + env.action_to_string(action))
            obs, reward, done, info = env.step(action)
            env.render()
            if done:
                if reward > 0:
                    print('player 2 (computer player) win')
                else:
                    print('draw')
                break

        import time
        time.sleep(1)
        print(f"---------------------------------------")
        print(f"| gomoku_env_legal_actions_cython  | {gomoku_env_legal_actions_cython:.3f} |")
        print(f"---------------------------------------")
        print(f"---------------------------------------")
        print(f"| gomoku_env_legal_actions_cython_lru  | {gomoku_env_legal_actions_cython_lru:.3f} |")
        print(f"---------------------------------------")

    def test_play_with_bot_mode(self):
        cfg = EasyDict(
            board_size=15,
            battle_mode='play_with_bot_mode',
            prob_random_agent=0,
            channel_last=False,
            scale=True,
            agent_vs_human=False,
            bot_action_type='v0',
            prob_random_action_in_bot=0.,
            check_action_to_connect4_in_bot_v0=False,
        )
        env = GomokuEnv(cfg)
        env.reset()
        print('init board state: ')
        env.render()
        while True:
            """player 1"""
            # action = env.human_to_action()
            action = env.random_action()
            print('player 1: ' + env.action_to_string(action))
            obs, reward, done, info = env.step(action)
            # reward is in the perspective of player1
            env.render()
            if done:
                if reward != 0 and info['next player to play'] == 2:
                    print('player 1 (human player) win')
                elif reward != 0 and info['next player to play'] == 1:
                    print('player 2 (computer player) win')
                else:
                    print('draw')
                break


# test = TestGomokuEnv()
# test.test_self_play_mode()

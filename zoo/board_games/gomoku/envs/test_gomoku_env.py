import pytest
from easydict import EasyDict

from zoo.board_games.gomoku.envs.gomoku_env import GomokuEnv


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
            # (str) The render mode. Options are 'None', 'state_realtime_mode', 'image_realtime_mode' or 'image_savefile_mode'.
            # If None, then the game will not be rendered.
            render_mode=None,
            screen_scaling=9,
        )
        env = GomokuEnv(cfg)
        obs = env.reset()
        print('init board state: ')
        while True:
            action = env.random_action()
            # action = env.human_to_action()
            print('player 1: ' + env.action_to_string(action))
            obs, reward, done, info = env.step(action)
            env.render(mode=cfg.render_mode)
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
            env.render(mode=cfg.render_mode)
            if done:
                if reward > 0:
                    print('player 2 (computer player) win')
                else:
                    print('draw')
                break

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
            # (str) The render mode. Options are 'None', 'state_realtime_mode', 'image_realtime_mode' or 'image_savefile_mode'.
            # If None, then the game will not be rendered.
            render_mode='state_realtime_mode',  # 'image_realtime_mode' # "state_realtime_mode",
            screen_scaling=9,
        )
        env = GomokuEnv(cfg)
        env.reset()
        print('init board state: ')
        env.render(mode=cfg.render_mode)
        while True:
            """player 1"""
            # action = env.human_to_action()
            action = env.random_action()

            print('player 1: ' + env.action_to_string(action))
            obs, reward, done, info = env.step(action)
            # reward is in the perspective of player1
            env.render(mode=cfg.render_mode)
            if done:
                if reward != 0 and info['next player to play'] == 2:
                    print('player 1 (human player) win')
                elif reward != 0 and info['next player to play'] == 1:
                    print('player 2 (computer player) win')
                else:
                    print('draw')
                break


# test = TestGomokuEnv()
# test.test_play_with_bot_mode()

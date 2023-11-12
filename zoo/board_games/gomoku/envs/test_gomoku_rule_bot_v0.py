import pytest
from easydict import EasyDict

from zoo.board_games.gomoku.envs.gomoku_env import GomokuEnv

cfg = EasyDict(
    prob_random_agent=0,
    board_size=6,
    # board_size=9,
    battle_mode='self_play_mode',
    channel_last=False,
    scale=True,
    agent_vs_human=False,
    bot_action_type='v0',  # {'v0', 'v1', 'alpha_beta_pruning'}
    prob_random_action_in_bot=0.,
    check_action_to_connect4_in_bot_v0=False,
    screen_scaling=9,
    render_mode=None,
)


@pytest.mark.envtest
class TestExpertActionV0:

    def test_naive(self):
        env = GomokuEnv(cfg)
        test_episodes = 10
        for i in range(test_episodes):
            obs = env.reset()
            # print('init board state: ', obs)
            # env.render('image_realtime_mode')
            while True:
                action = env.bot_action()
                # action = env.random_action()
                # action = env.human_to_action()
                print('action index of player 1 is:', action)
                print('player 1: ' + env.action_to_string(action))
                obs, reward, done, info = env.step(action)
                # env.render('image_realtime_mode')
                if done:
                    print('=' * 20)
                    if reward > 0:
                        print('player 1 win')
                    else:
                        print('draw')
                    print('=' * 20)
                    break

                # action = env.bot_action()
                # action = env.random_action()
                action = env.human_to_action()
                print('action index of player 2 is:', action)
                print('player 2: ' + env.action_to_string(action))
                obs, reward, done, info = env.step(action)
                # env.render('image_realtime_mode')
                if done:
                    print('=' * 20)
                    if reward > 0:
                        print('player 2 win')
                    else:
                        print('draw')
                    print('=' * 20)
                    break

    def test_v0_vs_v1(self):
        """
        board_size=6, test 10 episodes:
        =================================================
        v0 vs v1: 0 bot_v0 win, 2 bot_v1 win, 8 draw
        v1 vs v0: 0 bot_v0 win, 4 bot_v1 win, 6 draw
        v0 vs v0: 0 player1 win, 4 player2 win, 6 draw
        v1 vs v1: 0 player1 win, 0 player2 win, 10 draw
        v0 vs random: 10 bot_v1 win, 0 random win, 0 draw
        v1 vs random: 10 bot_v1 win, 0 random win, 0 draw
        =================================================

        board_size=9, test 3 episodes:
        =================================================
        v0 vs v1: 0 bot_v0 win, 3 bot_v1 win, 0 draw
        v1 vs v0: 3 bot_v0 win, 0 bot_v1 win, 0 draw
        v0 vs v0: 3 player1 win, 0 player2 win, 0 draw
        v1 vs v1: 0 player1 win, 0 player2 win, 3 draw
        v0 vs random: 3 bot_v1 win, 0 random win, 0 draw
        v1 vs random: 3 bot_v1 win, 0 random win, 0 draw
        =================================================
        """
        env = GomokuEnv(cfg)
        test_episodes = 3
        for i in range(test_episodes):
            obs = env.reset()
            # print('init board state: ', obs)
            env.render()
            while True:
                env.bot_action_type = 'v0'
                action = env.bot_action()
                # action = env.random_action()
                # action = env.human_to_action()
                print('action index of player 1 is:', action)
                print('player 1: ' + env.action_to_string(action))
                obs, reward, done, info = env.step(action)
                env.render()
                if done:
                    print('=' * 20)
                    if reward > 0:
                        print('player 1 win')
                    else:
                        print('draw')
                    print('=' * 20)
                    break

                env.bot_action_type = 'v0'
                action = env.bot_action()
                # action = env.random_action()
                # action = env.human_to_action()
                print('action index of player 2 is:', action)
                print('player 2: ' + env.action_to_string(action))
                obs, reward, done, info = env.step(action)
                env.render()
                if done:
                    print('=' * 20)
                    if reward > 0:
                        print('player 2 win')
                    else:
                        print('draw')
                    print('=' * 20)
                    break


test = TestExpertActionV0()
test.test_v0_vs_v1()
# test.test_naive()

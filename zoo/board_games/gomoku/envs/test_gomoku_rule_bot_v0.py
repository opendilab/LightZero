import pytest
from easydict import EasyDict

from zoo.board_games.gomoku.envs.gomoku_env import GomokuEnv

cfg = EasyDict(
    prob_random_agent=0,
    board_size=6,
    battle_mode='self_play_mode',
    channel_last=False,
    scale=True,
    agent_vs_human=False,
    bot_action_type='v0',  # {'v0', 'v1', 'alpha_beta_pruning'}
    prob_random_action_in_bot=0.5,
    check_action_to_connect4_in_bot_v0=False,
)


@pytest.mark.envtest
class TestExpertActionV0:

    def test_naive(self):
        env = GomokuEnv(cfg)
        test_episodes = 100
        for i in range(test_episodes):
            obs = env.reset()
            # print('init board state: ', obs)
            env.render()
            while True:
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

    def test_v0_vs_v1(self):
        """
        board_size=6, test_100_episodes:
        =================================================
        v0 vs v1: 0 bot_v0 win, 16 bot_v1 win, 84 draw
        v1 vs v0: 1 bot_v0 win, 35 bot_v1 win, 64 draw
        v0 vs v0: 100 draw
        v1 vs v1: 100 draw
        v0 vs random: 93 bot_v0 win, 35 random win, 7 draw
        v1 vs random: 100 bot_v1 win, 0 random win, 0 draw
        =================================================

        board_size=5, test_100_episodes:
        =================================================
        v0 vs v1: 0 bot_v0 win, 0 bot_v1 win, 100 draw
        v1 vs v0: 1 bot_v0 win, 35 bot_v1 win, 64 draw
        v0 vs v0: 100 draw
        v1 vs v1: 100 draw
        v0 vs random: 68 bot_v0 win, 0 random win, 32 draw
        v1 vs random: 98 bot_v1 win, 0 random win, 2 draw
        =================================================
        """
        env = GomokuEnv(cfg)
        test_episodes = 10
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

                env.bot_action_type = 'v1'
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
# test.test_v0_vs_v1()
test.test_naive()

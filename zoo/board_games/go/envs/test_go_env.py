import pytest
from easydict import EasyDict

from zoo.board_games.go.envs.go_env import GoEnv
import time

cfg = EasyDict(
    board_size=6,
    komi=7.5,
    battle_mode='self_play_mode',
    prob_random_agent=0,
    channel_last=False,
    scale=True,
    agent_vs_human=False,
    bot_action_type='v0',
    prob_random_action_in_bot=0.,
    check_action_to_connect4_in_bot_v0=False,
    stop_value=1,
)


@pytest.mark.envtest
class TestGoEnv:

    def test_naive(self):

        env = GoEnv(cfg)
        test_episodes = 1
        for _ in range(test_episodes):
            print('NOTE：actions are counted by column, such as action 9, which is the second column and the first row')
            obs = env.reset()
            print(obs['observation'].shape, obs['action_mask'].shape)
            print(obs['observation'], obs['action_mask'])

            actions_black = [0, 2, 0]
            actions_white = [1, 6]

            # env.render()

            for i in range(1000):
                print('turn: ', i)
                """player 1"""
                # action = env.human_to_action()
                action = env.random_action()
                # action = actions_black[i]
                print('player 1 (black): ', action)
                obs, reward, done, info = env.step(action)
                time.sleep(0.1)
                # print(obs, reward, done, info)
                assert isinstance(obs, dict)
                assert isinstance(done, bool)
                assert isinstance(reward, float) or isinstance(reward, int)
                # env.render('board')
                env.render('human')

                if done:
                    if reward > 0:
                        print('player 1 (black) win')
                    elif reward < 0:
                        print('player 2 (white) win')
                    else:
                        print('draw')
                    break

                """player 2"""
                action = env.random_action()
                # action = actions_white[i]
                print('player 2 (white): ', action)
                obs, reward, done, info = env.step(action)
                time.sleep(0.1)

                # print(obs, reward, done, info)
                # env.render('board')
                env.render('human')
                if done:
                    if reward > 0:
                        print('player 2 (white) win')
                    elif reward < 0:
                        print('player 1 (black) win')
                    else:
                        print('draw')
                    break

import pytest
from easydict import EasyDict

from zoo.board_games.gomoku.envs.gomoku_env_ui import GomokuEnv


@pytest.mark.envtest
class TestExpertActionV1:

    def test_naive(self):
        cfg = EasyDict(
            prob_random_agent=0,
            board_size=6,
            battle_mode='self_play_mode',
            channel_last=False,
            scale=False,
            agent_vs_human=False,
            bot_action_type='v1',  # {'v0', 'v1', 'alpha_beta_pruning'}
            prob_random_action_in_bot=0.,
            check_action_to_connect4_in_bot_v0=False,
        )
        env = GomokuEnv(cfg)
        test_episodes = 1
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
                    if reward > 0:
                        print('player 1 win')
                    else:
                        print('draw')
                    break

                action = env.bot_action()
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


test = TestExpertActionV1()
test.test_naive()

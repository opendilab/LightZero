import pytest
from easydict import EasyDict
from connect4_env import Connect4Env

# 这是啥！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
@pytest.mark.envtest
class TestConnect4Env:

    def test_self_play_mode(self):
        # 实现cfg接口！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
        cfg = EasyDict(
            battle_mode='self_play_mode',
            channel_last=True,
            scale=True,
            agent_vs_human=False,
            prob_random_agent=0,
            prob_expert_agent=0,
            bot_action_type='v0'
        )
        env = Connect4Env(cfg)
        env.reset()
        print('init board state: ')
        env.render()
        while True:
            """player 1"""
            # action = env.human_to_action()
            action = env.random_action()
            # action = env.bot_action()

            # test legal_actions
            # legal_actions = env.legal_actions
            # print('legal_actions: ', legal_actions)
            # action = legal_actions[-1]

            # 实现acttostring!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            print('player 1: ' + env.action_to_string(action))
            obs, reward, done, info = env.step(action)
            print(reward)
            env.render()
            if done:
                if reward > 0:
                    print('player 1 win')
                else:
                    print('draw')
                break

            """player 2"""
            action = env.random_action()
            print('player 2 : ' + env.action_to_string(action))
            obs, reward, done, info = env.step(action)
            print(reward)
            env.render()
            if done:
                if reward > 0:
                    print('player 2 win')
                else:
                    print('draw')
                break

    def test_play_with_bot_mode(self):
        cfg = EasyDict(
            battle_mode='play_with_bot_mode',
            channel_last=True,
            scale=True,
            # channel_last=False,
            # scale=False,
            agent_vs_human=False,
            prob_random_agent=0,
            prob_expert_agent=0,
            bot_action_type='v0'
        )
        env = Connect4Env(cfg)
        env.reset()
        print('init board state: ')
        env.render()
        while True:
            """player 1"""
            # action = env.human_to_action()
            action = env.random_action()

            # test legal_actions
            # legal_actions = env.legal_actions
            # print('legal_actions: ', legal_actions)
            # action = legal_actions[-1]

            print('player 1: ' + env.action_to_string(action))
            obs, reward, done, info = env.step(action)
            # reward is in the perspective of player1
            env.render()
            if done:
                if reward != 0 and env.current_player == 2:
                    print('player 1 (human player) win')
                elif reward != 0 and env.current_player == 1:
                    print('player 2 (computer player) win')
                else:
                    print('draw')
                break


test = TestConnect4Env()
test.test_self_play_mode()
test.test_play_with_bot_mode()

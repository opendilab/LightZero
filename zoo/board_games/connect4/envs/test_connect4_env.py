import pytest
from easydict import EasyDict

from connect4_env import Connect4Env


@pytest.mark.envtest
class TestConnect4Env:

    def test_self_play_mode(self):
        cfg = EasyDict(
            battle_mode='self_play_mode',
            mcts_mode='self_play_mode',
            channel_last=False,
            scale=True,
            screen_scaling=9,
            prob_random_action_in_bot=0.,
            save_replay=False,
            agent_vs_human=False,
            prob_random_agent=0,
            prob_expert_agent=0,
            bot_action_type='rule',  # {'rule', 'mcts'}
        )
        env = Connect4Env(cfg)
        env.reset()
        print('init board state: ')
        env.render()
        while True:
            """player 1"""
            # action = env.human_to_action()
            # action = env.bot_action()
            action = env.random_action()

            # test legal_actions
            # legal_actions = env.legal_actions
            # print('legal_actions: ', legal_actions)
            # action = legal_actions[-1]

            print('player 1: ' + env.action_to_string(action))
            obs, reward, done, info = env.step(action)
            # print(reward)
            env.render()
            if done:
                if reward > 0:
                    print('player 1 win')
                else:
                    print('draw')
                break

            """player 2"""
            action = env.bot_action()
            print('player 2 : ' + env.action_to_string(action))
            obs, reward, done, info = env.step(action)
            # print(reward)
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
            mcts_mode='play_with_bot_mode',
            channel_last=False,
            scale=True,
            screen_scaling=9,
            prob_random_action_in_bot=0.,
            save_replay=False,
            agent_vs_human=False,
            prob_random_agent=0,
            prob_expert_agent=0,
            bot_action_type='rule',  # {'rule', 'mcts'}
        )
        env = Connect4Env(cfg)
        env.reset()
        print('init board state: ')
        env.render()
        while True:
            """player 1"""
            action = env.human_to_action()
            # action = env.bot_action()

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

    def test_eval_mode(self):
        cfg = EasyDict(
            battle_mode='eval_mode',
            mcts_mode='play_with_bot_mode',
            channel_last=False,
            scale=True,
            screen_scaling=9,
            prob_random_action_in_bot=0.,
            save_replay=False,
            agent_vs_human=True,
            prob_random_agent=0,
            prob_expert_agent=0,
            bot_action_type='mcts',

        )
        env = Connect4Env(cfg)
        env.reset()
        print('init board state: ')
        env.render(mode='state_realtime_mode')
        while True:
            """player 1"""
            # action = env.human_to_action()
            action = env.bot_action()

            # test legal_actions
            # legal_actions = env.legal_actions
            # print('legal_actions: ', legal_actions)
            # action = legal_actions[-1]

            print('player 1: ' + env.action_to_string(action))
            obs, reward, done, info = env.step(action)
            # reward is in the perspective of player1
            env.render(mode='state_realtime_mode')
            if done:
                if reward != 0 and env.current_player == 2:
                    print('player 1 (human player) win')
                elif reward != 0 and env.current_player == 1:
                    print('player 2 (computer player) win')
                else:
                    print('draw')
                break

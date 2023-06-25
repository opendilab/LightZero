from easydict import EasyDict
from zoo.board_games.go.envs.go_env import GoEnv
from zoo.board_games.mcts_bot import MCTSBot

import pytest

cfg = EasyDict(dict(
    # board_size=6,
    # num_simulations=50,
    num_simulations=50,
    board_size=5,
    komi=7.5,
    prob_random_agent=0,
    prob_expert_agent=0,
    battle_mode='self_play_mode',
    scale=True,
    channel_last=True,
    agent_vs_human=False,
    bot_action_type='alpha_beta_pruning',  # {'v0', 'alpha_beta_pruning'}
    prob_random_action_in_bot=0.,
    check_action_to_connect4_in_bot_v0=False,
))


@pytest.mark.envtest
class TestGoBot:

    def test_go_mcts_vs_random(self):
        # player_0  num_simulation=1000, will win
        # player_1  num_simulation=1
        env = GoEnv(cfg)
        env.reset()
        state = env._raw_env._go.board
        player_0 = MCTSBot(GoEnv, cfg, 'player 1', cfg.num_simulations)  # player_index = 0, player = 1

        player_index = 0  # player 1 first
        print('#' * 15)
        print(state)
        print('#' * 15)
        print('\n')
        while not env.get_done_reward()[0]:
            if player_index == 0:
                action = player_0.get_actions(state, player_index=player_index)
                player_index = 1
            else:
                print('-' * 40)
                # action = player_1.get_actions(state, player_index=player_index)
                action = env.random_action()
                player_index = 0
                # print('-' * 40)
            env.step(action)
            state = env._raw_env._go.board
            print('-' * 40)
            print(state)
            # print('#' * 15)
        assert env.get_done_winner()[1] == 1, f'winner is {env.get_done_winner()[1]}, player 1 should win'

    def test_go_self_play_mode_player0_win(self):
        # player_0  num_simulation=1000, will win
        # player_1  num_simulation=1
        env = GoEnv(cfg)
        env.reset()
        state = env._raw_env._go.board
        player_0 = MCTSBot(GoEnv, cfg, 'player 1', cfg.num_simulations)  # player_index = 0, player = 1
        player_1 = MCTSBot(GoEnv, cfg, 'player 2', int(cfg.num_simulations/2))  # player_index = 1, player = 2

        player_index = 0  # player 1 first
        print('#' * 15)
        print(state)
        print('#' * 15)
        print('\n')
        while not env.get_done_reward()[0]:
            if player_index == 0:
                action = player_0.get_actions(state, player_index=player_index)
                player_index = 1
            else:
                print('-' * 40)
                action = player_1.get_actions(state, player_index=player_index)
                player_index = 0
                # print('-' * 40)
            env.step(action)
            state = env._raw_env._go.board
            print('-' * 40)
            print(state)
            # print('#' * 15)
        assert env.get_done_winner()[1] == 1, f'winner is {env.get_done_winner()[1]}, player 1 should win'

    def test_go_self_play_mode_player1_win(self):
        # player_0  num_simulation=1
        # player_1  num_simulation=1000, will win
        env = GoEnv(cfg)
        env.reset()
        state = env._raw_env._go.board
        player_0 = MCTSBot(GoEnv, cfg, 'player 1', 1)  # player_index = 0, player = 1
        player_1 = MCTSBot(GoEnv, cfg, 'player 2', cfg.num_simulations)  # player_index = 1, player = 2

        player_index = 0  # player 1 first
        print('#' * 15)
        print(state)
        print('#' * 15)
        print('\n')
        while not env.get_done_reward()[0]:
            if player_index == 0:
                action = player_0.get_actions(state, player_index=player_index)
                player_index = 1
            else:
                print('-' * 40)
                action = player_1.get_actions(state, player_index=player_index)
                player_index = 0
            env.step(action)
            state = env._raw_env._go.board
            print('-' * 40)
            print(state)
        assert env.get_done_winner()[1] == 2, f'winner is {env.get_done_winner()[1]}, player 2 should win'


    def test_go_self_play_mode_draw(self):
        # player_0  num_simulation=1000
        # player_1  num_simulation=1000, will draw
        env = GoEnv(cfg)
        env.reset()
        state = env._raw_env._go.board
        player_0 = MCTSBot(GoEnv, cfg, 'player 1', cfg.num_simulations)  # player_index = 0, player = 1
        player_1 = MCTSBot(GoEnv, cfg, 'player 2', cfg.num_simulations)  # player_index = 1, player = 2

        player_index = 0  # player 1 fist
        print('#' * 15)
        print(state)
        print('#' * 15)
        print('\n')
        while not env.get_done_reward()[0]:
            if player_index == 0:
                action = player_0.get_actions(state, player_index=player_index)
                player_index = 1
            else:
                print('-' * 40)
                action = player_1.get_actions(state, player_index=player_index)
                player_index = 0
            env.step(action)
            state = env._raw_env._go.board
            print('-' * 40)
            print(state)
        assert env.get_done_winner()[1] == -1, f'winner is {env.get_done_winner()[1]}, two players should draw'

    # def test_go_self_play_mode_case_1(self):
    #     env = GoEnv(cfg)
    #     init_state = [
    #         [1, 1, 1, 1, 0],
    #         [1, 0, 0, 0, 2],
    #         [0, 0, 2, 0, 2],
    #         [0, 2, 0, 0, 2],
    #         [2, 1, 1, 0, 0],
    #     ]
    #     player_0 = MCTSBot(GoEnv, cfg, 'player 1', cfg.num_simulations)  # player_index = 0, player = 1
    #     player_1 = MCTSBot(GoEnv, cfg, 'player 2', cfg.num_simulations)  # player_index = 1, player = 2
    #     player_index = 1  # player 1 fist
    #
    #     env.reset(player_index, init_state)
    #     state = env._raw_env._go.board
    #
    #     print('#' * 15)
    #     print(state)
    #     print('#' * 15)
    #     print('\n')
    #     while not env.get_done_reward()[0]:
    #         if player_index == 0:
    #             action = player_0.get_actions(state, player_index=player_index)
    #             player_index = 1
    #         else:
    #             print('-' * 40)
    #             action = player_1.get_actions(state, player_index=player_index)
    #             player_index = 0
    #             print('-' * 40)
    #         env.step(action)
    #         state = env._raw_env._go.board
    #         print('#' * 15)
    #         print(state)
    #         print('#' * 15)
    #         row, col = env.action_to_coord(action)
    #     assert env.get_done_winner()[1] == 2
    #     assert state[0, 4] == 2
    #
    # def test_go_self_play_mode_case_2(self):
    #     env = GoEnv(cfg)
    #     init_state = [
    #         [0, 0, 2, 0, 0],
    #         [0, 1, 2, 0, 0],
    #         [2, 2, 1, 0, 0],
    #         [2, 0, 0, 1, 2],
    #         [1, 1, 1, 0, 0],
    #     ]
    #     player_0 = MCTSBot(GoEnv, cfg, 'player 1', cfg.num_simulations)  # player_index = 0, player = 1
    #     player_1 = MCTSBot(GoEnv, cfg, 'player 2', cfg.num_simulations)  # player_index = 1, player = 2
    #     player_index = 0  # player 1 fist
    #
    #     env.reset(player_index, init_state)
    #     state = env._raw_env._go.board
    #
    #     print('#' * 15)
    #     print(state)
    #     print('#' * 15)
    #     print('\n')
    #     while not env.get_done_reward()[0]:
    #         if player_index == 0:
    #             action = player_0.get_actions(state, player_index=player_index)
    #             player_index = 1
    #         else:
    #             print('-' * 40)
    #             action = player_1.get_actions(state, player_index=player_index)
    #             player_index = 0
    #             print('-' * 40)
    #         env.step(action)
    #         state = env._raw_env._go.board
    #         print('#' * 15)
    #         print(state)
    #         print('#' * 15)
    #         row, col = env.action_to_coord(action)
    #     assert env.get_done_winner()[1] == 1
    #     assert state[4, 4] == 1


test = TestGoBot().test_go_self_play_mode_player0_win()
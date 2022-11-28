import sys

from easydict import EasyDict
sys.path.append('/YOUR/PATH/LightZero')
from zoo.board_games.tictactoe.envs.tictactoe_env import TicTacToeEnv
from zoo.board_games.mcts_bot import MCTSBot

import pytest

@pytest.mark.envtest
class TestTicTacToeBot:
    def test_tictactoe_two_player_mode_player0_win(self):
        # player_0  num_simulation=1000, will win
        # player_1  num_simulation=1
        cfg = dict(
            prob_random_agent=0,
            prob_expert_agent=0,
            battle_mode='two_player_mode',
        )
        env = TicTacToeEnv(EasyDict(cfg))
        env.reset()
        state = env.board
        player_0 = MCTSBot(TicTacToeEnv, cfg, 'a', 1000)  # player_index = 0, player = 1
        player_1 = MCTSBot(TicTacToeEnv, cfg, 'b', 1)  # player_index = 1, player = 2

        player_index = 0  # A fist
        print('#' * 15)
        print(state)
        print('#' * 15)
        print('\n')
        while not env.is_game_over()[0]:
            if player_index == 0:
                action = player_0.get_actions(state, player_index=player_index)
                player_index = 1
            else:
                print('-' * 40)
                action = player_1.get_actions(state, player_index=player_index)
                player_index = 0
                print('-' * 40)
            env.step(action)
            state = env.board
            print('#' * 15)
            print(state)
            print('#' * 15)
        assert env.have_winner()[1] == 1

    def test_tictactoe_two_player_mode_player1_win(self):
        # player_0  num_simulation=1
        # player_1  num_simulation=1000, will win
        cfg = dict(
            prob_random_agent=0,
            prob_expert_agent=0,
            battle_mode='two_player_mode',
        )
        env = TicTacToeEnv(EasyDict(cfg))
        env.reset()
        state = env.board
        player_0 = MCTSBot(TicTacToeEnv, cfg, 'a', 1)  # player_index = 0, player = 1
        player_1 = MCTSBot(TicTacToeEnv, cfg, 'b', 1000)  # player_index = 1, player = 2

        player_index = 0  # A fist
        print('#' * 15)
        print(state)
        print('#' * 15)
        print('\n')
        while not env.is_game_over()[0]:
            if player_index == 0:
                action = player_0.get_actions(state, player_index=player_index)
                player_index = 1
            else:
                print('-' * 40)
                action = player_1.get_actions(state, player_index=player_index)
                player_index = 0
                print('-' * 40)
            env.step(action)
            state = env.board
            print('#' * 15)
            print(state)
            print('#' * 15)
        assert env.have_winner()[1] == 2

    def test_tictactoe_two_player_mode_draw(self):
        # player_0  num_simulation=1000
        # player_1  num_simulation=1000, will draw
        cfg = dict(
            prob_random_agent=0,
            prob_expert_agent=0,
            battle_mode='two_player_mode',
        )
        env = TicTacToeEnv(EasyDict(cfg))
        env.reset()
        state = env.board
        player_0 = MCTSBot(TicTacToeEnv, cfg, 'a', 1000)  # player_index = 0, player = 1
        player_1 = MCTSBot(TicTacToeEnv, cfg, 'b', 1000)  # player_index = 1, player = 2

        player_index = 0  # A fist
        print('#' * 15)
        print(state)
        print('#' * 15)
        print('\n')
        while not env.is_game_over()[0]:
            if player_index == 0:
                action = player_0.get_actions(state, player_index=player_index)
                player_index = 1
            else:
                print('-' * 40)
                action = player_1.get_actions(state, player_index=player_index)
                player_index = 0
                print('-' * 40)
            env.step(action)
            state = env.board
            print('#' * 15)
            print(state)
            print('#' * 15)
        assert env.have_winner()[1] == -1

    def test_tictactoe_two_player_mode_half_case_1(self):
        cfg = dict(
            prob_random_agent=0,
            prob_expert_agent=0,
            battle_mode='two_player_mode',
        )
        env = TicTacToeEnv(EasyDict(cfg))
        init_state = [[1, 1, 0],
                      [0, 2, 2],
                      [0, 0, 0]]
        player_0 = MCTSBot(TicTacToeEnv, cfg, 'a', 1000)  # player_index = 0, player = 1
        player_1 = MCTSBot(TicTacToeEnv, cfg, 'b', 1000)  # player_index = 1, player = 2
        player_index = 0  # A fist

        env.reset(player_index, init_state)
        state = env.board

        print('#' * 15)
        print(state)
        print('#' * 15)
        print('\n')
        while not env.is_game_over()[0]:
            if player_index == 0:
                action = player_0.get_actions(state, player_index=player_index)
                player_index = 1
            else:
                print('-' * 40)
                action = player_1.get_actions(state, player_index=player_index)
                player_index = 0
                print('-' * 40)
            env.step(action)
            state = env.board
            print('#' * 15)
            print(state)
            print('#' * 15)
            row, col = env.action_to_coord(action)
        assert env.have_winner()[1] == 1
        assert row == 0, col == 2

    def test_tictactoe_two_player_mode_half_case_2(self):
        cfg = dict(
            prob_random_agent=0,
            prob_expert_agent=0,
            battle_mode='two_player_mode',
        )
        env = TicTacToeEnv(EasyDict(cfg))
        init_state = [[1, 0, 1],
                      [0, 0, 2],
                      [2, 0, 1]]
        player_0 = MCTSBot(TicTacToeEnv, cfg, 'a', 1000)  # player_index = 0, player = 1
        player_1 = MCTSBot(TicTacToeEnv, cfg, 'b', 1000)  # player_index = 1, player = 2
        player_index = 1  # A fist

        env.reset(player_index, init_state)
        state = env.board

        print('#' * 15)
        print(state)
        print('#' * 15)
        print('\n')
        while not env.is_game_over()[0]:
            if player_index == 0:
                action = player_0.get_actions(state, player_index=player_index)
                player_index = 1
            else:
                print('-' * 40)
                action = player_1.get_actions(state, player_index=player_index)
                player_index = 0
                print('-' * 40)
            env.step(action)
            state = env.board
            print('#' * 15)
            print(state)
            print('#' * 15)
        assert env.have_winner()[1] == 1

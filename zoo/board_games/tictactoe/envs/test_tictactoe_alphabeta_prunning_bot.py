from easydict import EasyDict
from zoo.board_games.tictactoe.envs.tictactoe_env import TicTacToeEnv
from zoo.board_games.alphabeta_pruning_bot import AlphaBetaPruningBot

import pytest

cfg = dict(
    prob_random_agent=0,
    prob_expert_agent=0,
    battle_mode='self_play_mode',
    agent_vs_human=False,
    bot_action_type='alpha_beta_pruning',  # {'v0', 'alpha_beta_pruning'}
)

@pytest.mark.envtest
class TestTicTacToeAlphaBetaPruningBot:

    def test_tictactoe_self_play_mode_draw(self):
        # player_0: AlphaBetaPruningBot
        # player_1: AlphaBetaPruningBot
        env = TicTacToeEnv(EasyDict(cfg))
        env.reset()
        state = env.board
        player_0 = AlphaBetaPruningBot(TicTacToeEnv, cfg, 'player 1')  # player_index = 0, player = 1
        player_1 = AlphaBetaPruningBot(TicTacToeEnv, cfg, 'player 2')  # player_index = 1, player = 2

        player_index = 0  # player 1 fist
        print('-' * 15)
        print(state)

        while not env.get_done_reward()[0]:
            if player_index == 0:
                action = player_0.get_best_action(state, player_index=player_index)
                player_index = 1
            else:
                print('-' * 40)
                action = player_1.get_best_action(state, player_index=player_index)
                player_index = 0
            env.step(action)
            state = env.board
            print('-' * 15)
            print(state)

        assert env.get_done_winner()[0] == False, env.get_done_winner()[1] == -1

    def test_tictactoe_self_play_mode_half_case_1(self):
        env = TicTacToeEnv(EasyDict(cfg))
        player_0 = AlphaBetaPruningBot(TicTacToeEnv, cfg, 'player 1')  # player_index = 0, player = 1
        player_1 = AlphaBetaPruningBot(TicTacToeEnv, cfg, 'player 2')  # player_index = 1, player = 2

        player_index = 0  # player 1 fist
        init_state = [[1, 1, 0], [0, 2, 2], [0, 0, 0]]
        env.reset(player_index, init_state)

        state = env.board
        print('-' * 15)
        print(state)

        while not env.get_done_reward()[0]:
            if player_index == 0:
                action = player_0.get_best_action(state, player_index=player_index)
                player_index = 1
            else:
                print('-' * 40)
                action = player_1.get_best_action(state, player_index=player_index)
                player_index = 0
            env.step(action)
            state = env.board
            print('-' * 15)
            print(state)
            row, col = env.action_to_coord(action)

        assert env.get_done_winner()[1] == 1
        assert row == 0, col == 2
        assert env.get_done_winner()[0] == True, env.get_done_winner()[1] == 1

    def test_tictactoe_self_play_mode_half_case_2(self):
        env = TicTacToeEnv(EasyDict(cfg))
        player_0 = AlphaBetaPruningBot(TicTacToeEnv, cfg, 'player 1')  # player_index = 0, player = 1
        player_1 = AlphaBetaPruningBot(TicTacToeEnv, cfg, 'player 2')  # player_index = 1, player = 2

        player_index = 0  # player 1 fist
        init_state = [[1, 0, 1], [0, 0, 2], [2, 0, 1]]
        env.reset(player_index, init_state)

        state = env.board
        print('-' * 15)
        print(state)

        while not env.get_done_reward()[0]:
            if player_index == 0:
                action = player_0.get_best_action(state, player_index=player_index)
                player_index = 1
            else:
                print('-' * 40)
                action = player_1.get_best_action(state, player_index=player_index)
                player_index = 0
            env.step(action)
            state = env.board
            print('-' * 15)
            print(state)
            row, col = env.action_to_coord(action)

        assert env.get_done_winner()[1] == 1
        assert (row == 0, col == 1) or (row == 1, col == 1)
        assert env.get_done_winner()[0] is True, env.get_done_winner()[1] == 1

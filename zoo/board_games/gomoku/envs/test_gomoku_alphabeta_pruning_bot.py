from easydict import EasyDict
import pytest
import time

from zoo.board_games.alphabeta_pruning_bot import AlphaBetaPruningBot
from zoo.board_games.gomoku.envs.gomoku_env import GomokuEnv

cfg = dict(
    board_size=5,
    prob_random_agent=0,
    prob_expert_agent=0,
    battle_mode='self_play_mode',
    scale=True,
    channel_last=True,
    agent_vs_human=False,
    bot_action_type='alpha_beta_pruning',  # options: {'v0', 'alpha_beta_pruning'}
    prob_random_action_in_bot=0.,
    check_action_to_connect4_in_bot_v0=False,
)


@pytest.mark.envtest
class TestGomokuBot:

    def test_gomoku_self_play_mode_draw(self):
        # player_0: AlphaBetaPruningBot
        # player_1: AlphaBetaPruningBot
        # will draw
        env = GomokuEnv(EasyDict(cfg))
        env.reset()
        state = env.board
        player_0 = AlphaBetaPruningBot(GomokuEnv, cfg, 'player 1')  # player_index = 0, player = 1
        player_1 = AlphaBetaPruningBot(GomokuEnv, cfg, 'player 2')  # player_index = 1, player = 2

        player_index = 0  # player 1 fist
        print('-' * 15)
        print(state)

        while not env.get_done_reward()[0]:
            if player_index == 0:
                start = time.time()
                action = player_0.get_best_action(state, player_index=player_index)
                print('player 1 action time: ', time.time() - start)
                player_index = 1
            else:
                start = time.time()
                action = player_1.get_best_action(state, player_index=player_index)
                print('player 2 action time: ', time.time() - start)
                player_index = 0
            env.step(action)
            state = env.board
            print('-' * 15)
            print(state)

        assert env.get_done_winner()[0] is False, env.get_done_winner()[1] == -1

    def test_gomoku_self_play_mode_case_1(self):
        env = GomokuEnv(EasyDict(cfg))
        player_0 = AlphaBetaPruningBot(GomokuEnv, cfg, 'player 1')  # player_index = 0, player = 1
        player_1 = AlphaBetaPruningBot(GomokuEnv, cfg, 'player 2')  # player_index = 1, player = 2

        player_index = 1  # player 2 fist
        init_state = [
            [1, 1, 1, 1, 0],
            [1, 0, 0, 0, 2],
            [0, 0, 2, 0, 2],
            [0, 2, 0, 0, 2],
            [2, 1, 1, 0, 0],
        ]
        env.reset(player_index, init_state)

        state = env.board
        print('-' * 15)
        print(state)

        while not env.get_done_reward()[0]:
            if player_index == 0:
                start = time.time()
                action = player_0.get_best_action(state, player_index=player_index)
                print('player 1 action time: ', time.time() - start)
                player_index = 1
            else:
                start = time.time()
                action = player_1.get_best_action(state, player_index=player_index)
                print('player 2 action time: ', time.time() - start)
                player_index = 0
            env.step(action)
            state = env.board
            print('-' * 15)
            print(state)
        row, col = env.action_to_coord(action)

        # the player 2 win when place piece in (0, 4)
        assert env.get_done_winner()[1] == 2
        assert row == 0, col == 4

    def test_gomoku_self_play_mode_case_2(self):
        env = GomokuEnv(EasyDict(cfg))
        player_0 = AlphaBetaPruningBot(GomokuEnv, cfg, 'player 1')  # player_index = 0, player = 1
        player_1 = AlphaBetaPruningBot(GomokuEnv, cfg, 'player 2')  # player_index = 1, player = 2

        player_index = 0  # player 1 fist
        init_state = [
            [0, 0, 2, 0, 0],
            [0, 1, 2, 0, 0],
            [2, 2, 1, 0, 0],
            [2, 0, 0, 1, 2],
            [1, 1, 1, 0, 0],
        ]
        env.reset(player_index, init_state)

        state = env.board
        print('-' * 15)
        print(state)

        while not env.get_done_reward()[0]:
            if player_index == 0:
                start = time.time()
                action = player_0.get_best_action(state, player_index=player_index)
                print('player 1 action time: ', time.time() - start)
                player_index = 1
            else:
                start = time.time()
                action = player_1.get_best_action(state, player_index=player_index)
                print('player 2 action time: ', time.time() - start)
                player_index = 0
            env.step(action)
            state = env.board
            print('-' * 15)
            print(state)
        row, col = env.action_to_coord(action)

        # the player 1 win when place piece in (4, 4)
        assert env.get_done_winner()[1] == 1
        assert row == 4, col == 4

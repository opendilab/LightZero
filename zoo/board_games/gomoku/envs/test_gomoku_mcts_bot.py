from easydict import EasyDict
from zoo.board_games.gomoku.envs.gomoku_env import GomokuEnv
from zoo.board_games.mcts_bot import MCTSBot

import pytest

cfg = dict(
    board_size=5,
    prob_random_agent=0,
    prob_expert_agent=0,
    battle_mode='self_play_mode',
    scale=True,
    channel_last=True,
    agent_vs_human=False,
    bot_action_type='alpha_beta_pruning',  # {'v0', 'alpha_beta_pruning'}
    prob_random_action_in_bot=0.,
    check_action_to_connect4_in_bot_v0=False,
)


@pytest.mark.envtest
class TestGomokuBot:

    def test_gomoku_self_play_mode_player0_win(self):
        # player_0  num_simulation=1000, will win
        # player_1  num_simulation=1
        env = GomokuEnv(EasyDict(cfg))
        env.reset()
        state = env.board
        player_0 = MCTSBot(GomokuEnv, cfg, 'player 1', 100)  # player_index = 0, player = 1
        player_1 = MCTSBot(GomokuEnv, cfg, 'player 2', 1)  # player_index = 1, player = 2

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
                print('-' * 40)
            env.step(action)
            state = env.board
            print('#' * 15)
            print(state)
            print('#' * 15)
        assert env.get_done_winner()[1] == 1

    def test_gomoku_self_play_mode_player1_win(self):
        # player_0  num_simulation=1
        # player_1  num_simulation=1000, will win
        env = GomokuEnv(EasyDict(cfg))
        env.reset()
        state = env.board
        player_0 = MCTSBot(GomokuEnv, cfg, 'player 1', 1)  # player_index = 0, player = 1
        player_1 = MCTSBot(GomokuEnv, cfg, 'player 2', 100)  # player_index = 1, player = 2

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
                print('-' * 40)
            env.step(action)
            state = env.board
            print('#' * 15)
            print(state)
            print('#' * 15)
        assert env.get_done_winner()[1] == 2

    def test_gomoku_self_play_mode_draw(self):
        # player_0  num_simulation=1000
        # player_1  num_simulation=1000, will draw
        env = GomokuEnv(EasyDict(cfg))
        env.reset()
        state = env.board
        player_0 = MCTSBot(GomokuEnv, cfg, 'player 1', 100)  # player_index = 0, player = 1
        player_1 = MCTSBot(GomokuEnv, cfg, 'player 2', 100)  # player_index = 1, player = 2

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
                print('-' * 40)
            env.step(action)
            state = env.board
            print('#' * 15)
            print(state)
            print('#' * 15)
        assert env.get_done_winner()[1] == -1

    def test_gomoku_self_play_mode_case_1(self):
        env = GomokuEnv(EasyDict(cfg))
        init_state = [
            [1, 1, 1, 1, 0],
            [1, 0, 0, 0, 2],
            [0, 0, 2, 0, 2],
            [0, 2, 0, 0, 2],
            [2, 1, 1, 0, 0],
        ]
        player_0 = MCTSBot(GomokuEnv, cfg, 'player 1', 1000)  # player_index = 0, player = 1
        player_1 = MCTSBot(GomokuEnv, cfg, 'player 2', 1000)  # player_index = 1, player = 2
        player_index = 1  # player 1 fist

        env.reset(player_index, init_state)
        state = env.board

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
                print('-' * 40)
            env.step(action)
            state = env.board
            print('#' * 15)
            print(state)
            print('#' * 15)
            row, col = env.action_to_coord(action)
        assert env.get_done_winner()[1] == 2
        assert state[0, 4] == 2

    def test_gomoku_self_play_mode_case_2(self):
        env = GomokuEnv(EasyDict(cfg))
        init_state = [
            [0, 0, 2, 0, 0],
            [0, 1, 2, 0, 0],
            [2, 2, 1, 0, 0],
            [2, 0, 0, 1, 2],
            [1, 1, 1, 0, 0],
        ]
        player_0 = MCTSBot(GomokuEnv, cfg, 'player 1', 100)  # player_index = 0, player = 1
        player_1 = MCTSBot(GomokuEnv, cfg, 'player 2', 100)  # player_index = 1, player = 2
        player_index = 0  # player 1 fist

        env.reset(player_index, init_state)
        state = env.board

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
                print('-' * 40)
            env.step(action)
            state = env.board
            print('#' * 15)
            print(state)
            print('#' * 15)
            row, col = env.action_to_coord(action)
        assert env.get_done_winner()[1] == 1
        assert state[4, 4] == 1

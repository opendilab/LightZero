import sys
from easydict import EasyDict
sys.path.append('/YOUR/PATH/LightZero')
from zoo.board_games.mcts_bot import MCTSBot
from zoo.board_games.tictactoe.envs.tictactoe_env import TicTacToeEnv
from zoo.board_games.gomoku.envs.gomoku_env import GomokuEnv
import time
import numpy as np


def test_tictactoe_mcts_bot_vs_expert_bot(num_simulations=50):
    cfg = dict(
        prob_random_agent=0,
        prob_expert_agent=0,
        battle_mode='two_player_mode',
    )
    mcts_bot_time_list = []
    expert_action_time_list = []
    winner = []

    for i in range(100):
        print('-' * 10 + str(i) + '-' * 10)
        env = TicTacToeEnv(EasyDict(cfg))
        env.reset()
        state = env.board
        player = MCTSBot(TicTacToeEnv, cfg, 'a', num_simulations)  # player_index = 0, player = 1
        player_index = 0
        while not env.is_game_over()[0]:
            if player_index == 0:
                t1 = time.time()
                action = env.expert_action()
                #action = player.get_actions(state, player_index=player_index)
                t2 = time.time()
                #print("The time difference is :", t2-t1)
                mcts_bot_time_list.append(t2 - t1)
                player_index = 1
            else:
                t1 = time.time()
                #action = env.expert_action()
                action = player.get_actions(state, player_index=player_index)
                t2 = time.time()
                #print("The time difference is :", t2-t1)
                expert_action_time_list.append(t2 - t1)
                player_index = 0
            env.step(action)
            state = env.board
            print(state)

        winner.append(env.have_winner()[1])

    mcts_bot_mu = np.mean(mcts_bot_time_list)
    mcts_bot_var = np.var(mcts_bot_time_list)

    expert_action_mu = np.mean(expert_action_time_list)
    expert_action_var = np.var(expert_action_time_list)

    print('num_simulations={}\n'.format(num_simulations))
    print('mcts_bot_time_list={}\n'.format(mcts_bot_time_list))
    print('mcts_bot_mu={}, mcts_bot_var={}\n'.format(mcts_bot_mu, mcts_bot_var))

    print('expert_action_time_list={}\n'.format(expert_action_time_list))
    print('expert_action_mu={}, expert_action_var={}\n'.format(expert_action_mu, expert_action_var))

    print(
        'winner={}, draw={}, player1={}, player2={}\n'.format(
            winner, winner.count(-1), winner.count(1), winner.count(2)
        )
    )


def test_gomoku_mcts_bot_vs_expert_bot(num_simulations):
    cfg = dict(
        board_size=5,
        prob_random_agent=0,
        battle_mode='two_player_mode',
    )
    mcts_bot_time_list = []
    expert_action_time_list = []
    winner = []

    for i in range(50):
        print('-' * 10 + str(i) + '-' * 10)
        env = GomokuEnv(EasyDict(cfg))
        env.reset()
        state = env.board
        player = MCTSBot(GomokuEnv, cfg, 'a', num_simulations)  # player_index = 0, player = 1
        player_index = 0
        while not env.is_game_over()[0]:
            if player_index == 0:
                t1 = time.time()
                action = env.expert_action()
                #action = player.get_actions(state, player_index=player_index)
                t2 = time.time()
                #print("The time difference is :", t2-t1)
                mcts_bot_time_list.append(t2 - t1)
                player_index = 1
            else:
                t1 = time.time()
                #action = env.expert_action()
                action = player.get_actions(state, player_index=player_index)
                t2 = time.time()
                #print("The time difference is :", t2-t1)
                expert_action_time_list.append(t2 - t1)
                player_index = 0
            env.step(action)
            state = env.board
            print(state)

        winner.append(env.have_winner()[1])

    mcts_bot_mu = np.mean(mcts_bot_time_list)
    mcts_bot_var = np.var(mcts_bot_time_list)

    expert_action_mu = np.mean(expert_action_time_list)
    expert_action_var = np.var(expert_action_time_list)

    print('num_simulations={}\n'.format(num_simulations))
    print('mcts_bot_time_list={}\n'.format(mcts_bot_time_list))
    print('mcts_bot_mu={}, mcts_bot_var={}\n'.format(mcts_bot_mu, mcts_bot_var))

    print('expert_action_time_list={}\n'.format(expert_action_time_list))
    print('expert_action_mu={}, expert_action_var={}\n'.format(expert_action_mu, expert_action_var))

    print(
        'winner={}, draw={}, player1={}, player2={}\n'.format(
            winner, winner.count(-1), winner.count(1), winner.count(2)
        )
    )


if __name__ == '__main__':
    test_tictactoe_mcts_bot_vs_expert_bot(num_simulations=50)
    # test_tictactoe_mcts_bot_vs_expert_bot(num_simulations=100)
    # test_tictactoe_mcts_bot_vs_expert_bot(num_simulations=500)
    # test_tictactoe_mcts_bot_vs_expert_bot(num_simulations=1000)
    # test_gomoku_mcts_bot_vs_expert_bot(num_simulations=1000)

"""
Overview:
    Implemrnt games between different bots to test the win rates and the speed.
Example:
    test_tictactoe_mcts_bot_vs_alphabeta_bot means a game between mcts_bot and alphabeta_bot where 
    mcts_bot makes the first move(bots on the left make the first move).
"""
import time

import numpy as np
from easydict import EasyDict

from zoo.board_games.gomoku.envs.gomoku_env import GomokuEnv
from zoo.board_games.mcts_bot import MCTSBot
from zoo.board_games.tictactoe.envs.tictactoe_env import TicTacToeEnv

cfg_tictactoe = dict(
    battle_mode='self_play_mode',
    agent_vs_human=False,
    bot_action_type=['v0', 'alpha_beta_pruning'],  # {'v0', 'alpha_beta_pruning'}
    prob_random_agent=0,
    prob_expert_agent=0,
    channel_last=True,
    scale=True,
    prob_random_action_in_bot=0.,
)


def test_tictactoe_mcts_bot_vs_rule_bot_v0_bot(num_simulations=50):
    """
    Overview:
        A tictactoe game between mcts_bot and rule_bot, where rule_bot take the first move.
    Arguments:
        - num_simulations (:obj:`int`): The number of the simulations required to find the best move.
    """
    # List to record the time required for each decision round and the winner.
    mcts_bot_time_list = []
    bot_action_time_list = []
    winner = []

    # Repeat the game for 10 rounds.
    for i in range(100):
        print('-' * 10 + str(i) + '-' * 10)
        # Initialize the game, where there are two players: player 1 and player 2.
        env = TicTacToeEnv(EasyDict(cfg_tictactoe))
        # Reset the environment, set the board to a clean board and the  start player to be player 1.
        env.reset()
        state = env.board
        player = MCTSBot(env, 'a', num_simulations)  # player_index = 0, player = 1
        # Set player 1 to move first.
        player_index = 0
        while not env.get_done_reward()[0]:
            """
            Overview:
                The two players take turns to make moves, and the time required for each decision is recorded.
            """
            # Set rule_bot to be player 1.
            if player_index == 0:
                t1 = time.time()
                action = env.bot_action()
                # action = player.get_actions(state, player_index=player_index)
                t2 = time.time()
                # print("The time difference is :", t2-t1)
                mcts_bot_time_list.append(t2 - t1)
                player_index = 1
            # Set mcts_bot to be player 2.
            else:
                t1 = time.time()
                # action = env.bot_action()
                action = player.get_actions(state, player_index=player_index)
                t2 = time.time()
                # print("The time difference is :", t2-t1)
                bot_action_time_list.append(t2 - t1)
                player_index = 0
            env.step(action)
            state = env.board
            print(state)

        # Record the winner.
        winner.append(env.get_done_winner()[1])

    # Calculate the variance and mean of decision times.
    mcts_bot_mu = np.mean(mcts_bot_time_list)
    mcts_bot_var = np.var(mcts_bot_time_list)

    bot_action_mu = np.mean(bot_action_time_list)
    bot_action_var = np.var(bot_action_time_list)

    # Print the information of the games.
    print('num_simulations={}\n'.format(num_simulations))
    print('mcts_bot_time_list={}\n'.format(mcts_bot_time_list))
    print('mcts_bot_mu={}, mcts_bot_var={}\n'.format(mcts_bot_mu, mcts_bot_var))

    print('bot_action_time_list={}\n'.format(bot_action_time_list))
    print('bot_action_mu={}, bot_action_var={}\n'.format(bot_action_mu, bot_action_var))

    print(
        'winner={}, draw={}, player1={}, player2={}\n'.format(
            winner, winner.count(-1), winner.count(1), winner.count(2)
        )
    )


def test_tictactoe_alphabeta_bot_vs_rule_bot_v0_bot(num_simulations=50):
    """
    Overview:
        A tictactoe game between alphabeta_bot and rule_bot, where alphabeta_bot take the first move.
    Arguments:
        - num_simulations (:obj:`int`): The number of the simulations required to find the best move.
    """
    # List to record the time required for each decision round and the winner.
    alphabeta_pruning_time_list = []
    rule_bot_v0_time_list = []
    winner = []

    # Repeat the game for 10 rounds.
    for i in range(10):
        print('-' * 10 + str(i) + '-' * 10)
        # Initialize the game, where there are two players: player 1 and player 2.
        env = TicTacToeEnv(EasyDict(cfg_tictactoe))
        # Reset the environment, set the board to a clean board and the  start player to be player 1.
        env.reset(start_player_index=1)
        state = env.board
        player = MCTSBot(env, 'a', num_simulations)  # player_index = 0, player = 1
        # Set player 2 to move first.
        player_index = 1
        while not env.get_done_reward()[0]:
            """
            Overview:
                The two players take turns to make moves, and the time required for each decision is recorded.
            """
            # Set rule_bot to be player 1.
            if player_index == 0:
                t1 = time.time()
                action = env.rule_bot_v0()
                # action = player.get_actions(state, player_index=player_index)
                t2 = time.time()
                # print("The time difference is :", t2-t1)
                # mcts_bot_time_list.append(t2 - t1)
                rule_bot_v0_time_list.append(t2 - t1)

                player_index = 1
            # Set alpha_beta_bot to be player 2.
            else:
                t1 = time.time()
                action = env.bot_action_alpha_beta_pruning()
                # action = player.get_actions(state, player_index=player_index)
                t2 = time.time()
                # print("The time difference is :", t2-t1)
                alphabeta_pruning_time_list.append(t2 - t1)
                player_index = 0
            env.step(action)
            state = env.board
            if env.get_done_reward()[0]:
                print(state)

        # Record the winner.
        winner.append(env.get_done_winner()[1])

    # Calculate the variance and mean of decision times.
    alphabeta_pruning_mu = np.mean(alphabeta_pruning_time_list)
    alphabeta_pruning_var = np.var(alphabeta_pruning_time_list)

    rule_bot_v0_mu = np.mean(rule_bot_v0_time_list)
    rule_bot_v0_var = np.var(rule_bot_v0_time_list)

    # Print the information of the games.
    print('num_simulations={}\n'.format(num_simulations))
    print('alphabeta_pruning_time_list={}\n'.format(alphabeta_pruning_time_list))
    print('alphabeta_pruning_mu={}, alphabeta_pruning_var={}\n'.format(alphabeta_pruning_mu, alphabeta_pruning_var))

    print('rule_bot_v0_time_list={}\n'.format(rule_bot_v0_time_list))
    print('rule_bot_v0_mu={}, bot_action_var={}\n'.format(rule_bot_v0_mu, rule_bot_v0_var))

    print(
        'winner={}, draw={}, player1={}, player2={}\n'.format(
            winner, winner.count(-1), winner.count(1), winner.count(2)
        )
    )


def test_tictactoe_alphabeta_bot_vs_mcts_bot(num_simulations=50):
    """
    Overview:
        A tictactoe game between alphabeta_bot and mcts_bot, where mcts_bot take the first move.
    Arguments:
        - num_simulations (:obj:`int`): The number of the simulations required to find the best move.
    """
    # List to record the time required for each decision round and the winner.
    alphabeta_pruning_time_list = []
    mcts_bot_time_list = []
    winner = []

    # Repeat the game for 10 rounds.
    for i in range(10):
        print('-' * 10 + str(i) + '-' * 10)
        # Initialize the game, where there are two players: player 1 and player 2.
        env = TicTacToeEnv(EasyDict(cfg_tictactoe))
        # Reset the environment, set the board to a clean board and the  start player to be player 1.
        env.reset(start_player_index=1)
        state = env.board
        player = MCTSBot(env, 'a', num_simulations)  # player_index = 0, player = 1
        # Set player 2 to move first.
        player_index = 1
        while not env.get_done_reward()[0]:
            """
            Overview:
                The two players take turns to make moves, and the time required for each decision is recorded.
            """
            # Set mcts_bot to be player 1.
            if player_index == 0:
                t1 = time.time()
                # action = env.rule_bot_v0()
                action = player.get_actions(state, player_index=player_index)
                t2 = time.time()
                # print("The time difference is :", t2-t1)
                mcts_bot_time_list.append(t2 - t1)
                # rule_bot_v0_time_list.append(t2 - t1)

                player_index = 1
            # Set alpha_beta_bot to be player 2.
            else:
                t1 = time.time()
                action = env.bot_action_alpha_beta_pruning()
                # action = player.get_actions(state, player_index=player_index)
                t2 = time.time()
                # print("The time difference is :", t2-t1)
                alphabeta_pruning_time_list.append(t2 - t1)
                player_index = 0
            env.step(action)
            state = env.board
            print(state)
            print(action)
            if env.get_done_reward()[0]:
                print(state)

        # Record the winner.
        winner.append(env.get_done_winner()[1])

    # Calculate the variance and mean of decision times.
    alphabeta_pruning_mu = np.mean(alphabeta_pruning_time_list)
    alphabeta_pruning_var = np.var(alphabeta_pruning_time_list)

    mcts_bot_mu = np.mean(mcts_bot_time_list)
    mcts_bot_var = np.var(mcts_bot_time_list)

    # Print the information of the games.
    print('num_simulations={}\n'.format(num_simulations))
    print('alphabeta_pruning_time_list={}\n'.format(alphabeta_pruning_time_list))
    print('alphabeta_pruning_mu={}, alphabeta_pruning_var={}\n'.format(alphabeta_pruning_mu, alphabeta_pruning_var))

    print('mcts_bot_time_list={}\n'.format(mcts_bot_time_list))
    print('mcts_bot_mu={}, mcts_bot_var={}\n'.format(mcts_bot_mu, mcts_bot_var))

    print(
        'winner={}, draw={}, player1={}, player2={}\n'.format(
            winner, winner.count(-1), winner.count(1), winner.count(2)
        )
    )


def test_tictactoe_rule_bot_v0_bot_vs_alphabeta_bot(num_simulations=50):
    """
    Overview:
        A tictactoe game between rule_bot and alphabeta_bot, where rule_bot take the first move.
    Arguments:
        - num_simulations (:obj:`int`): The number of the simulations required to find the best move.
    """
    # List to record the time required for each decision round and the winner.
    alphabeta_pruning_time_list = []
    rule_bot_v0_time_list = []
    winner = []

    # Repeat the game for 10 rounds.
    for i in range(10):
        print('-' * 10 + str(i) + '-' * 10)
        # Initialize the game, where there are two players: player 1 and player 2.
        env = TicTacToeEnv(EasyDict(cfg_tictactoe))
        # Reset the environment, set the board to a clean board and the  start player to be player 1.
        env.reset()
        state = env.board
        player = MCTSBot(env, 'a', num_simulations)  # player_index = 0, player = 1
        # Set player 1 to move first.
        player_index = 0
        while not env.get_done_reward()[0]:
            """
            Overview:
                The two players take turns to make moves, and the time required for each decision is recorded.
            """
            # Set rule_bot to be player 1.
            if player_index == 0:
                t1 = time.time()
                action = env.rule_bot_v0()
                # action = player.get_actions(state, player_index=player_index)
                t2 = time.time()
                # print("The time difference is :", t2-t1)
                # mcts_bot_time_list.append(t2 - t1)
                rule_bot_v0_time_list.append(t2 - t1)

                player_index = 1
            # Set alpha_beta_bot to be player 2.
            else:
                t1 = time.time()
                action = env.bot_action_alpha_beta_pruning()
                # action = player.get_actions(state, player_index=player_index)
                t2 = time.time()
                # print("The time difference is :", t2-t1)
                alphabeta_pruning_time_list.append(t2 - t1)
                player_index = 0
            env.step(action)
            state = env.board
            if env.get_done_reward()[0]:
                print(state)

        # Record the winner.
        winner.append(env.get_done_winner()[1])

    # Calculate the variance and mean of decision times.
    alphabeta_pruning_mu = np.mean(alphabeta_pruning_time_list)
    alphabeta_pruning_var = np.var(alphabeta_pruning_time_list)

    rule_bot_v0_mu = np.mean(rule_bot_v0_time_list)
    rule_bot_v0_var = np.var(rule_bot_v0_time_list)

    # Print the information of the games.
    print('num_simulations={}\n'.format(num_simulations))
    print('alphabeta_pruning_time_list={}\n'.format(alphabeta_pruning_time_list))
    print('alphabeta_pruning_mu={}, alphabeta_pruning_var={}\n'.format(alphabeta_pruning_mu, alphabeta_pruning_var))

    print('rule_bot_v0_time_list={}\n'.format(rule_bot_v0_time_list))
    print('rule_bot_v0_mu={}, bot_action_var={}\n'.format(rule_bot_v0_mu, rule_bot_v0_var))

    print(
        'winner={}, draw={}, player1={}, player2={}\n'.format(
            winner, winner.count(-1), winner.count(1), winner.count(2)
        )
    )


def test_tictactoe_mcts_bot_vs_alphabeta_bot(num_simulations=50):
    """
    Overview:
        A tictactoe game between mcts_bot and alphabeta_bot, where mcts_bot take the first move.
    Arguments:
        - num_simulations (:obj:`int`): The number of the simulations required to find the best move.
    """
    # List to record the time required for each decision round and the winner.
    alphabeta_pruning_time_list = []
    mcts_bot_time_list = []
    winner = []

    # Repeat the game for 10 rounds.
    for i in range(10):
        print('-' * 10 + str(i) + '-' * 10)
        # Initialize the game, where there are two players: player 1 and player 2.
        env = TicTacToeEnv(EasyDict(cfg_tictactoe))
        # Reset the environment, set the board  to a clean board and the  start player to be player 1.
        env.reset()
        state = env.board
        player = MCTSBot(env, 'a', num_simulations)  # player_index = 0, player = 1
        # Set player 1 to move first.
        player_index = 0
        while not env.get_done_reward()[0]:
            """
            Overview:
                The two players take turns to make moves, and the time required for each decision is recorded.
            """
            # Set mcts_bot to be player 1.
            if player_index == 0:
                t1 = time.time()
                # action = env.mcts_bot()
                action = player.get_actions(state, player_index=player_index, best_action_type = "most_visit")
                t2 = time.time()
                # print("The time difference is :", t2-t1)
                # mcts_bot_time_list.append(t2 - t1)
                mcts_bot_time_list.append(t2 - t1)

                player_index = 1

            # Set alpha_beta_bot to be player 2.
            else:
                t1 = time.time()
                action = env.bot_action_alpha_beta_pruning()
                # action = player.get_actions(state, player_index=player_index)
                t2 = time.time()
                # print("The time difference is :", t2-t1)
                alphabeta_pruning_time_list.append(t2 - t1)
                player_index = 0
            env.step(action)
            state = env.board
            # Print the result of the game.
            if env.get_done_reward()[0]:
                print(state)
        # Record the winner.
        winner.append(env.get_done_winner()[1])

    # Calculate the variance and mean of decision times.
    alphabeta_pruning_mu = np.mean(alphabeta_pruning_time_list)
    alphabeta_pruning_var = np.var(alphabeta_pruning_time_list)

    mcts_bot_mu = np.mean(mcts_bot_time_list)
    mcts_bot_var = np.var(mcts_bot_time_list)

    # Print the information of the games.
    print('num_simulations={}\n'.format(num_simulations))
    print('alphabeta_pruning_time_list={}\n'.format(alphabeta_pruning_time_list))
    print('alphabeta_pruning_mu={}, alphabeta_pruning_var={}\n'.format(alphabeta_pruning_mu, alphabeta_pruning_var))

    print('mcts_bot_time_list={}\n'.format(mcts_bot_time_list))
    print('mcts_bot_mu={}, bot_action_var={}\n'.format(mcts_bot_mu, mcts_bot_var))

    print(
        'winner={}, draw={}, player1={}, player2={}\n'.format(
            winner, winner.count(-1), winner.count(1), winner.count(2)
        )
    )


cfg_gomoku = dict(
    board_size=5,
    battle_mode='self_play_mode',
    bot_action_type='v0',  # {'v0', 'alpha_beta_pruning'}
    agent_vs_human=False,
    prob_random_agent=0,
    channel_last=True,
    scale=True,
    prob_random_action_in_bot=0.,
    check_action_to_connect4_in_bot_v0=False,
)


def test_gomoku_mcts_bot_vs_rule_bot_v0_bot(num_simulations=50):
    """
    Overview:
        A tictactoe game between mcts_bot and rule_bot, where rule_bot take the first move.
    Arguments:
        - num_simulations (:obj:`int`): The number of the simulations required to find the best move.
    """
    # List to record the time required for each decision round and the winner.
    mcts_bot_time_list = []
    bot_action_time_list = []
    winner = []

    # Repeat the game for 10 rounds.
    for i in range(10):
        print('-' * 10 + str(i) + '-' * 10)
        # Initialize the game, where there are two players: player 1 and player 2.
        env = GomokuEnv(EasyDict(cfg_gomoku))
        # Reset the environment, set the board to a clean board and the  start player to be player 1.
        env.reset()
        state = env.board
        player = MCTSBot(env, 'a', num_simulations)  # player_index = 0, player = 1
        # Set player 1 to move first.
        player_index = 0
        while not env.get_done_reward()[0]:
            """
            Overview:
                The two players take turns to make moves, and the time required for each decision is recorded.
            """
            # Set rule_bot to be player 1.
            if player_index == 0:
                t1 = time.time()
                action = env.bot_action()
                # action = player.get_actions(state, player_index=player_index)
                t2 = time.time()
                # print("The time difference is :", t2-t1)
                mcts_bot_time_list.append(t2 - t1)
                player_index = 1
            
            # Set mcts_bot to be player 2.
            else:
                t1 = time.time()
                # action = env.bot_action()
                action = player.get_actions(state, player_index=player_index)
                t2 = time.time()
                # print("The time difference is :", t2-t1)
                bot_action_time_list.append(t2 - t1)
                player_index = 0
            env.step(action)
            state = env.board
            # Print the result of the game.
            if env.get_done_reward()[0]:
                print(state)

        # Record the winner.
        winner.append(env.get_done_winner()[1])

    # Calculate the variance and mean of decision times.
    mcts_bot_mu = np.mean(mcts_bot_time_list)
    mcts_bot_var = np.var(mcts_bot_time_list)

    bot_action_mu = np.mean(bot_action_time_list)
    bot_action_var = np.var(bot_action_time_list)

    # Print the information of the games.
    print('num_simulations={}\n'.format(num_simulations))
    print('mcts_bot_time_list={}\n'.format(mcts_bot_time_list))
    print('mcts_bot_mu={}, mcts_bot_var={}\n'.format(mcts_bot_mu, mcts_bot_var))

    print('bot_action_time_list={}\n'.format(bot_action_time_list))
    print('bot_action_mu={}, bot_action_var={}\n'.format(bot_action_mu, bot_action_var))

    print(
        'winner={}, draw={}, player1={}, player2={}\n'.format(
            winner, winner.count(-1), winner.count(1), winner.count(2)
        )
    )


if __name__ == '__main__':
    # ==============================================================
    # test win rate between alphabeta_bot and rule_bot_v0/mcts_bot
    # ==============================================================
    # test_tictactoe_alphabeta_bot_vs_rule_bot_v0_bot()
    # test_tictactoe_rule_bot_v0_bot_vs_alphabeta_bot()
    # test_tictactoe_alphabeta_bot_vs_mcts_bot(num_simulations=2000)
    test_tictactoe_mcts_bot_vs_alphabeta_bot(num_simulations=2000)

    # ==============================================================
    # test win rate between mcts_bot and rule_bot_v0
    # ==============================================================
    # test_tictactoe_mcts_bot_vs_rule_bot_v0_bot(num_simulations=50)
    # test_tictactoe_mcts_bot_vs_rule_bot_v0_bot(num_simulations=100)
    # test_tictactoe_mcts_bot_vs_rule_bot_v0_bot(num_simulations=500)
    # test_tictactoe_mcts_bot_vs_rule_bot_v0_bot(num_simulations=1000)
    # test_gomoku_mcts_bot_vs_rule_bot_v0_bot(num_simulations=1000)

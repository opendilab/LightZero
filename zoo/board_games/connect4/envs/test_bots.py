import pytest
from easydict import EasyDict
from connect4_env import Connect4Env
from zoo.board_games.mcts_bot import MCTSBot
import time
import numpy as np

cfg = EasyDict(
    battle_mode='self_play_mode',
    mcts_mode='self_play_mode',
    channel_last=True,
    scale=True,
    agent_vs_human=False,
    prob_random_agent=0,
    prob_expert_agent=0,
    bot_action_type='rule',
    screen_scaling=9,
    save_replay=True,
    prob_random_action_in_bot = 0
)


def test_mcts_bot_vs_rule_bot(num_simulations=200):
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
    for i in range(1):
        print('-' * 10 + str(i) + '-' * 10)
        # Initialize the game, where there are two players: player 1 and player 2.
        env = Connect4Env(EasyDict(cfg))
        # Reset the environment, set the board to a clean board and the  start player to be player 1.
        env.reset()
        state = env.board
        cfg_temp = EasyDict(cfg.copy())
        cfg_temp.save_replay = False
        env_mcts = Connect4Env(EasyDict(cfg_temp))
        player = MCTSBot(env_mcts, 'a', num_simulations)  # player_index = 0, player = 1
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
                # action = env.bot_action()
                action = player.get_actions(state, player_index=player_index)
                t2 = time.time()
                # print("The time difference is :", t2-t1)
                mcts_bot_time_list.append(t2 - t1)
                player_index = 1
            # Set mcts_bot to be player 2.
            else:
                t1 = time.time()
                action = env.bot_action()
                # action = player.get_actions(state, player_index=player_index)
                t2 = time.time()
                # print("The time difference is :", t2-t1)
                bot_action_time_list.append(t2 - t1)
                player_index = 0
            env.step(action)
            state = env.board
            print(np.array(state).reshape(6, 7))

        # Record the winner.
        winner.append(env.get_done_winner()[1])

    # Calculate the variance and mean of decision times.
    mcts_bot_mu = np.mean(mcts_bot_time_list)
    mcts_bot_var = np.var(mcts_bot_time_list)

    bot_action_mu = np.mean(bot_action_time_list)
    bot_action_var = np.var(bot_action_time_list)

    # Print the information of the games.
    print('num_simulations={}\n'.format(num_simulations))
    # print('mcts_bot_time_list={}\n'.format(mcts_bot_time_list))
    print('mcts_bot_mu={}, mcts_bot_var={}\n'.format(mcts_bot_mu, mcts_bot_var))

    # print('bot_action_time_list={}\n'.format(bot_action_time_list))
    print('bot_action_mu={}, bot_action_var={}\n'.format(bot_action_mu, bot_action_var))

    print(
        'winner={}, draw={}, player1={}, player2={}\n'.format(
            winner, winner.count(-1), winner.count(1), winner.count(2)
        )
    )


def test_mcts_bot_vs_mcts_bot(num_simulations=50):
    """
    Overview:
        A tictactoe game between mcts_bot and rule_bot, where rule_bot take the first move.
    Arguments:
        - num_simulations (:obj:`int`): The number of the simulations required to find the best move.
    """
    # List to record the time required for each decision round and the winner.
    mcts_bot1_time_list = []
    mcts_bot2_time_list = []
    winner = []

    # Repeat the game for 10 rounds.
    for i in range(10):
        print('-' * 10 + str(i) + '-' * 10)
        # Initialize the game, where there are two players: player 1 and player 2.
        env = Connect4Env(EasyDict(cfg))
        # Reset the environment, set the board to a clean board and the  start player to be player 1.
        env.reset()
        state = env.board
        player1 = MCTSBot(env, 'a', 200)  # player_index = 0, player = 1
        player2 = MCTSBot(env, 'a', 1000)
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
                # action = env.bot_action()
                action = player1.get_actions(state, player_index=player_index)
                t2 = time.time()
                # print("The time difference is :", t2-t1)
                mcts_bot1_time_list.append(t2 - t1)
                player_index = 1
            # Set mcts_bot to be player 2.
            else:
                t1 = time.time()
                # action = env.bot_action()
                action = player2.get_actions(state, player_index=player_index)
                t2 = time.time()
                # print("The time difference is :", t2-t1)
                mcts_bot2_time_list.append(t2 - t1)
                player_index = 0
            env.step(action)
            state = env.board
            print(np.array(state).reshape(6, 7))

        # Record the winner.
        winner.append(env.get_done_winner()[1])

    # Calculate the variance and mean of decision times.
    mcts_bot1_mu = np.mean(mcts_bot1_time_list)
    mcts_bot1_var = np.var(mcts_bot1_time_list)

    mcts_bot2_mu = np.mean(mcts_bot2_time_list)
    mcts_bot2_var = np.var(mcts_bot2_time_list)

    # Print the information of the games.
    print('num_simulations={}\n'.format(200))
    print('mcts_bot1_time_list={}\n'.format(mcts_bot1_time_list))
    print('mcts_bot1_mu={}, mcts_bot1_var={}\n'.format(mcts_bot1_mu, mcts_bot1_var))

    print('num_simulations={}\n'.format(1000))
    print('mcts_bot2_time_list={}\n'.format(mcts_bot2_time_list))
    print('mcts_bot2_mu={}, mcts_bot2_var={}\n'.format(mcts_bot2_mu, mcts_bot2_var))

    print(
        'winner={}, draw={}, player1={}, player2={}\n'.format(
            winner, winner.count(-1), winner.count(1), winner.count(2)
        )
    )


if __name__ == '__main__':
    # test_mcts_bot_vs_rule_bot(50)
    test_mcts_bot_vs_rule_bot(200)
    # test_mcts_bot_vs_rule_bot(1000) 
    # test_mcts_bot_vs_mcts_bot()

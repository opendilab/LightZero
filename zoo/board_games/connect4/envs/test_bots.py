import time

import numpy as np
import pytest
from easydict import EasyDict
import psutil

from connect4_env import Connect4Env
from zoo.board_games.mcts_bot import MCTSBot


def get_memory_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    return memory_info.rss

@pytest.mark.unittest
class TestConnect4Bot():
    """
    Overview:
        This class is used to test the Connect4 Bots.
    """

    def setup(self) -> None:
        """
        Overview:
            This method is responsible for setting up the initial configurations required for the game environment.
            It creates an instance of the Connect4Env class and Connect4RuleBot class.
        """
        self.cfg = EasyDict(
            battle_mode='self_play_mode',
            mcts_mode='self_play_mode',
            channel_last=True,
            scale=True,
            agent_vs_human=False,
            prob_random_agent=0,
            prob_expert_agent=0,
            bot_action_type='rule',
            screen_scaling=9,
            render_mode= None,
            prob_random_action_in_bot=0,
        )

    def test_mcts_bot_vs_rule_bot(self, num_simulations: int = 200) -> None:
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
            memory_usage = get_memory_usage()
            print(f"初始内存使用量: {memory_usage} 字节")
            # Initialize the game, where there are two players: player 1 and player 2.
            env = Connect4Env(EasyDict(self.cfg))
            # Reset the environment, set the board to a clean board and the  start player to be player 1.
            env.reset(replay_name_suffix=f'test{i}')
            state = env.board
            self.cfg_temp = EasyDict(self.cfg.copy())
            self.cfg_temp.save_replay = False
            env_mcts = Connect4Env(EasyDict(self.cfg_temp))
            player = MCTSBot(env_mcts, 'a', num_simulations)  # player_index = 0, player = 1
            # Set player 1 to move first.
            player_index = 0
            step = 1
            while not env.get_done_reward()[0]:
                """
                Overview:
                    The two players take turns to make moves, and the time required for each decision is recorded.
                """
                # Set rule_bot to be player 1.
                if player_index == 0:
                    t1 = time.time()
                    # action = env.bot_action()
                    action, node = player.get_actions(state, step, player_index=player_index)
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
                step += 1
                print(np.array(state).reshape(6, 7))
                temp = memory_usage
                memory_usage = get_memory_usage()
                memory_cost = memory_usage - temp
                print(f"搜索后内存使用量: {memory_usage} 字节")
                print(f"搜索增加的内存使用量: {memory_cost} 字节")

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

    def test_mcts_bot_vs_mcts_bot(self, num_simulations_1: int = 50, num_simulations_2: int = 50) -> None:
        """
        Overview:
            A tictactoe game between two mcts_bots. 
        Arguments:
            - num_simulations_1 (:obj:`int`): The number of the simulations of player 1 required to find the best move.
            - num_simulations_2 (:obj:`int`): The number of the simulations of player 2 required to find the best move.
        """
        # List to record the time required for each decision round and the winner.
        mcts_bot1_time_list = []
        mcts_bot2_time_list = []
        winner = []

        # Repeat the game for 10 rounds.
        for i in range(1):
            print('-' * 10 + str(i) + '-' * 10)
            memory_usage = get_memory_usage()
            print(f"初始内存使用量: {memory_usage} 字节")
            # Initialize the game, where there are two players: player 1 and player 2.
            env = Connect4Env(EasyDict(self.cfg))
            # Reset the environment, set the board to a clean board and the  start player to be player 1.
            env.reset()
            state = env.board
            player1 = MCTSBot(env, 'a', num_simulations_1)  # player_index = 0, player = 1
            player2 = MCTSBot(env, 'b', num_simulations_2)
            # Set player 1 to move first.
            player_index = 0
            step = 1
            node = None
            while not env.get_done_reward()[0]:
                """
                Overview:
                    The two players take turns to make moves, and the time required for each decision is recorded.
                """
                # Set rule_bot to be player 1.
                if player_index == 0:
                    t1 = time.time()
                    # action = env.bot_action()
                    action, node, visit = player1.get_actions(state, step, player_index)
                    t2 = time.time()
                    # print("The time difference is :", t2-t1)
                    mcts_bot1_time_list.append(t2 - t1)
                    player_index = 1
                # Set mcts_bot to be player 2.
                else:
                    t1 = time.time()
                    # action = env.bot_action()
                    action, node, visit = player2.get_actions(state, step, player_index, num_simulation=visit)
                    t2 = time.time()
                    # print("The time difference is :", t2-t1)
                    mcts_bot2_time_list.append(t2 - t1)
                    player_index = 0
                env.step(action)
                step += 1
                state = env.board
                print(np.array(state).reshape(6, 7))
                temp = memory_usage
                memory_usage = get_memory_usage()
                memory_cost = memory_usage - temp
                print(f"搜索后内存使用量: {memory_usage} 字节")
                print(f"搜索增加的内存使用量: {memory_cost} 字节")

            # Record the winner.
            winner.append(env.get_done_winner()[1])

        # Calculate the variance and mean of decision times.
        mcts_bot1_mu = np.mean(mcts_bot1_time_list)
        mcts_bot1_var = np.var(mcts_bot1_time_list)

        mcts_bot2_mu = np.mean(mcts_bot2_time_list)
        mcts_bot2_var = np.var(mcts_bot2_time_list)

        # Print the information of the games.
        print('num_simulations={}\n'.format(num_simulations_1))
        print('mcts_bot1_time_list={}\n'.format(mcts_bot1_time_list))
        print('mcts_bot1_mu={}, mcts_bot1_var={}\n'.format(mcts_bot1_mu, mcts_bot1_var))

        print('num_simulations={}\n'.format(num_simulations_2))
        print('mcts_bot2_time_list={}\n'.format(mcts_bot2_time_list))
        print('mcts_bot2_mu={}, mcts_bot2_var={}\n'.format(mcts_bot2_mu, mcts_bot2_var))

        print(
            'winner={}, draw={}, player1={}, player2={}\n'.format(
                winner, winner.count(-1), winner.count(1), winner.count(2)
            )
        )

    def test_rule_bot_vs_rule_bot(self) -> None:
        """
        Overview:
            A tictactoe game between mcts_bot and rule_bot, where rule_bot take the first move.
        Arguments:
            - num_simulations (:obj:`int`): The number of the simulations required to find the best move.
        """
        # List to record the time required for each decision round and the winner.
        bot_action_time_list2 = []
        bot_action_time_list1 = []
        winner = []

        # Repeat the game for 10 rounds.
        for i in range(10):
            print('-' * 10 + str(i) + '-' * 10)
            memory_usage = get_memory_usage()
            print(f"初始内存使用量: {memory_usage} 字节")
            # Initialize the game, where there are two players: player 1 and player 2.
            env = Connect4Env(EasyDict(self.cfg))
            # Reset the environment, set the board to a clean board and the  start player to be player 1.
            env.reset(replay_name_suffix=f'test{i}')
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
                    t2 = time.time()
                    # print("The time difference is :", t2-t1)
                    bot_action_time_list1.append(t2 - t1)
                    player_index = 1
                # Set mcts_bot to be player 2.
                else:
                    t1 = time.time()
                    action = env.bot_action()
                    # action = player.get_actions(state, player_index=player_index)
                    t2 = time.time()
                    # print("The time difference is :", t2-t1)
                    bot_action_time_list2.append(t2 - t1)
                    player_index = 0
                env.step(action)
                state = env.board
                print(np.array(state).reshape(6, 7))
                temp = memory_usage
                memory_usage = get_memory_usage()
                memory_cost = memory_usage - temp
                print(f"搜索后内存使用量: {memory_usage} 字节")
                print(f"搜索增加的内存使用量: {memory_cost} 字节")

            # Record the winner.
            winner.append(env.get_done_winner()[1])

        # Calculate the variance and mean of decision times.
        bot_action_mu1 = np.mean(bot_action_time_list1)
        bot_action_var1 = np.var(bot_action_time_list1)

        bot_action_mu2 = np.mean(bot_action_time_list2)
        bot_action_var2 = np.var(bot_action_time_list2)

        # Print the information of the games.
        # print('bot_action_time_list1={}\n'.format(bot_action_time_list1))
        print('bot_action_mu1={}, bot_action_var1={}\n'.format(bot_action_mu1, bot_action_var1))

        # print('bot_action_time_list={}\n'.format(bot_action_time_list))
        print('bbot_action_mu2={}, bot_action_var2={}\n'.format(bot_action_mu2, bot_action_var2))

        print(
            'winner={}, draw={}, player1={}, player2={}\n'.format(
                winner, winner.count(-1), winner.count(1), winner.count(2)
            )
        )


if __name__ == "__main__":
    test=TestConnect4Bot()
    test.setup()
    test.test_mcts_bot_vs_mcts_bot(2000,200)
    # test.test_mcts_bot_vs_rule_bot()
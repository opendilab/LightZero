import numpy as np
import pytest
from easydict import EasyDict

from connect4_env import Connect4Env
from zoo.board_games.connect4.envs.rule_bot import Connect4RuleBot


@pytest.mark.unittest
class TestConnect4RuleBot():
    """
    Overview:
        This class is used to test the Connect4RuleBot class methods.
    """

    def setup(self):
        """
        Overview:
            This method is responsible for setting up the initial configurations required for the game environment.
            It creates an instance of the Connect4Env class and Connect4RuleBot class.
        """
        cfg = EasyDict(
            battle_mode='self_play_mode',
            mcts_mode='self_play_mode',
            channel_last=False,
            scale=True,
            agent_vs_human=False,
            prob_random_agent=0,
            prob_expert_agent=0,
            bot_action_type='rule',
            screen_scaling=9,
            save_replay=False,
            prob_random_action_in_bot = 0
        )
        self.env = Connect4Env(cfg)
        self.player = 1
        self.bot = Connect4RuleBot(self.env, self.player)

    def test_is_winning_move(self):
        """
        Overview:
            This test method creates a game situation where the bot has three consecutive pieces in the board.
            It tests the `is_winning_move` method of the Connect4RuleBot class by asserting that the method returns True
            when a winning move is possible for the bot.
        """
        # Create a chessboard with three consecutive pieces.
        board = np.zeros((6, 7))
        board[5][3] = self.player
        board[5][4] = self.player
        board[5][5] = self.player
        self.bot.board = board
        assert self.bot.is_winning_move(2) is True  # Winning move is to place a piece in the second column.

    def test_is_winning_move_in_two_steps(self):
        board = np.zeros((6, 7))
        board[5][3] = self.player
        board[5][4] = self.player
        self.bot.board = board
        assert self.bot.is_winning_move_in_two_steps(2) is True
        board = np.zeros((6, 7))
        board[5][3] = self.player
        board[5][4] = self.player
        board[5][0] = 3 - self.player
        board[4][0] = 3 - self.player
        board[3][0] = 3 - self.player
        self.bot.board = board
        assert self.bot.is_winning_move_in_two_steps(2) is False

    def test_is_blocking_move(self):
        """
        Overview:
            This test method creates a game situation where the opponent has three consecutive pieces in the board.
            It tests the `is_blocking_move` method of the Connect4RuleBot class by asserting that the method returns True
            when a blocking move is necessary to prevent the opponent from winning.
        """
        """
        # Create a chessboard with three consecutive pieces.
        board = np.zeros((6, 7))
        opponent = 2 if self.player == 1 else 1
        board[5][3] = opponent
        board[5][4] = opponent
        board[5][5] = opponent
        self.bot.board = board
        assert self.bot.is_blocking_move(2) is True  # Placing a piece in the second column is a move to prevent the opponent from winning.
        """

        # Create a chessboard with three consecutive pieces of opponents.
        self.bot.current_player = 2
        board = np.array([[1, 0, 0, 0, 0, 0, 0],
                          [1, 0, 1, 0, 0, 0, 0],
                          [2, 0, 2, 0, 0, 0, 0],
                          [1, 1, 1, 0, 0, 0, 0],
                          [1, 2, 2, 1, 0, 0, 2],
                          [1, 2, 2, 2, 1, 0, 0]])
        self.bot.board = board
        assert self.bot.is_blocking_move(3) is True  # Placing a piece in the 4th column is a move to prevent the opponent from winning.

    def test_is_sequence_3_move(self):
        """
        Overview:
            This test method creates a game situation where the bot has two consecutive pieces in the board.
            It tests the `is_sequence_3_move` method of the Connect4RuleBot class by asserting that the method returns True
            when placing a piece next to these two consecutive pieces will create a sequence of 3 pieces.
        """
        # Create a chessboard with two consecutive pieces.
        board = np.zeros((6, 7))
        board[5][4] = self.player
        board[5][5] = self.player
        self.bot.board = board
        assert self.bot.is_sequence_3_move(3) is True  # Placing a piece in the third column should create a three-in-a-row.

    def test_is_sequence_2_move(self):
        """
        Overview:
            This test method creates a game situation where the bot has a single piece in the board.
            It tests the `is_sequence_2_move` method of the Connect4RuleBot class by asserting that the method returns True
            when placing a piece next to the single piece will create a sequence of 2 pieces.
            It also tests for situations where placing a piece will not result in a sequence of 2 pieces.
        """
        # Create a chessboard with one consecutive piece.
        board = np.zeros((6, 7))
        board[5][5] = self.player
        self.bot.board = board
        assert self.bot.is_sequence_2_move(4) is True  # Placing a move in the fourth column should create a two-in-a-row.

        # Create a chessboard with one and two consecutive pieces.
        board = np.array([[0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 2, 0, 0],
                          [0, 0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 2, 2, 0, 0]])
        self.bot.board = board
        assert self.bot.is_sequence_2_move(5) is True  # Placing a move in the 5th column should create a two-in-a-row.
        assert self.bot.is_sequence_2_move(4) is False  # Placing a move in the 5th column should not create a two-in-a-row.
        assert self.bot.is_sequence_2_move(6) is False  # Placing a move in the 6th column should not create a two-in-a-row.

    def test_get_action(self):
        """
        Overview:
            This test method creates a game situation with an empty board.
            It tests the `get_rule_bot_action` method of the Connect4RuleBot class by asserting that the method returns an action
            that is within the set of legal actions.
        """
        board = np.zeros((6, 7))
        self.bot.board = board
        action = self.bot.get_rule_bot_action(board, self.player)
        assert action in self.env.legal_actions

    def test_remove_actions(self):
        self.bot.next_player = 3 - self.player
        board = np.zeros((6, 7))
        board[5][0] = self.player
        board[5][3] = self.player
        board[5][4] = self.player
        board[5][5] = 3 - self.player
        board[4][3] = 3 - self.player
        board[4][4] = 3 - self.player
        board[4][5] = 3 - self.player
        self.bot.board = board
        self.bot.legal_actions = [i for i in range(7) if board[0][i] == 0]
        self.bot.remove_actions()
        assert self.bot.legal_actions == [0, 1, 3, 4, 5]
        board = np.zeros((6, 7))
        board[5][0] = self.player
        board[4][0] = self.player
        board[5][3] = 3 - self.player
        board[5][4] = 3 - self.player
        self.bot.board = board
        self.bot.legal_actions = [i for i in range(7) if board[0][i] == 0]
        self.bot.remove_actions()
        assert self.bot.legal_actions == [0, 2, 5]
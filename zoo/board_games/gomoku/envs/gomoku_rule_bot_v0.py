import copy
from typing import Any, Tuple, List

import numpy as np


class GomokuRuleBotV0:
    """
    Overview:
        The rule-based bot for the Gomoku game. The bot follows a set of rules in a certain order until a valid move is found.\
        The rules are: winning move, blocking move, do not take a move which may lead to opponent win in 3 steps, \
        forming a sequence of 4, forming a sequence of 3, forming a sequence of 2, and a random move.
    """

    def __init__(self, env: Any, player: int, search_only_in_neighbor: bool = False) -> None:
        """
        Overview:
            Initializes the bot with the game environment and the player it represents.
        Arguments:
            - env (:obj:`Any`): The game environment, which contains the game state and allows interactions with it.
            - player (:obj:`int`): The player that the bot represents in the game.
        """
        self.env = env
        self.current_player = player
        self.players = self.env.players
        self.board_size = self.env.board_size
        self.dp = None
        self.search_only_in_neighbor = search_only_in_neighbor

    def get_neighbor_actions(self, board: np.ndarray) -> List[int]:
        """
        Overview:
            Get the legal actions in the neighborhood of existing pieces on the board.
        Arguments:
            - board (:obj:`np.ndarray`): The current game board.
        Returns:
            - neighbor_actions (:obj:`list` of :obj:`int`): The legal actions in the neighborhood of existing pieces.
        """
        neighbor_actions = set()
        for i in range(self.board_size):
            for j in range(self.board_size):
                # If there is a piece at (i, j)
                if board[i, j] != 0:
                    # Check the neighborhood
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            nx, ny = i + dx, j + dy
                            # If the neighbor coordinate is valid and there is no piece at (nx, ny)
                            if 0 <= nx < self.board_size and 0 <= ny < self.board_size and board[nx, ny] == 0:
                                neighbor_action = self.env.coord_to_action(nx, ny)
                                neighbor_actions.add(neighbor_action)
        return list(neighbor_actions)

    def get_rule_bot_action(self, board: np.ndarray, player: int) -> int:
        """
        Overview:
            Determines the next action of the bot based on the current game board and player.
        Arguments:
            - board (:obj:`np.ndarray`): The current game board.
            - player (:obj:`int`): The current player.
        Returns:
            - action (:obj:`int`): The next action of the bot.
        """
        if self.search_only_in_neighbor:
            # Get the legal actions in the neighborhood of existing pieces.
            self.legal_actions = self.get_neighbor_actions(board)
        else:
            self.legal_actions = self.env.legal_actions.copy()
        self.current_player = player
        self.next_player = self.players[0] if self.current_player == self.players[1] else self.players[1]
        self.board = np.array(copy.deepcopy(board)).reshape(self.board_size, self.board_size)
        # Initialize dp array if it's None
        if self.dp is None:
            self.dp = np.zeros((self.board_size, self.board_size, 8), dtype=int)
            self.update_dp(self.board)

        # Check if there is a winning move.
        for action in self.legal_actions:
            if self.is_winning_move(action):
                return action

        # Check if there is a move to block opponent's winning move.
        for action in self.legal_actions:
            if self.is_blocking_move(action):
                return action

        # Remove the actions which may lead to opponent to win.
        self.remove_actions()

        # If all the actions are removed, then randomly select an action.
        if len(self.legal_actions) == 0:
            return np.random.choice(self.env.legal_actions)

        # Check if there is a move to form a sequence of 4.
        for action in self.legal_actions:
            if self.is_sequence_X_move(action, 4):
                return action

        # Check if there is a move to form a sequence of 3.
        for action in self.legal_actions:
            if self.is_sequence_X_move(action, 3):
                return action

        # Check if there is a move to form a sequence of 2.
        for action in self.legal_actions:
            if self.is_sequence_X_move(action, 2):
                return action

        # Randomly select a legal move.
        return np.random.choice(self.legal_actions)

    def is_winning_move(self, action: int) -> bool:
        """
        Overview:
            Checks if an action is a winning move.
        Arguments:
            - action (:obj:`int`): The action to be checked.
        Returns:
            - result (:obj:`bool`): True if the action is a winning move; False otherwise.
        """
        piece = self.current_player
        temp_board, dp_backup = self._place_piece(action, piece)

        result = self.check_five_in_a_row(temp_board, piece)
        # Restore the dp array
        self.dp = dp_backup
        return result

    def is_blocking_move(self, action: int) -> bool:
        """
        Overview:
            Checks if an action can block the opponent's winning move.
        Arguments:
            - action (:obj:`int`): The action to be checked.
        Returns:
            - result (:obj:`bool`): True if the action can block the opponent's winning move; False otherwise.
        """
        piece = 2 if self.current_player == 1 else 1
        temp_board, dp_backup = self._place_piece(action, piece)

        result = self.check_five_in_a_row(temp_board, piece)
        # Restore the dp array
        self.dp = dp_backup
        return result

    def is_winning_move_in_two_steps(self, action: int) -> bool:
        """
        Overview:
            Checks if the specified action can lead to a win in two steps.
        Arguments:
            - action (:obj:`int`): The action to be checked.
        Returns:
            - result (:obj:`bool`): True if the action can lead to a win in two steps; False otherwise.
        """
        # Simulate the action
        piece = self.current_player
        # player_current_1step (assessing_action_now) -> player_opponent_1step -> player_current_2step -> player_opponent_2step
        #                                               -- action is here --
        temp_board, dp_backup = self._place_piece(action, piece)
        temp = [self.board.copy(), self.current_player]

        # Swap players
        self.board = temp_board
        self.current_player = 3 - self.current_player

        # Get legal actions
        legal_actions = [
            action
            for action in range(self.board_size * self.board_size)
            if self.board[self.env.action_to_coord(action)] == 0
        ]
        # player_current_1step (assessing_action_now) -> player_opponent_1step -> player_current_2step -> player_opponent_2step
        #                                                                        -- action is here --
        # Check if the player_current_2step has a winning move.
        if any(self.is_winning_move(action) for action in legal_actions):
            self.board, self.current_player = temp
            return False

        # player_current_1step (assessing_action_now)  -> player_opponent_1step -> player_current_2step -> player_opponent_2step
        #                                                                        -- action is here --
        # Count blocking moves. If player_current_2step has more than two blocking_move, which means that
        # if player_current take assessing_action_now, then the player_opponent_2step will have at least one wining move
        blocking_count = sum(self.is_blocking_move(action) for action in legal_actions)

        # Restore the original state
        self.board, self.current_player = temp

        # Check if there are more than one blocking moves
        return blocking_count >= 2

    def remove_actions(self) -> None:
        """
        Overview:
            Removes the actions from `self.legal_actions` that could potentially lead to the opponent's win.
        """
        temp_list = self.legal_actions.copy()
        for action in temp_list:
            temp = [self.board.copy(), self.current_player]

            piece = self.current_player
            action_x, action_y = self.env.action_to_coord(action)
            self.board[action_x][action_y] = piece

            self.current_player = self.next_player
            # Get legal actions
            legal_actions = [
                action
                for action in range(self.board_size * self.board_size)
                if self.board[self.env.action_to_coord(action)] == 0
            ]
            # print(f'if we take action {action}, then the legal actions for opponent are {legal_actions}')
            for a in legal_actions:
                if self.is_winning_move(a) or self.is_winning_move_in_two_steps(a):
                    self.legal_actions.remove(action)
                    # print(f"if take action {action}, then opponent take{a} may win")
                    # print(f"so we should remove action from {self.legal_actions}")
                    break

            self.board, self.current_player = temp

    def is_sequence_X_move(self, action: int, X: int) -> bool:
        """
        Overview:
            Checks if the specified action can form a sequence of 'X' pieces for the bot.
        Arguments:
            - action (:obj:`int`): The action to be checked.
            - X (:obj:`int`): The length of the sequence to be checked.
        Returns:
            - result (:obj:`bool`): True if the action can form a sequence of 'X' pieces; False otherwise.
        """
        piece = self.current_player

        temp_board, dp_backup = self._place_piece(action, piece)

        result = self.check_sequence_in_neighbor_board(temp_board, piece, X, action)
        # Restore the dp array
        self.dp = dp_backup
        return result

    def _place_piece(self, action: int, piece: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Overview:
            Places a piece on the board and updates the 'dp' array.
        Arguments:
            - action (:obj:`int`): The action indicating where to place the piece.
            - piece (:obj:`int`): The piece to be placed.
        Returns:
            - temp_board (:obj:`np.ndarray`): The updated game board.
            - dp_backup (:obj:`np.ndarray`): The backup of the 'dp' array before updating.
        """
        action_x, action_y = self.env.action_to_coord(action)
        temp_board = self.board.copy()
        temp_board[action_x][action_y] = piece

        # Backup the dp array
        dp_backup = copy.deepcopy(self.dp)
        # Update dp array
        self.update_dp(temp_board)

        return temp_board, dp_backup

    def check_sequence_in_neighbor_board(self, board: np.ndarray, piece: int, seq_len: int, action: int) -> bool:
        """
        Overview:
            Checks if a sequence of the bot's pieces of a given length can be formed in the neighborhood of a given action.
        Arguments:
            - board (:obj:`np.ndarray`): The current game board.
            - piece (:obj:`int`): The piece of the bot.
            - seq_len (:obj:`int`): The length of the sequence to be checked.
            - action (:obj:`int`): The action to be checked.
        Returns:
            - result (:obj:`bool`): True if the sequence of the bot's pieces can be formed; False otherwise.
        """
        # Convert action to coordinates
        row, col = self.env.action_to_coord(action)

        # Check horizontal locations
        for c in range(max(0, col - seq_len + 1), min(self.board_size - seq_len + 1, col + 1)):
            window = list(board[row, c:c + seq_len])
            if window.count(piece) == seq_len:
                return True

        # Check vertical locations
        for r in range(max(0, row - seq_len + 1), min(self.board_size - seq_len + 1, row + 1)):
            window = list(board[r:r + seq_len, col])
            if window.count(piece) == seq_len:
                return True

        # Check positively sloped diagonals
        for r in range(max(0, row - seq_len + 1), min(self.board_size - seq_len + 1, row + 1)):
            for c in range(max(0, col - seq_len + 1), min(self.board_size - seq_len + 1, col + 1)):
                if r - c == row - col:
                    window = [board[r + i][c + i] for i in range(seq_len)]
                    if window.count(piece) == seq_len:
                        return True

        # Check negatively sloped diagonals
        for r in range(max(0, row - seq_len + 1), min(self.board_size - seq_len + 1, row + 1)):
            for c in range(max(0, col - seq_len + 1), min(self.board_size - seq_len + 1, col + 1)):
                if r + c == row + col:
                    window = [board[r + i][c - i] for i in range(seq_len)]
                    if window.count(piece) == seq_len:
                        return True

        return False

    def update_dp(self, board: np.ndarray = None) -> None:
        """
        Overview:
            Updates the dynamic programming (dp) array based on the current game board.
        Arguments:
            - board (:obj:`np.ndarray`): The current game board. Defaults to None.
        """
        directions = [(0, 1), (1, 0), (-1, 1), (1, 1)]
        for i in range(self.board_size):
            for j in range(self.board_size):
                if board[i, j]:
                    for d, (dx, dy) in enumerate(directions):
                        nx, ny = i + dx, j + dy
                        if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                            self.dp[nx, ny, d] = (self.dp[i, j, d] + 1) if board[i, j] == board[nx, ny] else 0

    def check_five_in_a_row(self, board: np.ndarray, piece: int) -> bool:
        """
        Overview:
            Uses the dynamic programming (dp) array to check if there are five of the bot's pieces in a row.
        Arguments:
            - board (:obj:`np.ndarray`): The current game board.
            - piece (:obj:`int`): The piece of the bot.
        Returns:
            - result (:obj:`bool`): True if there are five of the bot's pieces in a row; False otherwise.
        """
        directions = [(0, 1), (1, 0), (-1, 1), (1, 1)]  # Four possible directions: right, down, down-left, down-right
        for i in range(self.board_size):
            for j in range(self.board_size):
                if board[i, j] == piece:  # If the piece at this location is the same as the one we're checking
                    for d, (dx, dy) in enumerate(directions):
                        # Check if the previous location is within the board's range
                        if 0 <= i - dx < self.board_size and 0 <= j - dy < self.board_size:
                            if self.dp[
                                i, j, d] + 1 >= 5:  # If there are at least 4 more pieces of the same type in this direction
                                return True  # We found five in a row

        return False  # If we checked every location and didn't find five in a row

    def check_five_in_a_row_naive(self, board: np.ndarray, piece: int) -> bool:
        """
        Overview:
            Checks if there are five of the bot's pieces in a row on the current game board.
        Arguments:
            - board (:obj:`np.ndarray`): The current game board.
            - piece (:obj:`int`): The piece of the bot.
        Returns:
            - result (:obj:`bool`): True if there are five of the bot's pieces in a row; False otherwise.
        """
        # Check horizontal and vertical locations
        for i in range(self.board_size):
            for j in range(self.board_size - 5 + 1):
                # Check horizontal
                if np.all(board[i, j:j + 5] == piece):
                    return True
                # Check vertical
                if np.all(board[j:j + 5, i] == piece):
                    return True

        # Check diagonals
        for i in range(self.board_size - 5 + 1):
            for j in range(self.board_size - 5 + 1):
                # Check positively sloped diagonals
                if np.all(board[range(i, i + 5), range(j, j + 5)] == piece):
                    return True
                # Check negatively sloped diagonals
                if np.all(board[range(i, i + 5), range(j + 5 - 1, j - 1, -1)] == piece):
                    return True

        return False

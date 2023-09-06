import copy

import numpy as np


class Connect4RuleBot():
    """
    Overview:
        The rule-based bot for the Connect4 game. The bot follows a set of rules in a certain order until a valid move is found.
        The rules are: winning move, blocking move, forming a sequence of 3, forming a sequence of 2, and a random move.
    """

    def __init__(self, env, player):
        """
        Initializes the bot with the game environment and the player it represents.

        Args:
            env: The game environment, which contains the game state and allows interactions with it.
            player: The player that the bot represents in the game.
        """
        self.env = env
        self.current_player = player
        self.players = self.env.players

    def get_rule_bot_action(self, board, player):
        """
          Determines the next action of the bot based on the current game board and player.

          Args:
              board: The current game board.
              player: The current player.

          Returns:
              The next action of the bot.
          """
        self.legal_actions = self.env.legal_actions
        # print(f"legal actions are {self.legal_actions}")
        self.current_player = player
        self.next_player = self.players[0] if self.current_player == self.players[1] else self.players[1]
        self.board = np.array(copy.deepcopy(board)).reshape(6, 7)

        # Check if there is a winning move
        for action in self.legal_actions:
            if self.is_winning_move(action):
                return action

        # Check if there is a move to block opponent's winning move
        for action in self.legal_actions:
            if self.is_blocking_move(action):
                return action
            
        # Remove the actions that may cause the opponent win
        temp_list = self.legal_actions.copy()
        for action in temp_list:
            temp = [self.board.copy(), self.current_player]
            # print("current board:", self.board)
            # print("current player:", self.current_player)
            piece = self.current_player
            row = self.get_available_row(action)
            # print(f"the row for {action} is {row}")
            if row is None:
                break
            self.board[row][action] = piece
            self.current_player = self.next_player
            legal_actions = [i for i in range(7) if self.board[0][i] == 0]
            # print(f'if we take action {action}, then the legal actions for opponent are {legal_actions}')
            for a in legal_actions:
                if self.is_winning_move(a):
                    self.legal_actions.remove(action)
                    print(f"if take action {action}, then opponent take{a} may win")
                    print(f"so we should take action from {self.legal_actions}")
                    break
            self.board, self.current_player = temp
        
        # If all the actions are removed, then randomly select an 
        if len(self.legal_actions) == 0:
            return np.random.choice(self.env.legal_actions)

        # Check if there is a move to form a sequence of 3
        for action in self.legal_actions:
            if self.is_sequence_3_move(action):
                return action

        # Check if there is a move to form a sequence of 2
        for action in self.legal_actions:
            if self.is_sequence_2_move(action):
                return action

        # Randomly select a legal move
        return np.random.choice(self.legal_actions)

    def is_winning_move(self, action):
        """
         Checks if an action is a winning move.

         Args:
             action: The action to be checked.

         Returns:
             True if the action is a winning move; False otherwise.
         """
        piece = self.current_player
        row = self.get_available_row(action)
        if row is None:
            return False
        temp_board = self.board.copy()
        temp_board[row][action] = piece
        return self.check_four_in_a_row(temp_board, piece)

    def is_blocking_move(self, action):
        """
        Checks if an action can block the opponent's winning move.

        Args:
            action: The action to be checked.

        Returns:
            True if the action can block the opponent's winning move; False otherwise.
        """
        piece = 2 if self.current_player == 1 else 1
        row = self.get_available_row(action)
        if row is None:
            return False
        temp_board = self.board.copy()
        temp_board[row][action] = piece
        return self.check_four_in_a_row(temp_board, piece)

    def is_sequence_3_move(self, action):
        """
        Checks if an action can form a sequence of 3 pieces of the bot.

        Args:
            action: The action to be checked.

        Returns:
            True if the action can form a sequence of 3 pieces of the bot; False otherwise.
        """
        piece = self.current_player
        row = self.get_available_row(action)
        if row is None:
            return False
        temp_board = self.board.copy()
        temp_board[row][action] = piece
        return self.check_sequence_in_neighbor_board(temp_board, piece, 3, action)

    def is_sequence_2_move(self, action):
        """
        Checks if an action can form a sequence of 2 pieces of the bot.

        Args:
            action: The action to be checked.

        Returns:
            True if the action can form a sequence of 2 pieces of the bot; False otherwise.
        """
        piece = self.current_player
        row = self.get_available_row(action)
        if row is None:
            return False
        temp_board = self.board.copy()
        temp_board[row][action] = piece
        return self.check_sequence_in_neighbor_board(temp_board, piece, 2, action)

    def get_available_row(self, col):
        """
        Gets the available row for a given column.

        Args:
            col: The column to be checked.

        Returns:
            The available row in the given column; None if the column is full.
        """
        for row in range(5, -1, -1):
            if self.board[row][col] == 0:
                return row
        return None

    def check_sequence_in_neighbor_board(self, board, piece, seq_len, action):
        """
        Checks if a sequence of the bot's pieces of a given length can be formed in the neighborhood of a given action.

        Args:
            board: The current game board.
            piece: The piece of the bot.
            seq_len: The length of the sequence.
            action: The action to be checked.

        Returns:
            True if such a sequence can be formed; False otherwise.
        """
        # Determine the row index where the piece fell
        row = self.get_available_row(action)

        # Check horizontal locations
        for c in range(max(0, action - seq_len + 1), min(7 - seq_len + 1, action + 1)):
            window = list(board[row, c:c + seq_len])
            if window.count(piece) == seq_len:
                return True

        # Check vertical locations
        for r in range(max(0, row - seq_len + 1), min(6 - seq_len + 1, row + 1)):
            window = list(board[r:r + seq_len, action])
            if window.count(piece) == seq_len:
                return True

        # Check positively sloped diagonals
        for r in range(6):
            for c in range(7):
                if r - c == row - action:
                    window = [board[r - i][c - i] for i in range(seq_len) if 0 <= r - i < 6 and 0 <= c - i < 7]
                    if len(window) == seq_len and window.count(piece) == seq_len:
                        return True

        # Check negatively sloped diagonals
        for r in range(6):
            for c in range(7):
                if r + c == row + action:
                    window = [board[r - i][c + i] for i in range(seq_len) if 0 <= r - i < 6 and 0 <= c + i < 7]
                    if len(window) == seq_len and window.count(piece) == seq_len:
                        return True


        return False

    def check_four_in_a_row(self, board, piece):
        """
        Checks if there are four of the bot's pieces in a row on the current game board.

        Args:
            board: The current game board.
            piece: The piece of the bot.

        Returns:
            True if there are four of the bot's pieces in a row; False otherwise.
        """
        # Check horizontal locations
        for col in range(4):
            for row in range(6):
                if board[row][col] == piece and board[row][col+1] == piece and board[row][col+2] == piece and board[row][col+3] == piece:
                    return True

        # Check vertical locations
        for col in range(7):
            for row in range(3):
                if board[row][col] == piece and board[row+1][col] == piece and board[row+2][col] == piece and board[row+3][col] == piece:
                    return True

        # Check positively sloped diagonals
        for row in range(3):
            for col in range(4):
                if board[row][col] == piece and board[row+1][col+1] == piece and board[row+2][col+2] == piece and board[row+3][col+3] == piece:
                    return True

        # Check negatively sloped diagonals
        for row in range(3, 6):
            for col in range(4):
                if board[row][col] == piece and board[row-1][col+1] == piece and board[row-2][col+2] == piece and board[row-3][col+3] == piece:
                    return True

        return False

    # not used now in this class
    def check_sequence_in_whole_board(self, board, piece, seq_len):
        """
         Checks if a sequence of the bot's pieces of a given length can be formed anywhere on the current game board.

         Args:
             board: The current game board.
             piece: The piece of the bot.
             seq_len: The length of the sequence.

         Returns:
             True if such a sequence can be formed; False otherwise.
         """
        # Check horizontal locations
        for row in range(6):
            row_array = list(board[row, :])
            for c in range(8 - seq_len):
                window = row_array[c:c + seq_len]
                if window.count(piece) == seq_len:
                    return True

        # Check vertical locations
        for col in range(7):
            col_array = list(board[:, col])
            for r in range(7 - seq_len):
                window = col_array[r:r + seq_len]
                if window.count(piece) == seq_len:
                    return True

        # Check positively sloped diagonals
        for row in range(6 - seq_len):
            for col in range(7 - seq_len):
                window = [board[row + i][col + i] for i in range(seq_len)]
                if window.count(piece) == seq_len:
                    return True

        # Check negatively sloped diagonals
        for row in range(seq_len - 1, 6):
            for col in range(7 - seq_len):
                window = [board[row - i][col + i] for i in range(seq_len)]
                if window.count(piece) == seq_len:
                    return True
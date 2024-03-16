# Reference link:

# https://github.com/LouisCaixuran/gomoku/blob/c1b6d508522d9e8c78be827f326bbee54c4dfd8b/gomoku/expert.py
"""
Sometimes, when GomokuRuleBotV1 has 4-connect, and the opponent also have 4-connect, GomokuRuleBotV1 will block the opponent and don't
play piece to 4-connect to 5-connect.
"""

from collections import defaultdict
import numpy as np


class GomokuRuleBotV1(object):
    """
        Overview:
            The ``GomokuExpert`` used to output rule-based expert actions for Gomoku.
            Input: board obs(:obj:`dict`) containing 'observation' and 'action_mask'.
            Returns: action (:obj:`Int`). The output action is the index number i*board_w+j corresponding to the placement position (i, j).
        Interfaces:
            ``__init__``, ``get_action``.
    """

    def __init__(self):
        """
        Overview:
            Init the ``GomokuRuleBotV1``.
        """
        # The initial unit weight of pieces
        self.unit_weight = 100
        self.init_board_flag = False

    def location_to_action(self, i, j):
        """
        Overview:
            Convert coordinate to serial action number.
        Arguments:
            - i (:obj:`Int`): X-axis.
            - j (:obj:`Int`): Y-axis.
        Returns:
            - action (:obj:`Int`): The serial action number of the entered coordinates on the pieceboard.
        Examples:
            - board_size = 6, (i,j)=(2,3) , action=3+2*6=15
        """
        # location = (i,j), action=j+i*width
        return j + i * self.board_width

    def action_to_location(self, action):
        """
        Overview:
            Convert serial action number to coordinate.
        Arguments:
            - action (:obj:`Int`): The serial number of the entered coordinates on the pieceboard.
        Returns:
            - [i, j].
        """
        # location = (i,j), action=j+i*width
        j = action % self.board_width
        i = action // self.board_width
        return [i, j]

    def get_loc_player(self, i, j):
        """
        Overview:
            Returns the state of the piece at the given coordinates.
        Arguments:
            - [i, j](:obj:`[Int, Int]`): The coordinate on the pieceboard.
        Returns:
            - board_status:
                0: no pieces,
                1: player 1,
                2: player 2.
        """
        action = self.location_to_action(i, j)
        return self.board_state[action]

    def scan_leftright(self, i, j, player):
        """
        Overview:
            Calculate the estimated score of the piece from left to right when the player moves at (i,j)
        Arguments:
            - i (:obj:`Int`): X-axis.
            - j (:obj:`Int`): Y-axis.
            - player (:obj:`Int`): Current player.
        Returns:
            - score: the evaluation score about the situation in this direction.
        """
        # Count the number of consecutive pieces of the current player or empty pieces:
        # and evaluate the score in this direction when moving pieces (i, j)
        score = 0
        count = 0
        unit_weight = self.unit_weight
        # scan left
        m, n = i, j - 1
        while n >= 0:
            is_continue, score, unit_weight = self.evaluate_one_move(m, n, player, score, unit_weight)
            if is_continue:
                # Continue to move one step to the left
                n = n - 1
            else:
                break
            count += 1
        # Change the direction to the right,
        # the unit_weight are reset to the initial unit_weight
        unit_weight = self.unit_weight
        # scan right
        n = j + 1
        while n < self.board_width:
            is_continue, score, unit_weight = self.evaluate_one_move(m, n, player, score, unit_weight)
            if is_continue:
                # Continue to move one step to the right
                n = n + 1
            else:
                break
            count += 1
        # Returns the score if there are four consecutive piece in this direction, otherwise 0
        return score if count >= 4 else 0

    def scan_updown(self, i, j, player):
        """
        Overview:
            Calculate the estimated score of the piece from up to down when the player moves at (i,j)
        Arguments:
            - i (:obj:`Int`): X-axis.
            - j (:obj:`Int`): Y-axis.
            - player (:obj:`Int`): Current player.

        Returns:
            - score: Situation valuation in this direction.
        """
        score = 0
        count = 0
        # Count the number of consecutive pieces or empty pieces of the current player
        # and get the score in this direction when moving pieces (i, j)
        unit_weight = self.unit_weight

        m, n = i - 1, j
        # scan up
        while m >= 0:
            is_continue, score, unit_weight = self.evaluate_one_move(m, n, player, score, unit_weight)
            if is_continue:
                # Continue to move one step to the up
                m = m - 1
            else:
                break
            count += 1
        # Change the direction and change the weight back to the initial score
        unit_weight = self.unit_weight
        m = i + 1
        # scan down
        while m < self.board_height:
            is_continue, score, unit_weight = self.evaluate_one_move(m, n, player, score, unit_weight)
            if is_continue:
                # Continue to move one step to the down
                m = m + 1
            else:
                break
            count += 1
        # Returns the score if there are four consecutive piece in this direction, otherwise 0
        return score if count >= 4 else 0

    def scan_left_updown(self, i, j, player):
        """
        Overview:
            Calculate the estimated score of the piece from top left to bottom right when the player moves at (i,j)
        Arguments:
            - i (:obj:`Int`): X-axis.
            - j (:obj:`Int`): Y-axis.
            - player (:obj:`Int`): Current player.
        Returns:
            - score: Situation valuation in this direction.
        """
        # Count the number of consecutive pieces or empty pieces of the current player
        # and get the score in this direction when moving pieces (i, j)
        score = 0
        count = 0

        unit_weight = self.unit_weight
        m, n = i - 1, j - 1
        # scan left_up
        while m >= 0 and n >= 0:
            is_continue, score, unit_weight = self.evaluate_one_move(m, n, player, score, unit_weight)
            if is_continue:
                # Continue to move one step to the left up
                m, n = m - 1, n - 1
            else:
                break
            count += 1

        unit_weight = self.unit_weight
        # scan right_down
        m, n = i + 1, j + 1
        while m < self.board_height and n < self.board_width:
            is_continue, score, unit_weight = self.evaluate_one_move(m, n, player, score, unit_weight)
            if is_continue:
                # Continue to move one step down to the right
                m, n = m + 1, n + 1
            else:
                break
            count += 1
        # Returns the score if there are four consecutive piece in this direction, otherwise 0
        return score if count >= 4 else 0

    def scan_right_updown(self, i, j, player):
        """
        Overview:
            Calculate the estimated score of the piece from top right to bottom left when the player moves at (i,j)
        Arguments:
            - i (:obj:`Int`): X-axis.
            - j (:obj:`Int`): Y-axis.
            - player (:obj:`Int`): Current player.
        Returns:
            - score: Situation valuation in this direction.
        """
        # Count the number of consecutive pieces or empty pieces of the current player
        # and get the score in this direction when moving pieces (i, j)
        score = 0
        count = 0
        unit_weight = self.unit_weight
        # scan left_down
        m, n = i + 1, j - 1
        while m < self.board_height and n >= 0:
            is_continue, score, unit_weight = self.evaluate_one_move(m, n, player, score, unit_weight)
            if is_continue:
                m, n = m + 1, n - 1
            else:
                break
            count += 1
        unit_weight = self.unit_weight
        # scan right_up
        m, n = i - 1, j + 1
        while m >= 0 and n < self.board_width:
            is_continue, score, unit_weight = self.evaluate_one_move(m, n, player, score, unit_weight)
            if is_continue:
                # Continue to move up one step to the right
                m, n = m - 1, n + 1
            else:
                break
            count += 1
        # Returns the score if there are four consecutive piece in this direction, otherwise 0
        return score if count >= 4 else 0

    def evaluate_one_move(self, m, n, player, score, unit_weight):
        """
        Overview:
            Calculate the income brought by the pieces adjacent to the position (m,n) \
            when the player places the piece at the specified position (i,j) \
            in the current situation
        Arguments:
            - m (:obj:`Int`): x.
            - n (:obj:`Int`): y.
            - player (:obj:`Int`): current piece player.
            - score (:obj:`Int`): The current position (the piece is at (i,j)) is evaluated.
            - unit_weight (:obj:`Int`): The weight of the piece in the current position

        Returns:
            - is_continue: Whether there is a piece of current_player at the current position
            - score: The evaluation score of the move to (i,j)
            - unit_weight: The weight of a single one of our piece pieces
        """
        loc_player = self.get_loc_player(m, n)
        if loc_player == player:
            # When encountering an current_player's piece, add unit_weight to the score
            score += unit_weight
        elif loc_player == 0:
            # When encountering an empty piece, add 1 to the score
            score += 1
            # When encountering an empty piece, reduce the unit_weight of subsequent piece
            unit_weight = unit_weight / 10
        else:
            # When encountering an opponent_player's piece, minus 5 to the score
            score -= 5
            # score -= 1  # TODO
            # When encountering an opponent_player's piece, return
            # is_continue = 0
            return 0, score, unit_weight
        # is_continue = 1
        return 1, score, unit_weight

    def evaluate_all_legal_moves(self, player):
        """
        Overview:
            Calculate the scores of all legal moves and choose the most favorable move from them.
        Arguments:
            - player (:obj:`Int`): current player.
        Returns:
            - action: the most favorable action
            - self.action_score[action][4]: the evaluation score related to the situation under this action
        """
        self.action_score = defaultdict(lambda: [0, 0, 0, 0, 0])
        for action in self.legal_actions:
            i, j = self.action_to_location(action)

            self.action_score[action][0] = self.scan_updown(i, j, player)
            self.action_score[action][1] = self.scan_leftright(i, j, player)
            self.action_score[action][2] = self.scan_left_updown(i, j, player)
            self.action_score[action][3] = self.scan_right_updown(i, j, player)

            # Indicates that one direction can already be rushed to 4
            # TODO(pu): the meaning of the special number
            for k in range(4):
                if self.action_score[action][k] >= 390:
                    self.action_score[action][k] = 2000
                elif self.action_score[action][k] >= 302:
                    self.action_score[action][k] = 1000

            # ==============================================================
            # <302: 原值
            # 302<= x <= 390: 1000
            # x >= 390: 2000
            # ==============================================================

            # Combining the scores of each direction into a total action score
            self.action_score[action][4] = (
                self.action_score[action][0] + self.action_score[action][1] + self.action_score[action][2] +
                self.action_score[action][3]
            )

        action = max(self.legal_actions, key=lambda x: self.action_score[x][4])

        return action, self.action_score[action][4]

    def get_action(self, obs):
        """
        Overview:
            Given the Gomoku obs, returns a rule-based expert action.
        Arguments:
            - obs (:obj:`np.array`)

        Returns:
            - bot_action
        """
        self.obs = obs

        if self.obs['observation'].shape[0] == self.obs['observation'].shape[1]:
            # the following reshape is wrong implementation
            # self.obs['observation'] = self.obs['observation'].reshape(
            #     3, self.obs['observation'].shape[0], self.obs['observation'].shape[1]
            # )

            # shape: 6,6,3 -> 3,6,6
            self.obs['observation'] = self.obs['observation'].transpose(2, 0, 1)

        if self.init_board_flag is False:
            # obtain the board_width and board_height from the self.obs['observation']
            self.board_width = self.obs['observation'][0].shape[0]
            self.board_height = self.obs['observation'][0].shape[1]
            self.init_board_flag = True
        if self.obs['observation'][2][0][0] == 1:
            # the 2th dim of self.obs['observation'] indicates which player is the to_play player,
            # 1 means player 1, 2 means player 2
            self.current_player_id = 1
            self.opponent_player_id = 2
        else:
            self.current_player_id = 2
            self.opponent_player_id = 1
        # transform observation, action_mask to self.legal_actions, self.board_state

        self.legal_actions = []
        self.board_state = np.zeros(self.board_width * self.board_height, 'int8')
        for i in range(self.board_width):
            for j in range(self.board_height):
                action = self.location_to_action(i, j)
                if self.obs['action_mask'][action] == 1:
                    self.legal_actions.append(action)
                if self.obs['observation'][0][i][j] == 1:
                    self.board_state[action] = self.current_player_id
                elif self.obs['observation'][1][i][j] == 1:
                    self.board_state[action] = self.opponent_player_id

        current_best_action, current_score = self.evaluate_all_legal_moves(self.current_player_id)
        # logging.info("location:{loc},score:{score}".format(loc=self.action_to_location(current_best_action), score=current_score))

        opponent_best_action, opponent_score = self.evaluate_all_legal_moves(self.opponent_player_id)
        # logging.info("O_location:{loc},score:{score}".format(loc=self.action_to_location(opponent_best_action), score=opponent_score))

        if current_score >= opponent_score:
            # curent player should play current_best_action if the score that the current_player obtain when playing current_best_action
            # is larger than the score that the opponent_player obtains when it playing opponent_best_action
            return current_best_action
        else:
            # curent player should play (Block) this opponent_best_action position if current_score < opponent_score
            return opponent_best_action

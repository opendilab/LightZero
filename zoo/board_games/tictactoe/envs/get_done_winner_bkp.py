def get_done_winner(self):
    """
    Overview:
         Check if the game is over and who the winner is. Return 'done' and 'winner'.
    Returns:
        - outputs (:obj:`Tuple`): Tuple containing 'done' and 'winner',
            - if player 1 win,     'done' = True, 'winner' = 1
            - if player 2 win,     'done' = True, 'winner' = 2
            - if draw,             'done' = True, 'winner' = -1
            - if game is not over, 'done' = False, 'winner' = -1
    """
    have_winner, winner = self._get_have_winner_and_winner()
    if have_winner:
        done, winner = True, winner
    elif len(self.legal_actions) == 0:
        # the agent don't have legal_actions to move, so the episode is done
        # winner = -1 indicates draw.
        done, winner = True, -1
    else:
        # episode is not done.
        done, winner = False, -1

    return done, winner


def _get_have_winner_and_winner(self):
    # has_legal_actions i.e. not done
    # Horizontal and vertical checks
    for i in range(self.board_size):
        if len(set(self.board[i, :])) == 1 and (self.board[i, 0] != 0):
            return True, self.board[i, 0]
        if len(set(self.board[:, i])) == 1 and (self.board[0, i] != 0):
            return True, self.board[0, i]

    # Diagonal checks
    if self.board[0, 0] == self.board[1, 1] == self.board[2, 2] != 0:
        return True, self.board[0, 0]
    if self.board[2, 0] == self.board[1, 1] == self.board[0, 2] != 0:
        return True, self.board[2, 0]

    winner = -1
    have_winner = False
    return have_winner, winner
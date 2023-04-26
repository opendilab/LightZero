def get_done_winner(board_size, board):
    """
    Overview:
         Check if the game is over and who the winner is. Return 'done' and 'winner'.
    Arguments:
        - board_size (:obj:`int`): The size of the board.
        - board (:obj:`numpy.ndarray`): The board state.
    Returns:
        - outputs (:obj:`Tuple`): Tuple containing 'done' and 'winner',
            - if player 1 win,     'done' = True, 'winner' = 1
            - if player 2 win,     'done' = True, 'winner' = 2
            - if draw,             'done' = True, 'winner' = -1
            - if game is not over, 'done' = False, 'winner' = -1
    """
    # has_legal_actions i.e. not done
    has_legal_actions = False
    directions = ((1, -1), (1, 0), (1, 1), (0, 1))
    for i in range(board_size):
        for j in range(board_size):
            # if no stone is on the position, don't need to consider this position
            if board[i][j] == 0:
                has_legal_actions = True
                continue
            # value-value at a coord, i-row, j-col
            player = board[i][j]
            # check if there exist 5 in a line
            for d in directions:
                x, y = i, j
                count = 0
                for _ in range(5):
                    if (x not in range(board_size)) or (y not in range(board_size)):
                        break
                    if board[x][y] != player:
                        break
                    x += d[0]
                    y += d[1]
                    count += 1
                    # if 5 in a line, store positions of all stones, return value
                    if count == 5:
                        return True, player
    # if the players don't have legal actions, return done=True
    return not has_legal_actions, -1
from libc.stdint cimport int32_t
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef get_done_winner_cython(int32_t board_size, int32_t[:, :] board):
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
    cdef int32_t i, j, player, x, y, count
    cdef bint has_legal_actions = False
    cdef directions = ((1, -1), (1, 0), (1, 1), (0, 1))


    for i in range(board_size):
        for j in range(board_size):
            if board[i, j] == 0:
                has_legal_actions = True
                continue
            player = board[i, j]
            for d in directions:
                x, y = i, j
                count = 0
                for _ in range(5):
                    if x < 0 or x >= board_size or y < 0 or y >= board_size:
                        break
                    if board[x, y] != player:
                        break
                    x += d[0]
                    y += d[1]
                    count += 1
                    if count == 5:
                        return True, player
    return not has_legal_actions, -1
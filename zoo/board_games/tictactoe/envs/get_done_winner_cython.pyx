from libc.stdint cimport int32_t
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef get_done_winner_cython(int32_t[:, :] board):
    """
    Overview:
         Check if the tictactoe game is over and who the winner is. Return 'done' and 'winner'.
    Arguments:
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
    # Check for a winning condition in all 4 directions: diagonal left, horizontal, diagonal right, and vertical
    cdef directions = ((1, -1), (1, 0), (1, 1), (0, 1))

    # iterate through all positions in the board
    for i in range(3):
        for j in range(3):
            # If the position is empty, there are still legal actions available
            if board[i, j] == 0:
                has_legal_actions = True
                continue
            # Store the player number (1 or 2) of the current position
            player = board[i, j]

            # Check for a winning condition in all 4 directions:
            # diagonal left (1,-1), horizontal (1,0), diagonal right (1,1), and vertical (0,1)

            # Determine which directions to check based on the current position
            start_dir_idx = 0 if j > 0 else 1
            end_dir_idx = 4 if j < 3 - 1 else 3
            for d_idx in range(start_dir_idx, end_dir_idx):
                d = directions[d_idx]

                x, y = i, j
                count = 0

                # Check for 3 consecutive positions with the same player number
                for _ in range(3):
                    # If the current position is out of the board's boundaries, break the loop
                    if x < 0 or x >= 3 or y < 0 or y >= 3:
                        break
                    # If the current position doesn't have the same player number, break the loop
                    if board[x, y] != player:
                        break
                    # Move to the next position in the direction and increment the count
                    x += d[0]
                    y += d[1]
                    count += 1

                    # If 3 consecutive positions with the same player number are found, return 'done' as True and the 'winner' as the player number
                    if count == 3:
                        return True, player

    # If no legal actions are left, return 'done' as True and 'winner' as -1 (draw)
    return not has_legal_actions, -1
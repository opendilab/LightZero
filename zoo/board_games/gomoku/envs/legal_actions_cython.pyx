from libc.stdint cimport int32_t
import cython

@cython.boundscheck(False)  # Disable bounds checking for performance
@cython.wraparound(False)   # Disable negative indexing for performance
def legal_actions_cython(int board_size, int32_t[:, :] board):
    # Use a Python list to store possible legal actions
    cdef list legal_actions = []
    cdef int i, j

    # Iterate over each position on the board
    for i in range(board_size):
        for j in range(board_size):
            # If the current position is empty (value is 0), it is a legal action
            if board[i, j] == 0:
                # Add the legal action to the list, representing it as an integer
                legal_actions.append(i * board_size + j)

    # Return the Python list containing all legal actions
    return legal_actions
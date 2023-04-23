# cythonize -i ./legal_actions_cython_v2.pyx

def legal_actions_cython_v2(int board_size, list board):
    cdef list legal_actions = []
    cdef int pos
    cdef int i, j
    for i in range(board_size):
        for j in range(board_size):
            if board[i][j] == 0:
                pos = i * board_size + j
                legal_actions.append(pos)
    return legal_actions

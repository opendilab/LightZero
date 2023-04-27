# cythonize -i ./legal_actions_cython.pyx

def legal_actions_cython_v1(int board_size, list board):
    cdef list legal_actions = []
    for i in range(board_size):
        for j in range(board_size):
            if board[i][j] == 0:
                legal_actions.append(i * board_size + j)
    return legal_actions
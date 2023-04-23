# cythonize -i ./legal_actions_cython.pyx

def legal_actions_cython(list board):
    cdef list legal_actions = []
    cdef int i, j
    for i in range(3):
        for j in range(3):
            if board[i][j] == 0:
                legal_actions.append(i * 3 + j)
    return legal_actions

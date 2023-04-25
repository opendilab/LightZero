# cythonize -i ./legal_actions_cython_str.pyx


cpdef list legal_actions_cython_str(str board_str):
    cdef list legal_actions = []
    cdef int idx
    for idx, cell in enumerate(board_str):
        if cell == '0':
            legal_actions.append(idx)
    return legal_actions

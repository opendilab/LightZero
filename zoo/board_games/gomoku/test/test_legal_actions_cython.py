import numpy as np
from zoo.board_games.gomoku.envs.legal_actions_cython import legal_actions_cython


def _legal_actions_func(board_size, board_tuple):
    # Convert tuple to NumPy array.
    board_array = np.array(board_tuple, dtype=np.int32)
    # Convert NumPy array to memory view.
    board_view = board_array.view(dtype=np.int32).reshape(board_array.shape)
    return legal_actions_cython(board_size, board_view)


def test_legal_actions_cython():
    # case 1
    board_size = 2
    board = [[0, 0], [0, 0]]

    legal_actions = _legal_actions_func(board_size, tuple(map(tuple, board)))
    assert legal_actions == [0, 1, 2, 3], f"Error: {legal_actions}"

    # case 2
    board_size = 3
    board = [[0, 0, 0], [0, 1, 2], [0, 2, 1]]
    legal_actions = _legal_actions_func(board_size, tuple(map(tuple, board)))

    assert legal_actions == [0, 1, 2, 3, 6], f"Error: {legal_actions}"

    # case 3
    board_size = 4
    board = [[1, 1, 1, 0], [2, 2, 1, 0], [2, 2, 1, 0], [2, 1, 2, 0]]
    legal_actions = _legal_actions_func(board_size, tuple(map(tuple, board)))
    assert legal_actions == [3, 7, 11, 15], f"Error: {legal_actions}"

    # case 4
    board_size = 5
    board = [[1, 1, 1, 1, 0], [2, 2, 1, 1, 0], [2, 2, 1, 2, 0], [2, 1, 2, 2, 0], [2, 1, 1, 2, 0]]
    legal_actions = _legal_actions_func(board_size, tuple(map(tuple, board)))
    assert legal_actions == [4, 9, 14, 19, 24], f"Error: {legal_actions}"

    # case 5
    board_size = 6
    board = [[1, 1, 1, 1, 0, 1], [2, 2, 1, 1, 0, 2], [2, 2, 1, 2, 0, 1], [2, 1, 2, 2, 0, 2], [2, 1, 1, 2, 0, 1],  [1, 2, 1, 2, 0, 2]]
    legal_actions = _legal_actions_func(board_size, tuple(map(tuple, board)))
    assert legal_actions == [4, 10, 16, 22, 28, 34], f"Error: {legal_actions}"


test_legal_actions_cython()

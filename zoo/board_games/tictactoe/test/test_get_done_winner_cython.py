import numpy as np
from zoo.board_games.tictactoe.envs.get_done_winner_cython import get_done_winner_cython


def _get_done_winner_func(board_tuple):
    # Convert tuple to NumPy array.
    board_array = np.array(board_tuple, dtype=np.int32)
    # Convert NumPy array to memory view.
    board_view = board_array.view(dtype=np.int32).reshape(board_array.shape)
    return get_done_winner_cython(board_view)


def test_get_done_winner_cython():
    # case 1
    board = [[0, 0, 0], [0, 1, 2], [0, 2, 1]]
    done_winner = _get_done_winner_func(tuple(map(tuple, board)))

    assert done_winner == (False, -1), f"Error: {done_winner}"

    # case 2
    board = [[1, 1, 2], [2, 2, 1], [1, 2, 1]]
    done_winner = _get_done_winner_func(tuple(map(tuple, board)))
    assert done_winner == (True, -1), f"Error: {done_winner}"

    # case 3
    board = [[1, 1, 1], [2, 2, 1], [2, 2, 1]]
    done_winner = _get_done_winner_func(tuple(map(tuple, board)))
    assert done_winner == (True, 1), f"Error: {done_winner}"

    # case 4
    board = [[1, 2, 1], [0, 2, 0], [1, 2, 0]]
    done_winner = _get_done_winner_func(tuple(map(tuple, board)))
    assert done_winner == (True, 2), f"Error: {done_winner}"


test_get_done_winner_cython()

"""
Overview:
    Efficiency comparison of different vectorization methods based on legal_actions_function `legal_actions`:
    NOTE: The time may vary on different devices and software versions.
    =======================================
    ### execute legal_actions 1000,000 times###
    ---------------------------------------
    | Methods                      | Seconds
    ---------------------------------------
    | legal_actions_forloop        | 30.645
    | legal_actions_np             | 72.559
    | legal_actions_cython         | 36.111
    | legal_actions_cython_lru  | 8.123
"""

import numpy as np
from ding.utils import EasyTimer
from zoo.board_games.gomoku.envs.legal_actions_cython import legal_actions_cython
from functools import lru_cache


def _legal_actions_cython_func(board_size, board_tuple):
    # Convert tuple to NumPy array.
    board_array = np.array(board_tuple, dtype=np.int32)
    # Convert NumPy array to memory view.
    board_view = board_array.view(dtype=np.int32).reshape(board_array.shape)
    return legal_actions_cython(board_size, board_view)


@lru_cache(maxsize=512)
def _legal_actions_cython_lru_func(board_size, board_tuple):
    # Convert tuple to NumPy array.
    board_array = np.array(board_tuple, dtype=np.int32)
    # Convert NumPy array to memory view.
    board_view = board_array.view(dtype=np.int32).reshape(board_array.shape)
    return legal_actions_cython(board_size, board_view)


timer = EasyTimer(cuda=True)


def legal_actions_forloop(board_size, board):
    legal_actions = []
    for i in range(board_size):
        for j in range(board_size):
            if board[i][j] == 0:
                legal_actions.append(i * board_size + j)
    return legal_actions


def legal_actions_np(board_size, board):
    zero_positions = np.argwhere(board == 0)
    legal_actions = [i * board_size + j for i, j in zero_positions]
    return legal_actions


def eval_legal_actions_template(legal_actions_func):
    # case 1
    board_size = 5
    board = [[1, 2, 1, 2, 1], [1, 2, 1, 2, 1], [2, 1, 2, 1, 2], [2, 1, 2, 1, 2], [1, 2, 1, 2, 1]]
    if legal_actions_func in [_legal_actions_cython_func, _legal_actions_cython_lru_func]:
        board = tuple(map(tuple, board))
    legal_actions = legal_actions_func(board_size, board)

    # case 2
    board_size = 5
    board = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    if legal_actions_func in [_legal_actions_cython_func, _legal_actions_cython_lru_func]:
        board = tuple(map(tuple, board))
    legal_actions = legal_actions_func(board_size, board)

    # case 3
    board_size = 5
    board = [[0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0]]
    if legal_actions_func in [_legal_actions_cython_func, _legal_actions_cython_lru_func]:
        board = tuple(map(tuple, board))
    legal_actions = legal_actions_func(board_size, board)

    # case 4
    board_size = 5
    board = [[0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0]]
    if legal_actions_func in [_legal_actions_cython_func, _legal_actions_cython_lru_func]:
        board = tuple(map(tuple, board))
    legal_actions = legal_actions_func(board_size, board)

    # case 5
    board_size = 5
    board = [[1, 1, 0, 1, 1], [1, 1, 0, 1, 1], [1, 1, 0, 1, 1], [1, 1, 0, 1, 1], [1, 1, 0, 1, 1]]
    if legal_actions_func in [_legal_actions_cython_func, _legal_actions_cython_lru_func]:
        board = tuple(map(tuple, board))
    legal_actions = legal_actions_func(board_size, board)


def eval_legal_actions_forloop():
    eval_legal_actions_template(legal_actions_forloop)


def eval_legal_actions_np():
    eval_legal_actions_template(legal_actions_np)


def eval_legal_actions_cython():
    eval_legal_actions_template(_legal_actions_cython_func)


def eval_legal_actions_cython_lru():
    eval_legal_actions_template(_legal_actions_cython_lru_func)


if __name__ == "__main__":
    eval_times = 1000

    print(f"##### execute eval_legal_actions {eval_times} times #####")

    with timer:
        for _ in range(eval_times):
            eval_legal_actions_forloop()
    print(f"---------------------------------------")
    print(f"| legal_actions_forloop | {timer.value:.3f} |")
    print(f"---------------------------------------")
    with timer:
        for _ in range(eval_times):
            eval_legal_actions_np()
    print(f"---------------------------------------")
    print(f"| legal_actions_np      | {timer.value:.3f} |")
    print(f"---------------------------------------")

    with timer:
        for _ in range(eval_times):
            eval_legal_actions_cython()
    print(f"---------------------------------------")
    print(f"| legal_actions_cython  | {timer.value:.3f} |")
    print(f"---------------------------------------")
    with timer:
        for _ in range(eval_times):
            eval_legal_actions_cython_lru()
    print(f"---------------------------------------")
    print(f"| legal_actions_cython_lru  | {timer.value:.3f} |")
    print(f"---------------------------------------")

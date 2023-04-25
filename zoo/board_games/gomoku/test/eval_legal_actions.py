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
    | legal_actions_cython_v1      | 21.828
    | legal_actions_cython_v2      | 4.400
    | legal_actions_cython_v3      | 4.333
    | legal_actions_cython_v3_lru  | 0.011
"""

import numpy as np
from ding.utils import EasyTimer
from zoo.board_games.gomoku.test.legal_actions_cython_v1 import legal_actions_cython_v1
from zoo.board_games.gomoku.test.legal_actions_cython_v2 import legal_actions_cython_v2
from zoo.board_games.gomoku.test.legal_actions_cython_v3 import legal_actions_cython_v3

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
    board = [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]
    legal_actions = legal_actions_func(board_size, board)

    # case 2
    board_size = 5
    board = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    legal_actions = legal_actions_func(board_size, board)

    # case 3
    board_size = 5
    board = [[0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0]]
    legal_actions = legal_actions_func(board_size, board)

    # case 4
    board_size = 5
    board = [[0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0]]
    legal_actions = legal_actions_func(board_size, board)

    # case 4
    board_size = 5
    board = [[1, 1, 0, 1, 1], [1, 1, 0, 1, 1], [1, 1, 0, 1, 1], [1, 1, 0, 1, 1], [1, 1, 0, 1, 1]]
    legal_actions = legal_actions_func(board_size, board)


def eval_legal_actions_forloop():
    eval_legal_actions_template(legal_actions_forloop)


def eval_legal_actions_np():
    eval_legal_actions_template(legal_actions_np)


def eval_legal_actions_cython_v1():
    eval_legal_actions_template(legal_actions_cython_v1)


def eval_legal_actions_cython_v2():
    eval_legal_actions_template(legal_actions_cython_v2)


def eval_legal_actions_cython_v3():
    eval_legal_actions_template(legal_actions_cython_v3)


from functools import lru_cache


@lru_cache(maxsize=128)
def eval_legal_actions_cython_v3_lru():
    eval_legal_actions_template(legal_actions_cython_v3)


if __name__ == "__main__":
    eval_times = 1000000

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
            eval_legal_actions_cython_v1()
    print(f"---------------------------------------")
    print(f"| legal_actions_cython_v1  | {timer.value:.3f} |")
    print(f"---------------------------------------")
    with timer:
        for _ in range(eval_times):
            eval_legal_actions_cython_v2()
    print(f"---------------------------------------")
    print(f"| legal_actions_cython_v2  | {timer.value:.3f} |")
    print(f"---------------------------------------")
    with timer:
        for _ in range(eval_times):
            eval_legal_actions_cython_v3()
    print(f"---------------------------------------")
    print(f"| legal_actions_cython_v3  | {timer.value:.3f} |")
    print(f"---------------------------------------")
    with timer:
        for _ in range(eval_times):
            eval_legal_actions_cython_v3_lru()
    print(f"---------------------------------------")
    print(f"| legal_actions_cython_v3_lru  | {timer.value:.3f} |")
    print(f"---------------------------------------")

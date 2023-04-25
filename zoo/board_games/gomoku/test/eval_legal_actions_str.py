"""
Overview:
    Efficiency comparison of different vectorization methods based on legal_actions_function `legal_actions`:
    NOTE: The time may vary on different devices and software versions.
    =======================================
    ### execute legal_actions 1000,000 times###
    ---------------------------------------
    | Methods                      | Seconds
    ---------------------------------------
    | legal_actions_forloop        | 30.600
    | legal_actions_forloop_str    | 71.559
    | legal_actions_enumerate_str  | 4.000
    | legal_actions_cython_v3      | 5.533
    | legal_actions_cython_v3_lru  | 0.100
    | legal_actions_cython_str     | 5.633
    | legal_actions_cython_str_lru | 0.100

"""

import numpy as np
from ding.utils import EasyTimer
from zoo.board_games.gomoku.test.legal_actions_cython_v1 import legal_actions_cython_v1
from zoo.board_games.gomoku.test.legal_actions_cython_v2 import legal_actions_cython_v2
from zoo.board_games.gomoku.test.legal_actions_cython_v3 import legal_actions_cython_v3
from zoo.board_games.gomoku.test.legal_actions_cython_str import legal_actions_cython_str

timer = EasyTimer(cuda=True)


def eval_legal_actions_template_(legal_actions_func, board_size, board):
    if legal_actions_func.__name__ == 'legal_actions_forloop':
        legal_actions = legal_actions_func(board_size, board)
    elif legal_actions_func.__name__ == 'legal_actions_forloop_str':
        board_str = ''.join(str(cell) for row in board for cell in row)
        legal_actions = legal_actions_func(board_size, board_str)
    elif legal_actions_func.__name__ == ' legal_actions_enumerate_str':
        board_str = ''.join(str(cell) for row in board for cell in row)
        legal_actions = legal_actions_func(board_str)
    elif legal_actions_func.__name__ == ' legal_actions_cython_v3':
        legal_actions = legal_actions_func(board_size, board)
    elif legal_actions_func.__name__ == ' legal_actions_cython_str':
        board_str = ''.join(str(cell) for row in board for cell in row)
        legal_actions = legal_actions_func(board_str)


def eval_legal_actions_template(legal_actions_func):
    # case 1
    board_size = 5
    board = [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]
    eval_legal_actions_template_(legal_actions_func, board_size, board)

    # case 2
    board_size = 5
    board = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    eval_legal_actions_template_(legal_actions_func, board_size, board)

    # case 3
    board_size = 5
    board = [[0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0]]
    eval_legal_actions_template_(legal_actions_func, board_size, board)

    # case 4
    board_size = 5
    board = [[0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0]]
    eval_legal_actions_template_(legal_actions_func, board_size, board)

    # case 4
    board_size = 5
    board = [[1, 1, 0, 1, 1], [1, 1, 0, 1, 1], [1, 1, 0, 1, 1], [1, 1, 0, 1, 1], [1, 1, 0, 1, 1]]
    eval_legal_actions_template_(legal_actions_func, board_size, board)


def legal_actions_forloop(board_size, board):
    legal_actions = []
    for i in range(board_size):
        for j in range(board_size):
            if board[i][j] == 0:
                legal_actions.append(i * board_size + j)
    return legal_actions


def legal_actions_forloop_str(board_size, board_str: str):
    legal_actions = []
    for i in range(board_size):
        for j in range(board_size):
            pos = i * board_size + j
            if board_str[pos] == '0':
                legal_actions.append(int(pos))
    return legal_actions


def legal_actions_enumerate_str(board_str: str):
    return [idx for idx, cell in enumerate(board_str) if cell == '0']


def eval_legal_actions_forloop():
    eval_legal_actions_template(legal_actions_forloop)


def eval_legal_actions_forloop_str():
    eval_legal_actions_template(legal_actions_forloop_str)


def eval_legal_actions_enumerate_str():
    eval_legal_actions_template(legal_actions_enumerate_str)


def eval_legal_actions_cython_v3():
    eval_legal_actions_template(legal_actions_cython_v3)


from functools import lru_cache


@lru_cache(maxsize=128)
def eval_legal_actions_cython_v3_lru():
    eval_legal_actions_template(legal_actions_cython_v3)


def eval_legal_actions_cython_str():
    eval_legal_actions_template(legal_actions_cython_str)


@lru_cache(maxsize=128)
def eval_legal_actions_cython_str_lru():
    eval_legal_actions_template(legal_actions_cython_str)


if __name__ == "__main__":
    # eval_times = 1000000
    eval_times = 10000

    print(f"##### execute eval_legal_actions {eval_times} times #####")

    with timer:
        for _ in range(eval_times):
            eval_legal_actions_forloop()
    print(f"---------------------------------------")
    print(f"| legal_actions_forloop | {timer.value:.3f} |")
    print(f"---------------------------------------")

    with timer:
        for _ in range(eval_times):
            eval_legal_actions_forloop_str()
    print(f"---------------------------------------")
    print(f"| legal_actions_forloop_str | {timer.value:.3f} |")
    print(f"---------------------------------------")

    with timer:
        for _ in range(eval_times):
            eval_legal_actions_enumerate_str()
    print(f"---------------------------------------")
    print(f"| legal_actions_enumerate_str | {timer.value:.3f} |")
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

    with timer:
        for _ in range(eval_times):
            eval_legal_actions_cython_str()
    print(f"---------------------------------------")
    print(f"| legal_actions_cython_str  | {timer.value:.3f} |")
    print(f"---------------------------------------")

    with timer:
        for _ in range(eval_times):
            eval_legal_actions_cython_str_lru()
    print(f"---------------------------------------")
    print(f"| legal_actions_cython_str_lru  | {timer.value:.3f} |")
    print(f"---------------------------------------")

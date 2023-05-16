"""
Overview:
    Efficiency comparison of different vectorization methods based on get_done_winner_function `get_done_winner`:
    NOTE: The time may vary on different devices and software versions.
    =======================================
    ### execute get_done_winner 1000,000 times###
    ---------------------------------------
    | Methods                      | Seconds
    ---------------------------------------
    | get_done_winner_python        | 30.645
    | get_done_winner_cython      | 21.828
    | get_done_winner_cython_lru  | 0.011
"""

import numpy as np
from ding.utils import EasyTimer
from zoo.board_games.gomoku.envs.get_done_winner_cython import get_done_winner_cython

timer = EasyTimer(cuda=True)


def get_done_winner_python(board_size, board):
    """
    Overview:
         Check if the game is over and who the winner is. Return 'done' and 'winner'.
    Arguments:
        - board_size (:obj:`int`): The size of the board.
        - board (:obj:`numpy.ndarray`): The board state.
    Returns:
        - outputs (:obj:`Tuple`): Tuple containing 'done' and 'winner',
            - if player 1 win,     'done' = True, 'winner' = 1
            - if player 2 win,     'done' = True, 'winner' = 2
            - if draw,             'done' = True, 'winner' = -1
            - if game is not over, 'done' = False, 'winner' = -1
    """
    # has_get_done_winner i.e. not done
    has_get_done_winner = False
    directions = ((1, -1), (1, 0), (1, 1), (0, 1))
    for i in range(board_size):
        for j in range(board_size):
            # if no stone is on the position, don't need to consider this position
            if board[i][j] == 0:
                has_get_done_winner = True
                continue
            # value-value at a coord, i-row, j-col
            player = board[i][j]
            # check if there exist 5 in a line
            for d in directions:
                x, y = i, j
                count = 0
                for _ in range(5):
                    if (x not in range(board_size)) or (y not in range(board_size)):
                        break
                    if board[x][y] != player:
                        break
                    x += d[0]
                    y += d[1]
                    count += 1
                    # if 5 in a line, store positions of all stones, return value
                    if count == 5:
                        return True, player
    # if the players don't have legal actions, return done=True
    return not has_get_done_winner, -1


def eval_get_done_winner_template(get_done_winner_func):
    # case 1
    board_size = 5
    board = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]], dtype=np.int32)
    get_done_winner = get_done_winner_func(board_size, board)

    # case 2
    board_size = 5
    board = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=np.int32)
    get_done_winner = get_done_winner_func(board_size, board)

    # case 3
    board_size = 5
    board = np.array([[0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0]], dtype=np.int32)
    get_done_winner = get_done_winner_func(board_size, board)

    # case 4
    board_size = 5
    board = np.array([[0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0]], dtype=np.int32)
    get_done_winner = get_done_winner_func(board_size, board)

    # case 4
    board_size = 5
    board = np.array([[1, 1, 0, 1, 1], [1, 1, 0, 1, 1], [1, 1, 0, 1, 1], [1, 1, 0, 1, 1], [1, 1, 0, 1, 1]], dtype=np.int32)
    get_done_winner = get_done_winner_func(board_size, board)


def eval_get_done_winner_python():
    eval_get_done_winner_template(get_done_winner_python)


def eval_get_done_winner_cython():
    eval_get_done_winner_template(get_done_winner_cython)


from functools import lru_cache


@lru_cache(maxsize=128)
def eval_get_done_winner_cython_lru():
    eval_get_done_winner_template(get_done_winner_cython)


if __name__ == "__main__":
    eval_times = 10000

    print(f"##### execute eval_get_done_winner {eval_times} times #####")

    with timer:
        for _ in range(eval_times):
            eval_get_done_winner_python()
    print(f"---------------------------------------")
    print(f"| get_done_winner_python | {timer.value:.3f} |")
    print(f"---------------------------------------")

    with timer:
        for _ in range(eval_times):
            eval_get_done_winner_cython()
    print(f"---------------------------------------")
    print(f"| get_done_winner_cython  | {timer.value:.3f} |")
    print(f"---------------------------------------")

    with timer:
        for _ in range(eval_times):
            eval_get_done_winner_cython_lru()
    print(f"---------------------------------------")
    print(f"| get_done_winner_cython_lru  | {timer.value:.3f} |")
    print(f"---------------------------------------")

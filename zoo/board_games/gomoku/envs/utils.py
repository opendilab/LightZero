import copy
import sys

import numpy as np


def check_action_to_special_connect4_case1(board):
    board = copy.deepcopy(board)
    if (board == [0, -1, -1, -1, 0]).all() or (board == [0, 1, 1, 1, 0]).all():
        return True
    else:
        return False


def check_action_to_special_connect4_case2(board):
    board = copy.deepcopy(board)
    if (board == [1, 1, 1, 0, 0]).all() or (board == [-1, -1, -1, 0, 0]).all() or (
            np.flip(board) == [1, 1, 1, 0, 0]).all() or (
            np.flip(board) == [-1, -1, -1, 0, 0]).all() or \
                (board == [1, 1, 0, 1, 0]).all() or (board == [-1, -1, 0, -1, 0]).all() or (
            np.flip(board) == [1, 1, 0, 1, 0]).all() or (
            np.flip(board) == [-1, -1, 0, -1, 0]).all() or \
                (board == [1, 0, 1, 1, 0]).all() or (board == [-1, 0, -1, -1, 0]).all() or (
            np.flip(board) == [1, 0, 1, 1, 0]).all() or (
            np.flip(board) == [-1, 0, -1, -1, 0]).all():
        return True
    else:
        return False



def check_action_to_connect4(board):
    board = copy.deepcopy(board)
    if ((board == -1).sum() == 3 and (board == 0).sum() == 2) or \
            ((board == 1).sum() == 3 and (board == 0).sum() == 2):
        return True
    else:
        return False

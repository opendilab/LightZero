from functools import lru_cache
from typing import Tuple, Union

import numpy as np


# Define expectimax search bot for 2048 env
def expectimax_search(grid: np.array, fast_search: bool = True) -> int:
    """
    Overview:
        Use Expectimax search algorithm to find the best action for 2048 env.
        Adapted from https://github.com/xwjdsh/2048-ai/blob/master/ai/ai.go.
    """
    # please refer to https://codemyroad.wordpress.com/2014/05/14/2048-ai-the-intelligent-bot/
    model1 = np.array([[16, 15, 14, 13], [9, 10, 11, 12], [8, 7, 6, 5], [1, 2, 2, 4]])
    model2 = np.array([[16, 15, 12, 4], [14, 13, 11, 3], [10, 9, 8, 2], [7, 6, 5, 1]])
    model3 = np.array([[16, 15, 14, 4], [13, 12, 11, 3], [10, 9, 8, 2], [7, 6, 5, 1]])

    # Use lru_cache decorator for caching, speeding up subsequent look-ups
    @lru_cache(maxsize=512)
    def get_model_score(value, i, j):
        result = np.zeros(3 * 8)
        for k, m in enumerate([model1, model2, model3]):
            start = k * 8
            result[start] += m[i, j] * value
            # Scores of other 7 directions of the model
            result[start + 1] += m[i, 3 - j] * value
            result[start + 2] += m[j, i] * value
            result[start + 3] += m[3 - j, i] * value
            result[start + 4] += m[3 - i, 3 - j] * value
            result[start + 5] += m[3 - i, j] * value
            result[start + 6] += m[j, 3 - i] * value
            result[start + 7] += m[3 - j, 3 - i] * value
        return result

    def get_score(grid: np.array) -> float:
        # Calculate the score of the current layout
        result = np.zeros(3 * 8)
        for i in range(4):
            for j in range(4):
                if grid[i, j] != 0:
                    result += get_model_score(grid[i, j], i, j)

        return result.max()

    def expectation_search(grid: np.array, depth: int, chance_node: bool) -> Tuple[float, Union[int, None]]:
        # Use Expectimax search algorithm to find the best action
        # please refer to https://courses.cs.washington.edu/courses/cse473/11au/slides/cse473au11-adversarial-search.pdf
        if depth == 0:
            return get_score(grid), None
        if chance_node:
            cum_score = 0.
            if fast_search:
                choices = [[2, 0.9]]
            else:
                choices = zip([2, 4], [0.9, 0.1])
            for value, prob in choices:
                value, prob = 2, 0.9
                for i in range(4):
                    for j in range(4):
                        if grid[i, j] == 0:
                            grid[i, j] = value
                            cum_score += prob * expectation_search(grid, depth - 1, False)[0]
                            grid[i, j] = 0
            empty_count = np.sum(grid == 0)
            cum_score /= empty_count
            return cum_score, None
        else:
            best_score = 0
            best_action = None
            # 0, 1, 2, 3 mean top, right, bottom, left
            for dire in [0, 1, 2, 3]:
                new_grid, move_flag, _ = move(grid, dire)
                if move_flag:
                    score = expectation_search(new_grid, depth - 1, True)[0]
                    if score > best_score:
                        best_score = score
                        best_action = dire
            return best_score, best_action

    #  Select search depth based on the current maximum tile value
    grid_max = grid.max()
    if grid_max >= 2048:
        depth = 6
    elif grid_max >= 1024:
        depth = 5
    else:
        depth = 4
    # Call the expectation search algorithm and return the best action
    _, best_action = expectation_search(grid, depth, False)
    return best_action


# Define move function, implement move operation in 2048 game
def move(grid: np.array, action: int, game_score: int = 0) -> Tuple[np.array, bool, int]:
    # execute action in 2048 game
    # 0, 1, 2, 3 mean top, right, bottom, left
    assert action in [0, 1, 2, 3], action
    old_grid = grid
    grid = np.copy(grid)
    # rotate
    if action == 0:
        grid = np.rot90(grid)
    elif action == 1:
        grid = np.rot90(grid, k=2)
    elif action == 2:
        grid = np.rot90(grid, k=3)
    # simple move
    for i in range(4):
        for j in range(3):
            if grid[i, j] == 0:
                grid[i, j] = grid[i, j + 1]
                grid[i, j + 1] = 0
    # merge
    for i in range(4):
        for j in range(3):
            if grid[i, j] == grid[i, j + 1]:
                game_score += 2 * grid[i, j]
                grid[i, j] *= 2
                grid[i, j + 1] = 0
    # simple move
    for i in range(4):
        for j in range(3):
            if grid[i, j] == 0:
                grid[i, j] = grid[i, j + 1]
                grid[i, j + 1] = 0
    # rotate back
    if action == 0:
        grid = np.rot90(grid, k=3)
    elif action == 1:
        grid = np.rot90(grid, k=2)
    elif action == 2:
        grid = np.rot90(grid)
    move_flag = np.any(old_grid != grid)
    return grid, move_flag, game_score


# # Define generate function, randomly generate 2 or 4 in an empty location
def generate(grid: np.array) -> np.array:
    number = np.random.choice([2, 4], p=[0.9, 0.1])
    # get empty location
    empty = np.where(grid == 0)
    # random select one
    index = np.random.randint(len(empty[0]))
    # set new number
    grid[empty[0][index], empty[1][index]] = number
    # return new grid
    return grid
from zoo.board_games.gomoku.envs.legal_actions_cython import legal_actions_cython


def test_legal_actions_cython():
    # case 1
    board_size = 2
    board = [[0, 0], [0, 0]]
    legal_actions = legal_actions_cython(board_size, board)
    assert legal_actions == [0, 1, 2, 3], f"Error: {legal_actions}"

    # case 2
    board_size = 3
    board = [[0, 0, 0], [0, 1, 1], [0, 1, 1]]
    legal_actions = legal_actions_cython(board_size, board)
    assert legal_actions == [0, 1, 2, 3, 6], f"Error: {legal_actions}"

    # case 3
    board_size = 4
    board = [[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 0]]
    legal_actions = legal_actions_cython(board_size, board)
    assert legal_actions == [3, 7, 11, 15], f"Error: {legal_actions}"


test_legal_actions_cython()
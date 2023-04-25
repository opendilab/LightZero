from zoo.board_games.gomoku.test.legal_actions_cython_str import legal_actions_cython_str


def eval_legal_actions_template(legal_actions_func_str):
    # case 1
    board_size = 2
    board = [[0, 0], [0, 0]]
    board_str = ''.join(str(cell) for row in board for cell in row)
    if legal_actions_func_str is legal_actions_forloop_str:
        legal_actions = legal_actions_func_str(board_size, board_str)
    else:
        legal_actions = legal_actions_func_str(board_str)
    assert legal_actions == [0, 1, 2, 3], f"Error: {legal_actions}"

    # case 2
    board_size = 3
    board = [[0, 0, 0], [0, 1, 1], [0, 1, 1]]
    board_str = ''.join(str(cell) for row in board for cell in row)
    if legal_actions_func_str is legal_actions_forloop_str:
        legal_actions = legal_actions_func_str(board_size, board_str)
    else:
        legal_actions = legal_actions_func_str(board_str)
    assert legal_actions == [0, 1, 2, 3, 6], f"Error: {legal_actions}"

    # case 3
    board_size = 4
    board = [[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 0]]
    board_str = ''.join(str(cell) for row in board for cell in row)
    if legal_actions_func_str is legal_actions_forloop_str:
        legal_actions = legal_actions_func_str(board_size, board_str)
    else:
        legal_actions = legal_actions_func_str(board_str)
    assert legal_actions == [3, 7, 11, 15], f"Error: {legal_actions}"


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


def test_legal_actions_forloop_str():
    eval_legal_actions_template(legal_actions_forloop_str)


def test_legal_actions_enumerate_str():
    eval_legal_actions_template(legal_actions_enumerate_str)


def test_legal_actions_cython_str():
    eval_legal_actions_template(legal_actions_cython_str)


test_legal_actions_forloop_str()
test_legal_actions_enumerate_str()
test_legal_actions_cython_str()

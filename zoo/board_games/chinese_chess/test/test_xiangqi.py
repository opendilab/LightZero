import numpy as np

from zoo.board_games.chinese_chess.envs.action_encoding import move_to_uci, uci_to_move
from zoo.board_games.chinese_chess.envs.xiangqi import (
    BLACK,
    CANNON,
    ELEPHANT,
    HORSE,
    KING,
    RED,
    ROOK,
    XiangqiState,
)


def _empty_position(turn: int = RED) -> np.ndarray:
    board = np.zeros((10, 9), dtype=np.int8)
    board[0, 4] = KING
    board[9, 4] = -KING
    board[5, 4] = ROOK  # Prevent an initially illegal flying-general position.
    return board


def _uci_moves(state: XiangqiState):
    return {move_to_uci(move) for move in state.legal_moves()}


def test_initial_position_has_44_legal_moves() -> None:
    state = XiangqiState()
    assert len(state.legal_moves()) == 44
    assert {'a0a1', 'b0a2', 'b0c2', 'h2h9'} <= _uci_moves(state)


def test_horse_leg_and_elephant_eye_are_blocked() -> None:
    board = _empty_position()
    board[0, 1] = HORSE
    board[1, 1] = ROOK
    board[0, 2] = ELEPHANT
    board[1, 3] = ROOK
    moves = _uci_moves(XiangqiState(board))
    assert 'b0a2' not in moves
    assert 'b0c2' not in moves
    assert 'c0e2' not in moves


def test_cannon_requires_exactly_one_screen_to_capture() -> None:
    board = _empty_position()
    board[0, 0] = CANNON
    board[1, 0] = ROOK
    board[3, 0] = -ROOK
    board[4, 0] = -HORSE
    moves = _uci_moves(XiangqiState(board))
    assert 'a0a3' in moves
    assert 'a0a4' not in moves


def test_piece_cannot_expose_flying_generals() -> None:
    board = _empty_position()
    state = XiangqiState(board)
    moves = _uci_moves(state)
    assert 'e5d5' not in moves
    assert 'e5f5' not in moves
    assert 'e5e9' in moves


def test_stalemate_is_a_loss_for_side_to_move() -> None:
    board = np.zeros((10, 9), dtype=np.int8)
    board[0, 4] = KING
    board[9, 4] = -KING
    board[1, 3] = -ROOK
    board[1, 5] = -ROOK
    board[2, 4] = -ROOK
    state = XiangqiState(board, turn=RED)
    result = state.result()
    assert result.done
    assert result.winner == BLACK


def test_threefold_repetition_and_sixty_move_draws() -> None:
    state = XiangqiState()
    state.repetition[state.key()] = 3
    assert state.result().done and state.result().winner == 0

    state = XiangqiState(halfmove_clock=120)
    assert state.result().done and state.result().winner == 0


def test_push_switches_turn_and_rejects_illegal_move() -> None:
    state = XiangqiState()
    state.push(uci_to_move('a0a1'))
    assert state.turn == BLACK
    try:
        state.push(uci_to_move('a0a2'))
    except ValueError:
        pass
    else:
        raise AssertionError('expected an illegal move to be rejected')

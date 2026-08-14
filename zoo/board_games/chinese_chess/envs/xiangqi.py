"""Small, dependency-free Xiangqi rules engine used by the LightZero env.

It implements piece movement, horse/elephant blocking, cannon screens,
palaces, the river, flying generals, self-check filtering, stalemate as a
loss, threefold repetition, and the 60-move no-capture draw rule.
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

ROWS, COLS = 10, 9
RED, BLACK = 1, -1
EMPTY, KING, ADVISOR, ELEPHANT, HORSE, ROOK, CANNON, PAWN = range(8)
Move = Tuple[int, int]

INITIAL_BOARD = np.array(
    [
        [ROOK, HORSE, ELEPHANT, ADVISOR, KING, ADVISOR, ELEPHANT, HORSE, ROOK],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, CANNON, 0, 0, 0, 0, 0, CANNON, 0],
        [PAWN, 0, PAWN, 0, PAWN, 0, PAWN, 0, PAWN],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-PAWN, 0, -PAWN, 0, -PAWN, 0, -PAWN, 0, -PAWN],
        [0, -CANNON, 0, 0, 0, 0, 0, -CANNON, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-ROOK, -HORSE, -ELEPHANT, -ADVISOR, -KING, -ADVISOR, -ELEPHANT, -HORSE, -ROOK],
    ],
    dtype=np.int8
)


def _inside(row: int, col: int) -> bool:
    return 0 <= row < ROWS and 0 <= col < COLS


def _palace(color: int, row: int, col: int) -> bool:
    return 3 <= col <= 5 and ((color == RED and 0 <= row <= 2) or (color == BLACK and 7 <= row <= 9))


@dataclass
class PositionResult:
    done: bool
    winner: int  # RED, BLACK, or 0 for draw/no winner.


class XiangqiState:
    """Mutable Xiangqi position with cached legal move generation."""

    def __init__(
        self,
        board: Optional[np.ndarray] = None,
        turn: int = RED,
        halfmove_clock: int = 0,
        ply: int = 0,
        repetition: Optional[Dict[bytes, int]] = None,
        position_hashes: Optional[List[int]] = None,
    ) -> None:
        self.board = np.array(INITIAL_BOARD if board is None else board, dtype=np.int8, copy=True).reshape(ROWS, COLS)
        self.turn = int(turn)
        self.halfmove_clock = int(halfmove_clock)
        self.ply = int(ply)
        self.repetition = {} if repetition is None else dict(repetition)
        self.repetition[self.key()] = max(1, self.repetition.get(self.key(), 0))
        self.position_hashes = list(position_hashes) if position_hashes else [self.position_hash()]
        if self.position_hashes[-1] != self.position_hash():
            self.position_hashes.append(self.position_hash())
        self.position_hashes = self.position_hashes[-121:]
        self._legal_cache: Optional[List[Move]] = None

    def copy(self) -> 'XiangqiState':
        return copy.deepcopy(self)

    def key(self) -> bytes:
        return self.board.tobytes() + bytes((1 if self.turn == RED else 0, ))

    def position_hash(self) -> int:
        """Return a stable 64-bit hash used for compact CTree serialization."""
        return int.from_bytes(hashlib.blake2b(self.key(), digest_size=8).digest(), 'little')

    @staticmethod
    def _color(piece: int) -> int:
        return RED if piece > 0 else BLACK

    def _piece_moves(self, square: int) -> Iterable[Move]:
        row, col = divmod(square, COLS)
        signed_piece = int(self.board[row, col])
        if signed_piece == EMPTY:
            return
        color, piece = self._color(signed_piece), abs(signed_piece)

        if piece == KING:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                to_row, to_col = row + dr, col + dc
                if _palace(color, to_row, to_col):
                    target = int(self.board[to_row, to_col])
                    if target == EMPTY or self._color(target) != color:
                        yield square, to_row * COLS + to_col
            # Flying-general capture.
            direction = 1 if color == RED else -1
            to_row = row + direction
            while 0 <= to_row < ROWS and self.board[to_row, col] == EMPTY:
                to_row += direction
            if 0 <= to_row < ROWS and int(self.board[to_row, col]) == -color * KING:
                yield square, to_row * COLS + col

        elif piece == ADVISOR:
            for dr, dc in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                to_row, to_col = row + dr, col + dc
                if _palace(color, to_row, to_col):
                    target = int(self.board[to_row, to_col])
                    if target == EMPTY or self._color(target) != color:
                        yield square, to_row * COLS + to_col

        elif piece == ELEPHANT:
            for dr, dc in ((2, 2), (2, -2), (-2, 2), (-2, -2)):
                to_row, to_col = row + dr, col + dc
                own_side = to_row <= 4 if color == RED else to_row >= 5
                if _inside(to_row, to_col) and own_side and self.board[row + dr // 2, col + dc // 2] == EMPTY:
                    target = int(self.board[to_row, to_col])
                    if target == EMPTY or self._color(target) != color:
                        yield square, to_row * COLS + to_col

        elif piece == HORSE:
            for dr, dc, leg_dr, leg_dc in (
                (2, 1, 1, 0),
                (2, -1, 1, 0),
                (-2, 1, -1, 0),
                (-2, -1, -1, 0),
                (1, 2, 0, 1),
                (-1, 2, 0, 1),
                (1, -2, 0, -1),
                (-1, -2, 0, -1),
            ):
                to_row, to_col = row + dr, col + dc
                if _inside(to_row, to_col) and self.board[row + leg_dr, col + leg_dc] == EMPTY:
                    target = int(self.board[to_row, to_col])
                    if target == EMPTY or self._color(target) != color:
                        yield square, to_row * COLS + to_col

        elif piece in (ROOK, CANNON):
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                to_row, to_col, screened = row + dr, col + dc, False
                while _inside(to_row, to_col):
                    target = int(self.board[to_row, to_col])
                    if piece == ROOK:
                        if target == EMPTY:
                            yield square, to_row * COLS + to_col
                        else:
                            if self._color(target) != color:
                                yield square, to_row * COLS + to_col
                            break
                    elif not screened:
                        if target == EMPTY:
                            yield square, to_row * COLS + to_col
                        else:
                            screened = True
                    elif target != EMPTY:
                        if self._color(target) != color:
                            yield square, to_row * COLS + to_col
                        break
                    to_row, to_col = to_row + dr, to_col + dc

        elif piece == PAWN:
            forward = 1 if color == RED else -1
            directions = [(forward, 0)]
            crossed_river = row >= 5 if color == RED else row <= 4
            if crossed_river:
                directions.extend(((0, 1), (0, -1)))
            for dr, dc in directions:
                to_row, to_col = row + dr, col + dc
                if _inside(to_row, to_col):
                    target = int(self.board[to_row, to_col])
                    if target == EMPTY or self._color(target) != color:
                        yield square, to_row * COLS + to_col

    def pseudo_moves(self, color: int) -> Iterable[Move]:
        for square, piece in enumerate(self.board.flat):
            if piece != EMPTY and self._color(int(piece)) == color:
                yield from self._piece_moves(square)

    def king_square(self, color: int) -> Optional[int]:
        locations = np.flatnonzero(self.board.reshape(-1) == color * KING)
        return int(locations[0]) if len(locations) else None

    def is_in_check(self, color: int) -> bool:
        king = self.king_square(color)
        if king is None:
            return True
        return any(destination == king for _, destination in self.pseudo_moves(-color))

    def legal_moves(self) -> List[Move]:
        if self._legal_cache is not None:
            return list(self._legal_cache)
        legal: List[Move] = []
        color = self.turn
        for move in self.pseudo_moves(color):
            origin, destination = move
            from_row, from_col = divmod(origin, COLS)
            to_row, to_col = divmod(destination, COLS)
            captured = int(self.board[to_row, to_col])
            self.board[to_row, to_col] = self.board[from_row, from_col]
            self.board[from_row, from_col] = EMPTY
            safe = not self.is_in_check(color)
            self.board[from_row, from_col] = self.board[to_row, to_col]
            self.board[to_row, to_col] = captured
            if safe:
                legal.append(move)
        self._legal_cache = legal
        return list(legal)

    def push(self, move: Move) -> None:
        if move not in self.legal_moves():
            raise ValueError(f'illegal Xiangqi move: {move}')
        origin, destination = move
        from_row, from_col = divmod(origin, COLS)
        to_row, to_col = divmod(destination, COLS)
        captured = int(self.board[to_row, to_col])
        moving_piece = abs(int(self.board[from_row, from_col]))
        self.board[to_row, to_col] = self.board[from_row, from_col]
        self.board[from_row, from_col] = EMPTY
        self.halfmove_clock = 0 if captured or moving_piece == PAWN else self.halfmove_clock + 1
        self.ply += 1
        self.turn = -self.turn
        self._legal_cache = None
        key = self.key()
        self.repetition[key] = self.repetition.get(key, 0) + 1
        current_hash = self.position_hash()
        if captured or moving_piece == PAWN:
            # A position from before an irreversible move cannot recur.
            self.position_hashes = [current_hash]
        else:
            self.position_hashes.append(current_hash)
            self.position_hashes = self.position_hashes[-121:]

    def result(self, max_ply: int = 500) -> PositionResult:
        red_king, black_king = self.king_square(RED), self.king_square(BLACK)
        if red_king is None:
            return PositionResult(True, BLACK)
        if black_king is None:
            return PositionResult(True, RED)
        repeated = self.repetition.get(self.key(), 0) >= 3 or self.position_hashes.count(self.position_hash()) >= 3
        if repeated or self.halfmove_clock >= 120 or self.ply >= max_ply:
            return PositionResult(True, 0)
        if not self.legal_moves():
            return PositionResult(True, -self.turn)  # Xiangqi stalemate is a loss.
        return PositionResult(False, 0)

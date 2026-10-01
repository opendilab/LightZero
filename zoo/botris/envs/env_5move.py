from __future__ import annotations
from botris import TetrisGame
from .utils import encode_input, encode_move_index, encode_board, decode_move_index, encode_piece_coordinates, dencode_piece_coordinates, decode_queue, decode_board
from .modals import Piece, EncodedInput, Rotation, EncodedBoard, NUMBER_OF_ROWS, NUMBER_OF_COLS, NUMBER_OF_PIECES
from typing import Deque, List, Tuple
from botris.engine import Board, PieceData, Move
from botris.engine import Piece as BotrisPiece
from botris.engine.utils import place_piece
import numpy as np
from collections import deque
from PIL import Image

MOVES = np.array([Move.hold, Move.move_left, Move.move_right, Move.rotate_cw, Move.rotate_ccw, Move.drop, Move.sonic_drop, Move.sonic_left, Move.sonic_right, Move.hard_drop])
ACTION_SPACE_SIZE: int = MOVES.size


class GameEnvironment5Move:
    def __init__(self, score_scale: int | None = 5, piece_reward: int | None = 1) -> None:
        options = {}
        if score_scale is not None:
            options['attack_table'] = {
                "single": score_scale,
                "double": score_scale * 2,
                "triple": score_scale * 4,
                "quad": score_scale * 8,
                "ass": score_scale * 4,
                "asd": score_scale * 8,
                "ast": score_scale * 12,
                "pc": score_scale * 20,
                "b2b": score_scale * 2,
            }
            options['combo_table'] = [score_scale, score_scale, score_scale * 2, score_scale * 2, score_scale * 2, score_scale * 4, score_scale * 4, score_scale * 6, score_scale * 6, score_scale * 8]
        options['board_width'] = NUMBER_OF_COLS
        options['board_height'] = NUMBER_OF_ROWS
        self.game: TetrisGame = TetrisGame(options=options)
        self.piece_reward: int | None = piece_reward

    def copy(self) -> GameEnvironment5Move:
        new_env = GameEnvironment5Move()
        new_env.game = self.game.copy()
        return new_env

    def reset(self) -> None:
        self.game.reset()

    @property
    def terminal(self) -> bool:
        return self.game.dead

    def get_input_encoding(self) -> EncodedInput:
        _board: Board = self.game.board #place_piece(self.game.board, self.game.current, self.game.options.board_width)
        board: EncodedBoard = encode_board(_board)

        _queue: Deque[BotrisPiece] = self.game.queue
        queue: List[Piece] = [piece.index for piece in list(_queue)]

        _current_piece: BotrisPiece = self.game.current.piece
        current_piece: Piece = _current_piece.index

        _held_piece: BotrisPiece = self.game.held
        held_piece: Piece = _held_piece.index if _held_piece is not None else Piece.NONE

        garbage_queued: int = len(self.game.garbage_queue)
        combo: int = self.game.combo
        b2b: bool = self.game.b2b

        return encode_input(board, queue, current_piece, held_piece, garbage_queued, combo, b2b)

    def step(self, move_idx: int) -> None:
        move: Move = MOVES[move_idx]
        self.game.execute_move(move)

    def get_score(self, terminal_score=None) -> int:
        if self.game.dead and terminal_score is not None:
            return terminal_score
        if self.piece_reward is not None:
            return self.game.score + self.game.pieces_placed * self.piece_reward
        return self.game.score
    
    def render(self, render_current=False) -> None:
        self.game.render_board(render_current=render_current)

    def draw(self) -> Image:
        return self.game.draw_board()
from __future__ import annotations
from botris import TetrisGame
from .utils import encode_input, encode_move_index, encode_board, decode_move_index, encode_piece_coordinates, dencode_piece_coordinates, decode_queue, decode_board
from .modals import Piece, EncodedInput, Rotation, EncodedMove, ENCODED_MOVE_SHAPE, EncodedBoard, NUMBER_OF_ROWS, NUMBER_OF_COLS, NUMBER_OF_PIECES
from typing import Deque, List, Tuple
from botris.engine import Board, PieceData
from botris.engine import Piece as BotrisPiece
import numpy as np
from collections import deque
from PIL import Image

class GameEnvironment:
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
        self.game: TetrisGame = TetrisGame(options=options)
        self.piece_reward: int | None = piece_reward

    def copy(self) -> GameEnvironment:
        new_env = GameEnvironment()
        new_env.game = self.game.copy()
        return new_env

    def reset(self) -> None:
        self.game.reset()

    def is_terminal(self) -> bool:
        return self.game.dead

    def get_input_encoding(self) -> EncodedInput:
        _board: Board = self.game.board
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

    def step(self, move: Tuple[Piece, Rotation, int, int]) -> None:
        piece_type, rotation, row, col = move
        botris_piece: BotrisPiece = BotrisPiece.from_index(piece_type)
        x, y = dencode_piece_coordinates(botris_piece, rotation, row, col)
        piece_data = PieceData(botris_piece, x, y, rotation)
        self.game.dangerously_drop_piece(piece_data)

    def step_action(self, action) -> None:
        move = decode_move_index(action)
        self.step(move)

    def get_score(self, terminal_score=None) -> int:
        if self.game.dead and terminal_score is not None:
            return terminal_score
        if self.piece_reward is not None:
            return self.game.score + self.game.pieces_placed * self.piece_reward
        return self.game.score

    def legal_moves_mask(self) -> EncodedMove:
        legal_moves_dict = self.game.generate_moves()
        legal_moves = np.zeros(ENCODED_MOVE_SHAPE, dtype=bool)
        for piece_data in legal_moves_dict.keys():
            piece, rotation = piece_data.piece.index, piece_data.rotation
            col, row = encode_piece_coordinates(piece_data)
            if (col < 0) or (col >= NUMBER_OF_COLS) or (row < 0) or (row >= NUMBER_OF_ROWS):
                continue
            move_idx = encode_move_index(piece, rotation, row, col)
            legal_moves[move_idx] = True
        return legal_moves
    
    def render(self, render_current=False) -> None:
        self.game.render_board(render_current=render_current)

    def draw(self) -> Image:
        return self.game.draw_board()
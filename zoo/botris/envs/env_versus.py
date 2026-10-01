from __future__ import annotations
from botris import TetrisGame
from .utils import encode_input, encode_move_index, encode_board, decode_move_index, encode_piece_coordinates, dencode_piece_coordinates, decode_queue, decode_board
from .modals import Piece, EncodedInput, Rotation, EncodedMove, ENCODED_MOVE_SHAPE, EncodedBoard, NUMBER_OF_ROWS, NUMBER_OF_COLS, NUMBER_OF_PIECES
from typing import Deque, List, Tuple
from botris.engine import Board, PieceData, generate_garbage
from botris.engine import Piece as BotrisPiece
import numpy as np
from collections import deque
from PIL import Image

class GameEnvironment:
    def __init__(self) -> None:
        self.game1: TetrisGame = TetrisGame()
        self.game2: TetrisGame = TetrisGame()
        self.current_player: int = 0

    def copy(self) -> GameEnvironment:
        new_env = GameEnvironment()
        new_env.game1 = self.game1.copy()
        new_env.game2 = self.game2.copy()
        new_env.current_player = self.current_player
        return new_env

    def reset(self) -> None:
        self.game1.reset()
        self.game2.reset()
        self.current_player = 0

    def is_terminal(self) -> bool:
        return self.game1.dead or self.game2.dead

    def get_input_encoding(self) -> EncodedInput:
        game = self.game2 if self.current_player else self.game1
        _board: Board = game.board
        board: EncodedBoard = encode_board(_board)

        _queue: Deque[BotrisPiece] = game.queue
        queue: List[Piece] = [piece.index for piece in list(_queue)]

        _current_piece: BotrisPiece = game.current.piece
        current_piece: Piece = _current_piece.index

        _held_piece: BotrisPiece = game.held
        held_piece: Piece = _held_piece.index if _held_piece is not None else Piece.NONE

        garbage_queued: int = len(game.garbage_queue)
        combo: int = game.combo
        b2b: bool = game.b2b

        return encode_input(board, queue, current_piece, held_piece, garbage_queued, combo, b2b)

    def step(self, move: Tuple[Piece, Rotation, int, int]) -> None:
        piece_type, rotation, row, col = move
        botris_piece: BotrisPiece = BotrisPiece.from_index(piece_type)
        x, y = dencode_piece_coordinates(botris_piece, rotation, row, col)
        piece_data = PieceData(botris_piece, x, y, rotation)
        game = self.game2 if self.current_player else self.game1
        events = game.dangerously_drop_piece(piece_data)
        self.current_player = 1 - self.current_player
        other_game = self.game2 if self.current_player else self.game1
        for event in events:
            if event.type == "clear":
                attack: int = event.attack
                other_game.queue_attack(attack)

    def step_action(self, action) -> None:
        move = decode_move_index(action)
        self.step(move)

    def get_winner(self) -> int:
        if self.game1.dead and self.game2.dead:
            return -1
        if self.game1.dead:
            return 1
        if self.game2.dead:
            return 0
        return None

    def legal_moves_mask(self) -> EncodedMove:
        game = self.game2 if self.current_player else self.game1

        legal_moves_dict = game.generate_moves()
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
        game = self.game2 if self.current_player else self.game1
        game.render_board(render_current=render_current)

    def draw(self) -> Image:
        game = self.game2 if self.current_player else self.game1
        return game.draw_board()
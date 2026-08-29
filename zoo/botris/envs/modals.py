from __future__ import annotations
from typing import Tuple, Literal, Annotated
import numpy as np
from numpy.typing import NDArray

class Piece(int):
    I: Piece = 0
    O: Piece = 1
    J: Piece = 2
    L: Piece = 3
    S: Piece = 4
    Z: Piece = 5
    T: Piece = 6
    NONE: Piece = 7

NUMBER_OF_PIECES: int = 8
PIECES: Tuple[Piece] = (Piece.I, Piece.O, Piece.J, Piece.L, Piece.S, Piece.Z, Piece.T, Piece.NONE)

NUMBER_OF_ROWS: int = 8
NUMBER_OF_COLS: int = 10

Rotation = Literal[0, 1, 2, 3]
NUMBER_OF_ROTATIONS: int = 4

QUEUE_SIZE: int = 6
INCLUDE_CURRENT_PIECE: Literal[0, 1] = 1
INCLUDE_HELD_PIECE: Literal[0, 1] = 1
INCLUDE_GARBAGE_QUEUED: Literal[0, 1] = 0
INCLUDE_COMBO: Literal[0, 1] = 0
INCLUDE_B2B: Literal[0, 1] = 0

MAX_GARBAGE_QUEUED: int = 15
MAX_COMBO: int = 11
MAX_MOVE_SCORE: int = 100

ACTION_SPACE_SIZE: int = NUMBER_OF_PIECES * NUMBER_OF_ROTATIONS * NUMBER_OF_ROWS * NUMBER_OF_COLS
OBSERVATION_SPACE_SIZE: int = NUMBER_OF_ROWS * NUMBER_OF_COLS + QUEUE_SIZE * NUMBER_OF_PIECES + NUMBER_OF_PIECES * INCLUDE_CURRENT_PIECE + NUMBER_OF_PIECES * INCLUDE_HELD_PIECE + MAX_GARBAGE_QUEUED * INCLUDE_GARBAGE_QUEUED + MAX_COMBO * INCLUDE_COMBO + INCLUDE_B2B

ENCODED_MOVE_SHAPE: Tuple[int] = (ACTION_SPACE_SIZE,)
ENCODED_BOARD_SHAPE: Tuple[int] = (NUMBER_OF_ROWS, NUMBER_OF_COLS,)
ENCODED_INPUT_SHAPE: Tuple[int] = (OBSERVATION_SPACE_SIZE,)

EncodedMove = Annotated[NDArray[np.int8], ENCODED_MOVE_SHAPE]
EncodedBoard = Annotated[NDArray[np.int8], ENCODED_BOARD_SHAPE]
EncodedInput = Annotated[NDArray[np.int8], ENCODED_INPUT_SHAPE]
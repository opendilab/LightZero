import numpy as np
from .modals import (NUMBER_OF_COLS, NUMBER_OF_PIECES, NUMBER_OF_ROWS, NUMBER_OF_ROTATIONS, Rotation, 
                                      Piece, EncodedBoard, EncodedMove, QUEUE_SIZE, ENCODED_MOVE_SHAPE, ENCODED_INPUT_SHAPE, 
                                      ENCODED_BOARD_SHAPE, ACTION_SPACE_SIZE, INCLUDE_GARBAGE_QUEUED, INCLUDE_CURRENT_PIECE, 
                                      INCLUDE_HELD_PIECE, INCLUDE_COMBO, INCLUDE_B2B, MAX_COMBO, MAX_GARBAGE_QUEUED)
from botris.engine import Board, get_piece_border, PieceData
from botris.engine import Piece as BotrisPiece
from typing import Tuple, List


def encode_move_index(piece_type: Piece, rotation: Rotation, row: int, col: int) -> int:
    return piece_type * NUMBER_OF_ROTATIONS * NUMBER_OF_ROWS * NUMBER_OF_COLS + rotation * NUMBER_OF_ROWS * NUMBER_OF_COLS + row * NUMBER_OF_COLS + col

def decode_move_index(move_idx: int) -> Tuple[Piece, Rotation, int, int]:
    piece_type = move_idx // (NUMBER_OF_ROTATIONS * NUMBER_OF_ROWS * NUMBER_OF_COLS)
    move_idx -= piece_type * NUMBER_OF_ROTATIONS * NUMBER_OF_ROWS * NUMBER_OF_COLS
    rotation = move_idx // (NUMBER_OF_ROWS * NUMBER_OF_COLS)
    move_idx -= rotation * NUMBER_OF_ROWS * NUMBER_OF_COLS
    row = move_idx // NUMBER_OF_COLS
    col = move_idx % NUMBER_OF_COLS
    return piece_type, rotation, row, col


def encode_move(piece_type: Piece, rotation: Rotation, row: int, col: int) -> EncodedMove:
    move_encoding = np.zeros(ENCODED_MOVE_SHAPE, dtype=np.int8)
    move_index: int = encode_move_index(piece_type, rotation, row, col)
    move_encoding[move_index] = 1
    return move_encoding

def decode_move(move_encoding: EncodedMove) -> Tuple[Piece, Rotation, int, int]:
    move_idx = np.argmax(move_encoding)
    return decode_move_index(move_idx)


def encode_input(binary_plane: EncodedBoard, queue: List[Piece], current_piece: Piece, held_piece: Piece, garbage_queued: int, combo: int, b2b: bool):

    queue_encoded = np.zeros((QUEUE_SIZE, NUMBER_OF_PIECES), dtype=np.int8)
    for i, piece_type in enumerate(queue[:QUEUE_SIZE]):
        queue_encoded[i, piece_type] = 1
    
    # Encode current piece type
    if INCLUDE_CURRENT_PIECE:
        current_piece_encoded = np.zeros(NUMBER_OF_PIECES, dtype=np.int8)
        current_piece_encoded[current_piece] = 1
    else:
        current_piece_encoded = np.array([])
    
    # Encode held piece type
    if INCLUDE_HELD_PIECE:
        held_piece_encoded = np.zeros(NUMBER_OF_PIECES, dtype=np.int8)
        held_piece_encoded[held_piece] = 1
    else:
        held_piece_encoded = np.array([])
    
    # Encode garbage queued
    if INCLUDE_GARBAGE_QUEUED:
        garbage_queued = min(garbage_queued, MAX_GARBAGE_QUEUED)
        garbage_queued_encoded = np.zeros(MAX_GARBAGE_QUEUED, dtype=np.int8)
        if garbage_queued >= 0:
            garbage_queued_encoded[garbage_queued - 1] = 1
    else:
        garbage_queued_encoded = np.array([])
    
    # Encode combo'
    if INCLUDE_COMBO:
        combo = min(combo, MAX_COMBO)
        combo_encoded = np.zeros(MAX_COMBO, dtype=np.int8)
        if combo >= 0:
            combo_encoded[combo - 1] = 1
    else:
        combo_encoded = np.array([])
    
    # Encode b2b
    if INCLUDE_B2B:
        b2b_encoded = np.array([int(b2b)], dtype=np.int8)
    else:
        b2b_encoded = np.array([])
    
    # Concatenate all encodings
    input_encoding = np.concatenate([
        binary_plane.flatten(), 
        queue_encoded.flatten(), 
        current_piece_encoded, 
        held_piece_encoded, 
        garbage_queued_encoded, 
        combo_encoded, 
        b2b_encoded
    ])
    
    return input_encoding

def encode_board(board: Board) -> EncodedBoard:
    binary_plane = np.zeros(ENCODED_BOARD_SHAPE, dtype=np.int8)
    for row in range(NUMBER_OF_ROWS):
        if row >= len(board):
            continue
        for col in range(NUMBER_OF_COLS):
            if col >= len(board[row]):
                continue
            if board[row][col] is not None:
                binary_plane[row, col] = 1
    return binary_plane

def decode_board(binary_plane: EncodedBoard) -> Board:
    board = []
    for row in range(NUMBER_OF_ROWS):
        board_row = []
        for col in range(NUMBER_OF_COLS):
            board_row.append(BotrisPiece.I if binary_plane[row, col] else None)
        board.append(board_row)
    return board

def decode_queue(queue: List[Piece]) -> List[BotrisPiece]:
    return [BotrisPiece.from_index(piece_type) for piece_type in queue]

def softmax(x, temperature=1.0):
    x = x / temperature
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def softmax_sample(policy, mask, temperature=1.0):
    policy_probs = softmax(policy, temperature)
    masked_probs = policy_probs * mask
    masked_probs /= masked_probs.sum()
    selected_move = np.random.choice(ACTION_SPACE_SIZE, p=masked_probs)
    return selected_move

def encode_piece_coordinates(piece_data: PieceData) -> Tuple[int, int]:
    lowest_x, highest_x, lowest_y, highest_y = get_piece_border(piece_data.piece, piece_data.rotation)
    return piece_data.x + lowest_x, piece_data.y - highest_y

def dencode_piece_coordinates(piece: BotrisPiece, rotation: Rotation, row: int, col: int) -> Tuple[int, int]:
    lowest_x, highest_x, lowest_y, highest_y = get_piece_border(piece, rotation)
    return col - lowest_x, row + highest_y
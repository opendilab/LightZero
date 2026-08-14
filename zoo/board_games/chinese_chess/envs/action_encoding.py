"""Compact AlphaZero action encoding for a 9 x 10 Xiangqi board.

The 2,086 labels contain every geometrically possible rook/cannon/king/pawn
line move, horse move, and the palace/river-restricted advisor and elephant
moves. Legality for a concrete position is handled by :mod:`xiangqi`.
"""

from typing import Dict, List, Tuple

BOARD_ROWS = 10
BOARD_COLS = 9
ACTION_SPACE_SIZE = 2086
Move = Tuple[int, int]


def _label(from_row: int, from_col: int, to_row: int, to_col: int) -> str:
    return f'{from_col}{from_row}{to_col}{to_row}'


def _build_labels() -> List[str]:
    labels: List[str] = []
    horse_offsets = ((-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1))
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            destinations = (
                [(row, target_col)
                 for target_col in range(BOARD_COLS)] + [(target_row, col) for target_row in range(BOARD_ROWS)] +
                [(row + dr, col + dc) for dr, dc in horse_offsets]
            )
            for to_row, to_col in destinations:
                if (to_row, to_col) != (row, col) and 0 <= to_row < BOARD_ROWS and 0 <= to_col < BOARD_COLS:
                    labels.append(_label(row, col, to_row, to_col))

    # Advisor edges in the red and black palaces.
    labels.extend(
        [
            '3041',
            '5041',
            '3241',
            '5241',
            '4130',
            '4150',
            '4132',
            '4152',
            '3948',
            '5948',
            '3748',
            '5748',
            '4839',
            '4859',
            '4837',
            '4857',
        ]
    )
    # Elephant edges on each side of the river.
    labels.extend(
        [
            '2002',
            '2042',
            '6042',
            '6082',
            '2402',
            '2442',
            '6442',
            '6482',
            '0220',
            '4220',
            '4260',
            '8260',
            '0224',
            '4224',
            '4264',
            '8264',
            '2907',
            '2947',
            '6947',
            '6987',
            '2507',
            '2547',
            '6547',
            '6587',
            '0729',
            '4729',
            '4769',
            '8769',
            '0725',
            '4725',
            '4765',
            '8765',
        ]
    )
    if len(labels) != ACTION_SPACE_SIZE or len(set(labels)) != ACTION_SPACE_SIZE:
        raise RuntimeError('invalid Xiangqi action label table')
    return labels


ACTION_LABELS: List[str] = _build_labels()
LABEL_TO_ACTION: Dict[str, int] = {label: action for action, label in enumerate(ACTION_LABELS)}


def move_to_action(move: Move) -> int:
    """Convert ``(from_square, to_square)`` to a compact action index."""
    from_square, to_square = move
    from_row, from_col = divmod(int(from_square), BOARD_COLS)
    to_row, to_col = divmod(int(to_square), BOARD_COLS)
    try:
        return LABEL_TO_ACTION[_label(from_row, from_col, to_row, to_col)]
    except KeyError as error:
        raise ValueError(f'move {move} is outside the Xiangqi action space') from error


def action_to_move(action: int) -> Move:
    """Convert a compact action index to ``(from_square, to_square)``."""
    if not 0 <= int(action) < ACTION_SPACE_SIZE:
        raise ValueError(f'action must be in [0, {ACTION_SPACE_SIZE}), got {action}')
    label = ACTION_LABELS[int(action)]
    return int(label[1]) * BOARD_COLS + int(label[0]), int(label[3]) * BOARD_COLS + int(label[2])


def mirror_square(square: int) -> int:
    """Rotate a square by 180 degrees."""
    return BOARD_ROWS * BOARD_COLS - 1 - int(square)


MIRRORED_ACTIONS: Tuple[int, ...] = tuple(
    move_to_action((mirror_square(move[0]), mirror_square(move[1])))
    for move in map(action_to_move, range(ACTION_SPACE_SIZE))
)


def mirror_action(action: int) -> int:
    """Rotate an action by 180 degrees (an involution)."""
    if not 0 <= int(action) < ACTION_SPACE_SIZE:
        raise ValueError(f'action must be in [0, {ACTION_SPACE_SIZE}), got {action}')
    return MIRRORED_ACTIONS[int(action)]


def move_to_uci(move: Move) -> str:
    from_square, to_square = move
    from_row, from_col = divmod(from_square, BOARD_COLS)
    to_row, to_col = divmod(to_square, BOARD_COLS)
    return f'{chr(97 + from_col)}{from_row}{chr(97 + to_col)}{to_row}'


def uci_to_move(uci: str) -> Move:
    uci = uci.strip().lower()
    if len(uci) != 4 or uci[0] not in 'abcdefghi' or uci[2] not in 'abcdefghi' or not uci[1].isdigit(
    ) or not uci[3].isdigit():
        raise ValueError(f'invalid Xiangqi coordinate move: {uci!r}')
    from_row, from_col = int(uci[1]), ord(uci[0]) - 97
    to_row, to_col = int(uci[3]), ord(uci[2]) - 97
    if not (0 <= from_row < BOARD_ROWS and 0 <= to_row < BOARD_ROWS):
        raise ValueError(f'invalid Xiangqi coordinate move: {uci!r}')
    return from_row * BOARD_COLS + from_col, to_row * BOARD_COLS + to_col

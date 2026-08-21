from zoo.board_games.chinese_chess.envs.action_encoding import (
    ACTION_LABELS,
    ACTION_SPACE_SIZE,
    action_to_move,
    mirror_action,
    move_to_action,
    move_to_uci,
    uci_to_move,
)


def test_compact_action_space_is_unique_and_bijective() -> None:
    assert len(ACTION_LABELS) == ACTION_SPACE_SIZE == 2086
    assert len(set(ACTION_LABELS)) == ACTION_SPACE_SIZE
    for action in range(ACTION_SPACE_SIZE):
        assert move_to_action(action_to_move(action)) == action


def test_action_mirror_is_an_involution() -> None:
    for action in range(ACTION_SPACE_SIZE):
        assert mirror_action(mirror_action(action)) == action


def test_iccs_round_trip() -> None:
    for uci in ('a0a9', 'h2e2', 'b9c7', 'e0e1'):
        assert move_to_uci(uci_to_move(uci)) == uci

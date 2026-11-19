"""Standalone demo regressions; no LightZero, torch or pytest dependency.

Run from the repository root:
    python -m unittest discover -s lzero/agent/tests -p 'test_mcts_tictactoe_zh.py' -v
Plotting checks are skipped when matplotlib is unavailable.
"""

import contextlib
import importlib.util
import io
import math
from functools import lru_cache
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[1] / 'mcts_tictactoe_zh.py'
SPEC = importlib.util.spec_from_file_location('mcts_tictactoe_zh', SCRIPT)
demo = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(demo)


def outcome(board):
    """Independent board oracle: return X's terminal reward, or None."""
    rows = [board[i:i + 3] for i in (0, 3, 6)]
    columns = [board[i::3] for i in range(3)]
    diagonals = [board[::4], board[2:7:2]]
    for line in rows + columns + diagonals:
        if line == ('X', ) * 3:
            return 1
        if line == ('O', ) * 3:
            return -1
    return 0 if ' ' not in board else None


@lru_cache(None)
def minimax(board, player):
    result = outcome(board)
    if result is not None:
        return result
    values = []
    for move, cell in enumerate(board):
        if cell == ' ':
            next_board = board[:move] + ('X' if player == 1 else 'O', ) + board[move + 1:]
            values.append(minimax(next_board, -player))
    return max(values) if player == 1 else min(values)


def reachable_positions():
    positions = set()
    pending = [((' ', ) * 9, 1)]
    while pending:
        board, player = pending.pop()
        if (board, player) in positions:
            continue
        positions.add((board, player))
        if outcome(board) is None:
            for move, cell in enumerate(board):
                if cell == ' ':
                    next_board = board[:move] + ('X' if player == 1 else 'O', ) + board[move + 1:]
                    pending.append((next_board, -player))
    return sorted(positions)


POSITIONS = reachable_positions()


def game_from_moves(moves):
    game = demo.Game()
    for move in moves:
        game.make_move(move)
    return game


class GameTests(unittest.TestCase):

    def test_all_reachable_states_and_transitions(self):
        self.assertEqual(len(POSITIONS), 5478)
        terminals = 0
        for board, player in POSITIONS:
            game = demo.Game()
            game.board, game.current_player = list(board), player
            result = outcome(board)
            self.assertEqual(game.is_game_over(), (result is not None, result))
            if result is not None:
                terminals += 1
                self.assertEqual(game.get_legal_moves(), [])
                self.assertIsNone(demo.mcts(game, simulations=1))
                with self.assertRaises(ValueError):
                    game.make_move(next((i for i, cell in enumerate(board) if cell == ' '), 0))
            else:
                legal = [i for i, cell in enumerate(board) if cell == ' ']
                self.assertEqual(game.get_legal_moves(), legal)
                for move in legal:
                    copy = game.clone()
                    copy.make_move(move)
                    expected = board[:move] + ('X' if player == 1 else 'O', ) + board[move + 1:]
                    self.assertEqual(tuple(copy.board), expected)
                    self.assertEqual(copy.current_player, -player)
            self.assertEqual((tuple(game.board), game.current_player), (board, player))
        self.assertEqual(terminals, 958)

    def test_invalid_moves_preserve_state(self):
        game = game_from_moves([4])
        for move in (-1, 9, 100, 4, 1.5, '1', None, True, False):
            with self.subTest(move=move), self.assertRaises(ValueError):
                game.make_move(move)
            self.assertEqual(game.board, [' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' '])
            self.assertEqual(game.current_player, -1)


class SearchTests(unittest.TestCase):

    def test_uct_uses_current_player_perspective(self):
        for player in (1, -1):
            game = demo.Game() if player == 1 else game_from_moves([4])
            root = demo.Node(game)
            root.visits = 20
            for move, value in ((0, 8), (1, -8)):
                child_game = game.clone()
                child_game.make_move(move)
                child = demo.Node(child_game, parent=root, move_from_parent=move)
                child.visits, child.value = 10, value
                root.children[move] = child
            self.assertEqual(root.best_child().move_from_parent, 0 if player == 1 else 1)

    def test_exploration_bonus_stays_positive_for_both_players(self):
        for player in (1, -1):
            game = demo.Game() if player == 1 else game_from_moves([4])
            root = demo.Node(game)
            for move, visits in ((0, 1), (1, 10)):
                child_game = game.clone()
                child_game.make_move(move)
                child = demo.Node(child_game, parent=root, move_from_parent=move)
                child.visits = visits
                root.children[move] = child
            root.visits = 11
            self.assertEqual(root.best_child().move_from_parent, 0)
            root.children[1].visits = 0
            self.assertEqual(root.best_child().move_from_parent, 1)
        self.assertIsNone(demo.Node(demo.Game()).best_child())

    def test_budget_expansion_backup_and_robust_selection(self):
        node_type = demo.Node
        for budget in (1, 9, 50):
            created = []

            def record_node(*args, **kwargs):
                node = node_type(*args, **kwargs)
                created.append(node)
                return node

            game = demo.Game()
            with mock.patch.object(demo, 'Node', side_effect=record_node):
                move = demo.mcts(game, budget, rng=random.Random(0))
            root = created[0]
            self.assertEqual(root.visits, budget)
            self.assertEqual(sum(child.visits for child in root.children.values()), budget)
            self.assertEqual(len(created), budget + 1)
            self.assertTrue(all(node.visits > 0 and abs(node.value) <= node.visits for node in created))
            self.assertEqual(root.children[move].visits, max(child.visits for child in root.children.values()))
            self.assertEqual(game.board, [' '] * 9)
            self.assertEqual(game.current_player, 1)
            if budget == 1:
                self.assertEqual(root.value, root.children[move].value)

    def test_both_players_win_and_block(self):
        cases = [([0, 3, 1, 4], 2), ([0, 3, 8, 4, 1], 5), ([0, 3, 8, 4], 5), ([0, 4, 1], 2)]
        for moves, expected in cases:
            for seed in range(5):
                with self.subTest(moves=moves, seed=seed):
                    game = game_from_moves(moves)
                    self.assertEqual(demo.mcts(game, 500, rng=random.Random(seed)), expected)

    def test_all_late_positions_against_exact_minimax(self):
        # Exhaustive tactical coverage, including opponent replies and terminal draws.
        # Fixed randomness is intentional: finite-budget MCTS is not a perfect solver.
        checked = 0
        for board, player in POSITIONS:
            if board.count(' ') > 3 or outcome(board) is not None:
                continue
            game = demo.Game()
            game.board, game.current_player = list(board), player
            move = demo.mcts(game, 200, rng=random.Random(0))
            self.assertEqual((tuple(game.board), game.current_player), (board, player))
            self.assertIn(move, game.get_legal_moves())
            game.make_move(move)
            self.assertEqual(minimax(tuple(game.board), -player), minimax(board, player), (board, player, move))
            checked += 1
        self.assertGreater(checked, 1000)

    def test_invalid_search_parameters(self):
        for budget in (0, -1, 1.5, True, None):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                demo.mcts(demo.Game(), budget)
        for c_param in (-1, math.inf, math.nan):
            with self.subTest(c_param=c_param), self.assertRaises(ValueError):
                demo.mcts(demo.Game(), c_param=c_param)


class DemoTests(unittest.TestCase):

    def test_match_accounting_for_both_seats(self):
        # Force known legal games to check winner attribution independently of MCTS.
        games = [([0, 3, 1, 4, 2], 1), ([0, 3, 8, 4, 1, 5], -1), ([0, 1, 2, 4, 3, 5, 7, 6, 8], 0)]
        for player in (1, -1):
            for moves, winner in games:
                sequence = iter(moves)
                rng = mock.Mock()
                rng.choice.side_effect = lambda choices: next(sequence)
                with mock.patch.object(demo, 'mcts', side_effect=lambda *args, **kwargs: next(sequence)):
                    counts = demo.play_match(1, num_games=1, mcts_player=player, rng=rng)
                expected = (int(winner == player), int(winner == -player), int(winner == 0))
                self.assertEqual(counts, expected)

    def test_match_reproducibility_without_global_rng_side_effects(self):
        before = random.getstate()
        for player in (1, -1):
            first = demo.play_match(30, 20, player, rng=random.Random(3))
            second = demo.play_match(30, 20, player, rng=random.Random(3))
            self.assertEqual(first, second)
            self.assertEqual(sum(first), 20)
        self.assertEqual(random.getstate(), before)

    def test_invalid_evaluation_parameters(self):
        for kwargs in ({'num_games': 0}, {'mcts_player': 0}, {'mcts_player': True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                demo.play_match(1, **kwargs)
        for counts in ([], [0], [1, -5]):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                demo.evaluate_and_plot(sim_counts=counts)

    def test_human_input_and_completed_game(self):
        game = game_from_moves([4])
        with mock.patch('builtins.input', side_effect=['abc', '0', '10', '5', '1']):
            with contextlib.redirect_stdout(io.StringIO()):
                demo.human_move(game)
        self.assertEqual(game.board[0], 'O')
        self.assertEqual(game.current_player, 1)

        def move_o(game, **kwargs):
            game.make_move(3 if game.board[3] == ' ' else 4)

        with mock.patch('builtins.input', side_effect=['1', '2', '3']), mock.patch.object(demo, 'bot_move', move_o):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                demo.play_game()
        self.assertIn('玩家（X）获胜', output.getvalue())

    def test_standard_library_only_entrypoint_and_eof(self):
        for args, input_text in ((['--help'], ''), ([], '1\n'), (['--mode', 'play'], '')):
            result = subprocess.run(
                [sys.executable, '-S', str(SCRIPT)] + args,
                input=input_text,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=10,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn('Traceback', result.stderr)
        result = subprocess.run(
            [sys.executable, str(SCRIPT), '--simulations', '0'],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
        )
        self.assertEqual(result.returncode, 2)

    @unittest.skipUnless(importlib.util.find_spec('matplotlib'), 'matplotlib is optional')
    def test_evaluation_menu_creates_png_without_display(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'performance.png'
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT), '--sim-counts', '1', '10', '--num-games', '3', '--mcts-player', 'O', '--output',
                    str(path), '--no-show'
                ],
                input='2\n',
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('MCTS 执 O', result.stdout)
            self.assertTrue(path.read_bytes().startswith(b'\x89PNG\r\n\x1a\n'))


if __name__ == '__main__':
    unittest.main()

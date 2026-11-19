"""Shared English/Chinese demo regressions; no LightZero, torch or pytest dependency.

Run from the repository root:
    python -m unittest discover -s tests -p 'test_mcts_tictactoe.py' -v
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

DEMO_DIR = Path(__file__).resolve().parents[1] / 'lzero' / 'agent'


def load_demo(filename):
    spec = importlib.util.spec_from_file_location(Path(filename).stem, DEMO_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


class GameTests:

    def test_all_reachable_states_and_transitions(self):
        self.assertEqual(len(POSITIONS), 5478)
        terminals = 0
        for board, player in POSITIONS:
            game = self.demo.Game()
            game.board, game.current_player = list(board), player
            result = outcome(board)
            self.assertEqual(game.is_game_over(), (result is not None, result))
            if result is not None:
                terminals += 1
                self.assertEqual(game.get_legal_moves(), [])
                self.assertIsNone(self.demo.mcts(game, simulations=1))
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
        game = self.game_from_moves([4])
        for move in (-1, 9, 100, 4, 1.5, '1', None, True, False):
            with self.subTest(move=move), self.assertRaises(ValueError):
                game.make_move(move)
            self.assertEqual(game.board, [' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' '])
            self.assertEqual(game.current_player, -1)


class SearchTests:

    def test_uct_uses_current_player_perspective(self):
        for player in (1, -1):
            game = self.demo.Game() if player == 1 else self.game_from_moves([4])
            root = self.demo.Node(game)
            root.visits = 20
            for move, value in ((0, 8), (1, -8)):
                child_game = game.clone()
                child_game.make_move(move)
                child = self.demo.Node(child_game, parent=root, move_from_parent=move)
                child.visits, child.value = 10, value
                root.children[move] = child
            self.assertEqual(root.best_child().move_from_parent, 0 if player == 1 else 1)

    def test_exploration_bonus_stays_positive_for_both_players(self):
        for player in (1, -1):
            game = self.demo.Game() if player == 1 else self.game_from_moves([4])
            root = self.demo.Node(game)
            for move, visits in ((0, 1), (1, 10)):
                child_game = game.clone()
                child_game.make_move(move)
                child = self.demo.Node(child_game, parent=root, move_from_parent=move)
                child.visits = visits
                root.children[move] = child
            root.visits = 11
            self.assertEqual(root.best_child().move_from_parent, 0)
            root.children[1].visits = 0
            self.assertEqual(root.best_child().move_from_parent, 1)
        self.assertIsNone(self.demo.Node(self.demo.Game()).best_child())

    def test_budget_expansion_backup_and_robust_selection(self):
        node_type = self.demo.Node
        for budget in (1, 9, 50):
            created = []

            def record_node(*args, **kwargs):
                node = node_type(*args, **kwargs)
                created.append(node)
                return node

            game = self.demo.Game()
            with mock.patch.object(self.demo, 'Node', side_effect=record_node):
                move = self.demo.mcts(game, budget, rng=random.Random(0))
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
                    game = self.game_from_moves(moves)
                    self.assertEqual(self.demo.mcts(game, 500, rng=random.Random(seed)), expected)

    def test_all_late_positions_against_exact_minimax(self):
        # Exhaustive tactical coverage, including opponent replies and terminal draws.
        # Fixed randomness is intentional: finite-budget MCTS is not a perfect solver.
        checked = 0
        for board, player in POSITIONS:
            if board.count(' ') > 3 or outcome(board) is not None:
                continue
            game = self.demo.Game()
            game.board, game.current_player = list(board), player
            move = self.demo.mcts(game, 200, rng=random.Random(0))
            self.assertEqual((tuple(game.board), game.current_player), (board, player))
            self.assertIn(move, game.get_legal_moves())
            game.make_move(move)
            self.assertEqual(minimax(tuple(game.board), -player), minimax(board, player), (board, player, move))
            checked += 1
        self.assertGreater(checked, 1000)

    def test_invalid_search_parameters(self):
        for budget in (0, -1, 1.5, True, None):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                self.demo.mcts(self.demo.Game(), budget)
        for c_param in (-1, math.inf, math.nan):
            with self.subTest(c_param=c_param), self.assertRaises(ValueError):
                self.demo.mcts(self.demo.Game(), c_param=c_param)


class DemoTests:

    def game_from_moves(self, moves):
        game = self.demo.Game()
        for move in moves:
            game.make_move(move)
        return game

    def test_match_accounting_for_both_seats(self):
        # Force known legal games to check winner attribution independently of MCTS.
        games = [([0, 3, 1, 4, 2], 1), ([0, 3, 8, 4, 1, 5], -1), ([0, 1, 2, 4, 3, 5, 7, 6, 8], 0)]
        for player in (1, -1):
            for moves, winner in games:
                sequence = iter(moves)
                rng = mock.Mock()
                rng.choice.side_effect = lambda choices: next(sequence)
                with mock.patch.object(self.demo, 'mcts', side_effect=lambda *args, **kwargs: next(sequence)):
                    counts = self.demo.play_match(1, num_games=1, mcts_player=player, rng=rng)
                expected = (int(winner == player), int(winner == -player), int(winner == 0))
                self.assertEqual(counts, expected)

    def test_match_reproducibility_without_global_rng_side_effects(self):
        before = random.getstate()
        for player in (1, -1):
            first = self.demo.play_match(30, 20, player, rng=random.Random(3))
            second = self.demo.play_match(30, 20, player, rng=random.Random(3))
            self.assertEqual(first, second)
            self.assertEqual(sum(first), 20)
        self.assertEqual(random.getstate(), before)

    def test_invalid_evaluation_parameters(self):
        for kwargs in ({'num_games': 0}, {'mcts_player': 0}, {'mcts_player': True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.demo.play_match(1, **kwargs)
        for counts in ([], [0], [1, -5]):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                self.demo.evaluate_and_plot(sim_counts=counts)

    def test_human_input_and_completed_game(self):
        game = self.game_from_moves([4])
        with mock.patch('builtins.input', side_effect=['abc', '0', '10', '5', '1']):
            with contextlib.redirect_stdout(io.StringIO()):
                self.demo.human_move(game)
        self.assertEqual(game.board[0], 'O')
        self.assertEqual(game.current_player, 1)

        def move_o(game, **kwargs):
            game.make_move(3 if game.board[3] == ' ' else 4)

        with mock.patch('builtins.input', side_effect=['1', '2', '3']), mock.patch.object(self.demo, 'bot_move',
                                                                                          move_o):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.demo.play_game()
        self.assertIn(self.messages['human_win'], output.getvalue())

    def test_standard_library_only_entrypoint_and_eof(self):
        for args, input_text in ((['--help'], ''), ([], '1\n'), (['--mode', 'play'], '')):
            result = subprocess.run(
                [sys.executable, '-S', str(self.script)] + args,
                input=input_text,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=10,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn('Traceback', result.stderr)
        result = subprocess.run(
            [sys.executable, str(self.script), '--simulations', '0'],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
        )
        self.assertEqual(result.returncode, 2)

    def test_invalid_menu_choice_reprompts_without_starting_game(self):
        result = subprocess.run(
            [sys.executable, '-S', str(self.script)],
            input='invalid\n',
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(self.messages['invalid_choice'], result.stdout)
        self.assertEqual(result.stdout.count(self.messages['choice_prompt']), 2)
        self.assertNotIn(self.messages['play_intro'], result.stdout)

    @unittest.skipUnless(importlib.util.find_spec('matplotlib'), 'matplotlib is optional')
    def test_headless_evaluation_preserves_backend_and_existing_figures(self):
        # A separate process keeps this test's initial backend out of other tests.
        code = '''
import matplotlib
matplotlib.use('svg')
import matplotlib.pyplot as plt
import runpy
import sys
demo = runpy.run_path(sys.argv[1])
fig = plt.figure()
backend = matplotlib.get_backend()
figures = plt.get_fignums()
settings = dict(matplotlib.rcParams)
rows = demo['evaluate_and_plot']([1, 5], num_games=3, output_path=sys.argv[2], show=False)
assert matplotlib.get_backend() == backend
assert plt.get_fignums() == figures
assert plt.figure(fig.number) is fig
assert dict(matplotlib.rcParams) == settings
assert len(rows) == 2
assert all(row['wins'] + row['losses'] + row['draws'] == 3 for row in rows)
plt.close(fig)
'''
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'performance.png'
            result = subprocess.run(
                [sys.executable, '-c', code, str(self.script), str(path)],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(path.read_bytes().startswith(b'\x89PNG\r\n\x1a\n'))

    @unittest.skipUnless(importlib.util.find_spec('matplotlib'), 'matplotlib is optional')
    def test_evaluation_menu_creates_png_without_display(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'performance.png'
            result = subprocess.run(
                [
                    sys.executable,
                    str(self.script), '--sim-counts', '1', '10', '--num-games', '3', '--mcts-player', 'O', '--output',
                    str(path), '--no-show'
                ],
                input='2\n',
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(self.messages['evaluation_intro'], result.stdout)
            self.assertTrue(path.read_bytes().startswith(b'\x89PNG\r\n\x1a\n'))


class ChineseDemoTests(GameTests, SearchTests, DemoTests, unittest.TestCase):
    script = DEMO_DIR / 'mcts_tictactoe_zh.py'
    demo = load_demo(script.name)
    messages = {
        'human_win': '玩家（X）获胜',
        'invalid_choice': '无效的选项',
        'choice_prompt': '请输入选项',
        'play_intro': '你执 X 先手',
        'evaluation_intro': 'MCTS 执 O',
    }


class EnglishDemoTests(GameTests, SearchTests, DemoTests, unittest.TestCase):
    script = DEMO_DIR / 'mcts_tictactoe.py'
    demo = load_demo(script.name)
    messages = {
        'human_win': 'Human (X) wins!',
        'invalid_choice': 'Invalid choice',
        'choice_prompt': 'Enter your choice',
        'play_intro': 'You play first as X',
        'evaluation_intro': 'MCTS plays O',
    }


class LanguageParityTests(unittest.TestCase):

    def test_seeded_search_and_matches_agree(self):
        english = EnglishDemoTests.demo
        chinese = ChineseDemoTests.demo
        # Sample opening, middle-game, and terminal positions for both players.
        for board, player in POSITIONS[::97]:
            for budget in (1, 10, 50):
                moves = []
                for module in (english, chinese):
                    game = module.Game()
                    game.board, game.current_player = list(board), player
                    moves.append(module.mcts(game, budget, rng=random.Random(7)))
                self.assertEqual(moves[0], moves[1], (board, player, budget))
        for player in (1, -1):
            self.assertEqual(
                english.play_match(50, 20, player, rng=random.Random(7)),
                chinese.play_match(50, 20, player, rng=random.Random(7)),
            )


if __name__ == '__main__':
    unittest.main()

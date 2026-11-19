"""Chapter 2: a pure-search Tic-Tac-Toe agent in one standalone file.

No LightZero installation is required. Run the interactive menu:
    python mcts_tictactoe.py
Play as X against MCTS as O:
    python mcts_tictactoe.py --mode play --simulations 1000 --seed 0
Evaluate MCTS (X by default) against random and save win/draw/loss curves:
    python mcts_tictactoe.py --mode evaluate --num-games 100 --seed 0 --no-show
    python mcts_tictactoe.py --mode evaluate --mcts-player O --no-show

The teaching interfaces are Game, Node, mcts, play_match, and evaluate_and_plot.
All W values use X's perspective: +1 for an X win, -1 for O, and 0 for a draw.
Selection converts Q to the player to move's perspective: X maximizes Q and O
minimizes Q, while both players receive a positive exploration bonus.
Finite search budgets and evaluation samples do not guarantee monotonically
improving win rates or prove perfect play.
"""

import argparse
import math
import random


class Game:
    """Tic-Tac-Toe: moves are indices 0-8; X is player 1 and O is player -1."""

    WINNING_LINES = (
        (0, 1, 2),
        (3, 4, 5),
        (6, 7, 8),  # Rows
        (0, 3, 6),
        (1, 4, 7),
        (2, 5, 8),  # Columns
        (0, 4, 8),
        (2, 4, 6),  # Diagonals
    )

    def __init__(self):
        self.board = [' ' for _ in range(9)]
        self.current_player = 1

    def get_current_player(self):
        return self.current_player

    def get_legal_moves(self):
        # A winning position has no legal moves even if empty cells remain.
        if self.is_game_over()[0]:
            return []
        return [i for i in range(9) if self.board[i] == ' ']

    def make_move(self, move):
        if isinstance(move, bool) or not isinstance(move, int) or not 0 <= move < 9:
            raise ValueError("A move must be an integer index from 0 to 8")
        if self.is_game_over()[0]:
            raise ValueError("The game is over; no more moves are allowed")
        if self.board[move] != ' ':
            raise ValueError(f"Invalid move: {move}; the cell is occupied")
        self.board[move] = 'X' if self.current_player == 1 else 'O'
        self.current_player *= -1

    def is_game_over(self):
        """Return (is_over, winner): 1=X, -1=O, 0=draw, None=ongoing."""
        for a, b, c in self.WINNING_LINES:
            if self.board[a] != ' ' and self.board[a] == self.board[b] == self.board[c]:
                return True, 1 if self.board[a] == 'X' else -1
        if ' ' not in self.board:
            return True, 0
        return False, None

    def clone(self):
        cloned_game = Game()
        cloned_game.board = self.board.copy()
        cloned_game.current_player = self.current_player
        return cloned_game

    def print_board(self):
        print("Current board (numbers indicate available cells):")
        cells = [mark if mark != ' ' else str(i + 1) for i, mark in enumerate(self.board)]
        print("\n---------\n".join(" | ".join(cells[i:i + 3]) for i in (0, 3, 6)))
        print()


class Node:
    """MCTS node: visits is N; value is cumulative reward W from X's perspective."""

    def __init__(self, game, parent=None, move_from_parent=None):
        self.game = game
        self.parent = parent
        self.move_from_parent = move_from_parent
        self.children = {}
        self.visits = 0
        self.value = 0.0

    def is_fully_expanded(self):
        return len(self.children) == len(self.game.get_legal_moves())

    def best_child(self, c_param=math.sqrt(2), rng=None):
        """UCT = Q from the current player's perspective + c * sqrt(ln N_parent / N_child).

        The default c is sqrt(2); there is no additional factor of 2 under
        the square root. Only exploitation changes sign on O's turn.
        Prioritize unvisited children and break ties randomly to avoid cell bias.
        """
        if not self.children:
            return None
        rng = rng if rng is not None else random
        unvisited = [child for child in self.children.values() if child.visits == 0]
        if unvisited:
            return rng.choice(unvisited)

        best_score = -math.inf
        candidates = []
        for child in self.children.values():
            exploitation = self.game.current_player * child.value / child.visits
            exploration = c_param * math.sqrt(math.log(self.visits) / child.visits)
            score = exploitation + exploration
            if score > best_score:
                best_score = score
                candidates = [child]
            elif score == best_score:
                candidates.append(child)
        return rng.choice(candidates)


def _require_positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def mcts(root_game, simulations=1000, c_param=math.sqrt(2), rng=None):
    """Run the four search steps and return a most-visited move, or None at a terminal state.

    Accept a Game, matching the chapter's interface. Search and random rollouts
    operate on clones, preserving the live board. simulations must be a
    positive integer. Supply random.Random(seed) for reproducible runs;
    otherwise use the standard library's global random number generator.
    """
    _require_positive_int(simulations, 'simulations')
    if not math.isfinite(c_param) or c_param < 0:
        raise ValueError("c_param must be finite and nonnegative")
    if root_game.is_game_over()[0]:
        return None
    rng = rng if rng is not None else random
    root_node = Node(root_game.clone())

    for _ in range(simulations):
        node = root_node
        game = root_game.clone()

        # 1. Selection: stop at a terminal node or one that is not fully expanded.
        while not game.is_game_over()[0] and node.is_fully_expanded():
            node = node.best_child(c_param=c_param, rng=rng)
            game.make_move(node.move_from_parent)

        # 2. Expansion: add one child and start evaluation from that child.
        if not game.is_game_over()[0]:
            unexpanded_moves = [move for move in game.get_legal_moves() if move not in node.children]
            move = rng.choice(unexpanded_moves)
            game.make_move(move)
            child = Node(game.clone(), parent=node, move_from_parent=move)
            node.children[move] = child
            node = child

        # 3. Simulation: both players act uniformly at random until the game ends.
        sim_game = game.clone()
        while not sim_game.is_game_over()[0]:
            sim_game.make_move(rng.choice(sim_game.get_legal_moves()))
        _, winner = sim_game.is_game_over()

        # 4. Backup: store X-relative rewards; convert perspective only during selection.
        while node is not None:
            node.visits += 1
            node.value += winner
            node = node.parent

    # A positive budget at a nonterminal root expands at least one child.
    most_visits = max(child.visits for child in root_node.children.values())
    candidates = [move for move, child in root_node.children.items() if child.visits == most_visits]
    return rng.choice(candidates)


def human_move(game):
    """Read a human move in 1-9; invalid input leaves the board unchanged."""
    if game.is_game_over()[0]:
        return
    while True:
        try:
            move = int(input("Enter your move (1-9): ")) - 1
        except ValueError:
            print("Invalid input; enter an integer from 1 to 9.")
            continue
        if move not in game.get_legal_moves():
            print("Invalid move; choose a cell that still displays a number (1-9).")
            continue
        game.make_move(move)
        return


def bot_move(game, simulations=1000, rng=None):
    move = mcts(game, simulations=simulations, rng=rng)
    if move is not None:
        game.make_move(move)
        print(f"Bot chose move: {move + 1}")


def play_game(simulations=1000, rng=None):
    """The human plays first as X; MCTS plays second as O."""
    _require_positive_int(simulations, 'simulations')
    game = Game()
    print("You play first as X; MCTS plays as O. Press Ctrl+C to exit.")
    game.print_board()
    while not game.is_game_over()[0]:
        if game.current_player == 1:
            human_move(game)
        else:
            bot_move(game, simulations=simulations, rng=rng)
        game.print_board()
    _, winner = game.is_game_over()
    print({1: "Human (X) wins!", -1: "MCTS Bot (O) wins!", 0: "It's a draw!"}[winner])


def play_match(mcts_simulations, num_games=100, mcts_player=1, rng=None):
    """Play MCTS against random and return (MCTS wins, random wins, draws).

    MCTS plays first as X by default, matching the chapter's evaluation.
    Use mcts_player=-1 to evaluate MCTS separately as the second player.
    """
    _require_positive_int(mcts_simulations, 'mcts_simulations')
    _require_positive_int(num_games, 'num_games')
    if isinstance(mcts_player, bool) or mcts_player not in (1, -1):
        raise ValueError("mcts_player must be 1 (X) or -1 (O)")
    rng = rng if rng is not None else random
    mcts_wins = random_wins = draws = 0
    for _ in range(num_games):
        game = Game()
        while not game.is_game_over()[0]:
            if game.current_player == mcts_player:
                move = mcts(game, simulations=mcts_simulations, rng=rng)
            else:
                move = rng.choice(game.get_legal_moves())
            game.make_move(move)
        _, winner = game.is_game_over()
        if winner == mcts_player:
            mcts_wins += 1
        elif winner == -mcts_player:
            random_wins += 1
        else:
            draws += 1
    return mcts_wins, random_wins, draws


def evaluate_and_plot(
    sim_counts=None,
    num_games=100,
    mcts_player=1,
    seed=0,
    output_path='mcts_performance.png',
    show=True,
):
    """Evaluate search budgets, save win/draw/loss curves, and return raw counts.

    Use show=False in a headless environment. Curves show sample frequencies;
    observing zero losses does not establish perfect play.
    """
    sim_counts = [1, 5, 10, 20, 50, 100, 200, 500] if sim_counts is None else list(sim_counts)
    if not sim_counts:
        raise ValueError("sim_counts must not be empty")
    for count in sim_counts:
        _require_positive_int(count, 'each simulation count')
    _require_positive_int(num_games, 'num_games')
    if isinstance(mcts_player, bool) or mcts_player not in (1, -1):
        raise ValueError("mcts_player must be 1 (X) or -1 (O)")

    import matplotlib
    if show:
        import matplotlib.pyplot as plt
    else:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

    player = 'X' if mcts_player == 1 else 'O'
    print(
        f"MCTS plays {player} ({'first' if mcts_player == 1 else 'second'}), {num_games} games per budget, seed={seed}"
    )
    print("Simulations\tMCTS win\tDraw\tRandom win")
    results = []
    for simulations in sim_counts:
        # Restart from the supplied seed so results do not depend on budget ordering.
        wins, losses, draws = play_match(simulations, num_games, mcts_player, rng=random.Random(seed))
        results.append(dict(simulations=simulations, wins=wins, losses=losses, draws=draws))
        print(f"{simulations}\t\t{wins / num_games:.1%}\t\t{draws / num_games:.1%}\t{losses / num_games:.1%}")

    # Keep plotting configuration local to this evaluation.
    with matplotlib.rc_context():
        if show:
            fig, ax = plt.subplots(figsize=(9, 6))
        else:
            # A standalone canvas preserves the caller's backend and existing figures.
            fig = Figure(figsize=(9, 6))
            FigureCanvasAgg(fig)
            ax = fig.subplots()
        try:
            labels = ('MCTS win', 'Draw', 'Random win')
            for key, label in zip(('wins', 'draws', 'losses'), labels):
                ax.plot(sim_counts, [row[key] / num_games for row in results], marker='o', label=label)
            ax.set_xscale('log')
            ax.set_xticks(sim_counts)
            ax.set_xticklabels([str(count) for count in sim_counts])
            ax.set_ylim(0, 1.05)
            ax.set_xlabel('Simulations per move (log scale)')
            ax.set_ylabel('Fraction of games')
            ax.set_title(f'MCTS ({player}) vs random, {num_games} games per budget')
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(output_path, dpi=200)
            print(f"Performance plot saved to: {output_path}")
            print("Note: sample win rates may fluctuate; zero observed losses do not prove perfect play.")
            if show:
                plt.show()
        finally:
            if show:
                plt.close(fig)
    return results


def _positive_int_arg(text):
    try:
        value = int(text)
        _require_positive_int(value, 'argument')
        return value
    except ValueError as error:
        raise argparse.ArgumentTypeError('Enter a positive integer') from error


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--mode', choices=('play', 'evaluate'), help='Show the interactive menu when omitted')
    parser.add_argument('--simulations', type=_positive_int_arg, default=1000, help='Simulations per move in play mode')
    parser.add_argument('--sim-counts', type=_positive_int_arg, nargs='+', help='Search budgets to evaluate')
    parser.add_argument('--num-games', type=_positive_int_arg, default=100, help='Games per evaluation budget')
    parser.add_argument('--mcts-player', choices=('X', 'O'), default='X', help='MCTS side in evaluation mode')
    parser.add_argument('--seed', type=int, default=0, help='Random seed (default: 0)')
    parser.add_argument('--output', default='mcts_performance.png', help='Output path for the performance plot')
    parser.add_argument('--no-show', action='store_true', help='Save the plot without opening a window')
    args = parser.parse_args()
    try:
        mode = args.mode
        if mode is None:
            print("Choose a mode:\n1. Play against MCTS\n2. Evaluate MCTS")
            while True:
                choice = input("Enter your choice (1 or 2, default: 1): ").strip()
                if choice in ('', '1', '2'):
                    break
                print("Invalid choice; enter 1 or 2.")
            mode = 'evaluate' if choice == '2' else 'play'
        if mode == 'play':
            play_game(simulations=args.simulations, rng=random.Random(args.seed))
        else:
            evaluate_and_plot(
                sim_counts=args.sim_counts,
                num_games=args.num_games,
                mcts_player=1 if args.mcts_player == 'X' else -1,
                seed=args.seed,
                output_path=args.output,
                show=not args.no_show,
            )
    except (EOFError, KeyboardInterrupt):
        print("\nExited.")


if __name__ == '__main__':
    main()

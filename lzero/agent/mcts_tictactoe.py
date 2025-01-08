import math
import random

# Game class representing the state of Tic-Tac-Toe
class Game:
    def __init__(self):
        # Initialize the board using a list of 9 cells, initially empty
        self.board = [' ' for _ in range(9)]
        # Current player: 1 represents Player 1 (X), -1 represents Player 2 (O)
        self.current_player = 1

    def get_current_player(self):
        # Return the current player
        return self.current_player

    def get_legal_moves(self):
        # Return all legal moves, i.e., the indices of empty cells on the board
        return [i for i in range(9) if self.board[i] == ' ']

    def make_move(self, move):
        # Make a move; raise an exception if the target cell is not empty
        if self.board[move] != ' ':
            raise ValueError("Invalid move")
        # Mark the cell based on the current player
        self.board[move] = 'X' if self.current_player == 1 else 'O'
        # Switch the player
        self.current_player *= -1

    def is_game_over(self):
        # Define all possible winning lines
        lines = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
            [0, 4, 8], [2, 4, 6]              # Diagonals
        ]
        # Check if any player has won
        for line in lines:
            a, b, c = line
            if self.board[a] == self.board[b] == self.board[c] and self.board[a] != ' ':
                return True, self.board[a]  # Return game over and the winner
        # Check for a draw
        if ' ' not in self.board:
            return True, 0  # Draw
        # Game is not over
        return False, None

    def clone(self):
        # Clone the current game state for simulation
        cloned_game = Game()
        cloned_game.board = self.board.copy()
        cloned_game.current_player = self.current_player
        return cloned_game

    def print_board(self):
        # Print the current state of the board
        print("Current board state:")
        print(f"{self.board[0]} | {self.board[1]} | {self.board[2]}")
        print("---------")
        print(f"{self.board[3]} | {self.board[4]} | {self.board[5]}")
        print("---------")
        print(f"{self.board[6]} | {self.board[7]} | {self.board[8]}")
        print()

# Node class for the MCTS tree structure
class Node:
    def __init__(self, game, parent=None):
        self.game = game          # Current game state
        self.parent = parent      # Parent node
        self.children = {}        # Child nodes, key is the move, value is the node
        self.visits = 0           # Number of visits to this node
        self.value = 0.0          # Accumulated reward value

    # Strategy for selecting child nodes (using the UCB1 formula)
    def select_child(self):
        best_score = -float('inf')
        best_move = None
        best_child = None
        for move, child in self.children.items():
            if child.visits == 0:
                score = float('inf')  # Prioritize unvisited nodes
            else:
                exploitation = child.value / child.visits  # Exploitation term
                exploration = math.sqrt(2 * math.log(self.visits) / child.visits)  # Exploration term
                score = exploitation + exploration
            if score > best_score:
                best_score = score
                best_move = move
                best_child = child
        return best_move, best_child

    # Expand all possible child nodes for this node
    def expand(self, game):
        legal_moves = game.get_legal_moves()
        for move in legal_moves:
            new_game = game.clone()
            new_game.make_move(move)
            child_node = Node(new_game, parent=self)
            self.children[move] = child_node

    # Simulate the game until it ends, returning the game result
    def simulate(self):
        game = self.game.clone()
        while True:
            is_over, result = game.is_game_over()
            if is_over:
                break
            legal_moves = game.get_legal_moves()
            move = random.choice(legal_moves)  # Randomly choose a move
            game.make_move(move)
        return result  # Return 'X', 'O', or 0

# MCTS algorithm implementation
def mcts(root_node, simulations=1000):
    for _ in range(simulations):
        node = root_node
        game = node.game.clone()
        # Selection phase
        while node.children and not game.is_game_over()[0]:
            move, node = node.select_child()
            game.make_move(move)
        # Expansion phase
        if not node.children and not game.is_game_over()[0]:
            node.expand(game)
        # Simulation phase
        if not game.is_game_over()[0]:
            result = node.simulate()
        else:
            _, result = game.is_game_over()
        # Backpropagation phase
        while node:
            node.visits += 1
            if result == 'X':
                node.value += 1.0 if node.game.current_player == -1 else -1.0
            elif result == 'O':
                node.value += -1.0 if node.game.current_player == -1 else 1.0
            else:
                node.value += 0.0  # Draw
            node = node.parent
    # Choose the move with the most visits as the best move
    best_move = max(root_node.children.keys(), key=lambda move: root_node.children[move].visits)
    return best_move

# Human player move input
def human_move(game):
    while True:
        try:
            move_input = input("Enter your move (1-9): ")
            move = int(move_input) - 1  # Convert to index
            if move not in game.get_legal_moves():
                print("Invalid move, please try again.")
            else:
                game.make_move(move)
                break
        except ValueError:
            print("Invalid input, please enter a number.")

# Bot player move (uses MCTS)
def bot_move(game):
    root_node = Node(game.clone())
    best_move = mcts(root_node, simulations=50)  # Adjust simulations for performance
    game.make_move(best_move)
    print(f"Bot chose move: {best_move + 1}")

# Main function: game loop
def main():
    game = Game()
    game.print_board()

    while not game.is_game_over()[0]:
        if game.get_current_player() == 1:
            human_move(game)  # Player 1 (X) move
        else:
            bot_move(game)    # Player 2 (O) move
        game.print_board()
        is_over, result = game.is_game_over()
        if is_over:
            if result == 'X':
                print("Player 1 (X) wins!")
            elif result == 'O':
                print("Player 2 (O) wins!")
            else:
                print("It's a draw!")
            break

# Run the main function
if __name__ == "__main__":
    """
    This file is a simple implementation of a Tic-Tac-Toe game, designed for educational purposes.
    Features:
    - Player 1 (X) competes against a bot (O) powered by Monte Carlo Tree Search (MCTS).
    - The game is played via command-line interaction, with the player providing inputs for their moves.
    - The bot uses the MCTS algorithm to determine the best moves.
    - Demonstrates the basic principles of MCTS: selection, expansion, simulation, and backpropagation.
    """
    main()
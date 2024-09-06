"""
Adapt the Chess environment in PettingZoo (https://github.com/Farama-Foundation/PettingZoo/blob/master/pettingzoo/classic/chess/chess.py) to the BaseEnv interface.
"""

import sys
from collections import defaultdict
from os import path

import chess
import gymnasium
import numpy as np
import pygame
from ding.envs import BaseEnv, BaseEnvTimestep
from ding.utils import ENV_REGISTRY
from gymnasium import spaces
from pettingzoo.classic.chess import chess_utils
from pettingzoo.utils.agent_selector import agent_selector


@ENV_REGISTRY.register('Chess')
class ChessEnv(BaseEnv):
    metadata = {
        "render_modes": ["human", "ansi", "rgb_array"],
        "name": "chess_v6",
        "is_parallelizable": False,
        "render_fps": 2,
    }

    def __init__(self, cfg={}):
        self.cfg = cfg
        self.current_player_index = 0
        self.next_player_index = 1

        self.board = chess.Board()

        self.agents = [f"player_{i + 1}" for i in range(2)]
        self.possible_agents = self.agents[:]

        self._agent_selector = agent_selector(self.agents)

        # Define action and observation spaces
        self._action_spaces = {name: spaces.Discrete(8 * 8 * 73) for name in self.agents}
        self._observation_spaces = {
            name: spaces.Dict(
                {
                    'observation': spaces.Box(low=0, high=1, shape=(8, 8, 111), dtype=bool),
                    'action_mask': spaces.Box(low=0, high=1, shape=(4672,), dtype=np.int8)
                }
            )
            for name in self.agents
        }

        self.rewards = None
        self.dones = None
        self.infos = {name: {} for name in self.agents}

        self.agent_selection = None

        self.board_history = np.zeros((8, 8, 104), dtype=bool)

        self.render_mode = self.cfg.get("render_mode", None)
        self.screen_height = self.screen_width = self.cfg.get("screen_size", 800)

        assert self.render_mode is None or self.render_mode in self.metadata["render_modes"]

        self.screen = None

        if self.render_mode in ["human", "rgb_array"]:
            self.BOARD_SIZE = (self.screen_width, self.screen_height)
            self.clock = pygame.time.Clock()
            self.cell_size = (self.BOARD_SIZE[0] / 8, self.BOARD_SIZE[1] / 8)

            bg_name = path.join(path.dirname(__file__), "img/chessboard.png")
            self.bg_image = pygame.transform.scale(
                pygame.image.load(bg_name), self.BOARD_SIZE
            )

            def load_piece(file_name):
                img_path = path.join(path.dirname(__file__), f"img/{file_name}.png")
                return pygame.transform.scale(
                    pygame.image.load(img_path), self.cell_size
                )

            self.piece_images = {
                "pawn": [load_piece("pawn_black"), load_piece("pawn_white")],
                "knight": [load_piece("knight_black"), load_piece("knight_white")],
                "bishop": [load_piece("bishop_black"), load_piece("bishop_white")],
                "rook": [load_piece("rook_black"), load_piece("rook_white")],
                "queen": [load_piece("queen_black"), load_piece("queen_white")],
                "king": [load_piece("king_black"), load_piece("king_white")],
            }

        self.transposition_table = defaultdict(dict)

    @property
    def current_player(self):
        return self.current_player_index

    def to_play(self):
        return self.next_player_index

    def to_play_str(self):
        return 'white' if self.next_player_index==0 else 'black'

    def reset(self):
        self.has_reset = True
        self.agents = self.possible_agents[:]
        self.board = chess.Board()

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()

        self.rewards = {name: 0 for name in self.agents}
        self._cumulative_rewards = {name: 0 for name in self.agents}
        self.dones = {name: False for name in self.agents}
        self.infos = {name: {} for name in self.agents}

        self.board_history = np.zeros((8, 8, 104), dtype=bool)
        self.current_player_index = 0

        agent = self.agent_selection
        current_index = self.agents.index(agent)
        self.current_player_index = current_index
        obs = self.observe(agent)
        self.transposition_table = defaultdict(dict)

        return obs

    def observe(self, agent):
        current_index = self.possible_agents.index(agent)

        observation = chess_utils.get_observation(self.board, current_index)
        observation = np.dstack((observation[:, :, :7], self.board_history))
        # We need to swap the white 6 channels with black 6 channels
        if current_index == 1:
            # 1. Mirror the board
            observation = np.flip(observation, axis=0)
            # 2. Swap the white 6 channels with the black 6 channels
            for i in range(1, 9):
                tmp = observation[..., 13 * i - 6 : 13 * i].copy()
                observation[..., 13 * i - 6 : 13 * i] = observation[
                    ..., 13 * i : 13 * i + 6
                ]
                observation[..., 13 * i : 13 * i + 6] = tmp
        legal_moves = (
            chess_utils.legal_moves(self.board) if agent == self.agent_selection else []
        )

        action_mask = np.zeros(4672, "int8")
        for i in legal_moves:
            action_mask[i] = 1
        return {'observation': observation, 'action_mask': action_mask}

    def set_game_result(self, result_val):
        # Set the game result and update rewards, dones, and infos
        for i, name in enumerate(self.agents):
            self.dones[name] = True
            result_coef = 1 if i == 0 else -1
            self.rewards[name] = result_val * result_coef
            self.infos[name] = {'legal_moves': []}

    def step(self, action):
        if self.dones[self.agent_selection]:
            return self._was_done_step(action)

        current_agent = self.agent_selection
        current_index = self.agents.index(current_agent)

        self.current_player_index = current_index

        # Update board history
        next_board = chess_utils.get_observation(self.board, current_agent)
        self.board_history = np.dstack((next_board[:, :, 7:], self.board_history[:, :, :-13]))

        # Execute the chosen action
        chosen_move = chess_utils.action_to_move(self.board, action, current_index)
        assert chosen_move in self.board.legal_moves
        self.board.push(chosen_move)

        # Check game termination conditions
        next_legal_moves = chess_utils.legal_moves(self.board)
        is_stale_or_checkmate = not any(next_legal_moves)
        is_repetition = self.board.is_repetition(3)
        is_50_move_rule = self.board.can_claim_fifty_moves()
        is_claimable_draw = is_repetition or is_50_move_rule
        game_over = is_claimable_draw or is_stale_or_checkmate

        if game_over:
            result = self.board.result(claim_draw=True)
            result_val = chess_utils.result_to_int(result)
            self.set_game_result(result_val)

        # Update cumulative rewards
        for agent, reward in self.rewards.items():
            self._cumulative_rewards[agent] += reward

        # Select the next agent
        self.agent_selection = self._agent_selector.next()
        agent = self.agent_selection
        self.next_player_index = self.agents.index(agent)

        observation = self.observe(agent)

        return BaseEnvTimestep(observation, self._cumulative_rewards[agent], self.dones[agent], self.infos[agent])

    @property
    def legal_actions(self):
        return chess_utils.legal_moves(self.board)

    def legal_moves(self):
        # Get the legal moves for the current board state
        legal_moves = chess_utils.legal_moves(self.board)
        return legal_moves

    def random_action(self):
        # Choose a random action from the legal moves
        action_list = self.legal_moves()
        return np.random.choice(action_list)

    def human_to_action(self, interactive=True):
        """
        Overview:
            This method allows the user to input a legal action, supporting both UCI format strings and integer indices.
            It returns the corresponding action index in the action space.

        Args:
            interactive (bool): If True, the method will prompt the user for input.
                                If False, it will automatically select the first available action.

        Returns:
            int: An integer representing the chosen action from the action space.
        """
        while True:
            try:
                # Print the current available legal moves for the current player.
                print(f"Current available actions for player {self.to_play_str()} are: {chess_utils.legal_moves(self.board)}")
                print(f"Current legal uci move is: {list(self.board.legal_moves)}")

                if interactive:
                    # Prompt the user to input the next move in either UCI string format or as an index.
                    choice = input(f"Enter the next move for player {self.to_play_str()} (UCI format or index): ").strip()

                    # If the input is a digit, assume it is the action index.
                    if choice.isdigit():
                        action = int(choice)
                        # Check if the action is a legal move.
                        if action in self.legal_actions:
                            return action
                    else:
                        # If the input is not a digit, assume it is a UCI string.
                        # Convert the UCI string to a chess.Move object.
                        move = chess.Move.from_uci(choice)

                        # Check if the move is legal in the current board state.
                        if move in self.board.legal_moves:
                            action_index = self.get_action_index(move, self.board)
                            return action_index
                        else:
                            # If the UCI move is not valid, prompt the user to try again.
                            print("Invalid UCI move, please try again.")
                else:
                    # If not in interactive mode, automatically select the first available legal action.
                    return self.legal_actions[0]
            except KeyboardInterrupt:
                # Handle user interruption (e.g., Ctrl+C).
                sys.exit(0)
            except Exception as e:
                # Handle any other exceptions, prompt the user to try again.
                print(f"Invalid input, please try again: {e}")

    def get_action_index(self, move, board):
        # Get all legal moves
        legal_moves = list(board.legal_moves)

        # Get the index of the move in the legal_moves list
        move_index = legal_moves.index(move)

        # Get the action index from the legal_moves list
        action_index = chess_utils.legal_moves(board)[move_index]

        return action_index

    def render(self, mode='ansi'):
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
        elif self.render_mode == "ansi":
            print(self.board)
            return str(self.board)
        elif self.render_mode in {"human", "rgb_array"}:
            return self._render_gui()
        else:
            raise ValueError(
                f"{self.render_mode} is not a valid render mode. Available modes are: {self.metadata['render_modes']}"
            )

    def _render_gui(self):
        if self.screen is None:
            pygame.init()

            if self.render_mode == "human":
                pygame.display.set_caption("Chess")
                self.screen = pygame.display.set_mode(self.BOARD_SIZE)
            elif self.render_mode == "rgb_array":
                self.screen = pygame.Surface(self.BOARD_SIZE)

        self.screen.blit(self.bg_image, (0, 0))
        for square, piece in self.board.piece_map().items():
            pos_x = square % 8 * self.cell_size[0]
            pos_y = (
                self.BOARD_SIZE[1] - (square // 8 + 1) * self.cell_size[1]
            )  # offset because pygame display is flipped
            piece_name = chess.piece_name(piece.piece_type)
            piece_img = self.piece_images[piece_name][piece.color]
            self.screen.blit(piece_img, (pos_x, pos_y))

        if self.render_mode == "human":
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    @property
    def observation_space(self):
        return self._observation_spaces

    @property
    def action_space(self):
        return self._action_spaces

    @property
    def reward_space(self):
        return self._reward_space

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def close(self) -> None:
        pass

    def bot_action(self):
        # Get all legal moves
        legal_moves = list(self.board.legal_moves)
        legal_actions = chess_utils.legal_moves(self.board)

        # Create a temporary board to simulate moves
        temp_board = self.board.copy()

        best_action = None
        best_score = -np.inf

        # Evaluate each legal move
        for action, move in zip(legal_actions, legal_moves):
            # Simulate the move
            temp_board.push(move)

            # Check transposition table for existing evaluation
            key = temp_board.fen()
            if key in self.transposition_table:
                score = self.transposition_table[key]
            else:
                # Evaluate the board after the move
                score = self.evaluate_board(temp_board)
                self.transposition_table[key] = score

            # Update best_action if the current move has a higher score
            if score > best_score:
                best_action = action
                best_score = score

            # Undo the move to return to the original state
            temp_board.pop()

        # Return the action with the highest score
        return best_action

    def evaluate_board(self, board):
        # Evaluate the board based on rules

        # Check if the game is over
        outcome = board.outcome()
        if outcome is not None:
            # If the game is over, return a definitive score
            if outcome.winner == chess.WHITE:
                return 10000  # White wins
            elif outcome.winner == chess.BLACK:
                return -10000  # Black wins
            else:
                return 0  # Draw

        # If the game is not over, calculate a score based on various factors
        score = 0

        # 1. Assign points based on the value of pieces
        # Piece values: Queen 9, Rook 5, Knight 3, Bishop 3, Pawn 1
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0
        }
        for piece_type, value in piece_values.items():
            score += len(board.pieces(piece_type, chess.WHITE)) * value
            score -= len(board.pieces(piece_type, chess.BLACK)) * value

        # 2. Pawn positions are also valued; pawns closer to the opponent's back rank are more valuable
        # Pawn position value table
        pawn_position_scores = [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [5, 5, 5, 5, 5, 5, 5, 5],
            [1, 1, 2, 3, 3, 2, 1, 1],
            [0.5, 0.5, 1, 2.5, 2.5, 1, 0.5, 0.5],
            [0, 0, 0, 2, 2, 0, 0, 0],
            [0.5, -0.5, -1, 0, 0, -1, -0.5, 0.5],
            [0.5, 1, 1, -2, -2, 1, 1, 0.5],
            [0, 0, 0, 0, 0, 0, 0, 0]
        ]
        for pawn_square in board.pieces(chess.PAWN, chess.WHITE):
            score += pawn_position_scores[chess.square_rank(pawn_square)][chess.square_file(pawn_square)]
        for pawn_square in board.pieces(chess.PAWN, chess.BLACK):
            score -= pawn_position_scores[7 - chess.square_rank(pawn_square)][chess.square_file(pawn_square)]

        # 3. Add points for rooks on open files
        score += 3 * self.count_open_files(board, chess.WHITE)
        score -= 3 * self.count_open_files(board, chess.BLACK)

        # 4. Knights on favorable positions
        knight_position_scores = [
            [-5, -4, -3, -3, -3, -3, -4, -5],
            [-4, -2, 0, 0, 0, 0, -2, -4],
            [-3, 0, 1, 1.5, 1.5, 1, 0, -3],
            [-3, 0.5, 1.5, 2, 2, 1.5, 0.5, -3],
            [-3, 0, 1.5, 2, 2, 1.5, 0, -3],
            [-3, 0.5, 1, 1.5, 1.5, 1, 0.5, -3],
            [-4, -2, 0, 0.5, 0.5, 0, -2, -4],
            [-5, -4, -3, -3, -3, -3, -4, -5]
        ]
        for knight_square in board.pieces(chess.KNIGHT, chess.WHITE):
            score += knight_position_scores[chess.square_rank(knight_square)][chess.square_file(knight_square)]
        for knight_square in board.pieces(chess.KNIGHT, chess.BLACK):
            score -= knight_position_scores[7 - chess.square_rank(knight_square)][chess.square_file(knight_square)]

        # 5. Bonus for having both bishops
        if len(board.pieces(chess.BISHOP, chess.WHITE)) == 2:
            score += 3
        if len(board.pieces(chess.BISHOP, chess.BLACK)) == 2:
            score -= 3

        # 6. Control of the center
        center_squares = [chess.D4, chess.E4, chess.D5, chess.E5]
        for square in center_squares:
            if board.is_attacked_by(chess.WHITE, square):
                score += 1
            if board.is_attacked_by(chess.BLACK, square):
                score -= 1

        # Additional rules can be added here, such as:
        # - Specific endgame scores
        # - King safety
        # - Penalties for isolated pawns
        # ...

        # Finally, return the evaluated score
        # Note that this score is not absolute and can be adjusted according to your understanding
        return score

    def count_open_files(self, board, color):
        # Count the number of open files occupied by the rooks of a given color
        open_files = 0
        for file_idx in range(8):
            if (not board.pieces(chess.PAWN, color) & chess.SquareSet(chess.BB_FILES[file_idx])
                    and board.pieces(chess.ROOK, color) & chess.SquareSet(chess.BB_FILES[file_idx])):
                open_files += 1
        return open_files

    def __repr__(self) -> str:
        return "LightZero Base Chess Env"
"""
Adapt the Chess environment in PettingZoo (https://github.com/Farama-Foundation/PettingZoo) to the BaseEnv interface.
"""

import sys
import chess
import numpy as np
from ding.envs import BaseEnv, BaseEnvTimestep
from ding.utils import ENV_REGISTRY
from gymnasium import spaces
from pettingzoo.classic.chess import chess_utils
from pettingzoo.utils.agent_selector import agent_selector


@ENV_REGISTRY.register('Chess')
class ChessEnv(BaseEnv):
    def __init__(self, cfg=None):
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

    @property
    def current_player(self):
        return self.current_player_index

    def to_play(self):
        return self.next_player_index

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
        return obs

    def observe(self, agent):
        # Get the observation for the current agent
        observation = chess_utils.get_observation(self.board, self.possible_agents.index(agent))
        observation = np.dstack((observation[:, :, :7], self.board_history))
        action_mask = self.legal_actions
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
        # Get the legal action mask
        action_mask = np.zeros(4672, 'uint8')
        action_mask[chess_utils.legal_moves(self.board)] = 1
        return action_mask

    def legal_moves(self):
        # Get the legal moves for the current board state
        legal_moves = chess_utils.legal_moves(self.board)
        return legal_moves

    def random_action(self):
        # Choose a random action from the legal moves
        action_list = self.legal_moves()
        return np.random.choice(action_list)

    def human_to_action(self):
        """
        Overview:
            For multiplayer games, ask the user for a legal action
            and return the corresponding action number.
        Returns:
            An integer from the action space.
        """
        while True:
            try:
                print(f"Current available actions for the player {self.to_play()} are:{self.legal_moves()}")
                choice = int(input(f"Enter the index of next move for the player {self.to_play()}: "))
                if choice in self.legal_moves():
                    break
            except KeyboardInterrupt:
                sys.exit(0)
            except Exception as e:
                print("Wrong input, try again")
        return choice

    def render(self, mode='human'):
        # Print the current board state
        print(self.board)

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

            # Evaluate the board after the move
            score = self.evaluate_board(temp_board)

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

        # Additional rules can be added here, such as:
        # - Knight position value
        # - Control of the center
        # - Bonus for having both bishops
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
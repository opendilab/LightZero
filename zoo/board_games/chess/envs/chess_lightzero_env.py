"""
Adapt the Chess environment in PettingZoo (https://github.com/Farama-Foundation/PettingZoo) to the BaseEnv interface.
"""

import copy
import logging
import os
import sys
from datetime import datetime
from typing import List

import chess
import numpy as np
from ding.envs.env.base_env import BaseEnv, BaseEnvTimestep
from ding.utils.registry_factory import ENV_REGISTRY
from gymnasium import spaces
from pettingzoo.classic.chess import chess_utils


@ENV_REGISTRY.register('chess_lightzero')
class ChessLightZeroEnv(BaseEnv):
    def __init__(self, cfg=None):
        self.cfg = cfg
        self.board_size = 8
        self.players = [1, 2]
        self.channel_last = cfg.channel_last
        self.scale = cfg.scale
        self.battle_mode = cfg.battle_mode
        assert self.battle_mode in ['self_play_mode', 'play_with_bot_mode', 'eval_mode']
        self.battle_mode_in_simulation_env = 'self_play_mode'
        self.prob_random_agent = cfg.prob_random_agent
        self.prob_expert_agent = cfg.prob_expert_agent
        assert (self.prob_random_agent >= 0 and self.prob_expert_agent == 0) or (
                self.prob_random_agent == 0 and self.prob_expert_agent >= 0), \
            f'self.prob_random_agent:{self.prob_random_agent}, self.prob_expert_agent:{self.prob_expert_agent}'
        self._env = self
        self.agent_vs_human = cfg.agent_vs_human
        self.alphazero_mcts_ctree = cfg.alphazero_mcts_ctree
        self._replay_path = cfg.replay_path if hasattr(cfg, "replay_path") and cfg.replay_path is not None else None
        self._save_replay_count = 0
        self._observation_space = None
        self._action_space = None

    @property
    def legal_actions(self):
        return chess_utils.legal_moves(self.board)

    def observe(self, agent_index):
        # Get observation for the specified agent.
        try:
            observation = chess_utils.get_observation(self.board, agent_index).astype(float)  # TODO
        except Exception as e:
            print('debug')

        action_mask = np.zeros(4672, dtype=np.int8)
        action_mask[chess_utils.legal_moves(self.board)] = 1
        return {'observation': observation, 'action_mask': action_mask}

    def current_state(self):
        """
        Overview:
            Get the current state from the player's perspective.
            self.board is a Board object from the python-chess library, used to represent the state of the board.
        """
        # TODO: more efficient observation
        return None, self.observe(self.current_player_index)['observation']

    def get_done_winner(self):
        """
        Overview:
            Check if the game is over and determine the winning player. Returns 'done' and 'winner'.
        Returns:
            - outputs (:obj:`Tuple`): A tuple containing 'done' and 'winner'
                - If player 1 wins, 'done' = True, 'winner' = 1
                - If player 2 wins, 'done' = True, 'winner' = 2
                - If it's a draw, 'done' = True, 'winner' = -1
                - If the game is not over, 'done' = False, 'winner' = -1
        """
        done = self.board.is_game_over()
        result = self.board.result(claim_draw=True)
        if result == "*":
            winner = -1
        else:
            winner = chess_utils.result_to_int(result)

        if not done:
            winner = -1

        return done, winner

    def reset(self, start_player_index=0, init_state=None, katago_policy_init=False, katago_game_state=None):
        if self.alphazero_mcts_ctree and init_state is not None:
            # Convert byte string to np.ndarray
            init_state = np.frombuffer(init_state, dtype=np.int32)

        if self.scale:
            self._observation_space = spaces.Dict(
                {
                    'observation': spaces.Box(low=0, high=1, shape=(8, 8, 20), dtype=np.float32),
                    'action_mask': spaces.Box(low=0, high=1, shape=(4672,), dtype=np.float32)
                }
            )
        else:
            self._observation_space = spaces.Dict(
                {
                    'observation': spaces.Box(low=0, high=1, shape=(8, 8, 20), dtype=bool),
                    'action_mask': spaces.Box(low=0, high=1, shape=(4672,), dtype=np.int8)
                }
            )
        self._action_space = spaces.Discrete(8 * 8 * 73)
        self._reward_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.start_player_index = start_player_index
        self._current_player = self.players[self.start_player_index]
        if init_state is not None:
            self.board = chess.Board(init_state)
        else:
            self.board = chess.Board()

        action_mask = np.zeros(4672, dtype=np.int8)
        action_mask[chess_utils.legal_moves(self.board)] = 1

        if self.battle_mode == 'play_with_bot_mode' or self.battle_mode == 'eval_mode':
            obs = {
                'observation': self.observe(self.current_player_index)['observation'],
                'action_mask': action_mask,
                'board': self.board.fen(),
                'current_player_index': self.start_player_index,
                'to_play': -1
            }
        elif self.battle_mode == 'self_play_mode':
            obs = {
                'observation': self.observe(self.current_player_index)['observation'],
                'action_mask': action_mask,
                'board': self.board.fen(),
                'current_player_index': self.start_player_index,
                'to_play': self.current_player
            }
        if self._replay_path is not None:
            self._frames = []

        return obs

    def step(self, action):
        if self.battle_mode == 'self_play_mode':
            if self.prob_random_agent > 0:
                if np.random.rand() < self.prob_random_agent:
                    action = self.random_action()
            elif self.prob_expert_agent > 0:
                if np.random.rand() < self.prob_expert_agent:
                    action = self.bot_action()

            timestep = self._player_step(action)
            if timestep.done:
                # The eval_episode_return is calculated from Player 1's perspective。
                timestep.info['eval_episode_return'] = -timestep.reward if timestep.obs[
                                                                               'to_play'] == 1 else timestep.reward
            return timestep
        elif self.battle_mode == 'play_with_bot_mode':
            timestep_player1 = self._player_step(action)
            if timestep_player1.done:
                timestep_player1.obs['to_play'] = -1
                return timestep_player1

            bot_action = self.bot_action()
            timestep_player2 = self._player_step(bot_action)

            timestep_player2.info['eval_episode_return'] = -timestep_player2.reward
            timestep_player2 = timestep_player2._replace(reward=-timestep_player2.reward)

            timestep = timestep_player2
            timestep.obs['to_play'] = -1

            return timestep
        elif self.battle_mode == 'eval_mode':
            timestep_player1 = self._player_step(action)
            if timestep_player1.done:
                timestep_player1.obs['to_play'] = -1

                if self._replay_path is not None:
                    if not os.path.exists(self._replay_path):
                        os.makedirs(self._replay_path)
                    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                    path = os.path.join(
                        self._replay_path,
                        'chess_{}_{}_{}.mp4'.format(os.getpid(), timestamp, self._save_replay_count)
                    )
                    self.display_frames_as_mp4(self._frames, path)
                    print(f'replay {path} saved!')
                    self._save_replay_count += 1

                return timestep_player1

            if self.agent_vs_human:
                bot_action = self.human_to_action()
            else:
                bot_action = self.bot_action()

            if self._replay_path is not None:
                self._frames.append(self._env.render(mode='rgb_array'))
            timestep_player2 = self._player_step(bot_action)
            if self._replay_path is not None:
                self._frames.append(self._env.render(mode='rgb_array'))

            timestep_player2.info['eval_episode_return'] = -timestep_player2.reward
            timestep_player2 = timestep_player2._replace(reward=-timestep_player2.reward)

            timestep = timestep_player2
            timestep.obs['to_play'] = -1

            if timestep_player2.done:
                if self._replay_path is not None:
                    if not os.path.exists(self._replay_path):
                        os.makedirs(self._replay_path)
                    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                    path = os.path.join(
                        self._replay_path,
                        'chess_{}_{}_{}.mp4'.format(os.getpid(), timestamp, self._save_replay_count)
                    )
                    self.display_frames_as_mp4(self._frames, path)
                    print(f'replay {path} saved!')
                    self._save_replay_count += 1

            return timestep

    def _player_step(self, action):
        if isinstance(action, np.ndarray):
            action = int(action)

        if action not in self.legal_actions:
            logging.warning(
                f"You input illegal action: {action}, the legal_actions are {self.legal_actions}. "
                f"Now we randomly choice a action from self.legal_actions."
            )
            action = np.random.choice(self.legal_actions)

        current_agent = self.current_player_index
        try:
            chosen_move = chess_utils.action_to_move(self.board, action, current_agent)
        except Exception as e:
            print('debug')

        assert chosen_move in self.board.legal_moves
        self.board.push(chosen_move)

        done = self.board.is_game_over()
        result = self.board.result(claim_draw=True)
        if result == "*":
            reward = 0.
        else:
            reward = chess_utils.result_to_int(result)

        if self.current_player == 1:
            reward = -reward

        info = {}
        if done:
            info['eval_episode_return'] = reward

        action_mask = np.zeros(4672, dtype=np.int8)
        action_mask[chess_utils.legal_moves(self.board)] = 1

        obs = {
            'observation': self.observe(self.current_player_index)['observation'],
            'action_mask': action_mask,
            'board': self.board.fen(),
            'current_player_index': 1 - self.current_player_index,
            'to_play': self.next_player
        }

        self.current_player = self.next_player

        return BaseEnvTimestep(obs, reward, done, info)

    @property
    def current_player(self):
        return self._current_player

    @property
    def current_player_index(self):
        return 0 if self._current_player == 1 else 1

    @property
    def next_player(self):
        return self.players[0] if self.current_player == self.players[1] else self.players[1]

    @current_player.setter
    def current_player(self, value):
        self._current_player = value

    def random_action(self):
        action_list = chess_utils.legal_moves(self.board)
        return np.random.choice(action_list)

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

    def human_to_action(self):
        while True:
            try:
                print(f"Current available actions for the player {self.to_play()} are:{self.legal_moves()}")
                choice = int(input(f"Enter the index of next move for the player {self.to_play()}: "))
                if choice in chess_utils.legal_moves(self.board):
                    break
            except KeyboardInterrupt:
                sys.exit(0)
            except Exception as e:
                print("Wrong input, try again")
        return choice

    def simulate_action(self, action):
        if action not in chess_utils.legal_moves(self.board):
            raise ValueError("action {0} on board {1} is not legal".format(action, self.board.fen()))
        new_board = copy.deepcopy(self.board)
        new_board.push(chess_utils.action_to_move(self.board, action, self.current_player_index))
        if self.start_player_index == 0:
            start_player_index = 1
        else:
            start_player_index = 0
        next_simulator_env = copy.deepcopy(self)
        next_simulator_env.reset(start_player_index, init_state=new_board.fen())
        return next_simulator_env

    def render(self, mode='human'):
        print(self.board)

    @staticmethod
    def display_frames_as_mp4(frames: list, path: str, fps=5) -> None:
        import imageio
        imageio.mimwrite(path, frames, fps=fps)

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    @property
    def observation_space(self) -> spaces.Space:
        return self._observation_space

    @property
    def action_space(self) -> spaces.Space:
        return self._action_space

    @property
    def reward_space(self) -> spaces.Space:
        return self._reward_space

    @staticmethod
    def create_collector_env_cfg(cfg: dict) -> List[dict]:
        collector_env_num = cfg.pop('collector_env_num')
        cfg = copy.deepcopy(cfg)
        return [cfg for _ in range(collector_env_num)]

    @staticmethod
    def create_evaluator_env_cfg(cfg: dict) -> List[dict]:
        evaluator_env_num = cfg.pop('evaluator_env_num')
        cfg = copy.deepcopy(cfg)
        cfg.battle_mode = 'eval_mode'
        return [cfg for _ in range(evaluator_env_num)]

    def __repr__(self) -> str:
        return "LightZero Chess Env"

    def close(self) -> None:
        pass
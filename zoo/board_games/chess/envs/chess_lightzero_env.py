import copy
import logging
import os
from collections import defaultdict
from datetime import datetime
from typing import List

import chess
import numpy as np
from ding.envs.env.base_env import BaseEnvTimestep
from ding.utils.registry_factory import ENV_REGISTRY
from gymnasium import spaces
from pettingzoo.classic.chess import chess_utils

from zoo.board_games.chess.envs.chess_env import ChessEnv


@ENV_REGISTRY.register('chess_lightzero')
class ChessLightZeroEnv(ChessEnv):
    def __init__(self, cfg=None):
        if cfg is None:
            cfg = {}
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
        self.agent_vs_human = cfg.agent_vs_human
        self.alphazero_mcts_ctree = cfg.alphazero_mcts_ctree
        self._replay_path = cfg.replay_path if hasattr(cfg, "replay_path") and cfg.replay_path is not None else None
        self._save_replay_count = 0
        self._observation_space = None
        self._action_space = None

        # self.board_history = np.zeros((8, 8, 104), dtype=bool)

        self.render_mode = self.cfg.get("render_mode", None)
        self.screen_height = self.screen_width = self.cfg.get("screen_size", 800)
        assert self.render_mode is None or self.render_mode in self.metadata["render_modes"]

        self.transposition_table = defaultdict(dict)

    @property
    def legal_actions(self):
        return chess_utils.legal_moves(self.board)

    def observe(self, agent_index):
        try:
            observation = chess_utils.get_observation(self.board, agent_index).astype(float)  # TODO
        except Exception as e:
            print('debug')

        # TODO:
        # observation = np.dstack((observation[:, :, :7], self.board_history))
        # We need to swap the white 6 channels with black 6 channels
        # if agent_index == 1:
        #     # 1. Mirror the board
        #     observation = np.flip(observation, axis=0)
        #     # 2. Swap the white 6 channels with the black 6 channels
        #     for i in range(1, 9):
        #         tmp = observation[..., 13 * i - 6 : 13 * i].copy()
        #         observation[..., 13 * i - 6 : 13 * i] = observation[
        #             ..., 13 * i : 13 * i + 6
        #         ]
        #         observation[..., 13 * i : 13 * i + 6] = tmp

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
        # self.board_history = np.zeros((8, 8, 104), dtype=bool)

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

        self.transposition_table = defaultdict(dict)

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
                self._frames.append(self.render(mode='rgb_array'))
            timestep_player2 = self._player_step(bot_action)
            if self._replay_path is not None:
                self._frames.append(self.render(mode='rgb_array'))

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

        # TODO: Update board history
        # next_board = chess_utils.get_observation(self.board, current_agent)
        # self.board_history = np.dstack((next_board[:, :, 7:], self.board_history[:, :, :-13]))

        chosen_move = chess_utils.action_to_move(self.board, action, current_agent)
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
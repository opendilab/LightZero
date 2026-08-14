"""LightZero environment for Chinese chess (Xiangqi).

MDP definition
--------------
State: four board frames encoded as 14 binary piece planes each, plus one
side-to-move plane, for a ``(57, 10, 9)`` tensor. The board and actions are
rotated 180 degrees for Black, so the current player always moves upwards.
Action: one of 2,086 geometrically possible Xiangqi moves; ``action_mask``
selects the legal subset.
Reward: zero until termination, then +1/-1 from the player making the step,
or zero for a draw. Checkmate, stalemate, or a captured general loses; three
repetitions, 60 no-capture rounds, or ``max_episode_steps`` plies draw.

``play_with_bot_mode`` exposes one agent transition for an agent move followed
by a random-bot move. ``self_play_mode`` exposes every individual ply and is
also used by AlphaZero's Python/C++ simulation environments.
"""

from __future__ import annotations

import copy
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
from ding.envs import BaseEnv, BaseEnvTimestep
from ding.utils import ENV_REGISTRY
from easydict import EasyDict
from gymnasium import spaces

from .action_encoding import (
    ACTION_SPACE_SIZE,
    action_to_move,
    mirror_action,
    move_to_action,
    move_to_uci,
    uci_to_move,
)
from .xiangqi import ADVISOR, BLACK, CANNON, ELEPHANT, HORSE, KING, PAWN, RED, ROOK, XiangqiState

HISTORY_LENGTH = 4
OBSERVATION_SHAPE = (HISTORY_LENGTH * 14 + 1, 10, 9)
_MAX_POSITION_HASHES = 121
_HASH_CHUNKS = 4  # uint64 encoded as four uint16 values in the int16 state array.
_SERIALIZED_HEADER_SIZE = 5
_SERIALIZED_SIZE = _SERIALIZED_HEADER_SIZE + HISTORY_LENGTH * 90 + _MAX_POSITION_HASHES * _HASH_CHUNKS


@ENV_REGISTRY.register('chinese_chess')
class ChineseChessEnv(BaseEnv):
    """Chinese chess environment supporting AlphaZero and MuZero bot mode."""

    config = dict(
        env_id='ChineseChess',
        battle_mode='self_play_mode',
        battle_mode_in_simulation_env='self_play_mode',
        bot_action_type='random',
        agent_vs_human=False,
        prob_random_agent=0.0,
        prob_expert_agent=0.0,
        channel_last=False,
        scale=True,
        alphazero_mcts_ctree=True,
        max_episode_steps=500,
        render_mode=None,
        replay_path=None,
    )

    @classmethod
    def default_config(cls) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    def __init__(self, cfg: Optional[Dict[str, Any]] = None) -> None:
        merged = self.default_config()
        if cfg is not None:
            merged.update(cfg)
        self._cfg = merged
        self.battle_mode = merged.battle_mode
        if self.battle_mode not in ('self_play_mode', 'play_with_bot_mode', 'eval_mode'):
            raise ValueError(f'unsupported battle_mode: {self.battle_mode}')
        if merged.bot_action_type != 'random':
            raise ValueError("ChineseChessEnv currently supports bot_action_type='random'")
        self.battle_mode_in_simulation_env = merged.battle_mode_in_simulation_env
        self.agent_vs_human = bool(merged.agent_vs_human)
        self.prob_random_agent = float(merged.prob_random_agent)
        self.prob_expert_agent = float(merged.prob_expert_agent)
        if self.prob_random_agent < 0 or self.prob_expert_agent < 0:
            raise ValueError('bot probabilities must be non-negative')
        if self.prob_expert_agent:
            raise ValueError('prob_expert_agent requires an expert bot, which is not configured')
        self.channel_last = bool(merged.channel_last)
        self.scale = bool(merged.scale)
        self.alphazero_mcts_ctree = bool(merged.alphazero_mcts_ctree)
        self.max_episode_steps = int(merged.max_episode_steps)
        self.render_mode = merged.render_mode
        self.players = [1, 2]
        self.total_num_actions = ACTION_SPACE_SIZE
        obs_shape = (10, 9, OBSERVATION_SHAPE[0]) if self.channel_last else OBSERVATION_SHAPE
        self._observation_space = spaces.Dict(
            {
                'observation': spaces.Box(0.0, 1.0, shape=obs_shape, dtype=np.float32),
                'action_mask': spaces.Box(0, 1, shape=(ACTION_SPACE_SIZE, ), dtype=np.int8),
            }
        )
        self._action_space = spaces.Discrete(ACTION_SPACE_SIZE)
        self._reward_space = spaces.Box(-1.0, 1.0, shape=(1, ), dtype=np.float32)
        self._rng = np.random.default_rng()
        self._state = XiangqiState()
        self._history: Deque[np.ndarray] = deque(maxlen=HISTORY_LENGTH)
        self._current_player = 1
        self.start_player_index = 0
        self._env = self
        self.reset()

    @property
    def current_player(self) -> int:
        return self._current_player

    @current_player.setter
    def current_player(self, player: int) -> None:
        self._current_player = int(player)
        self._state.turn = RED if self._current_player == 1 else BLACK
        self._state._legal_cache = None

    @property
    def current_player_index(self) -> int:
        return self._current_player - 1

    @property
    def next_player(self) -> int:
        return 2 if self._current_player == 1 else 1

    @property
    def board(self) -> np.ndarray:
        return self._state.board

    def _physical_to_canonical_action(self, action: int) -> int:
        return action if self._state.turn == RED else mirror_action(action)

    def _canonical_to_physical_action(self, action: int) -> int:
        return action if self._state.turn == RED else mirror_action(action)

    @property
    def legal_actions(self) -> List[int]:
        return [self._physical_to_canonical_action(move_to_action(move)) for move in self._state.legal_moves()]

    def _serialize(self) -> np.ndarray:
        header = np.array(
            [
                self._state.turn,
                self._state.halfmove_clock,
                self._state.ply,
                len(self._history),
                len(self._state.position_hashes),
            ],
            dtype=np.int16
        )
        history = list(self._history)
        if not history:
            history = [self._state.board]
        history = ([history[0]] * (HISTORY_LENGTH - len(history))) + history[-HISTORY_LENGTH:]
        position_hashes = np.zeros(_MAX_POSITION_HASHES, dtype=np.uint64)
        recent_hashes = self._state.position_hashes[-_MAX_POSITION_HASHES:]
        position_hashes[-len(recent_hashes):] = recent_hashes
        hash_chunks = position_hashes.view(np.uint16).view(np.int16)
        return np.concatenate(
            [header] + [board.astype(np.int16, copy=False).reshape(-1) for board in history] + [hash_chunks]
        )

    def _deserialize(self, init_state: Any, start_player_index: int) -> Tuple[XiangqiState, Deque[np.ndarray]]:
        if isinstance(init_state, (bytes, bytearray, memoryview)):
            data = np.frombuffer(init_state, dtype=np.int16)
        else:
            data = np.asarray(init_state)
        if data.size == 90:
            state = XiangqiState(data.reshape(10, 9), RED if start_player_index == 0 else BLACK)
            history = deque([state.board.copy()] * HISTORY_LENGTH, maxlen=HISTORY_LENGTH)
            return state, history
        flat = np.asarray(data, dtype=np.int16).reshape(-1)
        if flat.size != _SERIALIZED_SIZE:
            raise ValueError(f'init_state has {flat.size} values; expected 90 or {_SERIALIZED_SIZE}')
        turn, halfmove_clock, ply, history_length, hash_count = map(int, flat[:_SERIALIZED_HEADER_SIZE])
        board_end = _SERIALIZED_HEADER_SIZE + HISTORY_LENGTH * 90
        boards = flat[_SERIALIZED_HEADER_SIZE:board_end].reshape(HISTORY_LENGTH, 10, 9).astype(np.int8)
        hash_count = min(_MAX_POSITION_HASHES, max(1, hash_count))
        stored_hashes = np.ascontiguousarray(flat[board_end:], dtype=np.int16).view(np.uint16).view(np.uint64)
        position_hashes = [int(value) for value in stored_hashes[-hash_count:]]
        state = XiangqiState(boards[-1], turn, halfmove_clock, ply, position_hashes=position_hashes)
        valid_length = min(HISTORY_LENGTH, max(1, history_length))
        history = deque((board.copy() for board in boards[-valid_length:]), maxlen=HISTORY_LENGTH)
        return state, history

    def reset(
        self,
        start_player_index: int = 0,
        init_state: Any = None,
        katago_policy_init: bool = False,
        katago_game_state: Any = None,
    ) -> Dict[str, Any]:
        del katago_policy_init, katago_game_state
        if start_player_index not in (0, 1):
            raise ValueError('start_player_index must be 0 or 1')
        self.start_player_index = int(start_player_index)
        if init_state is None:
            self._state = XiangqiState(turn=RED if start_player_index == 0 else BLACK)
            self._history = deque([self._state.board.copy()] * HISTORY_LENGTH, maxlen=HISTORY_LENGTH)
        else:
            self._state, self._history = self._deserialize(init_state, start_player_index)
        self._current_player = 1 if self._state.turn == RED else 2
        return self._make_obs()

    def _canonical_planes(self) -> np.ndarray:
        planes: List[np.ndarray] = []
        perspective = self._state.turn
        piece_types = (PAWN, ROOK, HORSE, CANNON, ADVISOR, ELEPHANT, KING)
        history = list(self._history)
        history = ([history[0]] * (HISTORY_LENGTH - len(history))) + history[-HISTORY_LENGTH:]
        for physical_board in history:
            board = physical_board if perspective == RED else np.rot90(physical_board, 2)
            for color in (perspective, -perspective):
                planes.extend((board == color * piece).astype(np.float32) for piece in piece_types)
        planes.append(np.full((10, 9), 1.0 if perspective == RED else 0.0, dtype=np.float32))
        observation = np.stack(planes, axis=0)
        return np.transpose(observation, (1, 2, 0)) if self.channel_last else observation

    def current_state(self) -> Tuple[np.ndarray, np.ndarray]:
        observation = self._canonical_planes()
        return observation.copy(), observation.copy()

    def _make_obs(self) -> Dict[str, Any]:
        action_mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.int8)
        legal_actions = self.legal_actions
        if legal_actions:
            action_mask[legal_actions] = 1
        to_play = self.current_player if self.battle_mode == 'self_play_mode' else -1
        return {
            'observation': self.current_state()[1],
            'action_mask': action_mask,
            'board': self._serialize(),
            'current_player_index': self.current_player_index,
            'to_play': to_play,
        }

    def get_done_winner(self) -> Tuple[bool, int]:
        result = self._state.result(self.max_episode_steps)
        winner = 1 if result.winner == RED else 2 if result.winner == BLACK else -1
        return result.done, winner

    def _player_step(self, action: int) -> BaseEnvTimestep:
        if isinstance(action, np.ndarray):
            if action.size != 1:
                raise ValueError(f'action array must contain one value, got shape {action.shape}')
            action = int(action.item())
        action = int(action)
        if action not in self.legal_actions:
            raise ValueError(f'illegal canonical action {action}; legal actions: {self.legal_actions}')
        acting_player = self.current_player
        physical_action = self._canonical_to_physical_action(action)
        self._state.push(action_to_move(physical_action))
        self._history.append(self._state.board.copy())
        self._current_player = 1 if self._state.turn == RED else 2
        done, winner = self.get_done_winner()
        reward = np.float32(0.0 if not done or winner == -1 else 1.0 if winner == acting_player else -1.0)
        info: Dict[str, Any] = {'next_player': self.current_player}
        if done:
            info['eval_episode_return'] = np.float32(1.0 if winner == 1 else -1.0 if winner == 2 else 0.0)
        return BaseEnvTimestep(self._make_obs(), reward, done, info)

    def step(self, action: int) -> BaseEnvTimestep:
        if self.battle_mode == 'self_play_mode':
            if self.prob_random_agent > 0 and self._rng.random() < self.prob_random_agent:
                action = self.random_action()
            return self._player_step(action)

        agent_timestep = self._player_step(action)
        if agent_timestep.done:
            agent_timestep.obs['to_play'] = -1
            return agent_timestep
        opponent_action = (
            self.human_to_action() if self.battle_mode == 'eval_mode' and self.agent_vs_human else self.bot_action()
        )
        opponent_timestep = self._player_step(opponent_action)
        reward = np.float32(-float(opponent_timestep.reward))
        opponent_timestep.info['eval_episode_return'] = reward
        opponent_timestep.obs['to_play'] = -1
        return opponent_timestep._replace(reward=reward)

    def random_action(self) -> int:
        actions = self.legal_actions
        if not actions:
            raise RuntimeError('no legal actions in terminal position')
        return int(self._rng.choice(actions))

    def bot_action(self) -> int:
        return self.random_action()

    def human_to_action(self) -> int:
        self.render('human')
        while True:
            try:
                text = input('Your move (ICCS, e.g. h9g7; q to quit): ').strip().lower()
                if text in ('q', 'quit', 'exit'):
                    raise KeyboardInterrupt
                physical = move_to_action(uci_to_move(text))
                action = self._physical_to_canonical_action(physical)
                if action in self.legal_actions:
                    return action
                print(f'Illegal move: {text}')
            except ValueError as error:
                print(error)

    def action_to_string(self, action: int) -> str:
        physical = self._canonical_to_physical_action(int(action))
        return move_to_uci(action_to_move(physical))

    def simulate_action(self, action: int) -> 'ChineseChessEnv':
        env = self.clone()
        env.battle_mode = 'self_play_mode'
        env._player_step(action)
        return env

    def clone(self) -> 'ChineseChessEnv':
        return copy.deepcopy(self)

    def render(self, mode: str = 'human') -> Optional[str]:
        if mode != 'human':
            raise ValueError("ChineseChessEnv supports render mode 'human' only")
        symbols = {
            0: '.',
            KING: 'K',
            ADVISOR: 'A',
            ELEPHANT: 'E',
            HORSE: 'H',
            ROOK: 'R',
            CANNON: 'C',
            PAWN: 'P',
            -KING: 'k',
            -ADVISOR: 'a',
            -ELEPHANT: 'e',
            -HORSE: 'h',
            -ROOK: 'r',
            -CANNON: 'c',
            -PAWN: 'p',
        }
        lines = ['    a b c d e f g h i']
        for row in range(9, -1, -1):
            lines.append(f'{row:>2}  ' + ' '.join(symbols[int(piece)] for piece in self.board[row]))
        lines.append(f"Turn: {'Red' if self._state.turn == RED else 'Black'}")
        output = '\n'.join(lines)
        print(output)
        return output

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        del dynamic_seed
        self._rng = np.random.default_rng(seed)

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
    def create_collector_env_cfg(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
        cfg = copy.deepcopy(cfg)
        env_num = cfg.pop('collector_env_num')
        return [copy.deepcopy(cfg) for _ in range(env_num)]

    @staticmethod
    def create_evaluator_env_cfg(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
        cfg = copy.deepcopy(cfg)
        env_num = cfg.pop('evaluator_env_num')
        cfg['battle_mode'] = 'eval_mode'
        return [copy.deepcopy(cfg) for _ in range(env_num)]

    def close(self) -> None:
        return None

    def __repr__(self) -> str:
        return 'LightZero Chinese Chess Env'

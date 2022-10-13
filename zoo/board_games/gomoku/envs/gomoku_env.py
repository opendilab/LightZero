import copy
import sys
from typing import List

import gym
import numpy as np
from ding.envs import BaseEnvTimestep
from ding.utils import ENV_REGISTRY
from ditk import logging
from easydict import EasyDict

from zoo.board_games.base_game_env import BaseGameEnv
from zoo.board_games.gomoku.envs.gomoku_expert import GomokuExpert


@ENV_REGISTRY.register('gomoku')
class GomokuEnv(BaseGameEnv):

    config = dict(
        prob_random_agent=0,
        board_size=15,
        battle_mode='two_player_mode',
    )

    @classmethod
    def default_config(cls: type) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    def __init__(self, cfg: dict = None):
        self.cfg = cfg
        self.battle_mode = cfg.battle_mode
        self.board_size = cfg.board_size
        self.prob_random_agent = cfg.prob_random_agent

        self.players = [1, 2]
        self.board_markers = [str(i + 1) for i in range(self.board_size)]
        self.total_num_actions = self.board_size * self.board_size
        self.expert = GomokuExpert()

    @property
    def current_player(self):
        return self._current_player

    @property
    def to_play(self):
        return self.players[0] if self.current_player == self.players[1] else self.players[1]

    @property
    def legal_actions(self):
        legal_actions = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i][j] == 0:
                    legal_actions.append(self.coord_to_action(i, j))
        return legal_actions

    def reset(self, start_player=0):
        self._observation_space = gym.spaces.Box(
            low=0, high=2, shape=(self.board_size, self.board_size, 3), dtype=np.int32
        )
        self._action_space = gym.spaces.Discrete(self.board_size ** 2)
        self._reward_space = gym.spaces.Box(low=0, high=1, shape=(1, ), dtype=np.float32)

        self._current_player = self.players[start_player]
        self.board = np.zeros((self.board_size, self.board_size), dtype="int32")
        action_mask = np.zeros(self.total_num_actions, 'int8')
        action_mask[self.legal_actions] = 1
        if self.battle_mode == 'two_player_mode':
            obs = {'observation': self.current_state(), 'action_mask': action_mask, 'to_play': self.current_player}
        else:
            obs = {'observation': self.current_state(), 'action_mask': action_mask, 'to_play': None}
        return obs

    def step(self, action):
        if self.battle_mode == 'two_player_mode':
            if np.random.rand() < self.prob_random_agent:
                action = self.random_action()
            timestep = self._player_step(action)
            # print(self.board)
            return timestep
        elif self.battle_mode == 'one_player_mode':
            # player 1 battle with expert player 2

            # player 1's turn
            timestep_player1 = self._player_step(action)
            # print('player 1 (muzero player): ' + self.action_to_string(action))  # TODO(pu): visualize
            if timestep_player1.done:
                # in one_player_mode, we set to_play as None, because we don't consider the alternation between players
                timestep_player1.obs['to_play'] = None
                return timestep_player1

            # player 2's turn
            expert_action = self.expert_action()
            # print('player 2 (computer random player): ' + self.action_to_string(expert_action))  # TODO(pu): visualize
            timestep_player2 = self._player_step(expert_action)
            # self.render()  # TODO(pu): visualize
            # the final_eval_reward is calculated from Player 1's perspective
            timestep_player2.info['final_eval_reward'] = -timestep_player2.reward
            timestep_player2 = timestep_player2._replace(reward=-timestep_player2.reward)

            timestep = timestep_player2
            # in one_player_mode, we set to_play as None, because we don't consider the alternation between players
            timestep.obs['to_play'] = None
            return timestep

    def _player_step(self, action):
        if action in self.legal_actions:
            row, col = self.action_to_coord(action)
            self.board[row, col] = self.current_player
        else:
            logging.warning(
                f"You input illegal action: {action}, the legal_actions are {self.legal_actions}. "
                f"Now we randomly choice a action from self.legal_actions."
            )
            action = np.random.choice(self.legal_actions)
            row, col = self.action_to_coord(action)
            self.board[row, col] = self.current_player

        # Check whether the game is ended or not and give the winner
        done, winner = self.have_winner()

        reward = np.array(float(winner == self.current_player)).astype(np.float32)
        info = {'next player to play': self.to_play}
        """
        NOTE: here exchange the player
        """
        self.current_player = self.to_play

        if done:
            info['final_eval_reward'] = reward
            # print('gomoku one episode done: ', info)

        action_mask = np.zeros(self.total_num_actions, 'int8')
        action_mask[self.legal_actions] = 1
        obs = {'observation': self.current_state(), 'action_mask': action_mask, 'to_play': self.current_player}
        return BaseEnvTimestep(obs, reward, done, info)

    def current_state(self):
        """
        Overview:
            self.board is nd-array, 0 indicates that no stones is placed here,
            1 indicates that player 1's stone is placed here, 2 indicates player 2's stone is placed here
        Arguments:
            - raw_obs (:obj:`array`):
                the 0 dim means which positions is occupied by self.current_player,
                the 1 dim indicates which positions are occupied by self.to_play,
                the 2 dim indicates which player is the to_play player, 1 means player 1, 2 means player 2
        """
        board_curr_player = np.where(self.board == self.current_player, 1, 0)
        board_opponent_player = np.where(self.board == self.to_play, 1, 0)
        board_to_play = np.full((self.board_size, self.board_size), self.current_player)
        raw_obs = np.array([board_curr_player, board_opponent_player, board_to_play], dtype=np.float32)
        # move channel dim to last axis to be compatible with EfficientZero
        return np.moveaxis(raw_obs, 0, 2)

    def coord_to_action(self, i, j):
        """
        Overview:
            convert coordinate i, j to action index a in [0, board_size**2)
        """
        return i * self.board_size + j

    def action_to_coord(self, a):
        """
        Overview:
            convert action index a in [0, board_size**2) to coordinate (i, j)
        """
        return a // self.board_size, a % self.board_size

    def have_winner(self):
        has_legal_actions = False
        directions = ((1, -1), (1, 0), (1, 1), (0, 1))
        for i in range(self.board_size):
            for j in range(self.board_size):
                # if no stone is on the position, don't need to consider this position
                if self.board[i][j] == 0:
                    has_legal_actions = True
                    continue
                # value-value at a coord, i-row, j-col
                player = self.board[i][j]
                # check if there exist 5 in a line
                for d in directions:
                    x, y = i, j
                    count = 0
                    for _ in range(5):
                        if (x not in range(self.board_size)) or (y not in range(self.board_size)):
                            break
                        if self.board[x][y] != player:
                            break
                        x += d[0]
                        y += d[1]
                        count += 1
                        # if 5 in a line, store positions of all stones, return value
                        if count == 5:
                            return True, player
        # if the players don't have legal actions, return done=True
        return not has_legal_actions, -1

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def random_action(self):
        action_list = self.legal_actions
        return np.random.choice(action_list)

    def expert_action(self):
        action_mask = np.zeros(self.total_num_actions, 'int8')
        action_mask[self.legal_actions] = 1
        obs = {'observation': self.current_state(), 'action_mask': action_mask}
        return self.expert.get_action(obs)

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
                row = int(
                    input(
                        f"Enter the row (1, 2, ...,{self.board_size}, from up to bottom) to play for the player {self.current_player}: "
                    )
                )
                col = int(
                    input(
                        f"Enter the column (1, 2, ...,{self.board_size}, from left to right) to play for the player {self.current_player}: "
                    )
                )
                choice = self.coord_to_action(row - 1, col - 1)
                if (choice in self.legal_actions and 1 <= row and 1 <= col and row <= self.board_size
                        and col <= self.board_size):
                    break
                else:
                    print("Wrong input, try again")
            except KeyboardInterrupt:
                print("exit")
                sys.exit(0)
            except Exception as e:
                print("Wrong input, try again")
        return choice

    def render(self, mode="human"):
        marker = "   "
        for i in range(self.board_size):
            if i <= 8:
                marker = marker + self.board_markers[i] + "  "
            else:
                marker = marker + self.board_markers[i] + " "
        print(marker)
        for row in range(self.board_size):
            if row <= 8:
                print(str(1 + row) + ' ', end=" ")
            else:
                print(str(1 + row), end=" ")
            for col in range(self.board_size):
                ch = self.board[row][col]
                if ch == 0:
                    print(".", end="  ")
                elif ch == 1:
                    print("X", end="  ")
                elif ch == 2:
                    print("O", end="  ")
            print()

    def action_to_string(self, action_number):
        """
        Overview:
            Convert an action number to a string representing the action.
        Arguments:
            - action_number: an integer from the action space.
        Returns:
            - String representing the action.
        """
        row = action_number // self.board_size + 1
        col = action_number % self.board_size + 1
        return f"Play row {row}, column {col}"

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        return self._action_space

    @property
    def reward_space(self) -> gym.spaces.Space:
        return self._reward_space

    def close(self) -> None:
        pass

    @current_player.setter
    def current_player(self, value):
        self._current_player = value

    @staticmethod
    def create_collector_env_cfg(cfg: dict) -> List[dict]:
        collector_env_num = cfg.pop('collector_env_num')
        cfg = copy.deepcopy(cfg)
        return [cfg for _ in range(collector_env_num)]

    @staticmethod
    def create_evaluator_env_cfg(cfg: dict) -> List[dict]:
        evaluator_env_num = cfg.pop('evaluator_env_num')
        cfg = copy.deepcopy(cfg)
        # NOTE: when in eval phase, we use 'one_player_mode' to evaluate the current agent with bot
        cfg.battle_mode = 'one_player_mode'
        return [cfg for _ in range(evaluator_env_num)]

    def __repr__(self) -> str:
        return "DI-engine Gomoku Env"

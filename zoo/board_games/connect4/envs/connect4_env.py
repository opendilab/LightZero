"""
Overview:
    Adapt the connect4 environment in PettingZoo (https://github.com/Farama-Foundation/PettingZoo) to the BaseEnv interface.
    Connect Four is a 2-player turn based game, where players must connect four of their tokens vertically, horizontally or diagonally. 
    The players drop their respective token in a column of a standing grid, where each token will fall until it reaches the bottom of the column or reaches an existing token. 
    Players cannot place a token in a full column, and the game ends when either a player has made a sequence of 4 tokens, or when all 7 columns have been filled.
Observation Space:
    The observation in the Connect4 environment is a dictionary with five elements, which contains key information about the current state. 
    - observation (:obj:`array`): An array that represents information about the current state, with a shape of (3, 6, 7). 
        The length of the first dimension is 3, which stores three two-dimensional game boards with shapes (6, 7).
        These boards represent the positions occupied by the current player, the positions occupied by the opponent player, and the identity of the current player, respectively.
    - action_mask (:obj:`array`): A mask for the actions, indicating which actions are executable. It is a one-dimensional array of length 7, corresponding to columns 1 to 7 of the game board. 
        It has a value of 1 for the columns where a move can be made, and a value of 0 for other positions.
    - board (:obj:`array`): A visual representation of the current game board, represented as a 6x7 array, in which the positions where player 1 and player 2 have placed their tokens are marked with values 1 and 2, respectively. 
    - current_player_index (:obj:`int`): The index of the current player, with player 1 having an index of 0 and player 2 having an index of 1. 
    - to_play (:obj:`int`): The player who needs to take an action in the current state, with a value of 1 or 2.
Action Space:
    A set of integers from 0 to 6 (inclusive), where the action represents which column a token should be dropped in.
Reward Space:
    For the ``self_play_mode``, a reward of 1 is returned at the time step when the game terminates, and a reward of 0 is returned at all other time steps.
    For the ``play_with_bot_mode``, at the time step when the game terminates, if the bot wins, the reward is -1; if the agent wins, the reward is 1; and in all other cases, the reward is 0.
"""

import os
import sys
import copy
from ditk import logging
from typing import List

import numpy as np
import pygame
from ding.envs import BaseEnv, BaseEnvTimestep
from ding.utils import ENV_REGISTRY
from gym import spaces
from easydict import EasyDict

# from pettingzoo.utils.agent_selector import agent_selector
from zoo.board_games.alphabeta_pruning_bot import AlphaBetaPruningBot
from zoo.board_games.connect4.envs.rule_bot import Connect4RuleBot

from zoo.board_games.mcts_bot import MCTSBot


@ENV_REGISTRY.register('connect4')
class Connect4Env(BaseEnv):
    config = dict(
        env_name="Connect4",
        battle_mode='self_play_mode',
        mcts_mode='self_play_mode',  # only used in AlphaZero
        # bot_action_type='mcts',
        bot_action_type='rule',
        agent_vs_human=False,
        prob_random_agent=0,
        prob_expert_agent=0,
        prob_random_action_in_bot=0.,
        channel_last=True,
        scale=False,
        stop_value=2,
    )

    @classmethod
    def default_config(cls: type) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    def __init__(self, cfg=None) -> None:

        self.cfg = cfg
        self.channel_last = cfg.channel_last
        self.scale = cfg.scale
        self.battle_mode = cfg.battle_mode
        self.mcts_mode = cfg.mcts_mode
        # The mode of interaction between the agent and the environment.
        assert self.battle_mode in ['self_play_mode', 'play_with_bot_mode', 'eval_mode']
        self.prob_random_agent = cfg.prob_random_agent
        self.prob_expert_agent = cfg.prob_expert_agent
        assert (self.prob_random_agent >= 0 and self.prob_expert_agent == 0) or (
                self.prob_random_agent == 0 and self.prob_expert_agent >= 0), \
            f'self.prob_random_agent:{self.prob_random_agent}, self.prob_expert_agent:{self.prob_expert_agent}'
        self.prob_random_action_in_bot = cfg.prob_random_action_in_bot
        self.agent_vs_human = cfg.agent_vs_human
        self.bot_action_type = cfg.bot_action_type
        # if 'alpha_beta_pruning' in self.bot_action_type:
        #     self.alpha_beta_bot = AlphaBetaPruningBot(self, cfg, 'alpha_beta_pruning_player')

        self._current_player = 1
        self.board = [0] * (6 * 7)
        self.players = [1, 2]

        if self.bot_action_type == 'mcts':
            self.mcts_bot = MCTSBot(self, 'mcts_player', 200)
        elif self.bot_action_type == 'rule':
            self.rule_bot = Connect4RuleBot(self, self._current_player)

        self._env = self

        # self.possible_agents = self.agents[:]

        # self.action_spaces = {i: spaces.Discrete(7) for i in self.players}
        # self.observation_spaces = {
        #     i: spaces.Dict(
        #         {
        #             "observation": spaces.Box(
        #                 low=0, high=1, shape=(6, 7, 2), dtype=np.int8
        #             ),
        #             "action_mask": spaces.Box(low=0, high=1, shape=(7,), dtype=np.int8),
        #         }
        #     )
        #     for i in self.players
        # }

    def current_state(self):
        """
        Overview:
            obtain the state from the view of current player.
            self.board is nd-array, 0 indicates that no stones is placed here,
            1 indicates that player 1's stone is placed here, 2 indicates player 2's stone is placed here
        Returns:
            - current_state (:obj:`array`):
                the 0 dim means which positions is occupied by self.current_player,
                the 1 dim indicates which positions are occupied by self.to_play,
                the 2 dim indicates which player is the to_play player, 1 means player 1, 2 means player 2
        """
        board_vals = np.array(self.board).reshape(6, 7)
        board_curr_player = np.where(board_vals == self.current_player, 1, 0)
        board_opponent_player = np.where(board_vals == self.next_player, 1, 0)
        board_to_play = np.full((6, 7), self.current_player)
        raw_obs = np.array([board_curr_player, board_opponent_player, board_to_play], dtype=np.float32)
        if self.scale:
            scale_obs = copy.deepcopy(raw_obs / 2)
        else:
            scale_obs = copy.deepcopy(raw_obs)
        if self.channel_last:
            # move channel dim to last axis
            # (C, W, H) -> (W, H, C)
            return np.transpose(raw_obs, [1, 2, 0]), np.transpose(scale_obs, [1, 2, 0])
        else:
            # (C, W, H)
            return raw_obs, scale_obs

    def observe(self):
        legal_moves = self.legal_actions

        action_mask = np.zeros(7, "int8")
        for i in legal_moves:
            action_mask[i] = 1

        if self.battle_mode == 'play_with_bot_mode' or self.battle_mode == 'eval_mode':
            return {"observation": self.current_state()[1], 
                    "action_mask": action_mask,
                    "board": copy.deepcopy(self.board),
                    "current_player_index": self.players.index(self._current_player),
                    "to_play": -1
                    }
        elif self.battle_mode == 'self_play_mode':
            return {"observation": self.current_state()[1], 
                    "action_mask": action_mask,
                    "board": copy.deepcopy(self.board),
                    "current_player_index": self.players.index(self._current_player),
                    "to_play": self._current_player
                    }

    @property
    def legal_actions(self):
        return [i for i in range(7) if self.board[i] == 0]

    # action in this case is a value from 0 to 6 indicating position to move on the flat representation of the connect4 board
    def step(self, action):
        if self.battle_mode == 'self_play_mode':
            if self.prob_random_agent > 0:
                if np.random.rand() < self.prob_random_agent:
                    action = self.random_action()
            elif self.prob_expert_agent > 0:
                if np.random.rand() < self.prob_expert_agent:
                    action = self.bot_action()

            # print(f'self playing now, the action is {action}')
            flag = "agent"
            timestep = self._player_step(action, flag)
            if timestep.done:
                # The eval_episode_return is calculated from Player 1's perspective。
                # 不是很明白episode_reward在train的时候是怎么被调用的#########################
                timestep.info['eval_episode_return'] = -timestep.reward if timestep.obs[
                                                                               'to_play'] == 1 else timestep.reward
            return timestep
        elif self.battle_mode == 'play_with_bot_mode':
            # player 1 battle with expert player 2

            # player 1's turn
            # print(f'playing with bot now, the action from algorithm is {action}')
            flag = "bot_agent"
            timestep_player1 = self._player_step(action, flag)
            # self.env.render()
            if timestep_player1.done:
                # NOTE: in play_with_bot_mode, we must set to_play as -1, because we don't consider the alternation between players.
                # And the to_play is used in MCTS.
                timestep_player1.obs['to_play'] = -1
                return timestep_player1

            # player 2's turn
            bot_action = self.bot_action()
            # print('player 2 (computer player): ' + self.action_to_string(bot_action))
            # print(f'playing with bot now, the action from bot is {action}')
            flag = "bot_bot"
            timestep_player2 = self._player_step(bot_action, flag)
            # the eval_episode_return is calculated from Player 1's perspective
            timestep_player2.info['eval_episode_return'] = -timestep_player2.reward
            timestep_player2 = timestep_player2._replace(reward=-timestep_player2.reward)

            timestep = timestep_player2
            # NOTE: in play_with_bot_mode, we must set to_play as -1, because we don't consider the alternation between players.
            # And the to_play is used in MCTS.
            timestep.obs['to_play'] = -1

            return timestep
        elif self.battle_mode == 'eval_mode':
            # player 1 battle with expert player 2

            # player 1's turn
            print('player 1 (agent player): ' + self.action_to_string(action))
            # print(f'evaluating now, the battle mode is {self.battle_mode}, the action is {action}')
            flag = "eval_agent"
            timestep_player1 = self._player_step(action, flag)
            self.render()
            if timestep_player1.done:
                # NOTE: in eval_mode, we must set to_play as -1, because we don't consider the alternation between players.
                # And the to_play is used in MCTS.
                timestep_player1.obs['to_play'] = -1
                return timestep_player1

            # player 2's turn
            if self.agent_vs_human:
                bot_action = self.human_to_action()
            else:
                # print(f'evaluating now, the battle mode is {self.battle_mode}, here comes the bot action{self.bot_action()}')
                bot_action = self.bot_action()
            # print('player 2 (computer player): ' + self.action_to_string(bot_action))
            flag = "eval_bot"
            timestep_player2 = self._player_step(bot_action, flag)
            self.render()
            # the eval_episode_return is calculated from Player 1's perspective
            timestep_player2.info['eval_episode_return'] = -timestep_player2.reward
            timestep_player2 = timestep_player2._replace(reward=-timestep_player2.reward)

            timestep = timestep_player2
            # NOTE: in eval_mode, we must set to_play as -1, because we don't consider the alternation between players.
            # And the to_play is used in MCTS.
            timestep.obs['to_play'] = -1

            return timestep

    def _player_step(self, action, flag):
        if action in self.legal_actions:
            piece = self.players.index(self._current_player) + 1
            for i in list(filter(lambda x: x % 7 == action, list(range(41, -1, -1)))):
                if self.board[i] == 0:
                    self.board[i] = piece
                    break
        else:
            print(np.array(self.board).reshape(6, 7))
            logging.warning(
                f"You input illegal action: {action}, the legal_actions are {self.legal_actions}. "
                f"flag is {flag}."
                f"Now we randomly choice a action from self.legal_actions."
            )
            action = self.random_action()
            print("the random action is", action)
            piece = self.players.index(self._current_player) + 1
            for i in list(filter(lambda x: x % 7 == action, list(range(41, -1, -1)))):
                if self.board[i] == 0:
                    self.board[i] = piece
                    break

        done, winner = self.get_done_winner()
        # check if there is a winner
        if not winner == -1:
            # self.rewards[self.agent_selection] += 1
            # self.rewards[next_agent] -= 1
            reward = np.array(1).astype(np.float32)
            # self.dones = {i: True for i in self.agents}
        # check if there is a tie
        # elif all(x in [1, 2] for x in self.board):
        #     # once either play wins or there is a draw, game over, both players are done
        #     # self.dones = {i: True for i in self.agents}
        #     reward = np.array(0).astype(np.float32)
        #     done = True
        else:
            reward = np.array(0).astype(np.float32)

        info = {}

        self._current_player = self.next_player

        if done:
            # 稀疏奖励，不需要累加，直接取最后一步的奖励
            info['eval_episode_return'] = reward
            # print('tictactoe one episode done: ', info)

        obs = self.observe()

        return BaseEnvTimestep(obs, reward, done, info)

    def reset(self, start_player_index=0, init_state=None):
        # reset environment
        if init_state is None:
            self.board = [0] * (6 * 7)
        else:
            # print("before:", self.board)
            self.board = init_state
            # print("after:", self.board)
        self.players = [1, 2]
        self.start_player_index = start_player_index
        self._current_player = self.players[self.start_player_index]

        self._action_space = spaces.Discrete(7)
        self._reward_space = spaces.Discrete(3)
        self._observation_space = spaces.Dict(
            {
                "observation": spaces.Box(low=0, high=1, shape=(3, 6, 7), dtype=np.int8),
                "action_mask": spaces.Box(low=0, high=1, shape=(7,), dtype=np.int8),
                "board": spaces.Box(low=0, high=2, shape=(6, 7), dtype=np.int8),
                "current_player_index": spaces.Discrete(2),
                "to_play": spaces.Discrete(2),
            }
        )

        # self.rewards = {i: 0 for i in self.agents}
        # self._cumulative_rewards = {name: 0 for name in self.agents}
        # self.dones = {i: False for i in self.agents}
        # self.infos = {i: {} for i in self.agents}

        # self._agent_selector = agent_selector(self.agents)

        # for agent, reward in self.rewards.items():
        #     self._cumulative_rewards[agent] += reward

        obs = self.observe()
        return obs

    def reset_v2(self, start_player_index=0, init_state=None):
        """
        Overview:
            only used in alpha-beta pruning bot.
        """
        self.start_player_index = start_player_index
        self._current_player = self.players[self.start_player_index]
        if init_state is not None:
            self.board = np.array(init_state, dtype="int32")
        else:
            self.board = [0] * (6 * 7)

    def render(self):
        print(np.array(self.board).reshape(6, 7))

    def close(self):
        pass

    def get_done_winner(self):
        # TODO
        board = copy.deepcopy(np.array(self.board)).reshape(6, 7)
        for piece in [1, 2]:
            # Check horizontal locations for win
            column_count = 7
            row_count = 6

            for c in range(column_count - 3):
                for r in range(row_count):
                    if (
                            board[r][c] == piece
                            and board[r][c + 1] == piece
                            and board[r][c + 2] == piece
                            and board[r][c + 3] == piece
                    ):
                        return True, piece

            # Check vertical locations for win
            for c in range(column_count):
                for r in range(row_count - 3):
                    if (
                            board[r][c] == piece
                            and board[r + 1][c] == piece
                            and board[r + 2][c] == piece
                            and board[r + 3][c] == piece
                    ):
                        return True, piece

            # Check positively sloped diagonals
            for c in range(column_count - 3):
                for r in range(row_count - 3):
                    if (
                            board[r][c] == piece
                            and board[r + 1][c + 1] == piece
                            and board[r + 2][c + 2] == piece
                            and board[r + 3][c + 3] == piece
                    ):
                        return True, piece

            # Check negatively sloped diagonals
            for c in range(column_count - 3):
                for r in range(3, row_count):
                    if (
                            board[r][c] == piece
                            and board[r - 1][c + 1] == piece
                            and board[r - 2][c + 2] == piece
                            and board[r - 3][c + 3] == piece
                    ):
                        return True, piece

        if all(x in [1, 2] for x in self.board):
            return True, -1

        return False, -1

    def get_done_reward(self):
        """
        Overview:
             Check if the game is over and what is the reward in the perspective of player 1.
             Return 'done' and 'reward'.
        Returns:
            - outputs (:obj:`Tuple`): Tuple containing 'done' and 'reward',
                - if player 1 win,     'done' = True, 'reward' = 1
                - if player 2 win,     'done' = True, 'reward' = -1
                - if draw,             'done' = True, 'reward' = 0
                - if game is not over, 'done' = False,'reward' = None
        """
        done, winner = self.get_done_winner()
        if winner == 1:
            reward = 1
        elif winner == 2:
            reward = -1
        elif winner == -1 and done:
            reward = 0
        elif winner == -1 and not done:
            # episode is not done
            reward = None
        return done, reward

    def random_action(self):
        action_list = self.legal_actions
        return np.random.choice(action_list)

    def bot_action(self):
        if np.random.rand() < self.prob_random_action_in_bot:
            return self.random_action()
        else:
            if self.bot_action_type == 'rule':
                return self.rule_bot.get_rule_bot_action(self.board, self._current_player)
            # elif self.bot_action_type == 'alpha_beta_pruning':
            #     return self.alpha_beta_bot.get_best_action(self.board, player_index=self.current_player_index)
            elif self.bot_action_type == 'mcts':
                return self.mcts_bot.get_actions(self.board, player_index=self.current_player_index)

    def action_to_string(self, action):
        """
        Overview:
            Convert an action number to a string representing the action.
        Arguments:
            - action: an integer from the action space.
        Returns:
            - String representing the action.
        """
        return f"Play column {action + 1}"

    def human_to_action(self):
        """
        Overview:
            For multiplayer games, ask the user for a legal action
            and return the corresponding action number.
        Returns:
            An integer from the action space.
        """
        # print(np.array(self.board).reshape(6, 7))
        while True:
            try:
                column = int(
                    input(
                        f"Enter the column to play for the player {self.current_player}: "
                    )
                )
                action = column - 1
                if action in self.legal_actions:
                    break
                else:
                    print("Wrong input, try again")
            except KeyboardInterrupt:
                print("exit")
                sys.exit(0)
            except Exception as e:
                print("Wrong input, try again")
        return action

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def __repr__(self) -> str:
        return "LightZero Connect4 Env"

    @property
    def current_player(self):
        return self._current_player

    @property
    def current_player_index(self):
        """
        Overview:
            current_player_index = 0, current_player = 1
            current_player_index = 1, current_player = 2
        """
        return 0 if self._current_player == 1 else 1

    @property
    def next_player(self):
        return self.players[0] if self._current_player == self.players[1] else self.players[1]

    @property
    def observation_space(self) -> spaces.Space:
        return self._observation_space

    @property
    def action_space(self) -> spaces.Space:
        return self._action_space

    @property
    def reward_space(self) -> spaces.Space:
        return self._reward_space

    def simulate_action(self, action):
        """
        Overview:
            execute action and get next_simulator_env. used in AlphaZero.
        Arguments:
            - action: an integer from the action space.
        Returns:
            - next_simulator_env: next simulator env after execute action.
        """
        if action not in self.legal_actions:
            raise ValueError("action {0} on board {1} is not legal".format(action, self.board))
        new_board = copy.deepcopy(self.board)
        piece = self.players.index(self._current_player) + 1
        for i in list(filter(lambda x: x % 7 == action, list(range(41, -1, -1)))):
            if new_board[i] == 0:
                new_board[i] = piece
                break
        if self.start_player_index == 0:
            start_player_index = 1  # self.players = [1, 2], start_player = 2, start_player_index = 1
        else:
            start_player_index = 0  # self.players = [1, 2], start_player = 1, start_player_index = 0
        next_simulator_env = copy.deepcopy(self)
        next_simulator_env.reset(start_player_index, init_state=new_board)
        return next_simulator_env

    def simulate_action_v2(self, board, start_player_index, action):
        """
        Overview:
            execute action from board and get new_board, new_legal_actions. used in alphabeta_pruning_bot.
        Arguments:
            - board (:obj:`np.array`): current board
            - start_player_index (:obj:`int`): start player index
            - action (:obj:`int`): action
        Returns:
            - new_board (:obj:`np.array`): new board
            - new_legal_actions (:obj:`list`): new legal actions
        """
        self.reset(start_player_index, init_state=board)
        if action not in self.legal_actions:
            raise ValueError("action {0} on board {1} is not legal".format(action, self.board))
        piece = self.players.index(self._current_player) + 1
        for i in list(filter(lambda x: x % 7 == action, list(range(41, -1, -1)))):
            if self.board[i] == 0:
                self.board[i] = piece
                break
        new_legal_actions = copy.deepcopy(self.legal_actions)
        new_board = copy.deepcopy(self.board)

        return new_board, new_legal_actions

    @staticmethod
    def create_collector_env_cfg(cfg: dict) -> List[dict]:
        collector_env_num = cfg.pop('collector_env_num')
        cfg = copy.deepcopy(cfg)
        return [cfg for _ in range(collector_env_num)]

    @staticmethod
    def create_evaluator_env_cfg(cfg: dict) -> List[dict]:
        evaluator_env_num = cfg.pop('evaluator_env_num')
        cfg = copy.deepcopy(cfg)
        # In eval phase, we use ``eval_mode`` to make agent play with the built-in bot to
        # evaluate the performance of the current agent.
        cfg.battle_mode = 'eval_mode'
        return [cfg for _ in range(evaluator_env_num)]

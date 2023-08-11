"""
Adapt the connect4 environment in PettingZoo (https://github.com/Farama-Foundation/PettingZoo) to the BaseEnv interface.
"""

import os
import sys
import copy
from ditk import logging

import numpy as np
import pygame
from ding.envs import BaseEnv, BaseEnvTimestep
from ding.utils import ENV_REGISTRY
from gym import spaces
from easydict import EasyDict

# from pettingzoo.utils.agent_selector import agent_selector
from zoo.board_games.alphabeta_pruning_bot import AlphaBetaPruningBot

@ENV_REGISTRY.register('connect4')
class Connect4Env(BaseEnv):
    config = dict(
        env_name="Connect4",
        battle_mode='self_play_mode',
        mcts_mode='self_play_mode',  # only used in AlphaZero
        bot_action_type='v0',  # {'v0', 'alpha_beta_pruning'}
        agent_vs_human=False,
        prob_random_agent=0,
        prob_expert_agent=0,
        channel_last=True,
        scale=False,
        stop_value=1,
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
        # The mode of interaction between the agent and the environment.
        assert self.battle_mode in ['self_play_mode', 'play_with_bot_mode', 'eval_mode']
        self.prob_random_agent = cfg.prob_random_agent
        self.prob_expert_agent = cfg.prob_expert_agent
        assert (self.prob_random_agent >= 0 and self.prob_expert_agent == 0) or (
                self.prob_random_agent == 0 and self.prob_expert_agent >= 0), \
            f'self.prob_random_agent:{self.prob_random_agent}, self.prob_expert_agent:{self.prob_expert_agent}'
        self.agent_vs_human = cfg.agent_vs_human
        self.bot_action_type = cfg.bot_action_type
        if 'alpha_beta_pruning' in self.bot_action_type:
            self.alpha_beta_pruning_player = AlphaBetaPruningBot(self, cfg, 'alpha_beta_pruning_player')
        
        self._env = self

        self.board = [0] * (6 * 7)

        self.players = [1, 2]
        # self.possible_agents = self.agents[:]

        # 这个变量的功能是否和battle_mode重复了###########################################
        self.mcts_mode = 'play_with_bot_mode'

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
            board_vals = np.array(self.board).reshape(6, 7)
            # cur_player = self.players.index(self._current_player)
            # opp_player = (cur_player + 1) % 2

            # cur_p_board = np.equal(board_vals, cur_player + 1)
            # opp_p_board = np.equal(board_vals, opp_player + 1)

            # observation = np.stack([cur_p_board, opp_p_board], axis=2).astype(np.int8)
            legal_moves = self.legal_actions

            action_mask = np.zeros(7, "int8")
            for i in legal_moves:
                action_mask[i] = 1

            if self.battle_mode == 'play_with_bot_mode' or self.battle_mode == 'eval_mode':
                return {"observation": self.current_state()[1], 
                        "action_mask": action_mask,
                        "board": copy.deepcopy(board_vals),
                        "current_player_index": self.players.index(self._current_player),
                        "to_play" : -1
                        }
            elif self.battle_mode == 'self_play_mode':
                return {"observation": self.current_state()[1], 
                        "action_mask": action_mask,
                        "board": copy.deepcopy(board_vals),
                        "current_player_index": self.players.index(self._current_player),
                        "to_play" : self._current_player
                        }
            
    # def observation_space(self, agent):
    #     return self.observation_spaces[agent]

    # def action_space(self, agent):
    #     return self.action_spaces[agent]

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

            timestep = self._player_step(action)
            if timestep.done:
                # The eval_episode_return is calculated from Player 1's perspective。
                # 不是很明白episode_reward在train的时候是怎么被调用的#########################
                timestep.info['eval_episode_return'] = -timestep.reward if timestep.obs['to_play'] == 1 else timestep.reward
            return timestep
        elif self.battle_mode == 'play_with_bot_mode':
            # player 1 battle with expert player 2

            # player 1's turn
            timestep_player1 = self._player_step(action)
            # self.env.render()
            if timestep_player1.done:
                # NOTE: in play_with_bot_mode, we must set to_play as -1, because we don't consider the alternation between players.
                # And the to_play is used in MCTS.
                timestep_player1.obs['to_play'] = -1
                return timestep_player1

            # player 2's turn
            bot_action = self.bot_action()
            # print('player 2 (computer player): ' + self.action_to_string(bot_action))
            timestep_player2 = self._player_step(bot_action)
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
            timestep_player1 = self._player_step(action)
            # self.env.render()
            if timestep_player1.done:
                # NOTE: in eval_mode, we must set to_play as -1, because we don't consider the alternation between players.
                # And the to_play is used in MCTS.
                timestep_player1.obs['to_play'] = -1
                return timestep_player1

            # player 2's turn
            if self.agent_vs_human:
                bot_action = self.human_to_action()
            else:
                bot_action = self.bot_action()
            # print('player 2 (computer player): ' + self.action_to_string(bot_action))
            timestep_player2 = self._player_step(bot_action)
            # the eval_episode_return is calculated from Player 1's perspective
            timestep_player2.info['eval_episode_return'] = -timestep_player2.reward
            timestep_player2 = timestep_player2._replace(reward=-timestep_player2.reward)

            timestep = timestep_player2
            # NOTE: in eval_mode, we must set to_play as -1, because we don't consider the alternation between players.
            # And the to_play is used in MCTS.
            timestep.obs['to_play'] = -1

            return timestep

    def _player_step(self, action):
        if action in self.legal_actions:
            piece = self.players.index(self._current_player) + 1
            for i in list(filter(lambda x: x % 7 == action, list(range(41, -1, -1)))):
                if self.board[i] == 0:
                    self.board[i] = piece
                    break
        else:
            logging.warning(
                f"You input illegal action: {action}, the legal_actions are {self.legal_actions}. "
                f"Now we randomly choice a action from self.legal_actions."
            )
            action = self.random_action()
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
        self.players = [1,2]
        self.start_player_index = start_player_index
        self._current_player = self.players[self.start_player_index]

        self._action_space = spaces.Discrete(7)
        self._reward_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self._observation_space = spaces.Dict(
                {
                    "observation": spaces.Box(low=0, high=1, shape=(3,6,7), dtype=np.int8),
                    "action_mask": spaces.Box(low=0, high=1, shape=(7,), dtype=np.int8),
                    # "board": spaces.Box(low=0, high=2, shape=(6,7), dtype=np.int8),
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


    def render(self):
        print(np.array(self.board).reshape(6, 7))

    def close(self):
        pass
    
    def get_done_winner(self):
        board = np.array(self.board).reshape(6, 7)
        for piece in [1,2]:
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
    


    def random_action(self):
        action_list = self.legal_actions
        return np.random.choice(action_list)
    
    def bot_action(self):
        # TODO
        pass

    def action_to_string(self, action):
        """
        Overview:
            Convert an action number to a string representing the action.
        Arguments:
            - action: an integer from the action space.
        Returns:
            - String representing the action.
        """
        return f"Play column {action+1}"

    # def set_game_result(self, result_val):
    #     for i, name in enumerate(self.agents):
    #         self.dones[name] = True
    #         result_coef = 1 if i == 0 else -1
    #         self.rewards[name] = result_val * result_coef
    #         self.infos[name] = {'legal_moves': []}

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
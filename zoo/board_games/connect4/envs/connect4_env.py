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

from pettingzoo.utils.agent_selector import agent_selector
from zoo.board_games.alphabeta_pruning_bot import AlphaBetaPruningBot

# def get_image(path):
#     from os import path as os_path

#     import pygame
#     cwd = os_path.dirname(__file__)
#     image = pygame.image.load(cwd + '/' + path)
#     sfc = pygame.Surface(image.get_size(), flags=pygame.SRCALPHA)
#     sfc.blit(image, (0, 0))
#     return sfc

@ENV_REGISTRY.register('Connect4')
class Connect4Env(BaseEnv):

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
        
        self.board = [0] * (6 * 7)

        self.agents = ["player_0", "player_1"]
        self.possible_agents = self.agents[:]

        # 这个变量有啥必要？？？？？？？？？？？？？？？？？？、
        self.mcts_mode = 'play_with_bot_mode'

        self.action_spaces = {i: spaces.Discrete(7) for i in self.agents}
        self.observation_spaces = {
            i: spaces.Dict(
                {
                    "observation": spaces.Box(
                        low=0, high=1, shape=(6, 7, 2), dtype=np.int8
                    ),
                    "action_mask": spaces.Box(low=0, high=1, shape=(7,), dtype=np.int8),
                }
            )
            for i in self.agents
        }

    def observe(self, agent):
            board_vals = np.array(self.board).reshape(6, 7)
            cur_player = self.possible_agents.index(agent)
            opp_player = (cur_player + 1) % 2

            cur_p_board = np.equal(board_vals, cur_player + 1)
            opp_p_board = np.equal(board_vals, opp_player + 1)

            # 为什么观测空间要设置成这样？
            observation = np.stack([cur_p_board, opp_p_board], axis=2).astype(np.int8)
            legal_moves = self._legal_moves() if agent == self.agent_selection else []

            action_mask = np.zeros(7, "int8")
            for i in legal_moves:
                action_mask[i] = 1

            return {"observation": observation, 
                    "action_mask": action_mask,
                    "board": copy.deepcopy(board_vals),
                    "current_player_index": self.agents.index(self.agent_selection),
                    "to_play" : self.agent_selection
                    }

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def _legal_moves(self):
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
                # 意义不明这边，维护环境的reward有啥用啊,MCTS的时候不是会再算一遍reward嘛
                timestep.info['eval_episode_return'] = -timestep.reward if timestep.obs['to_play'] == "player_0" else timestep.reward
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
        if action in self._legal_moves():
            piece = self.agents.index(self.agent_selection) + 1
            for i in list(filter(lambda x: x % 7 == action, list(range(41, -1, -1)))):
                if self.board[i] == 0:
                    self.board[i] = piece
                    break
        else:
            logging.warning(
                f"You input illegal action: {action}, the legal_actions are {self._legal_moves()}. "
                f"Now we randomly choice a action from self.legal_actions."
            )
            action = self.random_action()
            piece = self.agents.index(self.agent_selection) + 1
            for i in list(filter(lambda x: x % 7 == action, list(range(41, -1, -1)))):
                if self.board[i] == 0:
                    self.board[i] = piece
                    break

        next_agent = self._agent_selector.next()

        winner = self.check_for_winner()

        # check if there is a winner
        if winner:
            self.rewards[self.agent_selection] += 1
            self.rewards[next_agent] -= 1
            reward = np.array(1).astype(np.float32)
            self.dones = {i: True for i in self.agents}
        # check if there is a tie
        elif all(x in [1, 2] for x in self.board):
            # once either play wins or there is a draw, game over, both players are done
            self.dones = {i: True for i in self.agents}
            reward = np.array(0).astype(np.float32)
        else: reward = np.array(0).astype(np.float32)

    
        info = {'next player to play': next_agent}

        self.agent_selection = next_agent

        done = winner
        if done:
            info['eval_episode_return'] = reward
            # print('tictactoe one episode done: ', info)

        obs = self.observe(next_agent)

        # 想想这里是否还需要维护self.reward这些变量
        return BaseEnvTimestep(obs, reward, done, info)

    def reset(self, seed=None, options=None):
        # reset environment
        self.board = [0] * (6 * 7)

        self.agents = self.possible_agents[:]
        self.rewards = {i: 0 for i in self.agents}
        self._cumulative_rewards = {name: 0 for name in self.agents}
        self.dones = {i: False for i in self.agents}
        self.infos = {i: {} for i in self.agents}

        self._agent_selector = agent_selector(self.agents)

        #这边估计要写一个指定首发玩家的代码
        self.agent_selection = self._agent_selector.reset()

        for agent, reward in self.rewards.items():
            self._cumulative_rewards[agent] += reward

        agent = self.agent_selection
        self.current_player_index = self.agents.index(agent)
        obs = self.observe(agent)
        return obs


    def render(self):
        print(np.array(self.board).reshape(6, 7))

    def close(self):
        pass
    
    def check_for_winner(self):
        board = np.array(self.board).reshape(6, 7)
        piece = self.agents.index(self.agent_selection) + 1

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
                    return True

        # Check vertical locations for win
        for c in range(column_count):
            for r in range(row_count - 3):
                if (
                    board[r][c] == piece
                    and board[r + 1][c] == piece
                    and board[r + 2][c] == piece
                    and board[r + 3][c] == piece
                ):
                    return True

        # Check positively sloped diagonals
        for c in range(column_count - 3):
            for r in range(row_count - 3):
                if (
                    board[r][c] == piece
                    and board[r + 1][c + 1] == piece
                    and board[r + 2][c + 2] == piece
                    and board[r + 3][c + 3] == piece
                ):
                    return True

        # Check negatively sloped diagonals
        for c in range(column_count - 3):
            for r in range(3, row_count):
                if (
                    board[r][c] == piece
                    and board[r - 1][c + 1] == piece
                    and board[r - 2][c + 2] == piece
                    and board[r - 3][c + 3] == piece
                ):
                    return True

        return False
    


    def random_action(self):
        action_list = self._legal_moves()
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

    def set_game_result(self, result_val):
        for i, name in enumerate(self.agents):
            self.dones[name] = True
            result_coef = 1 if i == 0 else -1
            self.rewards[name] = result_val * result_coef
            self.infos[name] = {'legal_moves': []}

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def __repr__(self) -> str:
        return "LightZero Connect4 Env"
    
    @property
    def current_player(self):
        return self.current_player_index

    @property
    def to_play(self):
        return self.current_player_index
    


import copy
import os
import sys
from functools import lru_cache
from typing import List, Any

import gymnasium as gym
import imageio
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pygame
from ding.envs import BaseEnv, BaseEnvTimestep
from ding.utils import ENV_REGISTRY
from ditk import logging
from easydict import EasyDict
from zoo.board_games.gomoku.envs.get_done_winner_cython import get_done_winner_cython
from zoo.board_games.gomoku.envs.legal_actions_cython import legal_actions_cython

from zoo.board_games.alphabeta_pruning_bot import AlphaBetaPruningBot
from zoo.board_games.gomoku.envs.gomoku_rule_bot_v0 import GomokuRuleBotV0
from zoo.board_games.gomoku.envs.gomoku_rule_bot_v1 import GomokuRuleBotV1


@lru_cache(maxsize=512)
def _legal_actions_func_lru(board_size, board_tuple):
    # Convert tuple to NumPy array.
    board_array = np.array(board_tuple, dtype=np.int32)
    # Convert NumPy array to memory view.
    board_view = board_array.view(dtype=np.int32).reshape(board_array.shape)
    return legal_actions_cython(board_size, board_view)


@lru_cache(maxsize=512)
def _get_done_winner_func_lru(board_size, board_tuple):
    # Convert tuple to NumPy array.
    board_array = np.array(board_tuple, dtype=np.int32)
    # Convert NumPy array to memory view.
    board_view = board_array.view(dtype=np.int32).reshape(board_array.shape)
    return get_done_winner_cython(board_size, board_view)


@ENV_REGISTRY.register('gomoku')
class GomokuEnv(BaseEnv):
    """
    Overview:
        A Gomoku environment that inherits from the BaseEnv. This environment can be used for training and
        evaluating AI players for the game of Gomoku.

    .. note::
        For the latest macOS, you should set context='spawn' or 'forkserver' in ding/envs/env_manager/subprocess_env_manager.py
        to be able to use subprocess env_manager.
    """

    config = dict(
        # (str) The name of the environment registered in the environment registry.
        env_name="Gomoku",
        # (int) The size of the board.
        board_size=6,
        # (str) The mode of the environment when take a step.
        battle_mode='self_play_mode',
        # (str) The mode of the environment when doing the MCTS.
        mcts_mode='self_play_mode',  # only used in AlphaZero
        # (str) The render mode. Options are 'None', 'state_realtime_mode', 'image_realtime_mode' or 'image_savefile_mode'.
        # If None, then the game will not be rendered.
        render_mode=None,
        # (str or None) The directory in which to save the replay file. If None, the file is saved in the current directory.
        replay_path=None,
        # (float) The scale of the render screen.
        screen_scaling=9,
        # (bool) Whether to use the 'channel last' format for the observation space. If False, 'channel first' format is used.
        channel_last=False,
        # (bool) Whether to scale the observation.
        scale=True,
        # (bool) Whether to let human to play with the agent when evaluating. If False, then use the bot to evaluate the agent.
        agent_vs_human=False,
        # (str) The type of the bot of the environment.
        bot_action_type='v1',  # {'v0', 'v1', 'alpha_beta_pruning'}, 'v1' is faster and stronger than 'v0' now.
        # (float) The probability that a random agent is used instead of the learning agent.
        prob_random_agent=0,
        # (float) The probability that a random action will be taken when calling the bot.
        prob_random_action_in_bot=0.,
        # (float) The stop value when training the agent. If the evalue return reach the stop value, then the training will stop.
        stop_value=2,
        # (bool) Whether to use the MCTS ctree in AlphaZero. If True, then the AlphaZero MCTS ctree will be used.
        alphazero_mcts_ctree=False,
    )

    @classmethod
    def default_config(cls: type) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    @property
    def legal_actions(self):
        # Convert NumPy arrays to nested tuples to make them hashable.
        return _legal_actions_func_lru(self.board_size, tuple(map(tuple, self.board)))

    # only for evaluation speed
    @property
    def legal_actions_cython(self):
        # Convert tuple to NumPy array.
        board_array = np.array(tuple(map(tuple, self.board)), dtype=np.int32)
        # Convert NumPy array to memory view.
        board_view = board_array.view(dtype=np.int32).reshape(board_array.shape)
        return legal_actions_cython(self.board_size, board_view)

    # only for evaluation speed
    @property
    def legal_actions_cython_lru(self):
        # Convert NumPy arrays to nested tuples to make them hashable.
        return _legal_actions_func_lru(self.board_size, tuple(map(tuple, self.board)))

    def get_done_winner(self):
        # Convert NumPy arrays to nested tuples to make them hashable.
        return _get_done_winner_func_lru(self.board_size, tuple(map(tuple, self.board)))

    def __init__(self, cfg: dict = None):
        self.cfg = cfg
        self.battle_mode = cfg.battle_mode
        # The mode of interaction between the agent and the environment.
        assert self.battle_mode in ['self_play_mode', 'play_with_bot_mode', 'eval_mode']
        # The mode of MCTS is only used in AlphaZero.
        self.mcts_mode = 'self_play_mode'

        self.board_size = cfg.board_size
        self.prob_random_agent = cfg.prob_random_agent
        self.prob_random_action_in_bot = cfg.prob_random_action_in_bot
        self.channel_last = cfg.channel_last
        self.scale = cfg.scale
        self.agent_vs_human = cfg.agent_vs_human
        self.bot_action_type = cfg.bot_action_type

        # Set the parameters about replay render.
        self.screen_scaling = cfg.screen_scaling
        # options = {None, 'state_realtime_mode', 'image_realtime_mode', 'image_savefile_mode'}
        self.render_mode = cfg.render_mode
        self.replay_name_suffix = "test"
        self.replay_path = cfg.replay_path
        self.replay_format = 'gif'  # 'mp4' #
        self.screen = None
        self.frames = []

        self.players = [1, 2]
        self._current_player = 1
        self.board_markers = [str(i + 1) for i in range(self.board_size)]
        self.total_num_actions = self.board_size * self.board_size
        self.gomoku_rule_bot_v1 = GomokuRuleBotV1()
        self._env = self

        if self.bot_action_type == 'alpha_beta_pruning':
            self.alpha_beta_pruning_player = AlphaBetaPruningBot(self, cfg, 'alpha_beta_pruning_player')
        elif self.bot_action_type == 'v0':
            self.rule_bot = GomokuRuleBotV0(self, self._current_player)
        self.alphazero_mcts_ctree = cfg.alphazero_mcts_ctree
        if not self.alphazero_mcts_ctree:
            # plt is not work in mcts_ctree mode
            self.fig, self.ax = plt.subplots(figsize=(self.board_size, self.board_size))
            plt.ion()

    def reset(self, start_player_index=0, init_state=None, katago_policy_init=False, katago_game_state=None):
        """
        Overview:
            This method resets the environment and optionally starts with a custom state specified by 'init_state'.
        Arguments:
            - start_player_index (:obj:`int`, optional): Specifies the starting player. The players are [1,2] and
                their corresponding indices are [0,1]. Defaults to 0.
            - init_state (:obj:`Any`, optional): The custom starting state. If provided, the game starts from this state.
                Defaults to None.
            - katago_policy_init (:obj:`bool`, optional): This parameter is used to maintain compatibility with the
                handling of 'katago' related parts in 'alphazero_mcts_ctree' in Go. Defaults to False.
            - katago_game_state (:obj:`Any`, optional): This parameter is similar to 'katago_policy_init' and is used to
                maintain compatibility with 'katago' in 'alphazero_mcts_ctree'. Defaults to None.
        """
        if self.alphazero_mcts_ctree and init_state is not None:
            # Convert byte string to np.ndarray
            init_state = np.frombuffer(init_state, dtype=np.int32)

        self._observation_space = gym.spaces.Box(
            low=0, high=2, shape=(self.board_size, self.board_size, 3), dtype=np.int32
        )
        self._action_space = gym.spaces.Discrete(self.board_size ** 2)
        self._reward_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.start_player_index = start_player_index
        self._current_player = self.players[self.start_player_index]
        if init_state is not None:
            self.board = np.array(copy.deepcopy(init_state), dtype="int32")
            if self.alphazero_mcts_ctree:
                self.board = self.board.reshape((self.board_size, self.board_size))
        else:
            self.board = np.zeros((self.board_size, self.board_size), dtype="int32")
        action_mask = np.zeros(self.total_num_actions, 'int8')
        action_mask[self.legal_actions] = 1
        if self.battle_mode == 'play_with_bot_mode' or self.battle_mode == 'eval_mode':
            # In ``play_with_bot_mode`` and ``eval_mode``, we need to set the "to_play" parameter in the "obs" dict to -1,
            # because we don't take into account the alternation between players.
            # The "to_play" parameter is used in the MCTS algorithm.
            obs = {
                'observation': self.current_state()[1],
                'action_mask': action_mask,
                'board': copy.deepcopy(self.board),
                'current_player_index': self.start_player_index,
                'to_play': -1
            }
        elif self.battle_mode == 'self_play_mode':
            # In the "self_play_mode", we set to_play=self.current_player in the "obs" dict,
            # which is used to differentiate the alternation of 2 players in the game when calculating Q in the MCTS algorithm.
            obs = {
                'observation': self.current_state()[1],
                'action_mask': action_mask,
                'board': copy.deepcopy(self.board),
                'current_player_index': self.start_player_index,
                'to_play': self.current_player
            }

        # Render the beginning state of the game.
        if self.render_mode is not None:
            self.render(self.render_mode)

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
            self.board = np.zeros((self.board_size, self.board_size), dtype="int32")

    def step(self, action):
        if self.battle_mode == 'self_play_mode':
            if np.random.rand() < self.prob_random_agent:
                action = self.random_action()
            timestep = self._player_step(action)
            if timestep.done:
                # The eval_episode_return is calculated from Player 1's perspective.
                timestep.info['eval_episode_return'] = -timestep.reward if timestep.obs[
                                                                               'to_play'] == 1 else timestep.reward
            return timestep
        elif self.battle_mode == 'play_with_bot_mode':
            # player 1 battle with expert player 2

            # player 1's turn
            timestep_player1 = self._player_step(action)
            # print('player 1 (efficientzero player): ' + self.action_to_string(action))  # Note: visualize
            if timestep_player1.done:
                # in play_with_bot_mode, we set to_play as None/-1, because we don't consider the alternation between players
                timestep_player1.obs['to_play'] = -1
                return timestep_player1

            # player 2's turn
            bot_action = self.bot_action()
            # print('player 2 (expert player): ' + self.action_to_string(bot_action))  # Note: visualize
            timestep_player2 = self._player_step(bot_action)
            # self.render()  # Note: visualize
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

            self._env.render(self.render_mode)
            # player 1's turn
            timestep_player1 = self._player_step(action)
            self._env.render(self.render_mode)
            if self.agent_vs_human:
                print('player 1 (agent): ' + self.action_to_string(action))  # Note: visualize
                self.render(mode="image_realtime_mode")

            if timestep_player1.done:
                # in eval_mode, we set to_play as None/-1, because we don't consider the alternation between players
                timestep_player1.obs['to_play'] = -1
                return timestep_player1

            # player 2's turn
            if self.agent_vs_human:
                bot_action = self.human_to_action()
            else:
                bot_action = self.bot_action()
                # bot_action = self.random_action()

            timestep_player2 = self._player_step(bot_action)
            self._env.render(self.render_mode)
            if self.agent_vs_human:
                print('player 2 (human): ' + self.action_to_string(bot_action))  # Note: visualize
                self.render(mode="image_realtime_mode")

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
        done, winner = self.get_done_winner()

        reward = np.array(float(winner == self.current_player)).astype(np.float32)
        info = {'next player to play': self.to_play}
        """
        NOTE: here exchange the player
        """
        self.current_player = self.to_play

        # The following code will save the rendered images in both env step in collect/eval phase and the env step in
        # simulated mcts.
        # if self.render_mode is not None:
        #     self.render(self.render_mode)

        if done:
            info['eval_episode_return'] = reward
            self._env.render(self.render_mode)
            if self.render_mode == 'image_savefile_mode':
                self.save_render_output(replay_name_suffix=self.replay_name_suffix, replay_path=self.replay_path,
                                        format=self.replay_format)

        action_mask = np.zeros(self.total_num_actions, 'int8')
        action_mask[self.legal_actions] = 1
        obs = {
            'observation': self.current_state()[1],
            'action_mask': action_mask,
            'board': copy.deepcopy(self.board),
            'current_player_index': self.players.index(self.current_player),
            'to_play': self.current_player
        }
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
        if self.scale:
            scale_obs = copy.deepcopy(raw_obs / 2)
        else:
            scale_obs = copy.deepcopy(raw_obs)

        if self.channel_last:
            # move channel dim to last axis
            # (C, W, H) -> (W, H, C)
            # e.g. (3, 6, 6) -> (6, 6, 3)
            return np.transpose(raw_obs, [1, 2, 0]), np.transpose(scale_obs, [1, 2, 0])
        else:
            # (C, W, H) e.g. (3, 6, 6)
            return raw_obs, scale_obs

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
            if self.bot_action_type == 'v0':
                return self.rule_bot.get_rule_bot_action(self.board, self._current_player)
            elif self.bot_action_type == 'v1':
                return self.rule_bot_v1()
            elif self.bot_action_type == 'alpha_beta_pruning':
                return self.bot_action_alpha_beta_pruning()

    def bot_action_alpha_beta_pruning(self):
        action = self.alpha_beta_pruning_player.get_best_action(self.board, player_index=self.current_player_index)
        return action

    def rule_bot_v1(self):
        action_mask = np.zeros(self.total_num_actions, 'int8')
        action_mask[self.legal_actions] = 1
        # NOTE: we use the original raw_obs for ``gomoku_rule_bot_v1.get_action()``
        obs = {'observation': self.current_state()[0], 'action_mask': action_mask}
        return self.gomoku_rule_bot_v1.get_action(obs)

    @property
    def current_player(self):
        return self._current_player

    @property
    def current_player_index(self):
        """
        current_player_index = 0, current_player = 1
        current_player_index = 1, current_player = 2
        """
        return 0 if self._current_player == 1 else 1

    @property
    def to_play(self):
        return self.players[0] if self.current_player == self.players[1] else self.players[1]

    @property
    def current_player_to_compute_bot_action(self):
        """
        Overview: to compute expert action easily.
        """
        return -1 if self.current_player == 1 else 1

    def human_to_action(self):
        """
        Overview:
            For multiplayer games, ask the user for a legal action
            and return the corresponding action number.
        Returns:
            An integer from the action space.
        """
        # print(self.board)
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

    def simulate_action(self, action):
        """
        Overview:
            execute action and get next_simulator_env. used in AlphaZero.
        Returns:
            Returns Gomoku instance.
        """
        if action not in self.legal_actions:
            raise ValueError("action {0} on board {1} is not legal".format(action, self.board))
        new_board = copy.deepcopy(self.board)
        row, col = self.action_to_coord(action)
        new_board[row, col] = self.current_player
        if self.start_player_index == 0:
            start_player_index = 1  # self.players = [1, 2], start_player = 2, start_player_index = 1
        else:
            start_player_index = 0  # self.players = [1, 2], start_player = 1, start_player_index = 0
        next_simulator_env = copy.deepcopy(self)
        next_simulator_env.reset(start_player_index, init_state=new_board)  # index
        return next_simulator_env

    def simulate_action_v2(self, board, start_player_index, action):
        """
        Overview:
            execute action from board and get new_board, new_legal_actions. used in AlphaZero.
        Returns:
            - new_board (:obj:`np.array`):
            - new_legal_actions (:obj:`np.array`):
        """
        self.reset(start_player_index, init_state=board)
        if action not in self.legal_actions:
            raise ValueError("action {0} on board {1} is not legal".format(action, self.board))
        row, col = self.action_to_coord(action)
        self.board[row, col] = self.current_player
        new_legal_actions = copy.deepcopy(self.legal_actions)
        new_board = copy.deepcopy(self.board)
        return new_board, new_legal_actions

    def clone(self):
        return copy.deepcopy(self)

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def draw_board(self):
        """
        Overview:
            This method draws the Gomoku board using matplotlib.
        """

        # Clear the previous board
        self.ax.clear()

        # Set the limits of the x and y axes
        self.ax.set_xlim(0, self.board_size + 1)
        self.ax.set_ylim(self.board_size + 1, 0)

        # Set the board background color
        self.ax.set_facecolor('peachpuff')

        # Draw the grid lines
        for i in range(self.board_size + 1):
            self.ax.plot([i + 1, i + 1], [1, self.board_size], color='black')
            self.ax.plot([1, self.board_size], [i + 1, i + 1], color='black')

    def render(self, mode="state_realtime_mode"):
        """
        Overview:
            The render method is used to draw the current state of the game. The rendering mode can be
            set according to the needs of the user.
        Arguments:
            - mode (str): Rendering mode, options are "state_realtime_mode", "image_realtime_mode",
              and "image_savefile_mode".
        """
        if mode is None:
            return
        # Print the state of the board directly
        if mode == "state_realtime_mode":
            print(np.array(self.board).reshape(self.board_size, self.board_size))
            return
        # Render the game as an image
        elif mode == "image_realtime_mode" or mode == "image_savefile_mode":
            self.draw_board()
            # Draw the pieces on the board
            for x in range(self.board_size):
                for y in range(self.board_size):
                    if self.board[x][y] == 1:  # Black piece
                        circle = patches.Circle((y + 1, x + 1), 0.4, edgecolor='black',
                                                facecolor='black', zorder=3)
                        self.ax.add_patch(circle)
                    elif self.board[x][y] == 2:  # White piece
                        circle = patches.Circle((y + 1, x + 1), 0.4, edgecolor='black',
                                                facecolor='white', zorder=3)
                        self.ax.add_patch(circle)
            # Set the title of the game
            plt.title('Agent vs. Human: ' + ('Black Turn' if self.current_player == 1 else 'White Turn'))
            # If in realtime mode, draw and pause briefly
            if mode == "image_realtime_mode":
                plt.draw()
                plt.pause(0.1)
            # In savefile mode, save the current frame to the frames list
            elif mode == "image_savefile_mode":
                # Save the current frame to the frames list.
                self.fig.canvas.draw()
                image = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype='uint8')

                # Get the width and height of the figure
                width, height = self.fig.get_size_inches() * self.fig.get_dpi()
                width = int(width)
                height = int(height)
                image = image.reshape(height, width, 3)

                # image = image.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
                self.frames.append(image)

    def close(self):
        """
        Overview:
            This method is used to display the final game board to the user and turn off interactive
            mode in matplotlib.
        """
        plt.ioff()
        plt.show()

    def render_for_b15(self, mode: str = None) -> None:
        """
        Overview:
            Renders the Gomoku (Five in a Row) game environment. Now only support board_size=15.
        Arguments:
            - mode (:obj:`str`): The mode to render with. Options are: None, 'human', 'state_realtime_mode',
                'image_realtime_mode', 'image_savefile_mode'.
        """
        # 'state_realtime_mode' mode, print the current game board for rendering.
        if mode == "state_realtime_mode":
            print(np.array(self.board).reshape(self.board_size, self.board_size))
            return
        else:
            # Other modes, use a screen for rendering.
            screen_width = self.board_size * self.screen_scaling
            screen_height = self.board_size * self.screen_scaling
            pygame.init()
            self.screen = pygame.Surface((screen_width, screen_height))

            # Load and scale all of the necessary images.
            tile_size = screen_width / self.board_size

            black_chip = self.get_image(os.path.join("img", "Gomoku_BlackPiece.png"))
            black_chip = pygame.transform.scale(
                black_chip, (int(tile_size), int(tile_size))
            )

            white_chip = self.get_image(os.path.join("img", "Gomoku_WhitePiece.png"))
            white_chip = pygame.transform.scale(
                white_chip, (int(tile_size), int(tile_size))
            )

            board_img = self.get_image(os.path.join("img", "GomokuBoard.png"))
            board_img = pygame.transform.scale(
                board_img, (int(screen_width), int(screen_height))
            )

            self.screen.blit(board_img, (0, 0))

            # Blit the necessary chips and their positions.
            for row in range(self.board_size):
                for col in range(self.board_size):
                    if self.board[row][col] == 1:  # Black piece
                        self.screen.blit(
                            black_chip,
                            (
                                col * tile_size,
                                row * tile_size,
                            ),
                        )
                    elif self.board[row][col] == 2:  # White piece
                        self.screen.blit(
                            white_chip,
                            (
                                col * tile_size,
                                row * tile_size,
                            ),
                        )
            if mode == "image_realtime_mode":
                surface_array = pygame.surfarray.pixels3d(self.screen)
                surface_array = np.transpose(surface_array, (1, 0, 2))
                plt.imshow(surface_array)
                plt.draw()
                plt.pause(0.001)
            elif mode == "image_savefile_mode":
                # Draw the observation and save to frames.
                observation = np.array(pygame.surfarray.pixels3d(self.screen))
                self.frames.append(np.transpose(observation, axes=(1, 0, 2)))

            self.screen = None

            return None

    def save_render_output(self, replay_name_suffix: str = '', replay_path: str = None, format: str = 'gif') -> None:
        """
        Overview:
            Save the rendered frames as an output file.
        Arguments:
            - replay_name_suffix (:obj:`str`): The suffix to be added to the replay filename.
            - replay_path (:obj:`str`): The path to save the replay file. If None, the default filename will be used.
            - format (:obj:`str`): The format of the output file. Options are 'gif' or 'mp4'.
        """
        # At the end of the episode, save the frames.
        if replay_path is None:
            filename = f'gomoku_{self.board_size}_{replay_name_suffix}.{format}'
        else:
            if not os.path.exists(replay_path):
                os.makedirs(replay_path)
            filename = replay_path+f'/gomoku_{self.board_size}_{replay_name_suffix}.{format}'

        if format == 'gif':
            # Save frames as a GIF with a duration of 0.1 seconds per frame.
            # imageio.mimsave(filename, self.frames, 'GIF', duration=0.1)
            imageio.mimsave(filename, self.frames, 'GIF', fps=30, subrectangles=True)
        elif format == 'mp4':
            # Save frames as an MP4 video with a frame rate of 30 frames per second.
            imageio.mimsave(filename, self.frames, fps=30, codec='mpeg4')

        else:
            raise ValueError("Unsupported format: {}".format(format))
        logging.info("Saved output to {}".format(filename))
        self.frames = []

    def render_naive(self, mode="human"):
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

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        return self._action_space

    @property
    def reward_space(self) -> gym.spaces.Space:
        return self._reward_space

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
        # In eval phase, we use ``eval_mode`` to make agent play with the built-in bot to
        # evaluate the performance of the current agent.
        cfg.battle_mode = 'eval_mode'
        return [cfg for _ in range(evaluator_env_num)]

    def __repr__(self) -> str:
        return "LightZero Gomoku Env"

    def close(self) -> None:
        pass

    def get_image(self, path: str) -> Any:
        from os import path as os_path
        cwd = os_path.dirname(__file__)
        image = pygame.image.load(cwd + "/" + path)
        sfc = pygame.Surface(image.get_size(), flags=pygame.SRCALPHA)
        sfc.blit(image, (0, 0))
        return sfc

import copy
import logging
import sys
from typing import List

import gym
import imageio
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ding.envs import BaseEnvTimestep
from ding.torch_utils import to_ndarray
from ding.utils import ENV_REGISTRY
from easydict import EasyDict
from gym import spaces
from gym.utils import seeding


@ENV_REGISTRY.register('game_2048')
class Game2048Env(gym.Env):
    """
    Overview:
        The Game2048Env is a gym environment implementation of the 2048 game. The goal of the game is to slide numbered tiles
        on a grid to combine them and create a tile with the number 2048 (or larger). The environment provides an interface to interact with
        the game and receive observations, rewards, and game status information.

    Interfaces:
      - reset(init_board=None, add_random_tile_flag=True):
          Resets the game board and starts a new episode. It returns the initial observation of the game.
      - step(action):
          Advances the game by one step based on the provided action. It returns the new observation, reward, game status,
          and additional information.
      - render(mode='human'):
          Renders the current state of the game for visualization purposes.
    MDP Definition:
      - Observation Space:
          The observation space is a 4x4 grid representing the game board. Each cell in the grid can contain a number from
          0 to 2048. The observation can be in different formats based on the 'obs_type' parameter in the environment configuration.
          - If 'obs_type' is set to 'encode_observation' (default):
              The observation is a 3D numpy array of shape (4, 4, 16). Each cell in the array is represented as a one-hot vector
              encoding the value of the tile in that cell. The one-hot vector has a length of 16, representing the possible tile
              values from 0 to 2048. The first element in the one-hot vector corresponds to an empty cell (0 value).
          - If 'obs_type' is set to 'dict_encoded_board':
              The observation is a dictionary with the following keys:
                  - 'observation': A 3D numpy array representing the game board as described above.
                  - 'action_mask': A binary mask representing the legal actions that can be taken in the current state.
                  - 'to_play': A placeholder value (-1) indicating the current player (not applicable in this game).
                  - 'chance': A placeholder value representing the chance outcome (not applicable in this game).
          - If 'obs_type' is set to 'raw_board':
              The observation is the raw game board as a 2D numpy array of shape (4, 4).
      - Action Space:
          The action space is a discrete space with 4 possible actions:
              - 0: Move Up
              - 1: Move Right
              - 2: Move Down
              - 3: Move Left
      - Reward:
          The reward depends on the 'reward_type' parameter in the environment configuration.
          - If 'reward_type' is set to 'raw':
              The reward is a floating-point number representing the immediate reward obtained from the last action.
          - If 'reward_type' is set to 'merged_tiles_plus_log_max_tile_num':
              The reward is a floating-point number representing the number of merged tiles in the current step.
              If the maximum tile number on the board after the step is greater than the previous maximum tile number,
              the reward is further adjusted by adding the logarithm of the new maximum tile number multiplied by 0.1.
              The reward is calculated as follows: reward = num_of_merged_tiles + (log2(new_max_tile_num) * 0.1)
              If the new maximum tile number is the same as the previous maximum tile number, the reward does not
              include the second term. Note: This reward type requires 'reward_normalize' to be set to False.
      - Done:
          The game ends when one of the following conditions is met:
              - The maximum tile number (configured by 'max_tile') is reached.
              - There are no legal moves left.
              - The number of steps in the episode exceeds the maximum episode steps (configured by 'max_episode_steps').
      - Additional Information:
          The 'info' dictionary returned by the 'step' method contains additional information about the current state.
          The following keys are included in the dictionary:
              - 'raw_reward': The raw reward obtained from the last action.
              - 'current_max_tile_num': The current maximum tile number on the board.
      - Rendering:
          The 'render' method can be used to visualize the current state of the game. It supports two rendering modes:
              - 'human': Renders the game in a text-based format in the console.
              - 'rgb_array_render': Renders the game as an RGB image.
          Note: The rendering mode is set to 'human' by default.
      """

    # The default_config for game 2048 env.
    config = dict(
        # (str) The name of the environment registered in the environment registry.
        env_name="game_2048",
        # (bool) Whether to save the replay of the game.
        save_replay=False,
        # (str) The format in which to save the replay. 'gif' is a popular choice.
        replay_format='gif',
        # (str) A suffix for the replay file name to distinguish it from other files.
        replay_name_suffix='eval',
        # (str or None) The directory in which to save the replay file. If None, the file is saved in the current directory.
        replay_path=None,
        # (bool) Whether to render the game in real time. Useful for debugging, but can slow down training.
        render_real_time=False,
        # (bool) Whether to scale the actions. If True, actions are divided by the action space size.
        act_scale=True,
        # (bool) Whether to use the 'channel last' format for the observation space. If False, 'channel first' format is used.
        channel_last=True,
        # (str) The type of observation to use. Options are 'raw_board', 'raw_encoded_board', and 'dict_encoded_board'.
        obs_type='dict_encoded_board',
        # (bool) Whether to normalize rewards. If True, rewards are divided by the maximum possible reward.
        reward_normalize=False,
        # (float) The factor to scale rewards by when reward normalization is used.
        reward_norm_scale=100,
        # (str) The type of reward to use. 'raw' means the raw game score. 'merged_tiles_plus_log_max_tile_num' is an alternative.
        reward_type='raw',
        # (int) The maximum tile number in the game. A game is won when this tile appears. 2**11=2048, 2**16=65536
        max_tile=int(2 ** 16),
        # (int) The number of steps to delay rewards by. If > 0, the agent only receives a reward every this many steps.
        delay_reward_step=0,
        # (float) The probability that a random agent is used instead of the learning agent.
        prob_random_agent=0.,
        # (int) The maximum number of steps in an episode.
        max_episode_steps=int(1e6),
        # (bool) Whether to collect data during the game.
        is_collect=True,
        # (bool) Whether to ignore legal actions. If True, the agent can take any action, even if it's not legal.
        ignore_legal_actions=True,
        # (bool) Whether to flatten the observation space. If True, the observation space is a 1D array instead of a 2D grid.
        need_flatten=False,
        # (int) The number of possible tiles that can appear after each move.
        num_of_possible_chance_tile=2,
        # (numpy array) The possible tiles that can appear after each move.
        possible_tiles=np.array([2, 4]),
        # (numpy array) The probabilities corresponding to each possible tile.
        tile_probabilities=np.array([0.9, 0.1]),
    )

    @classmethod
    def default_config(cls: type) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        self._init_flag = False
        self._env_name = cfg.env_name
        self.replay_format = cfg.replay_format
        self.replay_name_suffix = cfg.replay_name_suffix
        self.replay_path = cfg.replay_path
        self.save_replay = cfg.save_replay
        self.render_real_time = cfg.render_real_time

        self._save_replay_count = 0
        self.channel_last = cfg.channel_last
        self.obs_type = cfg.obs_type
        self.reward_type = cfg.reward_type
        self.reward_normalize = cfg.reward_normalize
        self.reward_norm_scale = cfg.reward_norm_scale
        assert self.reward_type in ['raw', 'merged_tiles_plus_log_max_tile_num']
        assert self.reward_type == 'raw' or (
                self.reward_type == 'merged_tiles_plus_log_max_tile_num' and self.reward_normalize is False)
        self.max_tile = cfg.max_tile
        # Define the maximum tile that will end the game (e.g. 2048). None means no limit.
        # This does not affect the state returned.
        assert self.max_tile is None or isinstance(self.max_tile, int)

        self.max_episode_steps = cfg.max_episode_steps
        self.is_collect = cfg.is_collect
        self.ignore_legal_actions = cfg.ignore_legal_actions
        self.need_flatten = cfg.need_flatten
        self.chance = 0
        self.chance_space_size = 16  # 32 for 2 and 4, 16 for 2
        self.max_tile_num = 0
        self.size = 4
        self.w = self.size
        self.h = self.size
        self.squares = self.size * self.size
        self.episode_return = 0
        # Members for gym implementation:
        self._action_space = spaces.Discrete(4)
        self._observation_space = spaces.Box(0, 1, (self.w, self.h, self.squares), dtype=int)
        self._reward_range = (0., self.max_tile)

        # for render
        self.grid_size = 70
        # Initialise the random seed of the gym environment.
        self.seed()
        self.frames = []
        self.num_of_possible_chance_tile = cfg.num_of_possible_chance_tile
        self.possible_tiles = cfg.possible_tiles
        self.tile_probabilities = cfg.tile_probabilities
        if self.num_of_possible_chance_tile > 2:
            self.possible_tiles = np.array([2 ** (i + 1) for i in range(self.num_of_possible_chance_tile)])
            self.tile_probabilities = np.array(
                [1 / self.num_of_possible_chance_tile for _ in range(self.num_of_possible_chance_tile)])
            assert self.possible_tiles.shape[0] == self.tile_probabilities.shape[0]
            assert np.sum(self.tile_probabilities) == 1

    def reset(self, init_board=None, add_random_tile_flag=True):
        """Reset the game board-matrix and add 2 tiles."""
        self.episode_length = 0
        self.add_random_tile_flag = add_random_tile_flag
        if init_board is not None:
            self.board = copy.deepcopy(init_board)
        else:
            self.board = np.zeros((self.h, self.w), np.int32)
            # Add two tiles at the start of the game
            for _ in range(2):
                if self.num_of_possible_chance_tile > 2:
                    self.add_random_tile(self.possible_tiles, self.tile_probabilities)
                elif self.num_of_possible_chance_tile == 2:
                    self.add_random_2_4_tile()

        self.episode_return = 0
        self._final_eval_reward = 0.0
        self.should_done = False
        # Create a mask for legal actions
        action_mask = np.zeros(4, 'int8')
        action_mask[self.legal_actions] = 1

        # Encode the board, ensure correct datatype and shape
        observation = encode_board(self.board).astype(np.float32)
        assert observation.shape == (4, 4, 16)

        # Reshape or transpose the observation as per the requirement
        if not self.channel_last:
            # move channel dim to fist axis
            # (W, H, C) -> (C, W, H)
            # e.g. (4, 4, 16) -> (16, 4, 4)
            observation = np.transpose(observation, [2, 0, 1])
        if self.need_flatten:
            observation = observation.reshape(-1)

        # Based on the observation type, create the appropriate observation object
        if self.obs_type == 'dict_encoded_board':
            observation = {
                'observation': observation,
                'action_mask': action_mask,
                'to_play': -1,
                'chance': self.chance
            }
        elif self.obs_type == 'raw_board':
            observation = self.board
        elif self.obs_type == 'raw_encoded_board':
            observation = observation
        else:
            raise NotImplementedError

        # Render the game if the replay is to be saved
        if self.save_replay:
            self.render(mode='rgb_array_render')

        return observation

    def step(self, action):
        """
        Overview:
            Perform one step of the game. This involves making a move, adding a new tile, and updating the game state.
            New tile could be added randomly or from the tile probabilities.
            The rewards are calculated based on the game configuration ('merged_tiles_plus_log_max_tile_num' or 'raw').
            The observations are also returned based on the game configuration ('raw_board', 'raw_encoded_board' or 'dict_encoded_board').
        Arguments:
            - action (:obj:`int`): The action to be performed.
        Returns:
            - BaseEnvTimestep: Contains the new state observation, reward, and other game information.
        """

        # Increment the total episode length
        self.episode_length += 1

        # Check if the action is legal, otherwise choose a random legal action
        if action not in self.legal_actions:
            logging.warning(
                f"Illegal action: {action}. Legal actions: {self.legal_actions}. "
                "Choosing a random action from legal actions."
            )
            action = np.random.choice(self.legal_actions)

        # Calculate the reward differently based on the reward type
        if self.reward_type == 'merged_tiles_plus_log_max_tile_num':
            empty_num1 = len(self.get_empty_location())
        raw_reward = float(self.move(action))
        if self.reward_type == 'merged_tiles_plus_log_max_tile_num':
            empty_num2 = len(self.get_empty_location())
            num_of_merged_tiles = float(empty_num2 - empty_num1)
            reward_merged_tiles_plus_log_max_tile_num = num_of_merged_tiles
            max_tile_num = self.highest()
            if max_tile_num > self.max_tile_num:
                reward_merged_tiles_plus_log_max_tile_num += np.log2(max_tile_num) * 0.1
                self.max_tile_num = max_tile_num

        # Update total reward and add new tile
        self.episode_return += raw_reward
        assert raw_reward <= 2 ** (self.w * self.h)
        if self.add_random_tile_flag:
            if self.num_of_possible_chance_tile > 2:
                self.add_random_tile(self.possible_tiles, self.tile_probabilities)
            elif self.num_of_possible_chance_tile == 2:
                self.add_random_2_4_tile()

        # Check if the game has ended
        done = self.is_done()

        # Convert rewards to float
        if self.reward_type == 'merged_tiles_plus_log_max_tile_num':
            reward_merged_tiles_plus_log_max_tile_num = float(reward_merged_tiles_plus_log_max_tile_num)
        elif self.reward_type == 'raw':
            raw_reward = float(raw_reward)

        # End the game if the maximum steps have been reached
        if self.episode_length >= self.max_episode_steps:
            done = True

        # Prepare the game state observation
        observation = encode_board(self.board)
        observation = observation.astype(np.float32)
        assert observation.shape == (4, 4, 16)
        if not self.channel_last:
            observation = np.transpose(observation, [2, 0, 1])
        if self.need_flatten:
            observation = observation.reshape(-1)
        action_mask = np.zeros(4, 'int8')
        action_mask[self.legal_actions] = 1

        # Return the observation based on the observation type
        if self.obs_type == 'dict_encoded_board':
            observation = {'observation': observation, 'action_mask': action_mask, 'to_play': -1, 'chance': self.chance}
        elif self.obs_type == 'raw_board':
            observation = self.board
        elif self.obs_type == 'raw_encoded_board':
            observation = observation
        else:
            raise NotImplementedError

        # Normalize the reward if necessary
        if self.reward_normalize:
            reward_normalize = raw_reward / self.reward_norm_scale
            reward = reward_normalize
        else:
            reward = raw_reward

        self._final_eval_reward += raw_reward

        # Convert the reward to ndarray
        if self.reward_type == 'merged_tiles_plus_log_max_tile_num':
            reward = to_ndarray([reward_merged_tiles_plus_log_max_tile_num]).astype(np.float32)
        elif self.reward_type == 'raw':
            reward = to_ndarray([reward]).astype(np.float32)

        # Prepare information to return
        info = {"raw_reward": raw_reward, "current_max_tile_num": self.highest()}
        if self.save_replay:
            self.render(mode='rgb_array_render')

        # If the game has ended, save additional information and the replay if necessary
        if done:
            info['eval_episode_return'] = self._final_eval_reward
            if self.save_replay:
                self.save_render_output(replay_name_suffix=self.replay_name_suffix, replay_path=self.replay_path,
                                        format=self.replay_format)

        return BaseEnvTimestep(observation, reward, done, info)

    def move(self, direction, trial=False):
        """
        Overview:
            Perform one move in the game. The game board can be shifted in one of four directions: up (0), right (1), down (2), or left (3).
            This method manages the shifting process and combines similar adjacent elements. It also returns the reward generated from the move.
        Arguments:
            - direction (:obj:`int`): The direction of the move.
            - trial (:obj:`bool`): If true, this move is only simulated and does not change the actual game state.
        """
        # TODO(pu): different transition dynamics
        # Logging the direction of the move if not a trial
        if not trial:
            logging.debug(["Up", "Right", "Down", "Left"][int(direction)])

        move_reward = 0
        # Calculate merge direction of the shift (0 for up/left, 1 for down/right) based on the input direction
        merge_direction = 0 if direction in [0, 3] else 1

        # Construct a range for extracting row/column into a list
        range_x = list(range(self.w))
        range_y = list(range(self.h))

        # If direction is up or down, process the board column by column
        if direction in [0, 2]:
            for y in range(self.h):
                old_col = [self.board[x, y] for x in range_x]
                new_col, reward = self.shift(old_col, merge_direction)
                move_reward += reward
                if old_col != new_col and not trial:  # Update the board if it's not a trial move
                    for x in range_x:
                        self.board[x, y] = new_col[x]
        # If direction is left or right, process the board row by row
        else:
            for x in range(self.w):
                old_row = [self.board[x, y] for y in range_y]
                new_row, reward = self.shift(old_row, merge_direction)
                move_reward += reward
                if old_row != new_row and not trial:  # Update the board if it's not a trial move
                    for y in range_y:
                        self.board[x, y] = new_row[y]

        return move_reward

    def shift(self, row, merge_direction):
        """
        Overview:
            This method shifts the elements in a given row or column of the 2048 board in a specified direction.
            It performs three main operations: removal of zeroes, combination of similar elements, and filling up the
            remaining spaces with zeroes. The direction of shift can be either left (0) or right (1).
        Arguments:
            - row: A list of integers representing a row or a column in the 2048 board.
            - merge_direction: An integer that dictates the direction of merge. It can be either 0 or 1.
                - 0: The elements in the 'row' will be merged towards left/up.
                - 1: The elements in the 'row' will be merged towards right/down.
        Returns:
            - combined_row: A list of integers of the same length as 'row' after shifting and merging.
            - move_reward: The reward gained from combining similar elements in 'row'. It is the sum of all new
                combinations.
        Note:
            This method assumes that the input 'row' is a list of integers and 'merge_direction' is either 0 or 1.
        """

        # Remove the zero elements from the row and store it in a new list.
        non_zero_row = [i for i in row if i != 0]

        # Determine the start, stop, and step values based on the direction of shift.
        # If the direction is left (0), we start at the first element and move forwards.
        # If the direction is right (1), we start at the last element and move backwards.
        start, stop, step = (0, len(non_zero_row), 1) if merge_direction == 0 else (len(non_zero_row) - 1, -1, -1)

        # Call the combine function to merge the adjacent, same elements in the row.
        combined_row, move_reward = self.combine(non_zero_row, start, stop, step)

        if merge_direction == 1:
            # If direction is 'right'/'down', reverse the row
            combined_row = combined_row[::-1]

        # Fill up the remaining spaces in the row with 0, if any.
        if merge_direction == 0:
            combined_row += [0] * (len(row) - len(combined_row))
        elif merge_direction == 1:
            combined_row = [0] * (len(row) - len(combined_row)) + combined_row

        return combined_row, move_reward

    def combine(self, row, start, stop, step):
        """
        Overview:
            Combine similar adjacent elements in the row, starting from the specified start index,
            ending at the stop index, and moving in the direction indicated by the step. The function
            also calculates the reward as the sum of all combined elements.
        """

        # Initialize the reward for this move as 0.
        move_reward = 0

        # Initialize the list to store the row after combining same elements.
        combined_row = []

        # Initialize a flag to indicate whether the next element should be skipped.
        skip_next = False

        # Iterate over the elements in the row based on the start, stop, and step values.
        for i in range(start, stop, step):
            # If the next element should be skipped, reset the flag and continue to the next iteration.
            if skip_next:
                skip_next = False
                continue

            # If the current element and the next element are the same, combine them.
            if i + step != stop and row[i] == row[i + step]:
                combined_row.append(row[i] * 2)
                move_reward += row[i] * 2
                # Set the flag to skip the next element in the next iteration.
                skip_next = True
            else:
                # If the current element and the next element are not the same, just append the current element to the result.
                combined_row.append(row[i])

        return combined_row, move_reward

    @property
    def legal_actions(self):
        """
        Overview:
            Return the legal actions for the current state. A move is considered legal if it changes the state of the board.
        """

        if self.ignore_legal_actions:
            return [0, 1, 2, 3]

        legal_actions = []

        # For each direction, simulate a move. If the move changes the board, add the direction to the list of legal actions
        for direction in range(4):
            # Calculate merge direction of the shift (0 for up/left, 1 for down/right) based on the input direction
            merge_direction = 0 if direction in [0, 3] else 1

            range_x = list(range(self.w))
            range_y = list(range(self.h))

            if direction % 2 == 0:
                for y in range(self.h):
                    old_col = [self.board[x, y] for x in range_x]
                    new_col, _ = self.shift(old_col, merge_direction)
                    if old_col != new_col:
                        legal_actions.append(direction)
                        break  # As soon as we know the move is legal, we can stop checking
            else:
                for x in range(self.w):
                    old_row = [self.board[x, y] for y in range_y]
                    new_row, _ = self.shift(old_row, merge_direction)
                    if old_row != new_row:
                        legal_actions.append(direction)
                        break  # As soon as we know the move is legal, we can stop checking

        return legal_actions

    # Implementation of game logic for 2048
    def add_random_2_4_tile(self):
        """Add a tile with value 2 or 4 with different probabilities."""
        possible_tiles = np.array([2, 4])
        tile_probabilities = np.array([0.9, 0.1])
        tile_val = self.np_random.choice(possible_tiles, 1, p=tile_probabilities)[0]
        empty_location = self.get_empty_location()
        if empty_location.shape[0] == 0:
            self.should_done = True
            return
        empty_idx = self.np_random.choice(empty_location.shape[0])
        empty = empty_location[empty_idx]
        logging.debug("Adding %s at %s", tile_val, (empty[0], empty[1]))

        # set the chance outcome
        if self.chance_space_size == 16:
            self.chance = 4 * empty[0] + empty[1]
        elif self.chance_space_size == 32:
            if tile_val == 2:
                self.chance = 4 * empty[0] + empty[1]
            elif tile_val == 4:
                self.chance = 16 + 4 * empty[0] + empty[1]

        self.board[empty[0], empty[1]] = tile_val

    def add_random_tile(self, possible_tiles: np.array = np.array([2, 4]),
                        tile_probabilities: np.array = np.array([0.9, 0.1])):
        """Add a tile with a value from possible_tiles array according to given probabilities."""
        if len(possible_tiles) != len(tile_probabilities):
            raise ValueError("Length of possible_tiles and tile_probabilities must be the same")
        if np.sum(tile_probabilities) != 1:
            raise ValueError("Sum of tile_probabilities must be 1")

        tile_val = self.np_random.choice(possible_tiles, 1, p=tile_probabilities)[0]
        tile_idx = np.where(possible_tiles == tile_val)[0][0]  # get the index of the tile value
        empty_location = self.get_empty_location()
        if empty_location.shape[0] == 0:
            self.should_done = True
            return
        empty_idx = self.np_random.choice(empty_location.shape[0])
        empty = empty_location[empty_idx]
        logging.debug("Adding %s at %s", tile_val, (empty[0], empty[1]))

        # set the chance outcome
        self.chance_space_size = len(possible_tiles) * 16  # assuming a 4x4 board
        self.chance = tile_idx * 16 + 4 * empty[0] + empty[1]

        self.board[empty[0], empty[1]] = tile_val

    def get_empty_location(self):
        """Return a 2d numpy array with the location of empty squares."""
        return np.argwhere(self.board == 0)

    def highest(self):
        """Report the highest tile on the board."""
        return np.max(self.board)

    def is_done(self):
        """Has the game ended. Game ends if there is a tile equal to the limit
           or there are no legal moves. If there are empty spaces then there
           must be legal moves."""

        if self.max_tile is not None and self.highest() == self.max_tile:
            return True
        elif len(self.legal_actions) == 0:
            # the agent don't have legal_actions to move, so the episode is done
            return True
        elif self.should_done:
            return True
        else:
            return False

    def get_board(self):
        """Get the whole board-matrix, useful for testing."""
        return self.board

    def set_board(self, new_board):
        """Set the whole board-matrix, useful for testing."""
        self.board = new_board

    def seed(self, seed=None, seed1=None):
        """Set the random seed for the gym environment."""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def random_action(self) -> np.ndarray:
        random_action = self.action_space.sample()
        if isinstance(random_action, np.ndarray):
            pass
        elif isinstance(random_action, int):
            random_action = to_ndarray([random_action], dtype=np.int64)
        return random_action

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
                action = int(
                    input(
                        f"Enter the action (0(Up), 1(Right), 2(Down), or 3(Left)) to play: "
                    )
                )
                if action in self.legal_actions:
                    break
                else:
                    print("Wrong input, try again")
            except KeyboardInterrupt:
                print("exit")
                sys.exit(0)
        return action

    def render(self, mode='human'):
        if mode == 'rgb_array_render':
            grey = (128, 128, 128)
            grid_size = self.grid_size

            # Render with Pillow
            pil_board = Image.new("RGB", (grid_size * 4, grid_size * 4))
            draw = ImageDraw.Draw(pil_board)
            draw.rectangle([0, 0, 4 * grid_size, 4 * grid_size], grey)
            fnt_path = fm.findfont(fm.FontProperties(family='DejaVu Sans'))
            fnt = ImageFont.truetype(fnt_path, 30)

            for y in range(4):
                for x in range(4):
                    o = self.board[y, x]
                    if o:
                        self.draw_tile(draw, x, y, o, fnt)

            # Instead of returning the image, we display it using pyplot
            if self.render_real_time:
                plt.imshow(np.asarray(pil_board))
                plt.draw()
                # plt.pause(0.001)
            # Append the frame to frames for gif
            self.frames.append(np.asarray(pil_board))
        elif mode == 'human':
            s = 'Current Return: {}, '.format(self.episode_return)
            s += 'Current Highest Tile number: {}\n'.format(self.highest())
            npa = np.array(self.board)
            grid = npa.reshape((self.size, self.size))
            s += "{}\n".format(grid)
            sys.stdout.write(s)
            return sys.stdout

    def draw_tile(self, draw, x, y, o, fnt):
        grid_size = self.grid_size
        white = (255, 255, 255)
        tile_colour_map = {
            0: (204, 192, 179),
            2: (238, 228, 218),
            4: (237, 224, 200),
            8: (242, 177, 121),
            16: (245, 149, 99),
            32: (246, 124, 95),
            64: (246, 94, 59),
            128: (237, 207, 114),
            256: (237, 204, 97),
            512: (237, 200, 80),
            1024: (237, 197, 63),
            2048: (237, 194, 46),
            4096: (237, 194, 46),
            8192: (237, 194, 46),
            16384: (237, 194, 46),
        }
        if o:
            draw.rectangle([x * grid_size, y * grid_size, (x + 1) * grid_size, (y + 1) * grid_size],
                           tile_colour_map[o])
            bbox = draw.textbbox((x, y), str(o), font=fnt)
            text_x_size, text_y_size = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.text((x * grid_size + (grid_size - text_x_size) // 2,
                       y * grid_size + (grid_size - text_y_size) // 2), str(o), font=fnt, fill=white)

    def save_render_output(self, replay_name_suffix: str = '', replay_path=None, format='gif'):
        # At the end of the episode, save the frames to a gif or mp4 file
        if replay_path is None:
            filename = f'game_2048_{replay_name_suffix}.{format}'
        else:
            filename = f'{replay_path}.{format}'

        if format == 'gif':
            imageio.mimsave(filename, self.frames, 'GIF')
        elif format == 'mp4':
            imageio.mimsave(filename, self.frames, fps=30, codec='mpeg4')

        else:
            raise ValueError("Unsupported format: {}".format(format))

        logging.info("Saved output to {}".format(filename))
        self.frames = []

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        return self._action_space

    @property
    def reward_space(self) -> gym.spaces.Space:
        return self._reward_range

    @staticmethod
    def create_collector_env_cfg(cfg: dict) -> List[dict]:
        collector_env_num = cfg.pop('collector_env_num')
        cfg = copy.deepcopy(cfg)
        # when in collect phase, sometimes we need to normalize the reward
        # reward_normalize is determined by the config.
        cfg.is_collect = True
        return [cfg for _ in range(collector_env_num)]

    @staticmethod
    def create_evaluator_env_cfg(cfg: dict) -> List[dict]:
        evaluator_env_num = cfg.pop('evaluator_env_num')
        cfg = copy.deepcopy(cfg)
        # when in evaluate phase, we don't need to normalize the reward.
        cfg.reward_normalize = False
        cfg.is_collect = False
        return [cfg for _ in range(evaluator_env_num)]

    def __repr__(self) -> str:
        return "LightZero game 2048 Env."


def encode_board(flat_board, num_of_template_tiles=16):
    """
    Overview:
        This function converts a [4, 4] raw game board into a [4, 4, num_of_template_tiles] one-hot encoded board.
    Arguments:
        - flat_board (:obj:`np.ndarray`): The raw game board, expected to be a 2D numpy array.
        - num_of_template_tiles (:obj:`int`): The number of unique tiles to consider in the encoding,
                                               default value is 16.
    Returns:
        - one_hot_board (:obj:`np.ndarray`): The one-hot encoded game board.
    """
    # Generate a sequence of powers of 2, corresponding to the unique tile values.
    # In the game, tile values are powers of 2. So, each unique tile is represented by 2 raised to some power.
    # The first tile is considered as 0 (empty tile).
    tile_values = 2 ** np.arange(num_of_template_tiles, dtype=int)
    tile_values[0] = 0  # The first tile represents an empty slot, so set its value to 0.

    # Create a 3D array from the 2D input board by repeating it along a new axis.
    # This creates a 'layered' view of the board, where each layer corresponds to one unique tile value.
    layered_board = np.repeat(flat_board[:, :, np.newaxis], num_of_template_tiles, axis=-1)

    # Perform the one-hot encoding:
    # For each layer of the 'layered_board', mark the positions where the tile value in the 'flat_board'
    # matches the corresponding value in 'tile_values'. If a match is found, mark it as 1 (True), else 0 (False).
    one_hot_board = (layered_board == tile_values).astype(int)

    return one_hot_board

import copy
import logging
import os
import sys
from typing import List, Literal

import gymnasium as gym
import imageio
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from ding.envs import BaseEnvTimestep
from ding.torch_utils import to_ndarray
from ding.utils import ENV_REGISTRY
from easydict import EasyDict
from gymnasium import spaces
from gymnasium.utils import seeding

from .modals import NUMBER_OF_COLS, NUMBER_OF_ROWS, ENCODED_BOARD_SHAPE, MAX_MOVE_SCORE, ENCODED_INPUT_SHAPE
from .env_5move import GameEnvironment5Move, ACTION_SPACE_SIZE


@ENV_REGISTRY.register('botris-5move')
class Botris5MoveEnv(gym.Env):
    """
    Overview:
        The BotrisEnv is a gym environment implementation of Botris. The environment provides an interface to interact with
        the game and receive observations, rewards, and game status information.

    Interfaces:
      - reset(init_board=None, add_random_tile_flag=True):
          Resets the game state and starts a new episode. It returns the initial observation of the game.
      - step(action):
          Advances the game by one step based on the provided action. It returns the new observation, reward, game status,
          and additional information.
      - render(mode='human'):
          Renders the current state of the game for visualization purposes.
    MDP Definition:
      - Observation Space:
          NOT ACCURATE!!!!!!!!!!!!!1
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
      - Action Space:
          NOT ACCURATE!!!!!!!!!!!!!1
          The action space is a discrete space with 4 possible actions:
              - 0: Move Up
              - 1: Move Right
              - 2: Move Down
              - 3: Move Left
      - Reward:
          The reward depends on the 'reward_type' parameter in the environment configuration.
          - If 'reward_type' is set to 'raw':
              The reward is a floating-point number representing the immediate reward obtained from the last action.
      - Done:
          The game ends when one of the following conditions is met:
              - The maximum score (configured by 'max_score') is reached.
              - There are no legal moves left.
              - The number of steps in the episode exceeds the maximum episode steps (configured by 'max_episode_steps').
      - Additional Information:
          The 'info' dictionary returned by the 'step' method contains additional information about the current state.
          The following keys are included in the dictionary:
              - 'raw_reward': The raw reward obtained from the last action.
      - Rendering:
          The render method provides a way to visually represent the current state of the game. It offers four distinct rendering modes:
            When set to None, the game state is not rendered.
            In 'state_realtime_mode', the game state is illustrated in a text-based format directly in the console.
            The 'image_realtime_mode' displays the game as an RGB image in real-time.
            With 'image_savefile_mode', the game is rendered as an RGB image but not displayed in real-time. Instead, the image is saved to a designated file.
            Please note that the default rendering mode is set to None.
      """

    # The default_config for Botris env.
    config = dict(
        # (str) The name of the environment registered in the environment registry.
        env_id="botris",
        # (str) The render mode. Options are 'None', 'state_realtime_mode', 'image_realtime_mode' or 'image_savefile_mode'.
        # If None, then the game will not be rendered.
        render_mode=None,
        # (str) The format in which to save the replay. 'gif' is a popular choice.
        replay_format='gif',
        # (str) A suffix for the replay file name to distinguish it from other files.
        replay_name_suffix='eval',
        # (str or None) The directory in which to save the replay file. If None, the file is saved in the current directory.
        replay_path=None,
        # (bool) Whether to scale the actions. If True, actions are divided by the action space size.
        act_scale=True,
        # (str) The type of observation to use. Options are 'raw_encoded_board' and 'dict_encoded_board'.
        obs_type='dict_encoded_board',
        # (bool) Whether to normalize rewards. If True, rewards are divided by the maximum possible reward.
        reward_normalize=False,
        # (float) The factor to scale rewards by when reward normalization is used.
        reward_norm_scale=100,
        # (str) The type of reward to use. 'raw' means the raw game score..
        reward_type='raw',
        # (int) The maximum score in the game. A game is won when this score is reached.
        max_score=int(10_000),
        # (int) The number of steps to delay rewards by. If > 0, the agent only receives a reward every this many steps.
        delay_reward_step=0,
        # (float) The probability that a random agent is used instead of the learning agent.
        prob_random_agent=0.,
        # (int) The maximum number of steps in an episode.
        max_episode_steps=int(1e6),
        # (bool) Whether to collect data during the game.
        is_collect=True,
        # (bool) Whether to ignore legal actions. If True, the agent can take any action, even if it's not legal.
        ignore_legal_actions=False,
    )

    @classmethod
    def default_config(cls: type) -> EasyDict:
        cfg: EasyDict = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    def __init__(self, cfg: dict) -> None:
        self._cfg: dict = cfg
        self._init_flag: bool = False
        self._env_id: str = cfg.env_id
        self.replay_format: str = cfg.replay_format
        self.replay_name_suffix: str = cfg.replay_name_suffix
        self.replay_path: str = cfg.replay_path
        self.render_mode: Literal['state_realtime_mode', 'image_realtime_mode', 'image_savefile_mode'] | None = cfg.render_mode

        self.obs_type: Literal['raw_encoded_board', 'dict_encoded_board'] = cfg.obs_type
        self.reward_type: Literal['raw'] = cfg.reward_type
        self.reward_normalize: bool = cfg.reward_normalize
        self.reward_norm_scale: int = cfg.reward_norm_scale
        assert self.reward_type in ['raw']
        assert self.reward_type == 'raw'
        self.max_score: int = cfg.max_score
        # Define the maximum score that will end the game (e.g. 1_000). None means no limit.
        # This does not affect the state returned.
        assert self.max_score is None or isinstance(self.max_score, int)

        self.max_episode_steps: int = cfg.max_episode_steps
        self.is_collect: bool = cfg.is_collect
        self.ignore_legal_actions: bool = cfg.ignore_legal_actions
        self.w: int = NUMBER_OF_COLS
        self.h: int = NUMBER_OF_ROWS
        self.episode_return: int = 0
        # Members for gym implementation:
        self._action_space = spaces.Discrete(ACTION_SPACE_SIZE)
        self._observation_space = spaces.Box(0, 1, ENCODED_INPUT_SHAPE, dtype=int)
        self._reward_range = (0., MAX_MOVE_SCORE)

        # Initialise the random seed of the gym environment.
        self.seed()
        self.frames = []

    def reset(self):
        """Reset the game."""
        self.episode_length = 0
        self.gameenv: GameEnvironment5Move = GameEnvironment5Move(20, 0.1)

        self.episode_return = 0
        self._final_eval_reward = 0.0
        # Create a mask for legal actions
        action_mask = np.ones(ACTION_SPACE_SIZE, np.int8)

        # Encode the board, ensure correct datatype and shape
        observation = self.gameenv.get_input_encoding()
        observation = observation.astype(np.float32)

        # Based on the observation type, create the appropriate observation object
        if self.obs_type == 'dict_encoded_board':
            observation = {
                'observation': observation,
                'action_mask': action_mask,
                'to_play': -1,
            }
        elif self.obs_type == 'raw_encoded_board':
            observation = observation
        else:
            raise NotImplementedError

        # Render the beginning state of the game.
        if self.render_mode is not None:
            self.render(self.render_mode)

        return observation

    def step(self, action):
        """
        Overview:
            Perform one step of the game. This involves making a move, and updating the game state.
            The rewards are calculated based on the game configuration ('raw').
            The observations are also returned based on the game configuration ('raw_encoded_board' or 'dict_encoded_board').
        Arguments:
            - action (:obj:`int`): The action to be performed.
        Returns:
            - BaseEnvTimestep: Contains the new state observation, reward, and other game information.
        """

        # Increment the total episode length
        self.episode_length += 1

        # Check if the action is legal, otherwise choose a random legal action
        raw_reward = float(self.move(action))

        # Update total reward and add new tile
        self.episode_return += raw_reward


        # Convert rewards to float
        if self.reward_type == 'raw':
            raw_reward = float(raw_reward)

        # Prepare the game state observation
        observation = self.gameenv.get_input_encoding()
        observation = observation.astype(np.float32)

        # Return the observation based on the observation type
        action_mask = np.ones(ACTION_SPACE_SIZE, np.int8)
        if self.obs_type == 'dict_encoded_board':
            observation = {'observation': observation, 'action_mask': action_mask, 'to_play': -1}
        elif self.obs_type == 'raw_encoded_board':
            observation = observation
        else:
            raise NotImplementedError

        # Check if the game has ended
        done = self.is_done()

        # End the game if the maximum steps have been reached
        if self.episode_length >= self.max_episode_steps:
            done = True

        # Normalize the reward if necessary
        if self.reward_normalize:
            reward_normalize = raw_reward / self.reward_norm_scale
            reward = reward_normalize
        else:
            reward = raw_reward

        self._final_eval_reward += raw_reward

        # Convert the reward to ndarray
        if self.reward_type == 'raw':
            reward = to_ndarray([reward]).astype(np.float32)

        # Prepare information to return
        info = {"raw_reward": raw_reward}

        # Render the new step.
        if self.render_mode is not None:
            self.render(self.render_mode)

        # If the game has ended, save additional information and the replay if necessary
        if done:
            info['eval_episode_return'] = self._final_eval_reward
            if self.render_mode == 'image_savefile_mode':
                self.save_render_output(replay_name_suffix=self.replay_name_suffix, replay_path=self.replay_path,
                                        format=self.replay_format)

        return BaseEnvTimestep(observation, reward, done, info)

    def move(self, action):
        """
        Overview:
            Perform one move in the game. The game board can be shifted in one of four directions: up (0), right (1), down (2), or left (3).
            This method manages the shifting process and combines similar adjacent elements. It also returns the reward generated from the move.
        Arguments:
            - direction (:obj:`int`): The direction of the move.
            - trial (:obj:`bool`): If true, this move is only simulated and does not change the actual game state.
        """
        # TODO(pu): different transition dynamics
        pre_move_score = self.gameenv.get_score()

        self.gameenv.step(action)

        move_reward = self.gameenv.get_score() - pre_move_score
        return move_reward

    def is_done(self):
        """Has the game ended. Game ends if there is a tile equal to the limit
           or there are no legal moves. If there are empty spaces then there
           must be legal moves."""
        if self.gameenv.terminal:
            return True
        elif (self.max_score is not None) and (self.gameenv.get_score() >= self.max_score):
            return True
        else:
            return False

    def seed(self, seed=None, seed1=None):
        """Set the random seed for the gym environment."""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode: str = None):
        """
        Overview:
            Renders the Botris game environment.
        Arguments:
            - mode (:obj:`str`): The rendering mode. Options are None, 'state_realtime_mode', 'image_realtime_mode' or 'image_savefile_mode'.
                When set to None, the game state is not rendered.
                In 'state_realtime_mode', the game state is illustrated in a text-based format directly in the console.
                The 'image_realtime_mode' displays the game as an RGB image in real-time.
                With 'image_savefile_mode', the game is rendered as an RGB image but not displayed in real-time. Instead, the image is saved to a designated file.
                Please note that the default rendering mode is set to None.
        """
        if mode == 'state_realtime_mode':
            s = 'Current Return: {}, '.format(self.episode_return)
            print(s)
            self.gameenv.render()
        else:
            pil_board = self.gameenv.draw()

            # Instead of returning the image, we display it using pyplot
            if mode == 'image_realtime_mode':
                plt.imshow(np.asarray(pil_board))
                plt.draw()
                # plt.pause(0.001)
            elif mode == 'image_savefile_mode':
                # Append the frame to frames for gif
                self.frames.append(np.asarray(pil_board))

    def save_render_output(self, replay_name_suffix: str = '', replay_path=None, format='gif'):
        # At the end of the episode, save the frames to a gif or mp4 file
        if replay_path is None:
            filename = f'botris_{replay_name_suffix}.{format}'
        else:
            if not os.path.exists(replay_path):
                os.makedirs(replay_path)
            filename = replay_path + f'/botris_{replay_name_suffix}.{format}'

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
        return "LightZero game botris Env."
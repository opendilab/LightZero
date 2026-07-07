import copy
import logging
import os
import sys
from typing import List, Literal, Tuple

import gymnasium as gym
import imageio
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ding.envs import BaseEnvTimestep
from ding.torch_utils import to_ndarray
from ding.utils import ENV_REGISTRY
from easydict import EasyDict
from gymnasium import spaces
from gymnasium.utils import seeding

from .modals import NUMBER_OF_COLS, NUMBER_OF_ROWS, ENCODED_BOARD_SHAPE, ACTION_SPACE_SIZE, MAX_MOVE_SCORE, ENCODED_INPUT_SHAPE
from .env_versus import GameEnvironment

@ENV_REGISTRY.register('botris-versus')
class BotrisEnv(gym.Env):
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
        self._reward_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

        # Initialise the random seed of the gym environment.
        self.seed()
        self.frames = []

    def reset(self):
        """Reset the game."""
        self.episode_length = 0
        self.gameenv: GameEnvironment = GameEnvironment()
        obs = self.observe()
        return obs

    def observe(self) -> dict:
        self.action_mask = self.gameenv.legal_moves_mask().astype(np.int8)
        return {"observation": self.current_state(),
                "action_mask": self.action_mask,
                "to_play": self.current_player
                }

    def current_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Overview:
            Obtain the state from the view of current player.\
            self.board is nd-array, 0 indicates that no stones is placed here,\
            1 indicates that player 1's stone is placed here, 2 indicates player 2's stone is placed here.
        Returns:
            - current_state (:obj:`array`):
                the 0 dim means which positions is occupied by ``self.current_player``,\
                the 1 dim indicates which positions are occupied by ``self.next_player``,\
                the 2 dim indicates which player is the to_play player, 1 means player 1, 2 means player 2.
        """
        observation = self.gameenv.get_input_encoding()
        observation = observation.astype(np.float32)
        return observation

    def get_done_winner(self) -> Tuple[bool, int]:
        """
        Overview:
            Check if the game is done and find the winner.
        Returns:
            - outputs (:obj:`Tuple`): Tuple containing 'done' and 'winner',
                - if player 1 win,     'done' = True, 'winner' = 1
                - if player 2 win,     'done' = True, 'winner' = 2
                - if draw,             'done' = True, 'winner' = -1
                - if game is not over, 'done' = False,'winner' = -1
        """
        done = self.gameenv.is_terminal()
        winner = self.gameenv.get_winner()
        return done, winner

    def _player_step(self, action: int) -> BaseEnvTimestep:
        """
        Overview:
            A function that implements the transition of the environment's state. \
            After taking an action in the environment, the function transitions the environment to the next state \
            and returns the relevant information for the next time step.
        Arguments:
            - action (:obj:`int`): A value from 0 to 6 indicating the position to move on the connect4 board.
            - flag (:obj:`str`): A marker indicating the source of an action, for debugging convenience.
        Returns:
            - timestep (:obj:`BaseEnvTimestep`): A namedtuple that records the observation and obtained reward after taking the action, \
            whether the game is terminated, and some other information. 
        """
        if self.legal_actions[action]:
            self.gameenv.step_action(action)
        else:
            logging.warning(
                f"You input illegal action: {action}, the legal_actions are {self.legal_actions}. "
                f"Now we randomly choice a action from self.legal_actions."
            )
            action = np.random.choice(self.legal_actions)
            self.gameenv.step_action(action)

        done, winner = self.get_done_winner()
        if winner != -1:
            reward = np.array(1).astype(np.float32)
        else:
            reward = np.array(0).astype(np.float32)

        info = {}

        obs = self.observe()

        # Render the new step.
        if self.render_mode is not None:
            self.render(self.render_mode)
        if done:
            info['eval_episode_return'] = reward
            if self.render_mode == 'image_savefile_mode':
                self.save_render_output(replay_name_suffix=self.replay_name_suffix, replay_path=self.replay_path,
                                        format=self.replay_format)

        return BaseEnvTimestep(obs, reward, done, info)

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
        timestep = self._player_step(action)

        if timestep.done:
            # The ``eval_episode_return`` is calculated from player 1's perspective.
            timestep.info['eval_episode_return'] = -timestep.reward if timestep.obs[
                                                                            'to_play'] == 1 else timestep.reward

        return timestep

    
    @property
    def observation_space(self) -> gym.spaces.Space:
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        return self._action_space

    @property
    def reward_space(self) -> gym.spaces.Space:
        return self._reward_space

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
    def legal_actions(self) -> List[int]:
        return np.where(self.legal_actions == 1)[0]
    
    @property
    def current_player(self):
        return self.gameenv.current_player
    
    @property
    def next_player(self):
        return 1 - self.current_player


    def simulate_action(self, action):
        """
        Overview:
            execute action and get next_simulator_env. used in AlphaZero.
        Returns:
            Returns Gomoku instance.
        """
        if not self.action_mask[action]:
            raise ValueError("action {0} on board {1} is not legal".format(action, self.board))
        next_simulator_env = copy.deepcopy(self)
        next_simulator_env.step(action)
        return next_simulator_env

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
        return "LightZero Botris Env"
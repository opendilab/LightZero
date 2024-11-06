import copy
import os
from datetime import datetime
from typing import List

# import gymnasium as gym
import gym

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from ding.envs import BaseEnv, BaseEnvTimestep
from ding.torch_utils import to_ndarray
from ding.utils import ENV_REGISTRY
from easydict import EasyDict
from matplotlib import animation


@ENV_REGISTRY.register('memory_maze_lightzero')
class MemoryMazeEnvLightZero(BaseEnv):
    """
    Overview:
        The MemoryMazeEnvLightZero class is an environment wrapper for the Memory Maze environment, adapting it for
        use with the LightZero framework. The environment involves randomized mazes where RL agents must rely on
        long-term memory to solve tasks such as finding specific target objects.

    Attributes:
        config (dict): Configuration dict for the environment. Default configurations can be updated.
        _cfg (dict): Internal configuration for runtime settings.
        _init_flag (bool): Flag to check if the environment is initialized.
        _env_id (str): The name of the environment variant (e.g., "MemoryMaze-9x9-v0").
        _save_replay (bool): Flag to control whether replay is saved as a GIF.
        _render (bool): Flag to control whether real-time rendering is enabled.
        _gif_images (list): List to store image frames for creating GIF replays.
        _max_step (int): Maximum number of steps per episode.
    """

    config = dict(
        env_id='memory_maze:MemoryMaze-9x9-v0',  # The default variant of the Memory Maze environment
        save_replay=False,  # Option to save episode as a GIF replay
        render=False,  # Enables real-time rendering using matplotlib
        scale_observation=True,  # Whether to scale observations to [0, 1]
        rgb_img_observation=True,  # Return RGB image observations
        flatten_observation=False,  # Flatten the observation tensor
        # max_steps=1000,  # The default maximum number of steps per episode
        max_steps=1e9,  # The default maximum number of steps per episode
    )

    @classmethod
    def default_config(cls: type) -> EasyDict:
        """
        Returns the default configuration for the environment.
        """
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    def __init__(self, cfg: dict) -> None:
        """
        Initializes the MemoryMaze environment with the specified configuration.

        Args:
            cfg (dict): A configuration dictionary containing environment settings.
        """
        self._cfg = cfg
        self._init_flag = False
        self._save_replay = cfg.save_replay
        self._render = cfg.render
        self._gif_images = []
        self._rng = np.random.RandomState()  # Random number generator

        self._env = None  # Initialize _env as None to avoid AttributeError

        # Observation and action space settings
        self.rgb_img_observation = cfg.rgb_img_observation
        self.scale_observation = cfg.scale_observation
        self.flatten_observation = cfg.flatten_observation
        self._max_steps = cfg.max_steps

    def reset(self) -> np.ndarray:
        """
        Resets the environment to its initial state and returns the first observation.

        Returns:
            obs (np.ndarray): The initial observation after resetting the environment.
        """
        self._current_step = 0
        self._episode_reward = 0

        # Set this if you are getting "Unable to load EGL library" error:
        os.environ['MUJOCO_GL'] = 'glfw'
        # os.environ['MUJOCO_GL'] = 'osmesa'

        # Initialize the Memory Maze environment
        self._env = gym.make(self._cfg.env_id)
        # print('==============env reset==============')
        observation = self._env.reset()

        # Define action and observation spaces
        self._action_space = self._env.action_space
        self._observation_space = self._env.observation_space
        self._reward_space = gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(1,), dtype=np.float32)

        if self._save_replay or self._render:
            img = self._convert_observation_to_image(observation)
            if self._save_replay:
                self._gif_images.append(img)

            if self._render:
                self._render_frame(img)

        # Process observation (scaling and flattening)
        observation = self._process_observation(observation)
        action_mask = np.ones(self.action_space.n, 'int8')
        
        observation = {'observation': observation, 'action_mask': action_mask, 'to_play': -1}

        return observation

    def step(self, action: np.ndarray) -> BaseEnvTimestep:
        """
        Takes a step in the environment using the provided action.

        Args:
            action (np.ndarray): The action to execute in the environment.

        Returns:
            BaseEnvTimestep: A tuple containing the next observation, reward, done flag, and info.
        """
        observation, reward, done, info = self._env.step(action)

        self._current_step += 1
        self._episode_reward += reward

        if self._save_replay or self._render:
            img = self._convert_observation_to_image(observation)
            if self._save_replay:
                self._gif_images.append(img)

            if self._render:
                self._render_frame(img)

        # Process the observation (scaling and flattening)
        observation = self._process_observation(observation)
        action_mask = np.ones(self.action_space.n, 'int8')

        # Check if episode is done or maximum steps reached
        if done or self._current_step >= self._max_steps:
            done = True
            info['eval_episode_return'] = self._episode_reward
            print(f'one episode done! eval_episode_return is {self._episode_reward}')

            if self._save_replay:
                self._save_gif()

        observation = {'observation': observation, 'action_mask': action_mask, 'to_play': -1}

        return BaseEnvTimestep(observation, reward, done, info)

    def random_action(self) -> np.ndarray:
        """
        Generates a random action from the action space.

        Returns:
            np.ndarray: A random action.
        """
        return self._env.action_space.sample()

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        """
        Set the seed for the environment's random number generator. Can handle both static and dynamic seeding.

        Args:
            seed (int): The seed for the random number generator.
            dynamic_seed (bool): Whether to modify the seed dynamically.
        """
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)
        self._rng = np.random.RandomState(self._seed)

    def close(self) -> None:
        """
        Closes the environment.
        """
        if self._env is not None:  # Check if _env has been initialized
            self._env.close()

    def _process_observation(self, observation: np.ndarray) -> np.ndarray:
        """
        Processes the observation by scaling and/or flattening it.

        Args:
            observation (np.ndarray): The raw observation from the environment.

        Returns:
            np.ndarray: The processed observation.
        """
        if self.rgb_img_observation and self.scale_observation:
            observation = observation / 255.0  # Scale RGB values to [0, 1]
            observation = np.transpose(observation, (-1, 0, 1))  # (H,W,C) -> (C,H,W)

        if self.flatten_observation:
            observation = observation.flatten()

        return observation

    def _convert_observation_to_image(self, observation: np.ndarray) -> Image:
        """
        Converts the observation into an image format for rendering and saving.

        Args:
            observation (np.ndarray): The observation from the environment.

        Returns:
            Image: The image representation of the observation.
        """
        if self.rgb_img_observation:
            img = Image.fromarray(observation, 'RGB')
        else:
            img = Image.fromarray(observation)

        return img

    def _render_frame(self, img: Image) -> None:
        """
        Renders the current frame using matplotlib.

        Args:
            img (Image): The image to render.
        """
        plt.imshow(img)
        plt.axis('off')
        plt.pause(0.0001)
        plt.clf()

    def _save_gif(self) -> None:
        """
        Saves the collected frames as a GIF.
        """
        gif_dir = os.path.join(os.path.dirname(__file__), 'replay')
        os.makedirs(gif_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        gif_file = os.path.join(gif_dir, f'episode_len{self._current_step}_{timestamp}.gif')
        self._gif_images[0].save(gif_file, save_all=True, append_images=self._gif_images[1:], duration=100, loop=0)
        print(f'Saved replay to {gif_file}')

    @property
    def observation_space(self) -> gym.spaces.Space:
        """
        Returns the observation space of the environment.

        Returns:
            gym.spaces.Space: The observation space.
        """
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        """
        Returns the action space of the environment.

        Returns:
            gym.spaces.Space: The action space.
        """
        return self._action_space

    @property
    def reward_space(self) -> gym.spaces.Space:
        return self._reward_space

    @staticmethod
    def create_collector_env_cfg(cfg: dict) -> List[dict]:
        """
        Creates the configuration for collector environments.

        Args:
            cfg (dict): The base configuration.

        Returns:
            List[dict]: A list of configuration dictionaries for collector environments.
        """
        collector_env_num = cfg.pop('collector_env_num')
        cfg = copy.deepcopy(cfg)
        cfg.is_train = True
        return [cfg for _ in range(collector_env_num)]

    @staticmethod
    def create_evaluator_env_cfg(cfg: dict) -> List[dict]:
        """
        Creates the configuration for evaluator environments.

        Args:
            cfg (dict): The base configuration.

        Returns:
            List[dict]: A list of configuration dictionaries for evaluator environments.
        """
        evaluator_env_num = cfg.pop('evaluator_env_num')
        cfg = copy.deepcopy(cfg)
        cfg.is_train = False
        return [cfg for _ in range(evaluator_env_num)]

    def __repr__(self):
        return "Memory Maze Environment for LightZero"
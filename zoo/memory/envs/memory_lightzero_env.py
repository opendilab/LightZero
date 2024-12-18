import copy
import os
from datetime import datetime
from typing import List

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from ding.envs import BaseEnv, BaseEnvTimestep
from ding.torch_utils import to_ndarray
from ding.utils import ENV_REGISTRY
from easydict import EasyDict
from matplotlib import animation


@ENV_REGISTRY.register('memory_lightzero')
class MemoryEnvLightZero(BaseEnv):
    """
    Overview:
        The MemoryEnvLightZero environment for LightZero, based on the Visual-Match and Key-to-Door Task from DeepMind.
    Attributes:
        config (dict): Configuration dict. Default configurations can be updated using this.
        _cfg (dict): Internal configuration dict that stores runtime configurations.
        _init_flag (bool): Flag to check if the environment is initialized.
        _env_id (str): The name of the Visual Match environment.
        _save_replay (bool): Flag to check if replays are saved.
        _render (bool): Flag to check if real-time rendering is enabled.
        _gif_images (list): List to store frames for creating GIF replay.
        _max_step (int): Maximum number of steps for the environment.
    """
    config = dict(
        env_id='visual_match',  # The name of the environment, options: 'visual_match', 'key_to_door'
        num_apples=10,  # Number of apples in the distractor phase
        apple_reward=(0, 0),  # Range of rewards for collecting an apple
        fix_apple_reward_in_episode=False,  # Whether to fix apple reward (DEFAULT_APPLE_REWARD) within an episode
        final_reward=1.0,  # Reward for choosing the correct door in the final phase
        respawn_every=300,  # Respawn interval for apples
        crop=True,  # Whether to crop the observation
        max_frames={
            "explore": 15,  # NOTE: "explore" should >=1, otherwise the agent won't be able to see the target color or key.
            "distractor": 30,
            "reward": 15
        },  # Maximum frames per phase
        save_replay=False,  # Whether to save GIF replay
        render=False,  # Whether to enable real-time rendering
        scale_entity_observation=True,  # Whether to scale the observation to [0, 1]
        entity_obs_max_scale=100,  # Maximum value of the observation
        rgb_img_observation=True,  # Whether to return RGB image observation
        scale_rgb_img_observation=True,  # Whether to scale the RGB image observation to [0, 1]
        flatten_observation=True,  # Whether to flatten the observation
    )

    @classmethod
    def default_config(cls: type) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    def __init__(self, cfg: dict) -> None:
        """
        Overview:
            Initialize the environment.
        Arguments:
            - cfg (:obj:`dict`): Configuration dict.
        """
        self._cfg = cfg
        self._init_flag = False
        self._save_replay = cfg.save_replay
        self._render = cfg.render
        self._gif_images = []
        self.entity_obs_max_scale = cfg.entity_obs_max_scale
        self.scale_entity_observation = cfg.scale_entity_observation
        self.rgb_img_observation = cfg.rgb_img_observation
        self.scale_rgb_img_observation = cfg.scale_rgb_img_observation
        self.flatten_observation = cfg.flatten_observation

    def reset(self) -> np.ndarray:
        """
        Overview:
            Reset the environment and return the initial observation.
        Returns:
            - obs (:obj:`np.ndarray`): Initial observation from the environment.
        """
        if hasattr(self, '_seed') and hasattr(self, '_dynamic_seed') and self._dynamic_seed:
            np_seed = 100 * np.random.randint(1, 1000)
            self._seed = self._seed + np_seed
            self._rng = np.random.RandomState(self._seed)
        elif hasattr(self, '_seed'):
            self._rng = np.random.RandomState(self._seed)
        else:
            self._seed = 0  # TODO
            self._rng = np.random.RandomState(self._seed)
        print(f'memory_lightzero_env reset self._seed: {self._seed}')
        if self._cfg.env_id == 'visual_match':
            from zoo.memory.envs.pycolab_tvt.visual_match import Game, PASSIVE_EXPLORE_GRID
            self._game = Game(
                self._rng,
                num_apples=self._cfg.num_apples,
                apple_reward=self._cfg.apple_reward,
                fix_apple_reward_in_episode=self._cfg.fix_apple_reward_in_episode,
                final_reward=self._cfg.final_reward,
                respawn_every=self._cfg.respawn_every,
                crop=self._cfg.crop,
                max_frames=self._cfg.max_frames,
                EXPLORE_GRID=PASSIVE_EXPLORE_GRID,
            )
        elif self._cfg.env_id == 'key_to_door':
            from zoo.memory.envs.pycolab_tvt.key_to_door import Game, REWARD_GRID_SR
            self._game = Game(
                self._rng,
                num_apples=self._cfg.num_apples,
                apple_reward=self._cfg.apple_reward,
                fix_apple_reward_in_episode=self._cfg.fix_apple_reward_in_episode,
                final_reward=self._cfg.final_reward,
                respawn_every=self._cfg.respawn_every,
                crop=self._cfg.crop,
                max_frames=self._cfg.max_frames,
                REWARD_GRID=REWARD_GRID_SR,
            )

        self._episode = self._game.make_episode()
        if self.rgb_img_observation:
            self._observation_space = gym.spaces.Box(0, 255, shape=(3, 5, 5), dtype='uint8')
            if self.scale_rgb_img_observation:
                self._observation_space = gym.spaces.Box(0, 1, shape=(3, 5, 5), dtype='float32')
        else:
            self._observation_space = gym.spaces.Box(0, self.entity_obs_max_scale, shape=(1, 5, 5), dtype='int64')
            if self.scale_entity_observation:
                self._observation_space = gym.spaces.Box(0, 1, shape=(1, 5, 5), dtype='float32')

        self._action_space = gym.spaces.Discrete(self._game.num_actions)
        self._reward_space = gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(1,), dtype=np.float32)

        self._current_step = 0
        self.episode_reward_list = []
        self._eval_episode_return = 0
        observation, _, _ = self._episode.its_showtime()
        observation = observation[0].reshape(1, 5, 5)
        observation = to_ndarray(observation, dtype=np.float32)
        action_mask = np.ones(self.action_space.n, 'int8')
        self._gif_images = []
        self._gif_images_numpy = []

        if self._save_replay or self._render or self.rgb_img_observation:
            # Convert observation to RGB format
            obs_rgb = np.zeros((5, 5, 3), dtype=np.uint8)
            for char, color in self._game._colours.items():
                # NOTE： self._game._colours is a dictionary that maps the characters in the game to their corresponding (true) colors, ranging in [0,999].
                #  Because the np.uint8 type array will perform a modulo 256 operation (taking the remainder), that is to say,
                #  any value greater than 255 will be subtracted by an integer multiple of 256 until the value falls within the range of 0-255.
                #  For example, 1000 will become 232 (because 1000 % 256 = 232)
                obs_rgb[observation.reshape(5, 5) == ord(char)] = color

            if self._save_replay or self._render:
                img = Image.fromarray(obs_rgb)
            if self._save_replay:
                self._gif_images.append(img)
                self._gif_images_numpy.append(obs_rgb)
            if self._render:
                plt.imshow(img)
                plt.axis('off')
                plt.pause(0.0001)
                plt.clf()

        if not self.rgb_img_observation and self.scale_entity_observation:
            observation = observation / self.entity_obs_max_scale
        if self.rgb_img_observation:
            observation = np.transpose(obs_rgb, (-1, 0, 1))  # (H,W,C) -> (C,H,W)
            if self.scale_rgb_img_observation:
                observation = observation / 255.0
        if self.flatten_observation:
            observation = observation.flatten()

        observation = {'observation': observation, 'action_mask': action_mask, 'to_play': -1}

        return observation

    def step(self, action: np.ndarray) -> BaseEnvTimestep:
        """
        Overview:
            Perform a step in the environment using the provided action, and return the next state of the environment.
        Arguments:
            - action (:obj:`np.ndarray`): The action to be performed in the environment.
        Returns:
            - timestep (:obj:`BaseEnvTimestep`): An object containing the new observation, reward, done flag,
              and info dictionary.
        """
        if isinstance(action, np.ndarray) and action.shape == (1,):
            action = action.squeeze()  # 0-dim array

        observation, reward, _ = self._episode.play(action)
        self.episode_reward_list.append(reward)
        observation = observation[0].reshape(1, 5, 5)

        self._current_step += 1
        self._eval_episode_return += reward
        done = self._episode.game_over

        info = {}
        if done:
            info['eval_episode_return'] = self._eval_episode_return
            info['success'] = 1 if self._eval_episode_return == self._cfg.final_reward else 0
            info['eval_episode_return'] = info['success']
            print(f'episode seed:{self._seed} done! self.episode_reward_list is: {self.episode_reward_list}')
            print(f"Step: {self._current_step}, Action: {action}, Reward: {reward}, Observation: {observation}, Done: {done}, Info: {info}")  # TODO

        # print(f"Step: {self._current_step}, Action: {action}, Reward: {reward}, Observation: {observation}, Done: {done}, Info: {info}")  # TODO
        observation = to_ndarray(observation, dtype=np.float32)
        # reward = to_ndarray([reward])
        reward = to_ndarray(reward)
        action_mask = np.ones(self.action_space.n, 'int8')

        if self._save_replay or self._render or self.rgb_img_observation:
            # Convert observation to RGB format
            obs_rgb = np.zeros((5, 5, 3), dtype=np.uint8)
            for char, color in self._game._colours.items():
                # NOTE： self._game._colours is a dictionary that maps the characters in the game to their corresponding (true) colors, ranging in [0,999].
                #  Because the np.uint8 type array will perform a modulo 256 operation (taking the remainder), that is to say,
                #  any value greater than 255 will be subtracted by an integer multiple of 256 until the value falls within the range of 0-255.
                #  For example, 1000 will become 232 (because 1000 % 256 = 232)
                obs_rgb[observation.reshape(5, 5) == ord(char)] = color

            if self._save_replay or self._render:
                img = Image.fromarray(obs_rgb)
            if self._save_replay:
                self._gif_images.append(img)
                self._gif_images_numpy.append(obs_rgb)
            if self._render:
                plt.imshow(img)
                plt.axis('off')
                plt.pause(0.0001)
                plt.clf()

        if done and self._save_replay:
            gif_dir = os.path.join(os.path.dirname(__file__), 'replay')
            os.makedirs(gif_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            gif_file = os.path.join(gif_dir, f'episode_seed{self._seed}_len{self._current_step}_{timestamp}.gif')
            self._gif_images[0].save(gif_file, save_all=True, append_images=self._gif_images[1:], duration=100, loop=0)
            # self.display_frames_as_gif(self._gif_images_numpy, gif_file)  # the alternative way to save gif
            print(f'saved replay to {gif_file}')

        if not self.rgb_img_observation and self.scale_entity_observation:
            observation = observation / self.entity_obs_max_scale
        if self.rgb_img_observation:
            observation = np.transpose(obs_rgb, (-1, 0, 1))  # (H,W,C) -> (C,H,W)
            if self.scale_rgb_img_observation:
                observation = observation / 255.0
        if self.flatten_observation:
            observation = observation.flatten()

        observation = {'observation': observation, 'action_mask': action_mask, 'to_play': -1}

        return BaseEnvTimestep(observation, reward, done, info)

    def random_action(self) -> np.ndarray:
        """
        Generate a random action using the action space's sample method. Returns a numpy array containing the action.
        """
        random_action = self.action_space.sample()
        random_action = to_ndarray([random_action], dtype=np.int64)
        return random_action

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        """
        Set the seed for the environment's random number generator. Can handle both static and dynamic seeding.
        """
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)
        self._rng = np.random.RandomState(self._seed)

    def close(self) -> None:
        """
        Close the environment.
        """
        self._init_flag = False

    @staticmethod
    def display_frames_as_gif(frames: list, path: str) -> None:
        patch = plt.imshow(frames[0])
        plt.axis('off')

        def animate(i):
            patch.set_data(frames[i])

        anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=5)
        anim.save(path, writer='imagemagick', fps=20)

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        return self._action_space

    @property
    def reward_space(self) -> gym.spaces.Space:
        return self._reward_space

    @staticmethod
    def create_collector_env_cfg(cfg: dict) -> List[dict]:
        collector_env_num = cfg.pop('collector_env_num')
        cfg = copy.deepcopy(cfg)
        cfg.is_train = True
        return [cfg for _ in range(collector_env_num)]

    @staticmethod
    def create_evaluator_env_cfg(cfg: dict) -> List[dict]:
        evaluator_env_num = cfg.pop('evaluator_env_num')
        cfg = copy.deepcopy(cfg)
        cfg.is_train = False
        return [cfg for _ in range(evaluator_env_num)]

    def __repr__(self):
        return "Memory Env of LightZero"

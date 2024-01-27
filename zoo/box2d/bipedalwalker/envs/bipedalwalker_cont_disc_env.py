import copy
import os
from datetime import datetime
from itertools import product

import gymnasium as gym
import numpy as np
from ding.envs import BaseEnvTimestep
from ding.envs.common import affine_transform
from ding.torch_utils import to_ndarray
from ding.utils import ENV_REGISTRY
from easydict import EasyDict

from zoo.box2d.bipedalwalker.envs.bipedalwalker_env import BipedalWalkerEnv


@ENV_REGISTRY.register('bipedalwalker_cont_disc')
class BipedalWalkerDiscEnv(BipedalWalkerEnv):
    """
    Overview:
        The modified BipedalWalker environment with manually discretized action space. For each dimension, equally dividing the
        original continuous action into ``each_dim_disc_size`` bins and using their Cartesian product to obtain
        handcrafted discrete actions.
    """

    @classmethod
    def default_config(cls: type) -> EasyDict:
        """
        Overview:
            Get the default configuration of the BipedalWalker environment.
        Returns:
            - cfg (:obj:`EasyDict`): Default configuration dictionary.
        """
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    config = dict(
        # (str) The gym environment name.
        env_name="BipedalWalker-v3",
        # (int) The number of bins for each dimension of the action space.
        each_dim_disc_size=4,
        # (bool) If True, save the replay as a gif file.
        save_replay_gif=False,
        # (str or None) The path to save the replay gif. If None, the replay gif will not be saved.
        replay_path_gif=None,
        # (str or None) The path to save the replay. If None, the replay will not be saved.
        replay_path=None,
        # (bool) If True, the action will be scaled.
        act_scale=True,
        # (bool) If True, the reward will be clipped to [-10, +inf].
        rew_clip=True,
        # (int) The maximum number of steps for each episode during collection.
        collect_max_episode_steps=int(1.08e5),
        # (int) The maximum number of steps for each episode during evaluation.
        eval_max_episode_steps=int(1.08e5),
    )

    def __init__(self, cfg: dict) -> None:
        """
        Overview:
            Initialize the BipedalWalker environment with the given config dictionary.
        Arguments:
            - cfg (:obj:`dict`): Configuration dictionary.
        """
        self._cfg = cfg
        self._init_flag = False
        self._env_name = cfg.env_name
        self._act_scale = cfg.act_scale
        self._rew_clip = cfg.rew_clip
        self._replay_path = cfg.replay_path
        self._replay_path_gif = cfg.replay_path_gif
        self._save_replay_gif = cfg.save_replay_gif
        self._save_replay_count = 0

    def reset(self) -> np.ndarray:
        """
        Overview:
            Reset the environment. During the reset phase, the original environment will be created,
            and at the same time, the action space will be discretized into "each_dim_disc_size" bins.
        Returns:
            - info_dict (:obj:`Dict[str, Any]`): Including observation, action_mask, and to_play label.
        """
        if not self._init_flag:
            self._env = gym.make('BipedalWalker-v3', hardcore=True, render_mode="rgb_array")
            self._observation_space = self._env.observation_space
            self._action_space = self._env.action_space
            self._reward_space = gym.spaces.Box(
                low=self._env.reward_range[0], high=self._env.reward_range[1], shape=(1, ), dtype=np.float32
            )
            self._init_flag = True
        if self._replay_path is not None:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            video_name = f'{self._env.spec.id}-video-{timestamp}'
            self._env = gym.wrappers.RecordVideo(
                self._env,
                video_folder=self._replay_path,
                episode_trigger=lambda episode_id: True,
                name_prefix=video_name
            )
        if hasattr(self, '_seed') and hasattr(self, '_dynamic_seed') and self._dynamic_seed:
            np_seed = 100 * np.random.randint(1, 1000)
            self._seed = self._seed + np_seed
            obs, _ = self._env.reset(seed=self._seed)  # using the reset method of Gymnasium env
        elif hasattr(self, '_seed'):
            obs, _ = self._env.reset(seed=self._seed)
        else:
            obs, _ = self._env.reset()
        obs = to_ndarray(obs).astype(np.float32)
        self._eval_episode_return = 0
        if self._save_replay_gif:
            self._frames = []
        # disc_to_cont: transform discrete action index to original continuous action
        self._raw_action_space = self._env.action_space
        self.m = self._raw_action_space.shape[0]
        self.n = self._cfg.each_dim_disc_size
        self.K = self.n ** self.m
        self.disc_to_cont = list(product(*[list(range(self.n)) for _ in range(self.m)]))
        # the modified discrete action space
        self._action_space = gym.spaces.Discrete(self.K)

        action_mask = np.ones(self.K, 'int8')
        obs = {'observation': obs, 'action_mask': action_mask, 'to_play': -1}
        return obs

    def step(self, action: np.ndarray) -> BaseEnvTimestep:
        """
        Overview:
            Take an action in the environment. During the step phase, the environment first converts the discrete action into a continuous action,
            and then passes it into the original environment.
        Arguments:
            - action (:obj:`np.ndarray`): Discrete action to be taken in the environment.
        Returns:
            - BaseEnvTimestep (:obj:`BaseEnvTimestep`): A tuple containing observation, reward, done, and info.
        """
        # disc_to_cont: transform discrete action index to original continuous action
        action = [-1 + 2 / self.n * k for k in self.disc_to_cont[int(action)]]
        action = to_ndarray(action)
        if action.shape == (1, ):
            action = action.squeeze()
        if self._act_scale:
            action = affine_transform(action, min_val=self._raw_action_space.low, max_val=self._raw_action_space.high)
        if self._save_replay_gif:
            self._frames.append(self._env.render())
        obs, rew, terminated, truncated, info = self._env.step(action)
        done = terminated or truncated

        action_mask = np.ones(self.K, 'int8')
        obs = {'observation': obs, 'action_mask': action_mask, 'to_play': -1}
        self._eval_episode_return += rew
        if self._rew_clip:
            rew = max(-10, rew)
        rew = np.float32(rew)
        if done:
            info['eval_episode_return'] = self._eval_episode_return
            if self._save_replay_gif:
                if not os.path.exists(self._replay_path_gif):
                    os.makedirs(self._replay_path_gif)
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                path = os.path.join(
                    self._replay_path_gif,
                    '{}_episode_{}_seed{}_{}.gif'.format(self._env_name, self._save_replay_count, self._seed, timestamp)
                )
                self.display_frames_as_gif(self._frames, path)
                print(f'save episode {self._save_replay_count} in {self._replay_path_gif}!')
                self._save_replay_count += 1
        obs = to_ndarray(obs)
        rew = to_ndarray([rew])
        return BaseEnvTimestep(obs, rew, done, info)

    def __repr__(self) -> str:
        """
        Overview:
            Represent the environment instance as a string.
        Returns:
            - repr_str (:obj:`str`): Representation string of the environment instance.
        """
        return "LightZero BipedalWalker Env (with manually discretized action space)"


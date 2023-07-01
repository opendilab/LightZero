import os
from itertools import product
from typing import Union

import gym
import numpy as np
from ding.envs import BaseEnvTimestep
from ding.envs.common import save_frames_as_gif
from ding.torch_utils import to_ndarray
from ding.utils import ENV_REGISTRY
from dizoo.mujoco.envs.mujoco_disc_env import MujocoDiscEnv


@ENV_REGISTRY.register('mujoco_disc_lightzero')
class MujocoDiscEnvLZ(MujocoDiscEnv):
    """
    Overview:
        The modified Mujoco environment with manually discretized action space for LightZero's algorithms.
        For each dimension, equally dividing the original continuous action into ``each_dim_disc_size`` bins and
        using their Cartesian product to obtain handcrafted discrete actions.
    """

    config = dict(
        action_clip=False,
        delay_reward_step=0,
        replay_path=None,
        save_replay_gif=False,
        replay_path_gif=None,
        action_bins_per_branch=None,
        norm_obs=dict(use_norm=False, ),
        norm_reward=dict(use_norm=False, ),
    )

    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
        self._cfg = cfg
        # We use env_name to indicate the env_id in LightZero.
        self._cfg.env_id = self._cfg.env_name
        self._action_clip = cfg.action_clip
        self._delay_reward_step = cfg.delay_reward_step
        self._init_flag = False
        self._replay_path = None
        self._replay_path_gif = cfg.replay_path_gif
        self._save_replay_gif = cfg.save_replay_gif

    def reset(self) -> np.ndarray:
        if not self._init_flag:
            self._env = self._make_env()
            self._env.observation_space.dtype = np.float32
            self._observation_space = self._env.observation_space
            self._raw_action_space = self._env.action_space
            self._reward_space = gym.spaces.Box(
                low=self._env.reward_range[0], high=self._env.reward_range[1], shape=(1,), dtype=np.float32
            )
            self._init_flag = True
        if hasattr(self, '_seed') and hasattr(self, '_dynamic_seed') and self._dynamic_seed:
            np_seed = 100 * np.random.randint(1, 1000)
            self._env.seed(self._seed + np_seed)
        elif hasattr(self, '_seed'):
            self._env.seed(self._seed)
        if self._replay_path is not None:
            self._env = gym.wrappers.RecordVideo(
                self._env,
                video_folder=self._replay_path,
                episode_trigger=lambda episode_id: True,
                name_prefix='rl-video-{}'.format(id(self))
            )
        if self._save_replay_gif:
            self._frames = []
        obs = self._env.reset()
        obs = to_ndarray(obs).astype('float32')

        # disc_to_cont: transform discrete action index to original continuous action
        self.m = self._raw_action_space.shape[0]
        self.n = self._cfg.each_dim_disc_size
        self.K = self.n ** self.m
        self.disc_to_cont = list(product(*[list(range(self.n)) for _ in range(self.m)]))
        self._eval_episode_return = 0.
        # the modified discrete action space
        self._action_space = gym.spaces.Discrete(self.K)

        action_mask = np.ones(self.K, 'int8')
        obs = {'observation': obs, 'action_mask': action_mask, 'to_play': -1}

        return obs

    def step(self, action: Union[np.ndarray, list]) -> BaseEnvTimestep:
        # disc_to_cont: transform discrete action index to original continuous action
        action = [-1 + 2 / self.n * k for k in self.disc_to_cont[int(action)]]
        action = to_ndarray(action)

        if self._save_replay_gif:
            self._frames.append(self._env.render(mode='rgb_array'))
        if self._action_clip:
            action = np.clip(action, -1, 1)
        obs, rew, done, info = self._env.step(action)

        self._eval_episode_return += rew

        if done:
            if self._save_replay_gif:
                path = os.path.join(
                    self._replay_path_gif, '{}_episode_{}.gif'.format(self._cfg.env_name, self._save_replay_count)
                )
                save_frames_as_gif(self._frames, path)
                self._save_replay_count += 1
            info['eval_episode_return'] = self._eval_episode_return

        obs = to_ndarray(obs).astype(np.float32)
        rew = to_ndarray([rew]).astype(np.float32)

        action_mask = np.ones(self._action_space.n, 'int8')
        obs = {'observation': obs, 'action_mask': action_mask, 'to_play': -1}

        return BaseEnvTimestep(obs, rew, done, info)

    def __repr__(self) -> str:
        return "LightZero modified Mujoco Env({}) with manually discretized action space".format(self._cfg.env_name)

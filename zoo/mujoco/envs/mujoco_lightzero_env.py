import os
from typing import Union

import gymnasium as gym
import numpy as np
from ding.envs import BaseEnvTimestep
from ding.envs.common import save_frames_as_gif
from ding.torch_utils import to_ndarray
from ding.utils import ENV_REGISTRY
from dizoo.mujoco.envs.mujoco_env import MujocoEnv


@ENV_REGISTRY.register('mujoco_lightzero')
class MujocoEnvLZ(MujocoEnv):
    """
    Overview:
        The modified MuJoCo environment with continuous action space for LightZero's algorithms.
    """

    config = dict(
        stop_value=int(1e6),
        action_clip=False,
        delay_reward_step=0,
        # replay_path (str or None): The path to save the replay video. If None, the replay will not be saved.
        # Only effective when env_manager.type is 'base'.
        replay_path=None,
        # (bool) If True, save the replay as a gif file.
        save_replay_gif=False,
        # (str or None) The path to save the replay gif. If None, the replay gif will not be saved.
        replay_path_gif=None,
        action_bins_per_branch=None,
        norm_obs=dict(use_norm=False, ),
        norm_reward=dict(use_norm=False, ),
    )

    def __init__(self, cfg: dict) -> None:
        """
        Overview:
            Initialize the MuJoCo environment.
        Arguments:
            - cfg (:obj:`dict`): Configuration dict. The dict should include keys like 'env_id', 'replay_path', etc.
        """
        super().__init__(cfg)
        self._cfg = cfg
        # We use env_id to indicate the env_id in LightZero.
        self._cfg.env_id = self._cfg.env_id
        self._action_clip = cfg.action_clip
        self._delay_reward_step = cfg.delay_reward_step
        self._init_flag = False
        self._replay_path = None
        self._replay_path_gif = cfg.replay_path_gif
        self._save_replay_gif = cfg.save_replay_gif
        self._action_bins_per_branch = cfg.action_bins_per_branch

    def reset(self) -> np.ndarray:
        """
        Overview:
            Reset the environment and return the initial observation.
        Returns:
            - obs (:obj:`np.ndarray`): The initial observation after resetting.
        """
        if not self._init_flag:
            self._env = self._make_env()
            if self._replay_path is not None:
                self._env = gym.wrappers.RecordVideo(
                    self._env,
                    video_folder=self._replay_path,
                    episode_trigger=lambda episode_id: True,
                    name_prefix='rl-video-{}'.format(id(self))
                )

            self._env.observation_space.dtype = np.float32
            self._observation_space = self._env.observation_space
            self._action_space = self._env.action_space
            self._reward_space = gym.spaces.Box(
                low=self._env.reward_range[0], high=self._env.reward_range[1], shape=(1,), dtype=np.float32
            )
            self._init_flag = True
        if hasattr(self, '_seed') and hasattr(self, '_dynamic_seed') and self._dynamic_seed:
            np_seed = 100 * np.random.randint(1, 1000)
            self._env.seed(self._seed + np_seed)
        elif hasattr(self, '_seed'):
            self._env.seed(self._seed)
        obs = self._env.reset()
        obs = to_ndarray(obs).astype('float32')
        self._eval_episode_return = 0.

        action_mask = None
        obs = {'observation': obs, 'action_mask': action_mask, 'to_play': -1}

        return obs

    def step(self, action: Union[np.ndarray, list]) -> BaseEnvTimestep:
        """
        Overview:
            Perform a step in the environment using the provided action, and return the next state of the environment.
            The next state is encapsulated in a BaseEnvTimestep object, which includes the new observation, reward,
            done flag, and info dictionary.
        Arguments:
            - action (:obj:`Union[np.ndarray, list]`): The action to be performed in the environment. 
        Returns:
            - timestep (:obj:`BaseEnvTimestep`): An object containing the new observation, reward, done flag,
              and info dictionary.
        .. note::
            - The cumulative reward (`_eval_episode_return`) is updated with the reward obtained in this step.
            - If the episode ends (done is True), the total reward for the episode is stored in the info dictionary
              under the key 'eval_episode_return'.
            - An action mask is created with ones, which represents the availability of each action in the action space.
            - Observations are returned in a dictionary format containing 'observation', 'action_mask', and 'to_play'.
        """
        if self._action_bins_per_branch:
            action = self.map_action(action)
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
                    self._replay_path_gif, '{}_episode_{}.gif'.format(self._cfg.env_id, self._save_replay_count)
                )
                save_frames_as_gif(self._frames, path)
                self._save_replay_count += 1
            info['eval_episode_return'] = self._eval_episode_return

        obs = to_ndarray(obs).astype(np.float32)
        rew = to_ndarray([rew]).astype(np.float32)

        action_mask = None
        obs = {'observation': obs, 'action_mask': action_mask, 'to_play': -1}

        return BaseEnvTimestep(obs, rew, done, info)

    def __repr__(self) -> str:
        """
        String representation of the environment.
        """
        return "LightZero Mujoco Env({})".format(self._cfg.env_id)


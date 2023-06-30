import copy
import os
from typing import Union, List, Optional

import gym
import numpy as np
from ding.envs import BaseEnv, BaseEnvTimestep
from ding.envs.common import save_frames_as_gif
from ding.torch_utils import to_ndarray
from ding.utils import ENV_REGISTRY
from dizoo.mujoco.envs.mujoco_wrappers import wrap_mujoco
from easydict import EasyDict


@ENV_REGISTRY.register('mujoco_lightzero')
class MujocoEnv(BaseEnv):
    """
    Overview:
        The modified MuJoCo environment with continuous action space for LightZero's algorithms.
    """

    @classmethod
    def default_config(cls: type) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    config = dict(
        stop_value=int(1e6),
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
        self._cfg = cfg
        self._action_clip = cfg.action_clip
        self._delay_reward_step = cfg.delay_reward_step
        self._init_flag = False
        self._replay_path = None
        self._replay_path_gif = cfg.replay_path_gif
        self._save_replay_gif = cfg.save_replay_gif
        self._action_bins_per_branch = cfg.action_bins_per_branch

    def map_action(self, action: Union[np.ndarray, list]) -> Union[np.ndarray, list]:
        """
        Overview:
            Map the discretized action index to the action in the original action space.
        Arguments:
            - action (:obj:`np.ndarray or list`): The discretized action index. \
                The value ranges is {0, 1, ..., self._action_bins_per_branch - 1}.
        Returns:
            - outputs (:obj:`list`): The action in the original action space. \
                The value ranges is [-1, 1].
        Examples:
            >>> inputs = [2, 0, 4]
            >>> self._action_bins_per_branch = 5
            >>> outputs = map_action(inputs)
            >>> assert isinstance(outputs, list) and outputs == [0.0, -1.0, 1.0]
        """
        return [2 * x / (self._action_bins_per_branch - 1) - 1 for x in action]

    def reset(self) -> np.ndarray:
        if not self._init_flag:
            self._env = self._make_env()
            if self._replay_path is not None:
                self._env = gym.wrappers.RecordVideo(
                    self._env,
                    video_folder=self._replay_path,
                    episode_trigger=lambda episode_id: True,
                    name_prefix='rl-video-{}'.format(id(self))
                )

            self._env.observation_space.dtype = np.float32  # To unify the format of envs in DI-engine
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

    def close(self) -> None:
        if self._init_flag:
            self._env.close()
        self._init_flag = False

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def step(self, action: Union[np.ndarray, list]) -> BaseEnvTimestep:
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
                    self._replay_path_gif, '{}_episode_{}.gif'.format(self._cfg.env_name, self._save_replay_count)
                )
                save_frames_as_gif(self._frames, path)
                self._save_replay_count += 1
            info['eval_episode_return'] = self._eval_episode_return

        obs = to_ndarray(obs).astype(np.float32)
        rew = to_ndarray([rew]).astype(np.float32)

        action_mask = None
        obs = {'observation': obs, 'action_mask': action_mask, 'to_play': -1}

        return BaseEnvTimestep(obs, rew, done, info)

    def _make_env(self):
        return wrap_mujoco(
            self._cfg.env_name,
            norm_obs=self._cfg.get('norm_obs', None),
            norm_reward=self._cfg.get('norm_reward', None),
            delay_reward_step=self._delay_reward_step
        )

    def enable_save_replay(self, replay_path: Optional[str] = None) -> None:
        if replay_path is None:
            replay_path = './video'
        self._replay_path = replay_path

    def random_action(self) -> np.ndarray:
        return self.action_space.sample()

    def __repr__(self) -> str:
        return "LightZero Mujoco Env({})".format(self._cfg.env_name)

    @staticmethod
    def create_collector_env_cfg(cfg: dict) -> List[dict]:
        collector_cfg = copy.deepcopy(cfg)
        collector_env_num = collector_cfg.pop('collector_env_num', 1)
        return [collector_cfg for _ in range(collector_env_num)]

    @staticmethod
    def create_evaluator_env_cfg(cfg: dict) -> List[dict]:
        evaluator_cfg = copy.deepcopy(cfg)
        evaluator_env_num = evaluator_cfg.pop('evaluator_env_num', 1)
        evaluator_cfg.norm_reward.use_norm = False
        return [evaluator_cfg for _ in range(evaluator_env_num)]

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        return self._action_space

    @property
    def reward_space(self) -> gym.spaces.Space:
        return self._reward_space

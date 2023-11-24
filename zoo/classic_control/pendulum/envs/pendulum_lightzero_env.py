import copy
from typing import Optional

import gymnasium as gym
import numpy as np
from ding.envs import BaseEnv, BaseEnvTimestep
from ding.envs.common.common_function import affine_transform
from ding.torch_utils import to_ndarray
from ding.utils import ENV_REGISTRY
from easydict import EasyDict


@ENV_REGISTRY.register('pendulum_lightzero')
class PendulumEnv(BaseEnv):

    @classmethod
    def default_config(cls: type) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    config = dict(
        continuous=True,
        save_replay_gif=False,
        replay_path_gif=None,
        replay_path=None,
        act_scale=True,
        delay_reward_step=0,
        prob_random_agent=0.,
        collect_max_episode_steps=int(1.08e5),
        eval_max_episode_steps=int(1.08e5),
    )

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        self._act_scale = cfg.act_scale
        try:
            self._env = gym.make('Pendulum-v1')
        except:
            self._env = gym.make('Pendulum-v0')
        self._init_flag = False
        self._replay_path = None
        self._continuous = cfg.get("continuous", True)
        self._observation_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0, -8.0]), high=np.array([1.0, 1.0, 8.0]), shape=(3, ), dtype=np.float32
        )
        if self._continuous:
            self._action_space = gym.spaces.Box(low=-2.0, high=2.0, shape=(1, ), dtype=np.float32)
        else:
            self.discrete_action_num = 11
            self._action_space = gym.spaces.Discrete(self.discrete_action_num)
        self._action_space.seed(0)  # default seed
        self._reward_space = gym.spaces.Box(
            low=-1 * (3.14 * 3.14 + 0.1 * 8 * 8 + 0.001 * 2 * 2), high=0.0, shape=(1, ), dtype=np.float32
        )

    def reset(self) -> np.ndarray:
        if not self._init_flag:
            try:
                self._env = gym.make('Pendulum-v1')
            except:
                self._env = gym.make('Pendulum-v0')
            if self._replay_path is not None:
                self._env = gym.wrappers.RecordVideo(
                    self._env,
                    video_folder=self._replay_path,
                    episode_trigger=lambda episode_id: True,
                    name_prefix='rl-video-{}'.format(id(self))
                )
            self._init_flag = True
        if hasattr(self, '_seed') and hasattr(self, '_dynamic_seed') and self._dynamic_seed:
            np_seed = 100 * np.random.randint(1, 1000)
            self._seed = self._seed + np_seed
            self._action_space.seed(self._seed)
            obs, _ = self._env.reset(seed=self._seed)
        elif hasattr(self, '_seed'): 
            self._action_space.seed(self._seed)
            obs, _ = self._env.reset(seed=self._seed)
        else:
            self._action_space.seed(self._seed)
            obs, _ = self._env.reset()
        obs = to_ndarray(obs).astype(np.float32)
        self._eval_episode_return = 0.

        if not self._continuous:
            action_mask = np.ones(self.discrete_action_num, 'int8')
        else:
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

    def step(self, action: np.ndarray) -> BaseEnvTimestep:
        if isinstance(action, int):
            action = np.array(action)
        # if require discrete env, convert actions to [-1 ~ 1] float actions
        if not self._continuous:
            action = (action / (self.discrete_action_num - 1)) * 2 - 1
        # scale into [-2, 2]
        if self._act_scale:
            action = affine_transform(action, min_val=self._env.action_space.low, max_val=self._env.action_space.high)
        obs, rew, done, _, info = self._env.step(action)
        self._eval_episode_return += rew
        obs = to_ndarray(obs).astype(np.float32)
        # wrapped to be transferred to a array with shape (1,)
        rew = to_ndarray([rew]).astype(np.float32)

        if done:
            info['eval_episode_return'] = self._eval_episode_return

        if not self._continuous:
            action_mask = np.ones(self.discrete_action_num, 'int8')
        else:
            action_mask = None
        obs = {'observation': obs, 'action_mask': action_mask, 'to_play': -1}

        return BaseEnvTimestep(obs, rew, done, info)

    def enable_save_replay(self, replay_path: Optional[str] = None) -> None:
        if replay_path is None:
            replay_path = './video'
        self._replay_path = replay_path

    def random_action(self) -> np.ndarray:
        if self._continuous:
            random_action = self.action_space.sample().astype(np.float32)
        else:
            random_action = self.action_space.sample()
            random_action = to_ndarray([random_action], dtype=np.int64)
        return random_action

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        return self._action_space

    @property
    def reward_space(self) -> gym.spaces.Space:
        return self._reward_space

    def __repr__(self) -> str:
        return "LightZero Pendulum Env({})".format(self._cfg.env_id)

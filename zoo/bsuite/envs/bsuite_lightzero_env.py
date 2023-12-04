import copy
from typing import List

import bsuite
import gymnasium as gym
from zoo.atari.envs.atari_wrappers import GymToGymnasiumWrapper
import numpy as np
from bsuite import sweep
from bsuite.utils import gym_wrapper
from ding.envs import BaseEnv, BaseEnvTimestep
from ding.torch_utils import to_ndarray
from ding.utils import ENV_REGISTRY


@ENV_REGISTRY.register('bsuite_lightzero')
class BSuiteEnv(BaseEnv):

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        self._init_flag = False
        self.env_name = cfg.env_name

    def reset(self) -> np.ndarray:
        if not self._init_flag:
            raw_env = bsuite.load_from_id(bsuite_id=self.env_name)
            raw_env = GymToGymnasiumWrapper(raw_env)
            self._env = gym_wrapper.GymFromDMEnv(raw_env)
            self._observation_space = self._env.observation_space
            self._action_space = self._env.action_space
            self._reward_space = gym.spaces.Box(
                low=self._env.reward_range[0], high=self._env.reward_range[1], shape=(1,), dtype=np.float64
            )
            self._init_flag = True
        if hasattr(self, '_seed') and hasattr(self, '_dynamic_seed') and self._dynamic_seed:
            np_seed = 100 * np.random.randint(1, 1000)
            self._seed = self._seed + np_seed
            obs, _ = self._env.reset(seed=self._seed)
        elif hasattr(self, '_seed'):
            obs, _ = self._env.reset(seed=self._seed)
        else:
            obs, _ = self._env.reset()
        if obs.shape[0] == 1:
            obs = obs[0]
        obs = to_ndarray(obs).astype(np.float32)
        self._eval_episode_return = 0

        action_mask = np.ones(self.action_space.n, 'int8')
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
        obs, rew, terminated, truncated, info = self._env.step(action)
        done = terminated or truncated
        self._eval_episode_return += rew
        if done:
            info['eval_episode_return'] = self._eval_episode_return
        if obs.shape[0] == 1:
            obs = obs[0]
        obs = to_ndarray(obs)
        rew = to_ndarray([rew])  # wrapped to be transfered to an array with shape (1,)

        action_mask = np.ones(self.action_space.n, 'int8')
        obs = {'observation': obs, 'action_mask': action_mask, 'to_play': -1}

        return BaseEnvTimestep(obs, rew, done, info)

    def config_info(self) -> dict:
        config_info = sweep.SETTINGS[self.env_name]  # additional info that are specific to each env configuration
        config_info['num_episodes'] = self._env.bsuite_num_episodes
        return config_info

    def random_action(self) -> np.ndarray:
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

    def __repr__(self) -> str:
        return "LightZero BSuite Env({})".format(self.env_name)

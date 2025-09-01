# toy_env.py
import copy
from typing import List
import gym
import numpy as np
from ding.envs import BaseEnv, BaseEnvTimestep
from ding.utils import ENV_REGISTRY
from easydict import EasyDict

@ENV_REGISTRY.register('toy_lightzero')
class ToyEnv(BaseEnv):
    """
    Overview:
        A simple, deterministic toy environment for debugging KV cache and long-sequence processing in UniZero.
        - State: 4-dim vector.
        - Actions: 3 discrete actions (stay, increment, decrement).
        - Episode Length: Fixed at 15 steps.
        - Returns 'timestep' in observation.
    """
    config = dict(
        env_id='toy-v0',
        env_type='Toy',
        observation_shape=(4,),
        action_space_size=3,
        collect_max_episode_steps=15,
        eval_max_episode_steps=15,
        manager=dict(shared_memory=False),
        stop_value=100,
    )

    @classmethod
    def default_config(cls: type) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    def __init__(self, cfg: EasyDict) -> None:
        self.cfg = cfg
        self._init_flag = False
        self._observation_space = gym.spaces.Dict({
            'observation': gym.spaces.Box(low=-np.inf, high=np.inf, shape=self.cfg.observation_shape, dtype=np.float32),
            'action_mask': gym.spaces.Box(low=0, high=1, shape=(self.cfg.action_space_size,), dtype=np.int8),
            'to_play': gym.spaces.Box(low=-1, high=2, shape=(), dtype=np.int8),
            'timestep': gym.spaces.Box(low=0, high=self.cfg.collect_max_episode_steps, shape=(), dtype=np.int32),
        })
        self._action_space = gym.spaces.Discrete(self.cfg.action_space_size)
        self._reward_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

    def reset(self) -> dict:
        if not self._init_flag:
            self._init_flag = True
        self._state = np.zeros(self.cfg.observation_shape, dtype=np.float32)
        self._episode_steps = 0
        self._eval_episode_return = 0.0
        return self.observe()

    def step(self, action: int) -> BaseEnvTimestep:
        if action == 1:
            self._state += 1
        elif action == 2:
            self._state -= 1
        
        self._episode_steps += 1
        reward = np.array([1.0], dtype=np.float32)
        self._eval_episode_return += reward

        done = self._episode_steps >= self.cfg.collect_max_episode_steps
        info = {}
        if done:
            info['eval_episode_return'] = self._eval_episode_return

        return BaseEnvTimestep(self.observe(), reward, done, info)

    def observe(self) -> dict:
        return {
            'observation': self._state.copy(),
            'action_mask': np.ones(self.cfg.action_space_size, dtype=np.int8),
            'to_play': np.array(-1, dtype=np.int8),
            'timestep': np.array(self._episode_steps, dtype=np.int32)
        }

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def close(self) -> None:
        self._init_flag = False

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
        return "LightZero Toy Env"

    @staticmethod
    def create_collector_env_cfg(cfg: dict) -> List[dict]:
        collector_env_num = cfg.pop('collector_env_num')
        return [cfg for _ in range(collector_env_num)]

    @staticmethod
    def create_evaluator_env_cfg(cfg: dict) -> List[dict]:
        evaluator_env_num = cfg.pop('evaluator_env_num')
        return [cfg for _ in range(evaluator_env_num)]
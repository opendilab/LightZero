from typing import Union, Optional

import gym
import numpy as np
from itertools import product
import logging

from ding.envs import BaseEnv, BaseEnvTimestep
from ding.envs import ObsPlusPrevActRewWrapper
from ding.torch_utils import to_ndarray
from ding.utils import ENV_REGISTRY

import zoo.CrowdSim.envs.Crowdsim.env


@ENV_REGISTRY.register('crowdsim_lightzero')
class CrowdSimEnv(BaseEnv):

    def __init__(self, cfg: dict = {}) -> None:
        self._cfg = cfg
        self._init_flag = False
        self._replay_path = None
        self._robot_num = self._cfg.robot_num
        self._human_num = self._cfg.human_num
        self._observation_space = gym.spaces.Dict({
            'robot_state': gym.spaces.Box(
                low=float("-inf"),
                high=float("inf"),
                shape=(self._robot_num, 4),
                dtype=np.float32
            ),
            'human_state': gym.spaces.Box(
                low=float("-inf"),
                high=float("inf"),
                shape=(self._human_num, 4),
                dtype=np.float32
            )
        })
        # action space
        # one_uav_action_space = [[0, 0], [30, 0], [-30, 0], [0, 30], [0, -30]]
        self.real_action_space = list(product(self._cfg.one_uav_action_space, repeat=self._robot_num))
        one_uav_action_n = len(self._cfg.one_uav_action_space)
        self._action_space = gym.spaces.Discrete(one_uav_action_n**self._robot_num)
        self._action_space.seed(0)  # default seed
        self._reward_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1, ), dtype=np.float32)
        self._continuous = False

    def reset(self) -> np.ndarray:
        if not self._init_flag:
            self._env = gym.make('CrowdSim-v0', dataset = self._cfg.dataset, custom_config = self._cfg)
            self._init_flag = True
        if hasattr(self, '_seed') and hasattr(self, '_dynamic_seed') and self._dynamic_seed:
            np_seed = 100 * np.random.randint(1, 1000)
            self._env.seed(self._seed + np_seed)
            self._action_space.seed(self._seed + np_seed)
        elif hasattr(self, '_seed'):
            self._env.seed(self._seed)
            self._action_space.seed(self._seed)
        self._eval_episode_return = 0
        # process obs
        raw_obs = self._env.reset()
        obs_list = list(raw_obs.to_tensor())
        obs = {'robot_state': obs_list[0], 'human_state': obs_list[1]}
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

    def step(self, action: Union[int, np.ndarray]) -> BaseEnvTimestep:
        if isinstance(action, np.ndarray) and action.shape == (1, ):
            action = action.squeeze()  # 0-dim array
        real_action = self.real_action_space[action]
        assert isinstance(real_action, tuple) and len(real_action) == self._robot_num, "illegal action!"
        raw_obs, rew, done, info = self._env.step(real_action)
        obs_list = list(raw_obs.to_array())
        obs = {'robot_state': obs_list[0], 'human_state': obs_list[1]}

        self._eval_episode_return += rew
        if done:
            info['eval_episode_return'] = self._eval_episode_return
            # logging.INFO('one game finish!')

        action_mask = np.ones(self.action_space.n, 'int8')
        obs = {'observation': obs, 'action_mask': action_mask, 'to_play': -1}
        rew = to_ndarray([rew]).astype(np.float32)
        return BaseEnvTimestep(obs, rew, done, info)

    def enable_save_replay(self, replay_path: Optional[str] = None) -> None:
        if replay_path is None:
            replay_path = './video'
        self._replay_path = replay_path

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

    def __repr__(self) -> str:
        return "LightZero CrowdSim Env"

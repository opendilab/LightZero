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
        self._replay_path = cfg.get('replay_path', None)
        self._robot_num = self._cfg.robot_num
        self._human_num = self._cfg.human_num
        self._observation_space = gym.spaces.Dict(
            {
                'robot_state': gym.spaces.Box(
                    low=float("-inf"), high=float("inf"), shape=(self._robot_num, 4), dtype=np.float32
                ),
                'human_state': gym.spaces.Box(
                    low=float("-inf"), high=float("inf"), shape=(self._human_num, 4), dtype=np.float32
                )
            }
        )
        # action space
        # one_uav_action_space = [[0, 0], [30, 0], [-30, 0], [0, 30], [0, -30]]
        self.real_action_space = list(product(self._cfg.one_uav_action_space, repeat=self._robot_num))
        one_uav_action_n = len(self._cfg.one_uav_action_space)
        self._action_space = gym.spaces.Discrete(one_uav_action_n ** self._robot_num)
        self._action_space.seed(0)  # default seed
        self._reward_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1, ), dtype=np.float32)
        self._continuous = False
        # obs_mode 'dict': {'robot_state': robot_state, 'human_state': human_state}
        # obs_mode '2-dim-array': np.concatenate((robot_state, human_state), axis=0)
        # obs_mode '1-dim-array': np.concatenate((robot_state, human_state), axis=0).flatten()
        self.obs_mode = self._cfg.get('obs_mode', '2-dim-array')
        assert self.obs_mode in [
            'dict', '2-dim-array', '1-dim-array'
        ], "obs_mode should be 'dict' or '2-dim-array' or '1-dim-array'!"
        # action_mode 'combine': combine all robot actions into one action, action space size = one_uav_action_n**robot_num
        # action_mode 'separate': separate robot actions, shape (robot_num,), for each robot action space size = one_uav_action_n
        self.action_mode = self._cfg.get('action_mode', 'combine')
        assert self.action_mode in ['combine', 'separate'], "action_mode should be 'combine' or 'separate'!"

    def reset(self) -> np.ndarray:
        if not self._init_flag:
            self._env = gym.make('CrowdSim-v0', dataset=self._cfg.dataset, custom_config=self._cfg)
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
        if self.obs_mode == 'dict':
            obs = {'robot_state': obs_list[0], 'human_state': obs_list[1]}
        elif self.obs_mode == '2-dim-array':
            # robot_state: (robot_num, 4), human_state: (human_num, 4)
            obs = np.concatenate((obs_list[0], obs_list[1]), axis=0)
        elif self.obs_mode == '1-dim-array':
            obs = np.concatenate((obs_list[0], obs_list[1]), axis=0).flatten()
        action_mask = np.ones(self.action_space.n, 'int8')
        obs = {'observation': obs, 'action_mask': action_mask, 'to_play': -1}
        if self._replay_path is not None:
            self._frame = []

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
        if self.action_mode == 'combine':
            if isinstance(action, np.ndarray) and action.shape == (1, ):
                action = action.squeeze()
            real_action = self.real_action_space[action]
        elif self.action_mode == 'separate':
            assert isinstance(action, np.ndarray) and action.shape == (self._robot_num, ), "illegal action!"
            real_action = tuple([self._cfg.one_uav_action_space[action[i]] for i in range(self._robot_num)])
        assert isinstance(real_action, tuple) and len(real_action) == self._robot_num, "illegal action!"
        raw_obs, rew, done, info = self._env.step(real_action)
        obs_list = list(raw_obs.to_array())
        if self.obs_mode == 'dict':
            obs = {'robot_state': obs_list[0], 'human_state': obs_list[1]}
        elif self.obs_mode == '2-dim-array':
            # robot_state: (robot_num, 4), human_state: (human_num, 4)
            obs = np.concatenate((obs_list[0], obs_list[1]), axis=0)
        elif self.obs_mode == '1-dim-array':
            obs = np.concatenate((obs_list[0], obs_list[1]), axis=0).flatten()

        self._eval_episode_return += rew
        if done:
            info['eval_episode_return'] = self._eval_episode_return
            # logging.INFO('one game finish!')

        action_mask = np.ones(self.action_space.n, 'int8')
        obs = {'observation': obs, 'action_mask': action_mask, 'to_play': -1}
        rew = to_ndarray([rew]).astype(np.float32)
        if self._replay_path is not None:
            self._frame.append(self._env.render())
            if done:
                import imageio, os
                if not os.path.exists(self._replay_path):
                    os.makedirs(self._replay_path)
                imageio.mimsave(self._replay_path + '/replay.gif', self._frame)
                # save env.human_df as csv
                self._env.human_df.to_csv(self._replay_path + '/human_df.csv')
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

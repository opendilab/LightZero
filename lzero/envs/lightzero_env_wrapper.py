import copy
from typing import List

import gym
import numpy as np
from ding.envs import BaseEnvTimestep
from ding.utils import ENV_WRAPPER_REGISTRY
from itertools import product
from ding.torch_utils import to_ndarray, to_list
from easydict import EasyDict


@ENV_WRAPPER_REGISTRY.register('lightzero_env_wrapper')
class LightZeroEnvWrapper(gym.ObservationWrapper):
    """
    Overview:
       Package the classic_contol, box2d environment into the format required by LightZero.
       Wrap obs as a dict, containing keys: obs, action_mask and to_play.
    Interface:
        ``__init__``, ``step``, ``reset``
    Properties:
        - env (:obj:`gym.Env`): the environment to wrap.
    """
    
    def __init__(self, env: gym.Env, cfg: EasyDict) -> None:
        """
        Overview:
            Initialize ``self.`` See ``help(type(self))`` for accurate signature;  \
                setup the properties according to running mean and std.
        Arguments:
            - env (:obj:`gym.Env`): the environment to wrap.
        """
        super().__init__(env)
        assert 'is_train' in cfg, '`is_train` flag must set in the config of env'
        self.is_train = cfg.is_train
        if self.is_train:  # set member variables which are enabled when is_train = True
            self.is_train_enabled_varaible = True
        else:  # else case
            self.is_train_enabled_varaible = False
        self.env_type = cfg.get("env_type", None)
        if self.is_train_enabled_varaible:
            if self.env_type is not None and self.env_type == "Atari":
                cfg.max_episode_steps = cfg.collect_max_episode_steps
                cfg.episode_life = True
                cfg.clip_rewards = True
        else:
            if self.env_type is not None and self.env_type == "Atari":
                cfg.max_episode_steps = cfg.eval_max_episode_steps
                cfg.episode_life = False
                cfg.clip_rewards = False
            
        self._env = env
        self.cfg = cfg
        self.env_name = cfg.env_name
        self.continuous = cfg.continuous
        if self.continuous:
            if self.env_name == "Pendulum-v1":
                self._action_space = gym.spaces.Box(low=-2.0, high=2.0, shape=(1, ), dtype=np.float32)
            elif self.env_name == "LunarLanderContinuous-v2":
                self._action_space = env.action_space
            elif self.env_name == "BipedalWalker-v3":
                self._action_space = env.action_space

    def step(self, action):
        """
        Overview:
            Step the environment with the given action. Repeat action, sum reward,  \
                and update ``data_count``, and also update the ``self.rms`` property  \
                    once after integrating with the input ``action``.
        Arguments:
            - action (:obj:`Any`): the given action to step with.
        Returns:
            - ``self.observation(observation)`` : normalized observation after the  \
                input action and updated ``self.rms``
            - reward (:obj:`Any`) : amount of reward returned after previous action
            - done (:obj:`Bool`) : whether the episode has ended, in which case further  \
                step() calls will return undefined results
            - info (:obj:`Dict`) : contains auxiliary diagnostic information (helpful  \
                for debugging, and sometimes learning)

        """
        if isinstance(action, int):
            action = np.array(action)
        if self.env_name == "Pendulum-v1":
            if self.cfg.discretization:
                # if require discrete env, convert actions to [-1 ~ 1] float actions
                action = (action / (self.discrete_action_num - 1)) * 2 - 1
                if action.shape in [0, ()]:
                    # to be compatiable with pendulum
                    action = np.array([action])
        if self.env_name in ["LunarLanderContinuous-v2", "BipedalWalker-v3"]:
            if self.cfg.discretization:
                action = [-1 + 2 / self.n * k for k in self.disc_to_cont[int(action)]]
                action = to_ndarray(action)
                if action.shape == (1,):
                    action = action.item()  # 0-dim array

        obs, rew, done, info = self.env.step(action)

        self._eval_episode_return += rew
        if done:
            info['eval_episode_return'] = self._eval_episode_return

        self._raw_obs_space = self._env.observation_space
        if self.env_name == "Pendulum-v1":
            self.discrete_action_num = 11
            if self.cfg.discretization:
                action_mask = np.ones(self.discrete_action_num, 'int8')
            else:
                action_mask = None
        elif self.env_name in ["LunarLander-v2"]:
            self.raw_action_space = self._env.action_space
            action_mask = np.ones(self.raw_action_space.n, 'int8')
        elif self.env_name in ["LunarLanderContinuous-v2", "BipedalWalker-v3"]:
            if self.cfg.discretization:
                # disc_to_cont: transform discrete action index to original continuous action
                self.raw_action_space = self._env.action_space
                self.m = self.raw_action_space.shape[0]
                self.n = self.cfg.each_dim_disc_size
                self.K = self.n ** self.m
                self.disc_to_cont = list(product(*[list(range(self.n)) for dim in range(self.m)]))
                # the modified discrete action space
                self._action_space = gym.spaces.Discrete(self.K)
                action_mask = np.ones(self.K, 'int8')
            else:
                action_mask = None

        # to be compatible with LightZero model,shape: [W, H, C]
        obs = obs.reshape(self._raw_obs_space.shape[0], 1, 1)

        obs = {'observation': obs, 'action_mask': action_mask, 'to_play': -1}
        return BaseEnvTimestep(obs, rew, done, info)

    def reset(self, **kwargs):
        """
        Overview:
            Resets the state of the environment and reset properties.
        Arguments:
            - kwargs (:obj:`Dict`): Reset with this key argumets
        Returns:
            - observation (:obj:`Any`): New observation after reset

        """
        obs = self.env.reset(**kwargs)
        self._eval_episode_return = 0.
        self._raw_obs_space = self._env.observation_space

        if self.env_name == "Pendulum-v1":
            self.discrete_action_num = 11
            self._action_space = gym.spaces.Discrete(self.discrete_action_num)
            if self.cfg.discretization:
                action_mask = np.ones(self.discrete_action_num, 'int8')
            else:
                action_mask = None
        elif self.env_name in ["LunarLander-v2"]:
            self.raw_action_space = self._env.action_space
            action_mask = np.ones(self.raw_action_space.n, 'int8')
        elif self.env_name in ["LunarLanderContinuous-v2", "BipedalWalker-v3"]:
            if self.cfg.discretization:
                # disc_to_cont: transform discrete action index to original continuous action
                self.raw_action_space = self._env.action_space
                self.m = self.raw_action_space.shape[0]
                self.n = self.cfg.each_dim_disc_size
                self.K = self.n ** self.m
                self.disc_to_cont = list(product(*[list(range(self.n)) for dim in range(self.m)]))
                # the modified discrete action space
                self._action_space = gym.spaces.Discrete(self.K)
                action_mask = np.ones(self.K, 'int8')
            else:
                action_mask = None

        # to be compatible with LightZero model,shape: [W, H, C]
        obs = obs.reshape(self._raw_obs_space.shape[0], 1, 1)

        obs = {'observation': obs, 'action_mask': action_mask, 'to_play': -1}
        return obs

    def __repr__(self) -> str:
        return "LightZero Env."

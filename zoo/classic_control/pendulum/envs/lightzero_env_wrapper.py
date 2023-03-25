import copy
from typing import List

import gym
import numpy as np
from ding.envs import BaseEnvTimestep
from ding.utils import ENV_WRAPPER_REGISTRY


@ENV_WRAPPER_REGISTRY.register('obs_plus_action_mask_to_play')
class ObsActionMaskToPlayWrapper(gym.ObservationWrapper):
    """
    Overview:
       Normalize observations according to running mean and std.
    Interface:
        ``__init__``, ``step``, ``reset``, ``observation``, ``new_shape``
    Properties:
        - env (:obj:`gym.Env`): the environment to wrap.

        - ``data_count``, ``clip_range``, ``rms``
    """
    
    def __init__(self, env, cfg):
        """
        Overview:
            Initialize ``self.`` See ``help(type(self))`` for accurate signature;  \
                setup the properties according to running mean and std.
        Arguments:
            - env (:obj:`gym.Env`): the environment to wrap.
        """
        super().__init__(env)
        self._continuous = cfg.continuous
        if self._continuous:
            self._action_space = gym.spaces.Box(low=-2.0, high=2.0, shape=(1, ), dtype=np.float32)
        else:
            self._discrete_action_num = 11
            self._action_space = gym.spaces.Discrete(self._discrete_action_num)

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
        # if require discrete env, convert actions to [-1 ~ 1] float actions
        if not self._continuous:
            action = (action / (self._discrete_action_num - 1)) * 2 - 1
            if action.shape in [0, ()]:
                # to be compatiable with pendulum
                action = np.array([action])

        obs, rew, done, info = self.env.step(action)

        self._eval_episode_return += rew
        if done:
            info['eval_episode_return'] = self._eval_episode_return

        obs = obs.reshape(obs.shape[0], 1, 1)
        if not self._continuous:
            action_mask = np.ones(self._discrete_action_num, 'int8')
        else:
            action_mask = None
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
        # to be compatible with LightZero model,shape: [W, H, C]
        obs = obs.reshape(obs.shape[0], 1, 1)
        if not self._continuous:
            action_mask = np.ones(self._discrete_action_num, 'int8')
        else:
            action_mask = None
        obs = {'observation': obs, 'action_mask': action_mask, 'to_play': -1}
        return obs

    def __repr__(self) -> str:
        return "LightZero Env."

    @staticmethod
    def create_collector_env_cfg(cfg: dict) -> List[dict]:
        collector_env_num = cfg.pop('collector_env_num')
        cfg = copy.deepcopy(cfg)
        cfg.max_episode_steps = cfg.collect_max_episode_steps
        return [cfg for _ in range(collector_env_num)]

    @staticmethod
    def create_evaluator_env_cfg(cfg: dict) -> List[dict]:
        evaluator_env_num = cfg.pop('evaluator_env_num')
        cfg = copy.deepcopy(cfg)
        cfg.max_episode_steps = cfg.eval_max_episode_steps
        return [cfg for _ in range(evaluator_env_num)]
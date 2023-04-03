from itertools import product

import gym
import numpy as np
from easydict import EasyDict

from ding.envs import BaseEnvTimestep
from ding.torch_utils import to_ndarray
from ding.utils import ENV_WRAPPER_REGISTRY


@ENV_WRAPPER_REGISTRY.register('action_discretization_env_wrapper')
class ActionDiscretizationEnvWrapper(gym.Wrapper):
    """
    Overview:
        The modified environment with manually discretized action space. For each dimension, equally dividing the
        original continuous action into ``each_dim_disc_size`` bins and using their Cartesian product to obtain
        handcrafted discrete actions.
    Interface:
        ``__init__``,  ``reset``, ``step``
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
        self.cfg = cfg
        self.env_name = cfg.env_name
        self.continuous = cfg.continuous

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
        self._raw_action_space = self.env.action_space

        if self.cfg.discretization:
            # disc_to_cont: transform discrete action index to original continuous action
            self.m = self._raw_action_space.shape[0]
            self.n = self.cfg.each_dim_disc_size
            self.K = self.n ** self.m
            self.disc_to_cont = list(product(*[list(range(self.n)) for dim in range(self.m)]))
            # the modified discrete action space
            self._action_space = gym.spaces.Discrete(self.K)

        return obs

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
        if self.cfg.discretization:
            # disc_to_cont: transform discrete action index to original continuous action
            action = [-1 + 2 / self.n * k for k in self.disc_to_cont[int(action)]]
            action = to_ndarray(action)
            if action.shape == (1,):
                action = action.item()  # 0-dim array

        # The core original env step.
        obs, rew, done, info = self.env.step(action)

        return BaseEnvTimestep(obs, rew, done, info)

    def __repr__(self) -> str:
        return "Action Discretization Env."

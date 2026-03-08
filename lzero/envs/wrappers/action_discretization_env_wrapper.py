from itertools import product

import gymnasium as gym
import numpy as np
from easydict import EasyDict

from ding.envs import BaseEnvTimestep
from ding.torch_utils import to_ndarray
from ding.utils import ENV_WRAPPER_REGISTRY


class _LegacyEnvAdapter(gym.Env):
    """
    Adapt legacy env objects so they satisfy gymnasium's ``Env`` type checks.
    """

    metadata = {}
    reward_range = (-float("inf"), float("inf"))
    spec = None
    render_mode = None

    def __init__(self, env) -> None:
        self.legacy_env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        if hasattr(env, "metadata"):
            self.metadata = env.metadata
        if hasattr(env, "reward_range"):
            self.reward_range = env.reward_range
        if hasattr(env, "spec"):
            self.spec = env.spec
        if hasattr(env, "render_mode"):
            self.render_mode = env.render_mode

    def reset(self, **kwargs):
        return self.legacy_env.reset(**kwargs)

    def step(self, action):
        return self.legacy_env.step(action)

    def render(self):
        return self.legacy_env.render()

    def close(self):
        return self.legacy_env.close()

    @property
    def unwrapped(self):
        return getattr(self.legacy_env, "unwrapped", self.legacy_env)

    def __getattr__(self, name):
        return getattr(self.legacy_env, name)


def _coerce_gymnasium_env(env):
    if isinstance(env, gym.Env):
        return env
    return _LegacyEnvAdapter(env)


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
        super().__init__(_coerce_gymnasium_env(env))
        assert 'is_train' in cfg, '`is_train` flag must set in the config of env'
        self.is_train = cfg.is_train
        self.cfg = cfg
        self.env_id = cfg.env_id
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
        reset_result = self.env.reset(**kwargs)
        if isinstance(reset_result, tuple):
            obs, _ = reset_result
        else:
            obs = reset_result
        self._raw_action_space = self.env.action_space

        if self.cfg.manually_discretization:
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
        if self.cfg.manually_discretization:
            # disc_to_cont: transform discrete action index to original continuous action
            action = [-1 + 2 / self.n * k for k in self.disc_to_cont[int(action)]]
            action = to_ndarray(action)

        # The core original env step.
        step_result = self.env.step(action)
        if len(step_result) == 5:
            obs, rew, terminated, truncated, info = step_result
            done = terminated or truncated
            if truncated:
                info = dict(info)
                info.setdefault('TimeLimit.truncated', True)
        else:
            obs, rew, done, info = step_result

        return BaseEnvTimestep(obs, rew, done, info)

    def __repr__(self) -> str:
        return "Action Discretization Env."

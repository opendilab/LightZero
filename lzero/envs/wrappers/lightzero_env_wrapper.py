import gymnasium as gym
import numpy as np
from easydict import EasyDict

from ding.envs import BaseEnvTimestep
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


@ENV_WRAPPER_REGISTRY.register('lightzero_env_wrapper')
class LightZeroEnvWrapper(gym.Wrapper):
    """
    Overview:
       Package the classic_control, box2d environment into the format required by LightZero.
       Wrap obs as a dict, containing keys: obs, action_mask and to_play.
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
        # The core original env reset.
        reset_result = self.env.reset(**kwargs)
        if isinstance(reset_result, tuple):
            obs, _ = reset_result
        else:
            obs = reset_result
        self._eval_episode_return = 0.
        self._raw_observation_space = self.env.observation_space

        if self.cfg.continuous:
            action_mask = None
        else:
            action_mask = np.ones(self.env.action_space.n, 'int8')

        if self.cfg.continuous:
            self._observation_space = gym.spaces.Dict(
                {
                    'observation': self._raw_observation_space,
                    'action_mask': gym.spaces.Box(low=np.inf, high=np.inf,
                                                  shape=(1, )),  # TODO: gym.spaces.Constant(None)
                    'to_play': gym.spaces.Box(low=-1, high=-1, shape=(1, )),  # TODO: gym.spaces.Constant(-1)
                }
            )
        else:
            self._observation_space = gym.spaces.Dict(
                {
                    'observation': self._raw_observation_space,
                    'action_mask': gym.spaces.MultiDiscrete([2 for _ in range(self.env.action_space.n)])
                    if isinstance(self.env.action_space, gym.spaces.Discrete) else
                    gym.spaces.MultiDiscrete([2 for _ in range(self.env.action_space.shape[0])]),  # {0,1}
                    'to_play': gym.spaces.Box(low=-1, high=-1, shape=(1, )),  # TODO: gym.spaces.Constant(-1)
                }
            )

        lightzero_obs_dict = {'observation': obs, 'action_mask': action_mask, 'to_play': -1}
        return lightzero_obs_dict

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

        if self.cfg.continuous:
            action_mask = None
        else:
            action_mask = np.ones(self.env.action_space.n, 'int8')

        lightzero_obs_dict = {'observation': obs, 'action_mask': action_mask, 'to_play': -1}

        self._eval_episode_return += rew
        if done:
            info['eval_episode_return'] = self._eval_episode_return

        return BaseEnvTimestep(lightzero_obs_dict, rew, done, info)

    def __repr__(self) -> str:
        return "LightZero Env."

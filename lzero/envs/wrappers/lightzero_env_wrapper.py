import gym
import numpy as np
from easydict import EasyDict

from ding.envs import BaseEnvTimestep
from ding.utils import ENV_WRAPPER_REGISTRY

@ENV_WRAPPER_REGISTRY.register('lightzero_env_wrapper')
class LightZeroEnvWrapper(gym.Wrapper):
    """
    Wraps the gym environment into a format required by LightZero.
    The observation is wrapped into a dict containing 'obs', 'action_mask', and 'to_play'.
    """
    def __init__(self, env: gym.Env, cfg: EasyDict) -> None:
        """
        Initialize the environment wrapper.
        Args:
            env: Original gym environment.
            cfg: Configuration dictionary. Must contain 'is_train' flag.
        """
        super().__init__(env)
        assert 'is_train' in cfg, '`is_train` flag must set in the config of env'
        self.is_train = cfg.is_train
        self.cfg = cfg
        self.env_name = cfg.env_name
        self.continuous = cfg.continuous

        # If environment is continuous, action_mask is None, else it's an array of ones
        self.action_mask = None if self.continuous else np.ones(self.env.action_space.n, 'int8')

        # Define observation space based on whether the environment is continuous or not
        if self.continuous:
            self._observation_space = gym.spaces.Dict(
                {
                    'observation': self.env.observation_space,
                    'action_mask': gym.spaces.Box(low=np.inf, high=np.inf, shape=(1, )),
                    'to_play': gym.spaces.Box(low=-1, high=-1, shape=(1, )),
                }
            )
        else:
            self._observation_space = gym.spaces.Dict(
                {
                    'observation': self.env.observation_space,
                    'action_mask': gym.spaces.MultiDiscrete(
                        [2 for _ in range(self.env.action_space.shape[0])]
                    ),
                    'to_play': gym.spaces.Box(low=-1, high=-1, shape=(1, )),
                }
            )

    def reset(self, **kwargs):
        """
        Reset the environment.
        Args:
            **kwargs: Arbitrary keyword arguments.
        Returns:
            dict: New observation after reset.
        """
        obs = self.env.reset(**kwargs)
        self._eval_episode_return = 0.
        return {'observation': obs, 'action_mask': self.action_mask, 'to_play': -1}

    def step(self, action):
        """
        Step the environment with the given action.
        Args:
            action: Action to apply.
        Returns:
            BaseEnvTimestep: Observation, reward, done status, and info after the action.
        """
        obs, rew, done, info = self.env.step(action)
        lightzero_obs_dict = {'observation': obs, 'action_mask': self.action_mask, 'to_play': -1}
        self._eval_episode_return += rew
        if done:
            info['eval_episode_return'] = self._eval_episode_return
        return BaseEnvTimestep(lightzero_obs_dict, rew, done, info)

    def __repr__(self) -> str:
        return "LightZero Env."
import os
from typing import Union

import gym
import numpy as np
from ding.envs import BaseEnvTimestep
from ding.envs.common import save_frames_as_gif
from ding.torch_utils import to_ndarray
from ding.utils import ENV_REGISTRY
from dizoo.multiagent_mujoco.envs.multi_mujoco_env import MujocoEnv,MujocoMulti


@ENV_REGISTRY.register('multiagent_mujoco_lightzero')
class MAMujocoEnvLZ(MujocoEnv):
    """
    Overview:
        The modified Multi-agentMuJoCo environment with continuous action space for LightZero's algorithms. \
        You can find the original implementation at \
        [Multi-Agent Mujoco](https://robotics.farama.org/envs/MaMuJoCo/index.html). The class is registered \
        in ENV_REGISTRY with the key 'multiagent_mujoco_lightzero'.
    """

    config = dict(
        stop_value=int(1e6),
        norm_obs=dict(use_norm=False, ),
        norm_reward=dict(use_norm=False, ),
    )

    def __init__(self, cfg: dict) -> None:
        """
        Overview:
            Initialize the Multi-agent MuJoCo environment.
        Arguments:
            - cfg (:obj:`dict`): Config dict. The following keys must be specified:
                - 'env_name' (:obj:`str`): The name of the environment.
                - 'scenario' (:obj:`str`): The scenario of the environment.
                - 'agent_conf' (:obj:`str`): The configuration of the agents.
                - 'agent_obsk' (:obj:`int`): The observation space of the agents.
                - 'add_agent_id' (:obj:`bool`): Whether to add agent id to the observation.
                - 'episode_limit' (:obj:`int`): The maximum number of episodes.
        """
        super().__init__(cfg)
        self._cfg = cfg
        # We use env_name to indicate the env_id in LightZero.
        self._cfg.env_id = self._cfg.env_name
        self._init_flag = False

    def reset(self) -> np.ndarray:
        """
        Overview:
            Reset the environment and return the initial observation.
        Returns:
            - obs (:obj:`np.ndarray`): The initial observation after resetting. The observation is a dict with keys \
                'observation', 'action_mask', and 'to_play'. The 'observation' is a dict with keys 'agent_state' and \
                'global_state'.
        """
        if not self._init_flag:
            self._env = MujocoMulti(env_args=self._cfg)
            self._init_flag = True
            
        if hasattr(self, '_seed') and hasattr(self, '_dynamic_seed') and self._dynamic_seed:
            np_seed = 100 * np.random.randint(1, 1000)
            self._env.seed(self._seed + np_seed)
        elif hasattr(self, '_seed'):
            self._env.seed(self._seed)
        
        obs = self._env.reset()
        obs = to_ndarray(obs)
        self._eval_episode_return = 0.
        self.env_info = self._env.get_env_info()

        self._num_agents = self.env_info['n_agents']
        self._agents = [i for i in range(self._num_agents)]
        self._observation_space = gym.spaces.Dict(
            {
                'agent_state': gym.spaces.Box(
                    low=float("-inf"), high=float("inf"), shape=obs['agent_state'].shape, dtype=np.float32
                ),
                'global_state': gym.spaces.Box(
                    low=float("-inf"), high=float("inf"), shape=obs['global_state'].shape, dtype=np.float32
                ),
            }
        )
        self._action_space = gym.spaces.Dict({agent: self._env.action_space[agent] for agent in self._agents})
        single_agent_obs_space = self._env.action_space[self._agents[0]]
        if isinstance(single_agent_obs_space, gym.spaces.Box):
            self._action_dim = single_agent_obs_space.shape
        elif isinstance(single_agent_obs_space, gym.spaces.Discrete):
            self._action_dim = (single_agent_obs_space.n, )
        else:
            raise Exception('Only support `Box` or `Discrte` obs space for single agent.')
        self._reward_space = gym.spaces.Dict(
            {
                agent: gym.spaces.Box(low=float("-inf"), high=float("inf"), shape=(1, ), dtype=np.float32)
                for agent in self._agents
            }
        )

        action_mask = None
        obs = {'observation': obs, 'action_mask': action_mask, 'to_play': -1}

        return obs

    def step(self, action: Union[np.ndarray, list]) -> BaseEnvTimestep:
        """
        Overview:
            Take a step in the environment with the given action.
        Arguments:
            - action (:obj:`np.ndarray`): The action to be taken.
        Returns:
            - timestep (:obj:`BaseEnvTimestep`): The timestep information including observation, reward, done flag, and info.
        """
        action = to_ndarray(action)
        obs, rew, done, info = self._env.step(action)
        self._eval_episode_return += rew
        if done:
            info['eval_episode_return'] = self._eval_episode_return

        obs = to_ndarray(obs)
        rew = to_ndarray([rew]).astype(np.float32)

        action_mask = None
        obs = {'observation': obs, 'action_mask': action_mask, 'to_play': -1}

        return BaseEnvTimestep(obs, rew, done, info)

    def __repr__(self) -> str:
        return "LightZero MAMujoco Env({})".format(self._cfg.env_name)


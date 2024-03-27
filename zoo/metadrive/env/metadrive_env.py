from typing import Any, Dict, Optional
from easydict import EasyDict
import matplotlib.pyplot as plt
import gymnasium as gym
import copy
import numpy as np
from ding.envs.env.base_env import BaseEnv, BaseEnvTimestep
from ding.torch_utils.data_helper import to_ndarray
from ding.utils.default_helper import deep_merge_dicts
from ding.utils import ENV_REGISTRY

# 记得换import!!!!!!!!!!!!!!!!!!!!!!!!!!!
from zoo.metadrive.env.drive_env import MetaDrive
# from dizoo.metadrive.env.drive_utils import BaseDriveEnv
# from zoo.metadrive.env.drive_utils import BaseDriveEnv

def draw_multi_channels_top_down_observation(obs, show_time=0.5):
    num_channels = obs.shape[-1]
    assert num_channels == 5
    channel_names = [
        "Road and navigation", "Ego now and previous pos", "Neighbor at step t", "Neighbor at step t-1",
        "Neighbor at step t-2"
    ]
    fig, axs = plt.subplots(1, num_channels, figsize=(15, 4), dpi=80)
    count = 0

    def close_event():
        plt.close()

    timer = fig.canvas.new_timer(interval=show_time * 1000)
    timer.add_callback(close_event)
    for i, name in enumerate(channel_names):
        count += 1
        ax = axs[i]
        ax.imshow(obs[..., i], cmap="bone")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(name)
    fig.suptitle("Multi-channels Top-down Observation")
    timer.start()
    plt.show()
    plt.close()

@ENV_REGISTRY.register('metadrive_lightzero')
class MetaDriveEnv(BaseEnv):
    """
    MetaDrive environment in LightZero.
    """
    config = dict(
        # (bool) Whether to use continuous action space
        continuous=True,
        # replay_path (str or None): The path to save the replay video. If None, the replay will not be saved.
        # Only effective when env_manager.type is 'base'.
        replay_path=None,
        # (bool) Whether to scale action into [-2, 2]
        act_scale=True,

    )
    @classmethod
    def default_config(cls: type) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg
    
    def __init__(self, cfg: dict = {}) -> None:
        """
        Initialize the environment with a configuration dictionary. Sets up spaces for observations, actions, and rewards.
        """
        # Initialize a raw env
        self._cfg = cfg
        self._env = MetaDrive(self._cfg)
        self._init_flag = True

        # Initialize the spaces
        #!!!!!!!!!!!!!!!!!!!!!这个if能不能删掉啊
        # if not hasattr(self._env, 'reward_space'):
        self._reward_space = gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(1, ))
        self._action_space = self._env.action_space
        self._observation_space = self._env.observation_space

        # bird view
        self.show_bird_view = False

    def reset(self, *args, **kwargs) -> Any:
        """
        Overview:
            Wrapper of ``reset`` method in env. The observations are converted to ``np.ndarray`` and final reward
            are recorded.
        Returns:
            - Any: Observations from environment
        """
        obs = self._env.reset(*args, **kwargs)
        obs = to_ndarray(obs, dtype=np.float32)
        if isinstance(obs, np.ndarray) and len(obs.shape) == 3:
            # obs = obs.transpose((2, 0, 1))
            obs = obs
        elif isinstance(obs, dict):
            vehicle_state = obs['vehicle_state']
            # birdview = obs['birdview'].transpose((2, 0, 1))
            birdview = obs['birdview']
            obs = {'vehicle_state': vehicle_state, 'birdview': birdview}
        self._eval_episode_return = 0.0
        self._arrive_dest = False
        self._observation_space = self._env.observation_space
        
        metadrive_obs = {}
        metadrive_obs['observation'] = obs 
        # !!!!!!!!!!!!!!!!!!这边传none会不会有问题啊 
        metadrive_obs['action_mask'] = None 
        metadrive_obs['to_play'] = -1 
        return metadrive_obs
    
    def step(self, action: Any = None) -> BaseEnvTimestep:
        """
        Overview:
            Wrapper of ``step`` method in env. This aims to convert the returns of ``gym.Env`` step method into
            that of ``ding.envs.BaseEnv``, from ``(obs, reward, done, info)`` tuple to a ``BaseEnvTimestep``
            namedtuple defined in DI-engine. It will also convert actions, observations and reward into
            ``np.ndarray``, and check legality if action contains control signal.
        Arguments:
            - action (Any, optional): Actions sent to env. Defaults to None.
        Returns:
            - BaseEnvTimestep: DI-engine format of env step returns.
        """
        action = to_ndarray(action)
        obs, rew, done, info = self._env.step(action)
        if self.show_bird_view:
            draw_multi_channels_top_down_observation(obs, show_time=0.5)
        self._eval_episode_return += rew
        obs = to_ndarray(obs, dtype=np.float32)
        if isinstance(obs, np.ndarray) and len(obs.shape) == 3:
            # obs = obs.transpose((2, 0, 1))
            obs = obs
        elif isinstance(obs, dict):
            vehicle_state = obs['vehicle_state']
            # birdview = obs['birdview'].transpose((2, 0, 1))
            birdview = obs['birdview']
            obs = {'vehicle_state': vehicle_state, 'birdview': birdview}
        rew = to_ndarray([rew], dtype=np.float32)
        if done:
            info['eval_episode_return'] = self._eval_episode_return
        metadrive_obs = {}
        metadrive_obs['observation'] = obs  
        metadrive_obs['action_mask'] = None 
        metadrive_obs['to_play'] = -1 
        return BaseEnvTimestep(metadrive_obs, rew, done, info)
    
    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        """
        Set the seed for the environment's random number generator. Can handle both static and dynamic seeding.
        """
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def enable_save_replay(self, replay_path: Optional[str] = None) -> None:
        if replay_path is None:
            replay_path = './video'
        self._replay_path = replay_path
        self._env = gym.wrappers.Monitor(self._env, self._replay_path, video_callable=lambda episode_id: True, force=True)

    def render(self):
        self._env.render()

    @property
    def observation_space(self) -> gym.spaces.Space:
        """
        Property to access the observation space of the environment.
        """
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        """
        Property to access the action space of the environment.
        """
        return self._action_space

    @property
    def reward_space(self) -> gym.spaces.Space:
        """
        Property to access the reward space of the environment.
        """
        return self._reward_space
    
    def close(self) -> None:
        """
        Close the environment, and set the initialization flag to False.
        """
        if self._init_flag:
            self._env.close()
        self._init_flag = False
 
    def __repr__(self) -> str:
        return repr(self._env)

    def clone(self):
        cfg = copy.deepcopy(self._cfg)
        return MetaDriveEnv(cfg)
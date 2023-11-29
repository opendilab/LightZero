import copy
import os
from datetime import datetime
from typing import List, Optional, Dict

import gym
import numpy as np
from ding.envs import BaseEnvTimestep
from ding.envs import ObsPlusPrevActRewWrapper
from ding.envs.common import affine_transform
from ding.torch_utils import to_ndarray
from ding.utils import ENV_REGISTRY
from easydict import EasyDict

from zoo.classic_control.cartpole.envs.cartpole_lightzero_env import CartPoleEnv


@ENV_REGISTRY.register('lunarlander')
class LunarLanderEnv(CartPoleEnv):
    """
    Overview:
        The LunarLander Environment class for LightZero algo.. This class is a wrapper of the gym LunarLander environment, with additional
        functionalities like replay saving and seed setting. The class is registered in ENV_REGISTRY with the key 'lunarlander'.
    """

    config = dict(
        env_name="LunarLander-v2",
        save_replay_gif=False,
        replay_path_gif=None,
        replay_path=None,
        act_scale=True,
        delay_reward_step=0,
        prob_random_agent=0.,
        collect_max_episode_steps=int(1.08e5),
        eval_max_episode_steps=int(1.08e5),
    )

    @classmethod
    def default_config(cls: type) -> EasyDict:
        """
        Overview:
            Return the default configuration of the class.
        Returns:
            - cfg (:obj:`EasyDict`): Default configuration dict.
        """
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    def __init__(self, cfg: dict) -> None:
        """
        Overview:
            Initialize the LunarLander environment.
        Arguments:
            - cfg (:obj:`dict`): Configuration dict. The dict should include keys like 'env_name', 'replay_path', etc.
        """
        self._cfg = cfg
        self._init_flag = False
        # env_name options = {'LunarLander-v2', 'LunarLanderContinuous-v2'}
        self._env_name = cfg.env_name
        self._replay_path = cfg.get('replay_path', None)
        self._replay_path_gif = cfg.get('replay_path_gif', None)
        self._save_replay_gif = cfg.get('save_replay_gif', False)
        self._save_replay_count = 0
        if 'Continuous' in self._env_name:
            self._act_scale = cfg.act_scale  # act_scale only works in continuous env
        else:
            self._act_scale = False

    def reset(self) -> Dict[str, np.ndarray]:
        """
        Overview:
            Reset the environment and return the initial observation.
        Returns:
            - obs (:obj:`np.ndarray`): The initial observation after resetting.
        """
        if not self._init_flag:
            self._env = gym.make(self._cfg.env_name)
            if self._replay_path is not None:
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                video_name = f'{self._env.spec.id}-video-{timestamp}'
                self._env = gym.wrappers.RecordVideo(
                    self._env,
                    video_folder=self._replay_path,
                    episode_trigger=lambda episode_id: True,
                    name_prefix=video_name
                )
            if hasattr(self._cfg, 'obs_plus_prev_action_reward') and self._cfg.obs_plus_prev_action_reward:
                self._env = ObsPlusPrevActRewWrapper(self._env)
            self._observation_space = self._env.observation_space
            self._action_space = self._env.action_space
            self._reward_space = gym.spaces.Box(
                low=self._env.reward_range[0], high=self._env.reward_range[1], shape=(1,), dtype=np.float32
            )
            self._init_flag = True
        if hasattr(self, '_seed') and hasattr(self, '_dynamic_seed') and self._dynamic_seed:
            np_seed = 100 * np.random.randint(1, 1000)
            self._env.seed(self._seed + np_seed)
        elif hasattr(self, '_seed'):
            self._env.seed(self._seed)
        obs = self._env.reset()
        obs = to_ndarray(obs)
        self._eval_episode_return = 0.
        if self._save_replay_gif:
            self._frames = []
        if 'Continuous' not in self._env_name:
            action_mask = np.ones(4, 'int8')
            obs = {'observation': obs, 'action_mask': action_mask, 'to_play': -1}
        else:
            action_mask = None
            obs = {'observation': obs, 'action_mask': action_mask, 'to_play': -1}
        return obs

    def step(self, action: np.ndarray) -> BaseEnvTimestep:
        """
        Overview:
            Take a step in the environment with the given action.
        Arguments:
            - action (:obj:`np.ndarray`): The action to be taken.
        Returns:
            - timestep (:obj:`BaseEnvTimestep`): The timestep information including observation, reward, done flag, and info.
        """
        if action.shape == (1,):
            action = action.item()  # 0-dim array
        if self._act_scale:
            action = affine_transform(action, min_val=-1, max_val=1)
        if self._save_replay_gif:
            self._frames.append(self._env.render(mode='rgb_array'))

        obs, rew, done, info = self._env.step(action)
        if 'Continuous' not in self._env_name:
            action_mask = np.ones(4, 'int8')
            obs = {'observation': obs, 'action_mask': action_mask, 'to_play': -1}
        else:
            action_mask = None
            obs = {'observation': obs, 'action_mask': action_mask, 'to_play': -1}
        self._eval_episode_return += rew
        if done:
            info['eval_episode_return'] = self._eval_episode_return
            if self._save_replay_gif:
                if not os.path.exists(self._replay_path_gif):
                    os.makedirs(self._replay_path_gif)
                path = os.path.join(
                    self._replay_path_gif,
                    '{}_episode_{}_seed{}.gif'.format(self._env_name, self._save_replay_count, self._seed)
                )
                self.display_frames_as_gif(self._frames, path)
                print(f'save episode {self._save_replay_count} in {self._replay_path_gif}!')
                self._save_replay_count += 1
        obs = to_ndarray(obs)
        rew = to_ndarray([rew]).astype(np.float32)  # wrapped to be transferred to a array with shape (1,)
        return BaseEnvTimestep(obs, rew, done, info)

    @property
    def legal_actions(self) -> np.ndarray:
        """
        Overview:
            Get the legal actions in the environment.
        Returns:
            - legal_actions (:obj:`np.ndarray`): An array of legal actions.
        """
        return np.arange(self._action_space.n)

    def enable_save_replay(self, replay_path: Optional[str] = None) -> None:
        if replay_path is None:
            replay_path = './video'
        self._replay_path = replay_path
        self._save_replay_gif = True
        self._save_replay_count = 0

    @staticmethod
    def display_frames_as_gif(frames: list, path: str) -> None:
        import imageio
        imageio.mimsave(path, frames, fps=20)

    def random_action(self) -> np.ndarray:
        random_action = self.action_space.sample()
        if isinstance(random_action, np.ndarray):
            pass
        elif isinstance(random_action, int):
            random_action = to_ndarray([random_action], dtype=np.int64)
        return random_action

    def __repr__(self) -> str:
        return "LightZero LunarLander Env."

    @staticmethod
    def create_collector_env_cfg(cfg: dict) -> List[dict]:
        """
        Overview:
            Create a list of environment configurations for the collector.
        Arguments:
            - cfg (:obj:`dict`): The base configuration dict.
        Returns:
            - cfgs (:obj:`List[dict]`): The list of environment configurations.
        """
        collector_env_num = cfg.pop('collector_env_num')
        cfg = copy.deepcopy(cfg)
        cfg.max_episode_steps = cfg.collect_max_episode_steps
        return [cfg for _ in range(collector_env_num)]

    @staticmethod
    def create_evaluator_env_cfg(cfg: dict) -> List[dict]:
        """
        Overview:
            Create a list of environment configurations for the evaluator.
        Arguments:
            - cfg (:obj:`dict`): The base configuration dict.
        Returns:
            - cfgs (:obj:`List[dict]`): The list of environment configurations.
        """
        evaluator_env_num = cfg.pop('evaluator_env_num')
        cfg = copy.deepcopy(cfg)
        cfg.max_episode_steps = cfg.eval_max_episode_steps
        return [cfg for _ in range(evaluator_env_num)]

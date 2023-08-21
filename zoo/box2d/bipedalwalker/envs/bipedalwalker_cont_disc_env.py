import copy
import os
from itertools import product
from typing import List, Optional

import gym
import numpy as np
from ding.envs import BaseEnv, BaseEnvTimestep
from ding.envs.common.common_function import affine_transform
from ding.torch_utils import to_ndarray
from ding.utils import ENV_REGISTRY
from easydict import EasyDict


@ENV_REGISTRY.register('bipedalwalker_cont_disc')
class BipedalWalkerDiscEnv(BaseEnv):
    """
        Overview:
            The modified BipedalWalker environment with manually discretized action space. For each dimension, equally dividing the
            original continuous action into ``each_dim_disc_size`` bins and using their Cartesian product to obtain
            handcrafted discrete actions.
    """

    @classmethod
    def default_config(cls: type) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    config = dict(
        save_replay_gif=False,
        replay_path_gif=None,
        replay_path=None,
        act_scale=True,
        rew_clip=True,
        delay_reward_step=0,
        prob_random_agent=0.,
        collect_max_episode_steps=int(1.08e5),
        eval_max_episode_steps=int(1.08e5),
        each_dim_disc_size=4,
    )

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        self._init_flag = False
        self._act_scale = cfg.act_scale
        self._rew_clip = cfg.rew_clip
        self._replay_path = cfg.replay_path
        self._replay_path_gif = cfg.replay_path_gif
        self._save_replay_gif = cfg.save_replay_gif
        self._save_replay_count = 0

    def reset(self) -> np.ndarray:
        """
        Overview:
             During the reset phase, the original environment will be created,
             and at the same time, the action space will be discretized into "each_dim_disc_size" bins.
        Returns:
            - info_dict (:obj:`Dict[str, Any]`): Including observation, action_mask, and to_play label.
        """
        if not self._init_flag:
            self._env = gym.make('BipedalWalker-v3', hardcore=True)
            self._observation_space = self._env.observation_space
            self._action_space = self._env.action_space
            self._reward_space = gym.spaces.Box(
                low=self._env.reward_range[0], high=self._env.reward_range[1], shape=(1, ), dtype=np.float32
            )
            self._init_flag = True
        if hasattr(self, '_seed') and hasattr(self, '_dynamic_seed') and self._dynamic_seed:
            np_seed = 100 * np.random.randint(1, 1000)
            self._env.seed(self._seed + np_seed)
        elif hasattr(self, '_seed'):
            self._env.seed(self._seed)
        if self._replay_path is not None:
            self._env = gym.wrappers.RecordVideo(
                self._env,
                video_folder=self._replay_path,
                episode_trigger=lambda episode_id: True,
                name_prefix='rl-video-{}'.format(id(self))
            )
        obs = self._env.reset()
        obs = to_ndarray(obs).astype(np.float32)
        self._eval_episode_return = 0
        if self._save_replay_gif:
            self._frames = []
        # disc_to_cont: transform discrete action index to original continuous action
        self._raw_action_space = self._env.action_space
        self.m = self._raw_action_space.shape[0]
        self.n = self._cfg.each_dim_disc_size
        self.K = self.n ** self.m
        self.disc_to_cont = list(product(*[list(range(self.n)) for _ in range(self.m)]))
        # the modified discrete action space
        self._action_space = gym.spaces.Discrete(self.K)

        action_mask = np.ones(self.K, 'int8')
        obs = {'observation': obs, 'action_mask': action_mask, 'to_play': -1}
        return obs

    def close(self) -> None:
        if self._init_flag:
            self._env.close()
        self._init_flag = False

    def render(self) -> None:
        self._env.render()

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def step(self, action: np.ndarray) -> BaseEnvTimestep:
        """
        Overview:
             During the step phase, the environment first converts the discrete action into a continuous action,
             and then passes it into the original environment.
        Arguments:
            - action (:obj:`np.ndarray`): Discrete action
        Returns:
            - BaseEnvTimestep (:obj:`tuple`): Including observation, reward, done, and info.
        """
        # disc_to_cont: transform discrete action index to original continuous action
        action = [-1 + 2 / self.n * k for k in self.disc_to_cont[int(action)]]
        action = to_ndarray(action)
        if action.shape == (1, ):
            action = action.squeeze()
        if self._act_scale:
            action = affine_transform(action, min_val=self._raw_action_space.low, max_val=self._raw_action_space.high)
        if self._save_replay_gif:
            self._frames.append(self._env.render(mode='rgb_array'))
        obs, rew, done, info = self._env.step(action)

        action_mask = None
        obs = {'observation': obs, 'action_mask': action_mask, 'to_play': -1}
        self._eval_episode_return += rew
        if self._rew_clip:
            rew = max(-10, rew)
        rew = np.float32(rew)
        if done:
            info['eval_episode_return'] = self._eval_episode_return
            if self._save_replay_gif:
                if not os.path.exists(self._replay_path_gif):
                    os.makedirs(self._replay_path_gif)
                path = os.path.join(
                    self._replay_path_gif,
                    '{}_episode_{}_seed{}.gif'.format(self._env_id, self._save_replay_count, self._seed)
                )
                self.display_frames_as_gif(self._frames, path)
                print(f'save episode {self._save_replay_count} in {self._replay_path_gif}!')
                self._save_replay_count += 1
        obs = to_ndarray(obs)
        rew = to_ndarray([rew])
        return BaseEnvTimestep(obs, rew, done, info)

    @property
    def legal_actions(self):
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
        return "DI-engine BipedalWalker Env"

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

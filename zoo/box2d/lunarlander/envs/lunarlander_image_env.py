"""
Image-based LunarLander Environment for PriorZero VLM

Wraps the standard LunarLander-v2 to produce image observations (3, 64, 64)
instead of vector observations, enabling VLM-based prior generation.
"""
import copy
from typing import List, Dict

import cv2
import gymnasium as gym
import numpy as np
from ding.torch_utils import to_ndarray
from ding.utils import ENV_REGISTRY
from easydict import EasyDict

from zoo.box2d.lunarlander.envs.lunarlander_env import LunarLanderEnv


@ENV_REGISTRY.register('lunarlander_image')
class LunarLanderImageEnv(LunarLanderEnv):
    """
    Image-based LunarLander environment.

    Replaces the 8-dim vector observation with a (3, 64, 64) RGB image
    rendered from the environment. Everything else (actions, rewards, done)
    remains identical to the base LunarLanderEnv.
    """

    config = dict(
        env_id="LunarLander-v2",
        save_replay_gif=False,
        replay_path_gif=None,
        replay_path=None,
        act_scale=False,
        collect_max_episode_steps=int(1000),
        eval_max_episode_steps=int(1000),
        image_size=64,
    )

    @classmethod
    def default_config(cls) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
        self._image_size = cfg.get('image_size', 64)

    def _render_image_obs(self) -> np.ndarray:
        """Render the environment and return a (3, H, W) float32 image scaled to [0, 1]."""
        frame = self._env.render()  # (H, W, 3) RGB uint8
        # Resize to target size
        frame = cv2.resize(frame, (self._image_size, self._image_size), interpolation=cv2.INTER_AREA)
        # HWC -> CHW, scale to [0, 1] float32 (consistent with Atari env scale=True)
        frame = np.transpose(frame, (2, 0, 1)).astype(np.float32) / 255.0
        return frame

    def reset(self) -> Dict[str, np.ndarray]:
        if not self._init_flag:
            self._env = gym.make(self._cfg.env_id, render_mode="rgb_array")
            self._observation_space = gym.spaces.Box(
                low=0, high=1, shape=(3, self._image_size, self._image_size), dtype=np.float32
            )
            self._action_space = self._env.action_space
            self._reward_space = gym.spaces.Box(
                low=self._env.reward_range[0], high=self._env.reward_range[1], shape=(1,), dtype=np.float32
            )
            self._init_flag = True

        if hasattr(self, '_seed') and hasattr(self, '_dynamic_seed') and self._dynamic_seed:
            np_seed = 100 * np.random.randint(1, 1000)
            self._seed = self._seed + np_seed
            self._env.reset(seed=self._seed)
        elif hasattr(self, '_seed'):
            self._env.reset(seed=self._seed)
        else:
            self._env.reset()

        self._eval_episode_return = 0.0
        self._timestep = 0
        if self._save_replay_gif:
            self._frames = []

        # Render image observation
        obs_image = self._render_image_obs()
        action_mask = np.ones(4, 'int8')
        obs = {
            'observation': obs_image,
            'action_mask': action_mask,
            'to_play': -1,
            'timestep': self._timestep,
        }
        return obs

    def step(self, action: np.ndarray):
        from ding.envs import BaseEnvTimestep

        if isinstance(action, np.ndarray) and action.shape == (1,):
            action = action.item()
        elif not isinstance(action, np.ndarray):
            action = int(action)
        if self._save_replay_gif:
            self._frames.append(self._env.render())

        _, rew, terminated, truncated, info = self._env.step(action)
        done = terminated or truncated
        self._timestep += 1

        # Render image observation
        obs_image = self._render_image_obs()
        action_mask = np.ones(4, 'int8')
        obs = {
            'observation': obs_image,
            'action_mask': action_mask,
            'to_play': -1,
            'timestep': self._timestep,
        }

        self._eval_episode_return += rew
        if done:
            info['eval_episode_return'] = self._eval_episode_return
            info['score'] = self._eval_episode_return
            if self._save_replay_gif:
                import os
                from datetime import datetime
                if not os.path.exists(self._replay_path_gif):
                    os.makedirs(self._replay_path_gif)
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                path = os.path.join(
                    self._replay_path_gif,
                    f'{self._env_id}_episode_{self._save_replay_count}_seed{self._seed}_{timestamp}.gif'
                )
                self.display_frames_as_gif(self._frames, path)
                self._save_replay_count += 1

        obs = to_ndarray(obs)
        rew = to_ndarray(rew).astype(np.float32)
        return BaseEnvTimestep(obs, rew, done, info)

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

    def __repr__(self) -> str:
        return "LightZero LunarLander Image Env."

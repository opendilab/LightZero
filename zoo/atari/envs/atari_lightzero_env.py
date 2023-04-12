import copy
import sys
from typing import List

import gym
import numpy as np
from ding.envs import BaseEnv, BaseEnvTimestep
from ding.torch_utils import to_ndarray
from ding.utils import ENV_REGISTRY
from easydict import EasyDict

from zoo.atari.envs.atari_wrappers import wrap_lightzero


@ENV_REGISTRY.register('atari_lightzero')
class AtariLightZeroEnv(BaseEnv):
    config = dict(
        collector_env_num=8,
        evaluator_env_num=3,
        n_evaluator_episode=3,
        env_name='PongNoFrameskip-v4',
        env_type='Atari',
        obs_shape=(4, 96, 96),
        collect_max_episode_steps=int(1.08e5),
        eval_max_episode_steps=int(1.08e5),
        gray_scale=True,
        frame_skip=4,
        episode_life=True,
        clip_rewards=True,
        channel_last=True,
        render_mode_human=False,
        scale=True,
        warp_frame=True,
        save_video=False,
        transform2string=False,
        game_wrapper=True,
        manager=dict(shared_memory=False, ),
        stop_value=int(1e6),
    )

    @classmethod
    def default_config(cls: type) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    def __init__(self, cfg=None):
        self.cfg = cfg
        self._init_flag = False
        self.channel_last = cfg.channel_last
        self.clip_rewards = cfg.clip_rewards
        self.episode_life = cfg.episode_life

    def _make_env(self):
        return wrap_lightzero(self.cfg, episode_life=self.cfg.episode_life, clip_rewards=self.cfg.clip_rewards)

    def reset(self):
        if not self._init_flag:
            self._env = self._make_env()
            self._observation_space = self._env.env.observation_space
            self._action_space = self._env.env.action_space
            self._reward_space = gym.spaces.Box(
                low=self._env.env.reward_range[0], high=self._env.env.reward_range[1], shape=(1, ), dtype=np.float32
            )

            self._init_flag = True
        if hasattr(self, '_seed') and hasattr(self, '_dynamic_seed') and self._dynamic_seed:
            np_seed = 100 * np.random.randint(1, 1000)
            self._env.env.seed(self._seed + np_seed)
        elif hasattr(self, '_seed'):
            self._env.env.seed(self._seed)

        obs = self._env.reset()
        self.obs = to_ndarray(obs)
        self._eval_episode_return = 0.
        self.has_reset = True
        obs = self.observe()
        # obs.shape: 96,96,1
        return obs

    def observe(self):
        """
        Overview:
            add action_mask to obs to adapt with MCTS alg..
        """
        observation = self.obs

        if not self.channel_last:
            # move the channel dim to the fist axis
            # (96, 96, 3) -> (3, 96, 96)
            observation = np.transpose(observation, (2, 0, 1))

        action_mask = np.ones(self._action_space.n, 'int8')
        return {'observation': observation, 'action_mask': action_mask, 'to_play': -1}

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        # self._env.render()
        self.obs = to_ndarray(obs)
        self.reward = np.array(reward).astype(np.float32)
        self._eval_episode_return += self.reward
        observation = self.observe()
        if done:
            info['eval_episode_return'] = self._eval_episode_return

        return BaseEnvTimestep(observation, self.reward, done, info)

    @property
    def legal_actions(self):
        return np.arange(self._action_space.n)

    def random_action(self):
        action_list = self.legal_actions
        return np.random.choice(action_list)

    def render(self, mode='human'):
        self._env.render()

    def human_to_action(self):
        """
        Overview:
            For multiplayer games, ask the user for a legal action
            and return the corresponding action number.
        Returns:
            An integer from the action space.
        """
        while True:
            try:
                print(f"Current available actions for the player are:{self.legal_actions}")
                choice = int(input(f"Enter the index of next action: "))
                if choice in self.legal_actions:
                    break
                else:
                    print("Wrong input, try again")
            except KeyboardInterrupt:
                print("exit")
                sys.exit(0)
            except Exception as e:
                print("Wrong input, try again")
        return choice

    def close(self) -> None:
        if self._init_flag:
            self._env.close()
        self._init_flag = False

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

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
        return "LightZero Atari Env({})".format(self.cfg.env_name)

    @staticmethod
    def create_collector_env_cfg(cfg: dict) -> List[dict]:
        collector_env_num = cfg.pop('collector_env_num')
        cfg = copy.deepcopy(cfg)
        cfg.max_episode_steps = cfg.collect_max_episode_steps
        cfg.episode_life = True
        cfg.clip_rewards = True
        return [cfg for _ in range(collector_env_num)]

    @staticmethod
    def create_evaluator_env_cfg(cfg: dict) -> List[dict]:
        evaluator_env_num = cfg.pop('evaluator_env_num')
        cfg = copy.deepcopy(cfg)
        cfg.max_episode_steps = cfg.eval_max_episode_steps
        cfg.episode_life = False
        cfg.clip_rewards = False
        return [cfg for _ in range(evaluator_env_num)]

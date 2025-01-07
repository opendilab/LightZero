"""
env返回的obs不是id 是string
"""
import copy
from typing import List

import gym
import numpy as np
from transformers import AutoTokenizer
from ding.utils import ENV_REGISTRY
from ding.envs import BaseEnv, BaseEnvTimestep
from jericho import FrotzEnv


@ENV_REGISTRY.register('jericho')
class JerichoEnv(BaseEnv):
    """
    Overview:
        The environment for Jericho games. For more details about the game, please refer to the \
        `Jericho <https://github.com/microsoft/GameZero/tree/main/zoo/jericho>`.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.max_steps = cfg.max_steps
        self.game_path = cfg.game_path
        self.max_action_num = cfg.max_action_num
        self.max_seq_len = cfg.max_seq_len

        # 初始化分词器以供其他用途（如动作提示），不用于观察
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_path) 

        self._env = FrotzEnv(self.game_path)
        self._action_list = None
        self.finished = False
        self._init_flag = False
        self.episode_return = 0
        self.env_step = 0

        self.observation_space = gym.spaces.Dict({
            'observation': gym.spaces.Text(),
            'action_mask': gym.spaces.Box(low=0, high=1, shape=(self.max_action_num,), dtype=np.int8),
            'to_play': gym.spaces.Discrete(1)
        })
        self.action_space = gym.spaces.Discrete(self.max_action_num)
        self.reward_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)

    def prepare_obs(self, obs, return_str: bool = True):
        if self._action_list is None:
            self._action_list = self._env.get_valid_actions()
        full_obs = obs + "\nValid actions: " + str(self._action_list)

        # 始终返回字符串形式的观察
        if return_str:
            return {'observation': full_obs, 'action_mask': self._create_action_mask(), 'to_play': -1}
        else:
            raise ValueError("Current configuration only supports string observations.")

    def _create_action_mask(self):
        if len(self._action_list) <= self.max_action_num:
            action_mask = [1] * len(self._action_list) + [0] * (self.max_action_num - len(self._action_list))
        else:
            action_mask = [1] * self.max_action_num
        return np.array(action_mask, dtype=np.int8)

    def reset(self):
        initial_observation, info = self._env.reset()
        self.finished = False
        self._init_flag = True
        self._action_list = None
        self.episode_return = 0
        self.env_step = 0

        return self.prepare_obs(initial_observation)

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        """
        Set the seed for the environment.
        """
        self._seed = seed
        self._env.seed(seed)

    def close(self) -> None:
        self._init_flag = False

    def __repr__(self) -> str:
        return "LightZero Jericho Env"

    def step(self, action: int):
        try:
            action_str = self._action_list[action]
        except IndexError as e:
            # 处理非法动作
            print('='*20)
            print(e, 'Action is illegal. Randomly choosing a legal action!')
            action = np.random.choice(len(self._action_list))
            action_str = self._action_list[action]

        observation, reward, done, info = self._env.step(action_str)
        self.env_step += 1
        self.episode_return += reward
        self._action_list = None

        observation = self.prepare_obs(observation)

        if self.env_step >= self.max_steps:
            print('='*20)
            print('One episode done!')
            done = True

        if done:
            self.finished = True
            info['eval_episode_return'] = self.episode_return

        return BaseEnvTimestep(observation, reward, done, info)

    @staticmethod
    def create_collector_env_cfg(cfg: dict) -> List[dict]:
        collector_env_num = cfg.pop('collector_env_num')
        cfg = copy.deepcopy(cfg)
        cfg.is_collect = True
        return [cfg for _ in range(collector_env_num)]

    @staticmethod
    def create_evaluator_env_cfg(cfg: dict) -> List[dict]:
        evaluator_env_num = cfg.pop('evaluator_env_num')
        cfg = copy.deepcopy(cfg)
        cfg.reward_normalize = False
        cfg.is_collect = False
        return [cfg for _ in range(evaluator_env_num)]


if __name__ == '__main__':
    from easydict import EasyDict
    env_cfg = EasyDict(
        dict(
            max_steps=100,
            game_path="/mnt/afs/niuyazhe/code/LightZero/zoo/jericho/envs/z-machine-games-master/jericho-game-suite/detective.z5",
            max_action_num=50,
            max_seq_len=512,
            tokenizer_path="google-bert/bert-base-uncased",
        )
    )
    env = JerichoEnv(env_cfg)
    obs = env.reset()
    print(f'[OBS]:\n{obs["observation"]}')
    while True:
        try:
            action_id = int(input('Please input the action id:'))
            timestep = env.step(action_id)
            obs = timestep.obs
            reward = timestep.reward
            done = timestep.done
            info = timestep.info
            print(f'[OBS]:\n{obs["observation"]}')
            print(f'Reward: {reward}')
            if done:
                print('Episode finished.')
                break
        except Exception as e:
            print(f'Error: {e}. Please try again.')
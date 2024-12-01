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
    tokenizer = None

    def __init__(self, cfg):
        self.cfg = cfg
        self.game_path = cfg.game_path
        self.max_action_num = cfg.max_action_num
        self.max_seq_len = cfg.max_seq_len

        if JerichoEnv.tokenizer is None:
            JerichoEnv.tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_path)

        self._env = FrotzEnv(self.game_path)
        self._action_list = None

        self.finished = False
        self._init_flag = False
        self.episode_return = 0

        self.observation_space = gym.spaces.Dict()
        self.action_space = gym.spaces.Discrete(self.max_action_num)
        self.reward_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(1, ), dtype=np.float32)

    def prepare_obs(self, obs, return_str: bool = False):
        if self._action_list is None:
            self._action_list = self._env.get_valid_actions()
        full_obs = obs + "\nValid actions: " + str(self._action_list)
        if not return_str:
            full_obs = JerichoEnv.tokenizer(
                [full_obs], truncation=True, padding="max_length", max_length=self.max_seq_len)
            full_obs = np.array(full_obs['input_ids'][0], dtype=np.int32)
        action_mask = [1] * len(self._action_list) + [0] * \
            (self.max_action_num - len(self._action_list))
        action_mask = np.array(action_mask, dtype=np.int8)
        return {'observation': full_obs, 'action_mask': action_mask, 'to_play': -1}

    def reset(self, return_str: bool = False):
        initial_observation, info = self._env.reset()
        self.episode_return = 0
        return self.prepare_obs(initial_observation, return_str)

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

    def step(self, action: int, return_str: bool = False):
        action_str = self._action_list[action]
        observation, reward, done, info = self._env.step(action_str)
        self.episode_return += reward
        self._action_list = None
        observation = self.prepare_obs(observation, return_str)

        if done:
            self.finished = True
            info['eval_episode_return'] = self.episode_return

        return BaseEnvTimestep(observation, reward, done, info)

    @staticmethod
    def create_collector_env_cfg(cfg: dict) -> List[dict]:
        collector_env_num = cfg.pop('collector_env_num')
        cfg = copy.deepcopy(cfg)
        # when in collect phase, sometimes we need to normalize the reward
        # reward_normalize is determined by the config.
        cfg.is_collect = True
        return [cfg for _ in range(collector_env_num)]

    @staticmethod
    def create_evaluator_env_cfg(cfg: dict) -> List[dict]:
        evaluator_env_num = cfg.pop('evaluator_env_num')
        cfg = copy.deepcopy(cfg)
        # when in evaluate phase, we don't need to normalize the reward.
        cfg.reward_normalize = False
        cfg.is_collect = False
        return [cfg for _ in range(evaluator_env_num)]


if __name__ == '__main__':
    from easydict import EasyDict
    env_cfg = EasyDict(
        dict(
            game_path="z-machine-games-master/jericho-game-suite/zork1.z5",
            max_action_num=50,
            tokenizer_path="google-bert/bert-base-uncased",
            max_seq_len=512
        )
    )
    env = JerichoEnv(env_cfg)
    obs = env.reset(return_str=True)
    print(f'[OBS]:\n{obs["observation"]}')
    while True:
        action_id = int(input('Please input the action id:'))
        obs, reward, done, info = env.step(action_id, return_str=True)
        print(f'[OBS]:\n{obs["observation"]}')
        if done:
            break

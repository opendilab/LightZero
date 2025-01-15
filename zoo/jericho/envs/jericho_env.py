"""
env返回的obs是id 不是string
"""
import copy
from typing import List

import gym
import numpy as np
from transformers import AutoTokenizer
from ding.utils import ENV_REGISTRY
from ding.envs import BaseEnv, BaseEnvTimestep
from jericho import FrotzEnv
from ding.utils import set_pkg_seed, get_rank, get_world_size


@ENV_REGISTRY.register('jericho')
class JerichoEnv(BaseEnv):
    """
    Overview:
        The environment for Jericho games. For more details about the game, please refer to the \
        `Jericho <https://github.com/microsoft/GameZero/tree/main/zoo/jericho>`.
    """
    tokenizer = None

    def __init__(self, cfg):
        self.cfg = cfg
        self.max_steps = cfg.max_steps
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
        self.env_step = 0

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
            obs_attn_mask = full_obs['attention_mask']
            full_obs = np.array(full_obs['input_ids'][0], dtype=np.int32) # TODO: attn_mask
        if len(self._action_list) <= self.max_action_num:
            action_mask = [1] * len(self._action_list) + [0] * \
                (self.max_action_num - len(self._action_list))
        else:
            action_mask = [1] * len(self._action_list)

        action_mask = np.array(action_mask, dtype=np.int8)
        if return_str:
            return {'observation': full_obs, 'action_mask': action_mask, 'to_play': -1}
        else:
            return {'observation': full_obs, 'obs_attn_mask': obs_attn_mask, 'action_mask': action_mask, 'to_play': -1}

    def reset(self, return_str: bool = False):
        initial_observation, info = self._env.reset()
        self.finished = False
        self._init_flag = True
        self._action_list = None
        self.episode_return = 0
        self.env_step = 0

        # 获取当前的 world_size 和 rank
        self.world_size = get_world_size()
        self.rank = get_rank()

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
        try:
            action_str = self._action_list[action]
        except Exception as e:
            # TODO: why exits illegal action
            print('='*20)
            print(e, f'rank {self.rank}, action {action} is illegal now we randomly choose a legal action from {self._action_list}!')
            action = np.random.choice(len(self._action_list))
            action_str = self._action_list[action]

        observation, reward, done, info = self._env.step(action_str)
        self.env_step += 1
        self.episode_return += reward
        self._action_list = None
        observation = self.prepare_obs(observation, return_str)

        # print(f'rank {self.rank}, step: {self.env_step}')
        # print(f'self._action_list:{self._action_list}')
        # print(f'rank {self.rank}, step: {self.env_step}, observation:{observation}, action:{action}, reward:{reward}')

        if self.env_step >= self.max_steps:
            done = True

        if done:
            print('='*20)
            print(f'rank {self.rank} one episode done!')
            # print(f'self._action_list:{self._action_list}, action:{action}, reward:{reward}')
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
            max_steps=100,
            # game_path="z-machine-games-master/jericho-game-suite/zork1.z5",
            game_path="/mnt/afs/niuyazhe/code/LightZero/zoo/jericho/envs/z-machine-games-master/jericho-game-suite/detective.z5",
            # game_path="/mnt/afs/niuyazhe/code/LightZero/zoo/jericho/envs/z-machine-games-master/jericho-game-suite/905.z5",
            max_action_num=50,
            max_env_step=100,
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

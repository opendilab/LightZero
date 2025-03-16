import copy
from typing import List

import gym
import numpy as np
from transformers import AutoTokenizer
from ding.utils import ENV_REGISTRY
from ding.envs import BaseEnv, BaseEnvTimestep
from jericho import FrotzEnv
from ding.utils import set_pkg_seed, get_rank, get_world_size
import torch

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

        # 新增：记录上一次的观察和动作，以及阻止的动作列表
        self.last_observation = None
        self.last_action = None
        self.blocked_actions = set()

        # 获取当前的 world_size 和 rank
        self.world_size = get_world_size()
        self.rank = get_rank()

        # 新增：是否启用移除无效动作的功能
        self.remove_stuck_actions = cfg.get('remove_stuck_actions', False)
        self.add_location_and_inventory = cfg.get('add_location_and_inventory', False)

        if JerichoEnv.tokenizer is None:
            # 只让 rank 0 下载模型
            if self.rank == 0:
                JerichoEnv.tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_path) 
            if self.world_size > 1: 
                # 等待 rank 0 完成模型加载
                torch.distributed.barrier()
            if self.rank != 0:  # 非 rank 0 的进程从本地缓存加载
                JerichoEnv.tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_path) 

        self._env = FrotzEnv(self.game_path, 0)
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

        # full_obs = obs + "\nValid actions: " + str(self._action_list)

        # 根据是否启用移除无效动作的功能，调整可用动作列表
        if self.remove_stuck_actions:
            available_actions = [a for a in self._action_list if a not in self.blocked_actions]
            if len(available_actions) < 1 and len(self._action_list)>0:
                # TODO
                available_actions = [self._action_list[0]]
            self._action_list = available_actions
        else:
            available_actions = self._action_list
        
        # import ipdb;ipdb.set_trace()
        if self.add_location_and_inventory:
            look = self._env.get_player_location()
            inv = self._env.get_inventory()
            full_obs = "Location: " + str(look) + "\nInventory: " + str(inv) + obs + "\nValid actions: " + str(available_actions)
        else:
            full_obs = obs + "\nValid actions: " + str(available_actions)

        if not return_str:
            full_obs = JerichoEnv.tokenizer(
                [full_obs], truncation=True, padding="max_length", max_length=self.max_seq_len)
            obs_attn_mask = full_obs['attention_mask']
            full_obs = np.array(full_obs['input_ids'][0], dtype=np.int32)  # TODO: attn_mask


        if len(available_actions) == 0:
            # 避免action_maks全为0导致mcts报segment fault的错误
            action_mask = [1] + [0] * (self.max_action_num - 1)
        elif 0<len(available_actions) <= self.max_action_num:
            action_mask = [1] * len(available_actions) + [0] * (self.max_action_num - len(available_actions))
        elif len(available_actions) == self.max_action_num:
            action_mask = [1] * len(available_actions)
        else:
            action_mask = [1] * self.max_action_num

        # action_mask = [0] * self.max_action_num

        action_mask = np.array(action_mask, dtype=np.int8)

        # TODO: unizero需要加上'to_play', PPO不能加上'to_play'
        if return_str: 
            return {'observation': full_obs, 'action_mask': action_mask, 'to_play': -1}
            # return {'observation': full_obs, 'action_mask': action_mask}
        else:
            return {'observation': full_obs, 'obs_attn_mask': obs_attn_mask, 'action_mask': action_mask, 'to_play': -1}
            # return {'observation': full_obs, 'obs_attn_mask': obs_attn_mask, 'action_mask': action_mask}
             

    def reset(self, return_str: bool = False):
        initial_observation, info = self._env.reset()
        self.finished = False
        self._init_flag = True
        self._action_list = None
        self.episode_return = 0
        self.env_step = 0
        self.timestep = 0

        # 设置初始的 last_observation
        if self.remove_stuck_actions:
            self.last_observation = initial_observation
        else:
            self.last_observation = None

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
        # import ipdb;ipdb.set_trace()
        # print(action)
        self.blocked_actions = set()

        if isinstance(action, str):
            action_str = action
        else:
            if isinstance(action, np.ndarray):
                action = int(action)
            try:
                action_str = self._action_list[action]
            except Exception as e:
                # TODO: 为什么会有非法动作
                print('='*20)
                print(e, f'rank {self.rank}, action {action} is illegal now we randomly choose a legal action from {self._action_list}!')

                if len(self._action_list) > 0:
                    action = np.random.choice(len(self._action_list))
                    action_str = self._action_list[action]
                else:
                    action_str = 'go'
                    print(f"rank {self.rank}, len(self._action_list) == 0, self._env.get_valid_actions():{self._env.get_valid_actions()}, so we pass action_str='go'")

        # 记录上一次的观察
        if self.remove_stuck_actions and self.last_observation is not None:
            previous_obs = self.last_observation
        else:
            previous_obs = None

        # 执行动作
        observation, reward, done, info = self._env.step(action_str)

        self.timestep += 1
        # print(f'step: {self.timestep}, [OBS]:{observation} self._action_list:{self._action_list}')

        # TODO: only for PPO, 如果是unizero需要注释下面这行
        # reward = np.array([float(reward)])

        self.env_step += 1
        self.episode_return += reward
        self._action_list = None

        # 比较观察，判断动作是否有效
        if self.remove_stuck_actions and previous_obs is not None:
            if observation == previous_obs:
                # 动作无效，移除该动作
                self.blocked_actions.add(action_str)
                # print(f'[Removing action] "{action_str}" as it did not change the observation.')

        # 更新上一次的观察
        if self.remove_stuck_actions:
            self.last_observation = observation

        # 准备观察和动作掩码
        observation = self.prepare_obs(observation, return_str)

        # 检查是否超过最大步数
        if self.env_step >= self.max_steps:
            done = True

        if done:
            print('='*20)
            print(f'rank {self.rank} one episode done!')
            self.finished = True
            info['eval_episode_return'] = self.episode_return

        return BaseEnvTimestep(observation, reward, done, info)

    @staticmethod
    def create_collector_env_cfg(cfg: dict) -> List[dict]:
        collector_env_num = cfg.pop('collector_env_num')
        cfg = copy.deepcopy(cfg)
        # collect phase 可能需要归一化奖励
        cfg.is_collect = True
        return [cfg for _ in range(collector_env_num)]

    @staticmethod
    def create_evaluator_env_cfg(cfg: dict) -> List[dict]:
        evaluator_env_num = cfg.pop('evaluator_env_num')
        cfg = copy.deepcopy(cfg)
        # evaluate phase 不需要归一化奖励
        cfg.reward_normalize = False
        cfg.is_collect = False
        return [cfg for _ in range(evaluator_env_num)]


if __name__ == '__main__':
    from easydict import EasyDict
    env_cfg = EasyDict(
        dict(
            max_steps=400,
            game_path="../envs/z-machine-games-master/jericho-game-suite/"+ "zork1.z5",
            max_action_num=10,
            tokenizer_path="google-bert/bert-base-uncased",
            max_seq_len=512,
            remove_stuck_actions=False,
            add_location_and_inventory=False
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
            action_id = input('Would you like to RESTART, RESTORE a saved game, give the FULL score for that game or QUIT?')
            break
import logging
import copy
import os
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
from gym import spaces
import os
import sys
# import embodied
import pickle
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoTokenizer, AutoModel
import torch

from ding.utils import ENV_REGISTRY, get_rank, get_world_size
from ding.torch_utils import to_ndarray
from ding.envs import BaseEnv, BaseEnvTimestep
from easydict import EasyDict
from messenger.envs.stage_one import StageOne
from messenger.envs.stage_two import StageTwo
from messenger.envs.stage_three import StageThree
from messenger.envs.wrappers import TwoEnvWrapper
from messenger.envs.config import STATE_HEIGHT, STATE_WIDTH
# import from_gym


@ENV_REGISTRY.register('messenger')
class Messenger(BaseEnv):


    tokenizer: Optional[AutoTokenizer] = None
    manual_encoder: Optional[AutoModel] = None
    manual_embeds: Optional[torch.Tensor] = None

    config = dict(
        model_path="BAAI/bge-base-en-v1.5",
        # (int) The number of environment instances used for data collection.
        collector_env_num=1,
        # (int) The number of environment instances used for evaluator.
        evaluator_env_num=1,
        # (int) The number of episodes to evaluate during each evaluation period.
        n_evaluator_episode=1,
        # (str) The type of the environment, here it's Messenger.
        env_type='Messenger',
        n_entities=17,
        observation_shape=(17, STATE_HEIGHT, STATE_WIDTH),
        max_seq_len=256, # all manual sentence
        gray_scale=True,
        channel_last=False,
        max_steps=100,
        stop_value=int(1e6),
        max_action_num=5,
        mode="train",
        task="s1"
    )

    @classmethod
    def default_config(cls: type) -> EasyDict:
        """
        Overview:
            Return the default configuration for the Atari LightZero environment.
        Arguments:
            - cls (:obj:`type`): The class AtariEnvLightZero.
        Returns:
            - cfg (:obj:`EasyDict`): The default configuration dictionary.
        """
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    def __init__(self, cfg: Dict[str, Any]):
        """
        Overview:
            Initialize the Messenger environment.

        Arguments:
            - cfg (:obj:`Dict[str, Any]`): Configuration dictionary containing keys like max_steps, game_path, etc.
        """
        merged_cfg = self.default_config()
        merged_cfg.update(cfg)
        self.cfg = merged_cfg
        self._init_flag = False
        self.channel_last = self.cfg.channel_last
        self._timestep = 0

        self.max_steps = self.cfg.max_steps
        self.max_seq_len = self.cfg.max_seq_len
        self.n_entities = self.cfg.n_entities
        self.task = self.cfg.task
        self.mode = self.cfg.mode
        self.max_action_num = self.cfg.max_action_num

        self.manual = None
        self._init_flag = False
        self._eval_episode_return = 0.0

        # Get current world size and rank for distributed setups.
        self.world_size: int = get_world_size()
        self.rank: int = get_rank()

        if Messenger.tokenizer is None:
            if self.rank == 0:
                Messenger.tokenizer = AutoTokenizer.from_pretrained(self.cfg['model_path'])
                Messenger.manual_encoder = AutoModel.from_pretrained(self.cfg['model_path'])
            if self.world_size > 1:
                # Wait until rank 0 finishes loading the tokenizer
                torch.distributed.barrier()
            if self.rank != 0:
                Messenger.tokenizer = AutoTokenizer.from_pretrained(self.cfg['model_path'])
                Messenger.manual_encoder = AutoModel.from_pretrained(self.cfg['model_path'])
        
        print(f"Messenger config: {self.task} {self.mode} max_steps {self.max_steps}")
        assert self.task in ("s1", "s2", "s3")
        assert self.mode in ("train", "eval")
        
        if self.task == "s1":
            if self.mode == "train":
                self._env = TwoEnvWrapper(
                    stage=1,
                    split_1='train-mc',
                    split_2='train-sc',
                )
            else:
                self._env = StageOne(split="val")
        elif self.task == "s2":
            if self.mode == "train":
                    self._env = TwoEnvWrapper(
                    stage=2,
                    split_1='train-sc',
                    split_2='train-mc'
                )
            else:
                self._env = StageTwo(split='val')
        elif self.task == "s3":
            if self.mode == "train":
                    self._env = TwoEnvWrapper(
                    stage=3,
                    split_1='train-mc',
                    split_2='train-sc',
                )
            else:
                self._env = StageThree(split='val')

        observation_space = (
                self.cfg.observation_shape[0],
                self.cfg.observation_shape[1],
                self.cfg.observation_shape[2]
            )
        self._observation_space = spaces.Dict({
            'observation': spaces.Box(
                low=0, high=1, shape=observation_space, dtype=np.float32
            ),
            'action_mask': spaces.Box(
                low=0, high=1, shape=(self.cfg.max_action_num,), dtype=np.int8
            ),
            'to_play': spaces.Box(
                low=-1, high=2, shape=(), dtype=np.int8
            ),
            'timestep': spaces.Box(
                low=0, high=int(1.08e5), shape=(), dtype=np.int32
            ),
        })
        self._action_space = spaces.Discrete(int(self.max_action_num))
        self.reward_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)



    def __repr__(self) -> str:
        """
        Overview:
            Return a string representation of the environment.

        Returns:
            - (:obj:`str`): String representation of the environment.
        """
        return "LightZero Messenger Env"
    
    @property
    def observation_space(self) -> spaces.Space:
        """
        Property to access the observation space of the environment.
        """
        return self._observation_space

    @property
    def action_space(self) -> spaces.Space:
        """
        Property to access the action space of the environment.
        """
        return self._action_space

    def _symbolic_to_multihot(self, obs):
        # (h, w, 2)
        layers = np.concatenate((obs["entities"], obs["avatar"]),
                                axis=-1).astype(int)
        new_ob = np.maximum.reduce([np.eye(self.n_entities)[layers[..., i]] for i
                                    in range(layers.shape[-1])])
        new_ob[:, :, 0] = 0
        return new_ob.astype(np.float32)
    
    def observe(self) -> dict:
        """
        Overview:
            Return the current observation along with the action mask and to_play flag.
        Returns:
            - observation (:obj:`dict`): The dictionary containing current observation, action mask, and to_play flag.
        """
        observation = self.obs

        if not self.channel_last:
            # move the channel dim to the fist axis
            # (10, 10, 17) -> (17, 10, 10)
            observation = np.transpose(observation, (2, 0, 1))
        action_mask = np.ones(self.max_action_num, dtype=np.int8)
        # return {'observation': {'image': observation, 'manual_embeds': self.manual_embeds}, 'action_mask': action_mask, 'to_play': np.array(-1), 'timestep': np.array(self._timestep), }
        return {'observation': observation, 'action_mask': action_mask, 'to_play': np.array(-1), 'timestep': np.array(self._timestep), 'manual_embeds': self.manual_embeds}

    def reset(self):
        self._init_flag = True
        self._eval_episode_return = 0.0
        self._timestep = 0
        obs, self.manual = self._env.reset()
        maunal_sentence = ' '.join(self.manual)

        tokenized_output = self.tokenizer(
            [maunal_sentence], truncation=True, padding="max_length", max_length=self.max_seq_len, return_tensors='pt')
        # ts = {k: v.to(self.device) for k, v in ts.items()}
        with torch.no_grad():
            self.manual_embeds = self.manual_encoder(**tokenized_output).last_hidden_state[:,0,:].squeeze()

        obs["observation"] = self._symbolic_to_multihot(obs)
        del obs["entities"]
        del obs["avatar"]
        self.obs = to_ndarray(obs['observation'])
        obs = self.observe()
        return obs

    def step(self, action: int) -> BaseEnvTimestep:
        """
        Overview:
            Execute the given action and return the resulting environment timestep.
        Arguments:
            - action (:obj:`int`): The action to be executed.
        Returns:
            - timestep (:obj:`BaseEnvTimestep`): The environment timestep after executing the action.
        """
        obs, reward, done, info = self._env.step(action)
        new_obs = self._symbolic_to_multihot(obs)
        self.obs = to_ndarray(new_obs)
        
        self._timestep += 1 # don't increment step while reading
        self._eval_episode_return += reward        

        observation = self.observe()
        if info is None:
            info = {}
        if self._timestep >= self.max_steps:
            done = True
        
        if done:
            print('=' * 20)
            print(f'rank {self.rank} one episode done! episode_return:{self._eval_episode_return}')
            info['eval_episode_return'] = self._eval_episode_return

        return BaseEnvTimestep(observation, reward, done, info)

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        """
        Overview:
            Set the seed for the environment.

        Arguments:
            - seed (:obj:`int`): The seed value.
            - dynamic_seed (:obj:`bool`, optional): Whether to use a dynamic seed for randomness (defaults to True).
        """
        self._seed = seed

    def close(self) -> None:
        """
        Overview:
            Close the environment and release any resources.
        """
        self._init_flag = False

    @staticmethod
    def create_collector_env_cfg(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Overview:
            Create a list of environment configuration dictionaries for the collector phase.

        Arguments:
            - cfg (:obj:`Dict[str, Any]`): The original environment configuration.

        Returns:
            - (:obj:`List[Dict[str, Any]]`): A list of configuration dictionaries for collector environments.
        """
        collector_env_num: int = cfg.pop('collector_env_num')
        cfg = copy.deepcopy(cfg)
        return [cfg for _ in range(collector_env_num)]

    @staticmethod
    def create_evaluator_env_cfg(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Overview:
            Create a list of environment configuration dictionaries for the evaluator phase.

        Arguments:
            - cfg (:obj:`Dict[str, Any]`): The original environment configuration.

        Returns:
            - (:obj:`List[Dict[str, Any]]`): A list of configuration dictionaries for evaluator environments.
        """
        evaluator_env_num: int = cfg.pop('evaluator_env_num')
        cfg = copy.deepcopy(cfg)
        return [cfg for _ in range(evaluator_env_num)]


if __name__ == "__main__":
    from easydict import EasyDict
    env_type='detective' # zork1, acorncourt, detective, omniquest
    # Configuration dictionary for the environment.
    env_cfg = EasyDict(
        dict(
            max_steps=400,
            max_action_num=5,
            max_seq_len=512,
            collector_env_num=1,
            evaluator_env_num=1,
            mode="train",
            task="s1",
            vis=False
        )
    )
    env = Messenger(env_cfg)
    obs = env.reset()

    while True:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(f"Step: {env._timestep}, Action: {action}, Reward: {reward}")
        if done:
            print(f"Episode done with return: {info['eval_episode_return']}")
            break
    del env  
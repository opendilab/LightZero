import copy
import json
import logging
import random as _random
import re
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Union

import gym
import numpy as np
import torch
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from transformers import AutoTokenizer

from ding.utils import ENV_REGISTRY, get_rank, get_world_size
from ding.envs import BaseEnv, BaseEnvTimestep


ATOMIC_ACTIONS = [
    "turn left", "turn right", "move forward",
    "pickup", "drop", "toggle", "check available actions",
]


class BabyAIHttpClient:
    """HTTP client for AgentGym BabyAI server with retry and timeout."""

    def __init__(self, env_addr: str, timeout: float = 10.0, max_retries: int = 3):
        self._addr = env_addr.rstrip('/')
        self._timeout = timeout
        self._session = requests.Session()
        retries = Retry(
            total=max_retries,
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504],
        )
        self._session.mount('http://', HTTPAdapter(max_retries=retries))
        self._session.mount('https://', HTTPAdapter(max_retries=retries))

    def health_check(self) -> bool:
        try:
            r = self._session.get(f"{self._addr}/", timeout=self._timeout)
            return r.status_code == 200
        except Exception:
            return False

    def create(self) -> int:
        r = self._session.post(f"{self._addr}/create", timeout=self._timeout)
        r.raise_for_status()
        data = r.json()
        if "error" in data:
            raise RuntimeError(f"BabyAI create error: {data['error']}")
        return data["id"]

    def reset(self, env_id: int, data_idx: int) -> dict:
        r = self._session.post(
            f"{self._addr}/reset",
            json={"id": env_id, "data_idx": data_idx},
            timeout=self._timeout,
        )
        r.raise_for_status()
        data = r.json()
        if "error" in data:
            raise RuntimeError(f"BabyAI reset error: {data['error']}")
        return data

    def step(self, env_id: int, action: str) -> dict:
        r = self._session.post(
            f"{self._addr}/step",
            json={"id": env_id, "action": action},
            timeout=self._timeout,
        )
        r.raise_for_status()
        data = r.json()
        if "error" in data:
            raise RuntimeError(f"BabyAI step error: {data['error']}")
        return data

    def close(self, env_id: int):
        try:
            self._session.post(
                f"{self._addr}/close",
                json={"id": env_id},
                timeout=self._timeout,
            )
        except Exception:
            pass

    def close_session(self):
        self._session.close()


def _parse_mission(obs_text: str) -> str:
    """Extract mission from observation text. Format: 'Your goal: <mission>\n...'"""
    if obs_text.startswith("Your goal: "):
        first_line_end = obs_text.find('\n')
        if first_line_end == -1:
            return obs_text[len("Your goal: "):]
        return obs_text[len("Your goal: "):first_line_end].strip()
    return ""


def _parse_available_actions(obs_text: str) -> List[str]:
    """Extract available actions list from observation text.
    Format: '...\\nAvailable actions: ["action1", "action2", ...]'
    """
    marker = "\nAvailable actions: ["
    idx = obs_text.rfind(marker)
    if idx == -1:
        return []
    actions_str = obs_text[idx + len(marker) - 1:]  # include the '['
    try:
        actions = json.loads(actions_str)
        if isinstance(actions, list):
            return [str(a) for a in actions]
    except json.JSONDecodeError:
        pass
    # Fallback: regex extraction
    matches = re.findall(r'"([^"]*)"', actions_str)
    return matches if matches else []


def _strip_actions_suffix(obs_text: str) -> str:
    """Remove the 'Available actions: [...]' suffix from observation text."""
    marker = "\nAvailable actions: ["
    idx = obs_text.rfind(marker)
    if idx != -1:
        return obs_text[:idx].strip()
    return obs_text


@ENV_REGISTRY.register('babyai')
class BabyAIEnv(BaseEnv):
    """
    BabyAI environment wrapper for PriorZero.
    Communicates with AgentGym BabyAI HTTP server.
    Interface contract matches JerichoEnv for algorithm-layer compatibility.
    """
    tokenizer: Optional[AutoTokenizer] = None

    # aligned with ScalingInter-RL: class-level counter for evaluator task cycling
    _eval_cycle_counter = 0
    _eval_cycle_lock = threading.Lock()

    DEFAULT_CONFIG: Dict[str, Any] = {
        'env_addr': 'http://127.0.0.1:8000',
        'data_idx': 0,
        'data_idx_list': None,
        'train_data_idx_list': None,
        'eval_data_idx_list': None,
        'max_steps': 64,
        'max_action_num': 20,
        'tokenizer_path': 'BAAI/bge-base-en-v1.5',
        'max_seq_len': 512,
        'for_unizero': True,
        'save_replay': False,
        'use_high_level_actions': True,
        'is_collect': True,
        'collector_env_num': 1,
        'evaluator_env_num': 1,
    }

    def __init__(self, cfg: Dict[str, Any]) -> None:
        merged_cfg = copy.deepcopy(self.DEFAULT_CONFIG)
        merged_cfg.update(cfg)
        self.cfg = merged_cfg

        self.env_addr: str = self.cfg['env_addr']
        self.data_idx: int = self.cfg.get('data_idx', 0)
        self.data_idx_list: Optional[List[int]] = self.cfg.get('data_idx_list', None)
        self._is_collect: bool = self.cfg.get('is_collect', True)
        self.max_steps: int = self.cfg['max_steps']
        self.max_action_num: int = self.cfg['max_action_num']
        self.max_seq_len: int = self.cfg['max_seq_len']
        self.for_unizero: bool = self.cfg['for_unizero']
        self.save_replay: bool = self.cfg['save_replay']
        self.use_high_level_actions: bool = self.cfg['use_high_level_actions']

        self.world_size: int = get_world_size()
        self.rank: int = get_rank()

        if BabyAIEnv.tokenizer is None:
            if self.rank == 0:
                BabyAIEnv.tokenizer = AutoTokenizer.from_pretrained(self.cfg['tokenizer_path'])
            if self.world_size > 1:
                torch.distributed.barrier()
            if self.rank != 0:
                BabyAIEnv.tokenizer = AutoTokenizer.from_pretrained(self.cfg['tokenizer_path'])

        self._client = BabyAIHttpClient(self.env_addr)
        try:
            self._env_id: int = self._client.create()
        except Exception as e:
            logging.error(f"[BabyAIEnv] Failed to create env on server: {e}")
            self._env_id = -1

        self._action_list: Optional[List[str]] = None
        self._mission: str = ""
        self._server_halted: bool = False
        self.finished: bool = False
        self._init_flag: bool = False
        self.episode_return: float = 0.0
        self._last_reward: float = 0.0
        self._timestep: int = 0

        self.observation_space = gym.spaces.Dict()
        self.action_space = gym.spaces.Discrete(self.max_action_num)
        self.reward_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)

    def prepare_obs(self, obs: str, return_str: bool = False) -> Dict[str, Any]:
        raw_obs_text = obs
        available_actions = self._action_list if self._action_list else []

        full_obs = f"{obs}\nValid actions: {available_actions}"
        full_obs_str = copy.deepcopy(full_obs)

        if not return_str:
            tokenized = BabyAIEnv.tokenizer(
                [full_obs], truncation=True, padding="max_length", max_length=self.max_seq_len
            )
            obs_attn_mask = tokenized['attention_mask']
            full_obs = np.array(tokenized['input_ids'][0], dtype=np.int32)

        if len(available_actions) == 0:
            action_mask = [1] + [0] * (self.max_action_num - 1)
        elif len(available_actions) <= self.max_action_num:
            action_mask = [1] * len(available_actions) + [0] * (self.max_action_num - len(available_actions))
        else:
            action_mask = [1] * self.max_action_num
        action_mask = np.array(action_mask, dtype=np.int8)

        if return_str:
            result = {
                'observation': full_obs,
                'action_mask': action_mask,
                'valid_actions': available_actions,
                'raw_obs_text': raw_obs_text,
            }
            if self.for_unizero:
                result['to_play'] = -1
                result['timestep'] = self._timestep
            return result
        else:
            result = {
                'observation': full_obs,
                'obs_attn_mask': obs_attn_mask,
                'action_mask': action_mask,
                'valid_actions': available_actions,
                'raw_obs_text': raw_obs_text,
            }
            if self.for_unizero:
                result['to_play'] = -1
                result['timestep'] = self._timestep
            return result

    def reset(self, return_str: bool = False) -> Dict[str, Any]:
        # aligned with ScalingInter-RL: multi-task cycling
        if self.data_idx_list is not None:
            if self._is_collect:
                self.data_idx = _random.choice(self.data_idx_list)
            else:
                with BabyAIEnv._eval_cycle_lock:
                    self.data_idx = self.data_idx_list[
                        BabyAIEnv._eval_cycle_counter % len(self.data_idx_list)
                    ]
                    BabyAIEnv._eval_cycle_counter += 1

        if self._server_halted:
            try:
                self._env_id = self._client.create()
                self._server_halted = False
            except Exception:
                pass

        try:
            resp = self._client.reset(self._env_id, self.data_idx)
        except Exception as e:
            logging.warning(f"[BabyAIEnv] reset failed: {e}")
            self._server_halted = True
            self._action_list = []
            self._mission = ""
            self.finished = False
            self._init_flag = True
            self.episode_return = 0.0
            self._last_reward = 0.0
            self._timestep = 0
            return self.prepare_obs("[Server unreachable]", return_str)

        obs_text = resp.get('observation', '')
        self._mission = _parse_mission(obs_text)
        self._action_list = _parse_available_actions(obs_text)
        if not self.use_high_level_actions:
            self._action_list = list(ATOMIC_ACTIONS)
        raw_obs = _strip_actions_suffix(obs_text)

        self.finished = False
        self._init_flag = True
        self._server_halted = False
        self.episode_return = 0.0
        self._last_reward = 0.0
        self._timestep = 0

        return self.prepare_obs(raw_obs, return_str)

    def step(self, action: Union[int, np.ndarray, str], return_str: bool = False) -> BaseEnvTimestep:
        if self._server_halted:
            dummy_obs = self.prepare_obs("[Server halted]", return_str)
            info = {'action_str': 'noop', 'abnormal': True, 'eval_episode_return': self.episode_return}
            return BaseEnvTimestep(dummy_obs, 0.0, True, info)

        if isinstance(action, str):
            action_str = action
        else:
            if isinstance(action, np.ndarray):
                action = int(action)
            try:
                action_str = self._action_list[action]
            except (IndexError, TypeError):
                if self._action_list and len(self._action_list) > 0:
                    action = int(np.random.choice(len(self._action_list)))
                    action_str = self._action_list[action]
                else:
                    action_str = "check available actions"

        try:
            resp = self._client.step(self._env_id, action_str)
        except Exception as e:
            logging.warning(f"[BabyAIEnv] step failed on '{action_str}': {e}")
            self._server_halted = True
            dummy_obs = self.prepare_obs("[Server halted]", return_str)
            info = {'action_str': action_str, 'abnormal': True, 'eval_episode_return': self.episode_return, 'score': self.episode_return}
            return BaseEnvTimestep(dummy_obs, 0.0, True, info)

        obs_text = resp.get('observation', '')
        reward_from_server = float(resp.get('reward', 0.0))
        done = bool(resp.get('done', False))

        step_reward = reward_from_server - self._last_reward
        self._last_reward = reward_from_server
        self.episode_return = reward_from_server

        self._timestep += 1
        self._action_list = _parse_available_actions(obs_text)
        if not self.use_high_level_actions:
            self._action_list = list(ATOMIC_ACTIONS)
        raw_obs = _strip_actions_suffix(obs_text)

        if self._timestep >= self.max_steps:
            done = True

        processed_obs = self.prepare_obs(raw_obs, return_str)
        # aligned with ScalingInter-RL: include task identity for per-level eval logging
        info = {
            'action_str': action_str,
            'score': self.episode_return,
            'data_idx': self.data_idx,
            'level_id': self.data_idx % 40 + 1,
        }

        if done:
            self.finished = True
            info['eval_episode_return'] = self.episode_return

        return BaseEnvTimestep(processed_obs, step_reward, done, info)

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed

    def close(self) -> None:
        self._init_flag = False
        if hasattr(self, '_client') and self._client is not None:
            self._client.close(self._env_id)

    def __repr__(self) -> str:
        return "LightZero BabyAI Env"

    @staticmethod
    def create_collector_env_cfg(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
        collector_env_num = cfg.pop('collector_env_num')
        cfg = copy.deepcopy(cfg)
        cfg['is_collect'] = True
        # aligned with ScalingInter-RL: use train task list for collector
        if 'train_data_idx_list' in cfg and cfg['train_data_idx_list'] is not None:
            cfg['data_idx_list'] = cfg['train_data_idx_list']
        return [cfg for _ in range(collector_env_num)]

    @staticmethod
    def create_evaluator_env_cfg(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
        evaluator_env_num = cfg.pop('evaluator_env_num')
        cfg = copy.deepcopy(cfg)
        cfg['is_collect'] = False
        # aligned with ScalingInter-RL: use eval task list for evaluator
        if 'eval_data_idx_list' in cfg and cfg['eval_data_idx_list'] is not None:
            cfg['data_idx_list'] = cfg['eval_data_idx_list']
        return [cfg for _ in range(evaluator_env_num)]

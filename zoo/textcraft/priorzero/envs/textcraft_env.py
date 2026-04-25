import copy
import json
import logging
import re
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


class TextCraftHttpClient:
    """HTTP client for AgentGym TextCraft server with retry and timeout."""

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

    def create(self, commands: str = None, goal: str = None) -> int:
        payload = {}
        if commands is not None:
            payload["commands"] = commands
        if goal is not None:
            payload["goal"] = goal
        r = self._session.post(f"{self._addr}/create", json=payload, timeout=self._timeout)
        r.raise_for_status()
        data = r.json()
        if "error" in data:
            raise RuntimeError(f"TextCraft create error: {data['error']}")
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
            raise RuntimeError(f"TextCraft reset error: {data['error']}")
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
            raise RuntimeError(f"TextCraft step error: {data['error']}")
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


def _parse_goal(obs_text: str) -> str:
    """Extract goal from observation. Format: '...Goal: craft <item>.'"""
    match = re.search(r'Goal:\s*craft\s+(.+?)\.?\s*$', obs_text, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return ""


def _extract_candidate_actions(obs_text: str) -> List[str]:
    """
    Parse crafting recipes from observation text and generate candidate actions.

    Returns a list of executable commands:
      - 'craft <count> <item> using <ingredients>' for each recipe
      - 'get <count> <ingredient>' for each base (non-craftable) ingredient
      - 'inventory' always included
    """
    candidates: List[str] = []
    craftable_items: set = set()

    recipe_section = re.search(
        r'Crafting commands?:\s*\n(.*?)(?:\n\s*\n|\Z)',
        obs_text,
        re.DOTALL | re.IGNORECASE,
    )
    if recipe_section:
        recipe_block = recipe_section.group(1)
        for line in recipe_block.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            cmd_match = re.match(
                r'(craft\s+\d+\s+.+?\s+using\s+.+)', line, re.IGNORECASE,
            )
            if cmd_match:
                craft_cmd = cmd_match.group(1).strip()
                candidates.append(craft_cmd)
                output_match = re.match(
                    r'craft\s+\d+\s+(.+?)\s+using\s+', craft_cmd, re.IGNORECASE,
                )
                if output_match:
                    craftable_items.add(output_match.group(1).strip().lower())

    base_ingredients: OrderedDict = OrderedDict()
    for cmd in candidates:
        using_match = re.search(r'using\s+(.+)$', cmd, re.IGNORECASE)
        if using_match:
            parts = using_match.group(1).split(',')
            for part in parts:
                part = part.strip()
                ing_match = re.match(r'(\d+)\s+(.+)', part)
                if ing_match:
                    count = ing_match.group(1)
                    item = ing_match.group(2).strip()
                    if item.lower() not in craftable_items:
                        key = item.lower()
                        if key not in base_ingredients:
                            base_ingredients[key] = (count, item)

    for count, item in base_ingredients.values():
        candidates.append(f"get {count} {item}")

    candidates.append("inventory")
    return candidates


@ENV_REGISTRY.register('textcraft')
class TextCraftEnv(BaseEnv):
    """
    TextCraft environment wrapper for PriorZero.
    Communicates with AgentGym TextCraft HTTP server.
    Interface contract matches BabyAIEnv/JerichoEnv for algorithm-layer compatibility.
    """
    tokenizer: Optional[AutoTokenizer] = None

    DEFAULT_CONFIG: Dict[str, Any] = {
        'env_addr': 'http://127.0.0.1:36005',
        'data_idx': 0,
        'max_steps': 30,
        'max_action_num': 20,
        'tokenizer_path': 'BAAI/bge-base-en-v1.5',
        'max_seq_len': 512,
        'for_unizero': True,
        'save_replay': False,
        'collector_env_num': 1,
        'evaluator_env_num': 1,
    }

    def __init__(self, cfg: Dict[str, Any]) -> None:
        merged_cfg = copy.deepcopy(self.DEFAULT_CONFIG)
        merged_cfg.update(cfg)
        self.cfg = merged_cfg

        self.env_addr: str = self.cfg['env_addr']
        self.data_idx: int = self.cfg['data_idx']
        self.max_steps: int = self.cfg['max_steps']
        self.max_action_num: int = self.cfg['max_action_num']
        self.max_seq_len: int = self.cfg['max_seq_len']
        self.for_unizero: bool = self.cfg['for_unizero']
        self.save_replay: bool = self.cfg['save_replay']

        self.world_size: int = get_world_size()
        self.rank: int = get_rank()

        if TextCraftEnv.tokenizer is None:
            if self.rank == 0:
                TextCraftEnv.tokenizer = AutoTokenizer.from_pretrained(self.cfg['tokenizer_path'])
            if self.world_size > 1:
                torch.distributed.barrier()
            if self.rank != 0:
                TextCraftEnv.tokenizer = AutoTokenizer.from_pretrained(self.cfg['tokenizer_path'])

        self._client = TextCraftHttpClient(self.env_addr)
        try:
            self._env_id: int = self._client.create()
        except Exception as e:
            logging.error(f"[TextCraftEnv] Failed to create env on server: {e}")
            self._env_id = -1

        self._goal: str = ""
        self._action_list: List[str] = ["inventory"]
        self._server_halted: bool = False
        self.finished: bool = False
        self._init_flag: bool = False
        self.episode_return: float = 0.0
        self._timestep: int = 0

        self.observation_space = gym.spaces.Dict()
        self.action_space = gym.spaces.Discrete(self.max_action_num)
        self.reward_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)

    def prepare_obs(self, obs: str, return_str: bool = False) -> Dict[str, Any]:
        raw_obs_text = obs
        full_obs = obs
        full_obs_str = copy.deepcopy(full_obs)

        if not return_str:
            tokenized = TextCraftEnv.tokenizer(
                [full_obs], truncation=True, padding="max_length", max_length=self.max_seq_len
            )
            obs_attn_mask = tokenized['attention_mask']
            full_obs = np.array(tokenized['input_ids'][0], dtype=np.int32)

        action_mask = np.ones(self.max_action_num, dtype=np.int8)

        if return_str:
            result = {
                'observation': full_obs,
                'action_mask': action_mask,
                'valid_actions': list(self._action_list),
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
                'valid_actions': list(self._action_list),
                'raw_obs_text': raw_obs_text,
            }
            if self.for_unizero:
                result['to_play'] = -1
                result['timestep'] = self._timestep
            return result

    def reset(self, return_str: bool = False) -> Dict[str, Any]:
        if self._server_halted:
            try:
                self._env_id = self._client.create()
                self._server_halted = False
            except Exception:
                pass

        try:
            resp = self._client.reset(self._env_id, self.data_idx)
        except Exception as e:
            logging.warning(f"[TextCraftEnv] reset failed: {e}")
            self._server_halted = True
            self._goal = ""
            self.finished = False
            self._init_flag = True
            self.episode_return = 0.0
            self._timestep = 0
            return self.prepare_obs("[Server unreachable]", return_str)

        obs_text = resp.get('observation', '')
        self._goal = _parse_goal(obs_text)
        self._action_list = _extract_candidate_actions(obs_text)

        self.finished = False
        self._init_flag = True
        self._server_halted = False
        self.episode_return = 0.0
        self._timestep = 0

        return self.prepare_obs(obs_text, return_str)

    def step(self, action: Union[int, np.ndarray, str], return_str: bool = False) -> BaseEnvTimestep:
        if self._server_halted:
            dummy_obs = self.prepare_obs("[Server halted]", return_str)
            info = {'action_str': 'noop', 'abnormal': True, 'eval_episode_return': self.episode_return}
            return BaseEnvTimestep(dummy_obs, 0.0, True, info)

        if isinstance(action, str):
            action_str = action
        elif isinstance(action, (int, np.integer, np.ndarray)):
            action_idx = int(action.item() if isinstance(action, np.ndarray) else action)
            if 0 <= action_idx < len(self._action_list):
                action_str = self._action_list[action_idx]
            else:
                action_str = "inventory"
        else:
            action_str = "inventory"

        try:
            resp = self._client.step(self._env_id, action_str)
        except Exception as e:
            logging.warning(f"[TextCraftEnv] step failed on '{action_str}': {e}")
            self._server_halted = True
            dummy_obs = self.prepare_obs("[Server halted]", return_str)
            info = {'action_str': action_str, 'abnormal': True, 'eval_episode_return': self.episode_return, 'score': self.episode_return}
            return BaseEnvTimestep(dummy_obs, 0.0, True, info)

        obs_text = resp.get('observation', '')
        reward_from_server = float(resp.get('reward', 0.0))
        done = bool(resp.get('done', False))

        self._action_list = _extract_candidate_actions(obs_text)

        step_reward = reward_from_server
        self.episode_return = reward_from_server

        self._timestep += 1

        if self._timestep >= self.max_steps:
            done = True

        processed_obs = self.prepare_obs(obs_text, return_str)
        info = {'action_str': action_str, 'score': self.episode_return}

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
        return "LightZero TextCraft Env"

    @staticmethod
    def create_collector_env_cfg(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
        collector_env_num = cfg.pop('collector_env_num')
        cfg = copy.deepcopy(cfg)
        cfg['is_collect'] = True
        return [cfg for _ in range(collector_env_num)]

    @staticmethod
    def create_evaluator_env_cfg(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
        evaluator_env_num = cfg.pop('evaluator_env_num')
        cfg = copy.deepcopy(cfg)
        cfg['is_collect'] = False
        return [cfg for _ in range(evaluator_env_num)]

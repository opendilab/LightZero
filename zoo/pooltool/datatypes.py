"""This would be a great place to create PoolToolEnv and observation/action vector building"""
from __future__ import annotations

import copy
import random
from functools import cached_property
from typing import Any, Dict, List, Optional

import attrs
import numpy as np
from ding.envs import BaseEnv
from easydict import EasyDict
from gym import spaces
from numpy.typing import NDArray

from pooltool.ai.datatypes import State as PoolToolState

class BasePoolToolEnv(BaseEnv):
    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def manage_seeds(self) -> None:
        if (
            hasattr(self, "_seed")
            and hasattr(self, "_dynamic_seed")
            and self._dynamic_seed
        ):
            np_seed = 100 * np.random.randint(1, 1000)
            self._env.seed(self._seed + np_seed)
        elif hasattr(self, "_seed"):
            self._env.seed(self._seed)

    @property
    def observation_space(self) -> spaces.Space:
        return self._observation_space

    @property
    def action_space(self) -> spaces.Space:
        return self._action_space

    @property
    def reward_space(self) -> spaces.Space:
        return self._reward_space

    @classmethod
    def default_config(cls) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + "Dict"
        return cfg

    @staticmethod
    def create_collector_env_cfg(cfg: dict) -> List[dict]:
        collector_env_num = cfg.pop("collector_env_num")
        cfg = copy.deepcopy(cfg)
        return [cfg for _ in range(collector_env_num)]

    @staticmethod
    def create_evaluator_env_cfg(cfg: dict) -> List[dict]:
        evaluator_env_num = cfg.pop("evaluator_env_num")
        cfg = copy.deepcopy(cfg)
        return [cfg for _ in range(evaluator_env_num)]

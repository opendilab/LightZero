from __future__ import annotations

import importlib
import sys
from functools import partial
from pathlib import Path
from typing import Any, Tuple

import attrs
import numpy as np
import torch
from ding.config import compile_config
from ding.envs import create_env_manager, get_vec_env_setting, to_ndarray
from ding.policy import create_policy
from ding.utils import set_pkg_seed
from lzero.mcts.utils import prepare_observation
from lzero.worker import MuZeroEvaluator
from zoo.pooltool.datatypes import ObservationDict, SimulationEnv

import pooltool as pt


def _config_from_model_path(model_path: Path) -> Tuple[dict, dict]:
    config_path = model_path.parent.parent / "formatted_total_config.py"
    assert config_path.exists()

    sys.path.append(str(config_path.parent))
    config_module = importlib.import_module("formatted_total_config")
    return config_module.main_config, config_module.create_config


@attrs.define
class ActionInference:
    evaluator: MuZeroEvaluator

    def forward_ready_observation(
        self, observation: ObservationDict
    ) -> Tuple[Any, Any, Any]:
        obs = [observation["observation"]]
        obs = to_ndarray(obs)
        obs = prepare_observation(obs, self.evaluator.policy_config.model.model_type)
        obs = torch.from_numpy(obs).to(self.evaluator.policy_config.device).float()

        action_mask = [observation["action_mask"]]
        to_play = [np.array(observation["to_play"], dtype=np.int64)]

        return obs, action_mask, to_play

    def infer(self, observation: ObservationDict):
        obs = self.forward_ready_observation(observation)
        policy_output = self.evaluator._policy.forward(*obs)
        return policy_output[0]["action"]

    @classmethod
    def from_model_path(cls, model_path: Path) -> ActionInference:
        cfg, create_cfg = _config_from_model_path(model_path)

        # Otherwise the environment isn't registered
        create_cfg.policy.import_names = cfg.policy.import_names

        # We just need a single evaluator
        cfg.env.evaluator_env_num = 1
        cfg.env.n_evaluator_episode = 1

        # If cuda is available and was used for training, use it
        cfg.policy.device = (
            "cuda" if cfg.policy.cuda and torch.cuda.is_available() else "cpu"
        )

        cfg = compile_config(
            cfg, env=None, auto=True, create_cfg=create_cfg, save_cfg=False
        )

        env_fn, _, evaluator_env_cfg = get_vec_env_setting(cfg.env)
        evaluator_env = create_env_manager(
            cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg]
        )

        evaluator_env.seed(cfg.seed, dynamic_seed=False)
        set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)

        policy = create_policy(cfg.policy, enable_field=["learn", "eval"])
        policy.eval_mode.load_state_dict(
            torch.load(model_path, map_location=cfg.policy.device)
        )

        policy_config = cfg.policy

        return cls(
            MuZeroEvaluator(
                eval_freq=cfg.policy.eval_freq,
                n_evaluator_episode=cfg.env.n_evaluator_episode,
                stop_value=cfg.env.stop_value,
                env=evaluator_env,
                policy=policy.eval_mode,
                exp_name=cfg.exp_name,
                policy_config=policy_config,
            )
        )


if __name__ == "__main__":
    model_path = Path("./ckpt/ckpt_best.pth.tar")
    inference = ActionInference.from_model_path(model_path)
    gui = pt.ShotViewer()

    while True:
        env = SimulationEnv.sum_to_three(random_pos=True)
        action = inference.infer(env.observation())
        env.set_action(env.scale_action(action))
        env.simulate()
        gui.show(env.system)

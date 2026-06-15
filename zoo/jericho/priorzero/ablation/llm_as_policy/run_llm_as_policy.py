#!/usr/bin/env python3
"""LLM-as-policy ablation for PriorZero Jericho experiments.

This baseline keeps the PriorZero prompt/action-prior setup, but removes the
world model, MCTS, replay buffer, and all training. At each environment step it
scores the current valid actions with the frozen LLM and executes the best one.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
import contextlib
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import vllm


REPO_ROOT = Path(__file__).resolve().parents[2]
LIGHTZERO_ROOT = REPO_ROOT.parents[2]
for path in (REPO_ROOT / "src", LIGHTZERO_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from priorzero_config import get_model_config, get_priorzero_config  # noqa: E402
from priorzero_datafactory import DataProcessor  # noqa: E402
from zoo.jericho.envs.jericho_env import JerichoEnv  # noqa: E402


class LocalVLLMActor:
    """Minimal adapter matching the DataProcessor vLLM interface."""

    def __init__(self, model_path: str, tensor_parallel_size: int, max_model_len: int, gpu_memory_utilization: float):
        self.requests = []
        self.sampling_params = None
        self.llm = vllm.LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            dtype="bfloat16",
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
        )

    def add_requests(self, sampling_params, prompt_token_ids):
        from vllm.inputs import TokensPrompt

        self.sampling_params = sampling_params
        self.requests = [TokensPrompt(prompt_token_ids=r) for r in prompt_token_ids]

    def get_responses(self):
        outputs = self.llm.generate(
            prompts=self.requests,
            sampling_params=self.sampling_params,
            use_tqdm=False,
        )
        self.requests = []
        return outputs

    def close(self) -> None:
        engine = getattr(self.llm, "llm_engine", None)
        engine_core = getattr(engine, "engine_core", None)
        if engine_core is not None and hasattr(engine_core, "shutdown"):
            engine_core.shutdown()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_data_processor(llm_cfg, exp_name: str) -> DataProcessor:
    vllm_engine = LocalVLLMActor(
        model_path=llm_cfg.model_name_or_path,
        tensor_parallel_size=llm_cfg.vllm_tensor_parallel_size,
        max_model_len=llm_cfg.prompt_max_len + llm_cfg.generate_max_len,
        gpu_memory_utilization=llm_cfg.gpu_memory_utilization,
    )
    strategy = SimpleNamespace(args=llm_cfg)
    return DataProcessor(
        rank=0,
        world_size=1,
        vllm_engine=vllm_engine,
        strategy=strategy,
        model_path=llm_cfg.model_name_or_path,
        exp_name=exp_name,
        instance_name="llm_as_policy",
    )


def normalize_logprobs(logprobs: Dict[str, float], temperature: float) -> Dict[str, float]:
    if not logprobs:
        return {}
    if temperature <= 1e-8:
        best = max(logprobs, key=logprobs.get)
        return {k: 0.0 if k == best else float("-inf") for k in logprobs}

    scaled = {k: v / temperature for k, v in logprobs.items()}
    max_val = max(scaled.values())
    log_z = math.log(sum(math.exp(v - max_val) for v in scaled.values())) + max_val
    return {k: v - log_z for k, v in scaled.items()}


def choose_action(
    llm_prior: Dict[str, float],
    valid_actions: List[str],
    temperature: float,
    sample: bool,
) -> Tuple[int, str, Dict[str, float]]:
    if len(valid_actions) == 0:
        return 0, "go", {"go": 1.0}

    filtered = {a: llm_prior[a] for a in valid_actions if a in llm_prior}
    if not filtered:
        return 0, valid_actions[0], {valid_actions[0]: 1.0}

    norm_logprobs = normalize_logprobs(filtered, temperature)
    policy = {a: math.exp(lp) for a, lp in norm_logprobs.items()}
    z = sum(policy.values())
    policy = {a: p / z for a, p in policy.items()} if z > 0 else {valid_actions[0]: 1.0}

    if sample:
        action_names = list(policy.keys())
        probs = np.array([policy[a] for a in action_names], dtype=np.float64)
        probs = probs / probs.sum()
        action_name = str(np.random.choice(action_names, p=probs))
    else:
        action_name = max(policy, key=policy.get)

    return valid_actions.index(action_name), action_name, policy


def run_episode(
    seed: int,
    env_cfg: Dict[str, Any],
    data_processor: DataProcessor,
    history_len: int,
    temperature: float,
    sample: bool,
) -> Dict[str, Any]:
    set_seed(seed)
    env = JerichoEnv(env_cfg)
    env.seed(seed, dynamic_seed=False)
    obs = env.reset()
    history = deque(maxlen=history_len)
    trajectory = []
    total_reward = 0.0
    start = time.time()

    try:
        done = False
        step = 0
        while not done:
            valid_actions = list(obs.get("valid_actions", []))
            history_snapshot = list(history)
            prompt = data_processor.get_user_prompt(
                history=history_snapshot,
                current_obs=obs["raw_obs_text"],
                valid_actions=valid_actions,
            )
            llm_prior_per_seq, _, _ = data_processor.get_llm_prior(
                states=[obs["raw_obs_text"]],
                valid_actions_list=[valid_actions],
                histories=[history_snapshot],
                return_cot=True,
            )
            llm_prior = dict(llm_prior_per_seq[0])
            action_idx, action_str, policy = choose_action(
                llm_prior=llm_prior,
                valid_actions=valid_actions,
                temperature=temperature,
                sample=sample,
            )

            timestep = env.step(action_idx)
            reward = float(timestep.reward)
            done = bool(timestep.done)
            info = dict(timestep.info)
            total_reward += reward

            top_actions = sorted(policy.items(), key=lambda x: x[1], reverse=True)[:5]
            trajectory.append(
                {
                    "step": step,
                    "observation": obs["raw_obs_text"],
                    "prompt": prompt,
                    "history_len_cfg": history_len,
                    "history_len_used": len(history_snapshot),
                    "prompt_includes_valid_actions": bool(
                        getattr(data_processor.args.user_prompt_dict, "observation_with_valid_actions", False)
                    ),
                    "valid_actions": valid_actions,
                    "action": info.get("action_str", action_str),
                    "reward": reward,
                    "score": float(info.get("score", total_reward)),
                    "top_policy": [{"action": a, "prob": float(p)} for a, p in top_actions],
                    "done": done,
                }
            )
            history.append((obs["raw_obs_text"], info.get("action_str", action_str), reward))
            obs = timestep.obs
            step += 1

        final_score = float(trajectory[-1]["score"]) if trajectory else total_reward
        return {
            "seed": seed,
            "score": final_score,
            "total_reward": total_reward,
            "steps": len(trajectory),
            "duration_sec": time.time() - start,
            "trajectory": trajectory,
        }
    finally:
        worker = getattr(env, "_valid_actions_worker", None)
        if worker is not None:
            with contextlib.suppress(Exception):
                worker.close()
        env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PriorZero ablation: frozen LLM as policy")
    parser.add_argument("--env_id", type=str, default="detective.z5")
    parser.add_argument("--model", type=str, default="qwen2.5-3b")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--history_len", "--his_len", type=int, default=25)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--sample", action="store_true", help="Sample from the LLM action prior instead of greedy argmax.")
    parser.add_argument("--use_cot", action="store_true")
    parser.add_argument("--output_dir", type=str, default="ablation/llm_as_policy/results")
    parser.add_argument("--exp_name", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    env_name = args.env_id.replace(".z5", "")
    exp_name = args.exp_name or f"data_ablation/llm_as_policy/{env_name}_{args.model}_his{args.history_len}"
    main_cfg, _, llm_cfg = get_priorzero_config(
        env_id=args.env_id,
        seed=args.seeds[0],
        exp_name=exp_name,
        use_cot=args.use_cot,
        model_key=args.model,
        multi_gpu=False,
    )
    model_cfg = get_model_config(args.model)
    llm_cfg.enable_rft = False
    llm_cfg.enable_world_model = False
    llm_cfg.history_length = args.history_len
    llm_cfg.vllm_enable_sleep = False
    llm_cfg.gpu_memory_utilization = model_cfg["gpu_memory_utilization"]
    if args.temperature is not None:
        llm_cfg.llm_prior_temperature = args.temperature

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(exp_name, "log").mkdir(parents=True, exist_ok=True)

    data_processor = build_data_processor(llm_cfg=llm_cfg, exp_name=exp_name)
    try:
        env_cfg = dict(main_cfg.env)
        results = []

        for seed in args.seeds:
            result = run_episode(
                seed=seed,
                env_cfg=env_cfg,
                data_processor=data_processor,
                history_len=args.history_len,
                temperature=llm_cfg.llm_prior_temperature,
                sample=args.sample,
            )
            results.append(result)
            print(
                f"[LLM-as-policy] seed={seed} score={result['score']} "
                f"steps={result['steps']} duration={result['duration_sec']:.1f}s"
            )

        scores = [r["score"] for r in results]
        summary = {
            "ablation": "llm_as_policy",
            "env_id": args.env_id,
            "model": args.model,
            "model_path": llm_cfg.model_name_or_path,
            "history_len": args.history_len,
            "temperature": llm_cfg.llm_prior_temperature,
            "sample": args.sample,
            "seeds": args.seeds,
            "score_mean": float(np.mean(scores)) if scores else 0.0,
            "score_std": float(np.std(scores)) if scores else 0.0,
            "score_min": float(np.min(scores)) if scores else 0.0,
            "score_max": float(np.max(scores)) if scores else 0.0,
            "results": results,
        }

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"{args.env_id}_{args.model}_his{args.history_len}_{timestamp}.json"
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(
            "[LLM-as-policy] "
            f"mean={summary['score_mean']:.3f} std={summary['score_std']:.3f} "
            f"min={summary['score_min']:.3f} max={summary['score_max']:.3f}"
        )
        print(f"[LLM-as-policy] results saved to {output_path}")
    finally:
        vllm_engine = getattr(data_processor, "vllm_engine", None)
        if vllm_engine is not None and hasattr(vllm_engine, "close"):
            with contextlib.suppress(Exception):
                vllm_engine.close()


if __name__ == "__main__":
    main()

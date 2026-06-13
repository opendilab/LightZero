#!/usr/bin/env python3
"""RLFT ablation for PriorZero Jericho experiments.

This experiment keeps the PriorZero prompt/action-prior setup, but removes the
world model, MCTS, and replay buffer. It performs the common RLFT loop:

1. Roll out N full episodes with the current LLM policy in Jericho.
2. Merge all step-level samples into one rollout buffer.
3. Estimate advantages with GAE from a value head on the actor backbone.
4. Run K PPO epochs by sampling minibatches from that rollout buffer.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import sys
import time
import gc
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.distributed as dist


REPO_ROOT = Path(__file__).resolve().parents[2]
LIGHTZERO_ROOT = REPO_ROOT.parents[2]
for path in (REPO_ROOT / "src", LIGHTZERO_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from priorzero_config import get_priorzero_config  # noqa: E402
from strategy.deepspeed import get_strategy, torch_dist_barrier_and_cuda_sync  # noqa: E402
from zoo.jericho.envs.jericho_env import JerichoEnv  # noqa: E402
from local_ppo import RLFTPolicyModel, RLFTReferenceModel, RLFTTrainer  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


class PromptOnlyDataProcessor:
    """Use PriorZero prompt utilities without vLLM-backed action scoring."""

    def __init__(self, llm_cfg, model_path: str):
        self.args = llm_cfg
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, padding_side="left"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.use_cot = llm_cfg.use_cot
        self.prompt_max_len = llm_cfg.prompt_max_len
        self.generate_max_len = llm_cfg.generate_max_len

    def get_system_prompt(self) -> str:
        parts = [
            "You are an expert player in a text-based adventure game. Your goal is to maximize the score by choosing the optimal next action.",
            "Please analyze the game history and current observation to decide the single best next action.",
            "OUTPUT FORMAT:",
        ]
        if self.use_cot:
            parts.append(
                "You MUST produce exactly TWO parts in the following order:\n"
                "1. Reasoning: Analyze the current situation, available actions, constraints, and uncertainties. Do NOT reveal the final choice here.\n"
                "2. Action: The final chosen action.\n"
                "Strict Format Example:\n"
                "Reasoning: <detailed_analysis>\n"
                "Action: <single_action>"
            )
        else:
            parts.append(
                "Output exactly one line starting with 'Action:'.\n"
                "Example:\n"
                "Action: <your_action_here>"
            )
        return "\n".join(parts)

    def get_user_prompt(
        self,
        history: List[Tuple[str, str, float]] | None = None,
        current_obs: str | None = None,
        valid_actions: List[str] | None = None,
    ) -> str:
        prompt_parts = []
        user_prompt_dict = self.args.user_prompt_dict
        if history:
            prompt_parts.append("=== GAME HISTORY ===")
            for i, (obs, action, reward) in enumerate(history, start=1):
                prompt_parts.append(f"Step {i}:")
                prompt_parts.append(f"Observation: {obs.strip()}")
                prompt_parts.append(f"Action: {action.strip()}")
                if user_prompt_dict.history_with_reward:
                    prompt_parts.append(f"Reward: {reward}")
            prompt_parts.append("")

        prompt_parts.append("=== CURRENT OBSERVATION ===")
        prompt_parts.append((current_obs or "").strip())
        if user_prompt_dict.observation_with_valid_actions and valid_actions:
            actions_str = ", ".join([f"'{act}'" for act in valid_actions])
            prompt_parts.append(f"\n[Valid Actions]\nYou can choose from the following actions: {actions_str}")

        prompt_parts.append("\n=== INSTRUCTION ===")
        if self.use_cot:
            prompt_parts.append(
                "Please analyze the situation and provide your response in the following format:\n"
                "Reasoning: <detailed_analysis>\n"
                "Action: <single_action>"
            )
        else:
            prompt_parts.append(
                "Decide on the best next move and output it in the following format:\n"
                "Action: <your_action_here>"
            )
        return "\n".join(prompt_parts)

    def build_chat_context(self, user_prompt: str) -> str:
        return self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": self.get_system_prompt()},
                {"role": "user", "content": user_prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )


@torch.no_grad()
def score_valid_actions_with_actor(
    policy_model: RLFTPolicyModel,
    tokenizer,
    data_processor: PromptOnlyDataProcessor,
    prompt: str,
    valid_actions: List[str],
    prompt_max_len: int,
) -> Dict[str, Dict[str, Any]]:
    if not valid_actions:
        valid_actions = ["go"]

    all_context_texts = [data_processor.build_chat_context(prompt) for _ in valid_actions]
    context_ids = tokenizer(
        all_context_texts,
        add_special_tokens=False,
        max_length=prompt_max_len - 64,
        padding=False,
        truncation=True,
    )["input_ids"]
    label_texts = ["Action: " + action + tokenizer.eos_token for action in valid_actions]
    label_ids = tokenizer(label_texts, add_special_tokens=False, padding=False, truncation=False)["input_ids"]
    full_ids = [c + l for c, l in zip(context_ids, label_ids)]

    inputs = tokenizer.pad({"input_ids": full_ids}, padding=True, return_tensors="pt")
    max_tgt_len = max(len(ids) for ids in label_ids)
    action_mask = torch.zeros((len(valid_actions), max_tgt_len), dtype=torch.long)
    for idx, ids in enumerate(label_ids):
        action_mask[idx, -len(ids):] = 1

    log_probs, state_values = policy_model.forward_logprobs_values(
        sequences=inputs.input_ids,
        action_mask=action_mask,
        attention_mask=inputs.attention_mask,
    )
    log_probs_cpu = log_probs.detach().cpu()
    state_values_cpu = state_values.detach().cpu()
    action_mask_cpu = action_mask.cpu()

    scored = {}
    for idx, action in enumerate(valid_actions):
        lp_tokens = log_probs_cpu[idx, action_mask_cpu[idx].bool()].tolist()
        score = float(sum(lp_tokens) / max(len(lp_tokens), 1))
        scored[action] = {
            "score": score,
            "rollout_logprob": lp_tokens,
            "full_ids": full_ids[idx],
            "label_ids": label_ids[idx],
            "value": float(state_values_cpu[idx].item()),
        }
    return scored


def close_env(env: JerichoEnv) -> None:
    worker = getattr(env, "_valid_actions_worker", None)
    if worker is not None:
        with contextlib.suppress(Exception):
            worker.close()
    env.close()


def compute_gae(
    rewards: List[float],
    values: List[float],
    dones: List[bool],
    gamma: float,
    gae_lambda: float,
) -> Tuple[List[float], List[float]]:
    advantages = [0.0 for _ in rewards]
    last_gae = 0.0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_non_terminal = 0.0 if dones[t] else 1.0
            next_value = 0.0
        else:
            next_non_terminal = 0.0 if dones[t] else 1.0
            next_value = values[t + 1]
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[t] = last_gae
    returns = [adv + value for adv, value in zip(advantages, values)]
    return advantages, returns


def rollout_episode(
    env_cfg: Dict[str, Any],
    data_processor: PromptOnlyDataProcessor,
    seed: int,
    history_len: int,
    temperature: float,
    sample: bool,
    policy_model: RLFTPolicyModel,
) -> Dict[str, Any]:
    set_seed(seed)
    env = JerichoEnv(env_cfg)
    env.seed(seed, dynamic_seed=False)
    obs = env.reset()
    history = deque(maxlen=history_len)
    samples = []
    trajectory = []
    rewards = []
    total_reward = 0.0
    start = time.time()

    try:
        done = False
        step = 0
        while not done:
            valid_actions = list(obs.get("valid_actions", []))
            if not valid_actions:
                valid_actions = ["go"]
            history_snapshot = list(history)
            prompt = data_processor.get_user_prompt(
                history=history_snapshot,
                current_obs=obs["raw_obs_text"],
                valid_actions=valid_actions,
            )
            scored_actions = score_valid_actions_with_actor(
                policy_model=policy_model,
                tokenizer=data_processor.tokenizer,
                data_processor=data_processor,
                prompt=prompt,
                valid_actions=valid_actions,
                prompt_max_len=data_processor.prompt_max_len,
            )
            llm_prior = {action: info["score"] for action, info in scored_actions.items()}
            action_idx, action_str, policy = choose_action(
                llm_prior=llm_prior,
                valid_actions=valid_actions,
                temperature=temperature,
                sample=sample,
            )
            norm_logprobs = normalize_logprobs(llm_prior, temperature)

            action_info = scored_actions[action_str]
            if len(action_info["label_ids"]) > 0:
                candidate_actions = [a for a in valid_actions if a in scored_actions]
                samples.append(
                    {
                        "prompt": prompt,
                        "action": action_str,
                        "valid_actions": candidate_actions,
                        "full_ids": action_info["full_ids"],
                        "label_ids": action_info["label_ids"],
                        "candidate_full_ids": [scored_actions[a]["full_ids"] for a in candidate_actions],
                        "candidate_label_ids": [scored_actions[a]["label_ids"] for a in candidate_actions],
                        "chosen_candidate_index": candidate_actions.index(action_str),
                        "old_action_logprob": float(action_info["score"]),
                        "old_categorical_logprob": float(norm_logprobs[action_str]),
                        "value": action_info["value"],
                    }
                )

            timestep = env.step(action_idx)
            reward = float(timestep.reward)
            done = bool(timestep.done)
            info = dict(timestep.info)
            if samples:
                samples[-1]["reward"] = reward
                samples[-1]["done"] = done
            rewards.append(reward)
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
            "samples": samples,
            "trajectory": trajectory,
        }
    finally:
        close_env(env)


def gather_advantage_stats(local_advantages: List[float], world_size: int) -> Tuple[float, float]:
    if world_size <= 1:
        arr = np.asarray(local_advantages, dtype=np.float32)
    else:
        gathered = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, local_advantages)
        arr = np.asarray([x for rank_values in gathered for x in rank_values], dtype=np.float32)
    if arr.size == 0:
        return 0.0, 1.0
    return float(arr.mean()), float(arr.std() + 1e-8)


def attach_gae(
    rollouts: List[Dict[str, Any]],
    samples: List[Dict[str, Any]],
    gamma: float,
    gae_lambda: float,
) -> None:
    if not samples:
        return

    cursor = 0
    for rollout in rollouts:
        ep_samples = rollout["samples"]
        ep_n = len(ep_samples)
        ep_values = [float(s["value"]) for s in ep_samples]
        ep_rewards = [float(s["reward"]) for s in ep_samples]
        ep_dones = [bool(s["done"]) for s in ep_samples]
        ep_advantages, ep_returns = compute_gae(
            rewards=ep_rewards,
            values=ep_values,
            dones=ep_dones,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )
        for sample_item, value, advantage, ret in zip(ep_samples, ep_values, ep_advantages, ep_returns):
            sample_item["value"] = value
            sample_item["advantage"] = float(advantage)
            sample_item["return"] = float(ret)
        cursor += ep_n


def build_train_batch(
    samples: List[Dict[str, Any]],
    tokenizer,
    adv_mean: float,
    adv_std: float,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    List[Dict[str, float]],
]:
    if not samples:
        raise ValueError("Cannot build RLFT batch from empty samples.")

    full_ids_list = [s["full_ids"] for s in samples]
    label_ids_list = [s["label_ids"] for s in samples]
    inputs = tokenizer.pad({"input_ids": full_ids_list}, padding=True, return_tensors="pt")
    max_tgt_len = max(len(ids) for ids in label_ids_list)
    action_mask = torch.zeros((len(samples), max_tgt_len), dtype=torch.long)
    for idx, ids in enumerate(label_ids_list):
        action_mask[idx, -len(ids):] = 1

    old_action_logprobs = torch.zeros((len(samples),), dtype=torch.float32)
    returns = torch.zeros((len(samples),), dtype=torch.float32)
    old_values = torch.zeros((len(samples),), dtype=torch.float32)

    normalized_advantages = []
    log_status = []
    for idx, sample_item in enumerate(samples):
        raw_adv = float(sample_item["advantage"])
        norm_adv = (raw_adv - adv_mean) / adv_std
        old_action_logprobs[idx] = float(sample_item["old_action_logprob"])
        returns[idx] = float(sample_item["return"])
        old_values[idx] = float(sample_item["value"])
        normalized_advantages.append(norm_adv)
        log_status.append(
            {
                "value_advantage": float(norm_adv),
                "raw_gae_advantage": raw_adv,
                "value_target_return": float(sample_item["return"]),
                "old_value": float(sample_item["value"]),
                "valid_action_count": float(len(sample_item.get("valid_actions", []))),
            }
        )

    return (
        inputs.input_ids,
        inputs.attention_mask,
        action_mask,
        torch.tensor(normalized_advantages, dtype=torch.float32),
        old_action_logprobs,
        returns,
        old_values,
        log_status,
    )


def memory_cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def iter_fixed_count_minibatches(
    samples: List[Dict[str, Any]],
    minibatch_size: int,
    num_minibatches: int,
    rng: random.Random,
):
    epoch_samples = list(samples)
    rng.shuffle(epoch_samples)
    for batch_idx in range(num_minibatches):
        start_idx = batch_idx * minibatch_size
        minibatch = epoch_samples[start_idx : start_idx + minibatch_size]
        if len(minibatch) < minibatch_size:
            minibatch = minibatch + rng.choices(samples, k=minibatch_size - len(minibatch))
        yield minibatch


def write_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PriorZero ablation: RLFT without world model")
    parser.add_argument("--env_id", type=str, default="detective.z5")
    parser.add_argument("--model", type=str, default="qwen2.5-3b")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--history_len", "--his_len", type=int, default=25)
    parser.add_argument("--max_env_steps", type=int, default=100_000)
    parser.add_argument("--max_rlft_iters", type=int, default=1_000_000)
    parser.add_argument("--rollout_episodes_per_iter", type=int, default=50)
    parser.add_argument("--ppo_epochs", type=int, default=2)
    parser.add_argument("--ppo_minibatch_size", type=int, default=128)
    parser.add_argument("--eval_episodes", type=int, default=2)
    parser.add_argument("--eval_freq", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--micro_train_batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--kl_coef", type=float, default=0.01)
    parser.add_argument("--value_loss_coef", type=float, default=0.5)
    parser.add_argument("--value_clip_eps", type=float, default=0.2)
    parser.add_argument("--zero_stage", type=int, default=2)
    parser.add_argument("--use_cot", action="store_true")
    parser.add_argument("--sample_eval", action="store_true")
    parser.add_argument("--output_dir", type=str, default="ablation/rlft/results")
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--save_freq", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rollout_episodes_per_iter <= 0:
        raise ValueError("--rollout_episodes_per_iter must be positive.")
    if args.ppo_epochs <= 0:
        raise ValueError("--ppo_epochs must be positive.")
    if args.ppo_minibatch_size <= 0:
        raise ValueError("--ppo_minibatch_size must be positive.")
    if args.max_env_steps <= 0:
        raise ValueError("--max_env_steps must be positive.")
    if args.zero_stage != 2:
        raise ValueError("This ablation is configured for DeepSpeed ZeRO-2; use --zero_stage 2.")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    env_name = args.env_id.replace(".z5", "")
    exp_name = args.exp_name or f"data_ablation/rlft/{env_name}_{args.model}_his{args.history_len}"
    main_cfg, _, llm_cfg = get_priorzero_config(
        env_id=args.env_id,
        seed=args.seed,
        exp_name=exp_name,
        use_cot=args.use_cot,
        model_key=args.model,
        multi_gpu=True,
    )
    llm_cfg.enable_world_model = False
    llm_cfg.enable_rft = True
    llm_cfg.history_length = args.history_len
    llm_cfg.train_batch_size = args.ppo_minibatch_size
    llm_cfg.micro_train_batch_size = args.micro_train_batch_size
    llm_cfg.learning_rate = args.learning_rate
    llm_cfg.zero_stage = args.zero_stage
    llm_cfg.rft_kl_coef = args.kl_coef
    llm_cfg.enable_value_head = True
    llm_cfg.value_loss_coef = args.value_loss_coef
    llm_cfg.value_clip_eps = args.value_clip_eps
    llm_cfg.rlft_action_temperature = llm_cfg.llm_prior_temperature
    llm_cfg.policy_loss_type = "ppo"
    llm_cfg.use_rollout_as_old_policy = True
    llm_cfg.enable_vllm_is_correction = False
    llm_cfg.vllm_enable_sleep = False
    llm_cfg.enable_vllm = False
    llm_cfg.disable_vllm_sync = True
    llm_cfg.max_steps = args.max_rlft_iters
    llm_cfg.seed = args.seed
    llm_cfg.llm_save_freq = args.save_freq
    llm_cfg.save_path = f"./{exp_name}/llm_ckpt/"
    if args.temperature is not None:
        llm_cfg.llm_prior_temperature = args.temperature
        llm_cfg.rlft_action_temperature = args.temperature

    strategy = get_strategy(llm_cfg)
    strategy.setup_distributed()
    world_size = strategy.world_size
    if args.rollout_episodes_per_iter < world_size:
        raise ValueError(
            f"--rollout_episodes_per_iter ({args.rollout_episodes_per_iter}) must be >= world_size ({world_size})."
        )
    if args.ppo_minibatch_size < world_size:
        raise ValueError(
            f"--ppo_minibatch_size ({args.ppo_minibatch_size}) must be >= world_size ({world_size})."
        )
    if args.ppo_minibatch_size < args.micro_train_batch_size * world_size:
        raise ValueError(
            "--ppo_minibatch_size must be at least micro_train_batch_size * world_size "
            f"({args.micro_train_batch_size * world_size})."
        )
    local_ppo_minibatch_size = max(1, args.ppo_minibatch_size // world_size)
    set_seed(args.seed + rank)

    Path(exp_name, "log").mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rollout_log_path = output_dir / f"{args.env_id}_{args.model}_his{args.history_len}_seed{args.seed}_rollouts.jsonl"
    eval_log_path = output_dir / f"{args.env_id}_{args.model}_his{args.history_len}_seed{args.seed}_eval.jsonl"

    vllm_engine = None
    data_processor = PromptOnlyDataProcessor(llm_cfg=llm_cfg, model_path=llm_cfg.model_name_or_path)

    ref_model = RLFTReferenceModel(strategy=strategy, pretrain=llm_cfg.model_name_or_path) if llm_cfg.rft_kl_coef > 0 else None
    policy_model = RLFTPolicyModel(
        strategy=strategy,
        pretrain=llm_cfg.model_name_or_path,
        max_steps=llm_cfg.max_steps,
    )
    trainer = RLFTTrainer(
        cfg=llm_cfg,
        strategy=strategy,
        policy_model=policy_model,
        reference_model=ref_model,
    )

    env_cfg = dict(main_cfg.env)
    torch_dist_barrier_and_cuda_sync()
    total_env_steps = 0
    total_episodes = 0

    train_iter = 0
    while train_iter < args.max_rlft_iters and total_env_steps < args.max_env_steps:
        local_rollouts = []
        local_samples = []
        for episode_idx in range(args.rollout_episodes_per_iter):
            if episode_idx % world_size != rank:
                continue
            rollout_seed = args.seed + train_iter * args.rollout_episodes_per_iter + episode_idx
            rollout = rollout_episode(
                env_cfg=env_cfg,
                data_processor=data_processor,
                seed=rollout_seed,
                history_len=args.history_len,
                temperature=llm_cfg.llm_prior_temperature,
                sample=True,
                policy_model=policy_model,
            )
            local_samples.extend(rollout["samples"])
            local_rollouts.append(rollout)

        attach_gae(
            rollouts=local_rollouts,
            samples=local_samples,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
        )
        local_advantages = [float(s["advantage"]) for s in local_samples]
        adv_mean, adv_std = gather_advantage_stats(local_advantages, world_size=world_size)

        local_ready = int(len(local_samples) > 0)
        ready_flags = [None for _ in range(world_size)]
        dist.all_gather_object(ready_flags, local_ready)
        if min(ready_flags) == 0:
            if rank == 0:
                print(f"[RLFT] skip train_iter={train_iter}, ready_flags={ready_flags}")
            train_iter += 1
            continue

        gathered_rollouts = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_rollouts, local_rollouts)
        if rank == 0:
            flat_rollouts = [r for rank_rollouts in gathered_rollouts for r in rank_rollouts]
            iter_env_steps = int(sum(r["steps"] for r in flat_rollouts))
            iter_episodes = int(len(flat_rollouts))
            total_env_steps += iter_env_steps
            total_episodes += iter_episodes
        else:
            flat_rollouts = None
            iter_env_steps = 0
            iter_episodes = 0
        counters = [total_env_steps, total_episodes, iter_env_steps, iter_episodes]
        dist.broadcast_object_list(counters, src=0)
        total_env_steps, total_episodes, iter_env_steps, iter_episodes = [int(x) for x in counters]

        local_minibatches = int(math.ceil(len(local_samples) / local_ppo_minibatch_size))
        gathered_minibatches = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_minibatches, local_minibatches)
        minibatches_per_epoch = int(max(gathered_minibatches))

        train_rng = random.Random(args.seed + train_iter * 1_000_003 + rank)
        train_statuses = []
        minibatches_trained = 0
        optimizer_updates_before = int(getattr(policy_model, "train_iter", 0))
        for ppo_epoch in range(args.ppo_epochs):
            for minibatch_samples in iter_fixed_count_minibatches(
                samples=local_samples,
                minibatch_size=local_ppo_minibatch_size,
                num_minibatches=minibatches_per_epoch,
                rng=train_rng,
            ):
                batch = build_train_batch(
                    samples=minibatch_samples,
                    tokenizer=data_processor.tokenizer,
                    adv_mean=adv_mean,
                    adv_std=adv_std,
                )
                status = trainer.train_batch(batch, collect_env_steps=total_env_steps)
                train_statuses.extend(status or [])
                minibatches_trained += 1
                memory_cleanup()
        optimizer_updates_after = int(getattr(policy_model, "train_iter", optimizer_updates_before))
        optimizer_updates = optimizer_updates_after - optimizer_updates_before

        if rank == 0:
            scores = [r["score"] for r in flat_rollouts]
            record = {
                "iter": train_iter,
                "phase": "train_rollout",
                "env_steps": total_env_steps,
                "episode_count": total_episodes,
                "iter_env_steps": iter_env_steps,
                "iter_episode_count": iter_episodes,
                "scores": scores,
                "episode_return_mean": float(np.mean(scores)) if scores else 0.0,
                "episode_return_std": float(np.std(scores)) if scores else 0.0,
                "episode_return_min": float(np.min(scores)) if scores else 0.0,
                "episode_return_max": float(np.max(scores)) if scores else 0.0,
                "advantage_mean": adv_mean,
                "advantage_std": adv_std,
                "rollout_sample_count": int(sum(len(r["samples"]) for r in flat_rollouts)),
                "local_train_sample_count": len(local_samples),
                "ppo_epochs": args.ppo_epochs,
                "ppo_minibatch_size": args.ppo_minibatch_size,
                "local_ppo_minibatch_size": local_ppo_minibatch_size,
                "minibatches_per_epoch": minibatches_per_epoch,
                "ppo_minibatches_trained": minibatches_trained,
                "optimizer_updates": optimizer_updates,
                "last_train_status": train_statuses[-1] if train_statuses else {},
                "gamma": args.gamma,
                "gae_lambda": args.gae_lambda,
                "max_env_steps": args.max_env_steps,
                "history_len": args.history_len,
                "prompt_includes_valid_actions": True,
                "rollouts": flat_rollouts,
            }
            write_jsonl(rollout_log_path, record)
            print(
                f"[RLFT] iter={train_iter} env_steps={total_env_steps} episodes={total_episodes} "
                f"iter_episodes={iter_episodes} iter_env_steps={iter_env_steps} "
                f"episode_return_mean={record['episode_return_mean']:.3f} "
                f"episode_return_min={record['episode_return_min']:.3f} "
                f"episode_return_max={record['episode_return_max']:.3f} "
                f"samples={record['rollout_sample_count']} ppo_epochs={args.ppo_epochs} "
                f"minibatches_per_epoch={minibatches_per_epoch} "
                f"ppo_minibatches={minibatches_trained} optimizer_updates={optimizer_updates} "
                f"adv_mean={adv_mean:.3f} adv_std={adv_std:.3f}"
            )

        if args.eval_freq > 0 and (train_iter % args.eval_freq == 0 or train_iter == args.max_rlft_iters - 1):
            local_eval_rollouts = []
            for ep in range(args.eval_episodes):
                if ep % world_size != rank:
                    continue
                eval_seed = args.seed + 10_000 + train_iter * args.eval_episodes + ep
                eval_rollout = rollout_episode(
                    env_cfg=env_cfg,
                    data_processor=data_processor,
                    seed=eval_seed,
                    history_len=args.history_len,
                    temperature=llm_cfg.llm_prior_temperature,
                    sample=args.sample_eval,
                    policy_model=policy_model,
                )
                eval_rollout.pop("samples")
                local_eval_rollouts.append(eval_rollout)
            gathered_eval = [None for _ in range(world_size)]
            dist.all_gather_object(gathered_eval, local_eval_rollouts)
            if rank == 0:
                flat_eval = [r for rank_rollouts in gathered_eval for r in rank_rollouts]
                scores = [r["score"] for r in flat_eval]
                eval_env_steps = int(sum(r["steps"] for r in flat_eval))
                record = {
                    "iter": train_iter,
                    "phase": "eval",
                    "env_steps": total_env_steps,
                    "episode_count": total_episodes,
                    "eval_env_steps": eval_env_steps,
                    "eval_episode_count": int(len(flat_eval)),
                    "scores": scores,
                    "episode_return_mean": float(np.mean(scores)) if scores else 0.0,
                    "episode_return_std": float(np.std(scores)) if scores else 0.0,
                    "episode_return_min": float(np.min(scores)) if scores else 0.0,
                    "episode_return_max": float(np.max(scores)) if scores else 0.0,
                    "history_len": args.history_len,
                    "prompt_includes_valid_actions": True,
                    "rollouts": flat_eval,
                }
                write_jsonl(eval_log_path, record)
                print(
                    f"[RLFT][eval] iter={train_iter} env_steps={total_env_steps} episodes={total_episodes} "
                    f"eval_episodes={len(flat_eval)} eval_env_steps={eval_env_steps} "
                    f"episode_return_mean={record['episode_return_mean']:.3f} scores={scores}"
                )

        torch_dist_barrier_and_cuda_sync()
        train_iter += 1

    policy_model.save_model()
    if rank == 0:
        print(f"[RLFT] finished. rollout_log={rollout_log_path}, eval_log={eval_log_path}")

    torch_dist_barrier_and_cuda_sync()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

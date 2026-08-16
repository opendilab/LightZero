#!/usr/bin/env python3
import argparse
import json
import os
import random
import re
import shutil
import sys
import tempfile
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


LIGHTZERO_ROOT = Path(__file__).resolve().parents[2]
if str(LIGHTZERO_ROOT) not in sys.path:
    sys.path.insert(0, str(LIGHTZERO_ROOT))

from jericho.util import unabbreviate as jericho_unabbreviate  # noqa: E402
from zoo.jericho.envs.jericho_env import JerichoEnv  # noqa: E402


ENV_PRESETS: Dict[str, Dict[str, int]] = {
    "detective.z5": {"max_action_num": 12, "max_steps": 100},
    "omniquest.z5": {"max_action_num": 25, "max_steps": 100},
    "acorncourt.z5": {"max_action_num": 45, "max_steps": 50},
    "zork1.z5": {"max_action_num": 55, "max_steps": 500},
}

DEFAULT_ENVS = ["detective.z5", "omniquest.z5", "acorncourt.z5", "zork1.z5"]
DEFAULT_EXPERIMENT_MODES = ["without_valid_actions", "with_valid_actions"]

SYSTEM_PROMPT = (
    "You are an expert player in a text-based adventure game.\n"
    "Your goal is to maximize score by choosing the best next action.\n"
    "Always output exactly one line in this format:\n"
    "Action: <single_action>"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_ids", nargs="+", default=DEFAULT_ENVS)
    parser.add_argument(
        "--experiment_modes",
        nargs="+",
        default=DEFAULT_EXPERIMENT_MODES,
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="/mnt/afs/niuyazhe/workspace/xiongjyu/models/Qwen2.5-3B-Instruct",
    )
    parser.add_argument("--output_dir", type=str, default="./outputs/jericho_qwen25_3b_sft")
    parser.add_argument("--history_window", type=int, default=10)
    parser.add_argument("--collect_episodes_per_env", type=int, default=1)
    parser.add_argument("--eval_episodes_per_env", type=int, default=3)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--scoring_batch_size", type=int, default=8)
    parser.add_argument("--eval_action_mode", choices=["score", "generate"], default="score")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval_seed", type=int, default=1234)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--log_every", type=int, default=10)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def reward_to_float(reward: Any) -> float:
    if isinstance(reward, np.ndarray):
        return float(reward.item())
    if isinstance(reward, torch.Tensor):
        return float(reward.item())
    return float(reward)


def dump_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def dump_jsonl(path: str, records: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def normalize_action_text(action: str) -> str:
    return re.sub(r"\s+", " ", action.strip().lower())


def build_env_cfg(env_id: str, tokenizer_path: str) -> Dict[str, Any]:
    if env_id not in ENV_PRESETS:
        raise ValueError(f"Unknown env_id={env_id}")
    game_path = os.path.join(
        "/mnt/afs/niuyazhe/workspace/xiongjyu/LightZero/zoo/jericho/envs/z-machine-games-master/jericho-game-suite",
        env_id,
    )
    if not os.path.exists(game_path):
        raise FileNotFoundError(f"Game file not found: {game_path}")
    preset = ENV_PRESETS[env_id]
    return {
        "max_steps": int(preset["max_steps"]),
        "game_path": game_path,
        "max_action_num": int(preset["max_action_num"]),
        "tokenizer_path": tokenizer_path,
        "max_seq_len": 512,
        "remove_stuck_actions": False,
        "add_location_and_inventory": False,
        "for_unizero": False,
        "save_replay": False,
        "save_replay_path": None,
        "env_type": env_id.replace(".z5", ""),
        "collect_policy_mode": "expert",
        "use_cache": True,
        "cache_size": 100000,
    }


def build_user_prompt(
    history: Sequence[Tuple[str, str, float]],
    current_obs: str,
    valid_actions: Sequence[str],
    include_valid_actions: bool,
) -> str:
    parts: List[str] = []
    if history:
        parts.append("=== GAME HISTORY ===")
        for idx, (obs, action, reward) in enumerate(history, start=1):
            parts.append(f"Step {idx}:")
            parts.append(f"Observation: {obs.strip()}")
            parts.append(f"Action: {action.strip()}")
            parts.append(f"Reward: {reward:.4f}")
        parts.append("")

    parts.append("=== CURRENT OBSERVATION ===")
    parts.append(current_obs.strip())

    if include_valid_actions and valid_actions:
        parts.append("")
        parts.append("[Valid Actions]")
        parts.append("You must choose exactly one action from this list:")
        parts.append(", ".join([f"'{action}'" for action in valid_actions]))

    parts.append("")
    parts.append("=== INSTRUCTION ===")
    parts.append("Output only one line: Action: <single_action>")
    return "\n".join(parts)


def build_chat_prompt(tokenizer: Any, question: str, system_prompt: str = SYSTEM_PROMPT) -> str:
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def extract_action_from_generation(text: str, strict_regex_only: bool = False) -> str:
    matches = re.findall(r"Action\s*:\s*([^\n\r]+)", text, flags=re.IGNORECASE)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if matches:
        candidate = matches[-1]
    elif strict_regex_only:
        candidate = lines[0] if lines else ""
    else:
        candidate = lines[0] if lines else ""
    return candidate.strip().strip("`").strip("\"").strip("'").strip()


def match_action_in_valid(action: str, valid_actions: Sequence[str]) -> Optional[str]:
    if not valid_actions:
        return None
    valid_map = {normalize_action_text(valid_action): valid_action for valid_action in valid_actions}
    return valid_map.get(normalize_action_text(action))


def collect_walkthrough_data_for_env(
    env_id: str,
    env_cfg: Dict[str, Any],
    history_window: int,
    collect_episodes_per_env: int,
    seed: int,
) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    cfg = dict(env_cfg)
    cfg["collect_policy_mode"] = "expert"
    env = JerichoEnv(cfg)

    try:
        for episode_id in range(collect_episodes_per_env):
            env.seed(seed + episode_id, dynamic_seed=False)
            obs = env.reset(return_str=True)
            history: Deque[Tuple[str, str, float]] = deque(maxlen=history_window)
            walkthrough_actions = list(env.walkthrough_actions or [])

            for step_id, action in enumerate(walkthrough_actions):
                action = jericho_unabbreviate(str(action)).strip()
                current_obs = str(obs.get("raw_obs_text", ""))
                valid_actions = [str(item) for item in obs.get("valid_actions", [])]

                samples.append(
                    {
                        "env_id": env_id,
                        "episode_id": episode_id,
                        "step_id": step_id,
                        "history": list(history),
                        "current_obs": current_obs,
                        "valid_actions": valid_actions,
                        "target_action": action,
                    }
                )

                next_obs, reward, done, info = env.step(action, return_str=True)
                reward_value = reward_to_float(reward)
                executed_action = str(info.get("action_str", action))
                history.append((current_obs, executed_action, reward_value))
                obs = next_obs
                if done:
                    break
    finally:
        env.close()

    return samples

def collect_walkthrough_samples(args: argparse.Namespace) -> List[Dict[str, Any]]:
    all_samples: List[Dict[str, Any]] = []
    for env_id in args.env_ids:
        env_cfg = build_env_cfg(env_id, args.base_model_path)
        env_samples = collect_walkthrough_data_for_env(
            env_id=env_id,
            env_cfg=env_cfg,
            history_window=args.history_window,
            collect_episodes_per_env=args.collect_episodes_per_env,
            seed=args.seed,
        )
        all_samples.extend(env_samples)
        print(f"[Collect] env={env_id}, samples={len(env_samples)}")
    print(f"[Collect] total_samples={len(all_samples)}")
    return all_samples


def build_train_records(
    raw_samples: Sequence[Dict[str, Any]],
    include_valid_actions: bool,
) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    for sample in raw_samples:
        question = build_user_prompt(
            history=sample["history"],
            current_obs=str(sample["current_obs"]),
            valid_actions=sample["valid_actions"],
            include_valid_actions=include_valid_actions,
        )
        answer = f"Action: {sample['target_action']}"
        records.append({"question": question, "answer": answer})
    return records


def prepare_train_jsonl(
    raw_samples: Sequence[Dict[str, Any]],
    mode_dir: str,
    include_valid_actions: bool,
) -> Tuple[str, List[Dict[str, str]]]:
    train_jsonl_path = os.path.join(mode_dir, "train.jsonl")
    if len(raw_samples) == 0:
        raise RuntimeError(f"Missing raw walkthrough samples to build {train_jsonl_path}.")
    train_records = build_train_records(raw_samples, include_valid_actions=include_valid_actions)
    dump_jsonl(train_jsonl_path, train_records)
    return train_jsonl_path, train_records


class TrainJsonlDataset(Dataset):
    def __init__(self, train_records: Sequence[Dict[str, str]], tokenizer: Any, max_seq_len: int):
        self.items: List[Dict[str, List[int]]] = []
        eos_text = tokenizer.eos_token if tokenizer.eos_token is not None else ""

        for record in train_records:
            question = str(record["question"])
            answer = str(record["answer"])
            prompt_text = build_chat_prompt(tokenizer, question=question, system_prompt=SYSTEM_PROMPT)
            target_text = f"{answer}{eos_text}"
            full_text = prompt_text + target_text

            encoded_full = tokenizer(
                full_text,
                add_special_tokens=False,
                truncation=True,
                max_length=max_seq_len,
            )
            input_ids = encoded_full["input_ids"]
            attention_mask = encoded_full["attention_mask"]
            labels = [-100] * len(input_ids)

            target_ids = tokenizer(target_text, add_special_tokens=False, truncation=False)["input_ids"]
            target_len = min(len(target_ids), len(input_ids))
            labels[-target_len:] = input_ids[-target_len:]

            self.items.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        return self.items[idx]


class SFTCollator:
    def __init__(self, pad_token_id: int, padding_side: str):
        self.pad_token_id = int(pad_token_id)
        if padding_side not in {"left", "right"}:
            raise ValueError(f"Unsupported padding_side: {padding_side}")
        self.padding_side = padding_side

    def __call__(self, features: Sequence[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(feature["input_ids"]) for feature in features)
        input_ids: List[List[int]] = []
        attention_mask: List[List[int]] = []
        labels: List[List[int]] = []

        for feature in features:
            pad_len = max_len - len(feature["input_ids"])
            if self.padding_side == "left":
                input_ids.append([self.pad_token_id] * pad_len + feature["input_ids"])
                attention_mask.append([0] * pad_len + feature["attention_mask"])
                labels.append([-100] * pad_len + feature["labels"])
            else:
                input_ids.append(feature["input_ids"] + [self.pad_token_id] * pad_len)
                attention_mask.append(feature["attention_mask"] + [0] * pad_len)
                labels.append(feature["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def load_tokenizer_llm(model_path: str) -> Tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    if not torch.cuda.is_available():
        raise RuntimeError("BF16 is required, but CUDA is not available.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 is required, but current CUDA device does not support BF16.")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    return tokenizer, model


def train_sft(
    args: argparse.Namespace,
    model: Any,
    tokenizer: Any,
    train_records: Sequence[Dict[str, str]],
    work_dir: str,
) -> Dict[str, Any]:
    from transformers import Trainer, TrainingArguments

    if len(train_records) == 0:
        raise RuntimeError("No training records found.")

    dataset = TrainJsonlDataset(train_records, tokenizer, args.max_seq_len)
    trainer_output_dir = tempfile.mkdtemp(prefix="trainer_", dir=work_dir)
    original_use_cache = getattr(model.config, "use_cache", None)
    if original_use_cache is not None:
        model.config.use_cache = False
    training_args = TrainingArguments(
        output_dir=trainer_output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        bf16=True,
        logging_strategy="steps",
        logging_steps=max(1, args.log_every),
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,
        disable_tqdm=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=SFTCollator(tokenizer.pad_token_id, tokenizer.padding_side),
    )
    try:
        train_result = trainer.train()
        trainer.save_state()
        metrics = dict(train_result.metrics)
    finally:
        if original_use_cache is not None:
            model.config.use_cache = original_use_cache
        shutil.rmtree(trainer_output_dir, ignore_errors=True)
    return metrics


class JerichoLLMAgent:
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        device: torch.device,
        max_seq_len: int,
        max_new_tokens: int,
        scoring_batch_size: int,
        action_mode: str,
        include_valid_actions: bool,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_seq_len = max_seq_len
        self.max_new_tokens = max_new_tokens
        self.scoring_batch_size = scoring_batch_size
        self.action_mode = action_mode
        self.include_valid_actions = include_valid_actions
        self.eos_text = tokenizer.eos_token if tokenizer.eos_token is not None else ""

    @torch.no_grad()
    def _score_actions(self, chat_prompt: str, valid_actions: Sequence[str]) -> Dict[str, float]:
        if not valid_actions:
            return {"go": 0.0}

        scores: Dict[str, float] = {}
        completions = [f"Action: {action}{self.eos_text}" for action in valid_actions]
        for start in range(0, len(valid_actions), self.scoring_batch_size):
            end = min(start + self.scoring_batch_size, len(valid_actions))
            batch_actions = list(valid_actions[start:end])
            batch_completions = completions[start:end]
            full_texts = [chat_prompt + completion for completion in batch_completions]

            enc = self.tokenizer(
                full_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_seq_len,
                add_special_tokens=False,
            )
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)

            labels = torch.full_like(input_ids, -100)
            for idx, completion in enumerate(batch_completions):
                completion_ids = self.tokenizer(completion, add_special_tokens=False, truncation=False)["input_ids"]
                nonpad_pos = torch.nonzero(attention_mask[idx], as_tuple=False).squeeze(-1)
                if nonpad_pos.numel() == 0:
                    continue
                seq_end = int(nonpad_pos[-1].item()) + 1
                target_len = min(len(completion_ids), seq_end)
                labels[idx, seq_end - target_len : seq_end] = input_ids[idx, seq_end - target_len : seq_end]

            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, :-1, :]
            shifted_ids = input_ids[:, 1:]
            shifted_labels = labels[:, 1:]
            valid_mask = shifted_labels.ne(-100)
            token_logprobs = F.log_softmax(logits, dim=-1).gather(
                dim=-1,
                index=shifted_ids.unsqueeze(-1),
            ).squeeze(-1)
            denom = valid_mask.sum(dim=1).clamp(min=1)
            score_tensor = (token_logprobs * valid_mask).sum(dim=1) / denom

            for action, score in zip(batch_actions, score_tensor.detach().cpu().tolist()):
                scores[action] = float(score)
        return scores

    @torch.no_grad()
    def _generate_action(self, chat_prompt: str) -> str:
        enc = self.tokenizer(
            chat_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_len,
            add_special_tokens=False,
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        out = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        gen_ids = out[0, input_ids.size(1) :]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True)

    def select_action(
        self,
        history: Sequence[Tuple[str, str, float]],
        current_obs: str,
        valid_actions: Sequence[str],
    ) -> Tuple[str, str, str]:
        question = build_user_prompt(
            history=history,
            current_obs=current_obs,
            valid_actions=valid_actions,
            include_valid_actions=self.include_valid_actions,
        )
        chat_prompt = build_chat_prompt(self.tokenizer, question=question)

        if not self.include_valid_actions:
            raw_generation = self._generate_action(chat_prompt)
            action_str = extract_action_from_generation(raw_generation, strict_regex_only=True)
            return action_str, "generate_no_valid_direct", raw_generation

        if not valid_actions:
            return "go", "fallback_no_valid_actions", ""

        if self.action_mode == "score":
            scores = self._score_actions(chat_prompt, valid_actions)
            return max(scores.items(), key=lambda item: item[1])[0], "score", ""

        raw_generation = self._generate_action(chat_prompt)
        predicted_action = extract_action_from_generation(raw_generation)
        mapped_action = match_action_in_valid(predicted_action, valid_actions)
        if mapped_action is None:
            mapped_action = match_action_in_valid(jericho_unabbreviate(predicted_action), valid_actions)
        if mapped_action is not None:
            return mapped_action, "generate", raw_generation

        scores = self._score_actions(chat_prompt, valid_actions)
        return max(scores.items(), key=lambda item: item[1])[0], "generate_fallback_score", raw_generation


def summarize_eval(
    stage_name: str,
    include_valid_actions: bool,
    effective_action_policy: str,
    per_env_scores: Dict[str, List[float]],
    per_env_returns: Dict[str, List[float]],
) -> Dict[str, Any]:
    overall_scores = [score for values in per_env_scores.values() for score in values]
    overall_returns = [ret for values in per_env_returns.values() for ret in values]
    summary: Dict[str, Any] = {
        "stage": stage_name,
        "include_valid_actions": include_valid_actions,
        "effective_action_policy": effective_action_policy,
        "overall": {
            "num_episodes": len(overall_scores),
            "score_mean": float(np.mean(overall_scores)) if overall_scores else 0.0,
            "score_std": float(np.std(overall_scores)) if overall_scores else 0.0,
            "return_mean": float(np.mean(overall_returns)) if overall_returns else 0.0,
            "return_std": float(np.std(overall_returns)) if overall_returns else 0.0,
        },
        "per_env": {},
    }
    for env_id in per_env_scores:
        env_scores = per_env_scores[env_id]
        env_returns = per_env_returns[env_id]
        summary["per_env"][env_id] = {
            "num_episodes": len(env_scores),
            "score_mean": float(np.mean(env_scores)) if env_scores else 0.0,
            "score_std": float(np.std(env_scores)) if env_scores else 0.0,
            "return_mean": float(np.mean(env_returns)) if env_returns else 0.0,
            "return_std": float(np.std(env_returns)) if env_returns else 0.0,
        }
    return summary


def evaluate_model(
    args: argparse.Namespace,
    model: Any,
    tokenizer: Any,
    stage_name: str,
    stage_dir: str,
    include_valid_actions: bool,
) -> Dict[str, Any]:
    os.makedirs(stage_dir, exist_ok=True)
    agent = JerichoLLMAgent(
        model=model,
        tokenizer=tokenizer,
        device=model.device,
        max_seq_len=args.max_seq_len,
        max_new_tokens=args.max_new_tokens,
        scoring_batch_size=args.scoring_batch_size,
        action_mode=args.eval_action_mode,
        include_valid_actions=include_valid_actions,
    )

    effective_action_policy = "generate_direct" if not include_valid_actions else args.eval_action_mode
    episode_records: List[Dict[str, Any]] = []
    per_env_scores: Dict[str, List[float]] = {env_id: [] for env_id in args.env_ids}
    per_env_returns: Dict[str, List[float]] = {env_id: [] for env_id in args.env_ids}

    for env_id in args.env_ids:
        env_cfg = build_env_cfg(env_id, args.base_model_path)
        env_cfg["collect_policy_mode"] = "agent"
        env = JerichoEnv(env_cfg)

        try:
            for episode_id in range(args.eval_episodes_per_env):
                env.seed(args.eval_seed + episode_id, dynamic_seed=False)
                obs = env.reset(return_str=True)
                history: Deque[Tuple[str, str, float]] = deque(maxlen=args.history_window)
                trajectory: List[Dict[str, Any]] = []
                last_info: Dict[str, Any] = {}
                step_count = 0

                while True:
                    current_obs = str(obs.get("raw_obs_text", ""))
                    valid_actions = [str(action) for action in obs.get("valid_actions", [])]
                    selected_action, selection_method, raw_generation = agent.select_action(
                        history=list(history),
                        current_obs=current_obs,
                        valid_actions=valid_actions,
                    )

                    next_obs, reward, done, info = env.step(selected_action, return_str=True)
                    reward_value = reward_to_float(reward)
                    executed_action = str(info.get("action_str", selected_action))

                    trajectory.append(
                        {
                            "step_id": step_count,
                            "observation": current_obs,
                            "selected_action": selected_action,
                            "executed_action": executed_action,
                            "reward": reward_value,
                            "score": float(info.get("score", 0.0)),
                            "done": bool(done),
                            "selection_method": selection_method,
                            "raw_generation": raw_generation,
                        }
                    )

                    history.append((current_obs, executed_action, reward_value))
                    obs = next_obs
                    last_info = info
                    step_count += 1
                    if done:
                        break

                episode_return = float(last_info.get("eval_episode_return", env.episode_return))
                episode_score = float(last_info.get("score", 0.0))
                per_env_scores[env_id].append(episode_score)
                per_env_returns[env_id].append(episode_return)
                episode_records.append(
                    {
                        "stage": stage_name,
                        "env_id": env_id,
                        "episode_id": episode_id,
                        "seed": args.eval_seed + episode_id,
                        "score": episode_score,
                        "episode_return": episode_return,
                        "steps": step_count,
                        "trajectory": trajectory,
                    }
                )
        finally:
            env.close()

    episode_path = os.path.join(stage_dir, "eval_episode.jsonl")
    dump_jsonl(episode_path, episode_records)

    summary = summarize_eval(
        stage_name=stage_name,
        include_valid_actions=include_valid_actions,
        effective_action_policy=effective_action_policy,
        per_env_scores=per_env_scores,
        per_env_returns=per_env_returns,
    )
    summary["eval_episode_path"] = episode_path
    dump_json(os.path.join(stage_dir, "eval_return.json"), summary)
    return summary


def run_mode_experiment(
    args: argparse.Namespace,
    mode_name: str,
    include_valid_actions: bool,
    raw_samples: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    mode_dir = os.path.join(args.output_dir, mode_name)
    os.makedirs(mode_dir, exist_ok=True)

    train_jsonl_path, train_records = prepare_train_jsonl(
        raw_samples=raw_samples,
        mode_dir=mode_dir,
        include_valid_actions=include_valid_actions,
    )
    print(f"[Mode:{mode_name}] train_jsonl={train_jsonl_path}, records={len(train_records)}")

    tokenizer, model = load_tokenizer_llm(args.base_model_path)
    try:
        model.eval()
        pre_summary = evaluate_model(
            args=args,
            model=model,
            tokenizer=tokenizer,
            stage_name="pre_sft",
            stage_dir=os.path.join(mode_dir, "pre_sft"),
            include_valid_actions=include_valid_actions,
        )

        train_metrics = train_sft(
            args=args,
            model=model,
            tokenizer=tokenizer,
            train_records=train_records,
            work_dir=mode_dir,
        )

        model.eval()
        post_summary = evaluate_model(
            args=args,
            model=model,
            tokenizer=tokenizer,
            stage_name="post_sft",
            stage_dir=os.path.join(mode_dir, "post_sft"),
            include_valid_actions=include_valid_actions,
        )
    finally:
        del model
        torch.cuda.empty_cache()

    return {
        "mode_name": mode_name,
        "include_valid_actions": include_valid_actions,
        "train_jsonl": train_jsonl_path,
        "pre_sft": pre_summary,
        "post_sft": post_summary,
        "train_metrics": train_metrics,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[Config] output_dir={args.output_dir}")
    print(f"[Config] env_ids={args.env_ids}")
    print(f"[Config] experiment_modes={args.experiment_modes}")
    print(f"[Config] history_window={args.history_window}")

    raw_samples = collect_walkthrough_samples(args)
    
    mode_reports: List[Dict[str, Any]] = []
    for mode_name in args.experiment_modes:
        include_valid_actions = mode_name == "with_valid_actions"
        mode_report = run_mode_experiment(
            args=args,
            mode_name=mode_name,
            include_valid_actions=include_valid_actions,
            raw_samples=raw_samples,
        )
        mode_reports.append(mode_report)

    print("[Summary]")
    for report in mode_reports:
        pre_score = report["pre_sft"]["overall"]["score_mean"]
        post_score = report["post_sft"]["overall"]["score_mean"]
        pre_return = report["pre_sft"]["overall"]["return_mean"]
        post_return = report["post_sft"]["overall"]["return_mean"]
        print(
            f"  - {report['mode_name']}: "
            f"score_mean {pre_score:.3f} -> {post_score:.3f}, "
            f"return_mean {pre_return:.3f} -> {post_return:.3f}"
        )

if __name__ == "__main__":
    main()

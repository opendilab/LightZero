from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import re
import torch
import torch.distributed as dist
from vllm import SamplingParams
from ding.utils import build_logger
import random
import numpy as np
import math

_FMT_RE = re.compile(
    r'^\s*Reasoning:\s*(?P<reason>[\s\S]*?)\nAction:\s*(?P<action>[^\n\r]+)\s*$',
    flags=re.IGNORECASE
)
def _format_reward(text: str) -> int:
    """
    Return 1 if the output strictly matches:
      Reasoning: <any, may contain newlines>
      Action: <one line>
    Otherwise 0.
    """
    if not isinstance(text, str):
        return 0

    t = text.replace("\r\n", "\n").replace("\r", "\n").strip()

    m = _FMT_RE.match(t)
    if m is None:
        return 0

    if len(re.findall(r'Reasoning:', t, flags=re.IGNORECASE)) != 1:
        return 0
    if len(re.findall(r'Action:', t, flags=re.IGNORECASE)) != 1:
        return 0

    # Action 必须非空（regex 已经用 + 保证非空，这里再保险）
    if m.group("action").strip() == "":
        return 0

    return 1



def unique_dicts_hash(lst):
    import hashlib
    import pickle
    seen = set()
    res = []
    for d in lst:
        b = pickle.dumps(d)
        h = hashlib.md5(b).hexdigest()

        if h not in seen:
            seen.add(h)
            res.append(d)
    return res

class DataProcessor:
    """
      - build_llm_prompt / build_chat_context
      - priorzero_batch -> samples
      - (use_cot) 批量生成 prefix_cot
      - vLLM 计算 action prior score（prompt_logprobs）
      - samples -> Dataset/Dataloader（collate_fn 做 pack）
    """

    def __init__(self, rank, world_size, vllm_engine, strategy, model_path, exp_name=None, instance_name="vllm_output"):
        self.vllm_engine = vllm_engine
        self.strategy = strategy
        self.args = getattr(strategy, "args", None)
        
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, padding_side="left"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.use_cot = self.args.use_cot
        self.prompt_max_len = self.args.prompt_max_len
        self.generate_max_len = self.args.generate_max_len
        self.temperature = self.args.temperature
        self.top_p = self.args.top_p
        self.vllm_enable_sleep = self.args.vllm_enable_sleep
        self.reduction = self.args.reduction
        self.rank = rank
        self.world_size = world_size
        self.output_step = 0
        self.llm_prior_with_cot = False
        
        from collections import deque
        self.episode_output = []

        # Running statistics for advantage normalization
        self.value_running_mean = 0.0
        self.value_running_std = 1.0
        self.value_count = 0
        self.running_momentum = 0.99  # EMA momentum for running statistics
        
        self.global_batch_advantages = []

        if self.rank == 0:
            self._logger, _ = build_logger(
                path=f'./{exp_name}/log/{instance_name}', name=instance_name, need_tb=False
            )
        
        if self.args.value_norm_cfg.enable_stability_optimizer:
            from models.stability_optimizer import AdaptiveValueNormalizer
            self.value_normalizer = AdaptiveValueNormalizer(
                init_momentum=self.args.value_norm_cfg.value_norm_init_momentum,
                final_momentum=self.args.value_norm_cfg.value_norm_final_momentum,
                warmup_steps=self.args.value_norm_cfg.value_norm_warmup_steps,
                clip_method=self.args.value_norm_cfg.value_norm_clip_method,
                clip_percentile=self.args.value_norm_cfg.value_norm_clip_percentile,
                min_std=1e-6,
                history_size=self.args.value_norm_cfg.value_norm_history_size,
            )
        else:
            self.value_normalizer = None
    
    def get_system_prompt(self):
        """
        系统提示词：纯文本指令，定义角色、目标和严格的输出协议。
        """
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
        history: Optional[List[Tuple[str, str, float]]] = None, 
        current_obs: Optional[str] = None, 
        valid_actions: Optional[List[str]] = None
    ) -> str:
        """
        用户提示词：注入历史和当前状态，并触发输出。
        """
        prompt_parts = []
        user_prompt_dict = self.args.user_prompt_dict
        if history and len(history) > 0:
            prompt_parts.append("=== GAME HISTORY ===")
            for i, (obs, action, reward) in enumerate(history, start=1):
                prompt_parts.append(f"Step {i}:")
                prompt_parts.append(f"Observation: {obs.strip()}")
                prompt_parts.append(f"Action: {action.strip()}")
                if user_prompt_dict.history_with_reward:
                    prompt_parts.append(f"Reward: {reward}")
            prompt_parts.append("") # 空行分隔

        prompt_parts.append("=== CURRENT OBSERVATION ===")
        prompt_parts.append(current_obs.strip())
        if user_prompt_dict.observation_with_valid_actions:
            if valid_actions and len(valid_actions) > 0:
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
                {"role": "user", "content": user_prompt}
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

    def build_llm_samples(self,
        raw_obs_list: List[List[str]],
        history_obs_list: List[List[List[Tuple[str, str, float]]]],
        llm_prior_per_tok_list: Optional[List[List[Any]]] = None,
        pred_values: Optional[torch.Tensor] = None,   # [B, T-1] 
        target_values: Optional[torch.Tensor] = None,   # [B, T-1] 
        cot_prefix_list: Optional[List[List[str]]] = None,  # CoT reuse optimization
        llm_action_list: Optional[List[List[str]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Build training samples from collected data.

        Args:
            raw_obs_list: Raw observations
            history_obs_list: History observations
            llm_prior_per_tok_list: LLM prior per token from collect phase
            target_values: Target values for advantage calculation
            cot_prefix_list: CoT prefixes from collect phase (CoT reuse optimization)

        Returns:
            List of sample dictionaries
        """
        samples: List[Dict[str, Any]] = []
        B = len(raw_obs_list)
        if B == 0:
            return samples
        T = len(raw_obs_list[0])

        for b in range(B):
            for t in range(T - 1):
                current_obs = raw_obs_list[b][t]
                current_hist = history_obs_list[b][t]
                
                true_action = llm_action_list[b][t+1]
                rollout_logprob = llm_prior_per_tok_list[b][t+1]['rollout_action_logprob'][true_action]
                full_ids = llm_prior_per_tok_list[b][t+1]['full_ids'][true_action]
                label_ids = llm_prior_per_tok_list[b][t+1]['label_ids'][true_action]
                valid_actions = list(llm_prior_per_tok_list[b][t+1]['rollout_action_logprob'].keys())
                if 'go' in valid_actions:
                    valid_actions.remove('go')
                
                instruction = self.get_user_prompt(
                    history=current_hist,
                    current_obs=current_obs,
                    valid_actions=valid_actions
                )
                prompt = self.build_chat_context(instruction)
                
                if len(label_ids) == 0:
                    continue
                target_value = None
                if target_values is not None:
                    target_value = float(target_values[b][t].item())
                
                pred_value = None
                if pred_values is not None:
                    pred_value = float(pred_values[b][t].item())

                # CoT reuse optimization: get CoT prefix from stored data
                prefix_cot = None
                if self.use_cot and cot_prefix_list is not None:
                    prefix_cot = cot_prefix_list[b][t+1]

                samples.append(
                    {
                        "instruction": instruction,
                        "prompt": prompt,
                        "target": true_action,
                        "pred_value": pred_value,
                        "target_value": target_value,
                        "rollout_logprob": rollout_logprob,  # Reinforce++ ratio 需要
                        "prefix_cot": prefix_cot,  # CoT reuse optimization
                        "full_ids": full_ids, 
                        "label_ids": label_ids,
                    }
                )
        return samples

    def make_llm_train_samples(self, priorzero_batch, ddp: bool = False, max_samples: int = 32) -> List[Dict[str, Any]]:
        """
        Convert PriorZero batch to LLM training samples.

        Args:
            priorzero_batch: Tuple of  (raw_obs_list, history_obs_list, llm_prior_per_tok_list, target_value, pred_value, cot_prefix_list, llm_action_list
                            CoT prefix list is added for CoT reuse optimization.

        Returns:
            Tuple of (input_ids, attention_mask, action_mask, advantages, rollout_logprob)
        """
        # Support both 7-element (legacy) and 8-element (with action_list) batch formats
        if len(priorzero_batch) == 8:
            raw_obs_list, history_obs_list, llm_prior_per_tok_list, target_value, pred_value, cot_prefix_list, llm_action_list, _action_list = priorzero_batch
        else:
            raw_obs_list, history_obs_list, llm_prior_per_tok_list, target_value, pred_value, cot_prefix_list, llm_action_list = priorzero_batch

        assert len(raw_obs_list) == len(history_obs_list) == len(llm_prior_per_tok_list) == len(target_value) == len(pred_value) == len(cot_prefix_list) == len(llm_action_list), \
            f"Batch size mismatch: raw_obs={len(raw_obs_list)}, history_obs={len(history_obs_list)}, llm_prior_per_tok={len(llm_prior_per_tok_list)}, \
                target_value={len(target_value)}, pred_value={len(pred_value)}, cot_prefix={len(cot_prefix_list)}, llm_action={len(llm_action_list)}"

        # Build samples with CoT prefixes
        samples = self.build_llm_samples(
            raw_obs_list, history_obs_list, llm_prior_per_tok_list, pred_value, target_value, cot_prefix_list, llm_action_list
        )
        random.Random(0).shuffle(samples)
        
        
        def _select_samples_with_unique_priority(sample_list, keep_n):
            """优先取去重后的样本；如果去重后不够，则按原始顺序补齐。"""
            if len(sample_list) < keep_n:
                return None
            unique_samples = unique_dicts_hash(sample_list)
            if len(unique_samples) >= keep_n:
                return unique_samples[:keep_n]
            remain = keep_n - len(unique_samples)
            selected = unique_samples + sample_list[:remain]
            return selected[:keep_n]
        
        if ddp:
            gathered_samples = [None for _ in range(self.world_size)]
            dist.all_gather_object(gathered_samples, samples)
            
            global_samples = []
            for rank_samples in gathered_samples:
                if rank_samples is not None:
                    global_samples.extend(rank_samples)
            global_max_samples = self.world_size * max_samples
            selected_global_samples = _select_samples_with_unique_priority(global_samples, global_max_samples)

            if selected_global_samples is None:
                print(
                    f"[Rank {self.rank}] insufficient global samples after all_gather: "
                    f"total_global={len(global_samples)} < required={global_max_samples}"
                )
                return False, [global_samples]

            start = self.rank * max_samples
            end = (self.rank + 1) * max_samples
            real_samples = selected_global_samples[start:end]
            print(
                f"[Rank {self.rank}] local={len(samples)}, gathered_total={len(global_samples)}, "
                f"selected_global={len(selected_global_samples)}, take={start}:{end}"
            )
        else:
            selected_samples = _select_samples_with_unique_priority(samples, max_samples)
            if selected_samples is None:
                return False, [samples]
            
            per_rank = len(selected_samples) // self.world_size
            start = self.rank * per_rank
            end = (self.rank + 1) * per_rank if self.rank != self.world_size - 1 else len(selected_samples)
            print(f"[Rank {self.rank}] process {start}: {end} samples. Total {len(selected_samples)} samples collected by Rank 0.")
            real_samples = selected_samples[start:end]
        
        if self.use_cot:
            targets_only = [s["prefix_cot"] + " " + s["target"] + self.tokenizer.eos_token for s in real_samples]
            if self.args.reward_func.format_reward:
                fmt_rewards = torch.tensor([_format_reward(t) for t in targets_only])
            else:
                fmt_rewards = None
        else:
            targets_only = ["Action: " +s["target"] + self.tokenizer.eos_token for s in real_samples]
            fmt_rewards = None

        full_ids_list = [s['full_ids'] for s in real_samples]
        tgt_ids_list = [s['label_ids'] for s in real_samples]
        assert self.tokenizer.batch_decode(tgt_ids_list) == targets_only, "Decoded label ids do not match targets_only. Please check the tokenizer and data processing logic."
        inputs = self.tokenizer.pad({"input_ids": full_ids_list}, padding=True, return_tensors="pt")
        labels = torch.full_like(inputs.input_ids, -100)
        for i, tgt_ids in enumerate(tgt_ids_list):
            tgt_len = len(tgt_ids)
            labels[i, -tgt_len:] = inputs.input_ids[i, -tgt_len:]
        action_mask_full = (labels != -100).long()
        max_tgt_len = max(len(t) for t in tgt_ids_list)
        action_mask = action_mask_full[:, -max_tgt_len:] 
        log_status_tmp = {}
        log_status = []
        
        if fmt_rewards is not None:
            fmt_weight = self.args.reward_func.format_param.format_weight
            assert 0.0 <= fmt_weight < 1.0, f"format_weight should be in [0, 1), but got {fmt_weight}"
            log_status_tmp['fmt_rewards'] = fmt_rewards.tolist()
        
        # t 时刻的 target_value = td_step 步真实 r 的折扣和 + boostrap( t + td_step) 的 v
        target_value = torch.tensor([s["target_value"] for s in real_samples], dtype=torch.float32)
        # t 时刻的 pred_value = boostrap( t ) 的 v
        pred_value = torch.tensor([s["pred_value"] for s in real_samples], dtype=torch.float32)
        advantage = target_value - pred_value
        
        if self.args.advantage_type == "advantage":
            advantage = advantage
            log_status_tmp["value_advantage"] = advantage.tolist()
            if fmt_rewards is not None:
                advantage = (1 - fmt_weight) * advantage + fmt_weight * fmt_rewards
                log_status_tmp["final_advantage"] = advantage.tolist()
                

        elif self.args.advantage_type == "advantage_batch_norm":
            # Legacy implementation: batch normalization (not recommended)
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
            log_status_tmp["value_advantage"] = advantage.tolist()
            
            if fmt_rewards is not None:
                advantage = (1 - fmt_weight) * advantage + fmt_weight * fmt_rewards
                log_status_tmp["final_advantage"] = advantage.tolist()

        elif self.args.advantage_type == "advantage_global_batch_norm":
            # self.global_batch_advantages
            self.global_batch_advantages += advantage.tolist()
            advantage = (advantage - np.mean(self.global_batch_advantages)) / (np.std(self.global_batch_advantages) + 1e-8)
            log_status_tmp["value_advantage"] = advantage.tolist()
            
            if fmt_rewards is not None:
                advantage = (1 - fmt_weight) * advantage + fmt_weight * fmt_rewards
                log_status_tmp["final_advantage"] = advantage.tolist()
        elif self.args.advantage_type == "advantage_running_norm":
            if self.value_normalizer is not None:
                raw_mean = advantage.mean().item()
                raw_std = advantage.std().item()
                raw_min = advantage.min().item()
                raw_max = advantage.max().item()
                batch_size = advantage.numel()
                
                advantage, norm_stats = self.value_normalizer.normalize(
                    advantage,
                    clip_values=True,
                    return_stats=True
                )
                
                norm_min = advantage.min().item()
                norm_max = advantage.max().item()
                norm_mean = advantage.mean().item()
                norm_std = advantage.std().item()
                
                if self.rank == 0 and self.value_normalizer.update_count % 10 == 0:
                    print(
                        f"[Value Norm] step={self.value_normalizer.update_count} | "
                        f"batch_size={batch_size} | "
                        f"running: mean={norm_stats['running_mean']:.3f}, std={norm_stats['running_std']:.3f} | "
                        f"batch: mean={norm_stats['batch_mean']:.3f}, std={norm_stats['batch_std']:.3f} | "
                        f"raw: min={raw_min:.3f}, max={raw_max:.3f} | "
                        f"norm: min={norm_min:.3f}, max={norm_max:.3f} | "
                        f"clipped={norm_stats['clipped_count']}/{norm_stats['total_count']} | "
                        f"momentum={norm_stats['momentum']:.3f}"
                    )
            else:
                batch_mean = advantage.mean().item()
                batch_std = advantage.std().item()
                batch_min = advantage.min().item()
                batch_max = advantage.max().item()
                batch_size = advantage.numel()

                if self.value_count == 0:
                    self.value_running_mean = batch_mean
                    self.value_running_std = max(batch_std, 1e-8)  # Avoid zero std
                else:
                    self.value_running_mean = (
                        self.running_momentum * self.value_running_mean +
                        (1 - self.running_momentum) * batch_mean
                    )
                    self.value_running_std = (
                        self.running_momentum * self.value_running_std +
                        (1 - self.running_momentum) * max(batch_std, 1e-8)
                    )

                self.value_count += 1
                advantage = (advantage - self.value_running_mean) / (self.value_running_std + 1e-8)
                
                norm_min = advantage.min().item()
                norm_max = advantage.max().item()
                norm_mean = advantage.mean().item()
                norm_std = advantage.std().item()

                if self.rank == 0 and self.value_count % 10 == 0:
                    print(
                        f"[Advantage Running Norm] step={self.value_count} | "
                        f"batch_size={batch_size} | "
                        f"running: mean={self.value_running_mean:.3f}, std={self.value_running_std:.3f} | "
                        f"batch: mean={batch_mean:.3f}, std={batch_std:.3f} | "
                        f"raw: min={batch_min:.3f}, max={batch_max:.3f} | "
                        f"norm: min={norm_min:.3f}, max={norm_max:.3f}"
                    )
                    
            log_status_tmp["value_advantage"] = advantage.tolist()
            if fmt_rewards is not None:
                advantage = (1 - fmt_weight) * advantage + fmt_weight * fmt_rewards
                log_status_tmp["final_advantage"] = advantage.tolist()
        else:
            raise ValueError(f"Unknown advantage_type: {self.args.advantage_type}")
        
        log_status = [
            {k: log_status_tmp[k][i] for k in log_status_tmp.keys()} for i in range(len(log_status_tmp['value_advantage']))
        ]
        
        for i, s in enumerate(real_samples):
            if len(s['rollout_logprob']) != len(s['label_ids']):
                raise ValueError(
                    f"Length mismatch at sample {i}: "
                    f"len(rollout_logprob)={len(s['rollout_logprob'])}, "
                    f"len(label_ids)={len(s['label_ids'])}, "
                    f"target={repr(s['target'])}"
                )
        old_seq_max_len = max([len(s['rollout_logprob']) for s in real_samples])
        rollout_logprob = torch.zeros(len(real_samples), old_seq_max_len, dtype=torch.float32)
        for idx in range(len(real_samples)):
            logprob_token_list = real_samples[idx]['rollout_logprob']
            rollout_logprob[idx, -len(logprob_token_list):] = torch.tensor(logprob_token_list, dtype=torch.float32)
        
        return True, (inputs.input_ids, inputs.attention_mask, action_mask, advantage, rollout_logprob, log_status)
        
    @torch.no_grad()
    def _build_cot_prefix_texts(self, all_user_prompts: List[str]) -> List[str]:
        """
        生成CoT推理前缀。
        优化: 使用较短的max_tokens(128)和stop条件以减少不必要的生成。
        从最后一次出现的 "Action:" 截断出 prefix（包含 Action: 和其后的空格位置）。
        返回 prefix_cot_list，与 all_user_prompts 等长。
        """
        cot_sampling_params = SamplingParams(
            temperature=1.0,
            top_p=1.0,
            max_tokens=self.generate_max_len, 
            stop=["\n\n"],
            # stop=["Action:", "\n\n"]
            include_stop_str_in_output=True,
            logprobs=None,
            prompt_logprobs=None,
        )

        all_context_texts = [self.build_chat_context(p) for p in all_user_prompts]
        context_token_ids = self.tokenizer(
            all_context_texts,
            add_special_tokens=False,
            max_length=self.prompt_max_len,
            padding=False,
            truncation=True,
        )["input_ids"]

        self.vllm_engine.add_requests(sampling_params=cot_sampling_params, prompt_token_ids=context_token_ids)
        cot_outputs = self.vllm_engine.get_responses()

        prefix_cot_list, full_output = [], []
        reasoning_pattern = re.compile(r"Reasoning\s*:", re.IGNORECASE)
        action_pattern = re.compile(r"Action\s*:", re.IGNORECASE)
        
        for output in cot_outputs:
            gen_text = output.outputs[0].text
            full_output.append(gen_text)
            # TODO 这里是否要清洗数据？清洗过后，计算prior先验的时候比较正常，但是format_reward几乎没用
            # if not reasoning_pattern.search(gen_text):
            #     prefix_cot_list.append("Action:")
            #     continue
            action_match = action_pattern.search(gen_text)
            if action_match:
                end_index = action_match.end()
                prefix_piece = gen_text[:end_index].strip()
            else:
                # prefix_piece = gen_text.strip() 
                prefix_piece = gen_text.strip() + "\nAction:"
                
            prefix_cot_list.append(prefix_piece)

        return prefix_cot_list, full_output
    
    @torch.no_grad()
    def get_llm_prior(
        self,
        states: List[str],
        valid_actions_list: List[List[str]],
        histories: Optional[List[List[Tuple[str, str, float]]]] = None,
        return_cot: bool = False,  # CoT reuse optimization: return CoT prefixes
    ) -> List[Any]:
        """
        Get LLM prior scores for actions.

        Args:
            states: List of current state observations
            valid_actions_list: List of valid actions for each state
            histories: List of history observations
            return_cot: If True, return CoT prefixes for reuse (optimization)

        Returns:
            If return_cot=False: (llm_prior_per_seq, llm_prior_per_tok)
            If return_cot=True: (llm_prior_per_seq, llm_prior_per_tok, prefix_cots)
        """
        prompt_list = []
        assert len(states) == len(histories) == len(valid_actions_list)
        for state, history, valid_actions in zip(states, histories, valid_actions_list):
            prompt = self.get_user_prompt(current_obs=state, history=history, valid_actions=valid_actions)
            prompt_list.append(prompt)

        if self.use_cot:
            prefix_cots, full_output = self._build_cot_prefix_texts(prompt_list)
        else:
            prefix_cots = [None] * len(prompt_list)
            full_output = None

        all_prompts = []
        all_labels = []
        all_prefix_cots = []
        all_env_indices = []

        for env_idx, (prompt, actions, prefix) in enumerate(zip(prompt_list, valid_actions_list, prefix_cots)):
            actions2 = actions if "go" in actions else (actions + ["go"])   # 确保环境使用的动作都在valid actions里有对应的logprob
            for action in actions2:
                all_prompts.append(prompt)
                all_labels.append(action)
                all_prefix_cots.append(prefix)
                all_env_indices.append(env_idx)
        assert len(all_prompts) == len(all_labels) == len(all_prefix_cots) == len(all_env_indices)
        
        scores, rollout_action_logprob, full_ids, label_ids = self._score_labels_with_prompt_logprobs(all_prompts, all_labels, all_prefix_cots)
        assert len(all_prompts) == len(scores) == len(rollout_action_logprob) == len(full_ids) == len(label_ids)
        
        llm_prior_per_seq, llm_prior_per_tok = [],[], 
        cur_env_idx = 0
        seq_dict = {}
        tok_dict = {'rollout_action_logprob': {}, 'full_ids': {}, 'label_ids': {}}
        
        for idx, (env_idx, prompt, label, prefix_cot) in enumerate(zip(all_env_indices, all_prompts, all_labels, all_prefix_cots)):
            if env_idx != cur_env_idx:
                llm_prior_per_seq.append(seq_dict)
                llm_prior_per_tok.append(tok_dict)
                seq_dict = {}
                tok_dict = {'rollout_action_logprob': {}, 'full_ids': {}, 'label_ids': {}}
                cur_env_idx = env_idx
            
            seq_dict[label] = scores[idx]
            tok_dict['rollout_action_logprob'][label] = rollout_action_logprob[idx]  
            tok_dict['full_ids'][label] = full_ids[idx]
            tok_dict['label_ids'][label] = label_ids[idx]
            tok_dict['prompt'] = prompt
            tok_dict['prefix_cot'] = prefix_cot
            tok_dict['current_obs'] = states[env_idx]
            tok_dict['history'] = histories[env_idx]
            
        if len(seq_dict) > 0:
            llm_prior_per_seq.append(seq_dict)
            llm_prior_per_tok.append(tok_dict)  

        if self.use_cot:
            self.episode_output.append({
                "Instruction": prompt_list[0],
                "Response": full_output[0],
                "llm_prior_per_seq": llm_prior_per_seq[0]
            })
        # CoT reuse optimization: return CoT prefixes if requested
        if return_cot:
            return llm_prior_per_seq, llm_prior_per_tok, prefix_cots
        else:
            return llm_prior_per_seq, llm_prior_per_tok

    @torch.no_grad()
    def _score_labels_with_prompt_logprobs(self, all_prompts: List[str], all_labels: List[str], all_prefix_cots: List[str]) -> List[float]:
        assert len(all_prompts) == len(all_labels) == len(all_prefix_cots)
        sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=1,                       
            include_stop_str_in_output=True,
            logprobs=None,
            prompt_logprobs=1,
        )

        all_context_texts = [self.build_chat_context(p) for p in all_prompts]
        context_ids = self.tokenizer(all_context_texts, add_special_tokens=False, max_length=self.prompt_max_len - self.generate_max_len - 20, padding=False, truncation=True)["input_ids"]

        if self.use_cot:
            label_texts = [pc + " " + l + self.tokenizer.eos_token for pc, l in zip(all_prefix_cots, all_labels)]
            label_texts_no_cots =  [" " + l + self.tokenizer.eos_token for l in all_labels]
        else:
            label_texts = ["Action: " + l + self.tokenizer.eos_token for l in all_labels]
            label_texts_no_cots = label_texts
            
        label_ids = self.tokenizer(label_texts, add_special_tokens=False, padding=False, truncation=False)["input_ids"]
        label_ids_no_cots = self.tokenizer(label_texts_no_cots, add_special_tokens=False, padding=False, truncation=False)["input_ids"]
        
        for idx, (l_ids, l_ids_not_cot) in enumerate(zip(label_ids, label_ids_no_cots)):
            len_not_cot = len(l_ids_not_cot)
            if l_ids[-len_not_cot:] != l_ids_not_cot:
                raise ValueError(f"Label IDs mismatch: with CoT {l_ids[-len_not_cot:]}, without CoT {l_ids_not_cot}, label_text: {label_texts[idx]}")
            
        full_ids = [c + l for c, l in zip(context_ids, label_ids)]
        p_lens = [len(x) for x in context_ids]
        l_lens = [len(x) for x in label_ids]
        l_no_cots_lens = [len(x) for x in label_ids_no_cots]

        self.vllm_engine.add_requests(sampling_params=sampling_params, prompt_token_ids=full_ids)
        outs = self.vllm_engine.get_responses()

        scores = []
        rollout_action_logprob = []
        nan_found = False
        for i, (out, ids, p_len, l_len, l_no_cots_len) in enumerate(zip(outs, full_ids, p_lens, l_lens, l_no_cots_lens)):            
            prompt_logprobs = getattr(out, "prompt_logprobs", None)
            token_lps = []
            
            for j in range(1, len(ids)):
                tok_id = ids[j]
                lp_dict = prompt_logprobs[j]
                
                assert tok_id in lp_dict
                token_lps.append(lp_dict[tok_id].logprob)

            if not token_lps:
                scores.append(float("-inf"))
                rollout_action_logprob.append([])
            else:
                assert l_no_cots_len <= l_len
                if self.llm_prior_with_cot:
                    target_lps = token_lps[-l_len:]
                else:
                    target_lps = token_lps[-l_no_cots_len:]
                denom = len(target_lps)

                score = sum(target_lps) if self.reduction == "sum" else sum(target_lps) / denom
                scores.append(score)
                
                if (not nan_found) and math.isnan(score):
                    vllm_returned_nan = any(math.isnan(x) for x in target_lps)
                    token_level_debug = []
                    for t_id, t_lp in zip(ids[1:], token_lps):
                        token_level_debug.append(f"TokenID: {t_id} -> LogProb: {t_lp} {'(NaN HERE!)' if math.isnan(t_lp) else ''}")

                    nan_found = True
                    nan_debug_dump = (
                        f"\n{'='*20} [NaN DEBUG REPORT] {'='*20}\n"
                        f"Sample Index (i): {i}\n"
                        f"Reason: {'vLLM returned NaN logprob' if vllm_returned_nan else 'Math error during sum/div'}\n\n"
                        f"--- Text Info ---\n"
                        f"Prompt: ...{repr(all_prompts[i])}\n"
                        f"Label Action: {repr(all_labels[i])}\n"
                        f"Prefix CoT: {repr(all_prefix_cots[i])}\n\n"
                        f"--- Numerical Info (Copy this to reproduce) ---\n"
                        f"Full Input Token IDs (full_ids[{i}]): {ids}\n"
                        f"Context Length (p_len): {p_len}\n"
                        f"Label Length (l_len): {l_len}\n"
                        f"Target Length (l_no_cots_len): {l_no_cots_len}\n\n"
                        f"--- Critical Calculation Data ---\n"
                        f"Head 10 Token IDs: {ids[1:11]}\n"
                        f"LogProbs List: {token_lps[:10]}\n"
                        f"Detailed Mapping:\n" + "\n".join(token_level_debug[:10]) + "\n\n"
                        
                        f"Tail Token IDs: {ids[-l_len - 10: -l_len]}\n"
                        f"LogProbs List: {token_lps[-l_len - 10: -l_len]}\n"
                        f"Detailed Mapping:\n" + "\n".join(token_level_debug[-l_len - 10: -l_len]) + "\n\n"
                        
                        f"Target Token IDs: {ids[-l_no_cots_len:]}\n"
                        f"LogProbs List: {target_lps}\n"
                        f"Detailed Mapping:\n" + "\n".join(token_level_debug[-l_no_cots_len:]) + "\n"
                        f"{'='*60}\n"
                    )
                rollout_action_logprob.append(token_lps[-l_len:])

        if self.rank == 0:
            if nan_found:
                self._logger.info(nan_debug_dump)
            
        return scores, rollout_action_logprob, full_ids, label_ids

    @torch.no_grad()
    def get_llm_output_log(self, wm_train_iter: int = 0, llm_train_iter: int = 0):
        if self.rank != 0:
            return 
        
        self._logger.info(
            f"\n{'='*80}\n"
            f"[LLM Output Log] WM Iter: {wm_train_iter} | LLM Iter: {llm_train_iter}\n"
            f"{'='*80}"
        )
        
        for i, tmp_dict in enumerate(self.episode_output[:15]):
            instruction = tmp_dict["Instruction"]
            response = tmp_dict["Response"]
            llm_prior = tmp_dict["llm_prior_per_seq"]
            
            self._logger.info(
                f"\n{'-'*80}\n"
                f"[Step {i}]\n"
                f"{'-'*80}\n"
                f"Instruction:\n{instruction}\n\n"
                f"Response:\n{response}\n\n"
                f"Action Probabilities:"
            )

            action_probs = {a: math.exp(float(lp)) for a, lp in llm_prior.items() if lp is not None and math.isfinite(float(lp))}
            all_prob = sum(action_probs.values())
            
            for action, prob in sorted(action_probs.items(), key=lambda x: x[1], reverse=True):
                self._logger.info(f"  {action:30s} | unnorm={prob:.6f} | norm={(prob / all_prob):.6f}")
            self._logger.info(f"  {'<other>':30s} | unnorm={1-all_prob:.6f}")
        self.episode_output = []

        
    def clear_statis(self):
        if self.value_normalizer is not None:
            self.value_normalizer.clear()
        self.global_batch_advantages.clear()
        
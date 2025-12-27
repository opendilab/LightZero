from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import re
import torch
import torch.distributed as dist
from vllm import SamplingParams
from ding.utils import build_logger

class DataProcessor:
    """
      - build_llm_prompt / build_chat_context
      - priorzero_batch -> samples
      - (use_cot) 批量生成 prefix_cot
      - vLLM 计算 action prior score（prompt_logprobs）
      - samples -> Dataset/Dataloader（collate_fn 做 pack）
    """

    def __init__(self, rank, vllm_engine, strategy, model_path, exp_name=None, instance_name="vllm_output"):
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
        self.temperature = self.args.temperature
        self.top_p = self.args.top_p
        self.vllm_enable_sleep = self.args.vllm_enable_sleep
        self.reduction = self.args.reduction
        self.rank = rank
        self.output_step = 0
        
        from collections import deque
        self.vllm_output = deque(maxlen=10)
        
        if self.rank == 0:
            self._logger, _ = build_logger(
                path=f'./{exp_name}/log/{instance_name}', name=instance_name, need_tb=False
            )

    def build_llm_prompt(self, current_obs: str, history: Optional[List[Tuple[str, str, float]]] = None) -> str:
        prompt_parts = []
        prompt_parts.append(
            "You are an expert player in a text-based adventure game. "
            "Your goal is to maximize the score by choosing the best possible next action. "
            "You must choose exactly ONE best next action."
        )
        if history is not None and len(history) > 0:
            history = list(history)
            prompt_parts.append("\n=== Recent History ===")

            for i, (obs, action, reward) in enumerate(history, start=1):  
                obs_str = obs
                prompt_parts.append(f"Step {i}:")
                prompt_parts.append(f"  Observation: {obs_str}")
                prompt_parts.append(f"  Action: {action}")
                prompt_parts.append(f"  Reward: {reward}")

        prompt_parts.append("\n=== Current Situation ===")
        prompt_parts.append(current_obs)

        if self.use_cot:
            prompt_parts.append(
                "\n=== Task ===\n"
                "You must produce TWO parts in order: (1) Reasoning, then (2) Action.\n\n"
                "1) Reasoning:\n"
                "- Perform a detailed reasoning process based ONLY on the current state and the recent interaction history.\n"
                "- Analyze what environment or situation you are currently in.\n"
                "- Identify what actions are available or valid at this step, and the relevant constraints.\n"
                "- You may discuss observations, uncertainties, and implications of different possibilities.\n"
                "- IMPORTANT: Do NOT state, imply, or reveal which action will be chosen, and the reasoning section MUST output exactly in the following format: Reasoning:<REASONING>.\n\n"
                "2) Action:\n"
                "- After finishing the reasoning, output exactly ONE line in the following format:\nAction: <ACTION>\n"
                "Your output MUST strictly follow this format: \nReasoning: <your reasoning content>\nAction: <the chosen action>"
            )
        else:
            prompt_parts.append(
                "\n=== Task ===\n"
                "Analyze the recent history and the current situation, and decide on the SINGLE best next action."
                "Please keep the output concise, avoiding any other content.\n\n"
            )
        return "\n".join(prompt_parts)

    def build_chat_context(self, user_prompt: str) -> str:
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": user_prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def build_llm_samples(self,
        raw_obs_list: List[List[str]],
        history_obs_list: List[List[List[Tuple[str, str, float]]]],
        action_logprob_list: Optional[List[List[Any]]] = None,
        target_values: Optional[torch.Tensor] = None,   # [B, T-1] 的 G_t
    ) -> List[Dict[str, Any]]:
        
        samples: List[Dict[str, Any]] = []
        B = len(raw_obs_list)
        if B == 0:
            return samples
        T = len(raw_obs_list[0])

        for b in range(B):
            for t in range(T - 1):
                current_obs = raw_obs_list[b][t]
                current_hist = history_obs_list[b][t]
                next_hist = history_obs_list[b][t + 1]

                _, true_action, reward_value = next_hist[-1]
                if not true_action:
                    continue

                instruction = self.build_llm_prompt(
                    current_obs=current_obs,
                    history=current_hist,
                )
                prompt = self.build_chat_context(instruction)
                old_logprob = None
                if action_logprob_list is not None:
                    old_logprob = action_logprob_list[b][t + 1][true_action]

                target_value = None
                if target_values is not None:
                    target_value = float(target_values[b][t].item())

                samples.append(
                    {
                        "instruction": instruction,
                        "prompt": prompt,
                        "target": true_action,
                        "reward": float(reward_value) if reward_value is not None else 0.0,
                        "target_value": target_value,           
                        "old_logprob": old_logprob,  # Reinforce++ ratio 需要
                    }
                )
        return samples

    def make_llm_train_samples(self, priorzero_batch) -> List[Dict[str, Any]]:
        current_batch, target_batch = priorzero_batch
        obs_batch_ori, action_batch, target_action_batch, mask_batch, batch_index_tensor, weights, make_time, timestep_batch, raw_obs_list, history_obs_list, action_logprob_list = current_batch
        target_reward, target_value, target_policy = target_batch

        samples = self.build_llm_samples(raw_obs_list, history_obs_list, action_logprob_list, target_value)

        if self.use_cot:
            if self.vllm_enable_sleep:
                self.vllm_engine.wake_up()
            
            all_user_prompts = [s["instruction"] for s in samples]
            prefix_list = self._build_cot_prefix_texts(all_user_prompts)
            for s, p in zip(samples, prefix_list):
                s["prefix_cot"] = p
            
            if self.vllm_enable_sleep:
                self.vllm_engine.sleep()

        if self.use_cot:
            prompts_only = [s["prompt"] + s["prefix_cot"] + " " for s in samples]
        else:
            prompts_only = [s["prompt"] for s in samples]

        targets_only = [s["target"] + self.tokenizer.eos_token for s in samples]

        prompts_ids_list = self.tokenizer(prompts_only, add_special_tokens=False, truncation=True, max_length=self.prompt_max_len - 20)["input_ids"]
        tgt_ids_list = self.tokenizer(targets_only, add_special_tokens=False, truncation=True)["input_ids"]

        full_ids_list = [p + t for p, t in zip(prompts_ids_list, tgt_ids_list)]
        inputs = self.tokenizer.pad({"input_ids": full_ids_list}, padding=True, return_tensors="pt")

        labels = inputs.input_ids.clone()
        labels[inputs.attention_mask == 0] = -100

        for row, p_ids in enumerate(prompts_ids_list):
            pad_len = int((inputs.attention_mask[row] == 0).sum().item())
            real_prompt_len = pad_len + len(p_ids)
            labels[row, :real_prompt_len] = -100

        action_mask_full = (labels != -100).long()
        max_tgt_len = max(len(t) for t in tgt_ids_list)
        action_mask = action_mask_full[:, -max_tgt_len:] 

        gt = torch.tensor(
            [s["target_value"] if s["target_value"] is not None else s["reward"] for s in samples],
            dtype=torch.float32,
        )
        old_seq_max_len = max([len(s['old_logprob']) for s in samples])
        old_logprob = torch.zeros(len(samples), old_seq_max_len, dtype=torch.float32)
        for idx in range(len(samples)):
            logprob_token_list = samples[idx]['old_logprob']
            old_logprob[idx, -len(logprob_token_list):] = torch.tensor(logprob_token_list, dtype=torch.float32)

        return inputs.input_ids, inputs.attention_mask, action_mask, gt, old_logprob
        
    @torch.no_grad()
    def _build_cot_prefix_texts(self, all_user_prompts: List[str]) -> List[str]:
        """
        生成一次完整输出，从最后一次出现的 "Action:" 截断出 prefix（包含 Action: 和其后的空格位置）。
        返回 prefix_cot_list，与 all_user_prompts 等长。
        """
        cot_sampling_params = SamplingParams(
            temperature=1.0,
            top_p=1.0,
            max_tokens=self.prompt_max_len,
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

        prefix_cot_list = []
        for output in cot_outputs:
            gen_text = output.outputs[0].text

            matches = list(re.finditer(r"(?mi)^\s*Action\s*:\s*", gen_text))
            if not matches:
                matches = list(re.finditer(r"action\s*:\s*", gen_text, flags=re.IGNORECASE))

            if not matches:
                prefix_cot_list.append("")
                continue

            m = matches[-1]
            prefix_piece = gen_text[: m.end()].strip() 
            prefix_cot_list.append(prefix_piece)

        return prefix_cot_list
    
    @torch.no_grad()
    def get_llm_prior(
        self,
        states: List[str],
        valid_actions_list: List[List[str]], 
        histories: Optional[List[List[Tuple[str, str, float]]]] = None,
    ) -> List[Any]:

        all_prompts = []
        all_labels = []
        self.vllm_output.append((states[0], histories[0]))
        
        for i, actions in enumerate(valid_actions_list):  
            actions.append('go')   # 确保环境使用的动作都在valid actions里有对应的logprob
            state = states[i]
            history = histories[i]
            prompt = self.build_llm_prompt(current_obs=state, history=history)
            
            for action in actions:
                all_prompts.append(prompt)
                all_labels.append(action)
        
        scores, old_action_logprob = self._score_labels_with_prompt_logprobs(all_prompts, all_labels)
        llm_prior_per_seq, llm_prior_per_tok, idx = [],[], 0
        
        for env_id in range(len(states)):
            tmp_dict = {}
            tmp_dict2 = {}
            for action in valid_actions_list[env_id]:
                tmp_dict[action] = scores[idx]
                tmp_dict2[action] = old_action_logprob[idx]
                idx = idx + 1
            llm_prior_per_seq.append(tmp_dict)
            llm_prior_per_tok.append(tmp_dict2)
        return llm_prior_per_seq, llm_prior_per_tok

    @torch.no_grad()
    def _score_labels_with_prompt_logprobs(self, all_prompts: List[str], all_labels: List[str]) -> List[float]:
        assert len(all_prompts) == len(all_labels)
        
        if self.use_cot:
            all_prefix_cot = self._build_cot_prefix_texts(all_prompts)

        sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=1,                       
            include_stop_str_in_output=True,
            logprobs=None,
            prompt_logprobs=1,
        )

        all_context_texts = [self.build_chat_context(p) for p in all_prompts]
        if self.use_cot:
            all_context_texts = [c + pc + " " for c, pc in zip(all_context_texts, all_prefix_cot)]

        context_ids = self.tokenizer(all_context_texts, add_special_tokens=False, max_length=self.prompt_max_len - 20, padding=False, truncation=True)["input_ids"]

        label_texts = [l + self.tokenizer.eos_token for l in all_labels]
        label_ids = self.tokenizer(label_texts, add_special_tokens=False, padding=False, truncation=False)["input_ids"]

        full_ids = [c + l for c, l in zip(context_ids, label_ids)]
        p_lens = [len(x) for x in context_ids]
        l_lens = [len(x) for x in label_ids]

        self.vllm_engine.add_requests(sampling_params=sampling_params, prompt_token_ids=full_ids)
        outs = self.vllm_engine.get_responses()

        scores = []
        old_action_logprob = []
        for out, ids, p_len, l_len in zip(outs, full_ids, p_lens, l_lens):
            prompt_logprobs = getattr(out, "prompt_logprobs", None)

            token_lps = []
            for j in range(p_len, p_len + l_len):
                tok_id = ids[j]
                lp_dict = prompt_logprobs[j]
                if tok_id not in lp_dict:
                    token_lps.append(float("-inf"))
                else:
                    token_lps.append(lp_dict[tok_id].logprob)

            if not token_lps:
                scores.append(float("-inf"))
                old_action_logprob.append([])
            else:
                scores.append(sum(token_lps) if self.reduction == "sum" else sum(token_lps) / len(token_lps))
                old_action_logprob.append(token_lps)
            
        return scores, old_action_logprob

    @torch.no_grad()
    def get_llm_output_log(self):
        if self.rank != 0:
            return 
        sampling_params = SamplingParams(
            temperature=1.0,
            top_p=1.0,
            max_tokens=self.prompt_max_len,
            logprobs=None,
            prompt_logprobs=None,
        )

        all_context_texts = [self.build_chat_context(self.build_llm_prompt(state, history)) for state, history in list(self.vllm_output)]
        context_token_ids = self.tokenizer(
            all_context_texts,
            add_special_tokens=False,
            max_length=self.prompt_max_len,
            padding=False,
            truncation=True,
        )["input_ids"]

        self.vllm_engine.add_requests(sampling_params=sampling_params, prompt_token_ids=context_token_ids)
        outputs = self.vllm_engine.get_responses()
        
        self.output_step += 1
        # if not hasattr(self, "_logger") or self._logger is None:
            # return

        for i, ((state, history), out) in enumerate(zip(list(self.vllm_output), outputs)):
            self._logger.info(
                f"\n[vllm_output step={self.output_step} idx={i}]"
                f"\n--- INPUT ---\n{self.build_llm_prompt(state, history)}"
                f"\n--- OUTPUT ---\n{out.outputs[0].text}\n"
            )
        
        
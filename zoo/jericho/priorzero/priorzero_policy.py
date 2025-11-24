# priorzero_policy.py
"""
[PRIORZERO] PriorZero Policy Implementation

This module implements the PriorZero policy that combines:
1. UniZero world model for planning in latent space
2. LLM policy model for providing high-quality action priors

Key Features:
- Dual-model training: world model + LLM policy
- LLM-guided MCTS: inject LLM priors into MCTS root node
- SFT + RFT: supervised fine-tuning with MCTS policies + reinforcement fine-tuning with environment rewards
- Full alignment with UniZero implementation

Author: PriorZero Team
Date: 2025-01-20
"""

import copy
import re
import sys
import time
import cProfile
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import List, Dict, Any, Tuple, Union, Optional

# [CRITICAL] Ensure local LightZero is used
from ensure_local_lightzero import ensure_local_lightzero
ensure_local_lightzero()

import numpy as np
import torch
import torch.nn.functional as F
from ding.utils import POLICY_REGISTRY
from ding.model import model_wrap
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType
import os

# Import from local LightZero
from lzero.policy.unizero import UniZeroPolicy as OriginalUniZeroPolicy
from lzero.policy import phi_transform, InverseScalarTransform, scalar_transform, DiscreteSupport
from lzero.policy import to_torch_float_tensor,mz_network_output_unpack, prepare_obs
from lzero.policy.utils import select_action
from lzero.mcts import UniZeroMCTSCtree as MCTSCtree
from lzero.entry.utils import initialize_zeros_batch
import lzero.model.unizero_model  


def build_llm_prompt(
    current_obs: str,
    history: Optional[List[Tuple[str, str, float]]] = None,
    action_descriptions: Optional[Dict[str, str]] = None,
    use_cot: bool = True
) -> str:
    """
    [PRIORZERO-NEW]
    Build a high-quality prompt for LLM to generate the next action.

    When use_cot is True, the model should:
        - First output its reasoning inside <think></think>
        - Then output the SINGLE best next action inside <action></action>

    When use_cot is False, the model should:
        - Output ONLY the SINGLE best next action inside <action></action>

    Args:
        current_obs: Current observation text
        history: List of (observation, action, reward) tuples
        action_descriptions: Optional descriptions for each action
        use_cot: Whether to encourage chain-of-thought reasoning

    Returns:
        Formatted prompt string
    """
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

    # Current observation
    prompt_parts.append("\n=== Current Situation ===")
    prompt_parts.append(current_obs)

    # Available actions (if provided)
    if action_descriptions:
        prompt_parts.append("\n=== Available Actions ===")
        prompt_parts.append(
            "You MUST choose the best action from the list below. "
            "Do not invent actions that are not in this list."
        )
        for action_text, desc in action_descriptions.items():
            # action_text: should match exactly the string we want inside <action>...</action>
            prompt_parts.append(f"- {action_text}: {desc}")

    # Task + output format
    if use_cot:
        # CoT 模式：先 <think>，再 <action>
        prompt_parts.append(
            "\n=== Task ===\n"
            "Analyze the recent history and the current situation, and decide on the SINGLE best next action.\n\n"
            "OUTPUT FORMAT:\n"
            "- First, write your detailed reasoning inside <think>...</think>.\n"
            "- Then, on a new line, output ONLY the chosen action text inside <action>...</action>.\n"
            "- Finally, do not put any text outside the <think> and <action> tags.\n\n"
            "Example (format only):\n"
            "<think>your step-by-step reasoning here</think>\n"
            "<action>the best action text here</action>\n\n"
        )
    else:
        # 非 CoT：只要最终动作
        # prompt_parts.append(
        #     "\n=== Task ===\n"
        #     "Analyze the recent history and the current situation, and decide on the SINGLE best next action.\n\n"
        #     "Your result should be wrapped in <action></action>, and please keep the output concise, avoiding any other content."
        #     "\nExample: <action>turn on</action>"
        # )
        prompt_parts.append(
            "\n=== Task ===\n"
            "Analyze the recent history and the current situation, and decide on the SINGLE best next action."
            "Please keep the output concise, avoiding any other content.\n\n"
        )
    return "\n".join(prompt_parts)

# ==============================================================================
# PriorZero Policy Class
# ==============================================================================

@POLICY_REGISTRY.register('priorzero', force_overwrite=True)
class PriorZeroPolicy(OriginalUniZeroPolicy):
    """
    [PRIORZERO-MODIFIED]
    PriorZero policy that combines UniZero world model with LLM policy.

    Architecture:
        - UniZero World Model: Learns latent dynamics, value, and policy in latent space
        - LLM Policy Model: Provides high-quality action priors based on language understanding

    Training:
        - World Model: Trained with standard UniZero losses (value, policy, reward, latent)
        - LLM: Trained with SFT (using MCTS policies) + RFT (using environment rewards)

    Inference:
        - LLM generates action ranking → converted to policy prior
        - Policy prior injected into MCTS root node
        - MCTS search refines the policy → selects best action
    """

    config = dict(
        **OriginalUniZeroPolicy.config,
        # LLM-specific config
        llm_policy_cfg=dict(
            pretrain_llm_path="Qwen/Qwen1.5-1.8B-Chat",
            use_lora=False,  # Whether to use LoRA for efficient fine-tuning
            lora_r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            llm_learning_rate=1e-6,
            llm_weight_decay=0.01,
            llm_loss_weight=0.5,  # Weight of LLM loss in total loss
            rft_loss_weight=0.3,  # Weight of RFT loss in total loss
            prompt_max_len=2048,
            generate_max_len=128,
            history_length=5,  # Number of recent steps to include in prompt
            use_cot=True,  # Whether to use chain-of-thought prompting
            sft_target='mcts_policy',  # 'mcts_policy' or 'oracle_policy'
            enable_rft=True,  # Whether to enable RFT training
        ),
    )

    def __init__(self, cfg: Dict, model: torch.nn.Module = None, enable_field: List[str] = None, **kwargs):
        # [PRIORZERO-NEW] Initialize LLM-related attributes BEFORE super().__init__
        # because super().__init__ will call _init_learn which needs these attributes        self.llm_policy_model = None
        self.llm_tokenizer = None
        self._optimizer_llm = None
        self._lr_scheduler_llm = None
        self._last_llm_grad_norm = 0.0
        self.llm_policy_cfg = cfg.llm_policy_cfg  # Set from cfg, not self._cfg (not set yet)
        self.profile_cfg = getattr(cfg, 'profile_cfg', {})
        self._profile_enabled = bool(self.profile_cfg.get('enable_cprofile', False))
        self._profile_dir = f"./{kwargs['exp_name']}/log/profile"
        self._profile_log_interval = int(self.profile_cfg.get('log_interval', 50))
        self._profile_stats = { 'train_world_model': {'count': 0, 'total': 0.0, 'max': 0.0}, 
                                'train_llm_sft': {'count': 0, 'total': 0.0, 'max': 0.0},
                                'train_llm_rft': {'count': 0, 'total': 0.0, 'max': 0.0}
                            }
        self._profile_stats_file = f'{self._profile_dir}/train_time.log'
        if self._profile_enabled:
            os.makedirs(self._profile_dir, exist_ok=True)

        # Call parent init (this will trigger _init_learn, _init_collect, _init_eval)
        super().__init__(cfg, model, enable_field)

    def _init_learn(self) -> None:
        """
        [PRIORZERO-MODIFIED]
        Initialize both UniZero world model and LLM policy model with their optimizers.
        Align with UniZero implementation - use logging instead of self._logger.
        """
        # ======================================================================
        # 1. Initialize UniZero World Model (from parent class)
        # ======================================================================
        super()._init_learn()
        logging.info("✓ UniZero World Model and optimizer initialized")
        logging.info(f"Loading LLM from: {self.llm_policy_cfg.pretrain_llm_path}")

        # Load tokenizer
        self.llm_tokenizer = AutoTokenizer.from_pretrained(
            self.llm_policy_cfg.pretrain_llm_path,
            trust_remote_code=True,
            padding_side='left'  # For batch generation
        )
        if self.llm_tokenizer.pad_token is None:
            self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token

        # Load LLM
        self.llm_policy_model = AutoModelForCausalLM.from_pretrained(
            self.llm_policy_cfg.pretrain_llm_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,  # Use bfloat16 to save memory
            device_map=None,  # We'll manually move to device
        )

        # Apply LoRA if enabled
        if self.llm_policy_cfg.use_lora:
            logging.info("Applying LoRA for parameter-efficient fine-tuning")
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.llm_policy_cfg.lora_r,
                lora_alpha=self.llm_policy_cfg.lora_alpha,
                lora_dropout=self.llm_policy_cfg.lora_dropout,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Qwen-specific
            )
            self.llm_policy_model = get_peft_model(self.llm_policy_model, lora_config)
            self.llm_policy_model.print_trainable_parameters()

        self.llm_policy_model.to(self._cfg.device)
        self.llm_policy_model.train()

        # ======================================================================
        # 3. [PRIORZERO-NEW] Initialize LLM Optimizer
        # ======================================================================
        self._optimizer_llm = torch.optim.AdamW(
            self.llm_policy_model.parameters(),
            lr=self.llm_policy_cfg.llm_learning_rate,
            weight_decay=self.llm_policy_cfg.llm_weight_decay,
            betas=(0.9, 0.999),
        )

        # Optional: learning rate scheduler
        self._lr_scheduler_llm = torch.optim.lr_scheduler.CosineAnnealingLR(
            self._optimizer_llm,
            T_max=100000,  # Will be set from config
            eta_min=self.llm_policy_cfg.llm_learning_rate * 0.1
        )

        logging.info(f"✓ LLM Policy Model ({self.llm_policy_cfg.pretrain_llm_path}) initialized")
        logging.info(f"  - LLM learning rate: {self.llm_policy_cfg.llm_learning_rate}")
        logging.info(f"  - LoRA enabled: {self.llm_policy_cfg.use_lora}")

    @contextmanager
    def _profile_block(self, name: str):
        if not self._profile_enabled:
            yield None
            return
        profiler = cProfile.Profile()
        start_time = time.perf_counter()
        profiler.enable()
        try:
            yield profiler
        finally:
            profiler.disable()
            elapsed = time.perf_counter() - start_time
            self._record_profile_time(name, elapsed)

    def _record_profile_time(self, name: str, elapsed: float) -> None:
        log_every = max(1, self._profile_log_interval)
        self._profile_stats[name]['count'] += 1
        self._profile_stats[name]['total'] += elapsed
        self._profile_stats[name]['max'] = max(self._profile_stats[name]['max'], elapsed)
        if self._profile_stats[name]['count'] % log_every == 0:
            avg = self._profile_stats[name]['total'] / self._profile_stats[name]['count']
            with open(self._profile_stats_file, mode='a', encoding='utf-8') as f:
                f.write(
                    f"{time.time():.3f}\tname={name}\tcount={self._profile_stats[name]['count']}\t"
                    f"total_s={self._profile_stats[name]['total']:.4f}\tavg_s={avg:.4f}\tmax_s={self._profile_stats[name]['max']:.4f}\n"
                )
    

    def _build_llm_samples(
        self,
        raw_obs_list: List[List[str]],
        history_obs_list: List[List[List[Tuple[str, str, float]]]],
        action_logprob_list: Optional[List[List[Any]]] = None,
        target_values = None
    ) -> List[Dict[str, Any]]:
        """
        Build prompt/target pairs (and rewards) for LLM training.
        """
        samples: List[Dict[str, Any]] = []
        B = len(raw_obs_list)
        if B == 0:
            return samples
        T = len(raw_obs_list[0])

        for b in range(B):
            for t in range(T - 1):
                current_obs = raw_obs_list[b][t]
                current_history = history_obs_list[b][t]
                next_step_history = history_obs_list[b][t + 1]
                if target_values is not None:
                    value = target_values[b][t].item()
                else:
                    value = None
                if isinstance(next_step_history, np.ndarray):
                    next_step_history = next_step_history.tolist()
                if not next_step_history:
                    continue
                _, true_action, reward_value = next_step_history[-1]
                if not true_action:
                    continue

                instruction = build_llm_prompt(
                    current_obs=current_obs,
                    history=current_history,
                    use_cot=self.llm_policy_cfg.use_cot
                )
                prompt = self.llm_tokenizer.apply_chat_template(
                    [{"role": "user", "content": instruction}],
                    tokenize=False,
                    add_generation_prompt=True
                )
                old_logprob = None
                if action_logprob_list is not None:
                    old_logprob = action_logprob_list[b][t+1][true_action]


                samples.append(
                    dict(
                        prompt=prompt,
                        # target=f"<answer>{true_action}</answer>{self.llm_tokenizer.eos_token}",
                        target=f"{true_action}{self.llm_tokenizer.eos_token}",
                        reward=float(reward_value) if reward_value is not None else 0.0,
                        value=value,
                        old_logprob=old_logprob
                    )
                )
        return samples

    def compute_sft_loss(
        self,
        raw_obs_list: List[List[str]],
        history_obs_list: List[List[List[Tuple[str, str, float]]]]
    ) -> torch.Tensor:
        """
        Calculate SFT loss and apply gradient updates with accumulation.
        """
        samples = self._build_llm_samples(raw_obs_list, history_obs_list)
        if len(samples) == 0:
            return torch.tensor(0.0, device=self._cfg.device)

        micro_batch_size = min(self.llm_policy_cfg.llm_micro_batch_size, len(samples))
        num_micro_batches = (len(samples) + micro_batch_size - 1) // micro_batch_size
        grad_accum_steps = max(
            1, min(self.llm_policy_cfg.llm_gradient_accumulation_steps, num_micro_batches)
        )

        accumulated_loss = 0.0
        last_grad_norm = 0.0
        self.llm_policy_model.train()
        self._optimizer_llm.zero_grad()

        full_texts = [s['prompt'] + s['target'] for s in samples]
        prompts_only = [s['prompt'] for s in samples]

        for micro_batch_idx in range(num_micro_batches):
            start_idx = micro_batch_idx * micro_batch_size
            end_idx = min((micro_batch_idx + 1) * micro_batch_size, len(samples))

            batch_full_texts = full_texts[start_idx:end_idx]
            batch_prompts = prompts_only[start_idx:end_idx]

            inputs = self.llm_tokenizer(
                batch_full_texts,
                padding=True,
                truncation=True,
                max_length=self.llm_policy_cfg.prompt_max_len,
                return_tensors="pt"
            ).to(self._cfg.device)

            labels = inputs.input_ids.clone()
            labels[labels == self.llm_tokenizer.pad_token_id] = -100

            for i, prompt_str in enumerate(batch_prompts):
                prompt_tokens = self.llm_tokenizer.encode(prompt_str, add_special_tokens=False)
                prompt_len = len(prompt_tokens)
                if prompt_len < labels.shape[1]:
                    labels[i, :prompt_len] = -100
                else:
                    labels[i, :] = -100

            outputs = self.llm_policy_model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                labels=labels
            )
            loss = outputs.loss
            accumulated_loss += loss.item()
            scaled_loss = loss / grad_accum_steps
            scaled_loss.backward()

            should_step = ((micro_batch_idx + 1) % grad_accum_steps == 0) or (micro_batch_idx == num_micro_batches - 1)
            if should_step:
                last_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.llm_policy_model.parameters(),
                    self._cfg.grad_clip_value
                ).item()
                self._optimizer_llm.step()
                if self._lr_scheduler_llm is not None:
                    self._lr_scheduler_llm.step()
                self._optimizer_llm.zero_grad(set_to_none=True)

            del inputs, labels, outputs, loss

        self._last_llm_grad_norm = last_grad_norm
        mean_loss = accumulated_loss / max(1, num_micro_batches)
        return torch.tensor(mean_loss, device=self._cfg.device)
    
    def compute_rft_loss(
        self,
        raw_obs_list: List[List[str]],
        history_obs_list: List[List[List[Tuple[str, str, float]]]],
        action_logprob_list: Optional[List[List[Any]]] = None,
        target_values: Optional[List[List[float]]] = None
    ) -> torch.Tensor:
        """
        Reinforcement fine-tuning loss with in-function gradient/optimizer updates.
        """
        samples = self._build_llm_samples(raw_obs_list, history_obs_list, action_logprob_list, target_values)
        if len(samples) == 0:
            return torch.tensor(0.0, device=self._cfg.device)

        micro_batch_size = min(self.llm_policy_cfg.llm_micro_batch_size, len(samples))
        num_micro_batches = (len(samples) + micro_batch_size - 1) // micro_batch_size
        grad_accum_steps = max(
            1, min(self.llm_policy_cfg.llm_gradient_accumulation_steps, num_micro_batches)
        )

        accumulated_loss = 0.0
        last_grad_norm = 0.0
        self.llm_policy_model.train()
        self._optimizer_llm.zero_grad()

        full_texts = [s['prompt'] + s['target'] for s in samples]
        prompts_only = [s['prompt'] for s in samples]
        rewards_list = [s['reward'] for s in samples]
        values_list = [s['value'] for s in samples]
        old_logprob_list = [s.get('old_logprob', None) for s in samples]
        loss_type = getattr(self.llm_policy_cfg, 'rft_loss_type', 'reinforce').lower()
        clip_eps = getattr(self.llm_policy_cfg, 'rft_clip_epsilon', 0.2)

        for micro_batch_idx in range(num_micro_batches):
            start_idx = micro_batch_idx * micro_batch_size
            end_idx = min((micro_batch_idx + 1) * micro_batch_size, len(samples))

            batch_full_texts = full_texts[start_idx:end_idx]
            batch_prompts = prompts_only[start_idx:end_idx]
            batch_rewards = rewards_list[start_idx:end_idx]
            batch_old_logprob = old_logprob_list[start_idx:end_idx]
            batch_values = values_list[start_idx:end_idx]

            inputs = self.llm_tokenizer(
                batch_full_texts,
                padding=True,
                truncation=True,
                max_length=self.llm_policy_cfg.prompt_max_len,
                return_tensors="pt"
            ).to(self._cfg.device)

            labels = inputs.input_ids.clone()
            labels[labels == self.llm_tokenizer.pad_token_id] = -100
            for i, prompt_str in enumerate(batch_prompts):
                prompt_tokens = self.llm_tokenizer.encode(prompt_str, add_special_tokens=False)
                prompt_len = len(prompt_tokens)
                if prompt_len < labels.shape[1]:
                    labels[i, :prompt_len] = -100
                else:
                    labels[i, :] = -100

            outputs = self.llm_policy_model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask
            )
            log_probs = F.log_softmax(outputs.logits, dim=-1)
            shifted_log_probs = log_probs[:, :-1, :].contiguous()
            shifted_labels = labels[:, 1:].contiguous()
            gather_labels = shifted_labels.clone()
            gather_labels[gather_labels == -100] = self.llm_tokenizer.pad_token_id

            token_log_probs = shifted_log_probs.gather(
                dim=-1,
                index=gather_labels.unsqueeze(-1)
            ).squeeze(-1)
            mask = (shifted_labels != -100).float()
            token_log_probs = token_log_probs * mask
            sequence_log_probs = token_log_probs.sum(dim=-1) / (mask.sum(dim=-1) + 1e-8)
            
            if self.llm_policy_cfg.rft_reward=='value':
                rewards_tensor = torch.tensor(batch_values, device=self._cfg.device, dtype=torch.float32)
            elif self.llm_policy_cfg.rft_reward=='reward':
                rewards_tensor = torch.tensor(batch_rewards, device=self._cfg.device, dtype=torch.float32)
            else:
                pass
            if len(batch_rewards) > 1:
                rewards_tensor = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)

            if loss_type == 'reinforce++' and all(lp is not None for lp in batch_old_logprob):
                old_lp_tensor = torch.tensor(batch_old_logprob, device=self._cfg.device, dtype=torch.float32)
                ratio = torch.exp(sequence_log_probs - old_lp_tensor)
                clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
                surrogate1 = ratio * rewards_tensor
                surrogate2 = clipped_ratio * rewards_tensor
                loss_term = torch.min(surrogate1, surrogate2)
                loss = -loss_term.mean()
            else:
                loss = -(rewards_tensor * sequence_log_probs).mean()
            accumulated_loss += loss.item()
            scaled_loss = loss / grad_accum_steps
            scaled_loss.backward()

            should_step = ((micro_batch_idx + 1) % grad_accum_steps == 0) or (micro_batch_idx == num_micro_batches - 1)
            if should_step:
                last_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.llm_policy_model.parameters(),
                    self._cfg.grad_clip_value
                ).item()
                self._optimizer_llm.step()
                if self._lr_scheduler_llm is not None:
                    self._lr_scheduler_llm.step()
                self._optimizer_llm.zero_grad(set_to_none=True)

            del inputs, labels, outputs, loss

        self._last_llm_grad_norm = last_grad_norm
        mean_loss = accumulated_loss / max(1, num_micro_batches)
        return torch.tensor(mean_loss, device=self._cfg.device)
    
    
    def _forward_learn(self, data: Tuple[torch.Tensor]) -> Dict[str, Union[float, int]]:
        """
        [PRIORZERO-MODIFIED]
        Dual-model training: UniZero world model + LLM policy.

        Training process:
        1. Train UniZero world model with standard losses (value, policy, reward, latent)
        2. Train LLM with SFT (supervised by MCTS policies)
        3. Optionally train LLM with RFT (reinforced by environment rewards)
        4. Joint optimization with combined loss

        Args:
            data: Tuple containing (current_batch, target_batch, train_iter, game_segments)

        Returns:
            log_dict: Dictionary of training metrics
        """
        self._learn_model.train()
        self._target_model.train()
        self.llm_policy_model.train()

        current_batch, target_batch, train_iter = data

        obs_batch_ori, action_batch, target_action_batch, mask_batch, batch_index_tensor, weights, make_time, timestep_batch, raw_obs_list, history_obs_list, action_logprob_list = current_batch
        target_reward, target_value, target_policy = target_batch
        
        obs_batch, obs_target_batch = prepare_obs(obs_batch_ori, self._cfg)
        action_batch = torch.from_numpy(action_batch).to(self._cfg.device).unsqueeze(
            -1).long() 
        timestep_batch = torch.from_numpy(timestep_batch).to(self._cfg.device).unsqueeze(
            -1).long()

        data_list = [mask_batch, target_reward, target_value, target_policy, weights]
        (mask_batch, target_reward, target_value, target_policy, weights) = to_torch_float_tensor(data_list, self._cfg.device)

        batch_size = self._cfg.batch_size
        target_reward = target_reward.view(batch_size, -1)
        target_value = target_value.view(batch_size, -1)

        transformed_target_reward = scalar_transform(target_reward)
        transformed_target_value = scalar_transform(target_value)

        # Convert to categorical distribution (for distributional RL)
        target_reward_categorical = phi_transform(self.reward_support, transformed_target_reward)
        target_value_categorical = phi_transform(self.value_support, transformed_target_value)

        batch_for_gpt = {
            'actions': action_batch.squeeze(-1),
            'timestep': timestep_batch.squeeze(-1),
            'rewards': target_reward_categorical[:, :-1],
            'target_value': target_value_categorical[:, :-1],
            'target_policy': target_policy[:, :-1],
        }
        if isinstance(self._cfg.model.observation_shape, int) or len(self._cfg.model.observation_shape) == 1:
            batch_for_gpt['observations'] = torch.cat((obs_batch, obs_target_batch), dim=1).reshape(
                self._cfg.batch_size, -1, self._cfg.model.observation_shape)
        elif len(self._cfg.model.observation_shape) == 3:
            batch_for_gpt['observations'] = torch.cat((obs_batch, obs_target_batch), dim=1).reshape(
                self._cfg.batch_size, -1, *self._cfg.model.observation_shape)

        batch_for_gpt['mask_padding'] = mask_batch == 1.0  
        batch_for_gpt['observations'] = batch_for_gpt['observations'][:, :-1]  
        batch_for_gpt['mask_padding'] = batch_for_gpt['mask_padding'][:, :-1]
        batch_for_gpt['ends'] = torch.zeros(batch_for_gpt['mask_padding'].shape, dtype=torch.long, device=self._cfg.device)
        batch_for_gpt['scalar_target_value'] = target_value

        with self._profile_block(name="train_world_model"):
            wm_losses = self._learn_model.world_model.compute_loss(
                batch_for_gpt,
                self._target_model.world_model.tokenizer,
                self.value_inverse_scalar_transform_handle,
            )

            wm_total_loss = (weights * wm_losses.loss_total).mean()

        # ==============================================================================
        # PRIORZERO-NEW] LLM Policy Training (SFT + RFT)
        # ==============================================================================
        self._last_llm_grad_norm = 0.0
        if self.llm_policy_cfg.enable_llm and self.llm_policy_cfg.enable_sft:
            with self._profile_block(name="train_llm_sft"):
                llm_sft_loss = self.compute_sft_loss(raw_obs_list=raw_obs_list, history_obs_list=history_obs_list)
        else:
            llm_sft_loss = torch.tensor(0.0, device=self._cfg.device)
        if self.llm_policy_cfg.enable_llm and self.llm_policy_cfg.enable_rft:
            with self._profile_block(name="train_llm_rft"):
                llm_rft_loss = self.compute_rft_loss(
                    raw_obs_list=raw_obs_list,
                    history_obs_list=history_obs_list,
                    action_logprob_list=action_logprob_list,
                    target_values=target_value,
                )
        else:
            llm_rft_loss = torch.tensor(0.0, device=self._cfg.device)

        llm_loss = (
            self.llm_policy_cfg.llm_loss_weight * llm_sft_loss +
            self.llm_policy_cfg.rft_loss_weight * llm_rft_loss
        )
        total_loss = wm_total_loss + llm_loss  # For logging
        
        self._optimizer_world_model.zero_grad()
        wm_total_loss.backward()
        wm_grad_norm = torch.nn.utils.clip_grad_norm_(
            self._learn_model.world_model.parameters(),
            self._cfg.grad_clip_value
        )
        self._optimizer_world_model.step()
        self._target_model.update(self._learn_model.state_dict())


        intermediate_losses = wm_losses.intermediate_losses
        obs_loss = intermediate_losses.get('loss_obs', torch.tensor(0.0))
        reward_loss = intermediate_losses.get('loss_rewards', torch.tensor(0.0))
        policy_loss = intermediate_losses.get('loss_policy', torch.tensor(0.0))
        value_loss = intermediate_losses.get('loss_value', torch.tensor(0.0))
        latent_recon_loss = intermediate_losses.get('latent_recon_loss', torch.tensor(0.0))
        perceptual_loss = intermediate_losses.get('perceptual_loss', torch.tensor(0.0))
        orig_policy_loss = intermediate_losses.get('orig_policy_loss', torch.tensor(0.0))
        policy_entropy = intermediate_losses.get('policy_entropy', torch.tensor(0.0))
        first_step_losses = intermediate_losses.get('first_step_losses', {})
        middle_step_losses = intermediate_losses.get('middle_step_losses', {})
        last_step_losses = intermediate_losses.get('last_step_losses', {})

        # Analysis metrics (dormant ratio, weight magnitude, etc.)
        dormant_ratio_encoder = intermediate_losses.get('dormant_ratio_encoder', 0.0)
        dormant_ratio_transformer = intermediate_losses.get('dormant_ratio_transformer', 0.0)
        dormant_ratio_head = intermediate_losses.get('dormant_ratio_head', 0.0)
        avg_weight_mag_encoder = intermediate_losses.get('avg_weight_mag_encoder', 0.0)
        avg_weight_mag_transformer = intermediate_losses.get('avg_weight_mag_transformer', 0.0)
        avg_weight_mag_head = intermediate_losses.get('avg_weight_mag_head', 0.0)
        e_rank_last_linear = intermediate_losses.get('e_rank_last_linear', 0.0)
        e_rank_sim_norm = intermediate_losses.get('e_rank_sim_norm', 0.0)
        latent_state_l2_norms = intermediate_losses.get('latent_state_l2_norms', torch.tensor(0.0))
        latent_action_l2_norms = intermediate_losses.get('latent_action_l2_norms', 0.0)

        # Logits statistics
        logits_value_mean = intermediate_losses.get('logits_value_mean', 0.0)
        logits_value_max = intermediate_losses.get('logits_value_max', 0.0)
        logits_value_min = intermediate_losses.get('logits_value_min', 0.0)
        logits_policy_mean = intermediate_losses.get('logits_policy_mean', 0.0)
        logits_policy_max = intermediate_losses.get('logits_policy_max', 0.0)
        logits_policy_min = intermediate_losses.get('logits_policy_min', 0.0)

        # Temperature parameters
        temperature_value = intermediate_losses.get('temperature_value', 0.0)
        temperature_reward = intermediate_losses.get('temperature_reward', 0.0)
        temperature_policy = intermediate_losses.get('temperature_policy', 0.0)

        # Value priority for prioritized replay
        value_priority_tensor = intermediate_losses.get('value_priority', torch.tensor([0.0]))
        value_priority_np = value_priority_tensor.detach().cpu().numpy() + 1e-6

        # Compute target policy entropy (for analysis)
        valid_target_policy = batch_for_gpt['target_policy'][batch_for_gpt['mask_padding']]
        target_policy_entropy = -torch.sum(valid_target_policy * torch.log(valid_target_policy + 1e-9), dim=-1)
        average_target_policy_entropy = target_policy_entropy.mean()

        # Build comprehensive log dict (aligned with UniZero)
        log_dict = {
            # ============ Core Losses ============
            'wm_total_loss': wm_total_loss.item(),
            'wm_obs_loss': obs_loss.item() if torch.is_tensor(obs_loss) else obs_loss,
            'wm_reward_loss': reward_loss.item() if torch.is_tensor(reward_loss) else reward_loss,
            'wm_policy_loss': policy_loss.item() if torch.is_tensor(policy_loss) else policy_loss,
            'wm_value_loss': value_loss.item() if torch.is_tensor(value_loss) else value_loss,
            'wm_latent_recon_loss': latent_recon_loss.item() if torch.is_tensor(latent_recon_loss) else latent_recon_loss,
            'wm_perceptual_loss': perceptual_loss.item() if torch.is_tensor(perceptual_loss) else perceptual_loss,
            'wm_orig_policy_loss': orig_policy_loss.item() if torch.is_tensor(orig_policy_loss) else orig_policy_loss,
            'wm_policy_entropy': policy_entropy.item() if torch.is_tensor(policy_entropy) else policy_entropy,
            'wm_target_policy_entropy': average_target_policy_entropy.item(),


            # ============ Step-wise Losses ============
            'analysis/first_step_loss_value': first_step_losses.get('loss_value', torch.tensor(0.0)).item() if isinstance(first_step_losses.get('loss_value'), torch.Tensor) else 0.0,
            'analysis/first_step_loss_policy': first_step_losses.get('loss_policy', torch.tensor(0.0)).item() if isinstance(first_step_losses.get('loss_policy'), torch.Tensor) else 0.0,
            'analysis/first_step_loss_rewards': first_step_losses.get('loss_rewards', torch.tensor(0.0)).item() if isinstance(first_step_losses.get('loss_rewards'), torch.Tensor) else 0.0,
            'analysis/first_step_loss_obs': first_step_losses.get('loss_obs', torch.tensor(0.0)).item() if isinstance(first_step_losses.get('loss_obs'), torch.Tensor) else 0.0,

            'analysis/middle_step_loss_value': middle_step_losses.get('loss_value', torch.tensor(0.0)).item() if isinstance(middle_step_losses.get('loss_value'), torch.Tensor) else 0.0,
            'analysis/middle_step_loss_policy': middle_step_losses.get('loss_policy', torch.tensor(0.0)).item() if isinstance(middle_step_losses.get('loss_policy'), torch.Tensor) else 0.0,
            'analysis/middle_step_loss_rewards': middle_step_losses.get('loss_rewards', torch.tensor(0.0)).item() if isinstance(middle_step_losses.get('loss_rewards'), torch.Tensor) else 0.0,
            'analysis/middle_step_loss_obs': middle_step_losses.get('loss_obs', torch.tensor(0.0)).item() if isinstance(middle_step_losses.get('loss_obs'), torch.Tensor) else 0.0,

            'analysis/last_step_loss_value': last_step_losses.get('loss_value', torch.tensor(0.0)).item() if isinstance(last_step_losses.get('loss_value'), torch.Tensor) else 0.0,
            'analysis/last_step_loss_policy': last_step_losses.get('loss_policy', torch.tensor(0.0)).item() if isinstance(last_step_losses.get('loss_policy'), torch.Tensor) else 0.0,
            'analysis/last_step_loss_rewards': last_step_losses.get('loss_rewards', torch.tensor(0.0)).item() if isinstance(last_step_losses.get('loss_rewards'), torch.Tensor) else 0.0,
            'analysis/last_step_loss_obs': last_step_losses.get('loss_obs', torch.tensor(0.0)).item() if isinstance(last_step_losses.get('loss_obs'), torch.Tensor) else 0.0,

            # ============ Analysis Metrics ============
            'analysis/latent_state_l2_norms': latent_state_l2_norms.item() if torch.is_tensor(latent_state_l2_norms) else latent_state_l2_norms,
            'analysis/latent_action_l2_norms': latent_action_l2_norms,

            # ============ Logits Statistics ============
            'logits_value_mean': logits_value_mean,
            'logits_value_max': logits_value_max,
            'logits_value_min': logits_value_min,
            'logits_policy_mean': logits_policy_mean,
            'logits_policy_max': logits_policy_max,
            'logits_policy_min': logits_policy_min,

            # ============ Temperature Parameters ============
            'temperature_value': temperature_value,
            'temperature_reward': temperature_reward,
            'temperature_policy': temperature_policy,

            # ============ Targets ============
            'wm_target_reward': target_reward.mean().item(),
            'wm_target_value': target_value.mean().item(),
            'transformed_target_reward': transformed_target_reward.mean().item(),
            'transformed_target_value': transformed_target_value.mean().item(),
            'value_priority': value_priority_np.mean().item(),
            'value_priority_orig': value_priority_np,

            # ============ Gradient Norms ============
            'wm_grad_norm': wm_grad_norm.item(),
            'llm_grad_norm': self._last_llm_grad_norm,

            # ============ Learning Rates ============
            'cur_lr_world_model': self._optimizer_world_model.param_groups[0]['lr'],
            'llm_lr': self._optimizer_llm.param_groups[0]['lr'],

            # ============ [PRIORZERO] LLM-specific Metrics ============
            'llm_sft_loss': llm_sft_loss.item(),
            'llm_rft_loss': llm_rft_loss.item(),
            'llm_total_loss': llm_loss.item(),
            # 'num_sft_samples': float(num_sft_samples),
            # 'num_rft_samples': float(num_rft_samples),
            'total_loss': total_loss.item(),
        }

        return log_dict

    def _monitor_vars_learn(self) -> List[str]:
        """
        [PRIORZERO-MODIFIED]
        Register variables to be monitored in learn mode for TensorBoard logging.

        This extends UniZero's monitoring with PriorZero-specific LLM metrics.

        Returns:
            List of variable names that should be logged to TensorBoard/WandB
        """

        return [
            # ============ LLM Loss Metrics ============
            'llm_sft_loss',              # Supervised fine-tuning loss
            'llm_rft_loss',              # Reinforcement fine-tuning loss
            'llm_total_loss',            # Combined LLM loss
            'llm_grad_norm',             # LLM gradient norm
            'llm_lr',                    # LLM learning rate
            # ============ LLM Training Statistics ============
            # 'num_sft_samples',           # Number of SFT samples in batch
            # 'num_rft_samples',           # Number of RFT samples in batch
            # ============ Combined Metrics ============
            'total_loss',                # Total loss (WM + LLM)
            'wm_total_loss',             # World model total loss
            'wm_grad_norm',              # World model gradient norm
            # ============ World Model Component Losses ============
            'wm_value_loss',
            'wm_policy_loss',
            'wm_reward_loss',
            'wm_obs_loss',

            'analysis/dormant_ratio_encoder',
            'analysis/dormant_ratio_transformer',
            'analysis/dormant_ratio_head',
            'analysis/avg_weight_mag_encoder',
            'analysis/avg_weight_mag_transformer',
            'analysis/avg_weight_mag_head',
            'analysis/latent_state_l2_norms',

            'analysis/first_step_loss_value',
            'analysis/first_step_loss_policy',
            'analysis/first_step_loss_rewards',
            'analysis/first_step_loss_obs',

            'analysis/middle_step_loss_value',
            'analysis/middle_step_loss_policy',
            'analysis/middle_step_loss_rewards',
            'analysis/middle_step_loss_obs',

            'analysis/last_step_loss_value',
            'analysis/last_step_loss_policy',
            'analysis/last_step_loss_rewards',
            'analysis/last_step_loss_obs',

            'adaptive_alpha',
            "adaptive_target_entropy_ratio",
            'alpha_loss',

            'Current_GPU',
            'Max_GPU',
            'collect_epsilon',
            'collect_mcts_temperature',
            'cur_lr_world_model',
            'cur_lr_tokenizer',
            
            'wm_orig_policy_loss',
            'wm_policy_entropy',
            'wm_latent_recon_loss',
            'wm_target_policy_entropy',
            'consistency_loss',
            'value_priority',
            'wm_target_reward',
            'wm_target_value',
            'total_grad_norm_before_clip_wm',
            # tokenizer
            'commitment_loss',
            'reconstruction_loss',
            'wm_perceptual_loss',

            "logits_value_mean",
            "logits_value_max",
            "logits_value_min",
            "logits_policy_mean",
            "logits_policy_max",
            "logits_policy_min",

            "temperature_value",
            "temperature_reward",
            "temperature_policy",
            "current_policy_label_eps",
            'adaptive_alpha',
            "adaptive_target_entropy_ratio",
            'alpha_loss',
            "current_encoder_clip_value",
        ]
        # ========================================================================
    
    def pad_to_fixed_length(self, data, target_len=55, pad_val=-1e9, dtype=torch.float32):
        """
        data: List[Sequence[Number]]，每个元素长度可以不一样（比如 3 或 4）
        返回: tensor, 形状 [B, target_len]，多余部分全是 pad_val
        """
        batch_size = len(data)
        out = torch.full((batch_size, target_len), pad_val, dtype=dtype)
        for i, seq in enumerate(data):
            if isinstance(seq, np.ndarray):
                seq = seq.tolist()
            L = min(len(seq), target_len)
            if L > 0:
                out[i, :L] = torch.tensor(seq[:L], dtype=dtype)
        return out
    
    def _forward_collect(
        self,
        data: torch.Tensor,
        action_mask: List[np.ndarray],
        temperature: float = 1.0,
        to_play: List[int] = None,
        epsilon: float = 0.0,
        ready_env_id: List[int] = None,
        timestep: List = [0],
        **kwargs
    ) -> Dict[int, Dict[str, Any]]:
        """
        [PRIORZERO-MODIFIED]
        Forward pass for data collection with LLM-guided MCTS.

        Process:
        1. Get LLM prior outputs from kwargs
        2. Parse LLM outputs into policy priors
        3. Run world model initial inference
        4. Inject LLM priors into MCTS root node (replace policy logits)
        5. Run MCTS search with LLM-guided priors
        6. Return best action and statistics

        Args:
            data: Stacked observations (tensor)
            action_mask: Action masks for each environment
            temperature: Temperature for action selection
            to_play: Player IDs (for multi-agent)
            epsilon: Epsilon for epsilon-greedy exploration
            ready_env_id: List of ready environment IDs
            **kwargs: Additional arguments, including 'llm_prior_outputs'

        Returns:
            output_dict: Dictionary mapping env_id to action and search statistics
        """
        self._collect_model.eval()

        llm_prior_logprob = kwargs.pop('llm_prior_logprob', None)
        valid_actions_list = kwargs.get('valid_actions_list', None)

        if llm_prior_logprob is None:
            logging.debug("No LLM priors provided, using standard UniZero MCTS")
            return super()._forward_collect(
                data, action_mask, temperature, to_play, epsilon,
                ready_env_id=ready_env_id, timestep=timestep
            )
            
        policy_priors = []
        for idx, actions in enumerate(valid_actions_list):
            prior = []
            for action in actions:
                prior.append(llm_prior_logprob[idx][action])
            policy_priors.append(prior)
        policy_priors = self.pad_to_fixed_length(data=policy_priors, target_len=self.cfg.model.action_space_size, pad_val=-1e9)
        # ======================================================================
        # World Model Initial Inference
        # ======================================================================
        self._collect_mcts_temperature = temperature
        self._collect_epsilon = epsilon
        active_collect_env_num = data.shape[0]
        if ready_env_id is None:
            ready_env_id = np.arange(active_collect_env_num)
        output = {i: None for i in ready_env_id}
        with torch.no_grad():
            network_output = self._collect_model.initial_inference(self.last_batch_obs, self.last_batch_action, data, timestep)
            latent_state_roots, reward_roots, pred_values, policy_logits = mz_network_output_unpack(network_output)

            network_output.policy_logits = policy_priors
            if not self._cfg.mcts_ctree:
                raise NotImplementedError("Python MCTS not supported for PriorZero")

            # ======================================================================
            # MCTS Search with LLM-Guided Priors
            # ======================================================================
            pred_values_np = self.value_inverse_scalar_transform_handle(pred_values).detach().cpu().numpy()
            latent_state_roots_np = latent_state_roots.detach().cpu().numpy()
            policy_logits = policy_priors.detach().cpu().numpy().tolist()
            
            legal_actions = [[i for i, x in enumerate(action_mask[j]) if x == 1] for j in range(active_collect_env_num)]
            noises = [
                np.random.dirichlet([self._cfg.root_dirichlet_alpha] * int(sum(action_mask[j]))
                                    ).astype(np.float32).tolist() for j in range(active_collect_env_num)
            ]
            roots = MCTSCtree.roots(active_collect_env_num, legal_actions)
            roots.prepare(self._cfg.root_noise_weight, noises, reward_roots, policy_logits, to_play)
            self._mcts_collect.search(roots, self._collect_model, latent_state_roots_np, to_play, timestep=timestep)

            roots_visit_count = roots.get_distributions()
            roots_values = roots.get_values()

            batch_action = []
            for i, env_id in enumerate(ready_env_id):
                distributions = roots_visit_count[i]
                value = roots_values[i]

                action_index_in_legal_action_set, visit_count_distribution_entropy = select_action(
                    distributions,
                    temperature=self._collect_mcts_temperature,
                    deterministic=False
                )

                legal_action_indices = np.where(action_mask[i] == 1.0)[0]
                action = legal_action_indices[action_index_in_legal_action_set]

                output[env_id] = {
                    'action': int(action),
                    'visit_count_distributions': distributions,
                    'visit_count_distribution_entropy': visit_count_distribution_entropy,
                    'searched_value': value,
                    'predicted_value': pred_values_np[i],
                    'predicted_policy_logits': policy_logits[i],
                    'timestep': timestep[i],
                }
                batch_action.append(action)
            self.last_batch_obs = data
            self.last_batch_action = batch_action
        return output

    def _state_dict_learn(self) -> Dict[str, Any]:
        """
        [PRIORZERO-MODIFIED]
        Save state dict for both world model and LLM.
        """
        state_dict = super()._state_dict_learn()

        # Add LLM model and optimizer
        state_dict['llm_model'] = self.llm_policy_model.state_dict()
        state_dict['optimizer_llm'] = self._optimizer_llm.state_dict()

        if self._lr_scheduler_llm is not None:
            state_dict['lr_scheduler_llm'] = self._lr_scheduler_llm.state_dict()

        return state_dict

    def _load_state_dict_learn(self, state_dict: Dict[str, Any]) -> None:
        """
        [PRIORZERO-MODIFIED]
        Load state dict for both world model and LLM.
        """
        super()._load_state_dict_learn(state_dict)

        # Load LLM model and optimizer
        if 'llm_model' in state_dict:
            self.llm_policy_model.load_state_dict(state_dict['llm_model'])
            logging.info("✓ LLM model state loaded")

        if 'optimizer_llm' in state_dict:
            self._optimizer_llm.load_state_dict(state_dict['optimizer_llm'])
            logging.info("✓ LLM optimizer state loaded")

        if 'lr_scheduler_llm' in state_dict and self._lr_scheduler_llm is not None:
            self._lr_scheduler_llm.load_state_dict(state_dict['lr_scheduler_llm'])
            logging.info("✓ LLM scheduler state loaded")

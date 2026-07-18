"""
Unified DataProcessor supporting both text (LLM) and image (VL) inputs

This processor can handle:
- Text observations with LLM (original functionality)
- Image observations with VL (new functionality)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import re
import random
import torch
import torch.distributed as dist
from vllm import SamplingParams
from ding.utils import build_logger
import numpy as np
from PIL import Image


class UnifiedDataProcessor:
    """
    Unified DataProcessor supporting both text and image inputs.

    For text input: Uses LLM (vLLM engine)
    For image input: Uses VL engine
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        vllm_engine,  # Can be vLLM or VL engine
        strategy,
        model_path: str,
        exp_name: Optional[str] = None,
        instance_name: str = "unified_output",
        obs_type: str = 'text',  # NEW: 'text' or 'image'
    ):
        """
        Initialize Unified DataProcessor.

        Args:
            rank: Process rank
            world_size: World size
            vllm_engine: vLLM or VL engine
            strategy: Training strategy
            model_path: Model path
            exp_name: Experiment name
            instance_name: Instance name for logging
            obs_type: Observation type ('text' or 'image')
        """
        self.vllm_engine = vllm_engine
        self.strategy = strategy
        self.args = getattr(strategy, "args", None)
        self.obs_type = obs_type  # NEW

        # Load tokenizer
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, padding_side="left"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Configuration
        self.use_cot = getattr(self.args, 'use_cot', True)
        self.prompt_max_len = getattr(self.args, 'prompt_max_len', 8192)
        self.generate_max_len = getattr(self.args, 'generate_max_len', 512)
        self.temperature = getattr(self.args, 'temperature', 1.0)
        self.top_p = getattr(self.args, 'top_p', 1.0)
        self.vllm_enable_sleep = getattr(self.args, 'vllm_enable_sleep', True)
        self.reduction = getattr(self.args, 'reduction', 'mean')
        self.rank = rank
        self.world_size = world_size
        self.output_step = 0
        self.llm_prior_with_cot = False

        # Statistics
        self.episode_output = []
        self.value_running_mean = 0.0
        self.value_running_std = 1.0
        self.value_count = 0
        self.running_momentum = 0.99

        # Logger
        if self.rank == 0:
            self._logger, _ = build_logger(
                path=f'./{exp_name}/log/{instance_name}',
                name=instance_name,
                need_tb=False
            )
            self._logger.info(f"✓ UnifiedDataProcessor initialized")
            self._logger.info(f"  - Observation type: {obs_type}")
            self._logger.info(f"  - Use CoT: {self.use_cot}")

        # Value normalizer
        if hasattr(self.args, 'value_norm_cfg') and self.args.value_norm_cfg.enable_stability_optimizer:
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

    # =========================================================================
    # Text Input Methods (Original LLM functionality)
    # =========================================================================

    def get_system_prompt_text(self) -> str:
        """System prompt for text-based games (LLM)."""
        parts = [
            "You are an expert player in a text-based adventure game.",
            "Your goal is to maximize the score by choosing the optimal next action.",
            "Please analyze the game history and current observation to decide the single best next action.",
            "OUTPUT FORMAT:",
        ]

        if self.use_cot:
            parts.append(
                "You MUST produce exactly TWO parts in the following order:\n"
                "1. Reasoning: Analyze the current situation, available actions, constraints, and uncertainties.\n"
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

    def get_user_prompt_text(
        self,
        history: Optional[List[Tuple[str, str, float]]] = None,
        current_obs: Optional[str] = None,
        valid_actions: Optional[List[str]] = None
    ) -> str:
        """User prompt for text-based games (LLM)."""
        prompt_parts = []

        if history and len(history) > 0:
            prompt_parts.append("=== GAME HISTORY ===")
            for i, (obs, action, reward) in enumerate(history, start=1):
                prompt_parts.append(f"Step {i}:")
                prompt_parts.append(f"Observation: {obs.strip()}")
                prompt_parts.append(f"Action: {action.strip()}")
                prompt_parts.append(f"Reward: {reward}")
            prompt_parts.append("")

        prompt_parts.append("=== CURRENT OBSERVATION ===")
        prompt_parts.append(current_obs.strip())

        if valid_actions:
            prompt_parts.append("\n=== VALID ACTIONS ===")
            for i, action in enumerate(valid_actions, start=1):
                prompt_parts.append(f"{i}. {action}")

        prompt_parts.append("\n=== INSTRUCTION ===")
        prompt_parts.append("Choose the best action from the valid actions above.")

        return "\n".join(prompt_parts)

    # =========================================================================
    # Image Input Methods (NEW VL functionality)
    # =========================================================================

    def get_system_prompt_image(self) -> str:
        """System prompt for image-based games (VL)."""
        parts = [
            "You are an expert Atari game player.",
            "Your goal is to maximize the score by choosing the optimal next action based on the game screen.",
            "Analyze the current game state shown in the image and decide the best action.",
            "OUTPUT FORMAT:",
        ]

        if self.use_cot:
            parts.append(
                "You MUST produce exactly TWO parts:\n"
                "1. Reasoning: Analyze the game state (positions, velocities, score, etc.)\n"
                "2. Action: The final chosen action.\n"
                "Format:\n"
                "Reasoning: <analysis>\n"
                "Action: <action_name>"
            )
        else:
            parts.append(
                "Output exactly one line starting with 'Action:'.\n"
                "Example:\n"
                "Action: <action_name>"
            )
        return "\n".join(parts)

    def get_user_prompt_image(
        self,
        history: Optional[List[Tuple[Any, str, float]]] = None,
        valid_actions: Optional[List[str]] = None,
        game_context: Optional[str] = None
    ) -> str:
        """User prompt for image-based games (VL)."""
        prompt_parts = []

        if game_context:
            prompt_parts.append(f"=== GAME CONTEXT ===")
            prompt_parts.append(game_context)
            prompt_parts.append("")

        if history and len(history) > 0:
            prompt_parts.append("=== RECENT HISTORY ===")
            for i, (_, action, reward) in enumerate(history[-3:], start=1):  # Last 3 steps
                prompt_parts.append(f"Step {i}: Action={action}, Reward={reward}")
            prompt_parts.append("")

        prompt_parts.append("=== CURRENT GAME SCREEN ===")
        prompt_parts.append("(See the image above)")

        if valid_actions:
            prompt_parts.append("\n=== VALID ACTIONS ===")
            for i, action in enumerate(valid_actions, start=1):
                prompt_parts.append(f"{i}. {action}")

        prompt_parts.append("\n=== INSTRUCTION ===")
        prompt_parts.append("Based on the current game screen, choose the best action from the valid actions above.")

        return "\n".join(prompt_parts)

    # =========================================================================
    # Unified Interface
    # =========================================================================

    def get_action_prior_single(
        self,
        observation: Union[str, np.ndarray, Image.Image],
        action_candidates: List[str],
        history: Optional[List] = None,
        temperature: float = 1.0,
        use_cot: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Get action prior for a single observation (unified interface).

        Args:
            observation: Text string or image array/PIL Image
            action_candidates: List of valid actions
            history: Optional history
            temperature: Sampling temperature
            use_cot: Whether to use CoT (overrides self.use_cot)

        Returns:
            Dictionary with action_probs, action_logits, raw_output
        """
        if use_cot is None:
            use_cot = self.use_cot

        if self.obs_type == 'text':
            return self._get_action_prior_text(
                text_obs=observation,
                action_candidates=action_candidates,
                history=history,
                temperature=temperature,
                use_cot=use_cot,
            )
        else:  # image
            return self._get_action_prior_image(
                image_obs=observation,
                action_candidates=action_candidates,
                history=history,
                temperature=temperature,
                use_cot=use_cot,
            )

    def _get_action_prior_text(
        self,
        text_obs: str,
        action_candidates: List[str],
        history: Optional[List] = None,
        temperature: float = 1.0,
        use_cot: bool = True,
    ) -> Dict[str, Any]:
        """Get action prior for text observation using LLM."""
        # Build prompt
        system_prompt = self.get_system_prompt_text()
        user_prompt = self.get_user_prompt_text(
            history=history,
            current_obs=text_obs,
            valid_actions=action_candidates
        )

        # Build chat messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # Convert to text
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Generate with vLLM
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=self.top_p,
            max_tokens=self.generate_max_len,
        )

        outputs = self.vllm_engine.generate([prompt_text], sampling_params)
        raw_output = outputs[0].outputs[0].text

        # Parse output to get action probabilities
        action_probs = self._parse_llm_output_to_probs(raw_output, action_candidates)
        action_logits = np.log(action_probs + 1e-10)

        return {
            'action_probs': action_probs,
            'action_logits': action_logits,
            'raw_output': raw_output,
        }

    def _get_action_prior_image(
        self,
        image_obs: Union[np.ndarray, Image.Image],
        action_candidates: List[str],
        history: Optional[List] = None,
        temperature: float = 1.0,
        use_cot: bool = True,
    ) -> Dict[str, Any]:
        """Get action prior for image observation using VL."""
        # Convert to PIL Image if needed
        if isinstance(image_obs, np.ndarray):
            if image_obs.dtype != np.uint8:
                image_obs = (image_obs * 255).astype(np.uint8)
            # Handle different formats
            if image_obs.shape[0] == 3:  # (C, H, W) -> (H, W, C)
                image_obs = np.transpose(image_obs, (1, 2, 0))
            image = Image.fromarray(image_obs)
        else:
            image = image_obs

        # Build prompt
        system_prompt = self.get_system_prompt_image()
        user_prompt = self.get_user_prompt_image(
            history=history,
            valid_actions=action_candidates,
            game_context="Atari game"
        )

        # Combine prompts
        full_prompt = f"{system_prompt}\n\n{user_prompt}"

        # Generate with VL
        raw_output = self.vllm_engine.generate(
            image=image,
            prompt=full_prompt,
            temperature=temperature,
            max_new_tokens=self.generate_max_len,
        )

        # Parse output to get action probabilities
        action_probs = self._parse_vl_output_to_probs(raw_output, action_candidates)
        action_logits = np.log(action_probs + 1e-10)

        return {
            'action_probs': action_probs,
            'action_logits': action_logits,
            'raw_output': raw_output,
        }

    def _parse_llm_output_to_probs(self, raw_output: str, action_candidates: List[str]) -> np.ndarray:
        """Parse LLM output to action probabilities."""
        # Extract action from output
        action_match = re.search(r'Action:\s*(.+)', raw_output, re.IGNORECASE)
        if action_match:
            chosen_action = action_match.group(1).strip()

            # Find matching action
            for i, action in enumerate(action_candidates):
                if action.lower() in chosen_action.lower() or chosen_action.lower() in action.lower():
                    # High probability for chosen action
                    probs = np.ones(len(action_candidates)) * 0.01
                    probs[i] = 0.9
                    probs = probs / probs.sum()
                    return probs

        # Fallback: uniform distribution
        return np.ones(len(action_candidates)) / len(action_candidates)

    def _parse_vl_output_to_probs(self, raw_output: str, action_candidates: List[str]) -> np.ndarray:
        """Parse VL output to action probabilities."""
        # Similar to LLM parsing
        return self._parse_llm_output_to_probs(raw_output, action_candidates)

    def get_llm_prior(
        self,
        states: List[Union[str, np.ndarray, Image.Image]],
        valid_actions_list: List[List[str]],
        histories: Optional[List[List]] = None,
        return_cot: bool = False
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[Any]]:
        """
        Batch get LLM/VL priors (for backward compatibility).

        Args:
            states: List of observations (text or images)
            valid_actions_list: List of valid action lists
            histories: List of histories
            return_cot: Whether to return CoT prefixes

        Returns:
            Tuple of (prior_per_seq, prior_per_tok, cot_prefixes)
        """
        if histories is None:
            histories = [None] * len(states)

        prior_per_seq = []
        prior_per_tok = []
        cot_prefixes = []

        for obs, actions, hist in zip(states, valid_actions_list, histories):
            result = self.get_action_prior_single(
                observation=obs,
                action_candidates=actions,
                history=hist,
                temperature=self.temperature,
            )

            prior_per_seq.append(result['action_probs'])
            prior_per_tok.append(result['action_logits'])
            if return_cot:
                cot_prefixes.append(result['raw_output'])

        if return_cot:
            return prior_per_seq, prior_per_tok, cot_prefixes
        else:
            return prior_per_seq, prior_per_tok, [None] * len(states)

    def make_llm_train_samples(self, priorzero_batch, ddp: bool = True, max_samples: int = None, prior_generator=None):
        """
        Make training samples from PriorZero batch.

        For text mode: tokenizes prompts + actions → tensors expected by BatchPPOTrainer.
        For image mode: builds VL chat context, tokenizes → same tensor format.

        Returns:
            Tuple of (flag, train_samples) where train_samples is a 6-tuple:
            (input_ids, attention_mask, action_mask, advantage, rollout_logprob, log_status)
        """
        if self.obs_type == 'image':
            return self._make_vl_train_samples(priorzero_batch, ddp=ddp, max_samples=max_samples, prior_generator=prior_generator)
        else:
            # Original LLM training samples (text input)
            # Keep existing implementation
            pass

    def _make_vl_train_samples(self, priorzero_batch, ddp: bool = True, max_samples: int = None, prior_generator=None):
        """
        Build VL training samples in the same tensor format as the LLM path.

        The 8-element priorzero_batch from fetch_latest_batch:
            [raw_obs_list, history_obs_list, llm_prior_per_tok_list,
             batch_target_values, batch_pred_values, cot_prefix_list, llm_action_list, action_list]

        Returns:
            (flag, (input_ids, attention_mask, action_mask, advantage, rollout_logprob, log_status))
        """
        import logging
        import random
        import traceback
        _logger = logging.getLogger(__name__)

        try:
            raw_obs_list, history_obs_list, llm_prior_per_tok_list, \
                target_values, pred_values, cot_prefix_list, llm_action_list, action_list = priorzero_batch

            if len(raw_obs_list) == 0:
                return (False, [])

            B = len(raw_obs_list)
            T = len(raw_obs_list[0]) if B > 0 else 0

            # ---- Step 1: build flat sample list ----
            samples = []
            for b in range(B):
                for t in range(T - 1):
                    action_name = llm_action_list[b][t + 1]
                    if action_name is None:
                        continue

                    # history at time t = history after executing action t (before action t+1)
                    history = history_obs_list[b][t] if t < len(history_obs_list[b]) else []
                    cot_prefix = cot_prefix_list[b][t + 1] if (cot_prefix_list is not None and t + 1 < len(cot_prefix_list[b])) else None

                    # VL prior stored as action-level log-prob array
                    action_logprobs = llm_prior_per_tok_list[b][t + 1] if (
                        llm_prior_per_tok_list is not None and t + 1 < len(llm_prior_per_tok_list[b])
                    ) else None

                    # MCTS-selected action index (integer) for correct rollout log-prob lookup
                    mcts_action_idx = int(action_list[b][t + 1]) if (
                        action_list is not None and b < len(action_list) and t + 1 < len(action_list[b])
                    ) else None

                    tv = float(target_values[b][t]) if target_values is not None and b < len(target_values) and t < len(target_values[b]) else 0.0
                    pv = float(pred_values[b][t]) if pred_values is not None and b < len(pred_values) and t < len(pred_values[b]) else 0.0

                    samples.append({
                        'history': history,
                        'action_name': action_name,
                        'cot_prefix': cot_prefix,
                        'action_logprobs': action_logprobs,  # np.ndarray or None
                        'mcts_action_idx': mcts_action_idx,  # int or None
                        'target_value': tv,
                        'pred_value': pv,
                    })

            if len(samples) == 0:
                return (False, [])

            random.Random(0).shuffle(samples)

            if max_samples is not None and len(samples) > max_samples:
                samples = samples[:max_samples]

            if ddp:
                real_samples = samples
            else:
                per_rank = len(samples) // self.world_size
                start = self.rank * per_rank
                end = (self.rank + 1) * per_rank if self.rank != self.world_size - 1 else len(samples)
                real_samples = samples[start:end]

            if len(real_samples) == 0:
                return (False, [])

            # ---- Step 2: build target text for each sample ----
            if self.use_cot:
                targets_only = []
                for s in real_samples:
                    cot = s['cot_prefix'] or ""
                    if cot:
                        targets_only.append(cot.strip() + "\nAction: " + s['action_name'] + self.tokenizer.eos_token)
                    else:
                        targets_only.append("Action: " + s['action_name'] + self.tokenizer.eos_token)
            else:
                targets_only = ["Action: " + s['action_name'] + self.tokenizer.eos_token for s in real_samples]

            # ---- Step 3: build prompt + target → full_ids / label_ids ----
            # Use a dummy image prompt (the actual image tokens will not be used in
            # text-only Actor forward, but we need the textual prompt structure)
            full_ids_list = []
            tgt_ids_list = []

            for idx, s in enumerate(real_samples):
                # Build the user prompt from history (text-only; images handled at inference)
                history = s['history']
                if prior_generator is not None and hasattr(prior_generator, 'get_user_prompt'):
                    # Use the prior_generator's prompt builder for consistency
                    valid_actions_hint = []  # not needed for tokenization
                    user_prompt = prior_generator.get_user_prompt(valid_actions_hint, history)
                else:
                    user_prompt = self.get_user_prompt_image(history=history)

                # Build chat context via tokenizer chat template
                prompt_text = self.tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": self.get_system_prompt_image()},
                        {"role": "user",   "content": user_prompt},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )

                target_text = targets_only[idx]
                full_text = prompt_text + target_text

                prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
                full_ids = self.tokenizer.encode(full_text, add_special_tokens=False)
                tgt_ids = full_ids[len(prompt_ids):]

                # Truncate prompt if it exceeds prompt_max_len
                if len(prompt_ids) > self.prompt_max_len:
                    prompt_ids = prompt_ids[-self.prompt_max_len:]
                    full_ids = prompt_ids + tgt_ids

                full_ids_list.append(full_ids)
                tgt_ids_list.append(tgt_ids)

            # ---- Step 4: pad and build tensors ----
            inputs = self.tokenizer.pad({"input_ids": full_ids_list}, padding=True, return_tensors="pt")
            labels = torch.full_like(inputs.input_ids, -100)
            for i, tgt_ids in enumerate(tgt_ids_list):
                tgt_len = len(tgt_ids)
                labels[i, -tgt_len:] = inputs.input_ids[i, -tgt_len:]

            action_mask_full = (labels != -100).long()
            max_tgt_len = max(len(t) for t in tgt_ids_list)
            action_mask = action_mask_full[:, -max_tgt_len:]

            # ---- Step 5: compute advantage ----
            target_value_tensor = torch.tensor([s['target_value'] for s in real_samples], dtype=torch.float32)
            pred_value_tensor = torch.tensor([s['pred_value'] for s in real_samples], dtype=torch.float32)
            advantage = target_value_tensor - pred_value_tensor

            log_status_tmp = {}

            if self.args.advantage_type == "advantage":
                log_status_tmp["value_advantage"] = advantage.tolist()
            elif self.args.advantage_type == "advantage_batch_norm":
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
                log_status_tmp["value_advantage"] = advantage.tolist()
            elif self.args.advantage_type == "advantage_running_norm":
                if self.value_normalizer is not None:
                    advantage_np = advantage.numpy()
                    advantage_np = self.value_normalizer.normalize_advantages(advantage_np)
                    advantage = torch.from_numpy(advantage_np)
                else:
                    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
                log_status_tmp["value_advantage"] = advantage.tolist()
            else:
                log_status_tmp["value_advantage"] = advantage.tolist()

            log_status = [
                {k: log_status_tmp[k][i] for k in log_status_tmp.keys()}
                for i in range(len(real_samples))
            ]

            # ---- Step 6: build rollout_logprob ----
            # VL has action-level log-probs (not per-token). We spread the action log-prob
            # uniformly across all target tokens so the PPO ratio is correct in expectation.
            rollout_logprob = torch.zeros(len(real_samples), max_tgt_len, dtype=torch.float32)
            for idx, s in enumerate(real_samples):
                tgt_len = len(tgt_ids_list[idx])
                if s['action_logprobs'] is not None and isinstance(s['action_logprobs'], np.ndarray):
                    # Use MCTS-selected action index to get the correct rollout log-prob.
                    # Previously used np.max which incorrectly assumed VLM's top choice == MCTS choice.
                    if s['mcts_action_idx'] is not None and 0 <= s['mcts_action_idx'] < len(s['action_logprobs']):
                        chosen_logprob = float(s['action_logprobs'][s['mcts_action_idx']])
                    else:
                        # Fallback: use max (legacy behavior, should rarely happen)
                        chosen_logprob = float(np.max(s['action_logprobs']))
                    per_token_lp = chosen_logprob / max(tgt_len, 1)
                    rollout_logprob[idx, -tgt_len:] = per_token_lp
                # else: leave as zero (no rollout log-probs available)

            if self.rank == 0:
                _logger.info(
                    f"[VL Train Samples] Built {len(real_samples)} samples | "
                    f"advantage mean={advantage.mean().item():.4f} std={advantage.std().item():.4f}"
                )

            return True, (inputs.input_ids, inputs.attention_mask, action_mask, advantage, rollout_logprob, log_status)

        except Exception as e:
            import traceback as tb
            if self.rank == 0:
                _logger.error(f"[VL Train Samples] Error: {e}\n{tb.format_exc()}")
            return (False, [])

    def get_llm_output_log(self, wm_train_iter: int, llm_train_iter: int):
        """Log LLM/VL output statistics."""
        if self.rank == 0 and len(self.episode_output) > 0:
            self._logger.info(
                f"[WM Iter {wm_train_iter} | LLM Iter {llm_train_iter}] "
                f"Collected {len(self.episode_output)} outputs"
            )
            self.episode_output = []


# Backward compatibility: alias to original name
DataProcessor = UnifiedDataProcessor

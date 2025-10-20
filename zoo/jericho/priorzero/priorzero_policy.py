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
from typing import List, Dict, Any, Tuple, Union, Optional

import numpy as np
import torch
import torch.nn.functional as F
from ding.utils import POLICY_REGISTRY
from ding.model import model_wrap
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType

# Import from LightZero
from lzero.policy.unizero import UniZeroPolicy as OriginalUniZeroPolicy
from lzero.policy import phi_transform, InverseScalarTransform, to_torch_float_tensor, mz_network_output_unpack
from lzero.mcts import UniZeroMCTSCtree as MCTSCtree
from lzero.entry.utils import initialize_zeros_batch


# ==============================================================================
# Helper Functions for LLM Prior Processing
# ==============================================================================

def parse_llm_action_ranking(
    text: str,
    action_map: Dict[str, int],
    action_space_size: int,
    fallback_to_uniform: bool = True
) -> np.ndarray:
    """
    [PRIORZERO-NEW]
    Parse LLM generated action ranking text into a policy distribution.

    Args:
        text: LLM generated text with ranked actions (e.g., "1. take key\\n2. go north")
        action_map: Mapping from action text to action index
        action_space_size: Size of the action space
        fallback_to_uniform: If True, return uniform distribution when no valid action found

    Returns:
        policy: Probability distribution over actions (shape: [action_space_size])
    """
    # Extract ranked actions using regex
    # Supports formats: "1. action", "1) action", "1: action"
    ranked_actions = re.findall(r'(?:^|\n)\s*\d+[\.\):\s]+(.+?)(?=\n|$)', text, re.MULTILINE)

    policy = np.zeros(action_space_size, dtype=np.float32)
    found_count = 0

    for rank, action_text in enumerate(ranked_actions):
        action_text = action_text.strip().lower()

        # Try exact match first
        if action_text in action_map:
            action_idx = action_map[action_text]
            # Assign decreasing weights (higher rank = higher weight)
            policy[action_idx] = len(ranked_actions) - rank
            found_count += 1
        else:
            # Try fuzzy matching (find best substring match)
            best_match_score = 0
            best_action_idx = None
            for candidate_text, candidate_idx in action_map.items():
                if candidate_text in action_text or action_text in candidate_text:
                    score = len(set(candidate_text.split()) & set(action_text.split()))
                    if score > best_match_score:
                        best_match_score = score
                        best_action_idx = candidate_idx

            if best_action_idx is not None:
                policy[best_action_idx] = len(ranked_actions) - rank
                found_count += 1

    # Normalize to probability distribution
    if policy.sum() > 0:
        policy /= policy.sum()
    elif fallback_to_uniform:
        # If LLM didn't generate any valid actions, return uniform distribution
        policy = np.ones(action_space_size, dtype=np.float32) / action_space_size

    return policy


def format_mcts_policy_to_text(
    mcts_policy: np.ndarray,
    action_inv_map: Dict[int, str],
    top_k: int = 5
) -> str:
    """
    [PRIORZERO-NEW]
    Convert MCTS policy vector into ranked action text for SFT training.

    Args:
        mcts_policy: MCTS visit count distribution (shape: [action_space_size])
        action_inv_map: Mapping from action index to action text
        top_k: Number of top actions to include

    Returns:
        Formatted text with ranked actions (e.g., "1. take key\\n2. go north\\n...")
    """
    # Sort actions by policy probability (descending)
    sorted_indices = np.argsort(mcts_policy)[::-1]

    output_lines = []
    rank = 1
    for idx in sorted_indices:
        if mcts_policy[idx] > 0 and rank <= top_k:
            action_text = action_inv_map.get(idx, f"action_{idx}")
            output_lines.append(f"{rank}. {action_text}")
            rank += 1

    return "\n".join(output_lines) if output_lines else "No valid actions found."


def build_llm_prompt(
    current_obs: str,
    history: Optional[List[Tuple[str, str, float]]] = None,
    action_descriptions: Optional[Dict[str, str]] = None,
    use_cot: bool = True
) -> str:
    """
    [PRIORZERO-NEW]
    Build a high-quality prompt for LLM to generate action ranking.

    Args:
        current_obs: Current observation text
        history: List of (observation, action, reward) tuples
        action_descriptions: Optional descriptions for each action
        use_cot: Whether to encourage chain-of-thought reasoning

    Returns:
        Formatted prompt string
    """
    prompt_parts = []

    # System instruction
    prompt_parts.append(
        "You are an expert player in a text-based adventure game. "
        "Your goal is to maximize the score by taking the best actions."
    )

    # Add history if available
    if history and len(history) > 0:
        prompt_parts.append("\n=== Recent History ===")
        for i, (obs, action, reward) in enumerate(history[-5:]):  # Last 5 steps
            prompt_parts.append(f"Step {i+1}:")
            prompt_parts.append(f"  Observation: {obs[:100]}...")  # Truncate long obs
            prompt_parts.append(f"  Action: {action}")
            prompt_parts.append(f"  Reward: {reward}")

    # Current observation
    prompt_parts.append("\n=== Current Situation ===")
    prompt_parts.append(current_obs)

    # Task instruction
    if use_cot:
        prompt_parts.append(
            "\n=== Task ===\n"
            "Think step-by-step:\n"
            "1. Analyze the current situation and your goal\n"
            "2. Consider what actions might help you progress\n"
            "3. Rank the best actions in order of priority\n"
            "\nProvide your analysis and then list the top 5 actions in this format:\n"
            "1. [first action]\n"
            "2. [second action]\n"
            "..."
        )
    else:
        prompt_parts.append(
            "\n=== Task ===\n"
            "List the top 5 best actions in order of priority:\n"
            "1. [first action]\n"
            "2. [second action]\n"
            "..."
        )

    return "\n".join(prompt_parts)


# ==============================================================================
# PriorZero Policy Class
# ==============================================================================

@POLICY_REGISTRY.register('priorzero')
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

    def __init__(self, cfg: Dict, model: torch.nn.Module = None, enable_field: List[str] = None):
        super().__init__(cfg, model, enable_field)

        # [PRIORZERO-NEW] LLM-related attributes
        self.llm_policy_model = None
        self.llm_tokenizer = None
        self._optimizer_llm = None
        self._lr_scheduler_llm = None
        self.llm_policy_cfg = self._cfg.llm_policy_cfg

        # Action mapping (will be set from config)
        self.action_map = None  # str -> int
        self.action_inv_map = None  # int -> str

    def _init_learn(self) -> None:
        """
        [PRIORZERO-MODIFIED]
        Initialize both UniZero world model and LLM policy model with their optimizers.
        """
        # ======================================================================
        # 1. Initialize UniZero World Model (from parent class)
        # ======================================================================
        super()._init_learn()
        self._logger.info("✓ UniZero World Model and optimizer initialized")

        # ======================================================================
        # 2. [PRIORZERO-NEW] Initialize LLM Policy Model
        # ======================================================================
        self._logger.info(f"Loading LLM from: {self.llm_policy_cfg.pretrain_llm_path}")

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
            self._logger.info("Applying LoRA for parameter-efficient fine-tuning")
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

        self._logger.info(f"✓ LLM Policy Model ({self.llm_policy_cfg.pretrain_llm_path}) initialized")
        self._logger.info(f"  - LLM learning rate: {self.llm_policy_cfg.llm_learning_rate}")
        self._logger.info(f"  - LoRA enabled: {self.llm_policy_cfg.use_lora}")

        # ======================================================================
        # 4. [PRIORZERO-NEW] Load Action Mappings
        # ======================================================================
        if hasattr(self._cfg, 'action_map') and self._cfg.action_map is not None:
            self.action_map = self._cfg.action_map
            self.action_inv_map = {v: k for k, v in self.action_map.items()}
            self._logger.info(f"✓ Action mappings loaded ({len(self.action_map)} actions)")
        else:
            self._logger.warning("⚠ Action mappings not found in config. Will use index-based actions.")
            # Fallback: create dummy mappings
            action_space_size = self._cfg.model.action_space_size
            self.action_inv_map = {i: f"action_{i}" for i in range(action_space_size)}
            self.action_map = {v: k for k, v in self.action_inv_map.items()}

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
        self.llm_policy_model.train()

        # Unpack data
        # NOTE: game_segments is our custom GameSegment with mcts_policy_segment
        current_batch, target_batch, train_iter, game_segments = data

        # ==============================================================================
        # Part 1: UniZero World Model Training (Full Implementation)
        # ==============================================================================

        # Unpack batches
        (obs_batch_ori, action_batch, mask_batch, batch_index_tensor,
         weights, make_time) = current_batch[:6]
        target_reward, target_value, target_policy = target_batch

        # Handle optional timestep
        if len(current_batch) > 6:
            timestep_batch = current_batch[6]
        else:
            timestep_batch = None

        # Convert to tensors and move to device
        data_list = [mask_batch, target_reward, target_value, target_policy, weights]
        (mask_batch, target_reward, target_value,
         target_policy, weights) = to_torch_float_tensor(data_list, self._cfg.device)

        # Reshape targets
        batch_size = self._cfg.batch_size
        target_reward = target_reward.view(batch_size, -1)
        target_value = target_value.view(batch_size, -1)

        # Apply scalar transform (for value and reward)
        transformed_target_reward = self.scalar_transform(target_reward)
        transformed_target_value = self.scalar_transform(target_value)

        # Convert to categorical distribution (for distributional RL)
        target_reward_categorical = phi_transform(
            self.reward_support, transformed_target_reward
        )
        target_value_categorical = phi_transform(
            self.value_support, transformed_target_value
        )

        # Prepare batch for world model
        # NOTE: This follows the exact format required by UniZero world model
        if timestep_batch is not None:
            batch_for_gpt = {
                'observations': obs_batch_ori,
                'actions': action_batch.squeeze(-1).long(),
                'timestep': timestep_batch.squeeze(-1).long(),
                'rewards': target_reward_categorical[:, :-1],
                'mask_padding': (mask_batch == 1.0)[:, :-1],
                'target_value': target_value_categorical[:, :-1],
                'target_policy': target_policy[:, :-1],
            }
        else:
            batch_for_gpt = {
                'observations': obs_batch_ori,
                'actions': action_batch.squeeze(-1).long(),
                'rewards': target_reward_categorical[:, :-1],
                'mask_padding': (mask_batch == 1.0)[:, :-1],
                'target_value': target_value_categorical[:, :-1],
                'target_policy': target_policy[:, :-1],
            }

        # Compute world model loss
        wm_losses = self._learn_model.world_model.compute_loss(
            batch_for_gpt,
            self._target_model.world_model.tokenizer,
            self.value_inverse_scalar_transform_handle,
        )

        # Weighted world model loss (for prioritized experience replay)
        wm_total_loss = (weights * wm_losses.loss_total).mean()

        # ==============================================================================
        # Part 2: [PRIORZERO-NEW] LLM Policy Training (SFT + RFT)
        # ==============================================================================

        llm_sft_loss = torch.tensor(0.0, device=self._cfg.device)
        llm_rft_loss = torch.tensor(0.0, device=self._cfg.device)
        num_sft_samples = 0
        num_rft_samples = 0

        # Collect training data from game segments
        sft_prompts = []
        sft_targets = []
        rft_prompts = []
        rft_rewards = []

        for segment in game_segments:
            segment_length = len(segment.obs_segment)

            for i in range(segment_length):
                # Skip if no MCTS policy available
                if segment.mcts_policy_segment[i] is None:
                    continue

                # Get raw observation text (assume it's stored in obs_segment)
                # NOTE: For text environments, obs_segment should contain text
                raw_obs_text = str(segment.obs_segment[i])

                # Build history context
                history = []
                for j in range(max(0, i - self.llm_policy_cfg.history_length), i):
                    if j < len(segment.obs_segment):
                        history.append((
                            str(segment.obs_segment[j]),
                            self.action_inv_map.get(segment.action_segment[j], f"action_{segment.action_segment[j]}"),
                            float(segment.reward_segment[j]) if j < len(segment.reward_segment) else 0.0
                        ))

                # Build prompt
                instruction = build_llm_prompt(
                    current_obs=raw_obs_text,
                    history=history,
                    use_cot=self.llm_policy_cfg.use_cot
                )

                # Apply chat template
                prompt = self.llm_tokenizer.apply_chat_template(
                    [{"role": "user", "content": instruction}],
                    tokenize=False,
                    add_generation_prompt=True
                )

                # ============================================================
                # SFT: Supervised Fine-Tuning with MCTS Policy
                # ============================================================
                if self.llm_policy_cfg.sft_target == 'mcts_policy':
                    mcts_policy_vec = segment.mcts_policy_segment[i]

                    # Convert MCTS policy to ranked action text
                    target_text = format_mcts_policy_to_text(
                        mcts_policy_vec,
                        self.action_inv_map,
                        top_k=5
                    )

                    sft_prompts.append(prompt)
                    sft_targets.append(target_text)
                    num_sft_samples += 1

                # ============================================================
                # RFT: Reinforcement Fine-Tuning with Environment Reward
                # ============================================================
                if self.llm_policy_cfg.enable_rft and i < len(segment.reward_segment):
                    env_reward = float(segment.reward_segment[i])

                    # Only use transitions with non-zero reward for RFT
                    if abs(env_reward) > 1e-6:
                        rft_prompts.append(prompt)
                        rft_rewards.append(env_reward)
                        num_rft_samples += 1

        # ============================================================
        # Train LLM with SFT
        # ============================================================
        if num_sft_samples > 0:
            # Prepare full texts (prompt + target + eos)
            full_texts = [
                p + t + self.llm_tokenizer.eos_token
                for p, t in zip(sft_prompts, sft_targets)
            ]

            # Tokenize
            inputs = self.llm_tokenizer(
                full_texts,
                padding=True,
                truncation=True,
                max_length=self.llm_policy_cfg.prompt_max_len,
                return_tensors="pt"
            ).to(self._cfg.device)

            # Create labels (mask prompt tokens to only compute loss on target)
            labels = inputs.input_ids.clone()
            labels[labels == self.llm_tokenizer.pad_token_id] = -100

            # Mask prompt tokens
            for i in range(len(sft_prompts)):
                prompt_tokens = self.llm_tokenizer.encode(sft_prompts[i], add_special_tokens=False)
                prompt_len = len(prompt_tokens)
                labels[i, :prompt_len] = -100

            # Forward pass
            llm_outputs = self.llm_policy_model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                labels=labels
            )
            llm_sft_loss = llm_outputs.loss

        # ============================================================
        # Train LLM with RFT (Policy Gradient)
        # ============================================================
        if num_rft_samples > 0 and self.llm_policy_cfg.enable_rft:
            # Tokenize prompts
            inputs = self.llm_tokenizer(
                rft_prompts,
                padding=True,
                truncation=True,
                max_length=self.llm_policy_cfg.prompt_max_len,
                return_tensors="pt"
            ).to(self._cfg.device)

            # Forward pass to get logits
            with torch.no_grad():
                outputs = self.llm_policy_model(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask
                )

            # Compute policy gradient loss (REINFORCE)
            # Loss = -reward * log_prob(action)
            logits = outputs.logits
            log_probs = F.log_softmax(logits, dim=-1)

            # Get log probability of actual tokens
            shifted_log_probs = log_probs[:, :-1, :].contiguous()
            shifted_labels = inputs.input_ids[:, 1:].contiguous()

            # Gather log probs of actual tokens
            token_log_probs = shifted_log_probs.gather(
                dim=-1,
                index=shifted_labels.unsqueeze(-1)
            ).squeeze(-1)

            # Mask padding tokens
            mask = (shifted_labels != self.llm_tokenizer.pad_token_id).float()
            token_log_probs = token_log_probs * mask

            # Sum log probs per sequence
            sequence_log_probs = token_log_probs.sum(dim=-1) / (mask.sum(dim=-1) + 1e-8)

            # Compute REINFORCE loss
            rewards_tensor = torch.tensor(
                rft_rewards,
                device=self._cfg.device,
                dtype=torch.float32
            )

            # Normalize rewards (important for stable training)
            rewards_tensor = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)

            llm_rft_loss = -(rewards_tensor * sequence_log_probs).mean()

        # ==============================================================================
        # Part 3: Joint Optimization
        # ==============================================================================

        # Combine losses
        llm_loss = (
            self.llm_policy_cfg.llm_loss_weight * llm_sft_loss +
            self.llm_policy_cfg.rft_loss_weight * llm_rft_loss
        )
        total_loss = wm_total_loss + llm_loss

        # Zero gradients
        self._optimizer_world_model.zero_grad()
        self._optimizer_llm.zero_grad()

        # Backward pass
        total_loss.backward()

        # Gradient clipping
        wm_grad_norm = torch.nn.utils.clip_grad_norm_(
            self._learn_model.world_model.parameters(),
            self._cfg.grad_clip_value
        )
        llm_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.llm_policy_model.parameters(),
            self._cfg.grad_clip_value
        )

        # Optimizer step
        self._optimizer_world_model.step()
        self._optimizer_llm.step()

        # Learning rate scheduler step (optional)
        if self._lr_scheduler_llm is not None:
            self._lr_scheduler_llm.step()

        # Update target model (soft update)
        self._target_model.update(self._learn_model.state_dict())

        # ==============================================================================
        # Part 4: Logging
        # ==============================================================================

        # Get base logs from parent class
        # NOTE: We need to extract individual loss components from wm_losses
        log_dict = {
            # World model losses
            'wm_total_loss': wm_total_loss.item(),
            'wm_value_loss': wm_losses.value_loss.mean().item() if hasattr(wm_losses, 'value_loss') else 0.0,
            'wm_policy_loss': wm_losses.policy_loss.mean().item() if hasattr(wm_losses, 'policy_loss') else 0.0,
            'wm_reward_loss': wm_losses.reward_loss.mean().item() if hasattr(wm_losses, 'reward_loss') else 0.0,

            # LLM losses
            'llm_sft_loss': llm_sft_loss.item(),
            'llm_rft_loss': llm_rft_loss.item(),
            'llm_total_loss': llm_loss.item(),

            # Combined
            'total_loss': total_loss.item(),

            # Gradient norms
            'wm_grad_norm': wm_grad_norm.item(),
            'llm_grad_norm': llm_grad_norm.item(),

            # Sample counts
            'num_sft_samples': num_sft_samples,
            'num_rft_samples': num_rft_samples,

            # Learning rates
            'wm_lr': self._optimizer_world_model.param_groups[0]['lr'],
            'llm_lr': self._optimizer_llm.param_groups[0]['lr'],
        }

        return log_dict

    def _forward_collect(
        self,
        data: torch.Tensor,
        action_mask: List[np.ndarray],
        temperature: float = 1.0,
        to_play: List[int] = None,
        epsilon: float = 0.0,
        ready_env_id: List[int] = None,
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

        # ======================================================================
        # [PRIORZERO-NEW] Get LLM Prior Outputs
        # ======================================================================
        llm_prior_outputs = kwargs.pop('llm_prior_outputs', None)

        if llm_prior_outputs is None:
            # If no LLM prior available, fall back to standard UniZero behavior
            self._logger.debug("No LLM priors provided, using standard UniZero MCTS")
            return super()._forward_collect(
                data, action_mask, temperature, to_play, epsilon,
                ready_env_id=ready_env_id, **kwargs
            )

        # ======================================================================
        # Parse LLM Outputs into Policy Priors
        # ======================================================================
        policy_priors = []
        for output in llm_prior_outputs:
            # Extract generated text
            generated_text = output.outputs[0].text if hasattr(output, 'outputs') else str(output)

            # Parse into policy distribution
            prior_policy = parse_llm_action_ranking(
                generated_text,
                self.action_map,
                self._cfg.model.action_space_size,
                fallback_to_uniform=True
            )

            # Convert to log probabilities (for compatibility with MCTS)
            policy_logits = torch.log(torch.from_numpy(prior_policy) + 1e-9)
            policy_priors.append(policy_logits)

        policy_priors = torch.stack(policy_priors).to(self._cfg.device)

        # ======================================================================
        # World Model Initial Inference
        # ======================================================================
        with torch.no_grad():
            # Run representation network to get latent state
            network_output = self._collect_model.initial_inference(data)

            # Unpack network outputs
            latent_state_roots, reward_roots, pred_values, policy_logits_roots = \
                mz_network_output_unpack(network_output)

            # [PRIORZERO-KEY] Replace policy logits with LLM priors
            network_output.policy_logits = policy_priors

            # Prepare for MCTS
            if not self._cfg.mcts_ctree:
                # Python implementation (not recommended for performance)
                raise NotImplementedError("Python MCTS not supported for PriorZero")

            # ======================================================================
            # MCTS Search with LLM-Guided Priors
            # ======================================================================
            # This is the key part where LLM priors guide the search

            # Prepare action masks
            action_mask_np = np.array(action_mask, dtype=np.float32)

            # Get timestep if available
            timestep = kwargs.get('timestep', None)

            # Run MCTS
            policy_logits_for_mcts = policy_priors.cpu().numpy()
            latent_state_roots_np = latent_state_roots.cpu().numpy()
            reward_roots_np = reward_roots.cpu().numpy()
            pred_values_np = pred_values.cpu().numpy()

            # Create MCTS roots with LLM priors
            roots = MCTSCtree.roots(
                latent_state_roots_np.shape[0],
                self._cfg.model.action_space_size,
            )

            # Prepare root nodes
            noises = [
                np.random.dirichlet([self._cfg.root_dirichlet_alpha] *
                                    self._cfg.model.action_space_size).astype(np.float32)
                for _ in range(latent_state_roots_np.shape[0])
            ]

            roots.prepare(
                self._cfg.root_noise_weight,
                noises,
                list(pred_values_np),
                policy_logits_for_mcts.tolist(),
                to_play if to_play is not None else [-1] * latent_state_roots_np.shape[0],
            )

            # Run MCTS search
            MCTSCtree(self._cfg).search(
                roots,
                self._collect_model,
                latent_state_roots,
                reward_roots,
                to_play if to_play is not None else [-1] * latent_state_roots_np.shape[0],
            )

            # Extract search results
            roots_visit_count = roots.get_distributions()
            roots_values = roots.get_values()

            # ======================================================================
            # Select Actions and Prepare Output
            # ======================================================================
            output = {}

            for i, env_id in enumerate(ready_env_id):
                # Get visit count distribution
                visit_count_dist = np.array(roots_visit_count[i], dtype=np.float32)

                # Normalize to get policy
                if visit_count_dist.sum() > 0:
                    policy = visit_count_dist / visit_count_dist.sum()
                else:
                    policy = np.ones_like(visit_count_dist) / len(visit_count_dist)

                # Apply action mask
                masked_policy = policy * action_mask[i]
                if masked_policy.sum() > 0:
                    masked_policy /= masked_policy.sum()
                else:
                    # If all actions masked, use uniform over valid actions
                    masked_policy = action_mask[i] / action_mask[i].sum()

                # Select action (with temperature)
                if temperature == 0:
                    # Greedy selection
                    action = np.argmax(masked_policy)
                else:
                    # Sample from temperature-scaled distribution
                    action_probs = masked_policy ** (1.0 / temperature)
                    action_probs /= action_probs.sum()
                    action = np.random.choice(len(action_probs), p=action_probs)

                # Compute visit count entropy (for logging)
                entropy = -np.sum(policy * np.log(policy + 1e-9))

                output[env_id] = {
                    'action': int(action),
                    'visit_count_distributions': visit_count_dist.tolist(),
                    'visit_count_distribution_entropy': float(entropy),
                    'searched_value': float(roots_values[i]),
                    'predicted_value': float(pred_values_np[i]),
                }

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
            self._logger.info("✓ LLM model state loaded")

        if 'optimizer_llm' in state_dict:
            self._optimizer_llm.load_state_dict(state_dict['optimizer_llm'])
            self._logger.info("✓ LLM optimizer state loaded")

        if 'lr_scheduler_llm' in state_dict and self._lr_scheduler_llm is not None:
            self._lr_scheduler_llm.load_state_dict(state_dict['lr_scheduler_llm'])
            self._logger.info("✓ LLM scheduler state loaded")

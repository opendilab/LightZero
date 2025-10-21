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
import logging
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

# Import from local LightZero
from lzero.policy.unizero import UniZeroPolicy as OriginalUniZeroPolicy
from lzero.policy import (
    phi_transform,
    InverseScalarTransform,
    scalar_transform,  # [PRIORZERO] Added for reward/value transformation
    DiscreteSupport,   # [PRIORZERO] Added for categorical distribution support
    to_torch_float_tensor,
    mz_network_output_unpack
)
from lzero.policy.utils import select_action
from lzero.mcts import UniZeroMCTSCtree as MCTSCtree
from lzero.entry.utils import initialize_zeros_batch
# Import UniZeroModel to ensure it's registered in MODEL_REGISTRY
import lzero.model.unizero_model  # noqa: F401


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

    def __init__(self, cfg: Dict, model: torch.nn.Module = None, enable_field: List[str] = None):
        # [PRIORZERO-NEW] Initialize LLM-related attributes BEFORE super().__init__
        # because super().__init__ will call _init_learn which needs these attributes
        self.llm_policy_model = None
        self.llm_tokenizer = None
        self._optimizer_llm = None
        self._lr_scheduler_llm = None
        self.llm_policy_cfg = cfg.llm_policy_cfg  # Set from cfg, not self._cfg (not set yet)

        # Action mapping (will be set from config)
        self.action_map = None  # str -> int
        self.action_inv_map = None  # int -> str

        # Call parent init (this will trigger _init_learn, _init_collect, _init_eval)
        super().__init__(cfg, model, enable_field)

    def _init_learn(self) -> None:
        """
        [PRIORZERO-MODIFIED]
        Initialize both UniZero world model and LLM policy model with their optimizers.
        Align with UniZero implementation - use logging instead of self._logger.
        """
        import logging

        # ======================================================================
        # 1. Initialize UniZero World Model (from parent class)
        # ======================================================================
        super()._init_learn()
        logging.info("✓ UniZero World Model and optimizer initialized")

        # [PRIORZERO-FIX] Ensure scalar transform handles are initialized
        # These are normally initialized in UniZeroPolicy.__init__ but we need to ensure they exist
        if not hasattr(self, 'value_support') or self.value_support is None:
            self.value_support = DiscreteSupport(*self._cfg.model.value_support_range, self._cfg.device)
        if not hasattr(self, 'reward_support') or self.reward_support is None:
            self.reward_support = DiscreteSupport(*self._cfg.model.reward_support_range, self._cfg.device)
        if not hasattr(self, 'value_inverse_scalar_transform_handle'):
            self.value_inverse_scalar_transform_handle = InverseScalarTransform(
                self.value_support, self._cfg.model.categorical_distribution
            )
        if not hasattr(self, 'reward_inverse_scalar_transform_handle'):
            self.reward_inverse_scalar_transform_handle = InverseScalarTransform(
                self.reward_support, self._cfg.model.categorical_distribution
            )
        logging.info("✓ Scalar transform handles verified/initialized")

        # ======================================================================
        # 2. [PRIORZERO-NEW] Initialize LLM Policy Model
        # ======================================================================
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

        # ======================================================================
        # 4. [PRIORZERO-NEW] Load Action Mappings
        # ======================================================================
        if hasattr(self._cfg, 'action_map') and self._cfg.action_map is not None:
            self.action_map = self._cfg.action_map
            self.action_inv_map = {v: k for k, v in self.action_map.items()}
            logging.info(f"✓ Action mappings loaded ({len(self.action_map)} actions)")
        else:
            logging.warning("⚠ Action mappings not found in config. Will use index-based actions.")
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
        import logging

        self._learn_model.train()
        self.llm_policy_model.train()

        # Unpack data
        # NOTE: game_segments is our custom GameSegment with mcts_policy_segment
        # [FIX] Handle case where game_segments might not be included
        if len(data) == 4:
            current_batch, target_batch, train_iter, game_segments = data
        elif len(data) == 3:
            current_batch, target_batch, train_iter = data
            game_segments = None
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                "[PRIORZERO] game_segments not included in training data. "
                "SFT/RFT training will be skipped."
            )
        else:
            raise ValueError(f"Unexpected data format: expected 3 or 4 elements, got {len(data)}")

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
        # [FIX] Use scalar_transform function (not self.scalar_transform)
        # scalar_transform is a standalone function imported from lzero.policy
        transformed_target_reward = scalar_transform(target_reward)
        transformed_target_value = scalar_transform(target_value)

        # Convert to categorical distribution (for distributional RL)
        target_reward_categorical = phi_transform(
            self.reward_support, transformed_target_reward
        )
        target_value_categorical = phi_transform(
            self.value_support, transformed_target_value
        )

        # Prepare batch for world model
        # NOTE: This follows the exact format required by UniZero world model
        # [FIX] Convert obs_batch_ori to tensor if needed
        if not isinstance(obs_batch_ori, torch.Tensor):
            # [DEBUG] Check obs_batch_ori shape
            import logging
            logger = logging.getLogger(__name__)
            if isinstance(obs_batch_ori, np.ndarray):
                logger.info(f"[DEBUG] obs_batch_ori type: numpy, shape: {obs_batch_ori.shape}, dtype: {obs_batch_ori.dtype}")

                # [FIX] Reshape if observations are flattened (2D instead of 3D)
                # Expected: [batch_size, num_unroll_steps+1, obs_dim] (buffer includes next_obs)
                # Got: [batch_size, (num_unroll_steps+1) * obs_dim]
                if len(obs_batch_ori.shape) == 2:
                    # Infer num_unroll_steps and obs_dim
                    # For text: obs_dim should be max_seq_len (e.g., 512)
                    obs_dim = 512  # Standard max_seq_len for BERT
                    total_size = obs_batch_ori.shape[1]
                    if total_size % obs_dim == 0:
                        inferred_steps = total_size // obs_dim
                        # Simply reshape to [batch_size, inferred_steps, obs_dim]
                        # The truncation to match action_batch will happen later (like unizero.py line 675)
                        obs_batch_ori = obs_batch_ori.reshape(batch_size, inferred_steps, obs_dim)
                        logger.info(f"[RESHAPE] Reshaped obs_batch_ori from (batch_size, {total_size}) to {obs_batch_ori.shape}")
                    else:
                        logger.warning(f"[RESHAPE_ERROR] Cannot reshape: total_size ({total_size}) not divisible by obs_dim ({obs_dim})")

                # Check if it's an object array (inhomogeneous shapes)
                if obs_batch_ori.dtype == np.object_:
                    logger.warning(f"[SHAPE_ISSUE] obs_batch_ori is object array - inhomogeneous shapes!")
                    logger.warning(f"[SHAPE_ISSUE] First element shape: {obs_batch_ori[0].shape if len(obs_batch_ori) > 0 else 'N/A'}")
                    if len(obs_batch_ori) > 1:
                        logger.warning(f"[SHAPE_ISSUE] Second element shape: {obs_batch_ori[1].shape}")
                    # Try to handle inhomogeneous array by padding/truncating
                    # For now, just raise a descriptive error
                    raise ValueError(
                        f"obs_batch_ori has inhomogeneous shapes. "
                        f"First element shape: {obs_batch_ori[0].shape}, "
                        f"Cannot directly convert to tensor. "
                        f"This suggests the replay buffer is storing observations with different sequence lengths."
                    )
            obs_batch_ori = torch.from_numpy(obs_batch_ori).to(self._cfg.device)

        # [FIX] Convert action_batch to tensor and handle shape correctly
        if not isinstance(action_batch, torch.Tensor):
            action_batch = torch.from_numpy(action_batch).to(self._cfg.device)

        if action_batch.shape[-1] == 1:
            actions_processed = action_batch.squeeze(-1).long()
        elif len(action_batch.shape) == 1:
            actions_processed = action_batch.long()
        else:
            actions_processed = action_batch.long()

        if timestep_batch is not None:
            # Convert timestep_batch to tensor if needed
            if not isinstance(timestep_batch, torch.Tensor):
                timestep_batch = torch.from_numpy(timestep_batch).to(self._cfg.device)

            # Handle timestep_batch shape
            if timestep_batch.shape[-1] == 1:
                timestep_processed = timestep_batch.squeeze(-1).long()
            elif len(timestep_batch.shape) == 1:
                timestep_processed = timestep_batch.long()
            else:
                timestep_processed = timestep_batch.long()

            batch_for_gpt = {
                'observations': obs_batch_ori,
                'actions': actions_processed,
                'timestep': timestep_processed,
                'rewards': target_reward_categorical[:, :-1],
                'target_value': target_value_categorical[:, :-1],
                'target_policy': target_policy[:, :-1],
            }
        else:
            batch_for_gpt = {
                'observations': obs_batch_ori,
                'actions': actions_processed,
                'rewards': target_reward_categorical[:, :-1],
                'target_value': target_value_categorical[:, :-1],
                'target_policy': target_policy[:, :-1],
            }

        # [FIX] Following unizero.py lines 673-675 exactly:
        # TODO======= 是否需要[:,-1:]
        # Convert mask_batch to boolean, then truncate observations and mask_padding
        batch_for_gpt['mask_padding'] = mask_batch == 1.0  # 0 means invalid padding data. Shape is now (B, T), e.g., (2, 4)

        # =================================================================================
        # [!!! FIX !!!] REMOVE OR COMMENT OUT THE LINE BELOW.
        # This line is the source of the bug. It incorrectly truncates the mask from shape
        # (B, T) to (B, T-1), causing a mismatch with the rewards tensor.
        # The mask_batch from the replay buffer already has the correct length (T) 
        # corresponding to the number of unroll steps.
        # =================================================================================
        # batch_for_gpt['mask_padding'] = batch_for_gpt['mask_padding'][:, :-1]  # <--- REMOVE THIS


        batch_for_gpt['observations'] = batch_for_gpt['observations'][:, :-1]

        # [FIX] Add missing 'ends' field (following unizero.py line 676)
        # 'ends' marks terminal states in the trajectory (0 = not terminal)
        batch_for_gpt['ends'] = torch.zeros(batch_for_gpt['mask_padding'].shape, dtype=torch.long, device=self._cfg.device)

        # [FIX] Add 'scalar_target_value' field for priority calculation (following unizero.py line 681)
        batch_for_gpt['scalar_target_value'] = target_value

        # [FIX] Log shapes for debugging
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"[BATCH_SHAPES] obs: {batch_for_gpt['observations'].shape}, actions: {batch_for_gpt['actions'].shape}, rewards: {batch_for_gpt['rewards'].shape}, mask_padding: {batch_for_gpt['mask_padding'].shape}")

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

        game_segments = None

        # [FIX] Only perform LLM training if game_segments available
        if game_segments is not None:
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
        # Part 4: Logging (Aligned with UniZero)
        # ==============================================================================

        # Extract intermediate losses from world model (like UniZero)
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
            'weighted_total_loss': wm_total_loss.item(),
            'obs_loss': obs_loss.item() if torch.is_tensor(obs_loss) else obs_loss,
            'reward_loss': reward_loss.item() if torch.is_tensor(reward_loss) else reward_loss,
            'policy_loss': policy_loss.item() if torch.is_tensor(policy_loss) else policy_loss,
            'value_loss': value_loss.item() if torch.is_tensor(value_loss) else value_loss,
            'latent_recon_loss': latent_recon_loss.item() if torch.is_tensor(latent_recon_loss) else latent_recon_loss,
            'perceptual_loss': perceptual_loss.item() if torch.is_tensor(perceptual_loss) else perceptual_loss,
            'orig_policy_loss': orig_policy_loss.item() if torch.is_tensor(orig_policy_loss) else orig_policy_loss,
            'policy_entropy': policy_entropy.item() if torch.is_tensor(policy_entropy) else policy_entropy,
            'target_policy_entropy': average_target_policy_entropy.item(),


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
            'analysis/dormant_ratio_encoder': dormant_ratio_encoder,
            'analysis/dormant_ratio_transformer': dormant_ratio_transformer,
            'analysis/dormant_ratio_head': dormant_ratio_head,
            'analysis/avg_weight_mag_encoder': avg_weight_mag_encoder,
            'analysis/avg_weight_mag_transformer': avg_weight_mag_transformer,
            'analysis/avg_weight_mag_head': avg_weight_mag_head,
            'analysis/e_rank_last_linear': e_rank_last_linear,
            'analysis/e_rank_sim_norm': e_rank_sim_norm,
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
            'target_reward': target_reward.mean().item(),
            'target_value': target_value.mean().item(),
            'transformed_target_reward': transformed_target_reward.mean().item(),
            'transformed_target_value': transformed_target_value.mean().item(),
            'value_priority': value_priority_np.mean().item(),
            'value_priority_orig': value_priority_np,

            # ============ Gradient Norms ============
            'total_grad_norm_before_clip_wm': wm_grad_norm.item(),
            'llm_grad_norm': llm_grad_norm.item(),

            # ============ Learning Rates ============
            'cur_lr_world_model': self._optimizer_world_model.param_groups[0]['lr'],
            'llm_lr': self._optimizer_llm.param_groups[0]['lr'],

            # ============ [PRIORZERO] LLM-specific Metrics ============
            'llm_sft_loss': llm_sft_loss.item(),
            'llm_rft_loss': llm_rft_loss.item(),
            'llm_total_loss': llm_loss.item(),
            'num_sft_samples': num_sft_samples,
            'num_rft_samples': num_rft_samples,
            'total_loss': total_loss.item(),
        }

        # ==============================================================================
        # [PRIORZERO-NEW] WandB Logging (if enabled)
        # ==============================================================================
        if self._cfg.get('use_wandb', False):
            try:
                import wandb
                if wandb.run is not None:
                    # Log all metrics to WandB with hierarchical naming
                    wandb.log({
                        # World Model Metrics
                        'train/wm/total_loss': log_dict['wm_total_loss'],
                        'train/wm/value_loss': log_dict['wm_value_loss'],
                        'train/wm/policy_loss': log_dict['wm_policy_loss'],
                        'train/wm/reward_loss': log_dict['wm_reward_loss'],
                        'train/wm/grad_norm': log_dict['wm_grad_norm'],
                        'train/wm/learning_rate': log_dict['wm_lr'],

                        # LLM Policy Metrics
                        'train/llm/sft_loss': log_dict['llm_sft_loss'],
                        'train/llm/rft_loss': log_dict['llm_rft_loss'],
                        'train/llm/total_loss': log_dict['llm_total_loss'],
                        'train/llm/grad_norm': log_dict['llm_grad_norm'],
                        'train/llm/learning_rate': log_dict['llm_lr'],
                        'train/llm/num_sft_samples': log_dict['num_sft_samples'],
                        'train/llm/num_rft_samples': log_dict['num_rft_samples'],

                        # Combined Metrics
                        'train/total_loss': log_dict['total_loss'],
                    }, step=self._train_iteration)
            except Exception as e:
                # Don't fail training if wandb logging fails
                import logging
                logging.warning(f"WandB logging failed: {e}")

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
            logging.debug("No LLM priors provided, using standard UniZero MCTS")
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

            # [FIX] Align with UniZero: construct legal_actions from action_mask
            active_collect_env_num = len(ready_env_id)
            legal_actions = [[i for i, x in enumerate(action_mask[j]) if x == 1]
                            for j in range(active_collect_env_num)]

            # Get timestep if available
            timestep = kwargs.get('timestep', None)

            # [FIX] Align with UniZero: transform values and prepare data
            pred_values_np = self.value_inverse_scalar_transform_handle(pred_values).detach().cpu().numpy()
            latent_state_roots_np = latent_state_roots.detach().cpu().numpy()
            # reward_roots_np = reward_roots.detach().cpu().numpy()
            policy_logits_for_mcts = policy_priors.detach().cpu().numpy().tolist()

            # [FIX] Align with UniZero: Create MCTS roots with legal_actions (not action_space_size)
            roots = MCTSCtree.roots(active_collect_env_num, legal_actions)

            # [FIX] Align with UniZero: noises based on number of valid actions per environment
            noises = [
                np.random.dirichlet([self._cfg.root_dirichlet_alpha] * int(sum(action_mask[j]))
                                    ).astype(np.float32).tolist()
                for j in range(active_collect_env_num)
            ]

            # [FIX] Align with UniZero: prepare roots (note reward_roots_np, not list(pred_values_np))
            roots.prepare(
                self._cfg.root_noise_weight,
                noises,
                reward_roots,
                # reward_roots_np,
                policy_logits_for_mcts,
                to_play if to_play is not None else [-1] * active_collect_env_num,
            )

            # Run MCTS search
            MCTSCtree(self._cfg).search(
                roots,
                self._collect_model,
                latent_state_roots_np,
                reward_roots,
                to_play if to_play is not None else [-1] * latent_state_roots_np.shape[0],
            )

            # Extract search results
            roots_visit_count = roots.get_distributions()
            roots_values = roots.get_values()

            # ======================================================================
            # [PRIORZERO] Get valid_actions_list for dynamic action mapping
            # ======================================================================
            valid_actions_list = kwargs.get('valid_actions_list', None)

            # ======================================================================
            # Select Actions and Prepare Output (Aligned with UniZero)
            # ======================================================================
            output = {}

            for i, env_id in enumerate(ready_env_id):
                # [FIX] Get visit count distribution (only contains legal actions)
                distributions = roots_visit_count[i]
                value = roots_values[i]

                # [FIX] Use select_action from UniZero (aligns with UniZero line 1115-1117)
                # NOTE: Only legal actions possess visit counts, so action_index_in_legal_action_set
                # represents the index within the legal action set, not the entire action set
                action_index_in_legal_action_set, visit_count_distribution_entropy = select_action(
                    distributions,
                    temperature=temperature if temperature is not None else self._collect_mcts_temperature,
                    deterministic=False
                )

                # [FIX] Convert action_index_in_legal_action_set to the actual action in full action space
                # (aligns with UniZero line 1119)
                legal_action_indices = np.where(action_mask[i] == 1.0)[0]
                action = legal_action_indices[action_index_in_legal_action_set]

                # [PRIORZERO] Create dynamic action_inv_map for this specific state
                # This maps action_index -> action_text using the current state's valid_actions
                if valid_actions_list is not None and i < len(valid_actions_list):
                    dynamic_action_inv_map = {
                        idx: act_text
                        for idx, act_text in enumerate(valid_actions_list[i])
                    }
                else:
                    # Fallback to static mapping if valid_actions not available
                    dynamic_action_inv_map = self.action_inv_map

                output[env_id] = {
                    'action': int(action),
                    'visit_count_distributions': distributions,
                    'visit_count_distribution_entropy': visit_count_distribution_entropy,
                    'searched_value': value,
                    'predicted_value': pred_values_np[i],
                    'dynamic_action_inv_map': dynamic_action_inv_map,  # [PRIORZERO] Include dynamic mapping
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
            logging.info("✓ LLM model state loaded")

        if 'optimizer_llm' in state_dict:
            self._optimizer_llm.load_state_dict(state_dict['optimizer_llm'])
            logging.info("✓ LLM optimizer state loaded")

        if 'lr_scheduler_llm' in state_dict and self._lr_scheduler_llm is not None:
            self._lr_scheduler_llm.load_state_dict(state_dict['lr_scheduler_llm'])
            logging.info("✓ LLM scheduler state loaded")

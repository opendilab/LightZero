import asyncio
import copy
import inspect
import re
import sys
import time
import cProfile
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import List, Dict, Any, Tuple, Union, Optional

import numpy as np
import torch
import torch.distributed as dist
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
from ding.utils import build_logger

from priorzero_utils import compute_approx_kl

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
            "Example:\n<think>your step-by-step reasoning here</think>\n<action>the best action text here</action>\n\n"
        )
    else:
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
        # because super().__init__ will call _init_learn which needs these attributes       
        self.llm_tokenizer = None
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
        self.vllm_engine = None

        super().__init__(cfg, model, enable_field)

    def _init_learn(self) -> None:
        """
        [PRIORZERO-MODIFIED]
        Initialize both UniZero world model and LLM policy model with their optimizers.
        Align with UniZero implementation - use logging instead of self._logger.
        """
        super()._init_learn()
        logging.info("✓ UniZero World Model and optimizer initialized")

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
            wm_losses, pred_values = self._learn_model.world_model.compute_loss(
                batch_for_gpt,
                self._target_model.world_model.tokenizer,
                self.value_inverse_scalar_transform_handle,
            )

            wm_total_loss = (weights * wm_losses.loss_total).mean()
        
        self._optimizer_world_model.zero_grad()
        wm_total_loss.backward()
        wm_grad_norm = torch.nn.utils.clip_grad_norm_(
            self._learn_model.world_model.parameters(),
            self._cfg.grad_clip_value
        )
        if self._cfg.multi_gpu:
            self.sync_gradients(self._learn_model)
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
            'rft_logprob_mean',
            'rft_seq_neglogprob_mean',
            'rft_advantage_mean',
            'rft_advantage_std',
            'rft_ratio_used_mean',
            'rft_kl_mean',
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

        return state_dict

    def _load_state_dict_learn(self, state_dict: Dict[str, Any]) -> None:
        """
        [PRIORZERO-MODIFIED]
        Load state dict for both world model and LLM.
        """
        super()._load_state_dict_learn(state_dict)
    

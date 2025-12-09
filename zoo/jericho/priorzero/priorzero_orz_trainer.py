import torch
import torch.nn as nn
import torch.nn.functional as F
import ray
from typing import Dict, List, Any, Optional

from orz.ppo.utils import get_strategy
from orz.ppo.actors import Actor


# ==============================================================================
# Helper: Strategy Configuration Adapter
# ==============================================================================
class StrategyArgs:
    """
    将 dict 配置转换为对象，供 get_strategy 读取。
    DeepSpeed 策略通常需要访问 args.local_rank, args.zero_stage 等属性。
    """
    def __init__(self, cfg: Dict):
        self.seed = cfg.get('seed', 42)
        self.local_rank = 0  # Ray Actor 内部为 0
        self.gradient_checkpointing = cfg.get('gradient_checkpointing', True)
        self.max_norm = cfg.get('grad_clip_value', 1.0)
        # Batch size settings
        self.micro_train_batch_size = cfg.get('llm_micro_batch_size', 1)
        self.train_batch_size = cfg.get('llm_micro_batch_size', 1)
        # DeepSpeed settings
        self.zero_stage = cfg.get('deepspeed_zero_stage', 2)
        self.bf16 = True 
        self.fp16 = False
        self.adam_offload = cfg.get('adam_offload', False)
        self.zpg = 1
        # LoRA settings
        self.lora_rank = cfg.get('lora_r', 0)
        self.lora_alpha = cfg.get('lora_alpha', 16)
        self.lora_dropout = cfg.get('lora_dropout', 0)
        self.target_modules = cfg.get('target_modules', ["q_proj", "v_proj", "k_proj", "o_proj"])
        # Misc
        self.flash_attn = True
        self.save_path = None
        self.save_steps = -1
        self.ckpt_path = None
        self.use_wandb = False

# ==============================================================================
# [MAIN ACTOR] OrzPPOTrainerActor
# ==============================================================================
@ray.remote(num_gpus=1)
class OrzPPOTrainerActor:
    """
    Remote Trainer for PriorZero.
    Includes explicit PPO Loss calculation (No Critic).
    """
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.device = torch.device("cuda:0") # Ray Worker 内部视角
        self.clip_eps = cfg.get('rft_clip_epsilon', 0.2)
        
        args = StrategyArgs(cfg)
        self.strategy = get_strategy(args)
        
        self.actor = Actor(
            cfg['pretrain_llm_path'],
            use_flash_attention_2=args.flash_attn,
            bf16=args.bf16,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.target_modules,
        )
        print(f'actor={self.actor}')
        self.actor_optim = self.strategy.create_optimizer(
            self.actor, 
            lr=cfg['llm_learning_rate'], 
            betas=(0.9, 0.95),
            weight_decay=cfg['llm_weight_decay']
        )
        print(f'self.actor_optim={self.actor_optim}')
        self.actor, self.actor_optim = self.strategy.prepare(
            self.actor, self.actor_optim, is_rlhf=True
        )

    def compute_actor_loss(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Manually implemented PPO Policy Loss.
        Formula: -min( ratio*A, clamp(ratio, 1-eps, 1+eps)*A )
        """
        # 1. Calculate Ratio: pi_new / pi_old = exp(log_new - log_old)
        # Detach old_log_probs to be safe
        ratio = torch.exp(log_probs - old_log_probs.detach())
        
        # 2. Calculate Surrogate Objectives
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
        
        # 3. Aggregate Loss
        loss = -torch.min(surr1, surr2)
        
        # 4. Apply Mask (Only calculate loss for Action tokens, ignore Prompt/Padding)
        if mask is not None:
            loss = (loss * mask).sum() / (mask.sum() + 1e-8)
        else:
            loss = loss.mean()
            
        # 5. Optional: Calculate Approx KL for monitoring
        # KL approx (k2 estimator): 0.5 * (logp_old - logp_new)^2
        with torch.no_grad():
            approx_kl = 0.5 * (old_log_probs - log_probs).pow(2)
            if mask is not None:
                approx_kl = (approx_kl * mask).sum() / (mask.sum() + 1e-8)
            else:
                approx_kl = approx_kl.mean()

        return {"loss": loss, "kl": approx_kl}

    def update_weights(self, state_dict_ref):
        """Sync: Main Process -> Actor"""
        state_dict = state_dict_ref # Ray resolves ObjectRef automatically
        unwrap_model = self.strategy.unwrap_model(self.actor)
        unwrap_model.load_state_dict(state_dict, strict=False)

    def get_weights(self):
        """Sync: Actor -> Main Process"""
        unwrap_model = self.strategy.unwrap_model(self.actor)
        return {k: v.cpu() for k, v in unwrap_model.state_dict().items()}

    def train_step(self, batch_data: Dict[str, Any]):
        """
        Execute one PPO step.
        """
        # --- 1. Unpack Data ---
        input_ids = torch.tensor(batch_data['input_ids'], device=self.device, dtype=torch.long)
        attention_mask = torch.tensor(batch_data['attention_mask'], device=self.device, dtype=torch.long)
        old_logprobs = torch.tensor(batch_data['old_logprobs'], device=self.device, dtype=torch.float32)
        advantages = torch.tensor(batch_data['advantages'], device=self.device, dtype=torch.float32)
        
        # Normalize Advantages
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # --- 2. Determine Masks ---
        # num_actions: 用于区分 Answer 和 Prompt
        num_actions = batch_data.get('num_actions')
        if num_actions is None:
            num_actions = input_ids.shape[1] # 默认全长

        # Construct Action Mask (1 for action tokens, 0 for prompt/padding)
        action_mask = torch.zeros_like(input_ids, dtype=torch.float)
        
        # Vectorized masking if num_actions varies
        if isinstance(num_actions, (list, tuple, torch.Tensor)):
            for i, n in enumerate(num_actions):
                action_mask[i, -int(n):] = 1.0
        else:
            # Fixed length
            action_mask[:, -int(num_actions):] = 1.0
        final_mask = action_mask * attention_mask
        curr_log_probs = self.actor(input_ids, num_actions, attention_mask)
        stats = self.compute_actor_loss(
            log_probs=curr_log_probs,
            old_log_probs=old_logprobs,
            advantages=advantages,
            mask=final_mask
        )
        loss = stats["loss"]
        self.strategy.backward(loss, self.actor, self.actor_optim)
        self.strategy.step(self.actor, self.actor_optim)
        return {
            'rft_loss': loss.item(),
            'rft_kl': stats["kl"].item()
        }
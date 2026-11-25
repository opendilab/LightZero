from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from utils import masked_mean

class PolicyLoss(nn.Module):
    """
    Policy Loss for PPO
    """

    def __init__(
        self,
        clip_eps_low: float = 0.2,
        clip_eps_high: float = 0.2,
        dual_clip: float = None,
        token_level_loss: bool = True,
        policy_loss_type: str = "ppo",
        enable_vllm_is_correction: bool = False,
        vllm_is_truncated_threshold: list = None,
        use_icepop: bool = False,
        use_cot: bool = False,
        use_mispo: bool = False,
        cot_weight: Optional[float] = None,
        mispo_token_truncated_threshold = None,
        mispo_traj_truncated_threshold = None,
        
    ) -> None:
        super().__init__()
        self.clip_eps_low = clip_eps_low
        self.clip_eps_high = clip_eps_high
        self.token_level_loss = token_level_loss
        self.dual_clip = dual_clip
        self.policy_loss_type = policy_loss_type
        self.enable_vllm_is_correction = enable_vllm_is_correction
        self.vllm_is_truncated_threshold = vllm_is_truncated_threshold
        self.use_icepop = use_icepop
        
        self.use_cot = use_cot
        self.cot_weight = cot_weight
        self.use_mispo = use_mispo
        self.mispo_token_truncated_threshold = mispo_token_truncated_threshold
        self.mispo_traj_truncated_threshold = mispo_traj_truncated_threshold

        # GSPO requires sequence-level loss
        if policy_loss_type == "gspo":
            self.token_level_loss = False

        # Dual-clip PPO: https://arxiv.org/pdf/1912.09729
        if dual_clip is not None:
            assert dual_clip > 1.0, f"dual_clip must be > 1.0, got {dual_clip}"

    def forward(
        self,
        input_ids: torch.LongTensor,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        rollout_log_probs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.policy_loss_type == "ppo":
            log_ratio = log_probs - old_log_probs
            ratio = log_ratio.exp()
        elif self.policy_loss_type == "gspo":
            # GSPO: https://arxiv.org/pdf/2507.18071
            if self.enable_vllm_is_correction:
                log_ratio = log_probs - rollout_log_probs
            else:
                log_ratio = log_probs - old_log_probs
            ratio = (log_ratio * action_mask).sum(dim=-1) / action_mask.sum(dim=-1)
            ratio = ratio.exp().unsqueeze(-1) * action_mask
        else:
            raise ValueError(f"Invalid policy loss type: {self.policy_loss_type}")
        if advantages.dim() == 1:
            advantages = advantages.unsqueeze(-1)
        
        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - self.clip_eps_low, 1 + self.clip_eps_high) * advantages

        if self.dual_clip is None:
            # Standard PPO
            loss = -torch.min(surr1, surr2)
        else:
            # Standard PPO clipping
            clip1 = torch.min(surr1, surr2)
            # Dual-clip: additional lower bound for negative advantages
            clip2 = torch.max(clip1, self.dual_clip * advantages)
            # Apply dual-clip: use clip2 for negative advantages, clip1 for positive advantages
            loss = -torch.where(advantages < 0, clip2, clip1)

        # Your Efficient RL Framework Secretly Brings You Off-Policy RL Training: https://fengyao.notion.site/off-policy-rl
        vllm_kl = None
        token_mask = None
        traj_mask = None
        effective_mask = action_mask
        if self.enable_vllm_is_correction and self.policy_loss_type == "ppo":
            low_threshold, high_threshold = self.vllm_is_truncated_threshold
            if self.use_mispo:
                token_low, token_high = self.mispo_token_truncated_threshold
                traj_low, traj_high = self.mispo_traj_truncated_threshold
                token_ratio = torch.exp(old_log_probs - rollout_log_probs).detach()
                token_mask = ((token_ratio >= token_low) & (token_ratio <= token_high)).float()
                traj_log_ratio = masked_mean(
                    old_log_probs - rollout_log_probs,
                    action_mask,
                    dim=-1,
                )
                traj_ratio = torch.exp(traj_log_ratio).detach()
                traj_mask = ((traj_ratio >= traj_low) & (traj_ratio <= traj_high)).float().unsqueeze(-1)
                mispo_mask = token_mask * traj_mask * action_mask
                loss = loss * mispo_mask
                effective_mask = mispo_mask
                if effective_mask.sum().item() == 0:
                    effective_mask = action_mask
                    
            elif self.use_icepop:
                # ICEPOP: set coefficients outside the interval to 0
                vllm_is = torch.exp(old_log_probs - rollout_log_probs).detach()
                mask = (vllm_is >= low_threshold) & (vllm_is <= high_threshold)
                vllm_is = vllm_is * mask
                loss = vllm_is * loss
            else:
                # Standard clamp with low and high thresholds
                vllm_is = (
                    torch.exp(old_log_probs - rollout_log_probs).clamp(min=low_threshold, max=high_threshold).detach()
                )
                loss = vllm_is * loss
            vllm_kl = masked_mean(rollout_log_probs - old_log_probs, effective_mask, dim=None)
        
        ###### 对 cot 前缀加权重
        if self.use_cot and self.cot_weight is not None:
            output_ids = input_ids[:, -action_mask.shape[1]:]
            is_split = (output_ids == 2512) & action_mask.bool()  
            token_weights = torch.ones_like(loss) 
            pos = torch.arange(action_mask.shape[1], device=input_ids.device).unsqueeze(0)
            last_split_pos = torch.where(is_split, pos, torch.full_like(pos, -1)).max(dim=1, keepdim=True).values
            token_weights = torch.where(
                (pos < last_split_pos) & action_mask.bool(),                   # 若想包含 2512 本身就改成 <=
                torch.full_like(token_weights, self.cot_weight),
                token_weights,
            )
            loss = loss * token_weights

        loss = (
            masked_mean(loss, effective_mask, dim=None)
            if self.token_level_loss
            else masked_mean(loss, effective_mask, dim=-1).mean()
        )

        clipped = ratio.gt(1 + self.clip_eps_high) | ratio.lt(1 - self.clip_eps_low)
        clipfrac = masked_mean(clipped, effective_mask, dim=None)

        clip_ratio = masked_mean(torch.lt(surr2, surr1).float(), effective_mask, dim=None)
        approx_kl = masked_mean(-log_ratio.detach(), effective_mask, dim=None)
        return loss, clipfrac, clip_ratio, approx_kl, vllm_kl, token_mask, traj_mask
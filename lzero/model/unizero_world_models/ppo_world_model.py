import torch
import torch.nn.functional as F

from lzero.policy.ppo_utils import ppo_policy_loss

from .tokenizer import Tokenizer
from .utils import LossWithIntermediateLosses
from .world_model import WorldModel


class PPOWorldModel(WorldModel):
    """PPO-specific actor/critic extension of the legacy UniZero world model."""

    def compute_ppo_loss(self, batch, inverse_scalar_transform_handle) -> LossWithIntermediateLosses:
        """Compute PPO directly on cached contextual latents.

        The encoder, Transformer, target tokenizer and reconstruction heads are
        deliberately bypassed.  Besides preserving the exact behavior context,
        this keeps Atari PPO minibatches proportional to two small heads instead
        of a complete world-model training pass.
        """
        features = batch['ppo_policy_features'].detach()
        valid_mask = batch['mask_padding'].bool()
        actions = batch['actions']
        batch_size, num_steps = actions.shape

        policy_logits = self.head_policy.head_module(features)
        if self.use_policy_logits_clip:
            policy_logits = self._apply_policy_logits_control(policy_logits)
        value_logits = self.head_value.head_module(features)
        policy_surrogate, policy_entropy_steps, ppo_metrics = ppo_policy_loss(
            policy_logits,
            batch['ppo_action_mask'],
            actions,
            batch['ppo_old_log_prob'],
            batch['ppo_advantages'],
            valid_mask,
            float(batch['ppo_clip_ratio']),
        )
        policy_loss_steps = (
            policy_surrogate - float(batch['ppo_entropy_weight']) * policy_entropy_steps
        )
        value_loss_steps = -(batch['target_value'] * F.log_softmax(value_logits, dim=-1)).sum(dim=-1)

        valid_float = valid_mask.float()
        valid_count = valid_float.sum().clamp_min(1)
        per_sample_count = valid_float.sum(dim=1).clamp_min(1)

        def masked_average(values):
            return (values * valid_float).sum() / valid_count

        loss_policy = masked_average(policy_loss_steps)
        orig_policy_loss = masked_average(policy_surrogate)
        policy_entropy = masked_average(policy_entropy_steps)
        loss_value = masked_average(value_loss_steps)
        zero = loss_policy.new_zeros(())

        per_sample_loss_policy = (policy_loss_steps * valid_float).sum(dim=1) / per_sample_count
        per_sample_orig_policy = (policy_surrogate * valid_float).sum(dim=1) / per_sample_count
        per_sample_entropy = (policy_entropy_steps * valid_float).sum(dim=1) / per_sample_count
        per_sample_value = (value_loss_steps * valid_float).sum(dim=1) / per_sample_count
        per_sample_zero = torch.zeros(batch_size, device=features.device, dtype=features.dtype)

        def step_average(values, mask, index):
            selected = values[:, index][mask[:, index]]
            return selected.mean() if selected.numel() else zero

        zero_obs_steps = torch.zeros(
            batch_size, max(num_steps - 1, 1), device=features.device, dtype=features.dtype
        )
        obs_mask = valid_mask[:, 1:] if num_steps > 1 else valid_mask[:, :1]
        zero_reward_steps = torch.zeros_like(valid_float)
        step_losses = []
        for index in (0, num_steps // 2, num_steps - 1):
            obs_index = min(index, zero_obs_steps.shape[1] - 1)
            step_losses.append({
                'loss_obs': step_average(zero_obs_steps, obs_mask, obs_index),
                'loss_rewards': step_average(zero_reward_steps, valid_mask, index),
                'loss_value': step_average(value_loss_steps, valid_mask, index),
                'loss_policy': step_average(policy_loss_steps, valid_mask, index),
                'orig_policy_loss': step_average(policy_surrogate, valid_mask, index),
                'policy_entropy': step_average(policy_entropy_steps, valid_mask, index),
            })

        with torch.no_grad():
            if self.config.use_priority:
                scalar_values = inverse_scalar_transform_handle(
                    value_logits.reshape(batch_size * num_steps, -1)
                ).reshape(batch_size, num_steps)
                target_values = batch['scalar_target_value'][:, :num_steps]
                value_priority = (
                    (scalar_values - target_values).abs() * valid_float
                ).sum(dim=1) / per_sample_count
            else:
                value_priority = per_sample_zero

        open_loop_components = {
            name: zero for name in ('latent', 'reward', 'value', 'policy', 'policy_ce', 'policy_entropy')
        }
        return LossWithIntermediateLosses(
            latent_recon_loss_weight=self.latent_recon_loss_weight,
            perceptual_loss_weight=self.perceptual_loss_weight,
            open_loop_consistency_loss_weight=self.open_loop_consistency_loss_weight,
            open_loop_recurrent_loss_weight=self.open_loop_recurrent_loss_weight,
            continuous_action_space=False,
            loss_obs=zero,
            loss_rewards=zero,
            loss_value=loss_value,
            loss_policy=loss_policy,
            latent_recon_loss=zero,
            perceptual_loss=zero,
            open_loop_consistency_loss=zero,
            open_loop_recurrent_loss=zero,
            open_loop_recurrent_latent_loss=open_loop_components['latent'],
            open_loop_recurrent_reward_loss=open_loop_components['reward'],
            open_loop_recurrent_value_loss=open_loop_components['value'],
            open_loop_recurrent_policy_loss=open_loop_components['policy'],
            open_loop_recurrent_policy_ce=open_loop_components['policy_ce'],
            open_loop_recurrent_policy_entropy=open_loop_components['policy_entropy'],
            orig_policy_loss=orig_policy_loss,
            policy_entropy=policy_entropy,
            first_step_losses=step_losses[0],
            middle_step_losses=step_losses[1],
            last_step_losses=step_losses[2],
            dormant_ratio_encoder=zero,
            dormant_ratio_transformer=zero,
            dormant_ratio_head=zero,
            avg_weight_mag_encoder=zero,
            avg_weight_mag_transformer=zero,
            avg_weight_mag_head=zero,
            e_rank_last_linear=zero,
            e_rank_sim_norm=zero,
            latent_state_l2_norms=features.norm(p=2, dim=-1)[valid_mask].mean(),
            value_priority=value_priority,
            intermediate_tensor_x=features,
            obs_embeddings=features,
            logits_value=value_logits.detach(),
            logits_policy=policy_logits.detach(),
            per_sample_loss_obs=per_sample_zero,
            per_sample_loss_rewards=per_sample_zero,
            per_sample_loss_value=per_sample_value,
            per_sample_loss_policy=per_sample_loss_policy,
            per_sample_loss_orig_policy=per_sample_orig_policy,
            per_sample_loss_policy_entropy=per_sample_entropy,
            **ppo_metrics,
        )

    def compute_loss(self, batch, target_tokenizer: Tokenizer = None, inverse_scalar_transform_handle=None, **kwargs):
        if batch.get('actor_critic_only', False):
            return self.compute_ppo_loss(batch, inverse_scalar_transform_handle)

        losses = super().compute_loss(batch, target_tokenizer, inverse_scalar_transform_handle, **kwargs)
        # PPO replay trains latent/reward dynamics only.  Removing these terms
        # from the final scalar also removes their gradients from shared latents.
        if batch.get('disable_policy_loss', False):
            losses.loss_total = losses.loss_total - losses.policy_loss_weight * losses.intermediate_losses['loss_policy']
        if batch.get('disable_value_loss', False):
            losses.loss_total = losses.loss_total - losses.value_loss_weight * losses.intermediate_losses['loss_value']
        return losses

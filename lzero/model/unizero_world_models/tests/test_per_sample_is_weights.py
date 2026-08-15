"""
Unit test for per-sample importance-sampling (IS) weighting in UniZero.

Background: ``LossWithIntermediateLosses.loss_total`` is a batch-level scalar, so
``(weights * loss_total).mean()`` degenerates to ``loss_total * mean(weights)`` and the per-sample
IS weights from the prioritized replay buffer have no effect. ``WorldModel.compute_loss`` therefore
additionally returns per-sample loss components (``per_sample_loss_*``, [B] tensors) for discrete
action spaces, and ``lzero.policy.unizero.apply_per_sample_is_weights`` rebuilds the total loss per
sample so that the IS weights are actually applied.
"""
from types import SimpleNamespace

import pytest
import torch
from easydict import EasyDict

from lzero.model.common import RepresentationNetworkUniZero
from lzero.model.unizero_world_models.tokenizer import Tokenizer
from lzero.model.unizero_world_models.world_model import WorldModel
from lzero.policy import DiscreteSupport, InverseScalarTransform
from lzero.policy.unizero import apply_per_sample_is_weights

B, T, A = 8, 6, 6
EMBED_DIM, GROUP_SIZE = 64, 8
SUPPORT_SIZE = 101
PS_KEYS = ['per_sample_loss_obs', 'per_sample_loss_rewards', 'per_sample_loss_value',
           'per_sample_loss_policy', 'per_sample_loss_orig_policy', 'per_sample_loss_policy_entropy']


def _build_world_model() -> WorldModel:
    cfg = EasyDict(dict(
        tokens_per_block=2, max_blocks=T + 2, max_tokens=2 * (T + 2), attention='causal',
        num_layers=1, num_heads=2, embed_dim=EMBED_DIM,
        embed_pdrop=0., resid_pdrop=0., attn_pdrop=0.,
        rotary_emb=False,
        lora_r=0, lora_alpha=1, lora_dropout=0.0, lora_target_modules=None,
        curriculum_stage_num=1, min_stage0_iters=10_000, max_stage_iters=20_000, lora_scale_init=1.0,
        task_embed_option='none', register_token_num=4, register_token_shared=True,
        gru_gating=False, moe_in_transformer=False, multiplication_moe_in_transformer=False,
        num_experts_of_moe_in_transformer=1,
        policy_entropy_weight=5e-3,
        predict_latent_loss_type='mse',
        group_size=GROUP_SIZE,
        obs_type='image',
        gamma=0.997,
        context_length=2 * T,
        dormant_threshold=0.1,
        analysis_dormant_ratio_weight_rank=False,
        latent_recon_loss_weight=0.0,
        perceptual_loss_weight=0.0,
        support_size=SUPPORT_SIZE,
        action_space_size=A,
        max_cache_size=100,
        env_num=2,
        continuous_action_space=False,
        num_simulations=5,
        game_segment_length=20,
        device='cpu',
        norm_type='LN',
        final_norm_option_in_obs_head='LayerNorm',
        use_priority=True,
    ))
    encoder = RepresentationNetworkUniZero(
        observation_shape=(3, 64, 64), num_res_blocks=1, num_channels=8, downsample=True,
        norm_type='LN', embedding_dim=EMBED_DIM, group_size=GROUP_SIZE,
        final_norm_option_in_encoder='LayerNorm')
    tokenizer = Tokenizer(encoder=encoder, decoder=None, with_lpips=False, obs_type='image')
    return WorldModel(cfg, tokenizer)


def _make_batch(mask_padding: torch.Tensor) -> dict:
    return dict(
        observations=torch.rand(B, T, 3, 64, 64),
        actions=torch.randint(0, A, (B, T)),
        timestep=torch.arange(T).unsqueeze(0).repeat(B, 1),
        rewards=torch.softmax(torch.randn(B, T, SUPPORT_SIZE), dim=-1),
        ends=torch.zeros(B, T, dtype=torch.long),
        mask_padding=mask_padding,
        target_value=torch.softmax(torch.randn(B, T, SUPPORT_SIZE), dim=-1),
        target_policy=torch.softmax(torch.randn(B, T, A), dim=-1),
        scalar_target_value=torch.randn(B, T),
    )


@pytest.mark.unittest
class TestPerSampleISWeights:

    def test_per_sample_losses_and_is_weighting(self):
        torch.manual_seed(0)
        wm = _build_world_model()
        handle = InverseScalarTransform(DiscreteSupport(-50, 51, 1), True)
        losses = wm.compute_loss(_make_batch(torch.ones(B, T, dtype=torch.bool)), wm.tokenizer, handle, global_step=0)
        il = losses.intermediate_losses

        # per-sample keys exist, have shape [B] and carry gradients
        for k in PS_KEYS:
            assert k in il, f'missing key: {k}'
            assert isinstance(il[k], torch.Tensor) and il[k].shape == (B,), (k, il[k].shape)
            assert il[k].requires_grad, k

        ps_total = (losses.obs_loss_weight * il['per_sample_loss_obs']
                    + losses.reward_loss_weight * il['per_sample_loss_rewards']
                    + losses.value_loss_weight * il['per_sample_loss_value']
                    + losses.policy_loss_weight * il['per_sample_loss_policy'])

        # full mask + uniform weights: per-sample weighting must equal the legacy scalar total
        weighted = apply_per_sample_is_weights(torch.ones(B), losses, il['per_sample_loss_policy'], losses.loss_total)
        assert torch.allclose(weighted, losses.loss_total, atol=1e-5), (weighted.item(), losses.loss_total.item())

        # non-uniform weights: exact match with the manual per-sample formula
        w = torch.linspace(0.5, 1.5, B)
        weighted_nu = apply_per_sample_is_weights(w, losses, il['per_sample_loss_policy'], losses.loss_total)
        assert torch.allclose(weighted_nu, (w * ps_total).mean(), atol=1e-6)

        # one-hot weights select exactly the selected sample's own loss
        w_onehot = torch.zeros(B)
        w_onehot[0] = 1.0
        weighted_oh = apply_per_sample_is_weights(w_onehot, losses, il['per_sample_loss_policy'], losses.loss_total)
        assert torch.allclose(weighted_oh, ps_total[0] / B, atol=1e-6)

        # fallback path: missing per-sample policy loss -> legacy scalar behavior
        w2 = torch.linspace(0.5, 1.5, B)
        weighted_fb = apply_per_sample_is_weights(w2, losses, None, losses.loss_total)
        assert torch.allclose(weighted_fb, (w2 * losses.loss_total).mean())

        # backward flows through the per-sample path
        wm.zero_grad()
        weighted.backward()
        grad_norms = [p.grad.norm().item() for p in wm.parameters() if p.grad is not None]
        assert len(grad_norms) > 0 and all(g == g for g in grad_norms)

    def test_per_sample_is_weights_discriminates_samples(self):
        """Stub container with strongly varied per-sample losses: only the per-sample path can
        weight samples differently from the scalar mean."""
        stub_ps = torch.arange(1, B + 1, dtype=torch.float32)
        stub_losses = SimpleNamespace(
            obs_loss_weight=1.0, reward_loss_weight=0.0, value_loss_weight=0.0, policy_loss_weight=0.0,
            latent_recon_loss_weight=0.0, perceptual_loss_weight=0.0,
            intermediate_losses={'per_sample_loss_obs': stub_ps,
                                 'per_sample_loss_rewards': torch.zeros(B),
                                 'per_sample_loss_value': torch.zeros(B),
                                 'latent_recon_loss': torch.tensor(0.), 'perceptual_loss': torch.tensor(0.)},
        )
        stub_scalar_total = stub_ps.mean()
        w = torch.ones(B)
        w[0], w[B - 1] = 10.0, 0.1
        per_sample_res = apply_per_sample_is_weights(w, stub_losses, torch.zeros(B), stub_scalar_total)
        assert torch.allclose(per_sample_res, (w * stub_ps).mean())
        legacy_res = (w * stub_scalar_total).mean()
        assert abs(per_sample_res.item() - legacy_res.item()) > 0.5

    def test_partial_mask(self):
        torch.manual_seed(0)
        wm = _build_world_model()
        handle = InverseScalarTransform(DiscreteSupport(-50, 51, 1), True)
        mask = torch.ones(B, T, dtype=torch.bool)
        mask[:, -2:] = False  # every sample loses its last 2 steps
        mask[0] = False       # sample 0 fully masked
        losses = wm.compute_loss(_make_batch(mask), wm.tokenizer, handle, global_step=0)
        il = losses.intermediate_losses
        for k in PS_KEYS:
            v = il[k]
            assert v.shape == (B,) and not torch.isnan(v).any(), k
        # fully-masked sample contributes zero loss (numerator 0, denominator clamped)
        assert il['per_sample_loss_policy'][0].item() == 0.0
        weighted = apply_per_sample_is_weights(torch.ones(B), losses, il['per_sample_loss_policy'], losses.loss_total)
        wm.zero_grad()
        weighted.backward()

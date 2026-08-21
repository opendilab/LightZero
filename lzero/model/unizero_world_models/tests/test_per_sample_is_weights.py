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
from lzero.model.unizero_world_models.ppo_world_model import PPOWorldModel
from lzero.policy import DiscreteSupport, InverseScalarTransform
from lzero.policy.unizero import (
    apply_open_loop_recurrent_entropy_weight,
    apply_per_sample_is_weights,
)

B, T, A = 8, 6, 6
EMBED_DIM, GROUP_SIZE = 64, 8
SUPPORT_SIZE = 101
PS_KEYS = ['per_sample_loss_obs', 'per_sample_loss_rewards', 'per_sample_loss_value',
           'per_sample_loss_policy', 'per_sample_loss_orig_policy', 'per_sample_loss_policy_entropy']


def _build_world_model(rotary_emb: bool = False, world_model_cls=WorldModel) -> WorldModel:
    cfg = EasyDict(dict(
        tokens_per_block=2, max_blocks=T + 2, max_tokens=2 * (T + 2), attention='causal',
        num_layers=1, num_heads=2, embed_dim=EMBED_DIM,
        embed_pdrop=0., resid_pdrop=0., attn_pdrop=0.,
        rotary_emb=rotary_emb,
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
    return world_model_cls(cfg, tokenizer)


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

    def test_open_loop_consistency_matches_cached_mcts_rollout_across_window_reset(self):
        torch.manual_seed(23)
        world_model = _build_world_model()
        world_model.rebuild_kv_window_from_tokens = True
        world_model.context_length = 6
        world_model.open_loop_diagnostic_batch_size = 2
        world_model.open_loop_consistency_batch_size = 2
        world_model.open_loop_consistency_horizon = T - 1
        observation_head_linears = [
            module for module in world_model.head_observations.modules()
            if isinstance(module, torch.nn.Linear)
        ]
        # Avoid a trivial all-zero comparison from the production zero initialization.
        torch.nn.init.normal_(observation_head_linears[-1].weight, std=0.02)
        batch = _make_batch(torch.ones(B, T, dtype=torch.bool))
        obs_embeddings = world_model.tokenizer.encode_to_obs_embeddings(batch['observations'])
        target_embeddings = obs_embeddings.detach().clone()

        diagnostics = world_model.compute_open_loop_latent_diagnostics(
            obs_embeddings.detach(), target_embeddings, batch['actions'], batch['mask_padding']
        )
        differentiable_loss = world_model.compute_open_loop_consistency_loss(
            obs_embeddings, target_embeddings, batch['actions'], batch['mask_padding']
        )

        # context_length=6 rebuilds the raw window before the third prediction,
        # so this compares both the pre-boundary and post-boundary rollout semantics.
        assert differentiable_loss.item() == pytest.approx(
            diagnostics['open_loop_latent_mse_mean'], rel=1e-5, abs=1e-6
        )

    def test_open_loop_consistency_loss_has_recurrent_gradients(self):
        torch.manual_seed(17)
        world_model = _build_world_model()
        world_model.rebuild_kv_window_from_tokens = True
        world_model.context_length = 6
        world_model.open_loop_consistency_loss_weight = 1.0
        world_model.open_loop_consistency_batch_size = 2
        world_model.open_loop_consistency_horizon = 3
        observation_head_linears = [
            module for module in world_model.head_observations.modules()
            if isinstance(module, torch.nn.Linear)
        ]
        # Production observation heads are zero-initialized, so their first backward pass
        # intentionally cannot reach the Transformer.  Simulate the post-first-update state to
        # verify that the recurrent auxiliary graph itself propagates upstream.
        torch.nn.init.normal_(observation_head_linears[-1].weight, std=0.02)
        handle = InverseScalarTransform(DiscreteSupport(-50, 51, 1), True)

        losses = world_model.compute_loss(
            _make_batch(torch.ones(B, T, dtype=torch.bool)),
            world_model.tokenizer,
            handle,
            global_step=1,
        )
        consistency_loss = losses.intermediate_losses['open_loop_consistency_loss']

        assert world_model.training is True
        assert consistency_loss.requires_grad
        assert torch.isfinite(consistency_loss)
        assert consistency_loss > 0
        expected_total = (
            losses.obs_loss_weight * losses.intermediate_losses['loss_obs']
            + losses.reward_loss_weight * losses.intermediate_losses['loss_rewards']
            + losses.value_loss_weight * losses.intermediate_losses['loss_value']
            + losses.policy_loss_weight * losses.intermediate_losses['loss_policy']
            + consistency_loss
        )
        assert torch.allclose(losses.loss_total, expected_total)

        transformer_parameter = next(
            parameter for parameter in world_model.transformer.parameters()
            if parameter.requires_grad and parameter.ndim > 1
        )
        observation_head_parameter = observation_head_linears[-1].weight
        gradients = torch.autograd.grad(
            consistency_loss,
            (transformer_parameter, observation_head_parameter),
            allow_unused=False,
        )
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        assert all(gradient.abs().sum() > 0 for gradient in gradients)
        assert all(gradient.abs().sum() > 0 for gradient in gradients)

    def test_open_loop_teacher_prefix_starts_each_prediction_from_steady_context(self):
        torch.manual_seed(19)
        world_model = _build_world_model()
        world_model.rebuild_kv_window_from_tokens = True
        world_model.context_length = 10
        world_model.open_loop_consistency_batch_size = 2
        world_model.open_loop_consistency_horizon = 2
        world_model.open_loop_prefix_transitions = 3
        observation_head_linears = [
            module for module in world_model.head_observations.modules()
            if isinstance(module, torch.nn.Linear)
        ]
        torch.nn.init.normal_(observation_head_linears[-1].weight, std=0.02)
        batch = _make_batch(torch.ones(B, T, dtype=torch.bool))
        obs_embeddings = world_model.tokenizer.encode_to_obs_embeddings(batch['observations'])

        transformer_input_lengths = []

        def record_input_length(_module, args):
            transformer_input_lengths.append(args[0].size(1))

        hook = world_model.transformer.register_forward_pre_hook(record_input_length)
        try:
            loss = world_model.compute_open_loop_consistency_loss(
                obs_embeddings, obs_embeddings.detach(), batch['actions'], batch['mask_padding']
            )
        finally:
            hook.remove()

        # Three real transitions form [o0,a0,o1,a1,o2,a2,o3] (7 retained tokens).
        # Each predicted transition then appends its action as token 8; after the prediction,
        # appending the predicted observation reaches 9 and rolls back to the same 7-token phase.
        assert transformer_input_lengths == [8, 8]
        assert loss.requires_grad and torch.isfinite(loss)

        # Prefix observations are context only. Changing their target-encoder copies must not
        # change the loss, while changing the two post-prefix targets (states 4 and 5) must.
        prefix_changed_targets = obs_embeddings.detach().clone()
        prefix_changed_targets.view(B, T, 1, EMBED_DIM)[:, :4].add_(100.)
        rollout_changed_targets = obs_embeddings.detach().clone()
        rollout_changed_targets.view(B, T, 1, EMBED_DIM)[:, 4:6].add_(100.)
        prefix_changed_loss = world_model.compute_open_loop_consistency_loss(
            obs_embeddings, prefix_changed_targets, batch['actions'], batch['mask_padding']
        )
        rollout_changed_loss = world_model.compute_open_loop_consistency_loss(
            obs_embeddings, rollout_changed_targets, batch['actions'], batch['mask_padding']
        )
        assert torch.allclose(prefix_changed_loss, loss)
        assert not torch.allclose(rollout_changed_loss, loss)

    def test_open_loop_teacher_prefix_diagnostic_measures_the_same_rollout(self):
        torch.manual_seed(31)
        world_model = _build_world_model()
        world_model.rebuild_kv_window_from_tokens = True
        world_model.context_length = 10
        world_model.open_loop_diagnostic_batch_size = 2
        world_model.open_loop_consistency_batch_size = 2
        world_model.open_loop_consistency_horizon = 2
        world_model.open_loop_prefix_transitions = 3
        observation_head_linears = [
            module for module in world_model.head_observations.modules()
            if isinstance(module, torch.nn.Linear)
        ]
        torch.nn.init.normal_(observation_head_linears[-1].weight, std=0.02)
        batch = _make_batch(torch.ones(B, T, dtype=torch.bool))
        obs_embeddings = world_model.tokenizer.encode_to_obs_embeddings(batch['observations'])
        target_embeddings = obs_embeddings.detach().clone()

        diagnostics = world_model.compute_open_loop_latent_diagnostics(
            obs_embeddings.detach(), target_embeddings, batch['actions'], batch['mask_padding']
        )
        differentiable_loss = world_model.compute_open_loop_consistency_loss(
            obs_embeddings, target_embeddings, batch['actions'], batch['mask_padding']
        )

        # The detached mechanism metric must inspect exactly the two transitions trained after
        # [o0,a0,o1,a1,o2,a2,o3], rather than silently reverting to a no-history rollout.
        assert differentiable_loss.item() == pytest.approx(
            diagnostics['open_loop_latent_mse_mean'], rel=1e-5, abs=1e-6
        )
        assert diagnostics['open_loop_latent_mse_first'] == pytest.approx(
            diagnostics['rolling_teacher_latent_mse_first'], rel=1e-5, abs=1e-6
        )
        assert diagnostics['open_loop_latent_mse_first'] == pytest.approx(
            diagnostics['teacher_forced_latent_mse_first'], rel=1e-5, abs=1e-6
        )

    def test_open_loop_recurrent_loss_supervises_all_muzero_style_heads(self):
        torch.manual_seed(29)
        world_model = _build_world_model()
        world_model.rebuild_kv_window_from_tokens = True
        world_model.context_length = 6
        world_model.open_loop_recurrent_loss_weight = 0.1
        world_model.open_loop_consistency_batch_size = 2
        world_model.open_loop_consistency_horizon = 3
        observation_head_linears = [
            module for module in world_model.head_observations.modules()
            if isinstance(module, torch.nn.Linear)
        ]
        torch.nn.init.normal_(observation_head_linears[-1].weight, std=0.02)
        handle = InverseScalarTransform(DiscreteSupport(-50, 51, 1), True)

        losses = world_model.compute_loss(
            _make_batch(torch.ones(B, T, dtype=torch.bool)),
            world_model.tokenizer,
            handle,
            global_step=1,
        )
        recurrent_loss = losses.intermediate_losses['open_loop_recurrent_loss']
        component_names = ('latent', 'reward', 'value', 'policy')
        components = [
            losses.intermediate_losses[f'open_loop_recurrent_{name}_loss']
            for name in component_names
        ]

        assert world_model.training is True
        assert recurrent_loss.requires_grad and torch.isfinite(recurrent_loss)
        assert all(component.requires_grad and torch.isfinite(component) for component in components)
        assert torch.allclose(
            recurrent_loss,
            10. * components[0] + components[1] + 0.5 * components[2] + components[3],
        )
        policy_ce = losses.intermediate_losses['open_loop_recurrent_policy_ce']
        policy_entropy = losses.intermediate_losses['open_loop_recurrent_policy_entropy']
        assert torch.allclose(
            components[3],
            policy_ce - world_model.policy_entropy_weight * policy_entropy,
        )
        expected_total = (
            losses.obs_loss_weight * losses.intermediate_losses['loss_obs']
            + losses.reward_loss_weight * losses.intermediate_losses['loss_rewards']
            + losses.value_loss_weight * losses.intermediate_losses['loss_value']
            + losses.policy_loss_weight * losses.intermediate_losses['loss_policy']
            + 0.1 * recurrent_loss
        )
        assert torch.allclose(losses.loss_total, expected_total)

        transformer_parameter = next(
            parameter for parameter in world_model.transformer.parameters()
            if parameter.requires_grad and parameter.ndim > 1
        )
        head_parameters = [
            [module for module in head.modules() if isinstance(module, torch.nn.Linear)][-1].weight
            for head in (
                world_model.head_observations,
                world_model.head_rewards,
                world_model.head_value,
                world_model.head_policy,
            )
        ]
        gradients = torch.autograd.grad(
            recurrent_loss,
            (transformer_parameter, *head_parameters),
            allow_unused=False,
        )
        assert all(torch.isfinite(gradient).all() for gradient in gradients)

    def test_adaptive_entropy_reweights_open_loop_recurrent_policy_component(self):
        recurrent_loss = torch.tensor(13.0)
        fixed_policy_loss = torch.tensor(3.0)
        policy_ce = torch.tensor(4.0)
        policy_entropy = torch.tensor(2.0)
        adaptive_alpha = torch.tensor(0.25)

        adjusted = apply_open_loop_recurrent_entropy_weight(
            recurrent_loss,
            fixed_policy_loss,
            policy_ce,
            policy_entropy,
            adaptive_alpha,
        )

        assert torch.equal(adjusted, torch.tensor(13.5))

    def test_open_loop_recurrent_teacher_prefix_uses_only_post_prefix_targets(self):
        """A teacher prefix builds context but must not contribute recurrent targets."""
        torch.manual_seed(37)
        world_model = _build_world_model()
        world_model.rebuild_kv_window_from_tokens = True
        world_model.context_length = 10
        world_model.open_loop_consistency_batch_size = 2
        world_model.open_loop_consistency_horizon = 2
        world_model.open_loop_prefix_transitions = 3
        # Production heads intentionally end in zero-initialized projections.  Use non-uniform
        # logits here so changing a categorical target is observable in the loss value.
        for head in (
                world_model.head_observations, world_model.head_rewards,
                world_model.head_value, world_model.head_policy):
            head_linears = [
                module for module in head.modules() if isinstance(module, torch.nn.Linear)
            ]
            torch.nn.init.normal_(head_linears[-1].weight, std=0.02)

        batch = _make_batch(torch.ones(B, T, dtype=torch.bool))
        obs_embeddings = world_model.tokenizer.encode_to_obs_embeddings(batch['observations'])
        target_embeddings = obs_embeddings.detach().clone()
        labels_rewards = batch['rewards'].reshape(-1, SUPPORT_SIZE)
        labels_policy = batch['target_policy'].reshape(-1, A)
        labels_value = batch['target_value'].reshape(-1, SUPPORT_SIZE)

        transformer_input_lengths = []
        hook = world_model.transformer.register_forward_pre_hook(
            lambda _module, inputs: transformer_input_lengths.append(inputs[0].size(1))
        )
        try:
            _, reference = world_model.compute_open_loop_recurrent_loss(
                obs_embeddings, target_embeddings, batch['actions'], batch['mask_padding'],
                labels_rewards, labels_policy, labels_value,
            )
        finally:
            hook.remove()

        # Prefix=3 builds [o0,a0,o1,a1,o2,a2,o3].  Each recurrent step then sees
        # eight tokens after its action and nine after its predicted observation.
        assert transformer_input_lengths == [8, 9, 8, 9]

        prefix_target_embeddings = target_embeddings.clone().view(B, T, 1, EMBED_DIM)
        prefix_target_embeddings[:2, :4] += 10.
        prefix_rewards = labels_rewards.clone().view(B, T, SUPPORT_SIZE)
        prefix_rewards[:2, :3] = torch.nn.functional.one_hot(
            torch.zeros(2, 3, dtype=torch.long), SUPPORT_SIZE
        ).float()
        prefix_policy = labels_policy.clone().view(B, T, A)
        prefix_policy[:2, :4] = torch.nn.functional.one_hot(
            torch.zeros(2, 4, dtype=torch.long), A
        ).float()
        prefix_value = labels_value.clone().view(B, T, SUPPORT_SIZE)
        prefix_value[:2, :4] = torch.nn.functional.one_hot(
            torch.zeros(2, 4, dtype=torch.long), SUPPORT_SIZE
        ).float()
        _, prefix_mutated = world_model.compute_open_loop_recurrent_loss(
            obs_embeddings, prefix_target_embeddings, batch['actions'], batch['mask_padding'],
            prefix_rewards, prefix_policy, prefix_value,
        )
        for name in ('latent', 'reward', 'value', 'policy'):
            assert torch.allclose(prefix_mutated[name], reference[name])

        post_target_embeddings = target_embeddings.clone().view(B, T, 1, EMBED_DIM)
        post_target_embeddings[:2, 4:6] += 10.
        post_rewards = labels_rewards.clone().view(B, T, SUPPORT_SIZE)
        post_rewards[:2, 3:5] = torch.nn.functional.one_hot(
            torch.full((2, 2), SUPPORT_SIZE - 1, dtype=torch.long), SUPPORT_SIZE
        ).float()
        post_policy = labels_policy.clone().view(B, T, A)
        post_policy[:2, 4:6] = torch.nn.functional.one_hot(
            torch.full((2, 2), A - 1, dtype=torch.long), A
        ).float()
        post_value = labels_value.clone().view(B, T, SUPPORT_SIZE)
        post_value[:2, 4:6] = torch.nn.functional.one_hot(
            torch.full((2, 2), SUPPORT_SIZE - 1, dtype=torch.long), SUPPORT_SIZE
        ).float()
        _, post_mutated = world_model.compute_open_loop_recurrent_loss(
            obs_embeddings, post_target_embeddings, batch['actions'], batch['mask_padding'],
            post_rewards, post_policy, post_value,
        )
        for name in ('latent', 'reward', 'value', 'policy'):
            assert not torch.allclose(post_mutated[name], reference[name])

    def test_open_loop_recurrent_reward_uses_transition_not_next_state_mask(self):
        """A terminal action keeps its reward even when its next-state targets are invalid."""
        torch.manual_seed(31)
        world_model = _build_world_model()
        world_model.rebuild_kv_window_from_tokens = True
        world_model.context_length = 6
        world_model.open_loop_consistency_batch_size = 2
        world_model.open_loop_consistency_horizon = 3
        observation_head_linears = [
            module for module in world_model.head_observations.modules()
            if isinstance(module, torch.nn.Linear)
        ]
        torch.nn.init.normal_(observation_head_linears[-1].weight, std=0.02)

        batch = _make_batch(torch.ones(B, T, dtype=torch.bool))
        obs_embeddings = world_model.tokenizer.encode_to_obs_embeddings(batch['observations'])
        target_embeddings = obs_embeddings.detach().clone()
        labels_rewards = batch['rewards'].reshape(-1, SUPPORT_SIZE)
        labels_policy = batch['target_policy'].reshape(-1, A)
        labels_value = batch['target_value'].reshape(-1, SUPPORT_SIZE)

        _, full_components = world_model.compute_open_loop_recurrent_loss(
            obs_embeddings, target_embeddings, batch['actions'], batch['mask_padding'],
            labels_rewards, labels_policy, labels_value,
        )
        terminal_mask = batch['mask_padding'].clone()
        terminal_mask[:2, 3:] = False
        _, terminal_components = world_model.compute_open_loop_recurrent_loss(
            obs_embeddings, target_embeddings, batch['actions'], terminal_mask,
            labels_rewards, labels_policy, labels_value,
        )

        # For the selected two samples, horizon=3 uses current states 0..2 for rewards in both
        # cases.  Only next state 3 becomes invalid, so the reward objective must be unchanged.
        assert torch.allclose(terminal_components['reward'], full_components['reward'])
        assert not torch.allclose(terminal_components['latent'], full_components['latent'])

    def test_compute_loss_exposes_cached_open_loop_diagnostics(self):
        torch.manual_seed(13)
        world_model = _build_world_model()
        world_model.rebuild_kv_window_from_tokens = True
        world_model.context_length = 6
        world_model.open_loop_diagnostic_freq = 1
        world_model.open_loop_diagnostic_batch_size = 2
        handle = InverseScalarTransform(DiscreteSupport(-50, 51, 1), True)

        losses = world_model.compute_loss(
            _make_batch(torch.ones(B, T, dtype=torch.bool)),
            world_model.tokenizer,
            handle,
            global_step=1,
        )

        for key in (
            'open_loop_latent_mse_mean',
            'open_loop_latent_mse_first',
            'open_loop_latent_mse_middle',
            'open_loop_latent_mse_last',
            'rolling_teacher_latent_mse_mean',
            'rolling_teacher_latent_mse_first',
            'rolling_teacher_latent_mse_middle',
            'rolling_teacher_latent_mse_last',
            'teacher_forced_latent_mse_mean',
            'teacher_forced_latent_mse_first',
            'rolling_context_ratio',
            'open_loop_exposure_ratio',
            'open_loop_total_ratio',
        ):
            value = losses.intermediate_losses[key]
            assert isinstance(value, torch.Tensor)
            assert torch.isfinite(value)

        assert losses.intermediate_losses['open_loop_latent_mse_first'] == pytest.approx(
            losses.intermediate_losses['teacher_forced_latent_mse_first'],
            rel=1e-5,
            abs=1e-6,
        )
        assert losses.intermediate_losses['rolling_teacher_latent_mse_first'] == pytest.approx(
            losses.intermediate_losses['teacher_forced_latent_mse_first'],
            rel=1e-5,
            abs=1e-6,
        )

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

    def test_ppo_actor_critic_update_isolated_from_world_model_gradients(self):
        """Large reconstruction gradients must not consume PPO's global clip budget."""
        torch.manual_seed(41)
        wm = _build_world_model(world_model_cls=PPOWorldModel)
        handle = InverseScalarTransform(DiscreteSupport(-50, 51, 1), True)
        batch = _make_batch(torch.ones(B, T, dtype=torch.bool))
        features = torch.randn(B, T, EMBED_DIM)
        actions = batch['actions']
        action_mask = torch.ones(B, T, A, dtype=torch.bool)
        with torch.no_grad():
            behavior_logits = wm.head_policy.head_module(features)
            old_log_prob = torch.distributions.Categorical(logits=behavior_logits).log_prob(actions)
        batch.update(
            actor_critic_only=True,
            ppo_policy_features=features,
            ppo_action_mask=action_mask,
            ppo_old_log_prob=old_log_prob,
            ppo_advantages=torch.randn(B, T),
            ppo_clip_ratio=0.2,
            ppo_entropy_weight=0.01,
        )

        def fail_if_encoded(*args, **kwargs):
            raise AssertionError('PPO fast path must not rerun either tokenizer')

        wm.tokenizer.encode_to_obs_embeddings = fail_if_encoded

        losses = wm.compute_loss(batch, wm.tokenizer, handle, global_step=0)
        assert losses.intermediate_losses['loss_obs'] == 0
        assert losses.intermediate_losses['loss_rewards'] == 0
        losses.loss_total.backward()

        def has_nonzero_grad(module):
            return any(
                parameter.grad is not None and parameter.grad.abs().sum() > 0
                for parameter in module.parameters()
            )

        assert has_nonzero_grad(wm.head_policy)
        assert has_nonzero_grad(wm.head_value)
        assert not has_nonzero_grad(wm.transformer)
        assert not has_nonzero_grad(wm.tokenizer)
        assert not has_nonzero_grad(wm.head_observations)
        assert not has_nonzero_grad(wm.head_rewards)

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

    def test_per_sample_is_weights_adds_auxiliary_losses_without_replay_weighting(self):
        per_sample_obs = torch.arange(1, B + 1, dtype=torch.float32)
        consistency = torch.tensor(2.5)
        recurrent = torch.tensor(4.0)
        stub_losses = SimpleNamespace(
            obs_loss_weight=1.0,
            reward_loss_weight=0.0,
            value_loss_weight=0.0,
            policy_loss_weight=0.0,
            latent_recon_loss_weight=0.0,
            perceptual_loss_weight=0.0,
            open_loop_consistency_loss_weight=0.2,
            open_loop_recurrent_loss_weight=0.3,
            intermediate_losses={
                'per_sample_loss_obs': per_sample_obs,
                'per_sample_loss_rewards': torch.zeros(B),
                'per_sample_loss_value': torch.zeros(B),
                'latent_recon_loss': torch.tensor(0.),
                'perceptual_loss': torch.tensor(0.),
                'open_loop_consistency_loss': consistency,
                'open_loop_recurrent_loss': recurrent,
            },
        )
        weights = torch.linspace(0.1, 1.7, B)

        actual = apply_per_sample_is_weights(
            weights,
            stub_losses,
            per_sample_policy_loss=torch.zeros(B),
            scalar_total_loss=torch.tensor(-999.),
        )
        expected = (
            (weights * per_sample_obs).mean()
            + 0.2 * consistency
            + 0.3 * recurrent
        )

        assert torch.allclose(actual, expected)

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

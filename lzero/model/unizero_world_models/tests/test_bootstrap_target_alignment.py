"""Regression tests for UniZero's H+1 bootstrap-observation target path."""

import numpy as np
import pytest
import torch

from lzero.model.unizero_world_models.tests.test_per_sample_is_weights import _build_world_model


def _randomize_last_linear(head: torch.nn.Module) -> None:
    """Undo the production zero-init so this unit test can observe input dependence."""
    for layer in reversed(head.head_module):
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.normal_(layer.weight, mean=0.0, std=0.05)
            if layer.bias is not None:
                torch.nn.init.zeros_(layer.bias)
            return
    raise AssertionError("head does not contain a linear output layer")


@pytest.mark.unittest
class TestBootstrapTargetAlignment:

    def test_target_initial_inference_latents_equal_direct_tokenizer_encoding(self):
        """Skipping the unused legacy Transformer must preserve every replay-root latent."""
        torch.manual_seed(9)
        world_model = _build_world_model().eval()
        sequence_count, action_steps = 2, 3
        observations = torch.rand(
            sequence_count * (action_steps + 1), 3, 64, 64
        )
        actions = np.random.default_rng(9).integers(
            0, world_model.action_space_size, (sequence_count, action_steps)
        )

        direct = world_model.tokenizer.encode_to_obs_embeddings(observations)
        _, initial_inference_latents = world_model.reset_for_initial_inference(
            {'obs': observations, 'action': actions, 'current_obs': None},
            start_pos=np.zeros(sequence_count, dtype=np.int64),
        )

        assert torch.equal(initial_inference_latents, direct)

    def test_combined_sequence_appends_final_observation_without_action(self):
        """H actions and H+1 observations must form H complete blocks plus a final obs block."""
        torch.manual_seed(10)
        world_model = _build_world_model().eval()
        batch_size, action_steps = 2, 3
        observations = torch.randn(
            batch_size,
            action_steps + 1,
            world_model.num_observations_tokens,
            world_model.config.embed_dim,
        )
        actions = torch.randint(
            0, world_model.action_space_size, (batch_size, action_steps, 1)
        )

        sequence, num_steps = world_model._process_obs_act_combined(
            {'obs_embeddings_and_act_tokens': (observations, actions)}, prev_steps=0
        )

        tokens_per_block = world_model.num_observations_tokens + 1
        assert num_steps == action_steps * tokens_per_block + world_model.num_observations_tokens
        expected_final = observations[:, -1] + world_model.pos_emb(
            torch.arange(action_steps * tokens_per_block, num_steps)
        )
        assert torch.allclose(sequence[:, -world_model.num_observations_tokens:], expected_final)

    def test_target_inference_uses_real_final_observation(self):
        """Buffer reanalysis must use o[t+H] for its final refreshed policy target."""
        torch.manual_seed(11)
        world_model = _build_world_model().eval()
        _randomize_last_linear(world_model.head_value)
        _randomize_last_linear(world_model.head_policy)
        batch_size, action_steps = 2, 3
        observations = torch.randn(
            batch_size * (action_steps + 1),
            world_model.num_observations_tokens,
            world_model.config.embed_dim,
        )
        actions = np.random.default_rng(11).integers(
            0, world_model.action_space_size, size=(batch_size, action_steps)
        )
        start_pos = np.zeros(batch_size, dtype=np.int64)
        world_model.reanalyze_phase = True

        baseline = world_model.wm_forward_for_initial_infererence(
            observations, actions, current_obs_embeddings=None, start_pos=start_pos
        )
        perturbed_observations = observations.clone().view(
            batch_size,
            action_steps + 1,
            world_model.num_observations_tokens,
            world_model.config.embed_dim,
        )
        final_perturbation = torch.linspace(
            -3.0, 3.0, world_model.config.embed_dim
        ).view(1, 1, -1)
        perturbed_observations[:, -1].add_(final_perturbation)
        perturbed = world_model.wm_forward_for_initial_infererence(
            perturbed_observations.view_as(observations),
            actions,
            current_obs_embeddings=None,
            start_pos=start_pos,
        )

        baseline_value = baseline.logits_value.view(batch_size, action_steps + 1, -1)
        perturbed_value = perturbed.logits_value.view(batch_size, action_steps + 1, -1)
        baseline_policy = baseline.logits_policy.view(batch_size, action_steps + 1, -1)
        perturbed_policy = perturbed.logits_policy.view(batch_size, action_steps + 1, -1)

        assert torch.equal(baseline_value[:, :-1], perturbed_value[:, :-1])
        assert torch.equal(baseline_policy[:, :-1], perturbed_policy[:, :-1])
        assert not torch.allclose(baseline_value[:, -1], perturbed_value[:, -1])
        assert not torch.allclose(baseline_policy[:, -1], perturbed_policy[:, -1])

    def test_regular_value_target_keeps_legacy_shape_without_extra_transformer_token(self):
        """The unused final value placeholder must not add compute to every learner sample."""
        world_model = _build_world_model().eval()
        batch_size, action_steps = 2, 3
        observations = torch.randn(
            batch_size * (action_steps + 1),
            world_model.num_observations_tokens,
            world_model.config.embed_dim,
        )
        actions = np.zeros((batch_size, action_steps), dtype=np.int64)

        outputs = world_model.wm_forward_for_initial_infererence(
            observations,
            actions,
            current_obs_embeddings=None,
            start_pos=np.zeros(batch_size, dtype=np.int64),
        )

        assert outputs.output_sequence.size(1) == action_steps * world_model.config.tokens_per_block
        assert outputs.logits_value.shape[0] == batch_size * (action_steps + 1)
        values = outputs.logits_value.view(batch_size, action_steps + 1, -1)
        assert torch.equal(values[:, -1], values[:, -2])

    def test_rejects_more_than_one_unpaired_observation_step(self):
        world_model = _build_world_model().eval()
        observations = torch.randn(1, 5, 1, world_model.config.embed_dim)
        actions = torch.zeros(1, 3, 1, dtype=torch.long)
        with pytest.raises(ValueError, match="or exceed it by one"):
            world_model._process_obs_act_combined(
                {'obs_embeddings_and_act_tokens': (observations, actions)}, prev_steps=0
            )

    def test_full_unroll_supports_bootstrap_partial_block_beyond_legacy_limit(self):
        """The real H+1 target must fit when H already equals configured max_blocks."""
        torch.manual_seed(12)
        world_model = _build_world_model().eval()
        action_steps = world_model.config.max_blocks
        observations = torch.randn(
            1, action_steps + 1, world_model.num_observations_tokens, world_model.config.embed_dim
        )
        actions = torch.randint(0, world_model.action_space_size, (1, action_steps, 1))

        outputs = world_model.forward(
            {'obs_embeddings_and_act_tokens': (observations, actions)}, start_pos=np.zeros(1, dtype=np.int64)
        )

        assert outputs.output_sequence.size(1) == world_model.config.max_tokens + world_model.num_observations_tokens
        assert outputs.logits_value.shape[:2] == (1, action_steps + 1)
        assert outputs.logits_policy.shape[:2] == (1, action_steps + 1)

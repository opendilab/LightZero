"""Regression tests for batched UniZero caches with unequal valid lengths."""

import copy
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from easydict import EasyDict

from lzero.model.unizero_world_models.transformer import (
    Transformer,
    TransformerConfig,
    apply_rotary_emb,
)
from lzero.model.unizero_world_models.utils import hash_state
from lzero.model.unizero_world_models.world_model_multitask import WorldModelMT
from lzero.model.unizero_world_models.tests.test_per_sample_is_weights import _build_world_model


def _build_transformer() -> Transformer:
    return Transformer(
        TransformerConfig(
            tokens_per_block=2,
            max_blocks=8,
            attention='causal',
            num_layers=2,
            num_heads=2,
            embed_dim=64,
            embed_pdrop=0.0,
            resid_pdrop=0.0,
            attn_pdrop=0.0,
            task_embed_option='none',
            register_token_num=0,
            register_token_shared=False,
        )
    ).eval()


def _build_rotary_transformer() -> Transformer:
    return Transformer(
        TransformerConfig(
            tokens_per_block=2,
            max_blocks=8,
            attention='causal',
            num_layers=2,
            num_heads=2,
            embed_dim=64,
            embed_pdrop=0.0,
            resid_pdrop=0.0,
            attn_pdrop=0.0,
            task_embed_option='none',
            register_token_num=0,
            register_token_shared=False,
            rotary_emb=True,
            rope_theta=10000.0,
            max_seq_len=128,
        )
    ).eval()


def _copy_into_left_padded_batch(transformer, short_cache, long_cache):
    batched = transformer.generate_empty_keys_values(n=2, max_tokens=16)
    for layer_index in range(transformer.config.num_layers):
        for cache_name in ('_k_cache', '_v_cache'):
            short = getattr(short_cache._keys_values[layer_index], cache_name)
            long = getattr(long_cache._keys_values[layer_index], cache_name)
            destination = getattr(batched._keys_values[layer_index], cache_name)
            destination._cache.zero_()
            destination._cache[0, :, 3:6, :] = short._cache[0, :, :3, :]
            destination._cache[1, :, :6, :] = long._cache[0, :, :6, :]
            destination._size = 6
    return batched


@pytest.mark.unittest
class TestVariableLengthKVCache:

    def test_single_and_multitask_root_cache_rings_share_the_search_capacity_rule(self):
        config = SimpleNamespace(max_cache_size=5000, num_simulations=50)
        assert WorldModelMT._initial_cache_pool_size(config) == 256

        memory_limited = SimpleNamespace(max_cache_size=100, num_simulations=50)
        assert WorldModelMT._initial_cache_pool_size(memory_limited) == 100

        with pytest.raises(ValueError, match='max_cache_size must be positive'):
            WorldModelMT._initial_cache_pool_size(
                SimpleNamespace(max_cache_size=0, num_simulations=50)
            )

    def test_reanalysis_root_contexts_include_replay_prefix_and_roll_exactly(self):
        """Every H+1 replay root should recover the same raw window as online inference."""
        torch.manual_seed(12)
        world_model = _build_world_model().eval()
        world_model.rebuild_kv_window_from_tokens = True
        world_model.context_length = 10

        roots_per_sequence = 4
        roots = torch.randn(2 * roots_per_sequence, 1, world_model.embed_dim)
        batch_actions = torch.tensor([[0, 1, 2], [3, 4, 5]])
        prefix_latents = [
            [torch.randn(1, world_model.embed_dim), torch.randn(1, world_model.embed_dim)],
            [],
        ]
        prefix_actions = [[4, 5], []]

        contexts = world_model.build_reanalysis_root_token_contexts(
            roots.numpy(),
            batch_actions,
            roots_per_sequence,
            prefix_latents,
            prefix_actions,
        )

        assert [context.size(0) for context in contexts] == [5, 7, 7, 7, 1, 3, 5, 7]
        expected_first = torch.cat((
            prefix_latents[0][0],
            world_model.act_embedding_table(torch.tensor([4])),
            prefix_latents[0][1],
            world_model.act_embedding_table(torch.tensor([5])),
            roots[0],
        ))
        assert torch.equal(contexts[0], expected_first)

        # Root 2 has overflowed the online threshold, so it must contain the
        # exact last seven raw tokens at positions 0..6.
        full_root_two = torch.cat((
            expected_first,
            world_model.act_embedding_table(torch.tensor([0])),
            roots[1],
            world_model.act_embedding_table(torch.tensor([1])),
            roots[2],
        ))
        assert torch.equal(contexts[2], full_root_two[-7:])

    def test_seeded_reanalysis_root_cache_matches_direct_prefix_forward(self):
        """MCTS root lookup must hit the K/V produced by its true replay prefix."""
        torch.manual_seed(13)
        world_model = _build_world_model().eval()
        world_model.rebuild_kv_window_from_tokens = True
        world_model.context_length = 10
        world_model.reanalyze_phase = True
        world_model.clear_caches()
        policy_linears = [
            module for module in world_model.head_policy.modules()
            if isinstance(module, torch.nn.Linear)
        ]
        torch.nn.init.normal_(policy_linears[-1].weight, std=0.02)

        # Deliberately use the same current latent with different histories.
        # A global state-hash map would silently overwrite one root context.
        roots = torch.randn(2, 1, world_model.embed_dim)
        roots[1].copy_(roots[0])
        contexts = [
            roots[0].clone(),
            torch.randn(3, world_model.embed_dim),
        ]
        contexts[1][-1].copy_(roots[1, 0])
        contextual_policy_logits = world_model.seed_reanalysis_root_caches(
            roots.numpy(), contexts
        )
        assert not torch.allclose(
            torch.as_tensor(contextual_policy_logits[0]),
            torch.as_tensor(contextual_policy_logits[1]),
        )

        for root_index, raw_context in enumerate(contexts):
            cache_key = hash_state(roots[root_index].reshape(-1).numpy())
            cache_index = world_model.past_kv_cache_init_infer_envs[root_index][cache_key]
            stored = world_model.shared_pool_init_infer[root_index][cache_index]
            expected = world_model.transformer.generate_empty_keys_values(
                n=1, max_tokens=world_model.context_length
            )
            positioned = raw_context.unsqueeze(0) + world_model.pos_emb(
                torch.arange(raw_context.size(0))
            )
            world_model.transformer(positioned, past_keys_values=expected)
            direct_hidden = world_model.transformer(positioned)
            direct_policy_logits = world_model.head_policy(
                direct_hidden, num_steps=raw_context.size(0), prev_steps=0
            )[:, -1]
            if world_model.use_policy_logits_clip:
                direct_policy_logits = world_model._apply_policy_logits_control(
                    direct_policy_logits
                )

            assert stored.size == raw_context.size(0)
            assert torch.allclose(
                torch.as_tensor(contextual_policy_logits[root_index]),
                direct_policy_logits[0],
            )
            assert torch.equal(
                world_model.past_token_context_init_infer_envs[root_index][cache_key], raw_context
            )
            for stored_layer, expected_layer in zip(stored._keys_values, expected._keys_values):
                size = raw_context.size(0)
                assert torch.allclose(
                    stored_layer._k_cache._cache[:, :, :size, :],
                    expected_layer._k_cache._cache[:, :, :size, :],
                )
                assert torch.allclose(
                    stored_layer._v_cache._cache[:, :, :size, :],
                    expected_layer._v_cache._cache[:, :, :size, :],
                )

        previous_hits = world_model.hit_count
        sizes = world_model.retrieve_or_generate_kvcache(roots.numpy(), ready_env_num=2)
        assert sizes == [1, 3]
        assert world_model.hit_count == previous_hits + 2
        assert world_model.reanalysis_root_seed_count == 2
        assert world_model.reanalysis_root_seed_hit_count == 2
        assert not world_model._reanalysis_seeded_root_keys

    def test_bootstrap_root_values_use_exact_online_context(self):
        """TD bootstrap logits must equal a direct forward of each rolling root history."""
        torch.manual_seed(29)
        world_model = _build_world_model().eval()
        world_model.context_length = 10
        value_linears = [
            module for module in world_model.head_value.modules()
            if isinstance(module, torch.nn.Linear)
        ]
        torch.nn.init.normal_(value_linears[-1].weight, std=0.02)

        current_root = torch.randn(1, world_model.embed_dim)
        contexts = [
            current_root.clone(),
            torch.cat((torch.randn(2, world_model.embed_dim), current_root)),
        ]
        contextual_values = world_model.evaluate_root_token_context_values(contexts)

        assert not torch.allclose(contextual_values[0], contextual_values[1])
        for root_index, raw_context in enumerate(contexts):
            positioned = raw_context.unsqueeze(0) + world_model.pos_emb(
                torch.arange(raw_context.size(0))
            )
            hidden = world_model.transformer(positioned)
            expected = world_model.head_value(
                hidden, num_steps=raw_context.size(0), prev_steps=0
            )[:, -1]
            assert torch.allclose(contextual_values[root_index], expected[0])

    def test_contextual_bootstrap_uses_the_requested_multitask_value_head(self):
        """A task-specific replay target must not silently use the stale single-task head."""
        torch.manual_seed(31)
        world_model = _build_world_model().eval()
        world_model.task_embed_option = 'none'
        task_zero_head = copy.deepcopy(world_model.head_value)
        task_one_head = copy.deepcopy(world_model.head_value)
        task_zero_last = [
            module for module in task_zero_head.modules()
            if isinstance(module, torch.nn.Linear)
        ][-1]
        task_one_last = [
            module for module in task_one_head.modules()
            if isinstance(module, torch.nn.Linear)
        ][-1]
        torch.nn.init.normal_(task_zero_last.weight, std=0.01)
        torch.nn.init.normal_(task_one_last.weight, std=0.03)
        world_model.head_value_multi_task = torch.nn.ModuleList([
            task_zero_head, task_one_head
        ])
        world_model.share_head = False
        world_model.use_moe_head = False
        world_model.use_softmoe_head = False

        context = torch.randn(3, world_model.embed_dim)
        task_zero = world_model.evaluate_root_token_context_values([context], task_id=0)
        task_one = world_model.evaluate_root_token_context_values([context], task_id=1)

        assert not torch.allclose(task_zero, task_one)
        positioned = context.unsqueeze(0) + world_model.pos_emb(torch.arange(3))
        hidden = world_model.transformer(positioned, task_id=1)
        expected = task_one_head(hidden, num_steps=3, prev_steps=0)[:, -1]
        assert torch.allclose(task_one[0], expected[0])

    def test_contextual_replay_fails_loudly_for_unsupported_task_token_modes(self):
        world_model = _build_world_model().eval()
        world_model.task_embed_option = 'add_task_embed'
        with pytest.raises(NotImplementedError, match='task_embed_option'):
            world_model.build_reanalysis_root_token_contexts(
                latent_state_roots=torch.randn(2, 1, world_model.embed_dim),
                batch_actions=torch.tensor([[0]]),
                roots_per_sequence=2,
                task_id=0,
            )

    def test_multitask_transformer_pass_propagates_task_id(self):
        calls = []

        class _Transformer:

            def __call__(self, sequences, past_keys_values=None, **kwargs):
                calls.append(kwargs.get('task_id'))
                return sequences

        holder = SimpleNamespace(transformer=_Transformer())
        sequences = torch.randn(2, 3, 4)
        output = WorldModelMT._transformer_pass(
            holder,
            sequences,
            past_keys_values=None,
            kvcache_independent=False,
            valid_context_lengths=None,
            task_id=3,
        )

        assert torch.equal(output, sequences)
        assert calls == [3]

    def test_multitask_rotary_mode_fails_loudly_without_per_root_positions(self):
        with pytest.raises(NotImplementedError, match='per-root episode positions'):
            WorldModelMT(EasyDict(rotary_emb=True), tokenizer=None)

    def test_position_differences_are_computed_from_final_initialized_weights(self):
        """Fresh-run correction tensors must not predate the model's final initialization."""
        torch.manual_seed(11)
        world_model = _build_world_model().eval()
        for layer_index in range(world_model.config.num_layers):
            expected_keys = world_model._get_positional_embedding(layer_index, 'key')
            expected_values = world_model._get_positional_embedding(layer_index, 'value')
            assert torch.equal(world_model.positional_embedding_k[layer_index], expected_keys)
            assert torch.equal(world_model.positional_embedding_v[layer_index], expected_values)

    def test_rotary_cached_inference_matches_full_sequence(self):
        """The next-token RoPE position must make cached and full inference equivalent."""
        torch.manual_seed(5)
        transformer = _build_rotary_transformer()
        tokens = torch.randn(1, 4, 64)

        with torch.no_grad():
            # Start beyond the initial 2*max_seq_len table to exercise exact
            # absolute-position extension (modulo wrapping would break cache equivalence).
            full_output = transformer(tokens, start_pos=255)
            cache = transformer.generate_empty_keys_values(n=1, max_tokens=16)
            transformer(tokens[:, :3], past_keys_values=cache, start_pos=255)
            cached_output = transformer(tokens[:, 3:], past_keys_values=cache, start_pos=258)

        assert torch.allclose(cached_output, full_output[:, 3:], atol=1e-5, rtol=1e-5)

    def test_rotary_cached_inference_uses_start_position(self):
        """Changing only a cached token's position must change its relative attention."""
        torch.manual_seed(6)
        transformer = _build_rotary_transformer()
        prefix = torch.randn(1, 3, 64)
        next_token = torch.randn(1, 1, 64)

        with torch.no_grad():
            cache = transformer.generate_empty_keys_values(n=1, max_tokens=16)
            transformer(prefix, past_keys_values=cache, start_pos=20)
            correct = transformer(next_token, past_keys_values=cache.clone(), start_pos=23)
            wrong = transformer(next_token, past_keys_values=cache.clone(), start_pos=31)

        assert not torch.allclose(correct, wrong, atol=1e-6, rtol=1e-6)

    def test_rotary_reanalysis_accepts_flat_root_position_vectors(self):
        """The recurrent MCTS path receives one absolute timestep per flattened root."""
        torch.manual_seed(7)
        world_model = _build_world_model(rotary_emb=True).eval()
        world_model.reanalyze_phase = True
        actions = torch.tensor([[1], [2]], dtype=torch.long)

        with torch.no_grad():
            output = world_model.forward(
                {'act_tokens': actions},
                is_init_infer=False,
                start_pos=np.array([10, 20], dtype=np.int64),
                search_depth=[0, 1],
            )

        assert output.output_sequence.shape[:2] == (2, 1)

    def test_rotary_attention_map_uses_the_same_query_key_rotation(self):
        """Attention-map diagnostics must not silently omit RoPE."""
        torch.manual_seed(10)
        transformer = _build_rotary_transformer()
        tokens = torch.randn(1, 4, 64)
        frequencies = transformer._get_rotary_frequencies(tokens, start_pos=17)

        block = transformer.blocks[0]
        normalized = block.ln1(tokens)
        attention = block.attn.get_attention_map(normalized, freqs_cis=frequencies)

        batch_size, token_count, embed_dim = normalized.shape
        head_size = embed_dim // transformer.config.num_heads
        query = block.attn.query(normalized).view(
            batch_size, token_count, transformer.config.num_heads, head_size
        ).transpose(1, 2)
        key = block.attn.key(normalized).view(
            batch_size, token_count, transformer.config.num_heads, head_size
        ).transpose(1, 2)
        rotated_query, rotated_key = apply_rotary_emb(query, key, frequencies)
        expected_logits = (rotated_query @ rotated_key.transpose(-2, -1)) / math.sqrt(head_size)
        causal_mask = block.attn.mask[:token_count, :token_count]
        expected = F.softmax(
            expected_logits.masked_fill(causal_mask == 0, float('-inf')), dim=-1
        )

        unrotated_logits = (query @ key.transpose(-2, -1)) / math.sqrt(head_size)
        unrotated = F.softmax(
            unrotated_logits.masked_fill(causal_mask == 0, float('-inf')), dim=-1
        )

        assert attention.shape == (1, transformer.config.num_heads, 4, 4)
        assert torch.isfinite(attention).all()
        assert torch.allclose(attention, expected, atol=1e-6, rtol=1e-6)
        assert not torch.allclose(attention, unrotated, atol=1e-6, rtol=1e-6)

    def test_masked_left_padding_matches_independent_inference(self):
        """Batching 3- and 6-token caches must preserve both independent outputs."""
        torch.manual_seed(0)
        transformer = _build_transformer()
        short_prefix = torch.randn(1, 3, 64)
        long_prefix = torch.randn(1, 6, 64)
        next_tokens = torch.randn(2, 1, 64)

        with torch.no_grad():
            short_cache = transformer.generate_empty_keys_values(n=1, max_tokens=16)
            long_cache = transformer.generate_empty_keys_values(n=1, max_tokens=16)
            transformer(short_prefix, past_keys_values=short_cache)
            transformer(long_prefix, past_keys_values=long_cache)

            short_reference = transformer(next_tokens[:1], past_keys_values=short_cache.clone())
            long_reference = transformer(next_tokens[1:], past_keys_values=long_cache.clone())

            batched_cache = _copy_into_left_padded_batch(transformer, short_cache, long_cache)
            batched_output = transformer(
                next_tokens,
                past_keys_values=batched_cache,
                valid_context_lengths=torch.tensor([3, 6]),
            )

        reference = torch.cat([short_reference, long_reference], dim=0)
        assert torch.allclose(batched_output, reference, atol=1e-5, rtol=1e-5)

    def test_root_cache_storage_removes_batch_padding(self):
        """Per-env root caches must not persist another env's left-padding as context."""
        torch.manual_seed(1)
        world_model = _build_world_model().eval()
        world_model.keys_values_wm = world_model.transformer.generate_empty_keys_values(
            n=2, max_tokens=world_model.context_length
        )

        expected_short = []
        for layer in world_model.keys_values_wm._keys_values:
            layer._k_cache._cache.normal_()
            layer._v_cache._cache.normal_()
            layer._k_cache._cache[0, :, :2, :].zero_()
            layer._v_cache._cache[0, :, :2, :].zero_()
            layer._k_cache._size = 5
            layer._v_cache._size = 5
            expected_short.append(layer._k_cache._cache[0, :, 2:5, :].clone())

        latent_state = torch.randn(2, 1, world_model.config.embed_dim)
        world_model.update_cache_context(
            latent_state,
            is_init_infer=True,
            valid_context_lengths=[3, 5],
            env_ids=[0, 1],
        )

        for env_id, expected_size in enumerate((3, 5)):
            cache_key = hash_state(latent_state[env_id].reshape(-1).numpy())
            cache_index = world_model.past_kv_cache_init_infer_envs[env_id][cache_key]
            stored = world_model.shared_pool_init_infer[env_id][cache_index]
            assert stored.size == expected_size
            if env_id == 0:
                for layer_index, expected in enumerate(expected_short):
                    actual = stored._keys_values[layer_index]._k_cache._cache[0, :, :3, :]
                    assert torch.equal(actual, expected)

    def test_root_position_embeddings_use_each_cache_length(self):
        """A root batch must not assign the longest env's position to every token."""
        torch.manual_seed(2)
        world_model = _build_world_model().eval()
        embeddings = torch.zeros(2, 1, world_model.config.embed_dim)
        positioned = world_model._add_position_embeddings(
            embeddings,
            prev_steps=6,
            num_steps=1,
            kvcache_independent=False,
            is_init_infer=True,
            valid_context_lengths=[3, 6],
        )
        expected = world_model.pos_emb(torch.tensor([[3], [6]]))
        assert torch.equal(positioned, expected)

    def test_recurrent_position_embeddings_extend_past_the_checkpoint_table(self):
        """A full H-step root must be able to append the H+1 bootstrap observation."""
        world_model = _build_world_model().eval()
        table_size = world_model.pos_emb.num_embeddings
        embeddings = torch.zeros(2, 1, world_model.embed_dim)

        positioned = world_model._add_position_embeddings(
            embeddings,
            prev_steps=0,
            num_steps=1,
            kvcache_independent=False,
            is_init_infer=True,
            valid_context_lengths=[table_size - 1, table_size],
        )

        expected = world_model._lookup_position_embeddings(
            torch.tensor([[table_size - 1], [table_size]])
        )
        assert torch.equal(positioned, expected)
        multitask_positioned = WorldModelMT._add_position_embeddings(
            world_model,
            embeddings,
            prev_steps=0,
            num_steps=1,
            kvcache_independent=False,
            is_init_infer=True,
            valid_context_lengths=[table_size - 1, table_size],
        )
        assert torch.equal(multitask_positioned, expected)
        assert torch.equal(positioned[0, 0], world_model.pos_emb.weight[-1])
        assert torch.allclose(
            positioned[1, 0],
            2 * world_model.pos_emb.weight[-1] - world_model.pos_emb.weight[-2],
        )

    def test_open_loop_diagnostic_matches_teacher_forcing_at_first_transition(self):
        """Before predictions feed back, open-loop and teacher-forced dynamics must agree."""
        torch.manual_seed(12)
        world_model = _build_world_model()
        world_model.rebuild_kv_window_from_tokens = True
        world_model.context_length = 6
        world_model.open_loop_diagnostic_batch_size = 2
        batch_size, sequence_length = 2, 6
        observations = torch.randn(
            batch_size,
            sequence_length,
            world_model.num_observations_tokens,
            world_model.embed_dim,
        )
        actions = torch.randint(
            0, world_model.action_space_size, (batch_size, sequence_length)
        )
        # Production UniZero uses non-zero transformer dropout.  The diagnostic must compare
        # open-loop and teacher-forced predictions under one shared eval-mode regime rather than
        # leaking independent training-mode dropout noise into the exposure-bias estimate.
        for module in world_model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.5
        world_model.train()
        metrics = world_model.compute_open_loop_latent_diagnostics(
            obs_embeddings=observations,
            target_obs_embeddings=observations,
            actions=actions,
            mask_padding=torch.ones(batch_size, sequence_length, dtype=torch.bool),
        )

        assert world_model.training is True
        assert metrics['open_loop_latent_mse_first'] == pytest.approx(
            metrics['teacher_forced_latent_mse_first'], rel=1e-5, abs=1e-6
        )
        assert metrics['rolling_teacher_latent_mse_first'] == pytest.approx(
            metrics['teacher_forced_latent_mse_first'], rel=1e-5, abs=1e-6
        )
        assert metrics['open_loop_latent_mse_mean'] >= 0
        assert torch.isfinite(torch.tensor(metrics['rolling_context_ratio']))
        assert torch.isfinite(torch.tensor(metrics['open_loop_exposure_ratio']))
        assert torch.isfinite(torch.tensor(metrics['open_loop_total_ratio']))
        assert metrics['open_loop_total_ratio'] == pytest.approx(
            metrics['rolling_context_ratio'] * metrics['open_loop_exposure_ratio'],
            rel=1e-5,
            abs=1e-6,
        )

    def test_exact_window_reset_recomputes_latest_observation(self):
        """A full window must be rebuilt, not algebraically shift contextual K/V tensors."""
        torch.manual_seed(3)
        world_model = _build_world_model().eval()
        world_model.exact_kv_window_reset = True
        world_model.keys_values_wm = world_model.transformer.generate_empty_keys_values(
            n=1, max_tokens=world_model.context_length
        )
        full_size = world_model.context_length - 1
        for layer in world_model.keys_values_wm._keys_values:
            layer._k_cache._cache.normal_()
            layer._v_cache._cache.normal_()
            layer._k_cache._size = full_size
            layer._v_cache._size = full_size

        latent_state = torch.randn(1, 1, world_model.config.embed_dim)
        expected = world_model.transformer.generate_empty_keys_values(
            n=1, max_tokens=world_model.context_length
        )
        world_model.forward(
            {'obs_embeddings': latent_state},
            past_keys_values=expected,
            is_init_infer=True,
            start_pos=0,
        )

        world_model.update_cache_context(
            latent_state,
            is_init_infer=True,
            valid_context_lengths=[full_size],
            env_ids=[0],
        )

        cache_key = hash_state(latent_state[0].reshape(-1).numpy())
        cache_index = world_model.past_kv_cache_init_infer_envs[0][cache_key]
        stored = world_model.shared_pool_init_infer[0][cache_index]
        assert stored.size == 1
        for stored_layer, expected_layer in zip(stored._keys_values, expected._keys_values):
            assert torch.allclose(
                stored_layer._k_cache._cache[:, :, :1, :],
                expected_layer._k_cache._cache[:, :, :1, :],
            )
            assert torch.allclose(
                stored_layer._v_cache._cache[:, :, :1, :],
                expected_layer._v_cache._cache[:, :, :1, :],
            )

    def test_exact_window_reset_batches_only_overflowing_samples(self):
        """Mixed cache phases must rebuild full samples and preserve shorter samples exactly."""
        torch.manual_seed(4)
        world_model = _build_world_model().eval()
        world_model.exact_kv_window_reset = True
        world_model.keys_values_wm = world_model.transformer.generate_empty_keys_values(
            n=2, max_tokens=world_model.context_length
        )
        full_size = world_model.context_length - 1
        short_size = 3
        expected_short = []
        for layer in world_model.keys_values_wm._keys_values:
            layer._k_cache._cache.normal_()
            layer._v_cache._cache.normal_()
            layer._k_cache._size = full_size
            layer._v_cache._size = full_size
            expected_short.append((
                layer._k_cache._cache[1:2, :, full_size - short_size:full_size, :].clone(),
                layer._v_cache._cache[1:2, :, full_size - short_size:full_size, :].clone(),
            ))

        latent_state = torch.randn(2, 1, world_model.config.embed_dim)
        expected_reset = world_model.transformer.generate_empty_keys_values(
            n=1, max_tokens=world_model.context_length
        )
        world_model.forward(
            {'obs_embeddings': latent_state[:1]},
            past_keys_values=expected_reset,
            is_init_infer=True,
            start_pos=0,
        )

        world_model.update_cache_context(
            latent_state,
            is_init_infer=True,
            valid_context_lengths=[full_size, short_size],
            env_ids=[0, 1],
        )

        stored = []
        for env_id in range(2):
            cache_key = hash_state(latent_state[env_id].reshape(-1).numpy())
            cache_index = world_model.past_kv_cache_init_infer_envs[env_id][cache_key]
            stored.append(world_model.shared_pool_init_infer[env_id][cache_index])

        assert stored[0].size == 1
        assert stored[1].size == short_size
        for layer_index, (reset_layer, short_layer) in enumerate(zip(
            expected_reset._keys_values, expected_short
        )):
            assert torch.allclose(
                stored[0]._keys_values[layer_index]._k_cache._cache[:, :, :1, :],
                reset_layer._k_cache._cache[:, :, :1, :],
            )
            assert torch.allclose(
                stored[0]._keys_values[layer_index]._v_cache._cache[:, :, :1, :],
                reset_layer._v_cache._cache[:, :, :1, :],
            )
            assert torch.equal(
                stored[1]._keys_values[layer_index]._k_cache._cache[:, :, :short_size, :],
                short_layer[0],
            )
            assert torch.equal(
                stored[1]._keys_values[layer_index]._v_cache._cache[:, :, :short_size, :],
                short_layer[1],
            )

    def test_raw_token_window_rebuild_matches_full_retained_prefix(self):
        """Absolute-position rolling must replay every retained raw token, not just the last obs."""
        torch.manual_seed(7)
        world_model = _build_world_model().eval()
        world_model.rebuild_kv_window_from_tokens = True
        world_model.exact_kv_window_reset = False
        world_model.keys_values_wm = world_model.transformer.generate_empty_keys_values(
            n=1, max_tokens=world_model.context_length
        )
        full_size = world_model.context_length - 1
        for layer in world_model.keys_values_wm._keys_values:
            layer._k_cache._cache.normal_()
            layer._v_cache._cache.normal_()
            layer._k_cache._size = full_size
            layer._v_cache._size = full_size

        raw_history = torch.randn(full_size, world_model.config.embed_dim)
        world_model.keys_values_wm_token_context_list = [raw_history.clone()]
        retained = raw_history[-(world_model.context_length - 3):]
        expected = world_model.transformer.generate_empty_keys_values(
            n=1, max_tokens=world_model.context_length
        )
        positioned = retained.unsqueeze(0) + world_model.pos_emb(
            torch.arange(retained.size(0))
        )
        world_model.transformer(positioned, past_keys_values=expected)

        latent_state = raw_history[-1:].unsqueeze(0)
        world_model.update_cache_context(
            latent_state,
            is_init_infer=True,
            valid_context_lengths=[full_size],
            env_ids=[0],
        )

        cache_key = hash_state(latent_state[0].reshape(-1).numpy())
        cache_index = world_model.past_kv_cache_init_infer_envs[0][cache_key]
        stored = world_model.shared_pool_init_infer[0][cache_index]
        assert stored.size == retained.size(0)
        assert torch.equal(
            world_model.past_token_context_init_infer_envs[0][cache_key], retained
        )
        for stored_layer, expected_layer in zip(stored._keys_values, expected._keys_values):
            size = retained.size(0)
            assert torch.allclose(
                stored_layer._k_cache._cache[:, :, :size, :],
                expected_layer._k_cache._cache[:, :, :size, :],
            )
            assert torch.allclose(
                stored_layer._v_cache._cache[:, :, :size, :],
                expected_layer._v_cache._cache[:, :, :size, :],
            )

    def test_root_inference_tracks_raw_observation_action_history(self):
        """Online root inference must keep raw histories aligned with their hashed KV entries."""
        torch.manual_seed(8)
        world_model = _build_world_model().eval()
        world_model.rebuild_kv_window_from_tokens = True
        world_model.exact_kv_window_reset = False
        world_model.context_length = 8

        observations = [torch.randn(2, 1, world_model.embed_dim)]
        expected_histories = [observations[0][index].clone() for index in range(2)]
        with torch.no_grad():
            world_model.wm_forward_for_initial_infererence(
                observations[0], [-1, -1], observations[0], [0, 0], [0, 1]
            )
            for step in range(1, 4):
                observations.append(torch.randn(2, 1, world_model.embed_dim))
                actions = [1, 2]
                embedded_actions = world_model.act_embedding_table(torch.tensor(actions))
                for index in range(2):
                    expected_histories[index] = torch.cat((
                        expected_histories[index],
                        embedded_actions[index:index + 1],
                        observations[-1][index],
                    ))
                    if expected_histories[index].size(0) >= world_model.context_length - 1:
                        expected_histories[index] = expected_histories[index][-(world_model.context_length - 3):]

                world_model.wm_forward_for_initial_infererence(
                    observations[-2], actions, observations[-1], [step, step], [0, 1]
                )

        for env_id in range(2):
            cache_key = hash_state(observations[-1][env_id].reshape(-1).numpy())
            cache_index = world_model.past_kv_cache_init_infer_envs[env_id][cache_key]
            stored = world_model.shared_pool_init_infer[env_id][cache_index]
            assert stored.size == expected_histories[env_id].size(0) == 5
            assert torch.equal(
                world_model.past_token_context_init_infer_envs[env_id][cache_key],
                expected_histories[env_id],
            )

    def test_root_inference_remains_exact_across_repeated_window_rollovers(self):
        """Every persisted root cache must equal a fresh forward after multiple window shifts."""
        torch.manual_seed(18)
        world_model = _build_world_model().eval()
        world_model.rebuild_kv_window_from_tokens = True
        world_model.exact_kv_window_reset = False
        world_model.context_length = 8

        observations = [torch.randn(2, 1, world_model.embed_dim)]
        expected_histories = [observations[0][index].clone() for index in range(2)]
        rollover_count = 0

        def assert_persisted_cache_matches_fresh_forward(current_observations):
            for env_id, raw_context in enumerate(expected_histories):
                cache_key = hash_state(current_observations[env_id].reshape(-1).numpy())
                cache_index = world_model.past_kv_cache_init_infer_envs[env_id][cache_key]
                stored = world_model.shared_pool_init_infer[env_id][cache_index]
                expected = world_model.transformer.generate_empty_keys_values(
                    n=1, max_tokens=world_model.context_length
                )
                positioned = raw_context.unsqueeze(0) + world_model._lookup_position_embeddings(
                    torch.arange(raw_context.size(0))
                )
                world_model.transformer(positioned, past_keys_values=expected)

                assert stored.size == raw_context.size(0)
                assert torch.equal(
                    world_model.past_token_context_init_infer_envs[env_id][cache_key],
                    raw_context,
                )
                for stored_layer, expected_layer in zip(
                        stored._keys_values, expected._keys_values
                ):
                    size = raw_context.size(0)
                    assert torch.allclose(
                        stored_layer._k_cache._cache[:, :, :size, :],
                        expected_layer._k_cache._cache[:, :, :size, :],
                        atol=1e-5,
                        rtol=1e-5,
                    )
                    assert torch.allclose(
                        stored_layer._v_cache._cache[:, :, :size, :],
                        expected_layer._v_cache._cache[:, :, :size, :],
                        atol=1e-5,
                        rtol=1e-5,
                    )

        with torch.no_grad():
            world_model.wm_forward_for_initial_infererence(
                observations[0], [-1, -1], observations[0], [0, 0], [0, 1]
            )
            assert_persisted_cache_matches_fresh_forward(observations[0])

            # Eight continuing steps force at least three exact window rebuilds
            # for context_length=8 (raw lengths progress 1 -> 3 -> 5 -> 5 ...).
            for step in range(1, 9):
                observations.append(torch.randn(2, 1, world_model.embed_dim))
                actions = [step % world_model.action_space_size,
                           (step + 1) % world_model.action_space_size]
                embedded_actions = world_model.act_embedding_table(torch.tensor(actions))
                for env_id in range(2):
                    untrimmed = torch.cat((
                        expected_histories[env_id],
                        embedded_actions[env_id:env_id + 1],
                        observations[-1][env_id],
                    ))
                    if untrimmed.size(0) >= world_model.context_length - 1:
                        rollover_count += 1
                        untrimmed = untrimmed[-(world_model.context_length - 3):]
                    expected_histories[env_id] = untrimmed

                world_model.wm_forward_for_initial_infererence(
                    observations[-2], actions, observations[-1], [step, step], [0, 1]
                )
                assert_persisted_cache_matches_fresh_forward(observations[-1])

        assert rollover_count >= 6

    def test_multitask_raw_window_rebuild_uses_the_shared_exact_semantics(self):
        """The MT cache writer must replay retained raw tokens just like the single-task path."""
        torch.manual_seed(9)
        world_model = _build_world_model().eval()
        world_model.rebuild_kv_window_from_tokens = True
        world_model.exact_kv_window_reset = False
        world_model.keys_values_wm = world_model.transformer.generate_empty_keys_values(
            n=1, max_tokens=world_model.context_length
        )
        full_size = world_model.context_length - 1
        for layer in world_model.keys_values_wm._keys_values:
            layer._k_cache._cache.normal_()
            layer._v_cache._cache.normal_()
            layer._k_cache._size = full_size
            layer._v_cache._size = full_size

        raw_history = torch.randn(full_size, world_model.embed_dim)
        retained = raw_history[-(world_model.context_length - 3):]
        world_model.keys_values_wm_token_context_list = [raw_history]
        latent_state = raw_history[-1:].unsqueeze(0)
        expected = world_model.transformer.generate_empty_keys_values(
            n=1, max_tokens=world_model.context_length
        )
        world_model.transformer(
            retained.unsqueeze(0) + world_model.pos_emb(torch.arange(retained.size(0))),
            past_keys_values=expected,
        )

        WorldModelMT._update_cache_context_exact(
            world_model,
            latent_state,
            is_init_infer=True,
            valid_context_lengths=[full_size],
        )

        cache_key = hash_state(latent_state[0].reshape(-1).numpy())
        cache_index = world_model.past_kv_cache_init_infer_envs[0][cache_key]
        stored = world_model.shared_pool_init_infer[0][cache_index]
        assert stored.size == retained.size(0)
        assert torch.equal(world_model.past_token_context_init_infer_envs[0][cache_key], retained)
        for stored_layer, expected_layer in zip(stored._keys_values, expected._keys_values):
            size = retained.size(0)
            assert torch.allclose(
                stored_layer._k_cache._cache[:, :, :size, :],
                expected_layer._k_cache._cache[:, :, :size, :],
            )
            assert torch.allclose(
                stored_layer._v_cache._cache[:, :, :size, :],
                expected_layer._v_cache._cache[:, :, :size, :],
            )

    def test_reanalysis_accepts_explicit_flattened_root_positions(self):
        world_model = _build_world_model().eval()
        positions = [11, 12, 13, 31, 0, 0]

        resolved = [
            world_model._reanalysis_root_start_position(positions, index, len(positions))
            for index in range(len(positions))
        ]

        assert resolved == positions

    def test_legacy_reanalysis_matrix_continues_final_observation_position(self):
        world_model = _build_world_model().eval()
        positions = torch.tensor([[10, 11], [30, 31]])

        resolved = [
            world_model._reanalysis_root_start_position(positions, index, 6)
            for index in range(6)
        ]

        assert resolved == [10, 11, 12, 30, 31, 32]

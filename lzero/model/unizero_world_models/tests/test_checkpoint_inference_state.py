"""Checkpoint loads must refresh weight-dependent inference scratch state."""

import copy
from types import SimpleNamespace

import pytest
import torch

from lzero.policy.unizero import UniZeroPolicy
from .test_per_sample_is_weights import _build_world_model


@pytest.mark.parametrize('route', ['parent', 'collect', 'eval', 'learn'])
def test_recursive_checkpoint_load_refreshes_real_position_projections_and_caches(route):
    torch.manual_seed(901)
    model = torch.nn.Module()
    model.world_model = _build_world_model().eval()
    wm = model.world_model
    checkpoint = copy.deepcopy(model.state_dict())
    # Simulate loading another trained model, not merely reloading identical weights.
    checkpoint['world_model.pos_emb.weight'].mul_(3.)
    checkpoint['world_model.transformer.blocks.0.attn.key.weight'].mul_(2.)
    stale = wm.positional_embedding_k[0].clone()
    with torch.no_grad():
        roots = torch.randn(2, 1, wm.embed_dim)
        wm.wm_forward_for_initial_infererence(roots, [-1, -1], roots, [0, 0], [0, 1])
    assert any(wm.past_kv_cache_init_infer_envs)
    if route == 'parent':
        model.load_state_dict(checkpoint)
    else:
        policy = object.__new__(UniZeroPolicy)
        policy._cfg = SimpleNamespace(model=SimpleNamespace(analysis_sim_norm=False))
        setattr(policy, '_' + route + '_model', model)
        state = {'model': checkpoint}
        if route == 'learn':
            policy.use_adaptive_entropy_weight = False
            policy._target_model = copy.deepcopy(model)
            state['target_model'] = checkpoint
        getattr(policy, '_load_state_dict_' + route)(state)
        if route == 'learn':
            target = policy._target_model.world_model
            assert not any(target.past_kv_cache_init_infer_envs)
            torch.testing.assert_close(target.positional_embedding_k[0], target._get_positional_embedding(0, 'key'))
    expected = wm._get_positional_embedding(0, 'key')
    assert not torch.allclose(stale, expected)
    torch.testing.assert_close(wm.positional_embedding_k[0], expected)
    start, end = 2, wm.context_length - 1
    torch.testing.assert_close(wm.pos_emb_diff_k[0][(start, end)],
                               expected[:, :, :end-start] - expected[:, :, start:end])
    assert not any(wm.past_kv_cache_init_infer_envs)
    assert not wm.past_kv_cache_recurrent_infer


@pytest.mark.parametrize('manager', [False, True])
def test_exact_cache_checkpoint_load_clears_history_without_approximate_rebase(manager):
    from .test_mcts_node_identity import make_model, seed_roots
    wm = make_model(manager=manager, exact=True)
    with torch.no_grad():
        seed_roots(wm)
    assert any(wm.past_token_context_init_infer_envs)
    def unnecessary_rebase():
        raise AssertionError('Exact windows do not use approximate position corrections')
    wm.precompute_pos_emb_diff_kv = unnecessary_rebase
    wm.load_state_dict(wm.state_dict())
    assert not any(wm.past_token_context_init_infer_envs)
    if manager:
        assert not any(pool._key_to_index for pool in wm.kv_cache_manager.init_pools)
    else:
        assert not any(wm.past_kv_cache_init_infer_envs)

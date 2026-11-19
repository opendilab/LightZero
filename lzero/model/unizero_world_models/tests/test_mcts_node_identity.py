"""Actual-model and C++ traversal regressions for history identity in UniZero."""

import numpy as np
import pytest
import torch
from easydict import EasyDict

from lzero.model.unizero_model import UniZeroModel
from lzero.model.unizero_world_models.world_model import WorldModel
from .test_per_sample_is_weights import _build_world_model


def make_model(manager=False, exact=True, simulations=5):
    torch.manual_seed(19)
    template = _build_world_model()
    cfg = EasyDict(dict(template.config, use_new_cache_manager=manager,
                        mcts_cache_by_node=True, rebuild_kv_window_from_tokens=exact,
                        num_simulations=simulations))
    wm = WorldModel(cfg, template.tokenizer).eval()
    # The production observation head is zero initialized: different histories
    # really do produce identical latents, without mocking inference.
    linears = [layer for layer in wm.head_value.modules() if isinstance(layer, torch.nn.Linear)]
    torch.nn.init.normal_(linears[-1].weight, std=0.05)
    return wm


def seed_roots(wm):
    roots = torch.randn(2, 1, wm.embed_dim)
    wm.wm_forward_for_initial_infererence(roots, [-1, -1], roots, [0, 0], [1, 0])
    wm.latent_state = roots
    return roots


@pytest.mark.parametrize('positions', [[3, 4], np.array([3, 4]), torch.tensor([3, 4]), 3])
def test_cache_miss_accepts_production_python_timesteps(positions):
    wm = make_model(exact=False)
    wm.mcts_cache_by_node = False
    roots = torch.randn(2, 1, wm.embed_dim)
    with torch.no_grad():
        wm.keys_values_wm_list, wm.keys_values_wm_size_list = [], []
        sizes = wm.retrieve_or_generate_kvcache(roots.numpy(), 2, start_pos=positions)
    assert sizes == [1, 1]
    assert wm.hit_count == 0


@pytest.mark.parametrize('initial', [True, False])
def test_ring_eviction_keeps_newer_copy_of_repeated_latent(initial):
    """Wrapping an older duplicate slot must not delete the current mapping."""
    wm = make_model(exact=True)
    wm.mcts_cache_by_node = False
    root = torch.randn(1, 1, wm.embed_dim)
    key = wm._root_cache_key(root[0].numpy())
    with torch.no_grad():
        wm.keys_values_wm = wm.transformer.generate_empty_keys_values(n=1, max_tokens=wm.context_length)
        wm.forward({'obs_embeddings': root}, past_keys_values=wm.keys_values_wm, is_init_infer=True)
        wm.keys_values_wm_size_list_current = [1]
        wm.keys_values_wm_token_context_list = [root[0]]
        for _ in range(2):
            wm.update_cache_context(root, is_init_infer=initial, valid_context_lengths=[1], env_ids=[0])
        mapping = wm.past_kv_cache_init_infer_envs[0] if initial else wm.past_kv_cache_recurrent_infer
        newest_slot = mapping[key]
        if initial:
            wm.shared_pool_index_init_envs[0] = 0
        else:
            wm.shared_pool_index = 0
        replacement = torch.randn_like(root)
        wm.keys_values_wm_token_context_list = [replacement[0]]
        wm.update_cache_context(replacement, is_init_infer=initial, valid_context_lengths=[1], env_ids=[0])
        assert mapping[key] == newest_slot
        contexts = wm.past_token_context_init_infer_envs[0] if initial else wm.past_token_context_recurrent_infer
        torch.testing.assert_close(contexts[key], root[0])


@pytest.mark.parametrize('manager', [False, True])
@pytest.mark.parametrize('exact', [False, True])
def test_equal_latents_keep_distinct_histories_within_and_across_roots(manager, exact):
    wm = make_model(manager, exact)
    with torch.no_grad():
        roots = seed_roots(wm)
        # Repeating whole searches also checks circular-pool metadata is reset:
        # old expansion IDs must never evict new entries from a different slot.
        for _ in range(4):
            wm.begin_mcts_search()
            children, histories = [], []
            for simulation, actions in enumerate((torch.tensor([1, 2]), torch.tensor([3, 4]))):
                wm.latent_state = roots
                out = wm.forward_recurrent_inference(
                    [(roots.numpy(), actions)], simulation, [1, 1], [0, 0], [(0, 0), (0, 1)]
                )
                children.append(out[1].clone())
                histories.append([torch.cat((roots[i], wm.act_embedding_table(actions[i:i+1]),
                                              out[1][i])) for i in range(2)])
            assert torch.equal(children[0], children[1])
            assert torch.equal(children[0][0], children[0][1])
            parents = torch.stack((children[0][0], children[1][1]))
            actions = torch.tensor([2, 3])
            output = wm.forward_recurrent_inference(
                [(parents.numpy(), actions)], 2, [2, 2], [0, 0], [(1, 0), (2, 1)]
            )
            for i, prefix in enumerate((histories[0][0], histories[1][1])):
                raw = torch.cat((prefix, wm.act_embedding_table(actions[i:i+1]), output[1][i]))
                hidden = wm._open_loop_hidden(raw.unsqueeze(0))
                expected = wm.head_value(hidden, num_steps=raw.size(0), prev_steps=0)[:, -1:]
                torch.testing.assert_close(output[4][i:i+1], expected, atol=2e-6, rtol=2e-5)
                if exact:
                    torch.testing.assert_close(
                        wm.past_token_context_recurrent_infer[('mcts_node', 3, i)], raw
                    )
        wm.begin_mcts_search()
        with pytest.raises(RuntimeError, match='Missing MCTS history'):
            wm.forward_recurrent_inference(
                [(parents.numpy(), actions)], 1, [2, 2], [0, 0], [(1, 0), (1, 1)]
            )


@pytest.mark.parametrize('manager', [False, True])
def test_cpp_search_passes_real_parent_node_ids(manager):
    from lzero.mcts.tree_search.mcts_ctree import UniZeroMCTSCtree
    wm = make_model(manager, simulations=16)
    model = UniZeroModel.__new__(UniZeroModel)
    torch.nn.Module.__init__(model)
    model.world_model = wm
    cfg = EasyDict(dict(model=dict(value_support_range=(-50, 51, 1),
                                   reward_support_range=(-50, 51, 1), categorical_distribution=True),
                        device='cpu', discount_factor=.997, num_simulations=16, deterministic=True))
    mcts = UniZeroMCTSCtree(cfg)
    with torch.no_grad():
        roots_latent = seed_roots(wm)
        roots = mcts.roots(2, [list(range(6))] * 2)
        roots.prepare_no_noise([0., 0.], [[0.] * 6] * 2, [-1, -1])
        mcts.search(roots, model, roots_latent.numpy(), [-1, -1], np.zeros(2, dtype=np.int64))
    assert all(sum(distribution) == 16 for distribution in roots.get_distributions())
    assert min(mcts.last_search_depth_mean) > 1  # Exercise real internal-parent traversal.
    keys = (wm.kv_cache_manager.recur_pool._key_to_index if manager
            else wm.past_kv_cache_recurrent_infer)
    assert len(keys) == 32  # All zero child latents still have distinct histories.
    assert all(key[0] == 'mcts_node' for key in keys)

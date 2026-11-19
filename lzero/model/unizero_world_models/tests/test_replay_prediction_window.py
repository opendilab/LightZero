"""Replay targets must match online predictions, including after window rolls."""

import numpy as np
import pytest
import torch
from easydict import EasyDict

from .test_per_sample_is_weights import _build_world_model
from lzero.model.unizero_world_models.world_model import WorldModel


def prediction_model():
    template = _build_world_model()
    return WorldModel(EasyDict(dict(template.config, num_layers=2)), template.tokenizer).eval()


def test_contextual_reanalysis_search_needs_no_prior_online_inference():
    """The target model's first use can be a seeded replay search."""
    from lzero.model.unizero_model import UniZeroModel
    from lzero.mcts.tree_search.mcts_ctree import UniZeroMCTSCtree
    wm = prediction_model()
    wm.rebuild_kv_window_from_tokens = True
    wm.mcts_cache_by_node = True
    wm.config.replay_context_use_prediction_window = True
    wm.reanalyze_phase = True
    model = UniZeroModel.__new__(UniZeroModel)
    torch.nn.Module.__init__(model)
    model.world_model = wm
    cfg = EasyDict(dict(model=dict(value_support_range=(-50, 51, 1),
                                   reward_support_range=(-50, 51, 1), categorical_distribution=True),
                        device='cpu', discount_factor=.997, num_simulations=5, deterministic=True))
    mcts = UniZeroMCTSCtree(cfg)
    roots_latent = torch.randn(2, 1, wm.embed_dim)
    with torch.no_grad():
        priors = wm.seed_reanalysis_root_caches(roots_latent.numpy(), [roots_latent[i] for i in range(2)])
        assert not hasattr(wm, 'latent_state')
        roots = mcts.roots(2, [list(range(6))] * 2)
        roots.prepare_no_noise([0.] * 2, priors, [-1] * 2)
        mcts.search(roots, model, roots_latent.numpy(), [-1] * 2, [0, 0])
    assert all(sum(counts) == 5 for counts in roots.get_distributions())
    assert wm.reanalysis_root_seed_hit_count == 2


@pytest.mark.parametrize('context_length', [8, 10])
def test_bootstrap_values_and_reanalysis_priors_match_online_rollovers(context_length):
    torch.manual_seed(42)
    wm = prediction_model()
    wm.rebuild_kv_window_from_tokens = True
    wm.context_length = context_length
    wm.config.replay_context_use_prediction_window = True
    # Nonconstant heads are essential: zero-initialized heads hide context bugs.
    for head in (wm.head_value, wm.head_policy):
        last_linear = [m for m in head.modules() if isinstance(m, torch.nn.Linear)][-1]
        torch.nn.init.normal_(last_linear.weight, std=.1)
    observations, actions = [], []
    with torch.no_grad():
        for step in range(10):
            current = torch.randn(2, 1, wm.embed_dim)
            previous = observations[-1] if observations else current
            action = torch.tensor([step % 6, (step + 1) % 6]) if step else [-1, -1]
            if step:
                actions.append(action)
            online = wm.wm_forward_for_initial_infererence(previous, action, current, [step] * 2, [0, 1])
            contexts = wm.build_reanalysis_root_token_contexts(
                current.numpy(), np.empty((2, 0), dtype=np.int64), 1,
                [[obs[i] for obs in observations] for i in range(2)],
                [[a[i] for a in actions] for i in range(2)],
            )
            assert [len(c) for c in contexts] == [min(2 * step + 1, context_length - 1)] * 2
            values = wm.evaluate_root_token_context_values(contexts)
            torch.testing.assert_close(values, online.logits_value[:, -1], atol=2e-5, rtol=2e-5)
            wm.reanalyze_phase = True
            priors = wm.seed_reanalysis_root_caches(current.numpy(), contexts)
            wm.reanalyze_phase = False
            torch.testing.assert_close(torch.tensor(priors), online.logits_policy[:, -1], atol=2e-5, rtol=2e-5)
            for i in range(2):
                key = wm._root_cache_key(current[i].numpy())
                stored = wm.past_token_context_init_infer_envs[i][key]
                assert len(stored) == min(2 * step + 1, context_length - 3)
            observations.append(current)

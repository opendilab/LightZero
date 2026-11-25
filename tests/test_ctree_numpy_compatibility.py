"""Native extension checks, deliberately independent of DI-engine's NumPy < 2 dependency."""
import importlib.util
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path

import numpy as np
import pytest


CTREE = Path(__file__).resolve().parents[1] / 'lzero' / 'mcts' / 'ctree'
EXTENSIONS = [
    ('ctree_muzero', 'mz_tree'),
    ('ctree_efficientzero', 'ez_tree'),
    ('ctree_sampled_muzero', 'smz_tree'),
    ('ctree_sampled_efficientzero', 'ezs_tree'),
    ('ctree_gumbel_muzero', 'gmz_tree'),
    ('ctree_stochastic_muzero', 'stochastic_mz_tree'),
]


def _load_extension(directory, name):
    # Loading the binary directly avoids lzero.mcts.__init__, which imports the
    # training stack. Missing binaries must fail this compatibility gate.
    path = next((CTREE / directory / (name + suffix) for suffix in EXTENSION_SUFFIXES
                 if (CTREE / directory / (name + suffix)).is_file()), None)
    assert path is not None, 'Build extensions first: python setup.py build_ext --inplace'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('directory,name', EXTENSIONS)
def test_native_extension_import(directory, name):
    module = _load_extension(directory, name)
    assert module.Roots is not None
    assert module.ResultsWrapper is not None


def test_muzero_search_with_numpy_inputs():
    tree = _load_extension('ctree_muzero', 'mz_tree')
    legal_actions = np.array([[0, 2], [1, 2]], dtype=np.int64)
    logits = np.zeros((2, 3), dtype=np.float32)
    rewards = np.zeros(2, dtype=np.float32)
    values = np.ones(2, dtype=np.float32)
    to_play = np.full(2, -1, dtype=np.int32)
    roots = tree.Roots(2, legal_actions.tolist())
    roots.prepare_no_noise(rewards.tolist(), logits.tolist(), to_play.tolist())
    stats = tree.MinMaxStatsList(2)
    stats.set_delta(0.01)

    for step in range(8):
        results = tree.ResultsWrapper(2)
        _, _, actions, virtual_to_play = tree.batch_traverse(
            roots, 19652, 1.25, 0.997, stats, results, to_play.tolist(), deterministic=True,
        )
        assert np.asarray(actions).shape == (2,)
        assert np.all((np.asarray(actions) >= 0) & (np.asarray(actions) < 3))
        if step == 0:
            assert actions == [0, 1]
        tree.batch_backpropagate(
            step + 1, 0.997, rewards.tolist(), values.tolist(), logits.tolist(),
            stats, results, virtual_to_play,
        )

    visits = np.asarray(roots.get_distributions())
    assert visits.shape == (2, 2)
    np.testing.assert_array_equal(visits.sum(axis=1), [8, 8])
    root_values = np.asarray(roots.get_values())
    assert root_values.shape == (2,)
    assert np.all(np.isfinite(root_values))
    assert np.all(root_values > 0)

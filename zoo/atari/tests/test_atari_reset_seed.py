import numpy as np

from zoo.atari.envs.atari_lightzero_env import _prepare_reset_seed


def test_static_eval_seed_replays_legacy_noop_rng_sequence():
    _prepare_reset_seed(base_seed=7, dynamic_seed=False)
    first_noop_count = np.random.randint(1, 31)

    _prepare_reset_seed(base_seed=7, dynamic_seed=False)
    second_noop_count = np.random.randint(1, 31)

    assert first_noop_count == second_noop_count


def test_dynamic_collect_seed_changes_but_is_reproducible():
    np.random.seed(123)
    first_sequence = [_prepare_reset_seed(base_seed=7, dynamic_seed=True) for _ in range(3)]

    np.random.seed(123)
    second_sequence = [_prepare_reset_seed(base_seed=7, dynamic_seed=True) for _ in range(3)]

    assert first_sequence == second_sequence
    assert len(set(first_sequence)) > 1

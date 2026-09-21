import numpy as np

from zoo.atari.envs.atari_lightzero_env import _prepare_reset_seed, _reset_with_numpy_seed


def test_static_eval_seed_replays_legacy_noop_rng_sequence():
    class _ResetSampler:

        def reset(self):
            return np.random.randint(1, 31)

    first_noop_count = _reset_with_numpy_seed(_ResetSampler(), reset_seed=7)
    second_noop_count = _reset_with_numpy_seed(_ResetSampler(), reset_seed=7)

    assert first_noop_count == second_noop_count


def test_dynamic_collect_seed_changes_but_is_reproducible():
    first_rng = np.random.RandomState(123)
    second_rng = np.random.RandomState(123)
    first_sequence = [
        _prepare_reset_seed(base_seed=7, dynamic_seed=True, rng=first_rng) for _ in range(3)
    ]
    second_sequence = [
        _prepare_reset_seed(base_seed=7, dynamic_seed=True, rng=second_rng) for _ in range(3)
    ]

    assert first_sequence == second_sequence
    assert len(set(first_sequence)) > 1


def test_legacy_noop_seed_scope_restores_process_rng_state():
    class _ResetSampler:

        def reset(self):
            np.random.randint(1, 31)
            return None

    np.random.seed(321)
    expected_next = np.random.randint(0, 1_000_000)
    np.random.seed(321)
    _reset_with_numpy_seed(_ResetSampler(), reset_seed=7)
    actual_next = np.random.randint(0, 1_000_000)

    assert actual_next == expected_next

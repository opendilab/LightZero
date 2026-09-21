import numpy as np
import pytest

from lzero.worker.muzero_evaluator import balanced_episode_targets


@pytest.mark.parametrize(
    'env_num,n_episode,expected',
    [
        (3, 3, [1, 1, 1]),
        (3, 5, [2, 2, 1]),
        (3, 8, [3, 3, 2]),
    ],
)
def test_balanced_episode_targets_match_vector_eval_monitor(env_num, n_episode, expected):
    targets = balanced_episode_targets(env_num, n_episode)

    np.testing.assert_array_equal(targets, expected)
    assert targets.sum() == n_episode


def test_balanced_episode_targets_reject_too_few_episodes():
    with pytest.raises(ValueError, match='at least env_num'):
        balanced_episode_targets(env_num=3, n_episode=2)

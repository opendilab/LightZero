from types import SimpleNamespace

import pytest

from lzero.worker.muzero_segment_collector import MuZeroSegmentCollector


@pytest.mark.unittest
def test_stash_inflight_segments_preserves_every_nonempty_env():
    collector = object.__new__(MuZeroSegmentCollector)
    collector._end_flag = True
    collector.last_game_segments = [None, None, None]
    collector.last_game_priorities = [None, None, None]
    collector.dones = [False, False, False]
    collector._compute_priorities = lambda env_id, *_: f'priority-{env_id}'

    segments = [
        SimpleNamespace(action_segment=[1, 2, 3]),
        SimpleNamespace(action_segment=[]),
        SimpleNamespace(action_segment=[4, 5]),
    ]
    pred_values = [[1., 2., 3.], [], [4., 5.]]
    search_values = [[1., 2., 3.], [], [4., 5.]]

    count = collector._stash_inflight_segments(segments, pred_values, search_values)

    assert count == 5
    assert collector.last_game_segments == [segments[0], None, segments[2]]
    assert collector.last_game_priorities == ['priority-0', None, 'priority-2']
    assert pred_values == [[], [], []]
    assert search_values == [[], [], []]

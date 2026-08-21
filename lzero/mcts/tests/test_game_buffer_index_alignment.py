"""
Unit test for buffer index/weight alignment in ``GameBuffer._sample_orig_data``.

Two invariants are covered:

1. Index alignment: the position resampling inside ``_sample_orig_data`` (e.g. when the last
   position of a short done segment is drawn) moves a sample to a different transition within the
   same game segment. The returned buffer indices must point at the positions actually used, so
   that the IS weights and the later ``update_priority`` refer to the trained transition.
2. Live priority updates: ``make_time`` must carry the real sampling time, because
   ``update_priority`` only writes priorities for samples newer than the last buffer reset
   (``clear_time``). Placeholder zeros would make that guard unconditionally False and silently
   drop all priority updates.
"""
from types import SimpleNamespace
import time

import numpy as np
import pytest
from easydict import EasyDict

from lzero.mcts.buffer.game_buffer_muzero import MuZeroGameBuffer


def _make_buffer() -> MuZeroGameBuffer:
    cfg = EasyDict(dict(
        env_type='not_board_games',
        action_type='fixed_action_space',
        replay_buffer_size=1000,
        batch_size=4,
        priority_prob_alpha=0.6,
        priority_prob_beta=0.4,
        use_priority=True,
        reanalyze_outdated=False,
        game_segment_length=20,
        num_unroll_steps=5,
        td_steps=10,
        model=dict(
            action_space_size=6,
            value_support_range=(-300, 301, 1),
            reward_support_range=(-300, 301, 1),
        ),
    ))
    return MuZeroGameBuffer(cfg)


@pytest.mark.unittest
class TestSampleOrigDataIndexAlignment:

    def test_partial_padded_segment_uses_real_transition_count_for_priorities(self):
        buf = _make_buffer()
        seg = SimpleNamespace(
            action_segment=list(range(20)),
            valid_transition_count=5,
        )
        meta = {
            'done': False,
            'unroll_plus_td_steps': 15,
            'priorities': np.arange(5, dtype=np.float32) + 1,
        }

        buf.push_game_segments(([seg], [meta]))

        assert buf.get_num_of_transitions() == 5
        assert len(buf.game_pos_priorities) == 5
        assert len(buf.game_segment_game_pos_look_up) == 5
        assert np.all(buf.game_pos_priorities == 0)

    def test_index_and_weight_alignment_after_position_resampling(self):
        buf = _make_buffer()
        # One short done segment with 3 transitions and non-uniform priorities.
        seg = SimpleNamespace(action_segment=[1, 2, 3])
        meta = {'done': True, 'unroll_plus_td_steps': 15, 'priorities': np.array([3.0, 2.0, 1.0])}
        buf.push_game_segments(([seg], [meta]))
        assert buf.get_num_of_transitions() == 3
        assert buf.game_segment_game_pos_look_up == [(0, 0), (0, 1), (0, 2)]

        num_transitions = buf.get_num_of_transitions()
        probs = buf.game_pos_priorities ** buf._alpha + 1e-6
        probs /= probs.sum()

        # batch_size == 3 with replace=False draws every index, so the pos=2 entry (last position
        # of the short segment) is always drawn and always resampled into {0, 1}.
        for _ in range(20):
            game_segment_list, pos_list, index_list, weights_list, make_time = buf._sample_orig_data(3)
            assert len(pos_list) == 3
            # IS weights are max-normalized within the batch
            raw = (num_transitions * probs[np.asarray([int(i) for i in index_list])]) ** (-buf._beta)
            expected_weights = raw / raw.max()
            for i in range(3):
                seg_idx, look_up_pos = buf.game_segment_game_pos_look_up[int(index_list[i])]
                # the returned flat index points at the (segment, pos) actually used
                assert seg_idx - buf.base_idx == 0
                assert look_up_pos == pos_list[i], \
                    f'index {index_list[i]} maps to pos {look_up_pos} but used pos is {pos_list[i]}'
                # IS weights come from the adjusted positions' probabilities
                assert abs(weights_list[i] - expected_weights[i]) < 1e-8
                # the resampled last position can never remain pos=2
                assert pos_list[i] != 2

    def test_make_time_and_update_priority_roundtrip(self):
        buf = _make_buffer()
        seg = SimpleNamespace(action_segment=[1, 2, 3])
        meta = {'done': True, 'unroll_plus_td_steps': 15, 'priorities': np.array([3.0, 2.0, 1.0])}
        buf.push_game_segments(([seg], [meta]))

        _, _, index_list, _, make_time = buf._sample_orig_data(3)
        # make_time must be real (nonzero) timestamps, otherwise update_priority drops everything
        assert all(t > 0 for t in make_time)

        # update_priority writes the new priorities to the adjusted positions (last write wins
        # when two samples share an adjusted index after resampling)
        new_prios = np.array([0.5, 0.6, 0.7])
        expected = buf.game_pos_priorities.copy()
        for i in range(3):
            expected[int(index_list[i])] = new_prios[i]
        train_data = [[None, None, None, np.asarray(index_list), None, np.asarray(make_time)], None]
        buf.update_priority(train_data, new_prios)
        assert np.allclose(buf.game_pos_priorities, expected)

        # samples with make_time older than a later buffer clear must be dropped
        buf.clear_time = time.time()  # simulate a buffer clear happening after the sample was made
        _, _, index_list2, _, make_time2 = buf._sample_orig_data(3)
        stale_make_time = [0.0 for _ in make_time2]  # simulates a sample made before the reset
        prios_before = buf.game_pos_priorities.copy()
        train_data_stale = [[None, None, None, np.asarray(index_list2), None, np.asarray(stale_make_time)], None]
        buf.update_priority(train_data_stale, np.array([9.9, 9.9, 9.9]))
        assert np.allclose(buf.game_pos_priorities, prios_before)

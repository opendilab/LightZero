from types import SimpleNamespace

import pytest

from lzero.entry.train_muzero_segment import (
    _prune_periodic_checkpoints,
    _required_replay_transitions,
    _restore_resume_counters,
)


class _Counter:
    def __init__(self):
        self.value = 0

    def update(self, value):
        self.value = value


def test_restore_resume_counters_updates_both_checkpoint_owners():
    learner = SimpleNamespace(_last_iter=_Counter(), collector_envstep=0)
    collector = SimpleNamespace(_total_envstep_count=0)
    _restore_resume_counters(learner, collector, train_iter=10079, envstep=43464)
    assert learner._last_iter.value == 10079
    assert learner.collector_envstep == 43464
    assert collector._total_envstep_count == 43464


def test_resume_replay_warmup_preserves_fresh_run_semantics():
    assert _required_replay_transitions(0, 256, 10000) == 257
    assert _required_replay_transitions(10079, 256, 10000) == 10000
    with pytest.raises(ValueError, match='non-negative'):
        _required_replay_transitions(10079, 256, -1)


def test_checkpoint_retention_keeps_zero_best_and_newest(tmp_path):
    checkpoint_dir = tmp_path / 'ckpt'
    checkpoint_dir.mkdir()
    for name in (
        'iteration_0.pth.tar', 'iteration_1000.pth.tar', 'iteration_2000.pth.tar',
        'iteration_3000.pth.tar', 'ckpt_best.pth.tar',
    ):
        (checkpoint_dir / name).touch()
    removed = _prune_periodic_checkpoints(str(tmp_path), keep_last=2)
    assert removed == [str(checkpoint_dir / 'iteration_1000.pth.tar')]
    assert (checkpoint_dir / 'iteration_0.pth.tar').exists()
    assert (checkpoint_dir / 'iteration_2000.pth.tar').exists()
    assert (checkpoint_dir / 'iteration_3000.pth.tar').exists()
    assert (checkpoint_dir / 'ckpt_best.pth.tar').exists()

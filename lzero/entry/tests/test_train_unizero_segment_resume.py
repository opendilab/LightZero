import pytest

from lzero.entry.train_unizero_segment import _required_replay_transitions, _restore_resume_counters


class _CountVar:

    def __init__(self):
        self.value = 0

    def update(self, value):
        self.value = value


class _Learner:

    def __init__(self):
        self._last_iter = _CountVar()
        self.collector_envstep = 0


class _Collector:

    def __init__(self):
        self._total_envstep_count = 0


def test_restore_resume_counters_updates_collector_and_checkpoint_owner():
    learner = _Learner()
    collector = _Collector()

    _restore_resume_counters(learner, collector, train_iter=10023, envstep=101278)

    assert learner._last_iter.value == 10023
    assert collector._total_envstep_count == 101278
    assert learner.collector_envstep == 101278


def test_restore_resume_counters_keeps_fresh_run_at_zero():
    learner = _Learner()
    collector = _Collector()

    _restore_resume_counters(learner, collector, train_iter=0, envstep=0)

    assert learner._last_iter.value == 0
    assert collector._total_envstep_count == 0
    assert learner.collector_envstep == 0


def test_resume_requires_replay_warmup_but_fresh_run_keeps_one_batch_threshold():
    assert _required_replay_transitions(0, batch_size=256, resume_buffer_min_transitions=10000) == 257
    assert _required_replay_transitions(10023, batch_size=256, resume_buffer_min_transitions=10000) == 10000
    assert _required_replay_transitions(10023, batch_size=256, resume_buffer_min_transitions=0) == 257


def test_resume_replay_warmup_rejects_invalid_configuration():
    with pytest.raises(ValueError, match='non-negative'):
        _required_replay_transitions(10, batch_size=256, resume_buffer_min_transitions=-1)

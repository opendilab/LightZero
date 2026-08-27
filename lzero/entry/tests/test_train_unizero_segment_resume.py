from pathlib import Path
from types import SimpleNamespace

import pytest

from lzero.entry.train_unizero_segment import (
    _make_checkpoint_errors_nonfatal,
    _prune_periodic_checkpoints,
    _required_replay_transitions,
    _resolve_segment_reanalyze_settings,
    _restore_resume_counters,
    _should_evaluate_at_train_iter,
)


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


class _Evaluator:

    def __init__(self, due_iters):
        self.due_iters = set(due_iters)
        self.calls = []

    def should_eval(self, train_iter):
        self.calls.append(train_iter)
        return train_iter in self.due_iters


class _CheckpointHook:

    def __init__(self, name, error=None):
        self.name = name
        self.error = error
        self.calls = 0

    def __call__(self, learner):
        self.calls += 1
        if self.error is not None:
            raise self.error


class _HookLearner:

    def __init__(self, hooks):
        self._hooks = hooks
        self.train_iter = 123
        self.collector_envstep = 456
        self.ckpt_name = 'iteration_123.pth.tar'


def test_restore_resume_counters_updates_collector_and_checkpoint_owner():
    learner = _Learner()
    collector = _Collector()

    _restore_resume_counters(learner, collector, train_iter=10023, envstep=101278)

    assert learner._last_iter.value == 10023
    assert collector._total_envstep_count == 101278
    assert learner.collector_envstep == 101278


@pytest.mark.parametrize(
    'error',
    [OSError('disk quota exceeded'), RuntimeError('PytorchStreamWriter failed writing file')],
)
def test_nonfatal_checkpoint_hooks_ignore_only_checkpoint_write_errors(error):
    save_hook = _CheckpointHook('save_ckpt_after_iter', error)
    unrelated_hook = _CheckpointHook('log_show')
    learner = _HookLearner({'after_iter': [save_hook, unrelated_hook]})

    assert _make_checkpoint_errors_nonfatal(learner) == 1
    for hook in learner._hooks['after_iter']:
        hook(learner)

    assert save_hook.calls == 1
    assert unrelated_hook.calls == 1


def test_nonfatal_checkpoint_hooks_do_not_hide_unexpected_errors():
    save_hook = _CheckpointHook('save_ckpt_after_run', ValueError('bad checkpoint state'))
    learner = _HookLearner({'after_run': [save_hook]})
    _make_checkpoint_errors_nonfatal(learner)

    with pytest.raises(ValueError, match='bad checkpoint state'):
        learner._hooks['after_run'][0](learner)


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


def test_initial_evaluation_is_not_repeated_during_replay_warmup():
    evaluator = _Evaluator(due_iters={5000})

    assert _should_evaluate_at_train_iter(0, None, evaluator)
    assert not _should_evaluate_at_train_iter(0, 0, evaluator)
    assert evaluator.calls == []

    assert _should_evaluate_at_train_iter(5000, 0, evaluator)
    assert evaluator.calls == [5000]


def test_segment_reanalyze_settings_support_minimal_and_explicit_configs():
    assert _resolve_segment_reanalyze_settings(SimpleNamespace()) == (
        1 / 100000,
        160,
        0.75,
    )
    assert _resolve_segment_reanalyze_settings(
        SimpleNamespace(
            buffer_reanalyze_freq=0.02,
            reanalyze_batch_size=32,
            reanalyze_partition=0.5,
        )
    ) == (0.02, 32, 0.5)


@pytest.mark.parametrize(
    ('config', 'message'),
    [
        (SimpleNamespace(buffer_reanalyze_freq=0), 'buffer_reanalyze_freq'),
        (SimpleNamespace(reanalyze_batch_size=0), 'reanalyze_batch_size'),
        (SimpleNamespace(reanalyze_partition=0), 'reanalyze_partition'),
        (SimpleNamespace(reanalyze_partition=1.1), 'reanalyze_partition'),
    ],
)
def test_segment_reanalyze_settings_reject_invalid_values(config, message):
    with pytest.raises(ValueError, match=message):
        _resolve_segment_reanalyze_settings(config)


def test_periodic_checkpoint_retention_keeps_initial_latest_and_best(tmp_path):
    checkpoint_dir = tmp_path / 'run' / 'ckpt'
    checkpoint_dir.mkdir(parents=True)
    for name in (
        'iteration_0.pth.tar',
        'iteration_20000.pth.tar',
        'iteration_40000.pth.tar',
        'iteration_60000.pth.tar',
        'ckpt_best.pth.tar',
        'iteration_invalid.pth.tar',
    ):
        (checkpoint_dir / name).write_bytes(b'checkpoint')

    removed = _prune_periodic_checkpoints(str(tmp_path / 'run'), keep_last=2)

    assert removed == [str(checkpoint_dir / 'iteration_20000.pth.tar')]
    assert sorted(path.name for path in checkpoint_dir.iterdir()) == [
        'ckpt_best.pth.tar',
        'iteration_0.pth.tar',
        'iteration_40000.pth.tar',
        'iteration_60000.pth.tar',
        'iteration_invalid.pth.tar',
    ]


def test_periodic_checkpoint_retention_disabled_and_rejects_negative(tmp_path):
    assert _prune_periodic_checkpoints(str(tmp_path / 'missing'), keep_last=0) == []
    with pytest.raises(ValueError, match='non-negative'):
        _prune_periodic_checkpoints(str(tmp_path / 'missing'), keep_last=-1)


def test_periodic_checkpoint_retention_does_not_crash_training_on_unlink_failure(
        tmp_path, monkeypatch
):
    checkpoint_dir = tmp_path / 'run' / 'ckpt'
    checkpoint_dir.mkdir(parents=True)
    stale = checkpoint_dir / 'iteration_20000.pth.tar'
    latest = checkpoint_dir / 'iteration_40000.pth.tar'
    stale.write_bytes(b'stale')
    latest.write_bytes(b'latest')
    original_unlink = Path.unlink

    def fail_stale_unlink(path, *args, **kwargs):
        if path == stale:
            raise OSError('synthetic shared-filesystem failure')
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'unlink', fail_stale_unlink)

    assert _prune_periodic_checkpoints(str(tmp_path / 'run'), keep_last=1) == []
    assert stale.exists()
    assert latest.exists()

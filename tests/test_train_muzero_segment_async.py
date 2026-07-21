import importlib.util
import sys
from pathlib import Path

import pytest
from easydict import EasyDict


def _load_async_entry_module():
    module_path = Path(__file__).resolve().parents[1] / 'lzero' / 'entry' / 'train_muzero_segment_async.py'
    spec = importlib.util.spec_from_file_location('train_muzero_segment_async_under_test', module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_async_cfg_defaults() -> None:
    module = _load_async_entry_module()
    cfg = EasyDict(policy=EasyDict(async_pipeline=EasyDict(enabled=True)))
    async_cfg = module._get_async_cfg(cfg)

    assert async_cfg.num_collector_actors == 1
    assert async_cfg.max_collect_inflight == 1
    assert async_cfg.max_eval_inflight == 1
    assert async_cfg.weight_sync_interval == 1


def test_should_eval_matches_serial_evaluator_gate() -> None:
    module = _load_async_entry_module()
    assert module._should_eval(train_iter=0, last_eval_iter=0, eval_freq=5) is False
    assert module._should_eval(train_iter=4, last_eval_iter=0, eval_freq=5) is False
    assert module._should_eval(train_iter=5, last_eval_iter=0, eval_freq=5) is True
    assert module._should_eval(train_iter=5, last_eval_iter=5, eval_freq=5) is False
    assert module._should_eval(train_iter=10, last_eval_iter=5, eval_freq=5) is True


def test_should_publish_model_state_respects_policy_lag() -> None:
    module = _load_async_entry_module()

    assert module._should_publish_model_state(
        current_version=0,
        last_published_version=-1,
        has_published_state=False,
        weight_sync_interval=10,
        max_policy_lag=0,
    ) is True
    assert module._should_publish_model_state(
        current_version=1,
        last_published_version=0,
        has_published_state=True,
        weight_sync_interval=10,
        max_policy_lag=0,
    ) is True
    assert module._should_publish_model_state(
        current_version=1,
        last_published_version=0,
        has_published_state=True,
        weight_sync_interval=10,
        max_policy_lag=2,
    ) is False
    assert module._should_publish_model_state(
        current_version=3,
        last_published_version=0,
        has_published_state=True,
        weight_sync_interval=10,
        max_policy_lag=2,
    ) is True
    assert module._should_publish_model_state(
        current_version=9,
        last_published_version=0,
        has_published_state=True,
        weight_sync_interval=10,
        max_policy_lag=-1,
    ) is False
    assert module._should_publish_model_state(
        current_version=10,
        last_published_version=0,
        has_published_state=True,
        weight_sync_interval=10,
        max_policy_lag=-1,
    ) is True


def test_reanalyze_interval_is_never_zero() -> None:
    module = _load_async_entry_module()

    assert module._reanalyze_interval(update_per_collect=1, buffer_reanalyze_freq=10) == 1
    assert module._reanalyze_interval(update_per_collect=25, buffer_reanalyze_freq=10) == 2
    assert module._reanalyze_interval(update_per_collect=25, buffer_reanalyze_freq=0) is None


def test_unizero_train_data_gets_train_iter() -> None:
    module = _load_async_entry_module()

    train_data = ['current_batch', 'target_batch']
    result = module._prepare_train_data_for_policy(train_data, 'unizero', 123)
    assert result == ['current_batch', 'target_batch', 123]

    train_data = ['current_batch', 'target_batch']
    result = module._prepare_train_data_for_policy(train_data, 'muzero', 123)
    assert result == ['current_batch', 'target_batch']


def test_has_enough_replay_data_respects_sample_type() -> None:
    module = _load_async_entry_module()

    class Buffer:
        def get_num_of_game_segments(self):
            return 3

        def get_num_of_transitions(self):
            return 5

    cfg = EasyDict(policy=EasyDict(sample_type='episode'))
    assert module._has_enough_replay_data(cfg, Buffer(), 2) is True
    assert module._has_enough_replay_data(cfg, Buffer(), 3) is False

    cfg = EasyDict(policy=EasyDict(sample_type='transition'))
    assert module._has_enough_replay_data(cfg, Buffer(), 4) is True
    assert module._has_enough_replay_data(cfg, Buffer(), 5) is False


def test_async_entry_requires_ray_when_ray_is_missing() -> None:
    if importlib.util.find_spec('ray') is not None:
        pytest.skip('ray is installed in this environment')

    module = _load_async_entry_module()
    cfg = EasyDict(policy=EasyDict(cuda=False, eval_offline=False, random_collect_episode_num=0))
    create_cfg = EasyDict(policy=EasyDict(type='muzero'))

    with pytest.raises(RuntimeError, match='Ray is required'):
        module.train_muzero_segment_async([cfg, create_cfg], seed=0, max_train_iter=1, max_env_step=1)

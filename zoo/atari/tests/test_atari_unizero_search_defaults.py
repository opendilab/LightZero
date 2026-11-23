from contextlib import nullcontext
from importlib import import_module

import pytest

from zoo.atari.config import atari_unizero_config, atari_unizero_segment_experimental_config as experimental


def assert_full_context(config):
    policy = config.policy
    world = policy.model.world_model_cfg
    assert world.context_length == world.max_tokens == 2 * policy.num_unroll_steps
    assert world.mcts_cache_by_node is True
    assert world.rebuild_kv_window_from_tokens is True
    assert world.replay_context_use_prediction_window is True
    assert policy.eval_mcts_deterministic is False
    if config.env.env_id in ('ALE/Pong-v5', 'PongNoFrameskip-v4'):
        assert config.env.stop_value == 19


@pytest.mark.parametrize('name', ['atari_unizero_config', 'atari_unizero_segment_config',
                                'atari_unizero_ddp_config', 'atari_unizero_stack4_config'])
def test_single_task_entry_search_settings(name):
    module = import_module('zoo.atari.config.' + name)
    config = module.main_config if hasattr(module, 'main_config') else module.build_config('ALE/MsPacman-v5')[0]
    assert_full_context(config)


def test_stack4_builds_independent_config_with_single_frame_env_space():
    from zoo.atari.config.atari_unizero_stack4_config import main_config
    base, _, _ = atari_unizero_config.build_config()
    assert base.env.observation_shape == (3, 64, 64)
    assert base.policy.model.observation_shape == (3, 64, 64)
    assert main_config.env.observation_shape == (4, 64, 64)
    assert main_config.env.frame_stack_num == main_config.policy.model.frame_stack_num == 4


def test_segment_ddp_preserves_distributed_training_recipe(monkeypatch):
    import ding.utils
    import lzero.config.utils
    import lzero.entry
    from zoo.atari.config import atari_unizero_segment_ddp_config

    captured = {}
    monkeypatch.setattr(ding.utils, 'DDPContext', nullcontext)
    monkeypatch.setattr(lzero.config.utils, 'lz_to_ddp_config', lambda config: config)
    monkeypatch.setattr(lzero.entry, 'train_unizero_segment', lambda pair, **kw: captured.update(config=pair[0]))
    atari_unizero_segment_ddp_config.main('ALE/Pong-v5', 0)
    config = captured['config']
    assert_full_context(config)
    assert config.policy.multi_gpu is True
    assert config.policy.batch_size == 128
    assert config.policy.replay_ratio == .25


@pytest.mark.parametrize('smoke', [False, True])
def test_async_launcher_adapts_search_capacity_to_smoke_horizon(monkeypatch, tmp_path, smoke):
    import lzero.entry
    from zoo.atari.config import atari_unizero_segment_async_config

    captured = {}
    monkeypatch.setattr(lzero.entry, 'train_unizero_segment', lambda pair, **kw: captured.update(config=pair[0]))
    atari_unizero_segment_async_config.main('ALE/Pong-v5', 0, exp_root=str(tmp_path), smoke_test=smoke)
    config = captured['config']
    assert_full_context(config)
    assert config.policy.num_unroll_steps == (2 if smoke else 10)


@pytest.mark.parametrize('kwargs, context, rebuild, node', [
    ({}, 20, True, True),
    ({'num_unroll_steps_override': 4}, 8, True, True),
    ({'infer_context_length_override': 3}, 6, True, True),
    ({'infer_context_length_override': 4, 'rebuild_kv_window_from_tokens': False,
      'mcts_cache_by_node': False}, 8, False, False),
    ({'exact_kv_window_reset': True}, 20, False, True),
])
def test_experimental_defaults_and_explicit_ablation_controls(monkeypatch, tmp_path, kwargs, context, rebuild, node):
    import lzero.entry

    captured = {}
    monkeypatch.setattr(lzero.entry, 'train_unizero_segment', lambda pair, **kw: captured.update(config=pair[0]))
    experimental.main('ALE/Pong-v5', 0, output_root=str(tmp_path), **kwargs)
    config = captured['config']
    world = config.policy.model.world_model_cfg
    assert world.context_length == context
    assert world.rebuild_kv_window_from_tokens is rebuild
    assert world.mcts_cache_by_node is node
    assert f'nodecache{int(node)}' in config.exp_name


@pytest.mark.parametrize('kwargs, message', [
    ({'exact_kv_window_reset': True, 'rebuild_kv_window_from_tokens': True}, 'mutually exclusive'),
    ({'num_unroll_steps_override': 1}, 'requires infer_context_length >= 2'),
    ({'num_unroll_steps_override': 4, 'infer_context_length_override': 5}, 'cannot exceed'),
])
def test_incompatible_search_options_fail_before_training(tmp_path, kwargs, message):
    with pytest.raises(ValueError, match=message):
        experimental.main('ALE/Pong-v5', 0, output_root=str(tmp_path), **kwargs)

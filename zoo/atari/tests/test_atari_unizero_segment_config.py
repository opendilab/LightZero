import pytest

from zoo.atari.config import (
    atari_unizero_segment_config,
    atari_unizero_segment_experimental_config,
)


def _unizero_policy():
    from lzero.policy.unizero import UniZeroPolicy
    return UniZeroPolicy


def test_stable_segment_config_is_the_best_mspacman_recipe_without_experimental_features():
    config, create_config, max_env_step = atari_unizero_segment_config.build_config(
        env_id='ALE/MsPacman-v5', seed=0
    )

    policy_config = config.policy
    world_model_config = policy_config.model.world_model_cfg
    assert max_env_step == int(3e6)
    assert config.env.collector_env_num == 8
    assert config.env.evaluator_env_num == 3
    assert config.env.n_evaluator_episode == 3
    assert policy_config.num_segments == 8
    assert policy_config.game_segment_length == 200
    assert policy_config.batch_size == 256
    assert policy_config.replay_ratio == 0.1
    assert policy_config.num_unroll_steps == 10
    assert policy_config.num_simulations == 50
    assert policy_config.collect_num_simulations == 25
    assert policy_config.evaluator_env_num == 3
    assert policy_config.eval_freq == int(1e4)
    assert policy_config.fixed_temperature_value == 0.25
    assert policy_config.obs_loss_weight == 10.0
    assert policy_config.value_loss_weight == 0.5
    assert policy_config.use_priority is False
    assert policy_config.use_augmentation is False
    assert policy_config.grad_clip_mode == 'global'
    assert config.exp_name.endswith(
        '_stabfix_norebuildkv_noctxreanalyze_reanalyze2e-10_'
        'nobootctx_olc0_noper_noaug_seed0_3m'
    )
    assert policy_config.use_adaptive_entropy_weight is False
    assert policy_config.bootstrap_value_context is False
    assert policy_config.buffer_reanalyze_freq == 2e-10
    assert policy_config.reanalyze_batch_size == 160
    assert policy_config.reanalyze_partition == 0.75
    assert policy_config.replay_buffer_size == int(5e5)
    assert world_model_config.context_length == 10
    assert world_model_config.use_priority is False
    assert world_model_config.rebuild_kv_window_from_tokens is False
    assert world_model_config.open_loop_consistency_loss_weight == 0
    assert world_model_config.open_loop_consistency_batch_size == 8
    assert world_model_config.open_loop_consistency_horizon == 4
    assert world_model_config.open_loop_prefix_transitions == 3
    assert world_model_config.policy_entropy_weight == 0.005
    assert create_config.policy.type == 'unizero'

    # Diagnostics, cache-isolation ablations, and resume tooling stay out of the
    # stable config; they belong to the experimental launcher.
    assert 'gradient_diagnostic_freq' not in policy_config
    assert 'open_loop_diagnostic_freq' not in world_model_config
    assert 'contextual_reanalysis' not in policy_config
    assert 'isolate_eval_cache' not in policy_config


def test_stable_segment_config_allows_max_env_step_override():
    _, _, max_env_step = atari_unizero_segment_config.build_config(
        env_id='ALE/MsPacman-v5', seed=0, max_env_step_override=int(5e6)
    )
    assert max_env_step == int(5e6)

    with pytest.raises(ValueError, match='max_env_step must be positive'):
        atari_unizero_segment_config.build_config(
            env_id='ALE/MsPacman-v5', seed=0, max_env_step_override=0
        )


def test_stable_segment_config_resolves_fixed_augmentation_and_unique_run_name():
    config, _, _ = atari_unizero_segment_config.build_config(
        env_id='ALE/Pong-v5', seed=0, use_augmentation=True
    )
    assert config.policy.use_augmentation is True
    assert config.policy.augmentation == ['shift', 'intensity']
    assert config.policy.grad_clip_mode == 'separate_encoder'
    assert config.exp_name.endswith(
        '_stabfix_norebuildkv_noctxreanalyze_reanalyze2e-10_'
        'nobootctx_olc0_noper_fixed-aug_seed0_3m'
    )

    overridden, _, _ = atari_unizero_segment_config.build_config(
        env_id='ALE/Pong-v5', seed=0, use_augmentation=True,
        grad_clip_mode_override='global', run_name='pong aug global smoke',
    )
    assert overridden.policy.grad_clip_mode == 'global'
    assert overridden.exp_name.endswith('/pong-aug-global-smoke')

    with pytest.raises(ValueError, match='grad_clip_mode'):
        atari_unizero_segment_config.build_config(
            env_id='ALE/Pong-v5', seed=0, grad_clip_mode_override='per_head'
        )


def test_prepare_run_directory_allows_only_explicit_checkpoint_resume(tmp_path):
    run_dir = tmp_path / 'run'
    run_dir.mkdir()
    atari_unizero_segment_experimental_config._prepare_run_directory(
        str(run_dir), resume_from='/tmp/checkpoint.pth.tar', resume_in_place=True
    )
    with pytest.raises(ValueError, match='requires resume_from'):
        atari_unizero_segment_experimental_config._prepare_run_directory(
            str(run_dir), resume_from=None, resume_in_place=True
        )


def test_prepare_run_directory_preserves_default_collision_protection(tmp_path):
    run_dir = tmp_path / 'run'
    run_dir.mkdir()
    with pytest.raises(FileExistsError, match='Run directory already exists'):
        atari_unizero_segment_experimental_config._prepare_run_directory(str(run_dir))


def test_experimental_collect_temperature_is_explicit_and_validated():
    assert atari_unizero_segment_experimental_config._resolve_collect_temperature(None) == 0.25
    assert atari_unizero_segment_experimental_config._resolve_collect_temperature(0.5) == 0.5
    with pytest.raises(ValueError, match='collect_temperature must be positive'):
        atari_unizero_segment_experimental_config._resolve_collect_temperature(0)


def test_experimental_default_run_name_records_resolved_training_features():
    run_name = atari_unizero_segment_experimental_config._default_run_name(
        'Pong', 0, '20260828_120000',
        num_unroll_steps=10,
        infer_context_length=5,
        game_segment_length=200,
        batch_size=256,
        replay_ratio=0.1,
        collect_temperature=0.25,
        obs_loss_weight=10.0,
        value_loss_weight=0.5,
        open_loop_consistency_weight=0.0,
        use_priority=False,
        use_augmentation=False,
        bootstrap_value_context=False,
        rebuild_kv_window_from_tokens=True,
        contextual_reanalysis=True,
        buffer_reanalyze_freq=0.02,
        stab_fix=True,
        max_env_step=500000,
    )

    assert '_stabfix_rebuildkv_ctxreanalyze_reanalyze0.02_' in run_name
    assert '_nobootctx_olc0_noper_noaug_' in run_name
    assert run_name.endswith('_seed0_0.5m_20260828_120000')


def test_augmentation_uses_separate_encoder_gradient_clipping_by_default():
    resolver = atari_unizero_segment_experimental_config._resolve_grad_clip_mode
    assert resolver(use_augmentation=True) == 'separate_encoder'
    assert resolver(use_augmentation=False) == 'global'
    assert resolver(use_augmentation=True, override='global') == 'global'
    with pytest.raises(ValueError, match='grad_clip_mode'):
        resolver(use_augmentation=True, override='per_head')


def test_experimental_cache_namespace_preserves_legacy_and_isolated_modes():
    resolver = atari_unizero_segment_experimental_config._resolve_inference_env_num
    assert resolver(8, 8, False) == 8
    assert resolver(8, 8, True) == 16


def test_experimental_defaults_use_fast_sparse_evaluation(monkeypatch, tmp_path):
    import lzero.entry

    captured = {}

    def fake_train(config_pair, **kwargs):
        captured['config'] = config_pair[0]

    monkeypatch.setattr(lzero.entry, 'train_unizero_segment', fake_train)
    atari_unizero_segment_experimental_config.main(
        env_id='ALE/MsPacman-v5',
        seed=0,
        output_root=str(tmp_path),
        run_name='experimental-eval-defaults',
        max_env_step_override=1,
    )

    config = captured['config']
    assert config.env.evaluator_env_num == 3
    assert config.env.n_evaluator_episode == 3
    assert config.policy.evaluator_env_num == 3
    assert config.policy.eval_freq == int(1e4)


def test_experimental_auto_name_tracks_implicit_contextual_reanalysis(monkeypatch, tmp_path):
    import lzero.entry

    captured = {}

    def fake_train(config_pair, **kwargs):
        captured['config'] = config_pair[0]

    monkeypatch.setattr(lzero.entry, 'train_unizero_segment', fake_train)
    atari_unizero_segment_experimental_config.main(
        env_id='ALE/Pong-v5',
        seed=0,
        output_root=str(tmp_path),
        max_env_step_override=500000,
        stab_fix=True,
        rebuild_kv_window_from_tokens=True,
        bootstrap_value_context=False,
        buffer_reanalyze_freq_override=0.02,
        open_loop_consistency_weight_override=0.0,
        use_priority=False,
        use_augmentation_override=False,
    )

    config = captured['config']
    assert config.policy.contextual_reanalysis is True
    # False-valued experimental overrides remain sparse until DI-engine merges
    # UniZeroPolicy defaults; the generated name must still record the resolved state.
    assert config.policy.get('bootstrap_value_context', False) is False
    assert config.policy.buffer_reanalyze_freq == 0.02
    assert config.policy.model.world_model_cfg.open_loop_consistency_loss_weight == 0.0
    assert (
        '_stabfix_rebuildkv_ctxreanalyze_reanalyze0.02_'
        'nobootctx_olc0_noper_noaug_' in config.exp_name
    )


def test_unizero_policy_defaults_disable_experimental_training_features():
    UniZeroPolicy = _unizero_policy()
    config = UniZeroPolicy.config
    world_model_config = config['model']['world_model_cfg']

    assert world_model_config['context_length'] == 2 * 4
    assert world_model_config['exact_kv_window_reset'] is False
    assert world_model_config['rebuild_kv_window_from_tokens'] is False
    assert config['contextual_reanalysis'] is False
    assert config['bootstrap_value_context'] is False
    assert world_model_config['open_loop_consistency_loss_weight'] == 0.
    assert world_model_config['open_loop_recurrent_loss_weight'] == 0.
    assert world_model_config['open_loop_prefix_transitions'] == 0
    assert config['gradient_diagnostic_freq'] == 0
    assert config['use_encoder_clip_annealing'] is False
    assert config['latent_norm_clip_threshold'] == 0.


def test_encoder_clip_diagnostics_are_registered_for_tensorboard():
    UniZeroPolicy = _unizero_policy()
    monitor_vars = set(UniZeroPolicy._monitor_vars_learn(None))
    assert {
        'encoder_clip/enabled',
        'encoder_clip/applied',
        'encoder_clip/apply_count',
        'encoder_clip/scale_factor',
        'encoder_clip/max_latent_norm',
        'encoder_clip/threshold',
    } <= monitor_vars


def test_optimization_diagnostics_are_registered_for_tensorboard():
    UniZeroPolicy = _unizero_policy()
    monitor_vars = set(UniZeroPolicy._monitor_vars_learn(None))
    assert {
        'replay/sample_age_fraction_mean',
        'value_calibration/bias',
        'value_calibration/correlation',
        'grad/clip_scale',
        'grad_component/value/encoder',
        'grad_component/policy/head_policy',
    } <= monitor_vars


def test_atari_experimental_overrides_are_sparse_and_explicit():
    policy_overrides, world_model_overrides = (
        atari_unizero_segment_experimental_config._experimental_config_overrides()
    )
    assert policy_overrides == {}
    assert world_model_overrides == {}

    policy_overrides, world_model_overrides = (
        atari_unizero_segment_experimental_config._experimental_config_overrides(
            infer_context_length=5,
            exact_kv_window_reset=True,
            contextual_reanalysis=True,
            bootstrap_value_context=True,
            open_loop_consistency_weight=1.,
            open_loop_consistency_batch_size=8,
            open_loop_consistency_horizon=4,
            open_loop_prefix_transitions=3,
            encoder_clip_enabled=True,
        )
    )
    assert policy_overrides == {
        'contextual_reanalysis': True,
        'bootstrap_value_context': True,
        'use_encoder_clip_annealing': True,
        'latent_norm_clip_threshold': 10.0,
    }
    assert world_model_overrides == {
        'context_length': 10,
        'exact_kv_window_reset': True,
        'open_loop_consistency_loss_weight': 1.,
        'open_loop_consistency_batch_size': 8,
        'open_loop_consistency_horizon': 4,
        'open_loop_prefix_transitions': 3,
    }

    _, world_model_overrides = atari_unizero_segment_experimental_config._experimental_config_overrides(
        rebuild_kv_window_from_tokens=True
    )
    assert world_model_overrides == {'rebuild_kv_window_from_tokens': True}


def test_explicit_encoder_clip_disable_sets_both_projection_owners():
    policy_overrides, _ = atari_unizero_segment_experimental_config._experimental_config_overrides(
        encoder_clip_enabled=False
    )
    assert policy_overrides == {
        'use_encoder_clip_annealing': False,
        'latent_norm_clip_threshold': 0.0,
    }


@pytest.mark.parametrize(
    ('kwargs', 'message'),
    [
        ({'infer_context_length': 0}, 'infer_context_length must be positive'),
        ({'open_loop_consistency_weight': -1.}, 'must be non-negative'),
        ({'open_loop_consistency_batch_size': 0}, 'batch_size must be positive'),
        ({'open_loop_consistency_horizon': 0}, 'horizon must be positive'),
        ({'open_loop_prefix_transitions': -1}, 'must be non-negative'),
    ],
)
def test_experimental_override_validation(kwargs, message):
    with pytest.raises(ValueError, match=message):
        atari_unizero_segment_experimental_config._experimental_config_overrides(**kwargs)

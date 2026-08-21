import pytest

from zoo.atari.config import (
    atari_unizero_segment_config,
    atari_unizero_segment_experimental_config,
)


def _unizero_policy():
    from lzero.policy.unizero import UniZeroPolicy
    return UniZeroPolicy


def test_stable_segment_config_is_small_baseline_without_experimental_overrides():
    config, create_config, max_env_step = atari_unizero_segment_config.build_config(
        env_id='ALE/Pong-v5', seed=0
    )

    policy_config = config.policy
    world_model_config = policy_config.model.world_model_cfg
    assert max_env_step == int(5e5)
    assert policy_config.num_segments == 8
    assert policy_config.game_segment_length == 20
    assert policy_config.batch_size == 64
    assert policy_config.replay_ratio == 0.25
    assert policy_config.buffer_reanalyze_freq == 1 / 100000
    assert policy_config.reanalyze_batch_size == 160
    assert policy_config.reanalyze_partition == 0.75
    assert world_model_config.context_length == 8
    assert create_config.policy.type == 'unizero'

    assert 'bootstrap_value_context' not in policy_config
    assert 'contextual_reanalysis' not in policy_config
    assert 'gradient_diagnostic_freq' not in policy_config
    assert 'use_adaptive_entropy_weight' not in policy_config
    assert 'open_loop_consistency_loss_weight' not in world_model_config
    assert 'open_loop_recurrent_loss_weight' not in world_model_config
    assert 'rebuild_kv_window_from_tokens' not in world_model_config


def test_stable_segment_config_allows_max_env_step_override():
    _, _, max_env_step = atari_unizero_segment_config.build_config(
        env_id='ALE/MsPacman-v5', seed=0, max_env_step_override=int(5e6)
    )
    assert max_env_step == int(5e6)

    with pytest.raises(ValueError, match='max_env_step must be positive'):
        atari_unizero_segment_config.build_config(
            env_id='ALE/MsPacman-v5', seed=0, max_env_step_override=0
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

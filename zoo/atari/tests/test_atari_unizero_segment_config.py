import pytest

from lzero.policy.unizero import UniZeroPolicy
from zoo.atari.config import atari_unizero_segment_config


def test_prepare_run_directory_allows_only_explicit_checkpoint_resume(tmp_path):
    run_dir = tmp_path / 'run'
    run_dir.mkdir()
    atari_unizero_segment_config._prepare_run_directory(
        str(run_dir), resume_from='/tmp/checkpoint.pth.tar', resume_in_place=True
    )
    with pytest.raises(ValueError, match='requires resume_from'):
        atari_unizero_segment_config._prepare_run_directory(
            str(run_dir), resume_from=None, resume_in_place=True
        )


def test_prepare_run_directory_preserves_default_collision_protection(tmp_path):
    run_dir = tmp_path / 'run'
    run_dir.mkdir()
    with pytest.raises(FileExistsError, match='Run directory already exists'):
        atari_unizero_segment_config._prepare_run_directory(str(run_dir))


def test_unizero_policy_defaults_disable_experimental_training_features():
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
        atari_unizero_segment_config._experimental_config_overrides()
    )
    assert policy_overrides == {}
    assert world_model_overrides == {}

    policy_overrides, world_model_overrides = (
        atari_unizero_segment_config._experimental_config_overrides(
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

    _, world_model_overrides = atari_unizero_segment_config._experimental_config_overrides(
        rebuild_kv_window_from_tokens=True
    )
    assert world_model_overrides == {'rebuild_kv_window_from_tokens': True}


def test_explicit_encoder_clip_disable_sets_both_projection_owners():
    policy_overrides, _ = atari_unizero_segment_config._experimental_config_overrides(
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
        atari_unizero_segment_config._experimental_config_overrides(**kwargs)

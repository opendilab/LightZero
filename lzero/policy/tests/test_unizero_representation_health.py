import pytest
import torch

from lzero.policy.unizero import (
    encoder_clip_metrics,
    replay_distribution_metrics,
    representation_health_metrics,
    should_run_periodic_monitor,
)


def test_representation_health_detects_collapse_even_when_token_norms_match():
    collapsed = torch.tensor([[[1.0, 0.0]], [[1.0, 0.0]]])
    diverse = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])

    # Both populations have identical per-token L2 norms, so the old metric is
    # zero for both and cannot distinguish healthy diversity from collapse.
    assert collapsed.norm(dim=-1).std().item() == pytest.approx(0.0)
    assert diverse.norm(dim=-1).std().item() == pytest.approx(0.0)

    collapsed_metrics = representation_health_metrics(collapsed)
    diverse_metrics = representation_health_metrics(diverse)

    assert collapsed_metrics['activation/x_token/feature_std_mean'] == pytest.approx(0.0)
    assert collapsed_metrics['activation/x_token/near_constant_fraction'] == pytest.approx(1.0)
    assert diverse_metrics['activation/x_token/feature_std_mean'] == pytest.approx(0.5)
    assert diverse_metrics['activation/x_token/near_constant_fraction'] == pytest.approx(0.0)


def test_representation_health_rejects_missing_feature_dimension():
    with pytest.raises(ValueError, match='Expected'):
        representation_health_metrics(torch.tensor(1.0))


def test_periodic_monitor_runs_immediately_after_resume_then_on_boundaries():
    assert should_run_periodic_monitor(train_iter=10023, frequency=10000, last_check_iter=-1)
    assert not should_run_periodic_monitor(train_iter=10024, frequency=10000, last_check_iter=10023)
    assert should_run_periodic_monitor(train_iter=20000, frequency=10000, last_check_iter=10023)
    assert not should_run_periodic_monitor(train_iter=20000, frequency=0, last_check_iter=-1)


def test_replay_distribution_metrics_expose_effective_sample_size():
    uniform = replay_distribution_metrics(torch.ones(4), torch.tensor([1.0, 2.0, 3.0, 4.0]))
    skewed = replay_distribution_metrics(torch.tensor([1.0, 0.1, 0.1, 0.1]), torch.ones(4))

    assert uniform['replay/is_weight_mean'] == pytest.approx(1.0)
    assert uniform['replay/is_weight_std'] == pytest.approx(0.0)
    assert uniform['replay/is_weight_ess_fraction'] == pytest.approx(1.0)
    assert uniform['replay/value_priority_std'] == pytest.approx(1.11803399)
    assert skewed['replay/is_weight_ess_fraction'] < 0.5


def test_encoder_clip_metrics_follow_learner_monitor_float_contract():
    metrics = encoder_clip_metrics(
        threshold=10,
        applied=True,
        apply_count=3,
        scale_factor=0.75,
        max_latent_norm=13,
    )

    assert metrics['encoder_clip/apply_count'] == 3.0
    assert all(type(value) is float for value in metrics.values())

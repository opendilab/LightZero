import pytest
import torch

from lzero.policy.unizero import (
    component_gradient_norms,
    encoder_clip_metrics,
    gradient_clip_metrics,
    replay_distribution_metrics,
    replay_sample_age_metrics,
    representation_health_metrics,
    search_exploration_metrics,
    should_run_periodic_monitor,
    value_calibration_metrics,
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


def test_replay_sample_age_metrics_distinguish_old_and_new_samples():
    metrics = replay_sample_age_metrics([0, 24, 74, 99], num_transitions=100, capacity=200)

    assert metrics['replay/buffer_fill_fraction'] == pytest.approx(0.5)
    assert metrics['replay/sample_oldest_quarter_fraction'] == pytest.approx(0.5)
    assert metrics['replay/sample_newest_quarter_fraction'] == pytest.approx(0.25)


def test_value_calibration_metrics_respect_mask_and_signed_bias():
    metrics = value_calibration_metrics(
        torch.tensor([[2., 100.], [4., 6.]]),
        torch.tensor([[1., 0.], [5., 7.]]),
        torch.tensor([[1, 0], [1, 1]], dtype=torch.bool),
    )

    assert metrics['value_calibration/bias'] == pytest.approx(-1. / 3.)
    assert metrics['value_calibration/mae'] == pytest.approx(1.)
    assert metrics['value_calibration/rmse'] == pytest.approx(1.)


def test_gradient_clip_metrics_match_global_norm_scaling():
    clipped = gradient_clip_metrics(total_norm=64., max_norm=20.)
    untouched = gradient_clip_metrics(total_norm=4., max_norm=20.)

    assert clipped['grad/clip_applied'] == 1.
    assert clipped['grad/clip_scale'] == pytest.approx(20. / (64. + 1e-6))
    assert clipped['grad/world_model_post_clip_norm'] == pytest.approx(20., rel=1e-5)
    assert untouched['grad/clip_scale'] == 1.


def test_search_exploration_metrics_separate_prior_search_and_sampling():
    metrics = search_exploration_metrics(
        policy_logits=[0., 0., 0.], visit_counts=[20., 4., 1.], temperature=0.25
    )

    assert metrics['exploration/prior_effective_actions'] == pytest.approx(3.)
    assert metrics['exploration/raw_visit_top1_probability'] == pytest.approx(0.8)
    assert metrics['exploration/sample_top1_probability'] > 0.99
    assert metrics['exploration/prior_visit_js_divergence'] > 0.


def test_component_gradient_norms_do_not_mutate_parameter_gradients():
    first = torch.nn.Linear(2, 1, bias=False)
    second = torch.nn.Linear(2, 1, bias=False)
    inputs = torch.tensor([[1., 2.]])
    components = {
        'first_only': first(inputs).square().mean(),
        'both': (first(inputs) + second(inputs)).square().mean(),
    }

    metrics = component_gradient_norms(components, {'first': first, 'second': second})

    assert metrics['grad_component/first_only/first'] > 0.
    assert metrics['grad_component/first_only/second'] == 0.
    assert metrics['grad_component/both/second'] > 0.
    assert all(parameter.grad is None for parameter in (*first.parameters(), *second.parameters()))

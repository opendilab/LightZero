"""Check the Bellman operator, independent of the Atari learner's own losses."""
import pytest
import torch
from easydict import EasyDict

from lzero.policy.scaling_transform import (
    DiscreteSupport, InverseScalarTransform, inverse_scalar_transform, phi_transform, scalar_transform,
)
from lzero.policy.target_smoothing import scalar_decode_smoothing_eps


@pytest.mark.parametrize('bounds', [(-300., 301.), (-2., 8.)])
@pytest.mark.parametrize('smoothing', [0.0, 0.1, 0.3])
def test_smoothed_categorical_mean_recovers_scalar_on_asymmetric_support(bounds, smoothing):
    support = DiscreteSupport(*bounds)
    values = torch.tensor([-1., 0., 1., 10.], dtype=torch.float64)
    probs = phi_transform(support, scalar_transform(values), smoothing)
    logits = probs.clamp_min(1e-100).log()
    function = inverse_scalar_transform(logits, support, label_smoothing_eps=smoothing).flatten()
    handle = InverseScalarTransform(support, label_smoothing_eps=smoothing)
    torch.testing.assert_close(function, values, atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(handle(logits).flatten(), values, atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(handle(logits[0]), values[:1], atol=1e-9, rtol=1e-9)


def test_compensated_five_step_bellman_operator_preserves_true_fixed_point():
    support = DiscreteSupport(-300., 301.)
    gamma = .997
    prefix = sum(gamma ** i for i in range(5))
    true_value = torch.tensor([1 / (1 - gamma)], dtype=torch.float64)
    decoder = InverseScalarTransform(support, label_smoothing_eps=.1)
    fitted = true_value.clone()
    legacy = true_value.clone()
    for _ in range(200):
        target = prefix + gamma ** 5 * fitted
        fitted = decoder(phi_transform(support, scalar_transform(target), .1).log()).flatten()
        old_target = prefix + gamma ** 5 * legacy
        legacy = inverse_scalar_transform(phi_transform(support, scalar_transform(old_target), .1).log(), support).flatten()
    torch.testing.assert_close(fitted, true_value, atol=1e-6, rtol=1e-6)
    assert legacy.item() == pytest.approx(23.96, abs=.02)


def test_unizero_search_keeps_reward_and_value_decode_smoothing_independent():
    from lzero.mcts.tree_search.mcts_ctree import UniZeroMCTSCtree
    cfg = EasyDict(device='cpu', compensate_scalar_label_smoothing=True,
                   reward_label_smoothing_eps=0., value_label_smoothing_eps=.1,
                   model=dict(value_support_range=(-300., 301., 1.),
                              reward_support_range=(-300., 301., 1.), categorical_distribution=True))
    tree = UniZeroMCTSCtree(cfg)
    scalar = torch.tensor([10.])
    for name, smooth in [('value', .1), ('reward', 0.)]:
        logits = phi_transform(tree.value_support, scalar_transform(scalar), smooth).clamp_min(1e-30).log()
        decoded = getattr(tree, name + '_inverse_scalar_transform_handle')(logits)
        torch.testing.assert_close(decoded.flatten(), scalar, atol=2e-4, rtol=2e-5)
    cfg.compensate_scalar_label_smoothing = False
    assert scalar_decode_smoothing_eps(cfg, 'value') == 0.
    assert UniZeroMCTSCtree(cfg).value_inverse_scalar_transform_handle.label_smoothing_eps == 0.


@pytest.mark.parametrize('smoothing', [-.1, 1., float('nan'), float('inf')])
def test_invalid_decode_smoothing_rejected(smoothing):
    with pytest.raises(ValueError):
        InverseScalarTransform(DiscreteSupport(-1, 2), label_smoothing_eps=smoothing)


def test_scalar_predictions_cannot_have_categorical_smoothing_compensation():
    with pytest.raises(ValueError, match='categorical'):
        InverseScalarTransform(DiscreteSupport(-1, 2), False, label_smoothing_eps=.1)

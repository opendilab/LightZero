import pytest
import torch

from lzero.policy.scaling_transform import DiscreteSupport, phi_transform, scalar_transform, inverse_scalar_transform
from lzero.policy.target_smoothing import scalar_smoothing_eps


def decode_target(values, epsilon):
    support = DiscreteSupport(-300., 301.)
    distribution = phi_transform(support, scalar_transform(values), epsilon)
    return inverse_scalar_transform(distribution.clamp_min(1e-30).log(), support).flatten()


def test_unsmoothed_targets_preserve_signed_scalar_values():
    values = torch.tensor([-100., -10., -1., 0., 1., 10., 100.])
    torch.testing.assert_close(decode_target(values, 0.), values, atol=2e-4, rtol=2e-5)


def test_uniform_smoothing_changes_the_bellman_fixed_point():
    # Closed-loop value fitting must preserve r/(1-gamma), not just minimize CE.
    gamma = 0.997
    reward_prefix = sum(gamma ** i for i in range(5))
    true_value = torch.tensor([1 / (1 - gamma)])
    target = reward_prefix + gamma ** 5 * true_value
    torch.testing.assert_close(decode_target(target, 0.), true_value, atol=2e-3, rtol=2e-5)
    assert decode_target(target, 0.1).item() < 0.85 * true_value.item()


def test_reward_and_value_overrides_are_independent_and_keep_legacy_compatibility():
    config = dict(label_smoothing_eps=0.1, value_label_smoothing_eps=0.)
    values = torch.tensor([10.])
    assert decode_target(values, scalar_smoothing_eps(config, 'reward')).item() == pytest.approx(8.52, abs=0.01)
    assert decode_target(values, scalar_smoothing_eps(config, 'value')).item() == pytest.approx(10., abs=2e-4)
    config['value_label_smoothing_eps'] = None
    assert scalar_smoothing_eps(config, 'value') == 0.1


@pytest.mark.parametrize('epsilon', [-0.1, 1., float('nan'), float('inf')])
def test_invalid_scalar_smoothing_is_rejected(epsilon):
    with pytest.raises(ValueError, match='finite and in'):
        scalar_smoothing_eps(dict(value_label_smoothing_eps=epsilon), 'value')

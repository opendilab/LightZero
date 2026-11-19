import pytest
import torch

from lzero.policy.scaling_transform import DiscreteSupport, phi_transform, scalar_transform, inverse_scalar_transform
from lzero.policy.target_smoothing import scalar_decode_smoothing_eps


@pytest.mark.parametrize('bounds', [(-300., 301., 1.), (-2., 3., .5), (0., 2., 1.), (0., 1., 1.)])
@pytest.mark.parametrize('eps', [0., .1, .9])
def test_local_smoothing_preserves_mass_and_mean_including_boundaries(bounds, eps):
    support = DiscreteSupport(*bounds)
    low, high = support.arange[0, 0], support.arange[0, -1]
    values = torch.linspace(low-1, high+1, 100, dtype=torch.float64).reshape(10, 10)
    q = phi_transform(support, values, eps, smoothing_mode='local')
    assert (q >= 0).all()
    torch.testing.assert_close(q.sum(-1), torch.ones_like(values))
    torch.testing.assert_close((q*support.arange).sum(-1), values.clamp(low, high))


def test_local_smoothing_spreads_only_to_neighbours_and_keeps_edges_absorbing():
    support = DiscreteSupport(-2., 3.)
    q = phi_transform(support, torch.tensor([0., 2.]), .1, smoothing_mode='local')
    torch.testing.assert_close(q[0], torch.tensor([0., .05, .9, .05, 0.]))
    torch.testing.assert_close(q[1], torch.tensor([0., 0., 0., 0., 1.]))


def test_local_smoothed_bellman_fixed_point_needs_no_decode_compensation():
    support = DiscreteSupport(-300., 301.)
    gamma = .997
    value = torch.tensor([1/(1-gamma)], dtype=torch.float64)
    target = sum(gamma**i for i in range(5)) + gamma**5 * value
    q = phi_transform(support, scalar_transform(target), .1, smoothing_mode='local')
    decoded = inverse_scalar_transform(q.clamp_min(1e-100).log(), support).flatten()
    torch.testing.assert_close(decoded, value, atol=1e-8, rtol=1e-8)
    assert scalar_decode_smoothing_eps(dict(compensate_scalar_label_smoothing=True,
        value_label_smoothing_mode='local', value_label_smoothing_eps=.1), 'value') == 0.


def test_local_target_has_no_uniform_probability_floor_far_from_its_scalar():
    support = DiscreteSupport(-300., 301.)
    q = phi_transform(support, torch.tensor([0.]), .1, smoothing_mode='local')
    assert (q[0, support.arange[0].abs() > 1] == 0).all()
    assert ((q*support.arange.square()).sum() == pytest.approx(.1))


def test_unknown_smoothing_mode_is_rejected():
    with pytest.raises(ValueError, match='mode'):
        phi_transform(DiscreteSupport(-1., 2.), torch.tensor([0.]), .1, smoothing_mode='typo')

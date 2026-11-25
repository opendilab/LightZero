from copy import deepcopy
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from easydict import EasyDict

from lzero.reward_model.rnd_reward_model import RNDNetworkRepr, RNDRewardModel


class ObservationError(torch.nn.Module):
    def forward(self, obs):
        return obs, torch.zeros_like(obs)


def make_model():
    cfg = EasyDict(deepcopy(RNDRewardModel.config))
    cfg.update(input_type='obs', obs_shape=4, latent_state_dim=4, rnd_buffer_size=16,
               weight_decay=0.0, hidden_size_list=[4, 4], input_norm=False,
               intrinsic_reward_weight=1.0, extrinsic_reward_norm=False)
    model = RNDRewardModel(cfg, 'cpu', tb_logger=Mock())
    model.reward_model = ObservationError()
    return model


def make_batch(padding_value=100.0, mask=(1, 1, 0)):
    obs = np.array([[1.0] * 4 + [2.0] * 4 + [padding_value] * 4], dtype=np.float32)
    rewards = np.array([[0.0, 0.5, 0.0]], dtype=np.float32)
    return [[obs, None, np.array([mask], dtype=np.float32)], [rewards, None, None]]


@pytest.mark.unittest
def test_padding_does_not_receive_bonus_or_change_valid_reward_scale():
    outputs = []
    for padding in (3.0, 100.0):
        model = make_model()
        batch = make_batch(padding)
        original_rewards = batch[1][0]
        outputs.append(model.estimate(batch)[1][0])
        np.testing.assert_array_equal(original_rewards, [[0.0, 0.5, 0.0]])
        assert model._running_mean_std_rnd_reward.mean == pytest.approx(2.5, abs=1e-3)
    np.testing.assert_allclose(outputs[0], outputs[1])
    np.testing.assert_allclose(outputs[0].reshape(-1), [0.0, 1.5, 0.0], atol=1e-6)


@pytest.mark.unittest
@pytest.mark.parametrize('mask', [(0, 0, 0), (1, 0, 0)])
def test_empty_or_single_valid_transition_has_finite_zero_bonus(mask):
    model = make_model()
    batch = make_batch(mask=mask)
    before = batch[1][0].copy()
    result = model.estimate(batch)[1][0]
    assert np.isfinite(result).all()
    np.testing.assert_array_equal(result.reshape(before.shape), before)


@pytest.mark.unittest
def test_all_valid_batch_keeps_existing_minmax_bonus():
    result = make_model().estimate(make_batch(3.0, mask=(1, 1, 1)))[1][0]
    np.testing.assert_allclose(result.reshape(-1), [0.0, 0.875, 1.0], atol=1e-6)


@pytest.mark.unittest
@pytest.mark.parametrize('input_type', ['latent_state', 'obs_latent_state'])
def test_single_valid_transition_preserves_shared_encoder_training_state(input_type):
    model = make_model()
    model.input_type = input_type
    encoder = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.BatchNorm1d(4))
    model.representation_network = encoder
    if input_type == 'obs_latent_state':
        model.reward_model = RNDNetworkRepr(4, 4, [4], encoder)
    running_mean = encoder[1].running_mean.clone()
    before_modes = [module.training for module in encoder.modules()]
    batch = make_batch(mask=(1, 0, 0))
    result = model.estimate(batch)[1][0]
    assert np.isfinite(result).all()
    assert before_modes == [module.training for module in encoder.modules()]
    torch.testing.assert_close(encoder[1].running_mean, running_mean)
    assert all(p.grad is None for p in encoder.parameters())


@pytest.mark.unittest
def test_inconsistent_transition_mask_reports_shape_error():
    batch = make_batch()
    batch[0][2] = np.ones((1, 2))
    with pytest.raises(ValueError, match='transition mask shape'):
        make_model().estimate(batch)

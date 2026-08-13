from types import SimpleNamespace

import pytest

from lzero.reward_model.rnd_reward_model import RNDRewardModel


def _model(step, final_weight=0.0, decay_start=25, decay_steps=45):
    model = RNDRewardModel.__new__(RNDRewardModel)
    model.cfg = SimpleNamespace(
        intrinsic_reward_weight=1 / 300,
        intrinsic_reward_weight_final=final_weight,
        intrinsic_reward_weight_decay_start=decay_start,
        intrinsic_reward_weight_decay_steps=decay_steps,
    )
    model.estimate_cnt_rnd = step
    return model


@pytest.mark.parametrize(
    'step, expected',
    [
        (0, 1 / 300),
        (25, 1 / 300),
        (47, (1 / 300) * (1 - 22 / 45)),
        (70, 0.0),
        (100, 0.0),
    ],
)
def test_intrinsic_reward_weight_linear_decay(step, expected):
    assert _model(step)._current_intrinsic_reward_weight() == pytest.approx(expected)


def test_intrinsic_reward_weight_legacy_constant_behavior():
    model = _model(step=100)
    model.cfg.intrinsic_reward_weight_final = None
    assert model._current_intrinsic_reward_weight() == pytest.approx(1 / 300)


def test_intrinsic_reward_weight_zero_length_decay():
    model = _model(step=26, final_weight=0.001, decay_steps=0)
    assert model._current_intrinsic_reward_weight() == pytest.approx(0.001)

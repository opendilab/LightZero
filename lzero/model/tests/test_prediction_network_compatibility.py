import io
import importlib

import pytest
import torch

from lzero.model.common import PredictionNetwork


@pytest.mark.unittest
@pytest.mark.parametrize('norm_type', ['BN', 'LN'])
@pytest.mark.parametrize('positional', [False, True])
@pytest.mark.parametrize('observation_shape,latent_shape,downsample', [
    ((3, 3, 5), (3, 5), False),
    ((3, 64, 64), (8, 8), True),
    ((3, 84, 84), (6, 6), True),
    ((3, 96, 96), (6, 6), True),
])
def test_prediction_head_legacy_api_and_optimizer(norm_type, positional, observation_shape, latent_shape, downsample):
    height, width = latent_shape
    args = dict(
        observation_shape=observation_shape, action_space_size=3, num_res_blocks=1,
        num_channels=8, value_head_channels=2, policy_head_channels=2,
        value_head_hidden_channels=[16], policy_head_hidden_channels=[16],
        output_support_size=5, flatten_input_size_for_value_head=2 * height * width,
        flatten_input_size_for_policy_head=2 * height * width, downsample=downsample,
    )
    if positional:
        model = PredictionNetwork(*args.values(), last_linear_layer_init_zero=False, norm_type=norm_type)
    else:
        model = PredictionNetwork(**args, last_linear_layer_init_zero=False, norm_type=norm_type)

    # Learners construct optimizers before any inference. All parameters must
    # already exist and retain their identities across the first forward pass.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    parameter_ids = [id(p) for p in model.parameters()]
    before = model.fc_policy[-1].weight.detach().clone()
    policy, value = model(torch.randn(2, 8, height, width))
    assert policy.shape == (2, 3)
    assert value.shape == (2, 5)
    assert parameter_ids == [id(p) for p in model.parameters()]
    (policy.square().mean() + value.square().mean()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    assert not torch.equal(before, model.fc_policy[-1].weight)

    checkpoint = io.BytesIO()
    torch.save(model.state_dict(), checkpoint)
    checkpoint.seek(0)
    restored = PredictionNetwork(**args, last_linear_layer_init_zero=False, norm_type=norm_type)
    restored.load_state_dict(torch.load(checkpoint, weights_only=True), strict=True)
    model.eval()
    restored.eval()
    latent = torch.randn(2, 8, height, width)
    for actual, expected in zip(restored(latent), model(latent)):
        torch.testing.assert_close(actual, expected)


@pytest.mark.unittest
def test_prediction_heads_keep_zero_initialization():
    model = PredictionNetwork((3, 3, 3), 4, 1, 8, 2, 2, [16], [16], 5, 18, 18)
    policy, value = model(torch.randn(2, 8, 3, 3))
    torch.testing.assert_close(policy, torch.zeros(2, 4))
    torch.testing.assert_close(value, torch.zeros(2, 5))


@pytest.mark.unittest
@pytest.mark.parametrize('module,name,shape', [
    ('muzero_model', 'MuZeroModel', (3, 3, 3)),
    ('efficientzero_model', 'EfficientZeroModel', (3, 3, 3)),
    ('stochastic_muzero_model', 'StochasticMuZeroModel', (3, 3, 3)),
    ('stochastic_muzero_model_mlp', 'StochasticMuZeroModelMLP', 4),
    ('muzero_model_multitask', 'MuZeroMTModel', (3, 3, 3)),
])
def test_model_callers_initial_inference(module, name, shape):
    cls = getattr(importlib.import_module('lzero.model.' + module), name)
    model = cls(
        observation_shape=shape, action_space_size=3, num_channels=8,
        num_res_blocks=1, self_supervised_learning_loss=False, downsample=False,
    )
    obs = torch.randn(2, shape) if isinstance(shape, int) else torch.randn(2, *shape)
    output = model.initial_inference(obs)
    assert output.policy_logits.shape == (2, 3)
    assert torch.isfinite(output.value).all()
    assert torch.isfinite(output.policy_logits).all()
    action = torch.zeros(2, dtype=torch.long)
    if name == 'EfficientZeroModel':
        output = model.recurrent_inference(output.latent_state, output.reward_hidden_state, action)
    elif name.startswith('Stochastic'):
        afterstate = model.recurrent_inference(output.latent_state, action)
        output = model.recurrent_inference(afterstate.latent_state, action, afterstate=True)
    else:
        output = model.recurrent_inference(output.latent_state, action)
    assert output.policy_logits.shape == (2, 3)
    assert torch.isfinite(output.value).all()

from itertools import product

import pytest
import torch
from ding.torch_utils import is_differentiable

from lzero.model.alphazero_model import PredictionNetwork

action_space_size = [2, 3]
batch_size = [100, 200]
num_res_blocks = [3]
num_channels = [3]
value_head_channels = [8]
policy_head_channels = [8]
fc_value_layers = [[
    16,
]]
fc_policy_layers = [[
    16,
]]
output_support_size = [2]
observation_shape = [1, 3, 3]

prediction_network_args = list(
    product(
        action_space_size,
        batch_size,
        num_res_blocks,
        num_channels,
        value_head_channels,
        policy_head_channels,
        fc_value_layers,
        fc_policy_layers,
        output_support_size,
    )
)


@pytest.mark.unittest
class TestAlphaZeroModel:

    def output_check(self, model, outputs):
        if isinstance(outputs, torch.Tensor):
            loss = outputs.sum()
        elif isinstance(outputs, list):
            loss = sum([t.sum() for t in outputs])
        elif isinstance(outputs, dict):
            loss = sum([v.sum() for v in outputs.values()])
        is_differentiable(loss, model)

    @pytest.mark.parametrize(
        'action_space_size, batch_size, num_res_blocks, num_channels, value_head_channels, policy_head_channels, fc_value_layers, fc_policy_layers, output_support_size',
        prediction_network_args
    )
    def test_prediction_network(
        self, action_space_size, batch_size, num_res_blocks, num_channels, value_head_channels, policy_head_channels,
        fc_value_layers, fc_policy_layers, output_support_size
    ):
        obs = torch.rand(batch_size, num_channels, 3, 3)
        flatten_output_size_for_value_head = value_head_channels * observation_shape[1] * observation_shape[2]
        flatten_output_size_for_policy_head = policy_head_channels * observation_shape[1] * observation_shape[2]
        # print('='*20)
        # print(batch_size, num_res_blocks, num_channels, action_space_size, fc_value_layers, fc_policy_layers, output_support_size)
        # print('='*20)
        prediction_network = PredictionNetwork(
            action_space_size=action_space_size,
            num_res_blocks=num_res_blocks,
            num_channels=num_channels,
            value_head_channels=value_head_channels,
            policy_head_channels=policy_head_channels,
            fc_value_layers=fc_value_layers,
            fc_policy_layers=fc_policy_layers,
            output_support_size=output_support_size,
            flatten_output_size_for_value_head=flatten_output_size_for_value_head,
            flatten_output_size_for_policy_head=flatten_output_size_for_policy_head,
            last_linear_layer_init_zero=True,
        )
        policy, value = prediction_network(obs)
        assert policy.shape == torch.Size([batch_size, action_space_size])
        assert value.shape == torch.Size([batch_size, output_support_size])


if __name__ == "__main__":
    action_space_size = 2
    batch_size = 100
    num_res_blocks = 3
    num_channels = 3
    reward_head_channels = 2
    value_head_channels = 8
    policy_head_channels = 8
    fc_value_layers = [16]
    fc_policy_layers = [16]
    output_support_size = 2
    observation_shape = [1, 3, 3]
    obs = torch.rand(batch_size, num_channels, 3, 3)
    flatten_output_size_for_value_head = value_head_channels * observation_shape[1] * observation_shape[2]
    flatten_output_size_for_policy_head = policy_head_channels * observation_shape[1] * observation_shape[2]
    print('=' * 20)
    print(
        batch_size, num_res_blocks, num_channels, action_space_size, reward_head_channels, fc_value_layers,
        fc_policy_layers, output_support_size
    )
    print('=' * 20)
    prediction_network = PredictionNetwork(
        action_space_size=action_space_size,
        num_res_blocks=num_res_blocks,
        num_channels=num_channels,
        value_head_channels=value_head_channels,
        policy_head_channels=policy_head_channels,
        fc_value_layers=fc_value_layers,
        fc_policy_layers=fc_policy_layers,
        output_support_size=output_support_size,
        flatten_output_size_for_value_head=flatten_output_size_for_value_head,
        flatten_output_size_for_policy_head=flatten_output_size_for_policy_head,
        last_linear_layer_init_zero=True,
    )
    policy, value = prediction_network(obs)
    assert policy.shape == torch.Size([batch_size, action_space_size])
    assert value.shape == torch.Size([batch_size, output_support_size])

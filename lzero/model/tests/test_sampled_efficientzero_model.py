from itertools import product

import pytest
import torch
from ding.torch_utils import is_differentiable

from lzero.model.sampled_efficientzero_model import PredictionNetwork, DynamicsNetwork

batch_size = [100, 10]
num_res_blocks = [3, 4, 5]
num_channels = [10]
lstm_hidden_size = [64]
action_space_size = [2, 3]
reward_head_channels = [2]
fc_reward_layers = [[16]]
output_support_size = [2]
flatten_output_size_for_reward_head = [180]
dynamics_network_args = list(
    product(
        batch_size, num_res_blocks, num_channels, lstm_hidden_size, action_space_size, reward_head_channels,
        fc_reward_layers, output_support_size, flatten_output_size_for_reward_head
    )
)

value_head_channels = [8]
policy_head_channels = [8]
fc_value_layers = [[
    16,
]]
fc_policy_layers = [[
    16,
]]
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
class TestSampledEfficientZeroModel:

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
        prediction_network = PredictionNetwork(
            observation_shape=observation_shape,
            continuous_action_space=True,
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
        assert policy.shape == torch.Size([batch_size, action_space_size * 2])
        assert value.shape == torch.Size([batch_size, output_support_size])

    @pytest.mark.parametrize(
        'batch_size, num_res_blocks, num_channels, lstm_hidden_size, action_space_size, reward_head_channels, fc_reward_layers, output_support_size,'
        'flatten_output_size_for_reward_head', dynamics_network_args
    )
    def test_dynamics_network(
        self, batch_size, num_res_blocks, num_channels, lstm_hidden_size, action_space_size, reward_head_channels,
        fc_reward_layers, output_support_size, flatten_output_size_for_reward_head
    ):
        print('=' * 20)
        print(
            batch_size, num_res_blocks, num_channels, lstm_hidden_size, action_space_size, reward_head_channels,
            fc_reward_layers, output_support_size, flatten_output_size_for_reward_head
        )
        print('=' * 20)
        observation_shape = [1, 3, 3]
        flatten_output_size_for_reward_head = reward_head_channels * observation_shape[1] * observation_shape[2]
        state_action_embedding = torch.rand(batch_size, num_channels, observation_shape[1], observation_shape[2])
        dynamics_network = DynamicsNetwork(
            observation_shape=observation_shape,
            action_encoding_dim=action_space_size,
            num_res_blocks=num_res_blocks,
            num_channels=num_channels,
            lstm_hidden_size=lstm_hidden_size,
            reward_head_channels=reward_head_channels,
            fc_reward_layers=fc_reward_layers,
            output_support_size=output_support_size,
            flatten_output_size_for_reward_head=flatten_output_size_for_reward_head
        )
        next_state, reward_hidden_state, value_prefix = dynamics_network(
            state_action_embedding,
            (torch.randn(1, batch_size, lstm_hidden_size), torch.randn(1, batch_size, lstm_hidden_size))
        )
        assert next_state.shape == torch.Size([batch_size, num_channels - action_space_size, 3, 3])
        assert reward_hidden_state[0].shape == torch.Size([1, batch_size, lstm_hidden_size])
        assert reward_hidden_state[1].shape == torch.Size([1, batch_size, lstm_hidden_size])
        assert value_prefix.shape == torch.Size([batch_size, output_support_size])


if __name__ == "__main__":
    batch_size = 2
    num_res_blocks = 3
    num_channels = 10
    lstm_hidden_size = 64
    action_space_size = 5
    reward_head_channels = 2
    fc_reward_layers = [16]
    output_support_size = 2
    observation_shape = [1, 3, 3]
    # flatten_output_size_for_reward_head = 180
    flatten_output_size_for_reward_head = reward_head_channels * observation_shape[1] * observation_shape[2]

    state_action_embedding = torch.rand(batch_size, num_channels, observation_shape[1], observation_shape[2])
    dynamics_network = DynamicsNetwork(
        observation_shape=observation_shape,
        action_encoding_dim=action_space_size,
        num_res_blocks=num_res_blocks,
        num_channels=num_channels,
        reward_head_channels=reward_head_channels,
        fc_reward_layers=fc_reward_layers,
        output_support_size=output_support_size,
        flatten_output_size_for_reward_head=flatten_output_size_for_reward_head
    )
    next_state, reward_hidden_state, value_prefix = dynamics_network(
        state_action_embedding,
        (torch.randn(1, batch_size, lstm_hidden_size), torch.randn(1, batch_size, lstm_hidden_size))
    )
    assert next_state.shape == torch.Size([batch_size, num_channels - action_space_size, 3, 3])
    assert reward_hidden_state[0].shape == torch.Size([1, batch_size, lstm_hidden_size])
    assert reward_hidden_state[1].shape == torch.Size([1, batch_size, lstm_hidden_size])
    assert value_prefix.shape == torch.Size([batch_size, output_support_size])

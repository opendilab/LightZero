from itertools import product

import pytest
import torch
from ding.torch_utils import is_differentiable

from lzero.model.muzero_model import PredictionNetwork, DynamicsNetwork

batch_size = [10, 100]
num_res_blocks = [3, 4]
num_channels = [3]
reward_head_channels = [2]
fc_reward_layers = [[16, 8]]
output_support_size = [2]
flatten_output_size_for_reward_head = [180]
dynamics_network_args = list(
    product(
        batch_size, num_res_blocks, num_channels, reward_head_channels, fc_reward_layers, output_support_size,
        flatten_output_size_for_reward_head
    )
)

action_space_size = [2, 3]
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
class TestMuZeroModel:

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
            observation_shape=observation_shape,
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

    @pytest.mark.parametrize(
        'batch_size, num_res_blocks, num_channels, reward_head_channels, fc_reward_layers, output_support_size,'
        'flatten_output_size_for_reward_head', dynamics_network_args
    )
    def test_dynamics_network(
        self, batch_size, num_res_blocks, num_channels, reward_head_channels, fc_reward_layers, output_support_size,
        flatten_output_size_for_reward_head
    ):
        observation_shape = [1, 3, 3]
        action_space_size = 1
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
        next_state, reward = dynamics_network(state_action_embedding)
        assert next_state.shape == torch.Size([batch_size, num_channels - action_space_size, 3, 3])
        assert reward.shape == torch.Size([batch_size, output_support_size])


if __name__ == "__main__":
    batch_size = 10
    num_res_blocks = 3
    num_channels = 3 + 1  # 3 channels for observation and 1 channel for action
    action_space_size = 1
    reward_head_channels = 2
    fc_reward_layers = [16, 8]
    output_support_size = 2
    flatten_output_size_for_reward_head = reward_head_channels * 3 * 3
    state_action_embedding = torch.rand(batch_size, num_channels, 3, 3)
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
    state, reward = dynamics_network(state_action_embedding)
    assert state.shape == torch.Size([10, num_channels - action_space_size, 3, 3])
    assert reward.shape == torch.Size([10, output_support_size])

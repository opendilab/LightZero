from itertools import product

import pytest
import torch
from ding.torch_utils import is_differentiable

from lzero.model.muzero_model import PredictionNetwork, DynamicsNetwork


@pytest.mark.parametrize('norm_type', ['BN', 'LN'])
def test_atari_prediction_heads_register_parameters_before_optimizer(norm_type):
    network = PredictionNetwork(
        observation_shape=(4, 64, 64), action_space_size=6, num_res_blocks=0, num_channels=4,
        value_head_channels=2, policy_head_channels=2,
        value_head_hidden_channels=[8], policy_head_hidden_channels=[8], output_support_size=11,
        flatten_input_size_for_value_head=2 * 8 * 8,
        flatten_input_size_for_policy_head=2 * 8 * 8, downsample=True, norm_type=norm_type,
    )
    optimizer = torch.optim.SGD(network.parameters(), lr=0.1)
    before = {name: (id(parameter), tuple(parameter.shape)) for name, parameter in network.named_parameters()}
    policy, value = network(torch.randn(2, 4, 8, 8))
    assert policy.shape == (2, 6) and value.shape == (2, 11)
    (policy.sum() + value.sum()).backward()
    optimizer.step()
    after = {name: (id(parameter), tuple(parameter.shape)) for name, parameter in network.named_parameters()}
    assert before == after
    assert network.fc_value[-1].bias.detach().abs().sum() > 0

@pytest.mark.parametrize('norm_type', ['BN', 'LN'])
def test_multitask_model_constructs_registered_prediction_heads(norm_type):
    from lzero.model.muzero_model_multitask import MuZeroMTModel

    model = MuZeroMTModel(
        observation_shape=(3, 8, 8), action_space_size=3, num_channels=4,
        reward_head_channels=2, value_head_channels=2, policy_head_channels=2,
        fc_reward_layers=[8], fc_value_layers=[8], fc_policy_layers=[8],
        reward_support_size=11, value_support_size=11, task_num=2, norm_type=norm_type,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    parameter_ids = {id(parameter) for group in optimizer.param_groups for parameter in group['params']}
    for task in range(2):
        output = model.initial_inference(torch.randn(2, 3, 8, 8), task_id=task)
        assert output.policy_logits.shape == (2, 3)
        assert output.value.shape == (2, 11)
        (output.policy_logits.sum() + output.value.sum()).backward()
    optimizer.step()
    for head in model.prediction_networks:
        assert all(id(parameter) in parameter_ids for parameter in head.parameters())
        assert head.fc_value[-1].bias.detach().abs().sum() > 0


batch_size = [10, 100]
num_res_blocks = [3, 4]
num_channels = [3]
reward_head_channels = [2]
reward_head_hidden_channels = [[16, 8]]
output_support_size = [2]
flatten_input_size_for_reward_head = [180]
dynamics_network_args = list(
    product(
        batch_size, num_res_blocks, num_channels, reward_head_channels, reward_head_hidden_channels, output_support_size,
        flatten_input_size_for_reward_head
    )
)

action_space_size = [2, 3]
value_head_channels = [8]
policy_head_channels = [8]
value_head_hidden_channels = [[
    16,
]]
policy_head_hidden_channels = [[
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
        value_head_hidden_channels,
        policy_head_hidden_channels,
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
        'action_space_size, batch_size, num_res_blocks, num_channels, value_head_channels, policy_head_channels, value_head_hidden_channels, policy_head_hidden_channels, output_support_size',
        prediction_network_args
    )
    def test_prediction_network(
        self, action_space_size, batch_size, num_res_blocks, num_channels, value_head_channels, policy_head_channels,
        value_head_hidden_channels, policy_head_hidden_channels, output_support_size
    ):
        obs = torch.rand(batch_size, num_channels, 3, 3)
        flatten_input_size_for_value_head = value_head_channels * observation_shape[1] * observation_shape[2]
        flatten_input_size_for_policy_head = policy_head_channels * observation_shape[1] * observation_shape[2]
        # print('='*20)
        # print(batch_size, num_res_blocks, num_channels, action_space_size, value_head_hidden_channels, policy_head_hidden_channels, output_support_size)
        # print('='*20)
        prediction_network = PredictionNetwork(
            observation_shape=observation_shape,
            action_space_size=action_space_size,
            num_res_blocks=num_res_blocks,
            num_channels=num_channels,
            value_head_channels=value_head_channels,
            policy_head_channels=policy_head_channels,
            value_head_hidden_channels=value_head_hidden_channels,
            policy_head_hidden_channels=policy_head_hidden_channels,
            output_support_size=output_support_size,
            flatten_input_size_for_value_head=flatten_input_size_for_value_head,
            flatten_input_size_for_policy_head=flatten_input_size_for_policy_head,
            last_linear_layer_init_zero=True,
        )
        policy, value = prediction_network(obs)
        assert policy.shape == torch.Size([batch_size, action_space_size])
        assert value.shape == torch.Size([batch_size, output_support_size])

    @pytest.mark.parametrize(
        'batch_size, num_res_blocks, num_channels, reward_head_channels, reward_head_hidden_channels, output_support_size,'
        'flatten_input_size_for_reward_head', dynamics_network_args
    )
    def test_dynamics_network(
        self, batch_size, num_res_blocks, num_channels, reward_head_channels, reward_head_hidden_channels, output_support_size,
        flatten_input_size_for_reward_head
    ):
        observation_shape = [1, 3, 3]
        action_space_size = 1
        flatten_input_size_for_reward_head = reward_head_channels * observation_shape[1] * observation_shape[2]
        state_action_embedding = torch.rand(batch_size, num_channels, observation_shape[1], observation_shape[2])
        dynamics_network = DynamicsNetwork(
            observation_shape=observation_shape,
            action_encoding_dim=action_space_size,
            num_res_blocks=num_res_blocks,
            num_channels=num_channels,
            reward_head_channels=reward_head_channels,
            reward_head_hidden_channels=reward_head_hidden_channels,
            output_support_size=output_support_size,
            flatten_input_size_for_reward_head=flatten_input_size_for_reward_head
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
    reward_head_hidden_channels = [16, 8]
    output_support_size = 2
    flatten_input_size_for_reward_head = reward_head_channels * 3 * 3
    state_action_embedding = torch.rand(batch_size, num_channels, 3, 3)
    dynamics_network = DynamicsNetwork(
        observation_shape=observation_shape,
        action_encoding_dim=action_space_size,
        num_res_blocks=num_res_blocks,
        num_channels=num_channels,
        reward_head_channels=reward_head_channels,
        reward_head_hidden_channels=reward_head_hidden_channels,
        output_support_size=output_support_size,
        flatten_input_size_for_reward_head=flatten_input_size_for_reward_head
    )
    state, reward = dynamics_network(state_action_embedding)
    assert state.shape == torch.Size([10, num_channels - action_space_size, 3, 3])
    assert reward.shape == torch.Size([10, output_support_size])

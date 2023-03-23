import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from ding.torch_utils import MLP, ResBlock
from ding.utils import MODEL_REGISTRY, SequenceType

from .common import RepresentationNetwork


@MODEL_REGISTRY.register('AlphaZeroModel')
class AlphaZeroModel(nn.Module):

    def __init__(
        self,
        observation_shape: SequenceType = (12, 96, 96),
        action_space_size: int = 6,
        representation_network_type: str = 'conv_res_blocks',
        categorical_distribution: bool = False,
        activation: Optional[nn.Module] = nn.ReLU(inplace=True),
        representation_network: nn.Module = None,
        batch_norm_momentum: float = 0.1,
        last_linear_layer_init_zero: bool = True,
        state_norm: bool = False,
        downsample: bool = False,
        num_res_blocks: int = 1,
        num_channels: int = 64,
        value_head_channels: int = 16,
        policy_head_channels: int = 16,
        fc_value_layers: SequenceType = [32],
        fc_policy_layers: SequenceType = [32],
        value_support_size: int = 601,
    ):
        """
        Overview:
            AlphaZero model.
        Arguments:
            - observation_shape (:obj:`SequenceType`): Observation space shape, e.g. [C, W, H]=[12, 96, 96].
            - action_space_size: (:obj:`int`): Action space size, such as 6.
            - representation_network_type (:obj:`Optional[str]`): The type of representation_network in AlphaZero model. options={'conv_res_blocks', 'identity'}.
            - categorical_distribution (:obj:`bool`): Whether to use discrete support to represent categorical distribution for value, reward/value_prefix.
            - activation (:obj:`Optional[nn.Module]`): the activation in AlphaZero model.
            - representation_network (:obj:`nn.Module`): the user-defined representation_network.
            - batch_norm_momentum (:obj:`float`):  Momentum of BN
            - last_linear_layer_init_zero (:obj:`bool`): Whether to use zero initialization for the last layer of value/policy mlp, default set it to True.
            - state_norm (:obj:`bool`): Whether to use normalization for hidden states, default set it to True.
            - downsample (:obj:`bool`): Whether to do downsampling for observations in ``representation_network``, default set it to True. \
                But sometimes, we do not need, e.g. board games.
            - num_res_blocks (:obj:`int`): number of res blocks in AlphaZero model.
            - num_channels (:obj:`int`): channels of hidden states.
            - value_head_channels (:obj:`int`): channels of value head.
            - policy_head_channels (:obj:`int`): channels of policy head.
            - fc_value_layers (:obj:`SequenceType`): hidden layers of the value prediction head (MLP head).
            - fc_policy_layers (:obj:`SequenceType`): hidden layers of the policy prediction head (MLP head).
            - value_support_size (:obj:`int`): dim of value output.
        """
        super(AlphaZeroModel, self).__init__()
        self.categorical_distribution = categorical_distribution
        self.observation_shape = observation_shape
        if self.categorical_distribution:
            self.value_support_size = value_support_size
        else:
            self.value_support_size = 1

        self.last_linear_layer_init_zero = last_linear_layer_init_zero
        self.state_norm = state_norm
        self.representation_network_type = representation_network_type
        self.representation_network = representation_network
        self.downsample = downsample

        self.action_space_size = action_space_size
        flatten_output_size_for_value_head = (
            (value_head_channels * math.ceil(self.observation_shape[1] / 16) *
             math.ceil(self.observation_shape[2] / 16)) if downsample else
            (value_head_channels * self.observation_shape[1] * self.observation_shape[2])
        )

        flatten_output_size_for_policy_head = (
            (policy_head_channels * math.ceil(self.observation_shape[1] / 16) *
             math.ceil(self.observation_shape[2] / 16)) if downsample else
            (policy_head_channels * self.observation_shape[1] * self.observation_shape[2])
        )

        if self.representation_network_type == 'identity':
            self.prediction_network = PredictionNetwork(
                action_space_size,
                num_res_blocks,
                self.observation_shape[0],  # in_channels
                num_channels,
                value_head_channels,
                policy_head_channels,
                fc_value_layers,
                fc_policy_layers,
                self.value_support_size,
                flatten_output_size_for_value_head,
                flatten_output_size_for_policy_head,
                momentum=batch_norm_momentum,
                last_linear_layer_init_zero=self.last_linear_layer_init_zero,
                activation=activation,
            )
        else:
            self.prediction_network = PredictionNetwork(
                action_space_size,
                num_res_blocks,
                None,  # in_channels
                num_channels,
                value_head_channels,
                policy_head_channels,
                fc_value_layers,
                fc_policy_layers,
                self.value_support_size,
                flatten_output_size_for_value_head,
                flatten_output_size_for_policy_head,
                momentum=batch_norm_momentum,
                last_linear_layer_init_zero=self.last_linear_layer_init_zero,
                activation=activation,
            )

        if self.representation_network is None:
            if self.representation_network_type == 'identity':
                self.representation_network = nn.Identity()
            elif self.representation_network_type == 'conv_res_blocks':
                self.representation_network = RepresentationNetwork(
                    self.observation_shape,
                    num_res_blocks,
                    num_channels,
                    downsample,
                    momentum=batch_norm_momentum,
                    activation=activation,
                )
        else:
            self.representation_network = self.representation_network

    def forward(self, encoded_state: torch.Tensor):
        encoded_state = self.representation_network(encoded_state)
        logit, value = self.prediction_network(encoded_state)
        return logit, value

    def compute_prob_value(self, state_batch: torch.Tensor):
        logits, values = self.forward(state_batch)
        dist = torch.distributions.Categorical(logits=logits)
        probs = dist.probs
        return probs, values

    def compute_logp_value(self, state_batch: torch.Tensor):
        logits, values = self.forward(state_batch)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs, values


class PredictionNetwork(nn.Module):

    def __init__(
        self,
        action_space_size,
        num_res_blocks,
        in_channels,
        num_channels,
        value_head_channels,
        policy_head_channels,
        fc_value_layers,
        fc_policy_layers,
        output_support_size,
        flatten_output_size_for_value_head,
        flatten_output_size_for_policy_head,
        momentum: float = 0.1,
        last_linear_layer_init_zero: bool = True,
        activation: Optional[nn.Module] = nn.ReLU(inplace=True),
    ):
        """
        Overview:
            Prediction network. Predict the value and policy given hidden state.
        Arguments:
            - action_space_size: (:obj:`int`): Action space size, such as 6.
            - num_res_blocks (:obj:`int`): number of res blocks in model.
            - in_channels (:obj:`int`): channels of input, if None, then in_channels = num_channels
            - num_channels (:obj:`int`): channels of hidden states.
            - value_head_channels (:obj:`int`): channels of value head.
            - policy_head_channels (:obj:`int`): channels of policy head.
            - fc_value_layers (:obj:`SequenceType`): hidden layers of the value prediction head (MLP head).
            - fc_policy_layers (:obj:`SequenceType`): hidden layers of the policy prediction head (MLP head).
            - output_support_size (:obj:`int`): dim of value output.
            - flatten_output_size_for_value_head (:obj:`int`): dim of flatten hidden states.
            - flatten_output_size_for_policy_head (:obj:`int`): dim of flatten hidden states.
            - last_linear_layer_init_zero (:obj:`bool`): Whether to use zero initialization for the last layer of value/policy mlp, default set it to True.
        """
        super().__init__()
        self.in_channels = in_channels
        if self.in_channels is not None:
            self.conv_input = nn.Conv2d(in_channels, num_channels, 1)

        self.resblocks = nn.ModuleList(
            [
                ResBlock(in_channels=num_channels, activation=activation, norm_type='BN', res_type='basic', bias=False)
                for _ in range(num_res_blocks)
            ]
        )

        self.conv1x1_value = nn.Conv2d(num_channels, value_head_channels, 1)
        self.conv1x1_policy = nn.Conv2d(num_channels, policy_head_channels, 1)
        self.bn_value = nn.BatchNorm2d(value_head_channels, momentum=momentum)
        self.bn_policy = nn.BatchNorm2d(policy_head_channels, momentum=momentum)
        self.flatten_output_size_for_value_head = flatten_output_size_for_value_head
        self.flatten_output_size_for_policy_head = flatten_output_size_for_policy_head
        self.fc_value = MLP(
            in_channels=self.flatten_output_size_for_value_head,
            hidden_channels=fc_value_layers[0],
            out_channels=output_support_size,
            layer_num=len(fc_value_layers) + 1,
            activation=activation,
            norm_type='LN',
            output_activation=nn.Identity(),
            output_norm_type=None,
            last_linear_layer_init_zero=last_linear_layer_init_zero
        )
        self.fc_policy = MLP(
            in_channels=self.flatten_output_size_for_policy_head,
            hidden_channels=fc_policy_layers[0],
            out_channels=action_space_size,
            layer_num=len(fc_policy_layers) + 1,
            activation=activation,
            norm_type='LN',
            output_activation=nn.Identity(),
            output_norm_type=None,
            last_linear_layer_init_zero=last_linear_layer_init_zero
        )

        self.activation = activation

    def forward(self, x: torch.Tensor):
        if self.in_channels is not None:
            x = self.conv_input(x)

        for block in self.resblocks:
            x = block(x)

        value = self.conv1x1_value(x)
        value = self.bn_value(value)
        value = self.activation(value)

        policy = self.conv1x1_policy(x)
        policy = self.bn_policy(policy)
        policy = self.activation(policy)

        value = value.reshape(-1, self.flatten_output_size_for_value_head)
        policy = policy.reshape(-1, self.flatten_output_size_for_policy_head)

        value = self.fc_value(value)
        policy = self.fc_policy(policy)
        return policy, value

"""
Overview:
    BTW, users can refer to the unittest of these model templates to learn how to use them.
"""
from typing import Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ding.torch_utils import MLP, ResBlock
from ding.utils import MODEL_REGISTRY, SequenceType

from .common import RepresentationNetwork


# use ModelRegistry to register the model, for more details about ModelRegistry, please refer to DI-engine's document.
@MODEL_REGISTRY.register('AlphaZeroModel')
class AlphaZeroModel(nn.Module):

    def __init__(
        self,
        observation_shape: SequenceType = (12, 96, 96),
        action_space_size: int = 6,
        categorical_distribution: bool = False,
        activation: Optional[nn.Module] = nn.ReLU(inplace=True),
        representation_network: nn.Module = None,
        last_linear_layer_init_zero: bool = True,
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
            The definition of AlphaZero model, which is a general model for AlphaZero algorithm.
        Arguments:
            - observation_shape (:obj:`SequenceType`): Observation space shape, e.g. [C, W, H]=[24, 19, 19] for go.
            - action_space_size: (:obj:`int`): Action space size, usually an integer number for discrete action space.
            - categorical_distribution (:obj:`bool`): Whether to use discrete support to represent categorical \
                distribution for value.
            - activation (:obj:`Optional[nn.Module]`): Activation function used in network, which often use in-place \
                operation to speedup, e.g. ReLU(inplace=True).
            - representation_network (:obj:`nn.Module`): The user-defined representation_network. In some complex \
                environment, we may need to define a customized representation_network.
            - last_linear_layer_init_zero (:obj:`bool`): Whether to use zero initializationss for the last layer of \
                value/policy mlp, default sets it to True.
            - downsample (:obj:`bool`): Whether to do downsampling for observations in ``representation_network``, \
                in board games, this argument is usually set to False.
            - num_res_blocks (:obj:`int`): The number of res blocks in AlphaZero model.
            - num_channels (:obj:`int`): The channels of hidden states.
            - value_head_channels (:obj:`int`): The channels of value head.
            - policy_head_channels (:obj:`int`): The channels of policy head.
            - fc_value_layers (:obj:`SequenceType`): The number of hidden layers used in value head (MLP head).
            - fc_policy_layers (:obj:`SequenceType`): The number of hidden layers used in policy head (MLP head).
            - value_support_size (:obj:`int`): The size of categorical value.
        """
        super(AlphaZeroModel, self).__init__()
        self.categorical_distribution = categorical_distribution
        self.observation_shape = observation_shape
        if self.categorical_distribution:
            self.value_support_size = value_support_size
        else:
            self.value_support_size = 1

        self.last_linear_layer_init_zero = last_linear_layer_init_zero
        self.representation_network = representation_network

        self.action_space_size = action_space_size
        # TODO use more adaptive way to get the flatten output size
        flatten_output_size_for_value_head = (
            (
                value_head_channels * math.ceil(self.observation_shape[1] / 16) *
                math.ceil(self.observation_shape[2] / 16)
            ) if downsample else (value_head_channels * self.observation_shape[1] * self.observation_shape[2])
        )

        flatten_output_size_for_policy_head = (
            (
                policy_head_channels * math.ceil(self.observation_shape[1] / 16) *
                math.ceil(self.observation_shape[2] / 16)
            ) if downsample else (policy_head_channels * self.observation_shape[1] * self.observation_shape[2])
        )

        self.prediction_network = PredictionNetwork(
            action_space_size,
            num_res_blocks,
            num_channels,
            value_head_channels,
            policy_head_channels,
            fc_value_layers,
            fc_policy_layers,
            self.value_support_size,
            flatten_output_size_for_value_head,
            flatten_output_size_for_policy_head,
            last_linear_layer_init_zero=self.last_linear_layer_init_zero,
            activation=activation,
        )

        if self.representation_network is None:
            self.representation_network = RepresentationNetwork(
                self.observation_shape,
                num_res_blocks,
                num_channels,
                downsample,
                activation=activation,
            )
        else:
            self.representation_network = self.representation_network

    def forward(self, state_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Overview:
            The common computation graph of AlphaZero model.
        Arguments:
            - state_batch (:obj:`torch.Tensor`): The input state data, e.g. 2D image with the shape of [C, H, W].
        Returns:
            - logit (:obj:`torch.Tensor`): The output logit to select discrete action.
            - value (:obj:`torch.Tensor`): The output value of input state to help policy improvement and evaluation.
        Shapes:
            - state_batch (:obj:`torch.Tensor`): :math:`(B, C, H, W)`, where B is batch size, C is channel, H is \
                height, W is width.
            - logit (:obj:`torch.Tensor`): :math:`(B, N)`, where B is batch size, N is action space size.
            - value (:obj:`torch.Tensor`): :math:`(B, 1)`, where B is batch size.
        """
        encoded_state = self.representation_network(state_batch)
        logit, value = self.prediction_network(encoded_state)
        return logit, value

    def compute_prob_value(self, state_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Overview:
            The computation graph of AlphaZero model to calculate action selection probability and value.
        Arguments:
            - state_batch (:obj:`torch.Tensor`): The input state data, e.g. 2D image with the shape of [C, H, W].
        Returns:
            - prob (:obj:`torch.Tensor`): The output probability to select discrete action.
            - value (:obj:`torch.Tensor`): The output value of input state to help policy improvement and evaluation.
        Shapes:
            - state_batch (:obj:`torch.Tensor`): :math:`(B, C, H, W)`, where B is batch size, C is channel, H is \
                height, W is width.
            - prob (:obj:`torch.Tensor`): :math:`(B, N)`, where B is batch size, N is action space size.
            - value (:obj:`torch.Tensor`): :math:`(B, 1)`, where B is batch size.
        """
        logit, value = self.forward(state_batch)
        # construct categorical distribution to calculate probability
        dist = torch.distributions.Categorical(logits=logit)
        prob = dist.probs
        return prob, value

    def compute_logp_value(self, state_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Overview:
            The computation graph of AlphaZero model to calculate log probability and value.
        Arguments:
            - state_batch (:obj:`torch.Tensor`): The input state data, e.g. 2D image with the shape of [C, H, W].
        Returns:
            - log_prob (:obj:`torch.Tensor`): The output log probability to select discrete action.
            - value (:obj:`torch.Tensor`): The output value of input state to help policy improvement and evaluation.
        Shapes:
            - state_batch (:obj:`torch.Tensor`): :math:`(B, C, H, W)`, where B is batch size, C is channel, H is \
                height, W is width.
            - log_prob (:obj:`torch.Tensor`): :math:`(B, N)`, where B is batch size, N is action space size.
            - value (:obj:`torch.Tensor`): :math:`(B, 1)`, where B is batch size.
        """
        logit, value = self.forward(state_batch)
        # use log_softmax to calculate log probability
        log_prob = F.log_softmax(logit, dim=-1)
        return log_prob, value


class PredictionNetwork(nn.Module):

    def __init__(
            self,
            action_space_size: int,
            num_res_blocks: int,
            num_channels: int,
            value_head_channels: int,
            policy_head_channels: int,
            fc_value_layers: SequenceType,
            fc_policy_layers: SequenceType,
            output_support_size: int,
            flatten_output_size_for_value_head: int,
            flatten_output_size_for_policy_head: int,
            last_linear_layer_init_zero: bool = True,
            activation: Optional[nn.Module] = nn.ReLU(inplace=True),
    ) -> None:
        """
        Overview:
            Prediction network. Predict the value and policy given the hidden state.
        Arguments:
            - action_space_size: (:obj:`int`): Action space size, usually an integer number for discrete action space.
            - num_res_blocks (:obj:`int`): The number of res blocks in AlphaZero model.
            - in_channels (:obj:`int`): The channels of input, if None, then in_channels = num_channels.
            - num_channels (:obj:`int`): The channels of hidden states.
            - value_head_channels (:obj:`int`): The channels of value head.
            - policy_head_channels (:obj:`int`): The channels of policy head.
            - fc_value_layers (:obj:`SequenceType`): The number of hidden layers used in value head (MLP head).
            - fc_policy_layers (:obj:`SequenceType`): The number of hidden layers used in policy head (MLP head).
            - output_support_size (:obj:`int`): The size of categorical value output.
            - flatten_output_size_for_value_head (:obj:`int`): The size of flatten hidden states, i.e. the input size \
                of the value head.
            - flatten_output_size_for_policy_head (:obj:`int`): The size of flatten hidden states, i.e. the input size \
                of the policy head.
            - last_linear_layer_init_zero (:obj:`bool`): Whether to use zero initializations for the last layer of \
                value/policy mlp, default sets it to True.
            - activation (:obj:`Optional[nn.Module]`): Activation function used in network, which often use in-place \
                operation to speedup, e.g. ReLU(inplace=True).
        """
        super().__init__()
        self.resblocks = nn.ModuleList(
            [
                ResBlock(in_channels=num_channels, activation=activation, norm_type='BN', res_type='basic', bias=False)
                for _ in range(num_res_blocks)
            ]
        )

        self.conv1x1_value = nn.Conv2d(num_channels, value_head_channels, 1)
        self.conv1x1_policy = nn.Conv2d(num_channels, policy_head_channels, 1)
        self.norm_value = nn.BatchNorm2d(value_head_channels)
        self.norm_policy = nn.BatchNorm2d(policy_head_channels)
        self.flatten_output_size_for_value_head = flatten_output_size_for_value_head
        self.flatten_output_size_for_policy_head = flatten_output_size_for_policy_head
        self.fc_value = MLP(
            in_channels=self.flatten_output_size_for_value_head,
            hidden_channels=fc_value_layers[0],
            out_channels=output_support_size,
            layer_num=len(fc_value_layers) + 1,
            activation=activation,
            norm_type='BN', 
            output_activation=False,
            output_norm=False,
            last_linear_layer_init_zero=last_linear_layer_init_zero
        )
        self.fc_policy = MLP(
            in_channels=self.flatten_output_size_for_policy_head,
            hidden_channels=fc_policy_layers[0],
            out_channels=action_space_size,
            layer_num=len(fc_policy_layers) + 1,
            activation=activation,
            norm_type='BN', 
            output_activation=False,
            output_norm=False,
            last_linear_layer_init_zero=last_linear_layer_init_zero
        )

        self.activation = activation

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Overview:
            Use the hidden state to predict the value and policy.
        Arguments:
            - x (:obj:`torch.Tensor`): The hidden state.
        Returns:
            - outputs (:obj:`Tuple[torch.Tensor, torch.Tensor]`): The value and policy.
        Shapes:
            - x (:obj:`torch.Tensor`): :math:`(B, C, H, W)`, where B is batch size, C is channel, H is \
                the height of the encoding state, W is width of the encoding state.
            - logit (:obj:`torch.Tensor`): :math:`(B, N)`, where B is batch size, N is action space size.
            - value (:obj:`torch.Tensor`): :math:`(B, 1)`, where B is batch size.
        """
        for block in self.resblocks:
            x = block(x)

        value = self.conv1x1_value(x)
        value = self.norm_value(value)
        value = self.activation(value)

        policy = self.conv1x1_policy(x)
        policy = self.norm_policy(policy)
        policy = self.activation(policy)

        value = value.reshape(-1, self.flatten_output_size_for_value_head)
        policy = policy.reshape(-1, self.flatten_output_size_for_policy_head)

        value = self.fc_value(value)
        logit = self.fc_policy(policy)
        return logit, value

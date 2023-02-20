"""
The following code is adapted from https://github.com/YeWR/EfficientZero/blob/main/config/atari/model.py
"""

import torch.nn.functional as F
import math

import torch
import torch.nn as nn
from ding.torch_utils import MLP, ResBlock
from ding.utils import MODEL_REGISTRY


class DownSample(nn.Module):

    def __init__(self, in_channels, out_channels, momentum=0.1, activation=nn.ReLU(inplace=True)):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels // 2,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels // 2, momentum=momentum)
        self.resblocks1 = nn.ModuleList(
            [
                ResBlock(
                    in_channels=out_channels // 2, activation=activation, norm_type='BN', res_type='basic', bias=False
                ) for _ in range(1)
            ]
        )
        self.conv2 = nn.Conv2d(
            out_channels // 2,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.downsample_block = ResBlock(
            in_channels=out_channels // 2,
            out_channels=out_channels,
            activation=activation,
            norm_type='BN',
            res_type='downsample',
            bias=False
        )
        self.resblocks2 = nn.ModuleList(
            [
                ResBlock(in_channels=out_channels, activation=activation, norm_type='BN', res_type='basic', bias=False)
                for _ in range(1)
            ]
        )
        self.pooling1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.resblocks3 = nn.ModuleList(
            [
                ResBlock(in_channels=out_channels, activation=activation, norm_type='BN', res_type='basic', bias=False)
                for _ in range(1)
            ]
        )
        self.pooling2 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.activation = activation

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        for block in self.resblocks1:
            x = block(x)
        x = self.downsample_block(x)
        for block in self.resblocks2:
            x = block(x)
        x = self.pooling1(x)
        for block in self.resblocks3:
            x = block(x)
        x = self.pooling2(x)
        return x


# Encode the observations into hidden states
class RepresentationNetwork(nn.Module):

    def __init__(
        self,
        observation_shape,
        num_res_blocks,
        num_channels,
        downsample,
        momentum=0.1,
        activation=nn.ReLU(inplace=True),
    ):
        """
        Overview: Representation network
        Arguments:
            - observation_shape (:obj:`Union[List, tuple]`):  shape of observations: [C, W, H]
            - num_res_blocks (:obj:`int`): number of res blocks
            - num_channels (:obj:`int`): channels of hidden states
            - downsample (:obj:`bool`): True -> do downsampling for observations. (For board games, do not need)
        """
        super().__init__()
        self.downsample = downsample
        if self.downsample:
            self.downsample_net = DownSample(
                observation_shape[0],
                num_channels,
            )
        else:
            self.conv = nn.Conv2d(observation_shape[0], num_channels, kernel_size=3, stride=1, padding=1, bias=False)

            self.bn = nn.BatchNorm2d(num_channels, momentum=momentum)
        self.resblocks = nn.ModuleList(
            [
                ResBlock(in_channels=num_channels, activation=activation, norm_type='BN', res_type='basic', bias=False)
                for _ in range(num_res_blocks)
            ]
        )
        self.activation = activation

    def forward(self, x):
        if self.downsample:
            x = self.downsample_net(x)
        else:
            x = self.conv(x)
            x = self.bn(x)
            x = self.activation(x)

        for block in self.resblocks:
            x = block(x)
        return x


# predict the value and policy given hidden states
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
        full_support_size,
        block_output_size_value,
        block_output_size_policy,
        momentum=0.1,
        last_linear_layer_init_zero=False,
        activation=nn.ReLU(inplace=True),
    ):
        """Prediction network
        Parameters
        ----------
        action_space_size: int
            action space
        num_res_blocks: int
            number of res blocks
        in_channels: int
            channels of input, if None, then in_channels = num_channels
        num_channels: int
            channels of hidden states
        value_head_channels: int
            channels of value head
        policy_head_channels: int
            channels of policy head
        fc_value_layers: list
            hidden layers of the value prediction head (MLP head)
        fc_policy_layers: list
            hidden layers of the policy prediction head (MLP head)
        full_support_size: int
            dim of value output
        block_output_size_value: int
            dim of flatten hidden states
        block_output_size_policy: int
            dim of flatten hidden states
        last_linear_layer_init_zero: bool
            True -> zero initialization for the last layer of value/policy mlp
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
        self.block_output_size_value = block_output_size_value
        self.block_output_size_policy = block_output_size_policy
        # TODO(pu)
        self.fc_value = MLP(
            in_channels=self.block_output_size_value,
            hidden_channels=fc_value_layers[0],
            out_channels=full_support_size,
            layer_num=len(fc_value_layers) + 1,
            activation=activation,
            norm_type='LN',
            output_activation=nn.Identity(),
            output_norm_type=None,
            last_linear_layer_init_zero=last_linear_layer_init_zero
        )
        self.fc_policy = MLP(
            in_channels=self.block_output_size_policy,
            hidden_channels=fc_policy_layers[0],
            out_channels=action_space_size,
            layer_num=len(fc_policy_layers) + 1,
            activation=activation,
            norm_type='LN',
            # norm_type=None,
            output_activation=nn.Identity(),
            output_norm_type=None,
            last_linear_layer_init_zero=last_linear_layer_init_zero
        )

        self.activation = activation

    def forward(self, x):
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

        value = value.reshape(-1, self.block_output_size_value)
        policy = policy.reshape(-1, self.block_output_size_policy)

        value = self.fc_value(value)
        policy = self.fc_policy(policy)
        return policy, value


@MODEL_REGISTRY.register('AlphaZeroNet')
class AlphaZeroNet(nn.Module):

    def __init__(
        self,
        observation_shape,
        action_space_size,
        num_res_blocks,
        num_channels,
        value_head_channels,
        policy_head_channels,
        fc_value_layers,
        fc_policy_layers,
        reward_support_size,
        value_support_size,
        downsample,
        representation_model_type: str = 'conv_res_blocks',
        representation_model: nn.Module = None,
        batch_norm_momentum=0.1,
        last_linear_layer_init_zero=False,
        state_norm=False,
        categorical_distribution=True,
        activation=nn.ReLU(inplace=True),
    ):
        """
        Overview:
            AlphaZero network
        Arguments:
            - representation_model_type
            - observation_shape: tuple or list. shape of observations: [C, W, H]
            - action_space_size: (:obj:`int`): . action space
            - num_res_blocks (:obj:`int`):  number of res blocks
            - num_channels (:obj:`int`): channels of hidden states
            - value_head_channels (:obj:`int`): channels of value head
            - policy_head_channels (:obj:`int`): channels of policy head
            - fc_value_layers (:obj:`list`):  hidden layers of the value prediction head (MLP head)
            - fc_policy_layers (:obj:`list`):  hidden layers of the policy prediction head (MLP head)
            - downsample (:obj:`bool`): True -> do downsampling for observations. (For board games, do not need)
            - batch_norm_momentum (:obj:`float`):  Momentum of BN
            - last_linear_layer_init_zero (:obj:`bool`): True -> zero initialization for the last layer of value/policy mlp
            - state_norm (:obj:`bool`):  True -> normalization for hidden states
        """
        super(AlphaZeroNet, self).__init__()
        self.categorical_distribution = categorical_distribution
        if not self.categorical_distribution:
            self.reward_support_size = 1
            self.value_support_size = 1
        else:
            self.reward_support_size = reward_support_size
            self.value_support_size = value_support_size
        self.last_linear_layer_init_zero = last_linear_layer_init_zero
        self.state_norm = state_norm
        self.representation_model_type = representation_model_type
        self.representation_model = representation_model
        self.downsample = downsample

        self.action_space_size = action_space_size
        block_output_size_value = (
            (value_head_channels * math.ceil(observation_shape[1] / 16) *
             math.ceil(observation_shape[2] / 16)) if downsample else
            (value_head_channels * observation_shape[1] * observation_shape[2])
        )

        block_output_size_policy = (
            (policy_head_channels * math.ceil(observation_shape[1] / 16) *
             math.ceil(observation_shape[2] / 16)) if downsample else
            (policy_head_channels * observation_shape[1] * observation_shape[2])
        )

        if self.representation_model_type == 'identity':
            self.prediction_network = PredictionNetwork(
                action_space_size,
                num_res_blocks,
                observation_shape[0],  # in_channels
                num_channels,
                value_head_channels,
                policy_head_channels,
                fc_value_layers,
                fc_policy_layers,
                self.value_support_size,
                block_output_size_value,
                block_output_size_policy,
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
                block_output_size_value,
                block_output_size_policy,
                momentum=batch_norm_momentum,
                last_linear_layer_init_zero=self.last_linear_layer_init_zero,
                activation=activation,
            )

        if self.representation_model is None:
            if self.representation_model_type == 'identity':
                self.representation_network = nn.Identity()
            elif self.representation_model_type == 'conv_res_blocks':
                self.representation_network = RepresentationNetwork(
                    observation_shape,
                    num_res_blocks,
                    num_channels,
                    downsample,
                    momentum=batch_norm_momentum,
                    activation=activation,
                )
            # elif
        else:
            self.representation_network = self.representation_model

    def forward(self, encoded_state):
        encoded_state = self.representation_network(encoded_state)
        logit, value = self.prediction_network(encoded_state)
        return logit, value

    def compute_prob_value(self, state_batch):
        logits, values = self.forward(state_batch)
        dist = torch.distributions.Categorical(logits=logits)
        probs = dist.probs
        return probs, values

    def compute_logp_value(self, state_batch):
        logits, values = self.forward(state_batch)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs, values

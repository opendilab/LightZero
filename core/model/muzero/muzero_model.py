"""
The following code is adapted from https://github.com/YeWR/EfficientZero/blob/main/config/atari/model.py
"""

import math

import numpy as np
import torch
import torch.nn as nn
from ding.torch_utils import MLP, ResBlock
from ding.utils import MODEL_REGISTRY

from .muzero_base_model import BaseNet, renormalize


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


class DownSample(nn.Module):

    def __init__(self, in_channels, out_channels, momentum=0.1):
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
                    in_channels=out_channels // 2,
                    activation=torch.nn.ReLU(inplace=True),
                    norm_type='BN',
                    res_type='basic',
                    bias=False
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
            activation=torch.nn.ReLU(inplace=True),
            norm_type='BN',
            res_type='downsample',
            bias=False
        )
        self.resblocks2 = nn.ModuleList(
            [
                ResBlock(
                    in_channels=out_channels,
                    activation=torch.nn.ReLU(inplace=True),
                    norm_type='BN',
                    res_type='basic',
                    bias=False
                ) for _ in range(1)
            ]
        )
        self.pooling1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.resblocks3 = nn.ModuleList(
            [
                ResBlock(
                    in_channels=out_channels,
                    activation=torch.nn.ReLU(inplace=True),
                    norm_type='BN',
                    res_type='basic',
                    bias=False
                ) for _ in range(1)
            ]
        )
        self.pooling2 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.activation = nn.ReLU(inplace=True)

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
            num_blocks,
            num_channels,
            downsample,
            momentum=0.1,
    ):
        """
        Overview: Representation network
        Arguments:
            - observation_shape (:obj:`Union[List, tuple]`):  shape of observations: [C, W, H]
            - num_blocks (:obj:`int`): number of res blocks
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
                ResBlock(
                    in_channels=num_channels,
                    activation=torch.nn.ReLU(inplace=True),
                    norm_type='BN',
                    res_type='basic',
                    bias=False
                ) for _ in range(num_blocks)
            ]
        )
        self.activation = nn.ReLU(inplace=True)

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


# Predict next hidden states given current states and actions
class DynamicsNetwork(nn.Module):

    def __init__(
            self,
            num_blocks,
            num_channels,
            reduced_channels_reward,
            fc_reward_layers,
            full_support_size,
            block_output_size_reward,
            momentum=0.1,
            last_linear_layer_init_zero=False,
    ):
        """
        Overview:
            Dynamics network
        Arguments:
            - num_blocks (:obj:int): number of res blocks
            - num_channels (:obj:int): channels of hidden states
            - fc_reward_layers (:obj:list):  hidden layers of the reward prediction head (MLP head)
            - full_support_size (:obj:int): dim of reward output
            - block_output_size_reward (:obj:int): dim of flatten hidden states
            - last_linear_layer_init_zero (:obj:bool): if True -> zero initialization for the last layer of reward mlp
        """
        super().__init__()
        self.num_channels = num_channels

        self.conv = nn.Conv2d(num_channels, num_channels - 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(num_channels - 1, momentum=momentum)
        self.resblocks = nn.ModuleList(
            [
                ResBlock(
                    in_channels=num_channels - 1,
                    activation=torch.nn.ReLU(inplace=True),
                    norm_type='BN',
                    res_type='basic',
                    bias=False
                ) for _ in range(num_blocks)
            ]
        )

        self.conv1x1_reward = nn.Conv2d(num_channels - 1, reduced_channels_reward, 1)
        self.bn_reward = nn.BatchNorm2d(reduced_channels_reward, momentum=momentum)
        self.block_output_size_reward = block_output_size_reward
        # TODO(pu)
        self.fc = MLP(
            self.block_output_size_reward,
            hidden_channels=fc_reward_layers[0],
            layer_num=len(fc_reward_layers) + 1,
            out_channels=full_support_size,
            activation=nn.ReLU(inplace=True),
            norm_type='BN',
            output_activation=nn.Identity(),
            output_norm_type=None,
            last_linear_layer_init_zero=last_linear_layer_init_zero
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        # take the state encoding,  x[:, -1, :, :] is action encoding
        state = x[:, :-1, :, :]
        x = self.conv(x)
        x = self.bn(x)

        x += state
        x = self.activation(x)

        for block in self.resblocks:
            x = block(x)
        state = x

        x = self.conv1x1_reward(x)
        x = self.bn_reward(x)
        x = self.activation(x)

        x = x.view(-1, self.block_output_size_reward)
        reward = self.fc(x)

        return state, reward

    def get_dynamic_mean(self):
        dynamic_mean = np.abs(self.conv.weight.detach().cpu().numpy().reshape(-1)).tolist()

        for block in self.resblocks:
            for name, param in block.named_parameters():
                dynamic_mean += np.abs(param.detach().cpu().numpy().reshape(-1)).tolist()
        dynamic_mean = sum(dynamic_mean) / len(dynamic_mean)
        return dynamic_mean

    def get_reward_mean(self):
        reward_w_dist = self.conv1x1_reward.weight.detach().cpu().numpy().reshape(-1)

        for name, param in self.fc.named_parameters():
            temp_weights = param.detach().cpu().numpy().reshape(-1)
            reward_w_dist = np.concatenate((reward_w_dist, temp_weights))
        reward_mean = np.abs(reward_w_dist).mean()
        return reward_w_dist, reward_mean


# predict the value and policy given hidden states
class PredictionNetwork(nn.Module):

    def __init__(
            self,
            action_space_size,
            num_blocks,
            in_channels,
            num_channels,
            reduced_channels_value,
            reduced_channels_policy,
            fc_value_layers,
            fc_policy_layers,
            full_support_size,
            block_output_size_value,
            block_output_size_policy,
            momentum=0.1,
            last_linear_layer_init_zero=False,
    ):
        """Prediction network
        Parameters
        ----------
        action_space_size: int
            action space
        num_blocks: int
            number of res blocks
        in_channels: int
            channels of input
        num_channels: int
            channels of hidden states
        reduced_channels_value: int
            channels of value head
        reduced_channels_policy: int
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
                ResBlock(
                    in_channels=num_channels,
                    activation=torch.nn.ReLU(inplace=True),
                    norm_type='BN',
                    res_type='basic',
                    bias=False
                ) for _ in range(num_blocks)
            ]
        )

        self.conv1x1_value = nn.Conv2d(num_channels, reduced_channels_value, 1)
        self.conv1x1_policy = nn.Conv2d(num_channels, reduced_channels_policy, 1)
        self.bn_value = nn.BatchNorm2d(reduced_channels_value, momentum=momentum)
        self.bn_policy = nn.BatchNorm2d(reduced_channels_policy, momentum=momentum)
        self.block_output_size_value = block_output_size_value
        self.block_output_size_policy = block_output_size_policy
        # TODO(pu)
        self.fc_value = MLP(
            in_channels=self.block_output_size_value,
            hidden_channels=fc_value_layers[0],
            out_channels=full_support_size,
            layer_num=len(fc_value_layers) + 1,
            activation=nn.ReLU(inplace=True),
            norm_type='BN',
            output_activation=nn.Identity(),
            output_norm_type=None,
            last_linear_layer_init_zero=last_linear_layer_init_zero
        )
        self.fc_policy = MLP(
            in_channels=self.block_output_size_policy,
            hidden_channels=fc_policy_layers[0],
            out_channels=action_space_size,
            layer_num=len(fc_policy_layers) + 1,
            activation=nn.ReLU(inplace=True),
            norm_type='BN',
            output_activation=nn.Identity(),
            output_norm_type=None,
            last_linear_layer_init_zero=last_linear_layer_init_zero
        )

        self.activation = nn.ReLU(inplace=True)

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

        value = value.reshape(-1,  self.block_output_size_value)
        policy = policy.reshape(-1, self.block_output_size_policy)

        value = self.fc_value(value)
        policy = self.fc_policy(policy)
        return policy, value


@MODEL_REGISTRY.register('MuZeroNet')
class MuZeroNet(BaseNet):

    def __init__(
            self,
            observation_shape,
            action_space_size,
            num_blocks,
            num_channels,
            reduced_channels_reward,
            reduced_channels_value,
            reduced_channels_policy,
            fc_reward_layers,
            fc_value_layers,
            fc_policy_layers,
            reward_support_size,
            value_support_size,
            downsample,
            representation_model_type: str = 'conv_res_blocks',
            representation_model: nn.Module = None,
            bn_mt=0.1,
            proj_hid=256,
            proj_out=256,
            pred_hid=64,
            pred_out=256,
            last_linear_layer_init_zero=False,
            state_norm=False,
            categorical_distribution=True,
    ):
        """
        Overview:
            EfficientZero network
        Arguments:
            - representation_model_type
            - observation_shape: tuple or list. shape of observations: [C, W, H]
            - action_space_size: (:obj:`int`): . action space
            - num_blocks (:obj:`int`):  number of res blocks
            - num_channels (:obj:`int`): channels of hidden states
            - reduced_channels_reward (:obj:`int`): channels of reward head
            - reduced_channels_value (:obj:`int`): channels of value head
            - reduced_channels_policy (:obj:`int`): channels of policy head
            - fc_reward_layers (:obj:`list`):  hidden layers of the reward prediction head (MLP head)
            - fc_value_layers (:obj:`list`):  hidden layers of the value prediction head (MLP head)
            - fc_policy_layers (:obj:`list`):  hidden layers of the policy prediction head (MLP head)
            - reward_support_size (:obj:`int`): dim of reward output
            - value_support_size (:obj:`int`): dim of value output
            - downsample (:obj:`bool`): True -> do downsampling for observations. (For board games, do not need)
            - bn_mt (:obj:`float`):  Momentum of BN
            - proj_hid (:obj:`int`): dim of projection hidden layer
            - proj_out (:obj:`int`): dim of projection output layer
            - pred_hid (:obj:`int`):dim of projection head (prediction) hidden layer
            - pred_out (:obj:`int`): dim of projection head (prediction) output layer
            - last_linear_layer_init_zero (:obj:`bool`): True -> zero initialization for the last layer of value/policy mlp
            - state_norm (:obj:`bool`):  True -> normalization for hidden states
            - categorical_distribution (:obj:`bool`): whether to use discrete support to represent categorical distribution for value, reward/reward
        """
        super(MuZeroNet, self).__init__()
        self.categorical_distribution = categorical_distribution
        if not self.categorical_distribution:
            self.reward_support_size = 1
            self.value_support_size = 1
        else:
            self.reward_support_size = reward_support_size
            self.value_support_size = value_support_size

        self.proj_hid = proj_hid
        self.proj_out = proj_out
        self.pred_hid = pred_hid
        self.pred_out = pred_out
        self.last_linear_layer_init_zero = last_linear_layer_init_zero
        self.state_norm = state_norm
        self.representation_model_type = representation_model_type
        self.representation_model = representation_model
        self.downsample = downsample

        self.action_space_size = action_space_size
        block_output_size_reward = (
            (reduced_channels_reward * math.ceil(observation_shape[1] / 16) *
             math.ceil(observation_shape[2] / 16)) if downsample else
            (reduced_channels_reward * observation_shape[1] * observation_shape[2])
        )

        block_output_size_value = (
            (reduced_channels_value * math.ceil(observation_shape[1] / 16) *
             math.ceil(observation_shape[2] / 16)) if downsample else
            (reduced_channels_value * observation_shape[1] * observation_shape[2])
        )

        block_output_size_policy = (
            (reduced_channels_policy * math.ceil(observation_shape[1] / 16) *
             math.ceil(observation_shape[2] / 16)) if downsample else
            (reduced_channels_policy * observation_shape[1] * observation_shape[2])
        )

        if self.representation_model is None:
            if self.representation_model_type == 'identity':
                self.representation_network = nn.Identity()
            elif self.representation_model_type == 'conv_res_blocks':
                self.representation_network = RepresentationNetwork(
                    observation_shape,
                    num_blocks,
                    num_channels,
                    downsample,
                    momentum=bn_mt,
                )
        else:
            self.representation_network = self.representation_model

        if self.representation_model_type == 'identity':
            self.dynamics_network = DynamicsNetwork(
                num_blocks,
                observation_shape[0] + 1,  # in_channels=observation_shape[0]
                reduced_channels_reward,
                fc_reward_layers,
                self.reward_support_size,
                block_output_size_reward,
                momentum=bn_mt,
                last_linear_layer_init_zero=self.last_linear_layer_init_zero,
            )
            self.prediction_network = PredictionNetwork(
                action_space_size,
                num_blocks,
                observation_shape[0],  # in_channels
                num_channels,
                reduced_channels_value,
                reduced_channels_policy,
                fc_value_layers,
                fc_policy_layers,
                self.value_support_size,
                block_output_size_value,
                block_output_size_policy,
                momentum=bn_mt,
                last_linear_layer_init_zero=self.last_linear_layer_init_zero,
            )
        else:
            self.dynamics_network = DynamicsNetwork(
                num_blocks,
                num_channels + 1,
                reduced_channels_reward,
                fc_reward_layers,
                self.reward_support_size,
                block_output_size_reward,
                momentum=bn_mt,
                last_linear_layer_init_zero=self.last_linear_layer_init_zero,
            )
            self.prediction_network = PredictionNetwork(
                action_space_size,
                num_blocks,
                None,  # in_channels
                num_channels,
                reduced_channels_value,
                reduced_channels_policy,
                fc_value_layers,
                fc_policy_layers,
                self.value_support_size,
                block_output_size_value,
                block_output_size_policy,
                momentum=bn_mt,
                last_linear_layer_init_zero=self.last_linear_layer_init_zero,
            )

    def prediction(self, encoded_state):
        policy, value = self.prediction_network(encoded_state)
        return policy, value

    def representation(self, observation):
        encoded_state = self.representation_network(observation)
        if not self.state_norm:
            return encoded_state
        else:
            encoded_state_normalized = renormalize(encoded_state)
            return encoded_state_normalized

    def dynamics(self, encoded_state, action):
        # Stack encoded_state with a game specific one hot encoded action
        action_one_hot = (
            torch.ones((
                encoded_state.shape[0],
                1,
                encoded_state.shape[2],
                encoded_state.shape[3],
            )).to(action.device).float()
        )
        action_one_hot = (action[:, :, None, None] * action_one_hot / self.action_space_size)

        x = torch.cat((encoded_state, action_one_hot), dim=1)
        next_encoded_state, reward = self.dynamics_network(x)
        if not self.state_norm:
            return next_encoded_state, reward
        else:
            next_encoded_state_normalized = renormalize(next_encoded_state)
            return next_encoded_state_normalized,reward

    def get_params_mean(self):
        representation_mean = self.representation_network.get_param_mean()
        dynamic_mean = self.dynamics_network.get_dynamic_mean()
        reward_w_dist, reward_mean = self.dynamics_network.get_reward_mean()

        return reward_w_dist, representation_mean, dynamic_mean, reward_mean


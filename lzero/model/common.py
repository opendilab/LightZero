"""
Acknowledgement: The following code is adapted from https://github.com/YeWR/EfficientZero/core/model.py
"""
import torch
from typing import List
from dataclasses import dataclass
import torch.nn as nn
from ding.torch_utils import MLP, ResBlock
import numpy as np


@dataclass
class EZNetworkOutput:
    # output format of the model
    value: float
    value_prefix: float
    policy_logits: List[float]
    hidden_state: List[float]
    reward_hidden_state: object


@dataclass
class MZNetworkOutput:
    # output format of the model
    value: float
    reward: float
    policy_logits: List[float]
    hidden_state: List[float]


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


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
        norm_type: str = 'BN',
    ):
        """
        Overview:
            Representation network
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
                ResBlock(
                    in_channels=num_channels, activation=activation, norm_type=norm_type, res_type='basic', bias=False
                ) for _ in range(num_res_blocks)
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

    def get_param_mean(self):
        mean = []
        for name, param in self.named_parameters():
            mean += np.abs(param.detach().cpu().numpy().reshape(-1)).tolist()
        mean = sum(mean) / len(mean)
        return mean
"""
Copyright 2020 Sensetime X-lab. All Rights Reserved

Main Function:
    1. build ResBlock: you can use this classes to build residual blocks
"""
import torch.nn as nn
from .nn_module import conv2d_block, fc_block,conv2d_block2,fc_block2
from .activation import build_activation
from .normalization import build_normalization


class ResBlock(nn.Module):
    r'''
    Overview:
        Residual Block with 2D convolution layers, including 2 types:
            basic block:
                input channel: C
                x -> 3*3*C -> norm -> act -> 3*3*C -> norm -> act -> out
                \__________________________________________/+
            bottleneck block:
                x -> 1*1*(1/4*C) -> norm -> act -> 3*3*(1/4*C) -> norm -> act -> 1*1*C -> norm -> act -> out
                \_____________________________________________________________________________/+

    Interface:
        __init__, forward
    '''

    def __init__(self, in_channels, out_channels=None,stride=1, downsample=None, activation='relu', norm_type='LN',):
        r"""
        Overview:
            Init the Residual Block

        Arguments:
            - in_channels (:obj:`int`): Number of channels in the input tensor
            - activation (:obj:`nn.Module`): the optional activation function
            - norm_type (:obj:`str`): type of the normalization, defalut set to batch normalization,
                                      support ['BN', 'IN', 'SyncBN', None]
            - res_type (:obj:`str`): type of residual block, support ['basic', 'bottleneck'], see overview for details
        """
        super(ResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = self.in_channels if out_channels is None else out_channels
        self.activation_type = activation
        self.norm_type = norm_type
        self.stride = stride
        self.downsample = downsample
        self.conv1 = conv2d_block(in_channels=self.in_channels,
                                  out_channels=self.out_channels,
                                  kernel_size=3,
                                  stride=self.stride,
                                  padding= 1,
                                  activation=self.activation_type,
                                  norm_type=self.norm_type)
        self.conv2 = conv2d_block(in_channels=self.out_channels,
                                  out_channels=self.out_channels,
                                  kernel_size=3,
                                  stride=self.stride,
                                  padding= 1,
                                  activation=None,
                                  norm_type=self.norm_type)
        self.activation = build_activation(self.activation_type)

    def forward(self, x):
        r"""
        Overview:
            return the redisual block output

        Arguments:
            - x (:obj:`tensor`): the input tensor

        Returns:
            - x(:obj:`tensor`): the resblock output tensor
        """
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.activation(out)
        return out


class ResBlock2(nn.Module):
    r'''
    Overview:
        Residual Block with 2D convolution layers, including 2 types:
            basic block:
                input channel: C
                x -> 3*3*C -> norm -> act -> 3*3*C -> norm -> act -> out
                \__________________________________________/+
            bottleneck block:
                x -> 1*1*(1/4*C) -> norm -> act -> 3*3*(1/4*C) -> norm -> act -> 1*1*C -> norm -> act -> out
                \_____________________________________________________________________________/+

    Interface:
        __init__, forward
    '''

    def __init__(self, in_channels, out_channels=None,stride=1, downsample=None, activation='relu', norm_type='LN',):
        r"""
        Overview:
            Init the Residual Block

        Arguments:
            - in_channels (:obj:`int`): Number of channels in the input tensor
            - activation (:obj:`nn.Module`): the optional activation function
            - norm_type (:obj:`str`): type of the normalization, defalut set to batch normalization,
                                      support ['BN', 'IN', 'SyncBN', None]
            - res_type (:obj:`str`): type of residual block, support ['basic', 'bottleneck'], see overview for details
        """
        super(ResBlock2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = self.in_channels if out_channels is None else out_channels
        self.activation_type = activation
        self.norm_type = norm_type
        self.stride = stride
        self.downsample = downsample
        self.conv1 = conv2d_block2(in_channels=self.in_channels,
                                  out_channels=self.out_channels,
                                  kernel_size=3,
                                  stride=self.stride,
                                  padding= 1,
                                  activation=self.activation_type,
                                  norm_type=self.norm_type)
        self.conv2 = conv2d_block2(in_channels=self.out_channels,
                                  out_channels=self.out_channels,
                                  kernel_size=3,
                                  stride=self.stride,
                                  padding= 1,
                                  activation=self.activation_type,
                                  norm_type=self.norm_type)
        self.activation = build_activation(self.activation_type)


    def forward(self, x):
        r"""
        Overview:
            return the redisual block output

        Arguments:
            - x (:obj:`tensor`): the input tensor

        Returns:
            - x(:obj:`tensor`): the resblock output tensor
        """
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return x

class ResFCBlock(nn.Module):
    def __init__(self, in_channels, activation='relu', norm_type=None):
        r"""
        Overview:
            Init the Residual Block

        Arguments:
            - activation (:obj:`nn.Module`): the optional activation function
            - norm_type (:obj:`str`): type of the normalization, defalut set to batch normalization
        """
        super(ResFCBlock, self).__init__()
        self.activation_type = activation
        self.norm_type = norm_type
        self.fc1 = fc_block(in_channels, in_channels, norm_type=self.norm_type, activation=self.activation_type)
        self.fc2 = fc_block(in_channels, in_channels,norm_type=self.norm_type,  activation=None)
        self.activation = build_activation(self.activation_type)


    def forward(self, x):
        r"""
        Overview:
            return  output of  the residual block with 2 fully connected block

        Arguments:
            - x (:obj:`tensor`): the input tensor

        Returns:
            - x(:obj:`tensor`): the resblock output tensor
        """
        residual = x
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.activation(x + residual)
        return x

class ResFCBlock2(nn.Module):
    r'''
    Overview:
        Residual Block with 2 fully connected block
        x -> fc1 -> norm -> act -> fc2 -> norm -> act -> out
        \_____________________________________/+

    Interface:
        __init__, forward
    '''

    def __init__(self, in_channels, activation='relu', norm_type='LN'):
        r"""
        Overview:
            Init the Residual Block

        Arguments:
            - activation (:obj:`nn.Module`): the optional activation function
            - norm_type (:obj:`str`): type of the normalization, defalut set to batch normalization
        """
        super(ResFCBlock2, self).__init__()
        self.activation_type = activation
        self.fc1 = fc_block2(in_channels, in_channels, activation=self.activation_type, norm_type=norm_type)
        self.fc2 = fc_block2(in_channels, in_channels, activation=self.activation_type, norm_type=norm_type)

    def forward(self, x):
        r"""
        Overview:
            return  output of  the residual block with 2 fully connected block

        Arguments:
            - x (:obj:`tensor`): the input tensor

        Returns:
            - x(:obj:`tensor`): the resblock output tensor
        """
        residual = x
        x = self.fc1(x)
        x = self.fc2(x)
        x = x + residual
        return x
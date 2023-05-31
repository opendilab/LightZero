from typing import Callable

import torch
import torch.nn as nn

from .activation import build_activation
from .normalization import build_normalization


def fc_block(
        in_channels: int,
        out_channels: int,
        activation: nn.Module = None,
        norm_type: str = None,
        use_dropout: bool = False,
        dropout_probability: float = 0.5
) -> nn.Sequential:
    r"""
    Overview:
        Create a fully-connected block with activation, normalization and dropout.
        Optional normalization can be done to the dim 1 (across the channels)
        x -> fc -> norm -> act -> dropout -> out
    Arguments:
        - in_channels (:obj:`int`): Number of channels in the input tensor
        - out_channels (:obj:`int`): Number of channels in the output tensor
        - activation (:obj:`nn.Module`): the optional activation function
        - norm_type (:obj:`str`): type of the normalization
        - use_dropout (:obj:`bool`) : whether to use dropout in the fully-connected block
        - dropout_probability (:obj:`float`) : probability of an element to be zeroed in the dropout. Default: 0.5
    Returns:
        - block (:obj:`nn.Sequential`): a sequential list containing the torch layers of the fully-connected block

    .. note::

        you can refer to nn.linear (https://pytorch.org/docs/master/generated/torch.nn.Linear.html)
    """
    block = []
    block.append(nn.Linear(in_channels, out_channels))
    if norm_type is not None and norm_type != 'none':
        block.append(build_normalization(norm_type, dim=1)(out_channels))
    if isinstance(activation, str) and activation != 'none':
        block.append(build_activation(activation))
    elif isinstance(activation, torch.nn.Module):
        block.append(activation)
    if use_dropout:
        block.append(nn.Dropout(dropout_probability))
    return nn.Sequential(*block)


def fc_block2(
        in_channels,
        out_channels,
        activation=None,
        norm_type=None,
        use_dropout=False,
        dropout_probability=0.5
):
    r"""
    Overview:
        create a fully-connected block with activation, normalization and dropout
        optional normalization can be done to the dim 1 (across the channels)
        x -> fc -> norm -> act -> dropout -> out
    Arguments:
        - in_channels (:obj:`int`): Number of channels in the input tensor
        - out_channels (:obj:`int`): Number of channels in the output tensor
        - init_type (:obj:`str`): the type of init to implement
        - activation (:obj:`nn.Moduel`): the optional activation function
        - norm_type (:obj:`str`): type of the normalization
        - use_dropout (:obj:`bool`) : whether to use dropout in the fully-connected block
        - dropout_probability (:obj:`float`) : probability of an element to be zeroed in the dropout. Default: 0.5
    Returns:
        - block (:obj:`nn.Sequential`): a sequential list containing the torch layers of the fully-connected block

    .. note::
        you can refer to nn.linear (https://pytorch.org/docs/master/generated/torch.nn.Linear.html)
    """
    block = []
    if norm_type is not None and norm_type != 'none':
        block.append(build_normalization(norm_type, dim=1)(in_channels))
    if isinstance(activation, str) and activation != 'none':
        block.append(build_activation(activation))
    elif isinstance(activation, torch.nn.Module):
        block.append(activation)
    block.append(nn.Linear(in_channels, out_channels))
    if use_dropout:
        block.append(nn.Dropout(dropout_probability))
    return nn.Sequential(*block)


def conv2d_block(
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        activation: str = None,
        norm_type: str = None,
        bias: bool = True,
) -> nn.Sequential:
    r"""
    Overview:
        Create a 2-dim convlution layer with activation and normalization.
    Arguments:
        - in_channels (:obj:`int`): Number of channels in the input tensor
        - out_channels (:obj:`int`): Number of channels in the output tensor
        - kernel_size (:obj:`int`): Size of the convolving kernel
        - stride (:obj:`int`): Stride of the convolution
        - padding (:obj:`int`): Zero-padding added to both sides of the input
        - dilation (:obj:`int`): Spacing between kernel elements
        - groups (:obj:`int`): Number of blocked connections from input channels to output channels
        - pad_type (:obj:`str`): the way to add padding, include ['zero', 'reflect', 'replicate'], default: None
        - activation (:obj:`nn.Module`): the optional activation function
        - norm_type (:obj:`str`): type of the normalization, default set to None, now support ['BN', 'IN', 'SyncBN']
    Returns:
        - block (:obj:`nn.Sequential`): a sequential list containing the torch layers of the 2 dim convlution layer

    .. note::

        Conv2d (https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d)
    """
    block = []
    block.append(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, groups=groups,bias=bias)
    )
    if norm_type is not None:
        block.append(nn.GroupNorm(num_groups=1,  num_channels=out_channels))
    if isinstance(activation, str) and activation != 'none':
        block.append(build_activation(activation))
    elif isinstance(activation, torch.nn.Module):
        block.append(activation)
    return nn.Sequential(*block)


def conv2d_block2(
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        activation: str = None,
        norm_type=None,
        bias: bool = True,
):
    r"""
    Overview:
        create a 2-dim convlution layer with activation and normalization.

        Note:
            Conv2d (https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d)

    Arguments:
        - in_channels (:obj:`int`): Number of channels in the input tensor
        - out_channels (:obj:`int`): Number of channels in the output tensor
        - kernel_size (:obj:`int`): Size of the convolving kernel
        - stride (:obj:`int`): Stride of the convolution
        - padding (:obj:`int`): Zero-padding added to both sides of the input
        - dilation (:obj:`int`): Spacing between kernel elements
        - groups (:obj:`int`): Number of blocked connections from input channels to output channels
        - init_type (:obj:`str`): the type of init to implement
        - pad_type (:obj:`str`): the way to add padding, include ['zero', 'reflect', 'replicate'], default: None
        - activation (:obj:`nn.Moduel`): the optional activation function
        - norm_type (:obj:`str`): type of the normalization, default set to None, now support ['BN', 'IN', 'SyncBN']

    Returns:
        - block (:obj:`nn.Sequential`): a sequential list containing the torch layers of the 2 dim convlution layer
    """

    block = []
    if norm_type is not None:
        block.append(nn.GroupNorm(num_groups=1,  num_channels=out_channels))
    if isinstance(activation, str) and activation != 'none':
        block.append(build_activation(activation))
    elif isinstance(activation, torch.nn.Module):
        block.append(activation)
    block.append(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, groups=groups,bias=bias)
    )
    return nn.Sequential(*block)


def MLP(
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        layer_num: int,
        layer_fn: Callable = None,
        activation: str = None,
        norm_type: str = None,
        use_dropout: bool = False,
        dropout_probability: float = 0.5
):
    r"""
    Overview:
        create a multi-layer perceptron using fully-connected blocks with activation, normalization and dropout,
        optional normalization can be done to the dim 1 (across the channels)
        x -> fc -> norm -> act -> dropout -> out
    Arguments:
        - in_channels (:obj:`int`): Number of channels in the input tensor
        - hidden_channels (:obj:`int`): Number of channels in the hidden tensor
        - out_channels (:obj:`int`): Number of channels in the output tensor
        - layer_num (:obj:`int`): Number of layers
        - layer_fn (:obj:`Callable`): layer function
        - activation (:obj:`nn.Module`): the optional activation function
        - norm_type (:obj:`str`): type of the normalization
        - use_dropout (:obj:`bool`): whether to use dropout in the fully-connected block
        - dropout_probability (:obj:`float`): probability of an element to be zeroed in the dropout. Default: 0.5
    Returns:
        - block (:obj:`nn.Sequential`): a sequential list containing the torch layers of the fully-connected block

    .. note::

        you can refer to nn.linear (https://pytorch.org/docs/master/generated/torch.nn.Linear.html)
    """
    assert layer_num >= 0, layer_num
    if layer_num == 0:
        return nn.Sequential(*[nn.Identity()])

    channels = [in_channels] + [hidden_channels] * (layer_num - 1) + [out_channels]
    if layer_fn is None:
        layer_fn = fc_block
    block = []
    for i, (in_channels, out_channels) in enumerate(zip(channels[:-1], channels[1:])):
        block.append(layer_fn(in_channels=in_channels,
                              out_channels=out_channels,
                              activation=activation,
                              norm_type=norm_type,
                              use_dropout=use_dropout,
                              dropout_probability=dropout_probability))
    return nn.Sequential(*block)

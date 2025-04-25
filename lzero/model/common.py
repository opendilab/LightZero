"""
Overview:
    In this Python file, we provide a collection of reusable model templates designed to streamline the development
    process for various custom algorithms. By utilizing these pre-built model templates, users can quickly adapt and
    customize their custom algorithms, ensuring efficient and effective development.
    BTW, users can refer to the unittest of these model templates to learn how to use them.
"""
import math
from dataclasses import dataclass
from typing import Callable, List, Optional
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from ding.torch_utils import MLP, ResBlock
from ding.torch_utils.network.normalization import build_normalization
from ding.utils import SequenceType
from ditk import logging
from ding.utils import set_pkg_seed, get_rank, get_world_size
import torch

def MLP_V2(
        in_channels: int,
        hidden_channels: List[int],
        out_channels: int,
        layer_fn: Callable = None,
        activation: Optional[nn.Module] = None,
        norm_type: Optional[str] = None,
        use_dropout: bool = False,
        dropout_probability: float = 0.5,
        output_activation: bool = True,
        output_norm: bool = True,
        last_linear_layer_init_zero: bool = False,
):
    """
    Overview:
        Create a multi-layer perceptron (MLP) using a list of hidden dimensions. Each layer consists of a fully
        connected block with optional activation, normalization, and dropout. The final layer is configurable
        to include or exclude activation, normalization, and dropout based on user preferences.

    Arguments:
        - in_channels (:obj:`int`): Number of input channels (dimensionality of the input tensor).
        - hidden_channels (:obj:`List[int]`): A list specifying the number of channels for each hidden layer.
            For example, [512, 256, 128] means the MLP will have three hidden layers with 512, 256, and 128 units, respectively.
        - out_channels (:obj:`int`): Number of output channels (dimensionality of the output tensor).
        - layer_fn (:obj:`Callable`, optional): Layer function to construct layers (default is `nn.Linear`).
        - activation (:obj:`nn.Module`, optional): Activation function to use after each layer
            (e.g., `nn.ReLU`, `nn.Sigmoid`). Default is None (no activation).
        - norm_type (:obj:`str`, optional): Type of normalization to apply after each layer.
            If None, no normalization is applied. Supported values depend on the implementation of `build_normalization`.
        - use_dropout (:obj:`bool`, optional): Whether to apply dropout after each layer. Default is False.
        - dropout_probability (:obj:`float`, optional): The probability of setting elements to zero in dropout. Default is 0.5.
        - output_activation (:obj:`bool`, optional): Whether to apply activation to the output layer. Default is True.
        - output_norm (:obj:`bool`, optional): Whether to apply normalization to the output layer. Default is True.
        - last_linear_layer_init_zero (:obj:`bool`, optional): Whether to initialize the weights and biases of the
            last linear layer to zeros. This is commonly used in reinforcement learning for stable initial outputs.

    Returns:
        - block (:obj:`nn.Sequential`): A PyTorch `nn.Sequential` object containing the layers of the MLP.

    Notes:
        - The final layer's normalization, activation, and dropout are controlled by `output_activation`,
          `output_norm`, and `use_dropout`.
        - If `last_linear_layer_init_zero` is True, the weights and biases of the last linear layer are initialized to 0.
    """
    assert len(hidden_channels) > 0, "The hidden_channels list must contain at least one element."
    if layer_fn is None:
        layer_fn = nn.Linear

    # Initialize the MLP block
    block = []
    channels = [in_channels] + hidden_channels + [out_channels]

    # Build all layers except the final layer
    for i, (in_channels, out_channels) in enumerate(zip(channels[:-2], channels[1:-1])):
        block.append(layer_fn(in_channels, out_channels))
        if norm_type is not None:
            block.append(build_normalization(norm_type, dim=1)(out_channels))
        if activation is not None:
            block.append(activation)
        if use_dropout:
            block.append(nn.Dropout(dropout_probability))

    # Build the final layer
    in_channels = channels[-2]
    out_channels = channels[-1]
    block.append(layer_fn(in_channels, out_channels))

    # Add optional normalization and activation for the final layer
    if output_norm and norm_type is not None:
        block.append(build_normalization(norm_type, dim=1)(out_channels))
    if output_activation and activation is not None:
        block.append(activation)
    if use_dropout:
        block.append(nn.Dropout(dropout_probability))

    # Initialize the weights and biases of the last linear layer to zero if specified
    if last_linear_layer_init_zero:
        for layer in reversed(block):
            if isinstance(layer, nn.Linear):
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)
                break

    return nn.Sequential(*block)

# use dataclass to make the output of network more convenient to use
@dataclass
class MZRNNNetworkOutput:
    # output format of the MuZeroRNN model
    value: torch.Tensor
    value_prefix: torch.Tensor
    policy_logits: torch.Tensor
    latent_state: torch.Tensor
    predict_next_latent_state: torch.Tensor
    reward_hidden_state: Tuple[torch.Tensor]


@dataclass
class EZNetworkOutput:
    # output format of the EfficientZero model
    value: torch.Tensor
    value_prefix: torch.Tensor
    policy_logits: torch.Tensor
    latent_state: torch.Tensor
    reward_hidden_state: Tuple[torch.Tensor]


@dataclass
class MZNetworkOutput:
    # output format of the MuZero model
    value: torch.Tensor
    reward: torch.Tensor
    policy_logits: torch.Tensor
    latent_state: torch.Tensor


class SimNorm(nn.Module):

    def __init__(self, simnorm_dim: int) -> None:
        """
        Overview:
            Simplicial normalization. Adapted from https://arxiv.org/abs/2204.00616.
        Arguments:
            - simnorm_dim (:obj:`int`): The dimension for simplicial normalization.
        """
        super().__init__()
        self.dim = simnorm_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Overview:
            Forward pass of the SimNorm layer.
        Arguments:
            - x (:obj:`torch.Tensor`): The input tensor to normalize.
        Returns:
            - x (:obj:`torch.Tensor`): The normalized tensor.
        """
        shp = x.shape
        # Ensure that there is at least one simplex to normalize across.
        if shp[1] != 0:
            x = x.view(*shp[:-1], -1, self.dim)
            x = F.softmax(x, dim=-1)
            return x.view(*shp)
        else:
            return x

    def __repr__(self) -> str:
        """
        Overview:
            String representation of the SimNorm layer.
        Returns:
            - output (:obj:`str`): The string representation.
        """
        return f"SimNorm(dim={self.dim})"


def AvgL1Norm(x, eps=1e-8):
    """
    Overview:
        Normalize the input tensor by the L1 norm.
    Arguments:
        - x (:obj:`torch.Tensor`): The input tensor to normalize.
        - eps (:obj:`float`): The epsilon value to prevent division by zero.
    Returns:
        - :obj:`torch.Tensor`: The normalized tensor.
    """
    return x / x.abs().mean(-1, keepdim=True).clamp(min=eps)


class FeatureAndGradientHook:

    def __init__(self):
        """
        Overview:
            Class to capture features and gradients at SimNorm.
        """
        self.features_before = []
        self.features_after = []
        self.grads_before = []
        self.grads_after = []

    def setup_hooks(self, model):
        # Hooks to capture features and gradients at SimNorm
        self.forward_handler = model.sim_norm.register_forward_hook(self.forward_hook)
        self.backward_handler = model.sim_norm.register_full_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        with torch.no_grad():
            self.features_before.append(input[0])
            self.features_after.append(output)

    def backward_hook(self, module, grad_input, grad_output):
        with torch.no_grad():
            self.grads_before.append(grad_input[0] if grad_input[0] is not None else None)
            self.grads_after.append(grad_output[0] if grad_output[0] is not None else None)

    def analyze(self):
        # Calculate L2 norms of features
        l2_norm_before = torch.mean(torch.stack([torch.norm(f, p=2, dim=1).mean() for f in self.features_before]))
        l2_norm_after = torch.mean(torch.stack([torch.norm(f, p=2, dim=1).mean() for f in self.features_after]))

        # Calculate norms of gradients
        grad_norm_before = torch.mean(
            torch.stack([torch.norm(g, p=2, dim=1).mean() for g in self.grads_before if g is not None]))
        grad_norm_after = torch.mean(
            torch.stack([torch.norm(g, p=2, dim=1).mean() for g in self.grads_after if g is not None]))

        # Clear stored data and delete tensors to free memory
        self.clear_data()

        # Optionally clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return l2_norm_before, l2_norm_after, grad_norm_before, grad_norm_after

    def clear_data(self):
        del self.features_before[:]
        del self.features_after[:]
        del self.grads_before[:]
        del self.grads_after[:]

    def remove_hooks(self):
        self.forward_handler.remove()
        self.backward_handler.remove()


class DownSample(nn.Module):

    def __init__(self, observation_shape: SequenceType, out_channels: int,
                 activation: nn.Module = nn.ReLU(inplace=True),
                 norm_type: Optional[str] = 'BN',
                 num_resblocks: int = 1,
                 ) -> None:
        """
        Overview:
            Define downSample convolution network. Encode the observation into hidden state.
            This network is often used in video games like Atari. In board games like go and chess,
            we don't need this module.
        Arguments:
            - observation_shape (:obj:`SequenceType`): The shape of observation space, e.g. [C, W, H]=[12, 96, 96]
                for video games like atari, RGB 3 channel times stack 4 frames.
            - out_channels (:obj:`int`): The output channels of output hidden state.
            - activation (:obj:`nn.Module`): The activation function used in network, defaults to nn.ReLU(inplace=True). \
                Use the inplace operation to speed up.
            - norm_type (:obj:`Optional[str]`): The normalization type used in network, defaults to 'BN'.
            - num_resblocks (:obj:`int`): The number of residual blocks. Defaults to 1.
        """
        super().__init__()
        assert norm_type in ['BN', 'LN'], "norm_type must in ['BN', 'LN']"

        assert num_resblocks == 1, "num_resblocks must be 1 in DownSample"
        
        self.observation_shape = observation_shape
        self.conv1 = nn.Conv2d(
            observation_shape[0],
            out_channels // 2,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,  # disable bias for better convergence
        )
        if norm_type == 'BN':
            self.norm1 = nn.BatchNorm2d(out_channels // 2)
        elif norm_type == 'LN':
            self.norm1 = nn.LayerNorm([out_channels // 2, observation_shape[-2] // 2, observation_shape[-1] // 2],
                                      eps=1e-5)

        self.resblocks1 = nn.ModuleList(
            [
                ResBlock(
                    in_channels=out_channels // 2,
                    activation=activation,
                    norm_type=norm_type,
                    res_type='basic',
                    bias=False
                ) for _ in range(num_resblocks)
            ]
        )
        self.downsample_block = ResBlock(
            in_channels=out_channels // 2,
            out_channels=out_channels,
            activation=activation,
            norm_type=norm_type,
            res_type='downsample',
            bias=False
        )
        self.resblocks2 = nn.ModuleList(
            [
                ResBlock(
                    in_channels=out_channels, activation=activation, norm_type=norm_type, res_type='basic', bias=False
                ) for _ in range(num_resblocks)
            ]
        )
        self.pooling1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.resblocks3 = nn.ModuleList(
            [
                ResBlock(
                    in_channels=out_channels, activation=activation, norm_type=norm_type, res_type='basic', bias=False
                ) for _ in range(num_resblocks)
            ]
        )
        self.pooling2 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Shapes:
            - x (:obj:`torch.Tensor`): :math:`(B, C_in, W, H)`, where B is batch size, C_in is channel, W is width, \
                H is height.
            - output (:obj:`torch.Tensor`): :math:`(B, C_out, W_, H_)`, where B is batch size, C_out is channel, W_ is \
                output width, H_ is output height.
        """
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)

        for block in self.resblocks1:
            x = block(x)
        x = self.downsample_block(x)
        for block in self.resblocks2:
            x = block(x)
        x = self.pooling1(x)
        for block in self.resblocks3:
            x = block(x)

        # 64, 84, 96 are the most common observation shapes in Atari games.
        if self.observation_shape[1] == 64:
            output = x
        elif self.observation_shape[1] == 84:
            x = self.pooling2(x)
            output = x
        elif self.observation_shape[1] == 96:
            x = self.pooling2(x)
            output = x
        else:
            raise NotImplementedError(f"DownSample for observation shape {self.observation_shape} is not implemented now. "
                                      f"You should transform the observation shape to 64 or 96 in the env.")

        return output


class HFLanguageRepresentationNetwork(nn.Module):
    def __init__(self,
                 model_path: str = 'google-bert/bert-base-uncased',
                 embedding_size: int = 768,
                 group_size: int = 8,
                 norm_type: str = "simnorm",
                #  norm_type: str = "layernorm", # TODO: Why does nan appear in the first step of training?
                 tokenizer=None):
        """
        Overview:
            This class defines a language representation network that utilizes a pretrained Hugging Face model.
            The network outputs embeddings with the specified dimension and can optionally use SimNorm or LayerNorm
            for normalization at the final stage to ensure training stability.
        Arguments:
            - model_path (str): The path to the pretrained Hugging Face model. Default is 'google-bert/bert-base-uncased'.
            - embedding_size (int): The dimension of the output embeddings. Default is 768.
            - group_size (int): The group size for SimNorm when using normalization.
            - norm_type (str): The type of normalization to use ("simnorm" or "layernorm"). Default is "layernorm".
            - tokenizer (Optional): An instance of a tokenizer. If None, the tokenizer will be loaded from the pretrained model.
        """
        super().__init__()

        from transformers import AutoModel, AutoTokenizer
        logging.info(f"Loading model from: {model_path}")

        # In distributed training, only the rank 0 process downloads the model, and other processes load from cache to speed up startup.
        if get_rank() == 0:
            self.model = AutoModel.from_pretrained(model_path)
        if get_world_size() > 1:
            # Wait for rank 0 to finish loading the model.
            torch.distributed.barrier()
        if get_rank() != 0:
            self.model = AutoModel.from_pretrained(model_path)

        if tokenizer is None:
            # Only rank 0 downloads the tokenizer, and then other processes load it from cache.
            if get_rank() == 0:
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            if get_world_size() > 1:
                torch.distributed.barrier()
            if get_rank() != 0:
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        else:
            self.tokenizer = tokenizer

        # Set the embedding dimension. A linear projection is added (the dimension remains unchanged here but can be extended for other mappings).
        self.embedding_size = embedding_size
        self.embed_proj_head = nn.Linear(self.model.config.hidden_size, self.embedding_size)

        # Select the normalization method based on the norm_type parameter.
        if norm_type.lower() == "simnorm":
            self.norm = SimNorm(simnorm_dim=group_size)
        elif norm_type.lower() == "layernorm":
            self.norm = nn.LayerNorm(embedding_size)
        else:
            raise NotImplementedError(f"Normalization type '{norm_type}' is not implemented. "
                                      f"Choose 'simnorm' or 'layernorm'.")

    def forward(self, x: torch.Tensor, no_grad: bool = True) -> torch.Tensor:
        """
        Forward Propagation:
            Compute the language representation based on the input token sequence.
            The [CLS] token’s representation is extracted from the output of the pretrained model,
            then passed through a linear projection and final normalization layer (SimNorm or LayerNorm).

        Arguments:
            - x (torch.Tensor): Input token sequence of shape [batch_size, seq_len].
            - no_grad (bool): Whether to run in no-gradient mode for memory efficiency. Default is True.
        Returns:
        - torch.Tensor: The processed language embedding with shape [batch_size, embedding_size].
        """
        # Construct the attention mask to exclude padding tokens.
        attention_mask = x != self.tokenizer.pad_token_id

        # Use no_grad context if specified to disable gradient computation.
        if no_grad:
            with torch.no_grad():
                x = x.long()  # Ensure the input tensor is of type long.
                outputs = self.model(x, attention_mask=attention_mask)
                # Get the hidden state from the last layer and select the output corresponding to the [CLS] token.
                cls_embedding = outputs.last_hidden_state[:, 0, :]
        else:
            x = x.long()
            outputs = self.model(x, attention_mask=attention_mask)
            cls_embedding = outputs.last_hidden_state[:, 0, :]

        # Apply linear projection to obtain the desired output dimension.
        cls_embedding = self.embed_proj_head(cls_embedding)
        # Normalize the embeddings using the selected normalization layer (SimNorm or LayerNorm) to ensure training stability.
        cls_embedding = self.norm(cls_embedding)

        return cls_embedding

from torch.nn.utils import weight_norm

# AdaptiveFeatureScaler：在对 1D 向量进行 scaling 时，加入 clamp 限制，避免 runaway
class AdaptiveFeatureScaler(nn.Module):
    def __init__(self, init_scale=0.1, max_scale=1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(init_scale))
        self.max_scale = max_scale
        
    def forward(self, x):
        # 限制 scale 参数的最大值，避免数值爆炸
        clamped_scale = torch.clamp(self.scale, 0.0, self.max_scale)
        return x * clamped_scale / math.sqrt(x.size(1))

# 假设 SimNorm, ResBlock, DownSample 在其他地方已经定义
# 下面仅给出 RepresentationNetworkUniZero 的实现

class RepresentationNetworkUniZero(nn.Module):
    def __init__(
            self,
            observation_shape: tuple = (3, 64, 64),
            num_res_blocks: int = 1,
            num_channels: int = 64,
            downsample: bool = True,
            activation: nn.Module = nn.GELU(approximate='tanh'),
            norm_type: str = 'BN',
            embedding_dim: int = 256,
            group_size: int = 8,
            final_norm_option_in_encoder: str = 'SimNorm',
            use_adaptive_scale: bool = False
            # use_global_pooling: bool = True  # 新增超参数：是否使用全局平均池化
            # use_global_pooling: bool = False # 新增超参数：是否使用全局平均池化
    ) -> None:
        """
        Representation network used in UniZero.
        对于 channel 数较大的场景，可使用全局平均池化来降低全连接层的输入维度，提高训练稳定性。
        """
        super().__init__()
        assert norm_type in ['BN', 'LN'], "norm_type must be in ['BN', 'LN']"
        # 打印日志信息（可选）
        print(f"Using norm type: {norm_type}")
        print(f"Using activation type: {activation}")

        self.use_global_pooling = False

        self.observation_shape = observation_shape
        self.downsample = downsample

        if self.downsample:
            # DownSample 对象的实现需自行定义
            self.downsample_net = DownSample(
                observation_shape,
                num_channels,
                activation=activation,
                norm_type=norm_type,
                num_resblocks=1,
            )
        else:
            self.conv = nn.Conv2d(observation_shape[0], num_channels, kernel_size=3, stride=1, padding=1, bias=False)
            if norm_type == 'BN':
                self.norm = nn.BatchNorm2d(num_channels)
            elif norm_type == 'LN':
                # 当不进行 downsample 时，观察图尺寸不变
                self.norm = nn.LayerNorm([num_channels, observation_shape[-2], observation_shape[-1]], eps=1e-5)

        # 构建 residual block 层
        self.resblocks = nn.ModuleList(
            [
                ResBlock(
                    in_channels=num_channels,
                    activation=activation,
                    norm_type=norm_type,
                    res_type='basic',
                    bias=False
                ) for _ in range(num_res_blocks)
            ]
        )
        self.activation = activation
        self.embedding_dim = embedding_dim

        # 根据观察图尺寸确定空间维度
        if self.observation_shape[1] == 64:
            spatial_size = 8
        elif self.observation_shape[1] in [84, 96]:
            spatial_size = 6
        else:
            spatial_size = self.observation_shape[1]  # 默认采用输入H
            
        if self.observation_shape[1] == 64:
            last_linear_in_dim = num_channels * 8 * 8
        elif self.observation_shape[1] in [84, 96]:
            last_linear_in_dim = num_channels * 6 * 6
        else:
            # 默认采用完整 flatten 的维度
            last_linear_in_dim = num_channels * self.observation_shape[1] * self.observation_shape[2]

        self.last_linear = nn.Linear(last_linear_in_dim, self.embedding_dim, bias=False)


        # 根据是否使用全局平均池化决定 last_linear 前的输入维度以及 norm 的形状
        if self.use_global_pooling:
            linear_in_dim = num_channels  # 全局池化后形状: (B, num_channels, 1, 1)
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
            # 对 1D 向量使用 LayerNorm
            self.norm_before_last_linear = nn.LayerNorm(linear_in_dim, eps=1e-5)
        else:
            linear_in_dim = num_channels * spatial_size * spatial_size
            if use_adaptive_scale:
                # 若通过 flatten 后进行 adaptive scaling，对 1D 向量归一化
                self.norm_before_last_linear = nn.LayerNorm(linear_in_dim, eps=1e-5)
            else:
                # 保留空间信息时，在 (C, H, W) 上归一化
                self.norm_before_last_linear = nn.LayerNorm([num_channels, spatial_size, spatial_size], eps=1e-5)

        self.last_linear = nn.Linear(linear_in_dim, self.embedding_dim, bias=False)

        self.use_adaptive_scale = use_adaptive_scale
        if self.use_adaptive_scale:
            self.adaptive_scaler = AdaptiveFeatureScaler(init_scale=0.1, max_scale=1.0)

        # 最后归一化层，根据 final_norm_option_in_encoder 进行选择
        if final_norm_option_in_encoder == 'LayerNorm':
            self.final_norm = nn.LayerNorm(self.embedding_dim, eps=1e-5)
        elif final_norm_option_in_encoder == 'SimNorm':
            self.final_norm = SimNorm(simnorm_dim=group_size)
        else:
            raise ValueError(f"Unsupported final_norm_option_in_encoder: {final_norm_option_in_encoder}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, H, W)
        Returns:
            x: (B, embedding_dim)
        """
        if self.downsample:
            x = self.downsample_net(x)
        else:
            x = self.conv(x)
            x = self.norm(x)
            x = self.activation(x)
        
        # 依次通过多个 residual block
        for block in self.resblocks:
            x = block(x)
        
        # 分支1：使用全局平均池化
        if self.use_global_pooling:
            x = self.global_pool(x)            # 输出 shape: (B, num_channels, 1, 1)
            x = x.view(x.size(0), -1)            # 展平为 (B, num_channels)
            x = self.norm_before_last_linear(x)  # 对 1D 向量做归一化
        else:
            # 分支2：不使用全局池化
            if self.use_adaptive_scale:
                # 若启用 adaptive scaling：先展平再做 fan-in 缩放
                x = x.view(x.size(0), -1)        # (B, num_channels * spatial_size^2)
                x = self.adaptive_scaler(x)
                x = self.norm_before_last_linear(x)  # 归一化 1D 向量
            else:
                # 保持完整空间信息：在 (B, C, H, W) 上归一化后，再展平
                x = self.norm_before_last_linear(x)
                x = x.view(x.size(0), -1)

        # 最后一层全连接映射与归一化
        x = self.last_linear(x)
        x = self.final_norm(x)
        return x


class RepresentationNetwork(nn.Module):

    def __init__(
            self,
            observation_shape: SequenceType = (4, 96, 96),
            num_res_blocks: int = 1,
            num_channels: int = 64,
            downsample: bool = True,
            activation: nn.Module = nn.ReLU(inplace=True),
            norm_type: str = 'BN',
            embedding_dim: int = 256,
            group_size: int = 8,
            use_sim_norm: bool = False,
    ) -> None:
        """
        Overview:
            Representation network used in MuZero and derived algorithms. Encode the 2D image obs into latent state.
            Currently, the network only supports obs images with both a width and height of 96.
        Arguments:
            - observation_shape (:obj:`SequenceType`): The shape of observation space, e.g. [C, W, H]=[4, 96, 96]
                for video games like atari, 1 gray channel times stack 4 frames.
            - num_res_blocks (:obj:`int`): The number of residual blocks.
            - num_channels (:obj:`int`): The channel of output hidden state.
            - downsample (:obj:`bool`): Whether to do downsampling for observations in ``representation_network``, \
                defaults to True. This option is often used in video games like Atari. In board games like go, \
                we don't need this module.
            - activation (:obj:`nn.Module`): The activation function used in network, defaults to nn.ReLU(inplace=True). \
                Use the inplace operation to speed up.
            - norm_type (:obj:`str`): The type of normalization in networks. defaults to 'BN'.
            - embedding_dim (:obj:`int`): The dimension of the output hidden state.
            - group_size (:obj:`int`): The size of group in the SimNorm layer.
            - use_sim_norm (:obj:`bool`): Whether to use SimNorm layer, defaults to False.
        """
        super().__init__()
        assert norm_type in ['BN', 'LN'], "norm_type must in ['BN', 'LN']"

        self.downsample = downsample
        if self.downsample:
            self.downsample_net = DownSample(
                observation_shape,
                num_channels,
                activation=activation,
                norm_type=norm_type,
            )
        else:
            self.conv = nn.Conv2d(observation_shape[0], num_channels, kernel_size=3, stride=1, padding=1, bias=False)

            if norm_type == 'BN':
                self.norm = nn.BatchNorm2d(num_channels)
            elif norm_type == 'LN':
                if downsample:
                    self.norm = nn.LayerNorm(
                        [num_channels, math.ceil(observation_shape[-2] / 16), math.ceil(observation_shape[-1] / 16)],
                        eps=1e-5)
                else:
                    self.norm = nn.LayerNorm([num_channels, observation_shape[-2], observation_shape[-1]], eps=1e-5)

        self.resblocks = nn.ModuleList(
            [
                ResBlock(
                    in_channels=num_channels, activation=activation, norm_type=norm_type, res_type='basic', bias=False
                ) for _ in range(num_res_blocks)
            ]
        )
        self.activation = activation

        self.use_sim_norm = use_sim_norm

        if self.use_sim_norm:
            self.embedding_dim = embedding_dim
            self.sim_norm = SimNorm(simnorm_dim=group_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Shapes:
            - x (:obj:`torch.Tensor`): :math:`(B, C_in, W, H)`, where B is batch size, C_in is channel, W is width, \
                H is height.
            - output (:obj:`torch.Tensor`): :math:`(B, C_out, W_, H_)`, where B is batch size, C_out is channel, W_ is \
                output width, H_ is output height.
        """
        if self.downsample:
            x = self.downsample_net(x)
        else:
            x = self.conv(x)
            x = self.norm(x)
            x = self.activation(x)

        for block in self.resblocks:
            x = block(x)

        if self.use_sim_norm:
            # NOTE: very important. 
            # for atari 64,8,8 = 4096 -> 768
            x = self.sim_norm(x)

        return x


class RepresentationNetworkMLP(nn.Module):

    def __init__(
            self,
            observation_shape: int,
            hidden_channels: int = 64,
            layer_num: int = 2,
            activation: nn.Module = nn.GELU(approximate='tanh'),
            norm_type: Optional[str] = 'BN',
            group_size: int = 8,
    ) -> torch.Tensor:
        """
        Overview:
            Representation network used in MuZero and derived algorithms. Encode the vector obs into latent state \
                with Multi-Layer Perceptron (MLP).
        Arguments:
            - observation_shape (:obj:`int`): The shape of vector observation space, e.g. N = 10.
            - num_res_blocks (:obj:`int`): The number of residual blocks.
            - hidden_channels (:obj:`int`): The channel of output hidden state.
            - downsample (:obj:`bool`): Whether to do downsampling for observations in ``representation_network``, \
                defaults to True. This option is often used in video games like Atari. In board games like go, \
                we don't need this module.
            - activation (:obj:`nn.Module`): The activation function used in network, defaults to nn.ReLU(inplace=True). \
                Use the inplace operation to speed up.
            - norm_type (:obj:`str`): The type of normalization in networks. defaults to 'BN'.
        """
        super().__init__()
        self.fc_representation = MLP(
            in_channels=observation_shape,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            layer_num=layer_num,
            activation=activation,
            norm_type=norm_type,
            # don't use activation and norm in the last layer of representation network is important for convergence.
            output_activation=False,
            output_norm=False,
            # last_linear_layer_init_zero=True is beneficial for convergence speed.
            last_linear_layer_init_zero=True,
        )
        self.sim_norm = SimNorm(simnorm_dim=group_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Shapes:
            - x (:obj:`torch.Tensor`): :math:`(B, N)`, where B is batch size, N is the length of vector observation.
            - output (:obj:`torch.Tensor`): :math:`(B, hidden_channels)`, where B is batch size.
        """
        x = self.fc_representation(x)
        # TODO
        x = self.sim_norm(x)
        return x


class LatentDecoder(nn.Module):
    
    def __init__(self, embedding_dim: int, output_shape: SequenceType, num_channels: int = 64, activation: nn.Module = nn.GELU(approximate='tanh')):
        """
        Overview:
            Decoder network used in UniZero. Decode the latent state into 2D image obs.
        Arguments:
            - embedding_dim (:obj:`int`): The dimension of the latent state.
            - output_shape (:obj:`SequenceType`): The shape of observation space, e.g. [C, W, H]=[3, 64, 64]
                for video games like atari, RGB 3 channel times stack 4 frames.
            - num_channels (:obj:`int`): The channel of output hidden state.
            - activation (:obj:`nn.Module`): The activation function used in network, defaults to nn.GELU(approximate='tanh').
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.output_shape = output_shape  # (C, H, W)
        self.num_channels = num_channels
        self.activation = activation

        # Assuming that the output shape is (C, H, W) = (12, 96, 96) and embedding_dim is 256
        # We will reverse the process of the representation network
        self.initial_size = (
            num_channels, output_shape[1] // 8, output_shape[2] // 8)  # This should match the last layer of the encoder
        self.fc = nn.Linear(self.embedding_dim, np.prod(self.initial_size))

        # Upsampling blocks
        self.conv_blocks = nn.ModuleList([
            # Block 1: (num_channels, H/8, W/8) -> (num_channels//2, H/4, W/4)
            nn.ConvTranspose2d(num_channels, num_channels // 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            self.activation,
            nn.BatchNorm2d(num_channels // 2),
            # Block 2: (num_channels//2, H/4, W/4) -> (num_channels//4, H/2, W/2)
            nn.ConvTranspose2d(num_channels // 2, num_channels // 4, kernel_size=3, stride=2, padding=1,
                               output_padding=1),
            self.activation,
            nn.BatchNorm2d(num_channels // 4),
            # Block 3: (num_channels//4, H/2, W/2) -> (output_shape[0], H, W)
            nn.ConvTranspose2d(num_channels // 4, output_shape[0], kernel_size=3, stride=2, padding=1,
                               output_padding=1),
        ])
        # TODO: last layer use sigmoid?

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        # Map embeddings back to the image space
        x = self.fc(embeddings)  # (B, embedding_dim) -> (B, C*H/8*W/8)
        x = x.view(-1, *self.initial_size)  # (B, C*H/8*W/8) -> (B, C, H/8, W/8)

        # Apply conv blocks
        for block in self.conv_blocks:
            x = block(x)  # Upsample progressively

        # The output x should have the shape of (B, output_shape[0], output_shape[1], output_shape[2])
        return x


class LatentEncoderForMemoryEnv(nn.Module):

    def __init__(
            self,
            image_shape=(3, 5, 5),
            embedding_size=100,
            channels=[16, 32, 64],
            kernel_sizes=[3, 3, 3],
            strides=[1, 1, 1],
            activation: nn.Module = nn.GELU(approximate='tanh'),
            normalize_pixel=False,
            group_size: int = 8,
            **kwargs,
    ):
        """
        Overview:
            Encoder network used in UniZero in MemoryEnv. Encode the 2D image obs into latent state.
        Arguments:
            - image_shape (:obj:`SequenceType`): The shape of observation space, e.g. [C, W, H]=[3, 64, 64]
                for video games like atari, RGB 3 channel times stack 4 frames.
            - embedding_size (:obj:`int`): The dimension of the latent state.
            - channels (:obj:`List[int]`): The channel of output hidden state.
            - kernel_sizes (:obj:`List[int]`): The kernel size of convolution layers.
            - strides (:obj:`List[int]`): The stride of convolution layers.
            - activation (:obj:`nn.Module`): The activation function used in network, defaults to nn.GELU(approximate='tanh'). \
                Use the inplace operation to speed up.
            - normalize_pixel (:obj:`bool`): Whether to normalize the pixel values to [0, 1], defaults to False.
            - group_size (:obj:`int`): The dimension for simplicial normalization
        """
        super(LatentEncoderForMemoryEnv, self).__init__()
        self.shape = image_shape
        self.channels = [image_shape[0]] + list(channels)

        layers = []
        for i in range(len(self.channels) - 1):
            layers.append(
                nn.Conv2d(
                    self.channels[i], self.channels[i + 1], kernel_sizes[i], strides[i],
                    padding=kernel_sizes[i] // 2  # keep the same size of feature map
                )
            )
            layers.append(nn.BatchNorm2d(self.channels[i + 1]))
            layers.append(activation)

        layers.append(nn.AdaptiveAvgPool2d(1))

        self.cnn = nn.Sequential(*layers)
        self.linear = nn.Sequential(
            nn.Linear(self.channels[-1], embedding_size, bias=False),
        )
        init.kaiming_normal_(self.linear[0].weight, mode='fan_out', nonlinearity='relu')

        self.normalize_pixel = normalize_pixel
        self.sim_norm = SimNorm(simnorm_dim=group_size)

    def forward(self, image):
        if self.normalize_pixel:
            image = image / 255.0
        x = self.cnn(image.float())  # (B, C, 1, 1)
        x = torch.flatten(x, start_dim=1)  # (B, C)
        x = self.linear(x)  # (B, embedding_size)
        x = self.sim_norm(x)
        return x


class LatentDecoderForMemoryEnv(nn.Module):

    def __init__(
            self,
            image_shape=(3, 5, 5),
            embedding_size=256,
            channels=[64, 32, 16],
            kernel_sizes=[3, 3, 3],
            strides=[1, 1, 1],
            activation: nn.Module = nn.LeakyReLU(negative_slope=0.01),
            **kwargs,
    ):
        """
        Overview:
            Decoder network used in UniZero in MemoryEnv. Decode the latent state into 2D image obs.
        Arguments:
            - image_shape (:obj:`SequenceType`): The shape of observation space, e.g. [C, W, H]=[3, 64, 64]
                for video games like atari, RGB 3 channel times stack 4 frames.
            - embedding_size (:obj:`int`): The dimension of the latent state.
            - channels (:obj:`List[int]`): The channel of output hidden state.
            - kernel_sizes (:obj:`List[int]`): The kernel size of convolution layers.
            - strides (:obj:`List[int]`): The stride of convolution layers.
            - activation (:obj:`nn.Module`): The activation function used in network, defaults to nn.LeakyReLU(). \
                Use the inplace operation to speed up.
        """
        super(LatentDecoderForMemoryEnv, self).__init__()
        self.shape = image_shape
        self.channels = list(channels) + [image_shape[0]]

        self.linear = nn.Linear(embedding_size, channels[0] * image_shape[1] * image_shape[2])

        layers = []
        for i in range(len(self.channels) - 1):
            layers.append(
                nn.ConvTranspose2d(
                    self.channels[i], self.channels[i + 1], kernel_sizes[i], strides[i],
                    padding=kernel_sizes[i] // 2, output_padding=strides[i] - 1
                )
            )
            if i < len(self.channels) - 2:
                layers.append(nn.BatchNorm2d(self.channels[i + 1]))
                layers.append(activation)
            else:
                layers.append(nn.Sigmoid())

        self.deconv = nn.Sequential(*layers)

    def forward(self, embedding):
        x = self.linear(embedding)
        x = x.view(-1, self.channels[0], self.shape[1], self.shape[2])
        x = self.deconv(x)  # (B, C, H, W)
        return x


class VectorDecoderForMemoryEnv(nn.Module):

    def __init__(
            self,
            embedding_dim: int,
            output_shape: SequenceType,
            hidden_channels: int = 64,
            layer_num: int = 2,
            activation: nn.Module = nn.LeakyReLU(negative_slope=0.01),  # TODO
            norm_type: Optional[str] = 'BN',
    ) -> torch.Tensor:
        """
        Overview:
            Decoder network used in UniZero in MemoryEnv. Decode the latent state into vector obs.
        Arguments:
            - observation_shape (:obj:`int`): The shape of vector observation space, e.g. N = 10.
            - num_res_blocks (:obj:`int`): The number of residual blocks.
            - hidden_channels (:obj:`int`): The channel of output hidden state.
            - downsample (:obj:`bool`): Whether to do downsampling for observations in ``representation_network``, \
                defaults to True. This option is often used in video games like Atari. In board games like go, \
                we don't need this module.
            - activation (:obj:`nn.Module`): The activation function used in network, defaults to nn.ReLU(). \
                Use the inplace operation to speed up.
            - norm_type (:obj:`str`): The type of normalization in networks. defaults to 'BN'.
        """
        super().__init__()
        self.fc_representation = MLP(
            in_channels=embedding_dim,
            hidden_channels=hidden_channels,
            out_channels=output_shape,
            layer_num=layer_num,
            activation=activation,
            norm_type=norm_type,
            # don't use activation and norm in the last layer of representation network is important for convergence.
            output_activation=False,
            output_norm=False,
            # last_linear_layer_init_zero=True is beneficial for convergence speed.
            last_linear_layer_init_zero=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Shapes:
            - x (:obj:`torch.Tensor`): :math:`(B, N)`, where B is batch size, N is the length of vector observation.
            - output (:obj:`torch.Tensor`): :math:`(B, hidden_channels)`, where B is batch size.
        """
        x = self.fc_representation(x)
        return x


class PredictionNetwork(nn.Module):

    def __init__(
            self,
            observation_shape: SequenceType,
            action_space_size: int,
            num_res_blocks: int,
            num_channels: int,
            value_head_channels: int,
            policy_head_channels: int,
            value_head_hidden_channels: int,
            policy_head_hidden_channels: int,
            output_support_size: int,
            flatten_input_size_for_value_head: int,
            flatten_input_size_for_policy_head: int,
            downsample: bool = False,
            last_linear_layer_init_zero: bool = True,
            activation: nn.Module = nn.ReLU(inplace=True),
            norm_type: Optional[str] = 'BN',
    ) -> None:
        """
        Overview:
            The definition of policy and value prediction network, which is used to predict value and policy by the
            given latent state.
        Arguments:
            - observation_shape (:obj:`SequenceType`): The shape of observation space, e.g. (C, H, W) for image.
            - action_space_size: (:obj:`int`): Action space size, usually an integer number for discrete action space.
            - num_res_blocks (:obj:`int`): The number of res blocks in AlphaZero model.
            - num_channels (:obj:`int`): The channels of hidden states.
            - value_head_channels (:obj:`int`): The channels of value head.
            - policy_head_channels (:obj:`int`): The channels of policy head.
            - value_head_hidden_channels (:obj:`SequenceType`): The number of hidden layers used in value head (MLP head).
            - policy_head_hidden_channels (:obj:`SequenceType`): The number of hidden layers used in policy head (MLP head).
            - output_support_size (:obj:`int`): The size of categorical value output.
            - self_supervised_learning_loss (:obj:`bool`): Whether to use self_supervised_learning related networks \
            - flatten_input_size_for_value_head (:obj:`int`): The size of flatten hidden states, i.e. the input size \
                of the value head.
            - flatten_input_size_for_policy_head (:obj:`int`): The size of flatten hidden states, i.e. the input size \
                of the policy head.
            - downsample (:obj:`bool`): Whether to do downsampling for observations in ``representation_network``.
            - last_linear_layer_init_zero (:obj:`bool`): Whether to use zero initializations for the last layer of \
                dynamics/prediction mlp, default sets it to True.
            - activation (:obj:`Optional[nn.Module]`): Activation function used in network, which often use in-place \
                operation to speedup, e.g. ReLU(inplace=True).
            - norm_type (:obj:`str`): The type of normalization in networks. defaults to 'BN'.
        """
        super(PredictionNetwork, self).__init__()
        assert norm_type in ['BN', 'LN'], "norm_type must in ['BN', 'LN']"

        self.resblocks = nn.ModuleList(
            [
                ResBlock(
                    in_channels=num_channels, activation=activation, norm_type=norm_type, res_type='basic', bias=False
                ) for _ in range(num_res_blocks)
            ]
        )

        self.conv1x1_value = nn.Conv2d(num_channels, value_head_channels, 1)
        self.conv1x1_policy = nn.Conv2d(num_channels, policy_head_channels, 1)

        if observation_shape[1] == 96:
            latent_shape = (observation_shape[1] // 16, observation_shape[2] // 16)
        elif observation_shape[1] == 64:
            latent_shape = (observation_shape[1] // 8, observation_shape[2] // 8)

        if norm_type == 'BN':
            self.norm_value = nn.BatchNorm2d(value_head_channels)
            self.norm_policy = nn.BatchNorm2d(policy_head_channels)
        elif norm_type == 'LN':
            if downsample:
                self.norm_value = nn.LayerNorm(
                    [value_head_channels, *latent_shape],
                    eps=1e-5)
                self.norm_policy = nn.LayerNorm([policy_head_channels, *latent_shape], eps=1e-5)
            else:
                self.norm_value = nn.LayerNorm([value_head_channels, observation_shape[-2], observation_shape[-1]],
                                               eps=1e-5)
                self.norm_policy = nn.LayerNorm([policy_head_channels, observation_shape[-2], observation_shape[-1]],
                                                eps=1e-5)

        self.flatten_input_size_for_value_head = flatten_input_size_for_value_head
        self.flatten_input_size_for_policy_head = flatten_input_size_for_policy_head

        self.activation = activation

        self.fc_value = MLP_V2(
            in_channels=self.flatten_input_size_for_value_head,
            hidden_channels=value_head_hidden_channels,
            out_channels=output_support_size,
            activation=self.activation,
            norm_type=norm_type,
            output_activation=False,
            output_norm=False,
            # last_linear_layer_init_zero=True is beneficial for convergence speed.
            last_linear_layer_init_zero=last_linear_layer_init_zero
        )
        self.fc_policy = MLP_V2(
            in_channels=self.flatten_input_size_for_policy_head,
            hidden_channels=policy_head_hidden_channels,
            out_channels=action_space_size,
            activation=self.activation,
            norm_type=norm_type,
            output_activation=False,
            output_norm=False,
            # last_linear_layer_init_zero=True is beneficial for convergence speed.
            last_linear_layer_init_zero=last_linear_layer_init_zero
        )

    def forward(self, latent_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Overview:
            Forward computation of the prediction network.
        Arguments:
            - latent_state (:obj:`torch.Tensor`): input tensor with shape (B, latent_state_dim).
        Returns:
            - policy (:obj:`torch.Tensor`): policy tensor with shape (B, action_space_size).
            - value (:obj:`torch.Tensor`): value tensor with shape (B, output_support_size).
        """
        for res_block in self.resblocks:
            latent_state = res_block(latent_state)

        value = self.conv1x1_value(latent_state)
        value = self.norm_value(value)
        value = self.activation(value)

        policy = self.conv1x1_policy(latent_state)
        policy = self.norm_policy(policy)
        policy = self.activation(policy)

        value = value.reshape(-1, self.flatten_input_size_for_value_head)
        policy = policy.reshape(-1, self.flatten_input_size_for_policy_head)

        value = self.fc_value(value)
        policy = self.fc_policy(policy)
        return policy, value


class PredictionNetworkMLP(nn.Module):

    def __init__(
            self,
            action_space_size,
            num_channels,
            common_layer_num: int = 2,
            value_head_hidden_channels: SequenceType = [32],
            policy_head_hidden_channels: SequenceType = [32],
            output_support_size: int = 601,
            last_linear_layer_init_zero: bool = True,
            activation: Optional[nn.Module] = nn.ReLU(inplace=True),
            norm_type: Optional[str] = 'BN',
    ):
        """
        Overview:
            The definition of policy and value prediction network with Multi-Layer Perceptron (MLP),
            which is used to predict value and policy by the given latent state.
        Arguments:
            - action_space_size: (:obj:`int`): Action space size, usually an integer number. For discrete action \
                space, it is the number of discrete actions.
            - num_channels (:obj:`int`): The channels of latent states.
            - value_head_hidden_channels (:obj:`SequenceType`): The number of hidden layers used in value head (MLP head).
            - policy_head_hidden_channels (:obj:`SequenceType`): The number of hidden layers used in policy head (MLP head).
            - output_support_size (:obj:`int`): The size of categorical value output.
            - last_linear_layer_init_zero (:obj:`bool`): Whether to use zero initializations for the last layer of \
                dynamics/prediction mlp, default sets it to True.
            - activation (:obj:`Optional[nn.Module]`): Activation function used in network, which often use in-place \
                operation to speedup, e.g. ReLU(inplace=True).
            - norm_type (:obj:`str`): The type of normalization in networks. defaults to 'BN'.
        """
        super().__init__()
        self.num_channels = num_channels

        # ******* common backbone ******
        self.fc_prediction_common = MLP(
            in_channels=self.num_channels,
            hidden_channels=self.num_channels,
            out_channels=self.num_channels,
            layer_num=common_layer_num,
            activation=activation,
            norm_type=norm_type,
            output_activation=True,
            output_norm=True,
            # last_linear_layer_init_zero=False is important for convergence
            last_linear_layer_init_zero=False,
        )

        # ******* value and policy head ******
        self.fc_value_head = MLP_V2(
            in_channels=self.num_channels,
            hidden_channels=value_head_hidden_channels,
            out_channels=output_support_size,
            activation=activation,
            norm_type=norm_type,
            output_activation=False,
            output_norm=False,
            # last_linear_layer_init_zero=True is beneficial for convergence speed.
            last_linear_layer_init_zero=last_linear_layer_init_zero
        )
        self.fc_policy_head = MLP_V2(
            in_channels=self.num_channels,
            hidden_channels=policy_head_hidden_channels,
            out_channels=action_space_size,
            activation=activation,
            norm_type=norm_type,
            output_activation=False,
            output_norm=False,
            # last_linear_layer_init_zero=True is beneficial for convergence speed.
            last_linear_layer_init_zero=last_linear_layer_init_zero
        )

    def forward(self, latent_state: torch.Tensor):
        """
        Overview:
            Forward computation of the prediction network.
        Arguments:
            - latent_state (:obj:`torch.Tensor`): input tensor with shape (B, latent_state_dim).
        Returns:
            - policy (:obj:`torch.Tensor`): policy tensor with shape (B, action_space_size).
            - value (:obj:`torch.Tensor`): value tensor with shape (B, output_support_size).
        """
        x_prediction_common = self.fc_prediction_common(latent_state)

        value = self.fc_value_head(x_prediction_common)
        policy = self.fc_policy_head(x_prediction_common)
        return policy, value


class PredictionHiddenNetwork(nn.Module):

    def __init__(
            self,
            observation_shape: SequenceType,
            action_space_size: int,
            num_res_blocks: int,
            num_channels: int,
            value_head_channels: int,
            policy_head_channels: int,
            value_head_hidden_channels: int,
            policy_head_hidden_channels: int,
            output_support_size: int,
            flatten_input_size_for_value_head: int,
            flatten_input_size_for_policy_head: int,
            downsample: bool = False,
            last_linear_layer_init_zero: bool = True,
            activation: nn.Module = nn.ReLU(inplace=True),
            norm_type: Optional[str] = 'BN',
            gru_hidden_size: int = 512,
    ) -> None:
        """
        Overview:
            The definition of policy and value prediction network, which is used to predict value and policy by the
            given latent state.
        Arguments:
            - observation_shape (:obj:`SequenceType`): The shape of observation space, e.g. (C, H, W) for image.
            - action_space_size: (:obj:`int`): Action space size, usually an integer number for discrete action space.
            - num_res_blocks (:obj:`int`): The number of res blocks in AlphaZero model.
            - num_channels (:obj:`int`): The channels of hidden states.
            - value_head_channels (:obj:`int`): The channels of value head.
            - policy_head_channels (:obj:`int`): The channels of policy head.
            - value_head_hidden_channels (:obj:`SequenceType`): The number of hidden layers used in value head (MLP head).
            - policy_head_hidden_channels (:obj:`SequenceType`): The number of hidden layers used in policy head (MLP head).
            - output_support_size (:obj:`int`): The size of categorical value output.
            - self_supervised_learning_loss (:obj:`bool`): Whether to use self_supervised_learning related networks \
            - flatten_input_size_for_value_head (:obj:`int`): The size of flatten hidden states, i.e. the input size \
                of the value head.
            - flatten_input_size_for_policy_head (:obj:`int`): The size of flatten hidden states, i.e. the input size \
                of the policy head.
            - downsample (:obj:`bool`): Whether to do downsampling for observations in ``representation_network``.
            - last_linear_layer_init_zero (:obj:`bool`): Whether to use zero initializations for the last layer of \
                dynamics/prediction mlp, default sets it to True.
            - activation (:obj:`Optional[nn.Module]`): Activation function used in network, which often use in-place \
                operation to speedup, e.g. ReLU(inplace=True).
            - norm_type (:obj:`str`): The type of normalization in networks. defaults to 'BN'.
        """
        super(PredictionHiddenNetwork, self).__init__()
        assert norm_type in ['BN', 'LN'], "norm_type must in ['BN', 'LN']"

        self.observation_shape = observation_shape
        self.gru_hidden_size = gru_hidden_size
        self.resblocks = nn.ModuleList(
            [
                ResBlock(
                    in_channels=num_channels, activation=activation, norm_type=norm_type, res_type='basic', bias=False
                ) for _ in range(num_res_blocks)
            ]
        )

        self.conv1x1_value = nn.Conv2d(num_channels, value_head_channels, 1)
        self.conv1x1_policy = nn.Conv2d(num_channels, policy_head_channels, 1)

        if norm_type == 'BN':
            self.norm_value = nn.BatchNorm2d(value_head_channels)
            self.norm_policy = nn.BatchNorm2d(policy_head_channels)
        elif norm_type == 'LN':
            if downsample:
                self.norm_value = nn.LayerNorm(
                    [value_head_channels, math.ceil(observation_shape[-2] / 16), math.ceil(observation_shape[-1] / 16)],
                    eps=1e-5)
                self.norm_policy = nn.LayerNorm([policy_head_channels, math.ceil(observation_shape[-2] / 16),
                                                 math.ceil(observation_shape[-1] / 16)], eps=1e-5)
            else:
                self.norm_value = nn.LayerNorm([value_head_channels, observation_shape[-2], observation_shape[-1]],
                                               eps=1e-5)
                self.norm_policy = nn.LayerNorm([policy_head_channels, observation_shape[-2], observation_shape[-1]],
                                                eps=1e-5)

        self.flatten_input_size_for_value_head = flatten_input_size_for_value_head
        self.flatten_input_size_for_policy_head = flatten_input_size_for_policy_head

        self.activation = activation

        self.fc_value = MLP(
            in_channels=self.flatten_input_size_for_value_head + self.gru_hidden_size,
            hidden_channels=value_head_hidden_channels[0],
            out_channels=output_support_size,
            layer_num=len(value_head_hidden_channels) + 1,
            activation=self.activation,
            norm_type=norm_type,
            output_activation=False,
            output_norm=False,
            # last_linear_layer_init_zero=True is beneficial for convergence speed.
            last_linear_layer_init_zero=last_linear_layer_init_zero
        )
        self.fc_policy = MLP(
            in_channels=self.flatten_input_size_for_policy_head + self.gru_hidden_size,
            hidden_channels=policy_head_hidden_channels[0],
            out_channels=action_space_size,
            layer_num=len(policy_head_hidden_channels) + 1,
            activation=self.activation,
            norm_type=norm_type,
            output_activation=False,
            output_norm=False,
            # last_linear_layer_init_zero=True is beneficial for convergence speed.
            last_linear_layer_init_zero=last_linear_layer_init_zero
        )

    def forward(self, latent_state: torch.Tensor, world_model_latent_history: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor]:
        """
        Overview:
            Forward computation of the prediction network.
        Arguments:
            - latent_state (:obj:`torch.Tensor`): input tensor with shape (B, latent_state_dim).
        Returns:
            - policy (:obj:`torch.Tensor`): policy tensor with shape (B, action_space_size).
            - value (:obj:`torch.Tensor`): value tensor with shape (B, output_support_size).
        """
        for res_block in self.resblocks:
            latent_state = res_block(latent_state)

        value = self.conv1x1_value(latent_state)
        value = self.norm_value(value)
        value = self.activation(value)

        policy = self.conv1x1_policy(latent_state)
        policy = self.norm_policy(policy)
        policy = self.activation(policy)

        latent_state_value = value.reshape(-1, self.flatten_input_size_for_value_head)
        latent_state_policy = policy.reshape(-1, self.flatten_input_size_for_policy_head)

        # TODO: world_model_latent_history.squeeze(0) shape: (num_layers * num_directions, batch_size, hidden_size) ->  ( batch_size, hidden_size)
        latent_history_value = torch.cat([latent_state_value, world_model_latent_history.squeeze(0)], dim=1)
        latent_history_policy = torch.cat([latent_state_policy, world_model_latent_history.squeeze(0)], dim=1)

        value = self.fc_value(latent_history_value)
        policy = self.fc_policy(latent_history_policy)
        return policy, value
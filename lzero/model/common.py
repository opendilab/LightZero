"""
Overview:
    This Python file provides a collection of reusable model templates designed to streamline the development
    process for various custom algorithms. By utilizing these pre-built model templates, users can quickly adapt and
    customize their algorithms, ensuring efficient and effective development.
    Users can refer to the unittest of these model templates to learn how to use them.
"""
import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from ditk import logging
# Assuming these imports are valid in the user's environment.
# If they are not, they should be replaced with the correct ones.
from ding.torch_utils import MLP, ResBlock
from ding.torch_utils.network.normalization import build_normalization
from ding.utils import SequenceType, get_rank, get_world_size
from transformers import AutoModelForCausalLM, AutoTokenizer
from ding.utils import set_pkg_seed, get_rank, get_world_size



def MLP_V2(
        in_channels: int,
        hidden_channels: List[int],
        out_channels: int,
        layer_fn: Callable = nn.Linear,
        activation: Optional[nn.Module] = None,
        norm_type: Optional[str] = None,
        use_dropout: bool = False,
        dropout_probability: float = 0.5,
        output_activation: bool = True,
        output_norm: bool = True,
        last_linear_layer_init_zero: bool = False,
) -> nn.Sequential:
    """
    Overview:
        Creates a multi-layer perceptron (MLP) using a list of hidden dimensions. Each layer consists of a fully
        connected block with optional activation, normalization, and dropout. The final layer is configurable
        to include or exclude activation and normalization.
    Arguments:
        - in_channels (:obj:`int`): Number of input channels (dimensionality of the input tensor).
        - hidden_channels (:obj:`List[int]`): A list specifying the number of channels for each hidden layer.
        - out_channels (:obj:`int`): Number of output channels (dimensionality of the output tensor).
        - layer_fn (:obj:`Callable`): The function to construct layers, defaults to `nn.Linear`.
        - activation (:obj:`Optional[nn.Module]`): Activation function to use after each layer, defaults to None.
        - norm_type (:obj:`Optional[str]`): Type of normalization to apply. If None, no normalization is applied.
        - use_dropout (:obj:`bool`): Whether to apply dropout after each layer, defaults to False.
        - dropout_probability (:obj:`float`): The probability for dropout, defaults to 0.5.
        - output_activation (:obj:`bool`): Whether to apply activation to the output layer, defaults to True.
        - output_norm (:obj:`bool`): Whether to apply normalization to the output layer, defaults to True.
        - last_linear_layer_init_zero (:obj:`bool`): Whether to initialize the last linear layer's weights and biases to zero.
    Returns:
        - block (:obj:`nn.Sequential`): A PyTorch `nn.Sequential` object containing the layers of the MLP.
    """
    if not hidden_channels:
        logging.warning("hidden_channels is empty, creating a single-layer MLP.")

    layers = []
    all_channels = [in_channels] + hidden_channels + [out_channels]
    num_layers = len(all_channels) - 1

    for i in range(num_layers):
        is_last_layer = (i == num_layers - 1)
        layers.append(layer_fn(all_channels[i], all_channels[i+1]))

        if not is_last_layer:
            # Intermediate layers
            if norm_type:
                layers.append(build_normalization(norm_type, dim=1)(all_channels[i+1]))
            if activation:
                layers.append(activation)
            if use_dropout:
                layers.append(nn.Dropout(dropout_probability))
        else:
            # Last layer
            if output_norm and norm_type:
                layers.append(build_normalization(norm_type, dim=1)(all_channels[i+1]))
            if output_activation and activation:
                layers.append(activation)
            # Note: Dropout on the final output is usually not recommended unless for specific regularization purposes.
            # The original logic applied it, so we keep it for consistency.
            if use_dropout:
                layers.append(nn.Dropout(dropout_probability))

    # Initialize the last linear layer to zero if specified
    if last_linear_layer_init_zero:
        for layer in reversed(layers):
            if isinstance(layer, nn.Linear):
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)
                break

    return nn.Sequential(*layers)


# --- Data-structures for Network Outputs ---

@dataclass
class MZRNNNetworkOutput:
    """
    Overview:
        Data structure for the output of the MuZeroRNN model.
    """
    value: torch.Tensor
    value_prefix: torch.Tensor
    policy_logits: torch.Tensor
    latent_state: torch.Tensor
    predict_next_latent_state: torch.Tensor
    reward_hidden_state: Tuple[torch.Tensor, torch.Tensor]


@dataclass
class EZNetworkOutput:
    """
    Overview:
        Data structure for the output of the EfficientZero model.
    """
    value: torch.Tensor
    value_prefix: torch.Tensor
    policy_logits: torch.Tensor
    latent_state: torch.Tensor
    reward_hidden_state: Tuple[torch.Tensor, torch.Tensor]


@dataclass
class MZNetworkOutput:
    """
    Overview:
        Data structure for the output of the MuZero model.
    """
    value: torch.Tensor
    reward: torch.Tensor
    policy_logits: torch.Tensor
    latent_state: torch.Tensor


# --- Core Network Components ---

class SimNorm(nn.Module):
    """
    Overview:
        Implements Simplicial Normalization as described in the paper: https://arxiv.org/abs/2204.00616.
        It groups features and applies softmax to each group.
    """

    def __init__(self, simnorm_dim: int) -> None:
        """
        Arguments:
            - simnorm_dim (:obj:`int`): The size of each group (simplex) to apply softmax over.
        """
        super().__init__()
        self.dim = simnorm_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Overview:
            Forward pass for SimNorm.
        Arguments:
            - x (:obj:`torch.Tensor`): The input tensor.
        Returns:
            - (:obj:`torch.Tensor`): The tensor after applying Simplicial Normalization.
        """
        if x.shape[1] == 0:
            return x
        # Reshape to (batch, groups, dim)
        x_reshaped = x.view(*x.shape[:-1], -1, self.dim)
        # Apply softmax over the last dimension (the simplex)
        x_softmax = F.softmax(x_reshaped, dim=-1)
        # Reshape back to the original tensor shape
        return x_softmax.view(*x.shape)

    def __repr__(self) -> str:
        return f"SimNorm(dim={self.dim})"


def AvgL1Norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Overview:
        Normalizes a tensor by the mean of its absolute values (L1 norm) along the last dimension.
    Arguments:
        - x (:obj:`torch.Tensor`): The input tensor to normalize.
        - eps (:obj:`float`): A small epsilon value to prevent division by zero.
    Returns:
        - (:obj:`torch.Tensor`): The normalized tensor.
    """
    return x / (x.abs().mean(dim=-1, keepdim=True) + eps)


class FeatureAndGradientHook:
    """
    Overview:
        A utility class to capture and analyze features and gradients of a specific module during
        the forward and backward passes. This is useful for debugging and understanding model dynamics.
    """

    def __init__(self, module: nn.Module):
        """
        Arguments:
            - module (:obj:`nn.Module`): The PyTorch module to attach the hooks to.
        """
        self.features_before = []
        self.features_after = []
        self.grads_before = []
        self.grads_after = []
        self.forward_handler = module.register_forward_hook(self._forward_hook)
        self.backward_handler = module.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module: nn.Module, inputs: Tuple[torch.Tensor], output: torch.Tensor) -> None:
        """Hook to capture input and output features during the forward pass."""
        with torch.no_grad():
            self.features_before.append(inputs[0].clone().detach())
            self.features_after.append(output.clone().detach())

    def _backward_hook(self, module: nn.Module, grad_inputs: Tuple[torch.Tensor], grad_outputs: Tuple[torch.Tensor]) -> None:
        """Hook to capture input and output gradients during the backward pass."""
        with torch.no_grad():
            self.grads_before.append(grad_inputs[0].clone().detach() if grad_inputs[0] is not None else None)
            self.grads_after.append(grad_outputs[0].clone().detach() if grad_outputs[0] is not None else None)

    def analyze(self) -> Tuple[float, float, float, float]:
        """
        Overview:
            Analyzes the captured features and gradients by computing their average L2 norms.
            This method clears the stored data after analysis to free memory.
        Returns:
            - (:obj:`Tuple[float, float, float, float]`): A tuple containing the L2 norms of
              (features_before, features_after, grads_before, grads_after).
        """
        if not self.features_before:
            return 0.0, 0.0, 0.0, 0.0

        l2_norm_before = torch.mean(torch.stack([torch.norm(f, p=2) for f in self.features_before])).item()
        l2_norm_after = torch.mean(torch.stack([torch.norm(f, p=2) for f in self.features_after])).item()

        valid_grads_before = [g for g in self.grads_before if g is not None]
        grad_norm_before = torch.mean(torch.stack([torch.norm(g, p=2) for g in valid_grads_before])).item() if valid_grads_before else 0.0

        valid_grads_after = [g for g in self.grads_after if g is not None]
        grad_norm_after = torch.mean(torch.stack([torch.norm(g, p=2) for g in valid_grads_after])).item() if valid_grads_after else 0.0

        self.clear_data()
        return l2_norm_before, l2_norm_after, grad_norm_before, grad_norm_after

    def clear_data(self) -> None:
        """Clears all stored feature and gradient tensors to free up memory."""
        self.features_before.clear()
        self.features_after.clear()
        self.grads_before.clear()
        self.grads_after.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def remove_hooks(self) -> None:
        """Removes the registered forward and backward hooks."""
        self.forward_handler.remove()
        self.backward_handler.remove()


class DownSample(nn.Module):
    """
    Overview:
        A convolutional network for downsampling image-based observations, commonly used in Atari environments.
        It consists of a series of convolutional, normalization, and residual blocks.
    """

    def __init__(
            self,
            observation_shape: Sequence[int],
            out_channels: int,
            activation: nn.Module = nn.ReLU(inplace=True),
            norm_type: str = 'BN',
            num_resblocks: int = 1,
    ) -> None:
        """
        Arguments:
            - observation_shape (:obj:`Sequence[int]`): The shape of the input observation, e.g., (C, H, W).
            - out_channels (:obj:`int`): The number of output channels.
            - activation (:obj:`nn.Module`): The activation function to use.
            - norm_type (:obj:`str`): The type of normalization ('BN' or 'LN').
            - num_resblocks (:obj:`int`): The number of residual blocks in each stage.
        """
        super().__init__()
        if norm_type not in ['BN', 'LN']:
            raise ValueError(f"Unsupported norm_type: {norm_type}. Must be 'BN' or 'LN'.")
        # The original design was fixed to 1 resblock per stage.
        if num_resblocks != 1:
            logging.warning(f"DownSample is designed for num_resblocks=1, but got {num_resblocks}.")

        self.observation_shape = observation_shape
        self.activation = activation

        # Initial convolution: stride 2
        self.conv1 = nn.Conv2d(observation_shape[0], out_channels // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm1 = build_normalization(norm_type, dim=2)(out_channels // 2)

        # Stage 1 with residual blocks
        self.resblocks1 = nn.ModuleList([
            ResBlock(in_channels=out_channels // 2, activation=activation, norm_type=norm_type, res_type='basic', bias=False)
            for _ in range(num_resblocks)
        ])
        
        # Downsample block: stride 2
        self.downsample_block = ResBlock(in_channels=out_channels // 2, out_channels=out_channels, activation=activation, norm_type=norm_type, res_type='downsample', bias=False)
        
        # Stage 2 with residual blocks
        self.resblocks2 = nn.ModuleList([
            ResBlock(in_channels=out_channels, activation=activation, norm_type=norm_type, res_type='basic', bias=False)
            for _ in range(num_resblocks)
        ])
        
        # Pooling 1: stride 2
        self.pooling1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        
        # Stage 3 with residual blocks
        self.resblocks3 = nn.ModuleList([
            ResBlock(in_channels=out_channels, activation=activation, norm_type=norm_type, res_type='basic', bias=False)
            for _ in range(num_resblocks)
        ])
        
        # Final pooling for specific input sizes
        self.pooling2 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Shapes:
            - x (:obj:`torch.Tensor`): (B, C_in, H, W)
            - output (:obj:`torch.Tensor`): (B, C_out, H_out, W_out)
        x = self.norm1(x)
        """
        x = self.conv1(x)
        x = self.activation(x)

        for block in self.resblocks1:
            x = block(x)
        
        x = self.downsample_block(x)
        for block in self.resblocks2:
            x = block(x)
        
        x = self.pooling1(x)
        for block in self.resblocks3:
            x = block(x)

        # This part handles specific Atari resolutions. A more general approach might be desirable,
        # but we maintain original behavior.
        obs_height = self.observation_shape[1]
        if obs_height == 64:
            return x
        elif obs_height in [84, 96]:
            return self.pooling2(x)
        else:
            raise NotImplementedError(
                f"DownSample for observation height {obs_height} is not implemented. "
                f"Supported heights are 64, 84, 96."
            )

class QwenNetwork(nn.Module):
    def __init__(self,
                 model_path: str = 'Qwen/Qwen3-1.7B',
                 embedding_size: int = 768,
                 final_norm_option_in_encoder: str = "layernorm",
                 group_size: int = 8,
                 tokenizer=None):
        super().__init__()

        logging.info(f"Loading Qwen model from: {model_path}")
        
        local_rank = get_rank()
        if local_rank == 0:
            self.pretrained_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype="auto", 
                device_map={"": local_rank},
                attn_implementation="flash_attention_2"
            )
        if get_world_size() > 1:
            torch.distributed.barrier()
        if local_rank != 0:
            self.pretrained_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype="auto",
                device_map={"": local_rank}, 
                attn_implementation="flash_attention_2"
            )
        
        for p in self.pretrained_model.parameters():
            p.requires_grad = False

        if tokenizer is None:
            if local_rank == 0:
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            if get_world_size() > 1:
                torch.distributed.barrier()
            if local_rank != 0:
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        else:
            self.tokenizer = tokenizer

        qwen_hidden_size = self.pretrained_model.config.hidden_size

        self.embedding_head = nn.Sequential(
            nn.Linear(qwen_hidden_size, embedding_size),
            self._create_norm_layer(final_norm_option_in_encoder, embedding_size, group_size)
        )

    def _create_norm_layer(self, norm_option, embedding_size, group_size):
        if norm_option.lower() == "simnorm":
            return SimNorm(simnorm_dim=group_size)
        elif norm_option.lower() == "layernorm":
            return nn.LayerNorm(embedding_size)
        else:
            raise NotImplementedError(f"Normalization type '{norm_option}' is not implemented.")

    def encode(self, x: torch.Tensor, no_grad: bool = True) -> torch.Tensor:
        """
        Overview:
            Encode the input token sequence `x` into a latent representation
            using a pretrained language model backbone followed by a projection head.
        Arguments:
            - x (:obj:`torch.Tensor`):  Input token ids of shape (B, L)
            - no_grad (:obj:`bool`, optional, default=True): If True, encoding is performed under `torch.no_grad()` to save memory and computation (no gradient tracking).
        Returns:
            - latent (:obj:`torch.Tensor`): Encoded latent state of shape (B, D).
        """
        pad_id = self.tokenizer.pad_token_id
        attention_mask = (x != pad_id).long().to(x.device)
        context = {'input_ids': x.long(), 'attention_mask': attention_mask}
        if no_grad:
            with torch.no_grad():
                outputs = self.pretrained_model(**context, output_hidden_states=True, return_dict=True)
        else:
            outputs = self.pretrained_model(**context, output_hidden_states=True, return_dict=True)
        last_hidden = outputs.hidden_states[-1]

        B, L, H = last_hidden.size()
        lengths = attention_mask.sum(dim=1)  # [B]
        positions = torch.clamp(lengths - 1, min=0)  # [B]
        batch_idx = torch.arange(B, device=last_hidden.device)

        selected = last_hidden[batch_idx, positions]  # [B, H]

        latent = self.embedding_head(selected.to(self.embedding_head[0].weight.dtype))
        return latent

    def decode(self, embeddings: torch.Tensor, max_length: int = 512) -> str:
        """
        Decodes embeddings into text via the decoder network.
        """
        embeddings_detached = embeddings.detach()
        self.pretrained_model.eval()
        
        # Directly generate using provided embeddings
        with torch.no_grad():
            param = next(self.pretrained_model.parameters())
            embeddings = embeddings_detached.to(device=param.device, dtype=param.dtype)
            gen_ids = self.pretrained_model.generate(
                inputs_embeds=embeddings,
                max_length=max_length
            )
        texts = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        self.pretrained_model.train()
        return texts[0] if len(texts) == 1 else texts

    def forward(self, x: torch.Tensor, no_grad: bool = True) -> torch.Tensor:
        return self.encode(x, no_grad=no_grad)


class HFLanguageRepresentationNetwork(nn.Module):
    def __init__(self,
                model_path: str = 'google-bert/bert-base-uncased',
                embedding_size: int = 768,
                group_size: int = 8,
                final_norm_option_in_encoder: str = "layernorm",
                tokenizer=None):
        """
        Arguments:
            - model_path (str): The path to the pretrained Hugging Face model. Default is 'google-bert/bert-base-uncased'.
            - embedding_size (int): The dimension of the output embeddings. Default is 768.
            - group_size (int): The group size for SimNorm when using normalization.
            - final_norm_option_in_encoder (str): The type of normalization to use ("simnorm" or "layernorm"). Default is "layernorm".
            - tokenizer (Optional): An instance of a tokenizer. If None, the tokenizer will be loaded from the pretrained model.
        """
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        # In distributed settings, ensure only rank 0 downloads the model/tokenizer.
        if get_rank() == 0:
            self.pretrained_model = AutoModel.from_pretrained(model_path)

        if get_world_size() > 1:
            # Wait for rank 0 to finish loading the model.
            torch.distributed.barrier()
        if get_rank() != 0:
            self.pretrained_model = AutoModel.from_pretrained(model_path)

        if get_rank() != 0:
            logging.info(f"Worker process is loading model from cache: {model_path}")
            self.model = AutoModel.from_pretrained(model_path)
            if tokenizer is None:
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        if tokenizer is not None:
            self.tokenizer = tokenizer

        self.embedding_size = embedding_size
        self.embed_proj_head = nn.Linear(self.pretrained_model.config.hidden_size, self.embedding_size)

        # # Select the normalization method based on the final_norm_option_in_encoder parameter.
        if final_norm_option_in_encoder.lower() == "simnorm":
            self.norm = SimNorm(simnorm_dim=group_size)
        elif final_norm_option_in_encoder.lower() == "layernorm":
            self.norm = nn.LayerNorm(embedding_size)
        else:
            raise NotImplementedError(f"Normalization type '{final_norm_option_in_encoder}' is not implemented. "
                                      f"Choose 'simnorm' or 'layernorm'.")

    def forward(self, x: torch.Tensor, no_grad: bool = True) -> torch.Tensor:
        """
        Overview:
            Computes language representation from input token IDs.
        Arguments:
            - x (:obj:`torch.Tensor`): Input token sequence of shape (B, seq_len).
            - no_grad (:obj:`bool`): If True, run the transformer model in `torch.no_grad()` context.
        Returns:
            - (:obj:`torch.Tensor`): The final language embedding of shape (B, embedding_size).
        """

        # Construct the attention mask to exclude padding tokens.
        attention_mask = x != self.tokenizer.pad_token_id

        if no_grad:
            with torch.no_grad():
                x = x.long()  # Ensure the input tensor is of type long.
                outputs = self.pretrained_model(x, attention_mask=attention_mask)
                # Get the hidden state from the last layer and select the output corresponding to the [CLS] token.
                cls_embedding = outputs.last_hidden_state[:, 0, :]
        else:
            x = x.long()
            outputs = self.pretrained_model(x, attention_mask=attention_mask)
            cls_embedding = outputs.last_hidden_state[:, 0, :]

        cls_embedding = self.embed_proj_head(cls_embedding)
        cls_embedding = self.norm(cls_embedding)
        
        return cls_embedding


class RepresentationNetworkUniZero(nn.Module):
    
    def __init__(
            self,
            observation_shape: SequenceType = (3, 64, 64),
            num_res_blocks: int = 1,
            num_channels: int = 64,
            downsample: bool = True,
            activation: nn.Module = nn.GELU(approximate='tanh'),
            norm_type: str = 'BN',
            embedding_dim: int = 256,
            group_size: int = 8,
            final_norm_option_in_encoder: str = 'LayerNorm', # TODO
    ) -> None:
        """
        Overview:
            Representation network used in UniZero. Encode the 2D image obs into latent state.
            Currently, the network only supports obs images with both a width and height of 64.
        Arguments:
            - observation_shape (:obj:`SequenceType`): The shape of observation space, e.g. [C, W, H]=[3, 64, 64]
                for video games like atari, RGB 3 channel.
            - num_res_blocks (:obj:`int`): The number of residual blocks.
            - num_channels (:obj:`int`): The channel of output hidden state.
            - downsample (:obj:`bool`): Whether to do downsampling for observations in ``representation_network``, \
                defaults to True. This option is often used in video games like Atari. In board games like go, \
                we don't need this module.
            - activation (:obj:`nn.Module`): The activation function used in network, defaults to nn.ReLU(inplace=True). \
                Use the inplace operation to speed up.
            - norm_type (:obj:`str`): The type of normalization in networks. defaults to 'BN'.
            - embedding_dim (:obj:`int`): The dimension of the latent state.
            - group_size (:obj:`int`): The dimension for simplicial normalization.
            - final_norm_option_in_encoder (:obj:`str`): The normalization option for the final layer, defaults to 'SimNorm'. \
                Options are 'SimNorm' and 'LayerNorm'.
        """
        super().__init__()
        assert norm_type in ['BN', 'LN'], "norm_type must in ['BN', 'LN']"
        logging.info(f"Using norm type: {norm_type}")
        logging.info(f"Using activation type: {activation}")

        self.observation_shape = observation_shape
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
        self.embedding_dim = embedding_dim

        # ==================== 修改开始 ====================
        if self.observation_shape[1] == 64:
            # 修复：将硬编码的 64 替换为 num_channels
            self.last_linear = nn.Linear(num_channels * 8 * 8, self.embedding_dim, bias=False)

        elif self.observation_shape[1] in [84, 96]:
            # 修复：将硬编码的 64 替换为 num_channels
            self.last_linear = nn.Linear(num_channels * 6 * 6, self.embedding_dim, bias=False)
        # ==================== 修改结束 ====================

        self.final_norm_option_in_encoder=final_norm_option_in_encoder 
        # 2. 在 __init__ 中统一初始化 final_norm
        if self.final_norm_option_in_encoder in ['LayerNorm', 'LayerNorm_Tanh']:
            self.final_norm = nn.LayerNorm(self.embedding_dim, eps=1e-5)
        elif self.final_norm_option_in_encoder == 'LayerNormNoAffine':
            self.final_norm = nn.LayerNorm(
                self.embedding_dim, eps=1e-5, elementwise_affine=False
            )
        elif self.final_norm_option_in_encoder == 'SimNorm':
            # 确保 SimNorm 已被定义
            self.final_norm = SimNorm(simnorm_dim=group_size)
        elif self.final_norm_option_in_encoder == 'L2Norm':
            # 直接实例化我们自定义的 L2Norm 模块
            self.final_norm = L2Norm(eps=1e-6)
        elif self.final_norm_option_in_encoder is None:
            # 如果不需要归一化，可以设置为 nn.Identity() 或 None
            self.final_norm = nn.Identity()
        else:
            raise ValueError(f"Unsupported final_norm_option_in_encoder: {self.final_norm_option_in_encoder}")
            
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

        # Important: Transform the output feature plane to the latent state.
        # For example, for an Atari feature plane of shape (64, 8, 8),
        # flattening results in a size of 4096, which is then transformed to 768.
        x = self.last_linear(x.view(x.size(0), -1))

        x = x.view(-1, self.embedding_dim)

        # NOTE: very important for training stability.
        # x = self.final_norm(x)

        # 3. 在 forward 中统一调用 self.final_norm
        # 这种结构更加清晰和可扩展
        if self.final_norm is not None:
            x = self.final_norm(x)

        # 针对 LayerNorm_Tanh 的特殊处理
        if self.final_norm_option_in_encoder == 'LayerNorm_Tanh':
            x = torch.tanh(x)

        return x


class RepresentationNetwork(nn.Module):
    """
    Overview:
        The standard representation network used in MuZero. It encodes a 2D image observation
        into a latent state, which retains its spatial dimensions.
    """
    def __init__(
            self,
            observation_shape: Sequence[int] = (4, 96, 96),
            num_res_blocks: int = 1,
            num_channels: int = 64,
            downsample: bool = True,
            activation: nn.Module = nn.ReLU(inplace=True),
            norm_type: str = 'BN',
            use_sim_norm: bool = False,
            group_size: int = 8,
    ) -> None:
        """
        Arguments:
            - observation_shape (:obj:`Sequence[int]`): Shape of the input observation (C, H, W).
            - num_res_blocks (:obj:`int`): The number of residual blocks.
            - num_channels (:obj:`int`): The number of channels in the convolutional layers.
            - downsample (:obj:`bool`): Whether to use the `DownSample` module.
            - activation (:obj:`nn.Module`): The activation function to use.
            - norm_type (:obj:`str`): Normalization type ('BN' or 'LN').
            - use_sim_norm (:obj:`bool`): Whether to apply a final `SimNorm` layer.
            - group_size (:obj:`int`): Group size for `SimNorm`.
        """
        super().__init__()
        if norm_type not in ['BN', 'LN']:
            raise ValueError(f"Unsupported norm_type: {norm_type}. Must be 'BN' or 'LN'.")

        self.downsample = downsample
        self.activation = activation

        if self.downsample:
            self.downsample_net = DownSample(observation_shape, num_channels, activation, norm_type)
        else:
            self.conv = nn.Conv2d(observation_shape[0], num_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.norm = build_normalization(norm_type, dim=3)(num_channels, *observation_shape[1:])

        self.resblocks = nn.ModuleList([
            ResBlock(in_channels=num_channels, activation=activation, norm_type=norm_type, res_type='basic', bias=False)
            for _ in range(num_res_blocks)
        ])

        self.use_sim_norm = use_sim_norm
        if self.use_sim_norm:
            self.sim_norm = SimNorm(simnorm_dim=group_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Shapes:
            - x (:obj:`torch.Tensor`): (B, C_in, H, W)
            - output (:obj:`torch.Tensor`): (B, C_out, H_out, W_out)
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
            # Flatten the spatial dimensions, apply SimNorm, and then reshape back.
            b, c, h, w = x.shape
            x_flat = x.view(b, c * h * w)
            x_norm = self.sim_norm(x_flat)
            x = x_norm.view(b, c, h, w)
            
        return x


class RepresentationNetworkMLP(nn.Module):
    """
    Overview:
        An MLP-based representation network for encoding vector observations into a latent state.
    """
    def __init__(
            self,
            observation_dim: int,
            hidden_channels: int = 64,
            num_layers: int = 2,
            activation: nn.Module = nn.GELU(approximate='tanh'),
            norm_type: Optional[str] = 'BN',
            group_size: int = 8,
            final_norm_option_in_encoder: str = 'LayerNorm', # TODO
    ) -> torch.Tensor:
        """
        Arguments:
            - observation_dim (:obj:`int`): The dimension of the input vector observation.
            - hidden_channels (:obj:`int`): The number of neurons in the hidden and output layers.
            - num_layers (:obj:`int`): The total number of layers in the MLP.
            - activation (:obj:`nn.Module`): The activation function to use.
            - norm_type (:obj:`Optional[str]`): The type of normalization ('BN', 'LN', or None).
            - group_size (:obj:`int`): The group size for the final `SimNorm` layer.
        """
        super().__init__()
        # Creating hidden layers list for MLP_V2
        hidden_layers = [hidden_channels] * (num_layers - 1) if num_layers > 1 else []
        
        self.fc_representation = MLP_V2(
            in_channels=observation_dim,
            hidden_channels=hidden_layers,
            out_channels=hidden_channels,
            activation=activation,
            norm_type=norm_type,
            output_activation=False,
            output_norm=False,
            last_linear_layer_init_zero=True,
        )

        # # Select the normalization method based on the final_norm_option_in_encoder parameter.
        if final_norm_option_in_encoder.lower() == "simnorm":
            self.norm = SimNorm(simnorm_dim=group_size)
        elif final_norm_option_in_encoder.lower() == "layernorm":
            self.norm = nn.LayerNorm(hidden_channels)
        else:
            raise NotImplementedError(f"Normalization type '{final_norm_option_in_encoder}' is not implemented. "
                                      f"Choose 'simnorm' or 'layernorm'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Shapes:
            - x (:obj:`torch.Tensor`): (B, observation_dim)
            - output (:obj:`torch.Tensor`): (B, hidden_channels)
        """
        x = self.fc_representation(x)
        x = self.norm(x)

        return x


class LatentDecoder(nn.Module):
    """
    Overview:
        A decoder network that reconstructs a 2D image from a 1D latent embedding.
        It acts as the inverse of a representation network like `RepresentationNetworkUniZero`.
    """
    def __init__(
        self,
        embedding_dim: int,
        output_shape: Tuple[int, int, int],
        num_channels: int = 64,
        activation: nn.Module = nn.GELU(approximate='tanh')
    ):
        """
        Arguments:
            - embedding_dim (:obj:`int`): The dimension of the input latent embedding.
            - output_shape (:obj:`Tuple[int, int, int]`): The shape of the target output image (C, H, W).
            - num_channels (:obj:`int`): The base number of channels for the initial upsampling stage.
            - activation (:obj:`nn.Module`): The activation function to use.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.output_shape = output_shape
        
        # This should match the spatial size of the encoder's feature map before flattening.
        # Assuming a total downsampling factor of 8 (e.g., for a 64x64 -> 8x8 encoder).
        self.initial_h = output_shape[1] // 8
        self.initial_w = output_shape[2] // 8
        self.initial_size = (num_channels, self.initial_h, self.initial_w)
        
        self.fc = nn.Linear(embedding_dim, np.prod(self.initial_size))

        self.deconv_blocks = nn.Sequential(
            # Block 1: (C, H/8, W/8) -> (C/2, H/4, W/4)
            nn.ConvTranspose2d(num_channels, num_channels // 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            activation,
            nn.BatchNorm2d(num_channels // 2),
            # Block 2: (C/2, H/4, W/4) -> (C/4, H/2, W/2)
            nn.ConvTranspose2d(num_channels // 2, num_channels // 4, kernel_size=3, stride=2, padding=1, output_padding=1),
            activation,
            nn.BatchNorm2d(num_channels // 4),
            # Block 3: (C/4, H/2, W/2) -> (output_C, H, W)
            nn.ConvTranspose2d(num_channels // 4, output_shape[0], kernel_size=3, stride=2, padding=1, output_padding=1),
            # A final activation like Sigmoid or Tanh is often used if pixel values are in a fixed range [0,1] or [-1,1].
            # We omit it here to maintain consistency with the original code.
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Shapes:
            - embeddings (:obj:`torch.Tensor`): (B, embedding_dim)
            - output (:obj:`torch.Tensor`): (B, C, H, W)
        """
        x = self.fc(embeddings)
        x = x.view(-1, *self.initial_size)
        x = self.deconv_blocks(x)
        return x


# --- Networks for MemoryEnv ---

class LatentEncoderForMemoryEnv(nn.Module):
    """
    Overview:
        An encoder for the MemoryEnv, converting a small image observation into a latent embedding.
        It uses a series of convolutions followed by adaptive average pooling.
    """
    def __init__(
            self,
            image_shape: Tuple[int, int, int] = (3, 5, 5),
            embedding_size: int = 100,
            channels: List[int] = [16, 32, 64],
            kernel_sizes: List[int] = [3, 3, 3],
            strides: List[int] = [1, 1, 1],
            activation: nn.Module = nn.GELU(approximate='tanh'),
            normalize_pixel: bool = False,
            group_size: int = 8,
    ):
        """
        Arguments:
            - image_shape (:obj:`Tuple[int, int, int]`): Shape of the input image (C, H, W).
            - embedding_size (:obj:`int`): Dimension of the output latent embedding.
            - channels (:obj:`List[int]`): List of output channels for each convolutional layer.
            - kernel_sizes (:obj:`List[int]`): List of kernel sizes for each convolutional layer.
            - strides (:obj:`List[int]`): List of strides for each convolutional layer.
            - activation (:obj:`nn.Module`): Activation function to use.
            - normalize_pixel (:obj:`bool`): Whether to normalize input pixel values to [0, 1].
            - group_size (:obj:`int`): Group size for the final `SimNorm` layer.
        """
        super().__init__()
        self.normalize_pixel = normalize_pixel
        all_channels = [image_shape[0]] + channels

        layers = []
        for i in range(len(channels)):
            layers.extend([
                nn.Conv2d(all_channels[i], all_channels[i+1], kernel_sizes[i], strides[i], padding=kernel_sizes[i]//2),
                nn.BatchNorm2d(all_channels[i+1]),
                activation
            ])
        layers.append(nn.AdaptiveAvgPool2d(1))
        self.cnn = nn.Sequential(*layers)
        
        self.linear = nn.Linear(channels[-1], embedding_size, bias=False)
        init.kaiming_normal_(self.linear.weight, mode='fan_out', nonlinearity='relu')

        self.sim_norm = SimNorm(simnorm_dim=group_size)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Shapes:
            - image (:obj:`torch.Tensor`): (B, C, H, W)
            - output (:obj:`torch.Tensor`): (B, embedding_size)
        """
        if self.normalize_pixel:
            image = image.float() / 255.0
        
        x = self.cnn(image.float())
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        x = self.sim_norm(x)
        return x


class LatentDecoderForMemoryEnv(nn.Module):
    """
    Overview:
        A decoder for the MemoryEnv, reconstructing a small image from a latent embedding.
        It uses a linear layer followed by a series of transposed convolutions.
    """
    def __init__(
            self,
            image_shape: Tuple[int, int, int] = (3, 5, 5),
            embedding_size: int = 256,
            channels: List[int] = [64, 32, 16],
            kernel_sizes: List[int] = [3, 3, 3],
            strides: List[int] = [1, 1, 1],
            activation: nn.Module = nn.LeakyReLU(negative_slope=0.01),
    ):
        """
        Arguments:
            - image_shape (:obj:`Tuple[int, int, int]`): Shape of the target output image (C, H, W).
            - embedding_size (:obj:`int`): Dimension of the input latent embedding.
            - channels (:obj:`List[int]`): List of channels for each deconvolutional layer.
            - kernel_sizes (:obj:`List[int]`): List of kernel sizes.
            - strides (:obj:`List[int]`): List of strides.
            - activation (:obj:`nn.Module`): Activation function for intermediate layers.
        """
        super().__init__()
        self.shape = image_shape
        self.deconv_channels = channels + [image_shape[0]]
        
        self.linear = nn.Linear(embedding_size, channels[0] * image_shape[1] * image_shape[2])

        layers = []
        for i in range(len(self.deconv_channels) - 1):
            layers.append(
                nn.ConvTranspose2d(
                    self.deconv_channels[i], self.deconv_channels[i+1], kernel_sizes[i], strides[i],
                    padding=kernel_sizes[i]//2, output_padding=strides[i]-1
                )
            )
            if i < len(self.deconv_channels) - 2:
                layers.extend([nn.BatchNorm2d(self.deconv_channels[i+1]), activation])
            else:
                # Final layer uses Sigmoid to output pixel values in [0, 1].
                layers.append(nn.Sigmoid())
        self.deconv = nn.Sequential(*layers)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Shapes:
            - embedding (:obj:`torch.Tensor`): (B, embedding_size)
            - output (:obj:`torch.Tensor`): (B, C, H, W)
        """
        x = self.linear(embedding)
        x = x.view(-1, self.deconv_channels[0], self.shape[1], self.shape[2])
        x = self.deconv(x)
        return x


class VectorDecoderForMemoryEnv(nn.Module):
    """
    Overview:
        An MLP-based decoder for MemoryEnv, reconstructing a vector observation from a latent embedding.
    """
    def __init__(
            self,
            embedding_dim: int,
            output_dim: int,
            hidden_channels: int = 64,
            num_layers: int = 2,
            activation: nn.Module = nn.LeakyReLU(negative_slope=0.01),
            norm_type: Optional[str] = 'BN',
    ) -> None:
        """
        Arguments:
            - embedding_dim (:obj:`int`): Dimension of the input latent embedding.
            - output_dim (:obj:`int`): Dimension of the target output vector.
            - hidden_channels (:obj:`int`): Number of neurons in the hidden layers.
            - num_layers (:obj:`int`): Total number of layers in the MLP.
            - activation (:obj:`nn.Module`): Activation function to use.
            - norm_type (:obj:`Optional[str]`): Normalization type ('BN', 'LN', or None).
        """
        super().__init__()
        hidden_layers = [hidden_channels] * (num_layers - 1) if num_layers > 1 else []
        
        self.fc_decoder = MLP_V2(
            in_channels=embedding_dim,
            hidden_channels=hidden_layers,
            out_channels=output_dim,
            activation=activation,
            norm_type=norm_type,
            output_activation=False,
            output_norm=False,
            last_linear_layer_init_zero=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Shapes:
            - x (:obj:`torch.Tensor`): (B, embedding_dim)
            - output (:obj:`torch.Tensor`): (B, output_dim)
        """
        return self.fc_decoder(x)

# --- Prediction Networks ---

class PredictionNetwork(nn.Module):
    """
    Overview:
        Predicts the policy and value from a given latent state. This network is typically used
        in the prediction step of MuZero-like algorithms. It processes a 2D latent state.
    """
    def __init__(
            self,
            action_space_size: int,
            num_res_blocks: int,
            num_channels: int,
            value_head_channels: int = 1,
            policy_head_channels: int = 2,
            value_head_hidden_channels: List[int] = [256],
            policy_head_hidden_channels: List[int] = [256],
            output_support_size: int = 601,
            last_linear_layer_init_zero: bool = True,
            activation: nn.Module = nn.ReLU(inplace=True),
            norm_type: str = 'BN',
    ) -> None:
        """
        Arguments:
            - action_space_size: (:obj:`int`): The size of the action space.
            - num_res_blocks (:obj:`int`): The number of residual blocks.
            - num_channels (:obj:`int`): The number of channels in the input latent state.
            - value_head_channels (:obj:`int`): Channels for the value head's convolutional layer.
            - policy_head_channels (:obj:`int`): Channels for the policy head's convolutional layer.
            - value_head_hidden_channels (:obj:`List[int]`): Hidden layer sizes for the value MLP head.
            - policy_head_hidden_channels (:obj:`List[int]`): Hidden layer sizes for the policy MLP head.
            - output_support_size (:obj:`int`): The size of the categorical value distribution.
            - last_linear_layer_init_zero (:obj:`bool`): Whether to initialize the last layer of heads to zero.
            - activation (:obj:`nn.Module`): The activation function.
            - norm_type (:obj:`str`): The normalization type ('BN' or 'LN').
        """
        super().__init__()
        if norm_type not in ['BN', 'LN']:
            raise ValueError(f"Unsupported norm_type: {norm_type}. Must be 'BN' or 'LN'.")

        self.resblocks = nn.ModuleList([
            ResBlock(in_channels=num_channels, activation=activation, norm_type=norm_type, res_type='basic', bias=False)
            for _ in range(num_res_blocks)
        ])
        
        self.conv1x1_value = nn.Conv2d(num_channels, value_head_channels, 1)
        self.conv1x1_policy = nn.Conv2d(num_channels, policy_head_channels, 1)

        self.norm_value = build_normalization(norm_type, dim=2)(value_head_channels)
        self.norm_policy = build_normalization(norm_type, dim=2)(policy_head_channels)
        self.activation = activation

        # The input size for the MLP heads depends on the spatial dimensions of the latent state.
        # This must be pre-calculated and passed correctly.
        # Example: for a 6x6 latent space, flatten_input_size = channels * 6 * 6
        # We assume the user will provide these values.
        # Here we just define placeholder attributes.
        self._flatten_input_size_for_value_head = None
        self._flatten_input_size_for_policy_head = None

        self.fc_value = MLP_V2(
            in_channels=-1, # Placeholder, will be determined at first forward pass
            hidden_channels=value_head_hidden_channels,
            out_channels=output_support_size,
            activation=activation,
            norm_type=norm_type,
            output_activation=False,
            output_norm=False,
            last_linear_layer_init_zero=last_linear_layer_init_zero
        )
        self.fc_policy = MLP_V2(
            in_channels=-1, # Placeholder
            hidden_channels=policy_head_hidden_channels,
            out_channels=action_space_size,
            activation=activation,
            norm_type=norm_type,
            output_activation=False,
            output_norm=False,
            last_linear_layer_init_zero=last_linear_layer_init_zero
        )

    def forward(self, latent_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Shapes:
            - latent_state (:obj:`torch.Tensor`): (B, C, H, W)
            - policy_logits (:obj:`torch.Tensor`): (B, action_space_size)
            - value (:obj:`torch.Tensor`): (B, output_support_size)
        """
        for res_block in self.resblocks:
            latent_state = res_block(latent_state)

        value_feat = self.activation(self.norm_value(self.conv1x1_value(latent_state)))
        policy_feat = self.activation(self.norm_policy(self.conv1x1_policy(latent_state)))
        
        value_flat = value_feat.view(value_feat.size(0), -1)
        policy_flat = policy_feat.view(policy_feat.size(0), -1)

        # Dynamically initialize in_channels on the first forward pass
        if self.fc_value.in_channels == -1:
            self.fc_value[0].in_features = value_flat.shape[1]
            self.fc_policy[0].in_features = policy_flat.shape[1]
            # PyTorch lazy modules handle this better, but this is a manual way.
            self.fc_value[0].weight.data.uniform_(-math.sqrt(1/value_flat.shape[1]), math.sqrt(1/value_flat.shape[1]))
            self.fc_policy[0].weight.data.uniform_(-math.sqrt(1/policy_flat.shape[1]), math.sqrt(1/policy_flat.shape[1]))


        value = self.fc_value(value_flat)
        policy_logits = self.fc_policy(policy_flat)
        return policy_logits, value


class PredictionNetworkMLP(nn.Module):
    """
    Overview:
        An MLP-based prediction network that predicts policy and value from a 1D latent state.
    """
    def __init__(
            self,
            action_space_size: int,
            num_channels: int,
            common_layer_num: int = 2,
            value_head_hidden_channels: List[int] = [32],
            policy_head_hidden_channels: List[int] = [32],
            output_support_size: int = 601,
            last_linear_layer_init_zero: bool = True,
            activation: nn.Module = nn.ReLU(inplace=True),
            norm_type: Optional[str] = 'BN',
    ):
        """
        Arguments:
            - action_space_size: (:obj:`int`): The size of the action space.
            - num_channels (:obj:`int`): The dimension of the input latent state.
            - common_layer_num (:obj:`int`): Number of layers in the shared backbone MLP.
            - value_head_hidden_channels (:obj:`List[int]`): Hidden layer sizes for the value MLP head.
            - policy_head_hidden_channels (:obj:`List[int]`): Hidden layer sizes for the policy MLP head.
            - output_support_size (:obj:`int`): The size of the categorical value distribution.
            - last_linear_layer_init_zero (:obj:`bool`): Whether to initialize the last layer of heads to zero.
            - activation (:obj:`nn.Module`): The activation function.
            - norm_type (:obj:`Optional[str]`): The normalization type.
        """
        super().__init__()
        
        common_hidden = [num_channels] * (common_layer_num - 1) if common_layer_num > 1 else []
        self.fc_prediction_common = MLP_V2(
            in_channels=num_channels,
            hidden_channels=common_hidden,
            out_channels=num_channels,
            activation=activation,
            norm_type=norm_type,
            output_activation=True,
            output_norm=True,
            last_linear_layer_init_zero=False,
        )

        self.fc_value_head = MLP_V2(
            in_channels=num_channels,
            hidden_channels=value_head_hidden_channels,
            out_channels=output_support_size,
            activation=activation,
            norm_type=norm_type,
            output_activation=False,
            output_norm=False,
            last_linear_layer_init_zero=last_linear_layer_init_zero
        )
        self.fc_policy_head = MLP_V2(
            in_channels=num_channels,
            hidden_channels=policy_head_hidden_channels,
            out_channels=action_space_size,
            activation=activation,
            norm_type=norm_type,
            output_activation=False,
            output_norm=False,
            last_linear_layer_init_zero=last_linear_layer_init_zero
        )

    def forward(self, latent_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Shapes:
            - latent_state (:obj:`torch.Tensor`): (B, num_channels)
            - policy_logits (:obj:`torch.Tensor`): (B, action_space_size)
            - value (:obj:`torch.Tensor`): (B, output_support_size)
        """
        x = self.fc_prediction_common(latent_state)
        value = self.fc_value_head(x)
        policy_logits = self.fc_policy_head(x)
        return policy_logits, value
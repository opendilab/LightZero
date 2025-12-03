# -*- coding: utf-8 -*-
"""
Optimized Vision Transformer (ViT) Model.

This script provides an optimized implementation of the Vision Transformer (ViT) architecture.
It includes improvements in code structure, clarity, and adherence to modern Python coding standards,
including comprehensive type hinting and documentation. The implementation also supports
integration with Low-Rank Adaptation (LoRA) through a flexible configuration system.

Author: [Your Name/Team Name]
Date: [Current Date]
"""

import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from lzero.model.common import SimNorm
from typing import Tuple, Union, Type, Optional

# ==================== LoRA Integration Section Start ====================

# Attempt to import core components from a local transformer.py for LoRA support.
# This allows for flexible adaptation (e.g., LoRA) of linear layers.
try:
    # Assuming transformer.py is in the same directory. Adjust the import path if necessary.
    from .transformer import _maybe_wrap_linear, TransformerConfig
except ImportError:
    # If the import fails (e.g., when running this file directly), provide a fallback.
    # This ensures the model remains functional without LoRA components.
    print("Warning: LoRA components could not be imported. Using standard nn.Linear.")
    _maybe_wrap_linear = lambda linear, config, label: linear
    
    # Define a placeholder class for TransformerConfig if it's not available.
    class TransformerConfig:
        """Placeholder for TransformerConfig when LoRA components are not available."""
        pass

# ==================== LoRA Integration Section End ====================


# ==================== Configuration Class ====================

class ViTConfig:
    """
    Overview:
        Configuration class for the Vision Transformer (ViT) model.
        This class centralizes all hyperparameters, making the model easier to configure and manage.
    """
    def __init__(self, **kwargs):
        """
        Overview:
            Initializes the ViTConfig object.
        Arguments:
            - **kwargs: Arbitrary keyword arguments to override default settings.
        """
        # Image and Patch Dimensions
        self.image_size: Union[int, Tuple[int, int]] = 64
        self.patch_size: Union[int, Tuple[int, int]] = 8
        self.channels: int = 3

        # Model Architecture
        self.num_classes: int = 768
        self.dim: int = 768
        self.depth: int = 12
        self.heads: int = 12
        self.mlp_dim: int = 3072
        self.dim_head: int = 64
        
        # Pooling and Normalization
        self.pool: str = 'cls'  # 'cls' or 'mean'
        self.final_norm_option_in_encoder: str = 'LayerNorm' # 'LayerNorm' or 'SimNorm'

        # Dropout Rates
        self.dropout: float = 0.1
        self.emb_dropout: float = 0.1
        
        # LoRA Configuration
        self.lora_config: Optional[TransformerConfig] = None

        # Update attributes with any provided keyword arguments
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: Ignoring unknown config parameter '{key}'")


# ==================== Helper Functions ====================

def pair(t: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    """
    Overview:
        Converts an integer to a tuple of two identical integers. If the input is already a tuple, it is returned as is.
        This is useful for handling kernel sizes, strides, etc., which can be specified as a single number or a tuple.
    Arguments:
        - t (:obj:`Union[int, Tuple[int, int]]`): The input value.
    Returns:
        - (:obj:`Tuple[int, int]`): A tuple of two integers.
    """
    return t if isinstance(t, tuple) else (t, t)


# ==================== Core Modules ====================

class FeedForward(nn.Module):
    """
    Overview:
        A standard feed-forward network block used in Transformer architectures.
        It consists of two linear layers with a GELU activation in between.
    """
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        config: Optional[TransformerConfig] = None
    ):
        """
        Overview:
            Initializes the FeedForward module.
        Arguments:
            - dim (:obj:`int`): The input and output dimension.
            - hidden_dim (:obj:`int`): The dimension of the hidden layer.
            - dropout (:obj:`float`): The dropout rate.
            - config (:obj:`Optional[TransformerConfig]`): Configuration for LoRA wrapping.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            _maybe_wrap_linear(nn.Linear(dim, hidden_dim), config, "feed_forward"),
            nn.GELU(),
            nn.Dropout(dropout),
            _maybe_wrap_linear(nn.Linear(hidden_dim, dim), config, "feed_forward"),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Overview:
            Forward pass for the FeedForward block.
        Arguments:
            - x (:obj:`torch.Tensor`): The input tensor of shape (batch_size, num_tokens, dim).
        Returns:
            - (:obj:`torch.Tensor`): The output tensor of the same shape as input.
        """
        return self.net(x)


class Attention(nn.Module):
    """
    Overview:
        Multi-Head Self-Attention (MHSA) module.
        It computes scaled dot-product attention across multiple heads.
    """
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        config: Optional[TransformerConfig] = None
    ):
        """
        Overview:
            Initializes the Attention module.
        Arguments:
            - dim (:obj:`int`): The input and output dimension.
            - heads (:obj:`int`): The number of attention heads.
            - dim_head (:obj:`int`): The dimension of each attention head.
            - dropout (:obj:`float`): The dropout rate for attention weights and output.
            - config (:obj:`Optional[TransformerConfig]`): Configuration for LoRA wrapping.
        """
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        # Linear layer to project input to Q, K, V. Potentially wrapped for LoRA.
        self.to_qkv = _maybe_wrap_linear(nn.Linear(dim, inner_dim * 3, bias=False), config, "attn")

        # Output projection layer.
        if project_out:
            # Wrap the linear layer inside the sequential module for LoRA.
            wrapped_linear = _maybe_wrap_linear(nn.Linear(inner_dim, dim), config, "attn")
            self.to_out = nn.Sequential(
                wrapped_linear,
                nn.Dropout(dropout)
            )
        else:
            self.to_out = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Overview:
            Forward pass for the Attention module.
        Arguments:
            - x (:obj:`torch.Tensor`): Input tensor of shape (batch_size, num_tokens, dim).
        Returns:
            - (:obj:`torch.Tensor`): Output tensor of the same shape as input.
        """
        x = self.norm(x)

        # Project to Q, K, V and split.
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        # Rearrange for multi-head attention: b n (h d) -> b h n d
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        # Scaled dot-product attention.
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)

        # Apply attention to values.
        out = torch.matmul(attn, v)
        # Rearrange back to original shape: b h n d -> b n (h d)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        return self.to_out(out)


class Transformer(nn.Module):
    """
    Overview:
        A stack of Transformer blocks, each containing a multi-head self-attention
        layer and a feed-forward network.
    """
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
        config: Optional[TransformerConfig] = None
    ):
        """
        Overview:
            Initializes the Transformer module.
        Arguments:
            - dim (:obj:`int`): The dimension of the token embeddings.
            - depth (:obj:`int`): The number of Transformer blocks.
            - heads (:obj:`int`): The number of attention heads.
            - dim_head (:obj:`int`): The dimension of each attention head.
            - mlp_dim (:obj:`int`): The hidden dimension of the feed-forward network.
            - dropout (:obj:`float`): The dropout rate.
            - config (:obj:`Optional[TransformerConfig]`): Configuration for LoRA.
        """
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout, config=config),
                FeedForward(dim, mlp_dim, dropout=dropout, config=config)
            ]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Overview:
            Forward pass for the Transformer stack.
        Arguments:
            - x (:obj:`torch.Tensor`): Input tensor of shape (batch_size, num_tokens, dim).
        Returns:
            - (:obj:`torch.Tensor`): Output tensor of the same shape.
        """
        for attn, ff in self.layers:
            x = attn(x) + x  # Apply attention and residual connection
            x = ff(x) + x    # Apply feed-forward and residual connection
        return self.norm(x)


class ViT(nn.Module):
    """
    Overview:
        Vision Transformer (ViT) model. This model applies the Transformer architecture
        to sequences of image patches for image classification tasks.
    """
    def __init__(self, config: ViTConfig):
        """
        Overview:
            Initializes the ViT model using a configuration object.
        Arguments:
            - config (:obj:`ViTConfig`): A configuration object containing all model hyperparameters.
        """
        super().__init__()
        self.config = config
        
        image_height, image_width = pair(config.image_size)
        patch_height, patch_width = pair(config.patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, \
            'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = config.channels * patch_height * patch_width
        assert config.pool in {'cls', 'mean'}, 'pool type must be either "cls" or "mean"'

        # Patch embedding layer
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, config.dim),
            nn.LayerNorm(config.dim),
        )

        # Positional embedding and CLS token
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, config.dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.dim))
        self.dropout = nn.Dropout(config.emb_dropout)

        # Transformer encoder stack
        self.transformer = Transformer(
            dim=config.dim,
            depth=config.depth,
            heads=config.heads,
            dim_head=config.dim_head,
            mlp_dim=config.mlp_dim,
            dropout=config.dropout,
            config=config.lora_config
        )

        self.pool = config.pool
        self.last_linear = nn.Linear(config.dim, config.num_classes)

        # Final normalization layer
        if config.final_norm_option_in_encoder == 'LayerNorm':
            self.final_norm = nn.LayerNorm(config.num_classes, eps=1e-5)
        elif config.final_norm_option_in_encoder == 'SimNorm':
            group_size = 8  # As specified in original code
            self.final_norm = SimNorm(simnorm_dim=group_size)
        else:
            raise ValueError(f"Unsupported final_norm_option_in_encoder: {config.final_norm_option_in_encoder}")

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Overview:
            Forward pass for the ViT model.
        Arguments:
            - img (:obj:`torch.Tensor`): Input image tensor of shape (batch_size, channels, height, width).
        Returns:
            - (:obj:`torch.Tensor`): Output logits tensor of shape (batch_size, num_classes).
        """
        # 1. Patch embedding
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        # 2. Prepend CLS token
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)

        # 3. Add positional embedding
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        # 4. Pass through Transformer encoder
        x = self.transformer(x)

        # 5. Pooling
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        # 6. Final classification head
        x = self.last_linear(x)
        x = self.final_norm(x)

        return x


# ==================== Test and Benchmark Code ====================
if __name__ == "__main__":
    import random
    import time

    # Fix random seeds for reproducibility
    torch.manual_seed(42)
    random.seed(42)

    # 1. Create a configuration object
    # This is now the standard way to configure the model.
    vit_config = ViTConfig(
        image_size=64,
        patch_size=8,
        num_classes=768,
        dim=768,
        depth=12,
        heads=12,
        mlp_dim=3072,
        dropout=0.1,
        emb_dropout=0.1,
        final_norm_option_in_encoder="LayerNorm"
    )

    # 2. Instantiate the model with the config
    model = ViT(config=vit_config)
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval() # Set model to evaluation mode for inference

    # Create a dummy input tensor
    dummy_input = torch.randn(256, 3, 64, 64).to(device)

    # Perform a single forward pass
    with torch.no_grad():
        out = model(dummy_input)
    
    print(f"Device: {device}")
    print(f"Output shape: {out.shape}")
    print(f"Output[0] (first 50 values): {out[0][:50]}")

    # 3. Simple Benchmark
    print("\nStarting benchmark...")
    warmup_reps, bench_reps = 5, 20
    
    with torch.no_grad():
        # Warm-up runs
        for _ in range(warmup_reps):
            _ = model(dummy_input)
        
        # Synchronize before timing (for CUDA)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        start_time = time.time()
        for _ in range(bench_reps):
            _ = model(dummy_input)
            
        # Synchronize after timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_time = time.time()

    total_time = end_time - start_time
    avg_latency_ms = (total_time / bench_reps) * 1000
    print(f"Average latency over {bench_reps} runs: {avg_latency_ms:.2f} ms")
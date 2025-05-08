"""
The following code is modified from https://github.com/karpathy/nanoGPT.
"""
import copy

import numpy as np
from typing import Optional, Tuple
import torch
import torch.nn as nn
from ding.torch_utils.network import GRUGatingUnit

from .transformer_config import TransformerConfig
from .kv_caching import KeysValues
from .attention import build_attention, Attention


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency components for the rotary positional embeddings.

    Arguments:
        - dim (int): The dimension of the embedding.
        - end (int): The length of the sequence for which frequencies are computed.
        - theta (float): A scaling factor for the frequencies, default is 10000.0.

    Returns:
        - freqs_cis (torch.Tensor): A tensor of complex numbers representing the precomputed frequencies.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Reshape the frequency components for broadcasting with the input tensor.

    Arguments:
        - freqs_cis (torch.Tensor): The frequency components tensor.
        - x (torch.Tensor): The input tensor to which the frequencies will be applied.

    Returns:
        - torch.Tensor: The reshaped frequency components tensor.
    """
    # Reference: https://github.com/meta-llama/llama3/blob/main/llama/model.py#L61
    ndim = x.ndim
    shape = [d if i in (0, 2, ndim - 1) else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # TODO: Check if any other positional embedding is relevant to this use case
    """
    Apply rotary positional embeddings to the query and key tensors.

    Arguments:
        - xq (torch.Tensor): The query tensor.
        - xk (torch.Tensor): The key tensor.
        - freqs_cis (torch.Tensor): The precomputed frequency components.

    Returns:
        - Tuple[torch.Tensor, torch.Tensor]: The transformed query and key tensors.
    
    Note:
        For more information on rotary positional embeddings, refer to the blog post:
        https://spaces.ac.cn/archives/8265/ or paper https://arxiv.org/abs/2104.09864
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(-2)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Transformer(nn.Module):
    """
    Transformer model class.

    Arguments:
        - config (:obj:`TransformerConfig`): Configuration for the Transformer model.

    Attributes:
        - config (:obj:`TransformerConfig`): Configuration object.
        - drop (:obj:`nn.Dropout`): Dropout layer for embedding dropout.
        - blocks (:obj:`nn.ModuleList`): List of Transformer blocks.
        - ln_f (:obj:`nn.LayerNorm`): Layer normalization applied to the final output.
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.drop = nn.Dropout(config.embed_pdrop)
        self.blocks = nn.ModuleList([
            Block(config, layer_idx=i)
            for i in range(config.num_layers)
        ])
        self.ln_f = nn.LayerNorm(config.embed_dim)

        if self.config.rotary_emb:
            freqs_cis = precompute_freqs_cis(
                self.config.embed_dim // self.config.num_heads,
                self.config.max_seq_len * 2,
                self.config.rope_theta,
            )
            self.register_buffer("freqs_cis", freqs_cis)

    def generate_empty_keys_values(self, n: int, max_tokens: int) -> KeysValues:
        """
        Generate a placeholder for keys and values.

        Arguments:
            - n (:obj:`int`): Batch size.
            - max_tokens (:obj:`int`): Maximum number of tokens in the sequence.

        Returns:
            - KeysValues: An object containing empty keys and values.
        """
        device = self.ln_f.weight.device  # Assumption: All submodules are on the same device
        return KeysValues(n, self.config.num_heads, max_tokens, self.config.embed_dim, self.config.num_layers, device)

    def forward(self, sequences: torch.Tensor, past_keys_values: Optional[KeysValues] = None,
                valid_context_lengths: Optional[torch.Tensor] = None, start_pos: int = 0) -> torch.Tensor:
        """
        Forward pass of the Transformer model.

        Arguments:
            - sequences (:obj:`torch.Tensor`): Input tensor of shape (batch_size, seq_length, embed_dim).
            - past_keys_values (:obj:`Optional[KeysValues]`): Precomputed keys and values for faster generation (default: None).
            - valid_context_lengths (:obj:`Optional[torch.Tensor]`): Valid lengths of context for masking (default: None).
            - start_pos (:obj:`int`): Starting position for rotary embeddings (default: 0).

        Returns:
            - torch.Tensor: Output tensor of shape (batch_size, seq_length, embed_dim).
        """
        seqlen = sequences.shape[1]
        # If using Rotary Position Embeddings (RoPE), slice the frequency components accordingly
        if self.config.rotary_emb:
            if isinstance(start_pos, (int, float, np.integer)):
                # In the reanalyze_phase or reset stage in collection/evaluation phase, create a tensor filled with start_pos, expanded to match the batch size, and adjust for sequence type,  e.g., start_pos=2.
                start_pos_tensor = torch.full((sequences.shape[0],), int(start_pos), device=sequences.device)
            elif isinstance(start_pos, (list, np.ndarray, torch.Tensor)):
                if isinstance(start_pos[0], (np.ndarray, torch.Tensor, list)):
                    # In the training phase, flatten start_pos, take the first element, convert to tensor, e.g., start_pos=[array([ 8, 10, 12, 14, 16]), array([12, 14, 16, 18, 20])]
                    start_pos_tensor = torch.as_tensor(
                    [x.reshape(-1)[0].item() for x in start_pos],  # Force flatten and take the first element
                        device=sequences.device
                    )
                elif isinstance(start_pos[0], (int, float, np.integer)):
                    # In the collection/evaluation phase, e.g., start_pos = [0, 0, 0, 0, 0, 0, 0, 0]
                    start_pos_tensor = torch.as_tensor([int(x) for x in start_pos], device=sequences.device)
            else:
                raise ValueError("start_pos must be an int, float, list, numpy array or torch.Tensor.")

            # TODO: Determine how to handle cases when episode length exceeds max_seq_len
            # Use modulo operation to ensure start_pos does not exceed max_seq_len
            start_pos_tensor = torch.remainder(start_pos_tensor, self.config.max_seq_len)
            # Convert each sample's start_pos to a list
            start_pos_list = start_pos_tensor.tolist()
            # For each sample, slice the corresponding range of freqs_cis based on start_pos
            freqs_cis_slices = [self.freqs_cis[int(pos): int(pos) + seqlen] for pos in start_pos_list]
            freqs_cis = torch.stack(freqs_cis_slices)

            if freqs_cis.ndim == 3 and freqs_cis.shape[1] == 1:
                # Convert shape [seq_len, 1, num_pairs] to [seq_len, num_pairs]
                freqs_cis = freqs_cis.squeeze(1)
        else:
            freqs_cis = None

        # print(f"freqs_cis.shape:{freqs_cis.shape}")

        # Ensure past keys and values match the number of transformer blocks
        assert past_keys_values is None or len(past_keys_values) == len(self.blocks)
        # Apply dropout to the input sequences
        x = self.drop(sequences)
        # Pass through each transformer block
        for i, block in enumerate(self.blocks):
            x = block(x, None if past_keys_values is None else past_keys_values[i], valid_context_lengths, freqs_cis)
        # Apply final layer normalization
        x = self.ln_f(x)
        return x


class Block(nn.Module):
    """
    Transformer block class.

    Arguments:
        config (:obj:`TransformerConfig`): Configuration for the Transformer block.

    Attributes:
        - gru_gating (:obj:`bool`): Flag to use GRU gating mechanism.
        - gru_bias (:obj:`float`): Bias for the GRU gating mechanism.
        - gate1 (:obj:`Optional[GRUGatingUnit]`): First GRU gating unit (if GRU gating is enabled).
        - gate2 (:obj:`Optional[GRUGatingUnit]`): Second GRU gating unit (if GRU gating is enabled).
        - ln1 (:obj:`nn.LayerNorm`): Layer normalization before the attention layer.
        - ln2 (:obj:`nn.LayerNorm`): Layer normalization before the MLP.
        - attn (:obj:`SelfAttention`): Self-attention mechanism.
        - mlp (:obj:`nn.Sequential`): Multi-layer perceptron.
    """

    def __init__(self, config: TransformerConfig, layer_idx : int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        # NOTE: GRU gating as in GTrXL
        self.gru_gating = config.gru_gating
        self.gru_bias = 1.0 # Fallback to 2.0
        if self.gru_gating:
            self.gate1 = GRUGatingUnit(config.embed_dim, self.gru_bias)
            self.gate2 = GRUGatingUnit(config.embed_dim, self.gru_bias)

        self.ln1 = nn.LayerNorm(config.embed_dim)
        self.ln2 = nn.LayerNorm(config.embed_dim)

        # Hybrid self-attention module
        if config.aha:
            if layer_idx < config.hybrid_local_layers:
                mode = 'local'
            else:
                mode = 'adaptive'
        elif config.interleave_local_causal:
            # even layers → local window, odd → full causal
            mode = 'local' if (layer_idx % 2 == 0) else 'causal'
        else:
            mode = config.attention

        cfg = copy.copy(config)
        cfg.attention = mode
        self.attn : Attention = build_attention(cfg) # Implements different attention mechanism
        self.mlp = nn.Sequential(
            nn.Linear(config.embed_dim, 4 * config.embed_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(4 * config.embed_dim, config.embed_dim),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x: torch.Tensor, past_keys_values: Optional[KeysValues] = None,
                valid_context_lengths: Optional[torch.Tensor] = None, freqs_cis: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the Transformer block.

        Arguments:
            - x (:obj:`torch.Tensor`): Input tensor of shape (batch_size, seq_length, embed_dim).
            - past_keys_values (:obj:`Optional[KeysValues]`): Precomputed keys and values for faster generation (default: None).
            - valid_context_lengths (:obj:`Optional[torch.Tensor]`): Valid lengths of context for masking (default: None).
            - freqs_cis (:obj:`torch.Tensor`): Frequency components for rotary position embeddings, used to modulate the attention mechanism (default: None).

        Returns:
            - torch.Tensor: Output tensor of shape (batch_size, seq_length, embed_dim).
        """
        x_attn = self.attn(self.ln1(x), past_keys_values, valid_context_lengths, freqs_cis)
        if self.gru_gating:
            x = self.gate1(x, x_attn)
        else:
            x = x + x_attn

        mlp_out = self.mlp(self.ln2(x))

        if self.gru_gating:
            x = self.gate2(x, mlp_out)
        else:
            x = x + mlp_out

        return x

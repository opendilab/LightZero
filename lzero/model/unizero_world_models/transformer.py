"""
Modified from https://github.com/karpathy/nanoGPT
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from ding.torch_utils.network import GRUGatingUnit
from einops import rearrange
from torch.nn import functional as F
import numpy as np

from .kv_caching import KeysValues


@dataclass
class TransformerConfig:
    tokens_per_block: int
    max_blocks: int
    attention: str

    num_layers: int
    num_heads: int
    embed_dim: int

    embed_pdrop: float
    resid_pdrop: float
    attn_pdrop: float
    
    # for RoPE
    rope_theta: float
    max_seq_len: int
    rotary_emb: bool = False

    @property
    def max_tokens(self):
        return self.tokens_per_block * self.max_blocks


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency components for the rotary positional embeddings.

    Arguments:
        - dim (int): The dimension of the embedding.
        - end (int): The length of the sequence for which frequencies are computed.
        - theta (float): A scaling factor for the frequencies, default is 10000.0.

    Returns:
        - torch.Tensor: A tensor of complex numbers representing the precomputed frequencies.
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
    shape = [d if i == ndim - 1 or i == 0 or i == 2 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
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
        https://spaces.ac.cn/archives/8265/ or paper https://arxiv.org/abs/2104.09864.
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    try:
        # print(f"freqs_cis.shape, xq_.shape:{freqs_cis.shape, xq_.shape}")
        freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    except Exception as e:
        print('='*20)
        # import ipdb;ipdb.set_trace()
        print(f"freqs_cis.shape, xq_.shape:{freqs_cis.shape, xq_.shape}")
        print(f"Error in reshaping freqs_cis: {e}")
        print('We are at the reset timestep!')
        print('='*20)

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
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.num_layers)])
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
            if isinstance(start_pos, int) or isinstance(start_pos, float):
                # Create a tensor filled with start_pos, expanded to match the batch size, and adjust for sequence type
                start_pos_tensor = torch.full((sequences.shape[0],), int(start_pos), device=sequences.device)
            elif isinstance(start_pos, (list, np.ndarray, torch.Tensor)):
                # print(f"start_pos: {start_pos}")
                # print(f"type(start_pos): {type(start_pos)}")
                # print(f"type(start_pos[0]): {type(start_pos[0])}")
                # print(f"start_pos[0].shape:{start_pos[0].shape}")
                # print(f"len(start_pos[0].shape):{len(start_pos[0].shape)}")
                if isinstance(start_pos[0], list):
                    # In the training phase, flatten start_pos, take the first element, convert to tensor, and double it
                    start_pos_tensor = torch.as_tensor(
                        [x.reshape(-1)[0].item() for x in start_pos],  # Force flatten and take the first element
                        device=sequences.device
                    )
                elif isinstance(start_pos[0], (np.ndarray, torch.Tensor)):
                    if len(start_pos[0].shape) <= 0:
                        # In the collection/evaluation phase, convert start_pos to a tensor and double it
                        start_pos_tensor = torch.as_tensor([x.item() for x in start_pos], device=sequences.device)
                    else:
                        # In the training phase, flatten start_pos, take the first element, convert to tensor, and double it
                        start_pos_tensor = torch.as_tensor(
                        [x.reshape(-1)[0].item() for x in start_pos],  # Force flatten and take the first element
                            device=sequences.device
                        )
                elif isinstance(start_pos[0], (int, float, np.integer)):
                    # Handle numpy integer types similarly to int and float
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

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        # NOTE: GRU gating as in GTrXL
        self.gru_gating = config.gru_gating
        self.gru_bias = 2.0
        if self.gru_gating:
            self.gate1 = GRUGatingUnit(config.embed_dim, self.gru_bias)
            self.gate2 = GRUGatingUnit(config.embed_dim, self.gru_bias)

        self.ln1 = nn.LayerNorm(config.embed_dim)
        self.ln2 = nn.LayerNorm(config.embed_dim)
        self.attn = SelfAttention(config)
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
            x = self.gate2(x, self.mlp(self.ln2(x)))
        else:
            x = x + x_attn
            x = x + self.mlp(self.ln2(x))

        return x


class SelfAttention(nn.Module):
    """
    Implements self-attention mechanism for transformers.

    Arguments:
        config (:obj:`TransformerConfig`): Configuration object containing hyperparameters.

    Attributes:
        - config (:obj:`TransformerConfig`): Stores the configuration for the self-attention module.
        - num_heads (:obj:`int`): Number of attention heads.
        - key (:obj:`nn.Linear`): Linear layer to project input to key vectors.
        - query (:obj:`nn.Linear`): Linear layer to project input to query vectors.
        - value (:obj:`nn.Linear`): Linear layer to project input to value vectors.
        - attn_drop (:obj:`nn.Dropout`): Dropout layer for attention weights.
        - resid_drop (:obj:`nn.Dropout`): Dropout layer for residual connection.
        - proj (:obj:`nn.Linear`): Final linear layer for projection.
        - mask (:obj:`torch.Tensor`): Mask tensor for causal or block-causal attention.
    """
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        assert config.embed_dim % config.num_heads == 0, "Embedding dimension must be divisible by number of heads."

        self.config = config
        self.num_heads = config.num_heads

        self.key = nn.Linear(config.embed_dim, config.embed_dim)
        self.query = nn.Linear(config.embed_dim, config.embed_dim)
        self.value = nn.Linear(config.embed_dim, config.embed_dim)

        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.proj = nn.Linear(config.embed_dim, config.embed_dim)

        causal_mask = torch.tril(torch.ones(config.max_tokens, config.max_tokens))
        self.register_buffer('mask', causal_mask)

    def forward(self, x: torch.Tensor, kv_cache: Optional[KeysValues] = None,
                valid_context_lengths: Optional[torch.Tensor] = None,  freqs_cis: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for the self-attention mechanism.

        Arguments:
            - x (:obj:`torch.Tensor`): Input tensor of shape (B, T, C) where B is batch size,
                                        T is sequence length, and C is embedding dimension.
            - kv_cache (:obj:`Optional[KeysValues]`): Optional key-value cache for faster inference.
            - valid_context_lengths (:obj:`Optional[torch.Tensor]`): Optional tensor containing valid context lengths.
            - freqs_cis (:obj:`torch.Tensor`): Frequency components for rotary position embeddings, used to modulate the attention mechanism (default: None).

        Returns:
            - torch.Tensor: Output tensor of shape (B, T, C).
        """
        B, T, C = x.size()
        if kv_cache is not None:
            b, nh, L, c = kv_cache.shape
            assert nh == self.num_heads and b == B and c * nh == C, "Cache dimensions do not match input dimensions."
        else:
            L = 0

        q = self.query(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)  # (B, num_heads, T, head_size)
        k = self.key(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)  # (B, num_heads, T, head_size)
        v = self.value(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)  # (B, num_heads, T, head_size)
        
        if self.config.rotary_emb:
            q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)

        if kv_cache is not None:
            kv_cache.update(k, v) # time occupancy 21%
            k, v = kv_cache.get() # time occupancy 5%

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        if valid_context_lengths is not None:
            # Final mask.shape: (B, T, L + T)
            # L is the context length, T is the current input length,
            # valid_context_lengths is the valid length at the end of the context.
            mask = torch.zeros(B, T, L + T, device=att.device)
            # For each sample, set the invalid parts to 0 based on its valid length.
            for i in range(B):
                mask[i] = self.mask[L:L + T, :L + T].clone()
                mask[i, :, :(L - valid_context_lengths[i])] = 0  # Set invalid parts to 0.
                # Adjust mask dimensions to match the last two dimensions of att.
                # (B, T, L + T) -> (B, 1, T, L + T) -> (B, num_heads, T, L + T)
                mask = mask.unsqueeze(1).expand(-1, att.size(1), -1, -1)
        else:
            # mask.shape: (T, L + T)
            mask = self.mask[L:L + T, :L + T]

        # att.shape: (B, num_heads, T, L + T)
        att = att.masked_fill(mask == 0, float('-inf'))

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, num_heads, T, L + T) x (B, num_heads, L + T, head_size) -> (B, num_heads, T, head_size)

        y = rearrange(y, 'b h t e -> b t (h e)')  # Combine the heads back together (B, T, embed_dim)
        y = self.resid_drop(self.proj(y))

        return y

    @torch.no_grad()
    def get_attention_map(self, x: torch.Tensor, kv_cache: Optional[KeysValues] = None,
                          valid_context_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute the attention map for the input sequence. This is useful for visualization purposes.
        More details can be found in visualizing_utils.py.

        Arguments:
            - x (:obj:`torch.Tensor`): Input sequence with shape (B, T, C).
            - kv_cache (:obj:`Optional[KeysValues]`): Cached keys and values for supporting long sequence inference.
            - valid_context_lengths (:obj:`Optional[torch.Tensor]`): Valid context lengths for handling variable-length contexts.

        Returns:
            - torch.Tensor: Attention map with shape (B, nh, T, L + T), representing the distribution of attention.
        """
        B, T, C = x.size()
        if kv_cache is not None:
            b, nh, L, c = kv_cache.shape
            assert nh == self.num_heads and b == B and c * nh == C, "Cache dimensions are inconsistent with input dimensions."
        else:
            L = 0

        # Compute query, key, and value projections
        q = self.query(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)  # (B, nh, T, hs)
        k = self.key(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)  # (B, nh, T, hs)

        if kv_cache is not None:
            # Update the kv_cache with the new keys and values
            kv_cache.update(k, v)
            k, v = kv_cache.get()

        # Compute the attention scores
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        if valid_context_lengths is not None:
            mask = torch.zeros(B, T, L + T, device=att.device)
            for i in range(B):
                # Create attention mask for each batch
                mask[i] = self.mask[L:L + T, :L + T].clone()
                mask[i, :, :(L - valid_context_lengths[i])] = 0
            mask = mask.unsqueeze(1).expand(-1, att.size(1), -1, -1)
        else:
            mask = self.mask[L:L + T, :L + T]

        # Apply the attention mask
        att = att.masked_fill(mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)

        return att
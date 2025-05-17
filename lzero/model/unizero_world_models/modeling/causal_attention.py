"""
Standard Autoregressive Self-Attention Module for Transformers (Causal Attention)
"""
import math
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange
from .kv_caching import KeysValues
from .transformer import apply_rotary_emb
from .attention import Attention
from .transformer_config import TransformerConfig


class CausalAttention(Attention):
    """
    Implements causal self-attention mechanism for transformers.

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
        super().__init__(config)
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
                valid_context_lengths: Optional[torch.Tensor] = None, freqs_cis: torch.Tensor = None) -> torch.Tensor:
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

        q = self.query(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1,                                                                          2)  # (B, num_heads, T, head_size)
        k = self.key(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)  # (B, num_heads, T, head_size)
        v = self.value(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1,
                                                                                    2)  # (B, num_heads, T, head_size)

        if self.config.rotary_emb:
            q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)

        if kv_cache is not None:
            kv_cache.update(k, v)  # time occupancy 21%
            k, v = kv_cache.get()  # time occupancy 5%

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
            mask = self.mask[L:L + T, :L + T]  # Causal Mask being Applied

        # att.shape: (B, num_heads, T, L + T)
        att = att.masked_fill(mask == 0, float('-inf'))

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, num_heads, T, L + T) x (B, num_heads, L + T, head_size) -> (B, num_heads, T, head_size)

        y = rearrange(y, 'b h t e -> b t (h e)')  # Combine the heads back together (B, T, embed_dim)
        y = self.resid_drop(self.proj(y))

        return y


    @torch.no_grad()
    def get_attention_map(
            self,
            x: torch.Tensor,
            kv_cache: Optional[KeysValues] = None,
            valid_context_lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute the attention map for the input sequence (B, T, C), reusing an existing kv_cache
        only to determine how much past context to mask, without mutating it.
        Returns attn weights of shape (B, nh, T, L+T).
        """
        B, T, C = x.size()
        device = x.device

        # 1) project Q and fresh K
        q = self.query(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        k_fresh = self.key(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)

        # 2) rotary embeddings if used
        #    forward only applies it to Q/K, not V
        if getattr(self.config, 'rotary_emb', False):
            # forward passes freqs_cis, but here we don’t have it—so user must supply it if needed
            # you can add freqs_cis param if you like
            raise RuntimeError("Rotary embedding case not supported in this helper; pass freqs_cis.")

        # 3) determine L via cache (read‐only)
        if kv_cache is not None:
            k_old, _ = kv_cache.get()  # shape (B, nh, L, head_dim)
            k = torch.cat([k_old, k_fresh], dim=2)  # (B, nh, L+T, head_dim)
            L = k_old.shape[2]
        else:
            k = k_fresh
            L = 0

        total_len = L + T

        # 4) raw scores
        scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # (B, nh, T, total_len)

        # 5) build mask exactly as in forward
        if valid_context_lengths is not None:
            # per‐batch mask tensor
            mask_bt = torch.zeros(B, T, total_len, device=device, dtype=torch.bool)
            for i in range(B):
                valid = int(valid_context_lengths[i].item())
                stale = L - valid
                sub = self.mask[L:L + T, :L + T].clone()  # (T, total_len)
                if stale > 0:
                    sub[:, :stale] = False
                mask_bt[i] = sub
            mask = mask_bt.unsqueeze(1).expand(-1, self.num_heads, -1, -1)  # (B, nh, T, total_len)
        else:
            mask = self.mask[L:L + T, :L + T]  # (T, total_len)
            mask = mask.unsqueeze(0).unsqueeze(1)  # (1,1, T, total_len)
            mask = mask.expand(B, self.num_heads, T, total_len)

        # 6) apply mask & softmax
        scores = scores.masked_fill(~mask, float('-inf'))
        attn = F.softmax(scores, dim=-1)

        return attn
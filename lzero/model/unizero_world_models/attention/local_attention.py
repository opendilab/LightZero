"""
Local Self-Attention Module for Transformers (Global Attention)
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

class LocalAttention(Attention):
    """
    Implements local self-attention mechanism for transformers.

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
        assert config.embed_dim % config.num_heads == 0, \
            "Embedding dimension must be divisible by number of heads."

        self.config = config
        self.num_heads = config.num_heads
        self.window = config.local_window_size

        # projectors
        self.key = nn.Linear(config.embed_dim, config.embed_dim)
        self.query = nn.Linear(config.embed_dim, config.embed_dim)
        self.value = nn.Linear(config.embed_dim, config.embed_dim)

        # dropout & output proj
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.proj = nn.Linear(config.embed_dim, config.embed_dim)

        # build a (max_tokens × max_tokens) local band mask once:
        indices = torch.arange(config.max_tokens)
        # mask[i,j] = True iff |i - j| <= window
        local_mask = (indices.unsqueeze(1) - indices.unsqueeze(0)).abs() <= self.window
        self.register_buffer('mask', local_mask)

    def forward(self,
                x: torch.Tensor,
                kv_cache: Optional[KeysValues] = None,
                valid_context_lengths: Optional[torch.Tensor] = None,
                freqs_cis: torch.Tensor = None
                ) -> torch.Tensor:
        B, T, C = x.shape
        if kv_cache is not None: # handle cache
            b, nh, L, c = kv_cache.shape
            assert nh == self.num_heads and b == B and c * nh == C, \
                "Cache dimensions do not match."
        else:
            L = 0

        # project Q, K, V into heads
        q = self.query(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = self.key(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = self.value(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)

        # rotary embeddings if used
        if getattr(self.config, 'rotary_emb', False):
            q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)

        # update / retrieve cache (for fast autoregressive inference)
        if kv_cache is not None:
            kv_cache.update(k, v)
            k, v = kv_cache.get()  # (B, nh, L+T, head_size)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # (B, nh, T, L+T)

        # apply local band mask slice [L:L+T, 0:L+T]
        #  so each new position t in [0,T) corresponds to global index (L+t)
        mask = self.mask[L:L + T, :L + T]  # (T, L+T)
        mask = mask.unsqueeze(0).unsqueeze(1)  # (1,1, T, L+T)
        mask = mask.expand(B, self.num_heads, T, L + T).to(att.device)
        att = att.masked_fill(~mask, float('-inf'))

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, head_size)

        # combine heads
        y = rearrange(y, 'b h t e -> b t (h e)')
        y = self.resid_drop(self.proj(y))
        return y

    @torch.no_grad()
    def get_attention_map(self,
                          x: torch.Tensor,
                          kv_cache: Optional[KeysValues] = None,
                          valid_context_lengths: Optional[torch.Tensor] = None
                          ) -> torch.Tensor:
        B, T, C = x.shape
        if kv_cache is not None:
            b, nh, L, c = kv_cache.shape
            kv_cache.update(*[t.view(B, nh, T, C // nh).transpose(1, 2)
                              for t in (self.query(x), self.key(x), self.value(x))][:2])
            k, v = kv_cache.get()
        else:
            L = 0

        q = self.query(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = self.key(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)

        # raw scores
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # local mask
        mask = self.mask[L:L + T, :L + T]
        mask = mask.unsqueeze(0).unsqueeze(1).expand(B, self.num_heads, T, L + T).to(att.device)
        att = att.masked_fill(~mask, float('-inf'))
        att = F.softmax(att, dim=-1)

        return att
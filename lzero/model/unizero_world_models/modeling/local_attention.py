"""
Local Self-Attention Module for Transformers
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
    Implements causal self-attention *with* a fixed local window constraint.
    Built on top of Causal Attn to ensure correctness of KV and autoregressive behavior.

    Arguments:
        config (TransformerConfig): Configuration object containing hyperparameters.

    Attributes:
        - config: stores the configuration.
        - num_heads: number of attention heads.
        - window: local attention window radius (tokens).
        - key, query, value: projection layers.
        - attn_drop, resid_drop: dropout layers.
        - proj: output projection.
        - mask: precomputed full causal mask (max_tokens × max_tokens).
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__(config)
        assert config.embed_dim % config.num_heads == 0, \
            "Embedding dimension must be divisible by number of heads."

        self.config = config
        self.num_heads = config.num_heads
        self.window = config.local_window_size  # radius of local window

        # projectors
        self.key   = nn.Linear(config.embed_dim, config.embed_dim)
        self.query = nn.Linear(config.embed_dim, config.embed_dim)
        self.value = nn.Linear(config.embed_dim, config.embed_dim)

        # dropouts & output projection
        self.attn_drop  = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.proj       = nn.Linear(config.embed_dim, config.embed_dim)

        # precompute full causal mask once
        causal_mask = torch.tril(torch.ones(config.max_tokens, config.max_tokens, dtype=torch.bool))
        self.register_buffer('mask', causal_mask)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[KeysValues] = None,
        valid_ctx_len: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for LocalAttention with window masking.

        Args:
            x (torch.Tensor): (B, T, C) input.
            kv_cache (KeysValues, optional): key/value cache.
            valid_ctx_len (torch.Tensor, optional): for each batch element, how many of
                the last L cached positions are valid.
            freqs_cis (torch.Tensor, optional): rotary embeddings slice.

        Returns:
            torch.Tensor: (B, T, C) output.
        """
        B, T, C = x.shape

        # ensure cache length is correct
        if kv_cache is not None:
            b, nh, L, c = kv_cache.shape
            assert b == B and nh == self.num_heads and c * nh == C, "Cache dims mismatch."
        else:
            L = 0

        # project Q, K, V
        q = self.query(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = self.key(x)  .view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = self.value(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)

        # apply rotary embeddings if used
        if self.config.rotary_emb:
            q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)

        # update and retrieve cache
        if kv_cache is not None:
            kv_cache.update(k, v)
            k, v = kv_cache.get()  # shapes: (B, nh, L+T, head_dim)

        # raw scores
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # (B, nh, T, L+T)

        # apply causal mask as pre-computed
        base = self.mask[L : L + T, : L + T]  # (T, L+T)

        # ensure no stale lots are used
        if valid_ctx_len is not None:
            mask_bt = torch.zeros(B, T, L + T, device=att.device, dtype=torch.bool)
            for i in range(B):
                valid = int(valid_ctx_len[i].item())
                stale = L - valid
                sub = base.clone()
                if stale > 0:
                    sub[:, :stale] = False
                mask_bt[i] = sub
                mask = mask_bt.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        else:
            mask = base.unsqueeze(0).unsqueeze(1).expand(B, self.num_heads, T, L + T)

        # applies local (window) attention mask
        if self.window is not None:
            qpos = torch.arange(L, L + T, device=att.device).unsqueeze(1)      # (T,1)
            kpos = torch.arange(0,      L + T, device=att.device).unsqueeze(0) # (1,L+T)
            wmask = (qpos - kpos).abs() <= self.window                         # (T,L+T)
            wmask = wmask.unsqueeze(0).unsqueeze(1).expand(B, self.num_heads, T, L + T)
            mask = mask & wmask

        # apply mask
        att = att.masked_fill(~mask, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        # attend and project
        y = att @ v                            # (B, nh, T, head_dim)
        y = rearrange(y, 'b h t d -> b t (h d)')
        return self.resid_drop(self.proj(y))

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

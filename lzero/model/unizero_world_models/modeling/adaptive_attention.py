"""
Adaptive Attention Mechanism for Transformer World Model
"""
import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .kv_caching import KeysValues
from .transformer import apply_rotary_emb
from .attention import Attention
from .transformer_config import TransformerConfig

class AdaptiveSpanAttention(Attention):
    """
    Implements adaptive span self-attention mechanism for transformers.

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

        self.config   = config
        self.num_heads= config.num_heads
        self.head_dim = config.embed_dim // config.num_heads
        self.max_len  = config.max_tokens

        # projections
        self.key   = nn.Linear(config.embed_dim, config.embed_dim)
        self.query = nn.Linear(config.embed_dim, config.embed_dim)
        self.value = nn.Linear(config.embed_dim, config.embed_dim)

        # dropouts + out proj
        self.attn_drop  = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.proj       = nn.Linear(config.embed_dim, config.embed_dim)

        # learnable span parameters (in softplus domain)
        init_span = config.init_adaptive_span or config.max_tokens # default value to max_tokens (i.e. 20)
        inv_softplus = lambda x: math.log(math.expm1(x))
        self.span_p = nn.Parameter(torch.full(
            (self.num_heads,), inv_softplus(init_span) # define Torch param
        ))

        # precompute full causal mask once
        causal = torch.tril(torch.ones(self.max_len, self.max_len, dtype=torch.bool))
        self.register_buffer('causal_mask', causal)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[KeysValues] = None,
        valid_ctx_len: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, T, C = x.size()
        device = x.device

        # project Q, K, V
        q = self.query(x).view(B, T, self.num_heads, self.head_dim).transpose(1,2)
        k = self.key(x)  .view(B, T, self.num_heads, self.head_dim).transpose(1,2)
        v = self.value(x).view(B, T, self.num_heads, self.head_dim).transpose(1,2)

        # apply rotary embeddings if used
        if getattr(self.config, 'rotary_emb', False) and freqs_cis is not None:
            q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)

        # update cache
        if kv_cache is not None:
            kv_cache.update(k, v)
            k, v = kv_cache.get()  # (B, nh, L+T, head_dim)
            L = k.shape[2] - T
        else:
            L = 0

        total_len = L + T

        # raw attention scores
        scores = (q @ k.transpose(-2,-1)) * (1.0 / math.sqrt(self.head_dim)) # (B, nh, T, total_len)

        # apply causal mask and remove stale slots
        #   slice out the T×(L+T) block
        base = self.causal_mask[L:L+T, :L+T]  # (T, total_len)
        if valid_ctx_len is not None:
            m = torch.zeros(B, T, total_len, dtype=torch.bool, device=device)
            for i in range(B):
                valid = int(valid_ctx_len[i].item())
                stale = L - valid
                sub = base.clone()
                if stale>0:
                    sub[:, :stale] = False
                m[i] = sub
            mask = m.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        else:
            mask = base.unsqueeze(0).unsqueeze(1).expand(B, self.num_heads, -1, -1)

        #  adaptive‐span mask
        spans = F.softplus(self.span_p)            # (nh,)
        spans_int = spans.floor().clamp(max=self.max_len).long()
        # positions in global indexing
        qpos = torch.arange(L, L+T, device=device).unsqueeze(1)   # (T,1)
        kpos = torch.arange(0, total_len, device=device).unsqueeze(0)  # (1, total_len)
        dist  = (qpos - kpos).abs()                               # (T, total_len)
        d_exp = dist.unsqueeze(0).expand(self.num_heads, -1, -1)  # (nh, T, total_len)
        s_exp = spans_int.unsqueeze(1).unsqueeze(2)               # (nh,1,1)
        adapt_mask = (d_exp <= s_exp)                             # (nh, T, total_len)
        adapt_mask = adapt_mask.unsqueeze(0).expand(B, -1, -1, -1)

        # combine masks
        final_mask = mask & adapt_mask

        # apply mask, softmax, dropout
        scores = scores.masked_fill(~final_mask, float('-inf'))
        attn   = F.softmax(scores, dim=-1)
        attn   = self.attn_drop(attn)

        # attend and project
        y = attn @ v                    # (B, nh, T, head_dim)
        y = rearrange(y, 'b h t d -> b t (h d)')
        return self.resid_drop(self.proj(y))

    @torch.no_grad()
    def get_attention_map(self, x: torch.Tensor, kv_cache: Optional[KeysValues] = None,
                          valid_context_lengths: Optional[torch.Tensor] = None):
        """
        Returns the attention weight maps for visualization.
        """
        B, T, C = x.shape
        device = x.device

        # same projection steps as forward, but only compute scores & mask
        q = self.query(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        if getattr(self.config, 'rotary_emb', False):
            q, k = apply_rotary_emb(q, k, freqs_cis=None)

        if kv_cache is not None:
            kv_cache.update(k, self.value(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2))
            k, _ = kv_cache.get()
            L = k.size(-2) - T
        else:
            L = 0

        total_len = L + T
        scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))

        # build same mask as in forward
        spans = F.softplus(self.span_p)
        spans_int = spans.floor().clamp(max=self.max_len).long()
        query_pos = torch.arange(L, L + T, device=device).unsqueeze(1)
        key_pos = torch.arange(0, total_len, device=device).unsqueeze(0)
        distances = (query_pos - key_pos).abs()
        d_exp = distances.unsqueeze(0).expand(self.num_heads, -1, -1)
        spans_e = spans_int.unsqueeze(1).unsqueeze(2)
        head_mask = d_exp <= spans_e
        mask = head_mask.unsqueeze(0).expand(B, -1, -1, -1)

        scores = scores.masked_fill(~mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        return attn

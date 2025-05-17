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
    Implements adaptive span self-attention with a fully-differentiable soft mask.
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__(config)
        assert config.embed_dim % config.num_heads == 0, \
            "Embedding dimension must be divisible by number of heads."

        self.config    = config
        self.num_heads = config.num_heads
        self.head_dim  = config.embed_dim // config.num_heads
        self.max_len   = config.max_tokens

        # projections
        self.key   = nn.Linear(config.embed_dim, config.embed_dim)
        self.query = nn.Linear(config.embed_dim, config.embed_dim)
        self.value = nn.Linear(config.embed_dim, config.embed_dim)

        # dropouts + out proj
        self.attn_drop  = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.proj       = nn.Linear(config.embed_dim, config.embed_dim)

        # learnable span parameters (in softplus domain)
        init_span = config.init_adaptive_span or config.max_tokens
        inv_softplus = lambda x: math.log(math.expm1(x))
        self.span_p = nn.Parameter(torch.full(
            (self.num_heads,), inv_softplus(init_span)
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

        # rotary embeddings
        if getattr(self.config, 'rotary_emb', False) and freqs_cis is not None:
            q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)

        # update & extract cache
        if kv_cache is not None:
            kv_cache.update(k, v)
            k, v = kv_cache.get()
            L = k.shape[2] - T
        else:
            L = 0

        total_len = L + T

        # raw attention scores
        scores = (q @ k.transpose(-2,-1)) * (1.0 / math.sqrt(self.head_dim))

        # causal & stale mask
        base = self.causal_mask[L:L+T, :L+T]
        if valid_ctx_len is not None:
            mask = torch.zeros(B, T, total_len, dtype=torch.bool, device=device)
            for i in range(B):
                valid = int(valid_ctx_len[i].item())
                stale = L - valid
                sub = base.clone()
                if stale > 0:
                    sub[:, :stale] = False
                mask[i] = sub
            mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        else:
            mask = base.unsqueeze(0).unsqueeze(1).expand(B, self.num_heads, T, total_len)

        # fully-differentiable soft span mask
        span_cont = F.softplus(self.span_p)                       # (nh,)
        qpos = torch.arange(L, L+T, device=device).unsqueeze(1)   # (T,1)
        kpos = torch.arange(0, total_len, device=device).unsqueeze(0)  # (1, total_len)
        dist  = (qpos - kpos).abs().unsqueeze(0)                  # (1, T, total_len)
        span_exp = span_cont.unsqueeze(1).unsqueeze(2)            # (nh,1,1)
        # triangular gating: max(0, 1 - dist / span)
        mask_weights = (1.0 - dist / span_exp).clamp(min=0.0)      # (nh, T, total_len)
        mask_weights = mask_weights.unsqueeze(0).expand(B, -1, -1, -1)

        # apply causal mask by zeroing weights outside causal region
        mask_weights = mask_weights.masked_fill(~mask, 0.0)

        # modulate scores and compute attention
        scores = scores * mask_weights
        attn   = F.softmax(scores, dim=-1)
        attn   = self.attn_drop(attn)

        # attend and project
        y = attn @ v
        y = rearrange(y, 'b h t d -> b t (h d)')
        return self.resid_drop(self.proj(y))

    @torch.no_grad()
    def get_attention_map(
        self,
        x: torch.Tensor,
        kv_cache: Optional[KeysValues] = None,
        valid_context_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the soft-masked attention map for analysis.
        """
        B, T, C = x.shape
        device = x.device

        # project Q, K (fresh)
        q = self.query(x).view(B, T, self.num_heads, self.head_dim).transpose(1,2)
        k_fresh = self.key(x).view(B, T, self.num_heads, self.head_dim).transpose(1,2)

        # read-only cache
        if kv_cache is not None:
            k, _ = kv_cache.get()
            L = k.shape[2] - T
        else:
            k = k_fresh
            L = 0

        total_len = L + T
        scores = (q @ k.transpose(-2,-1)) * (1.0 / math.sqrt(self.head_dim))

        # causal mask
        base = self.causal_mask[L:L+T, :L+T]
        if valid_context_lengths is not None:
            cm = torch.zeros(B, T, total_len, dtype=torch.bool, device=device)
            for i in range(B):
                valid = int(valid_context_lengths[i].item())
                stale = L - valid
                sub = base.clone()
                if stale > 0:
                    sub[:, :stale] = False
                cm[i] = sub
            mask = cm.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        else:
            mask = base.unsqueeze(0).unsqueeze(1).expand(B, self.num_heads, T, total_len)

        # compute same soft mask weights
        span_cont = F.softplus(self.span_p)
        qpos = torch.arange(L, L+T, device=device).unsqueeze(1)
        kpos = torch.arange(0, total_len, device=device).unsqueeze(0)
        dist  = (qpos - kpos).abs().unsqueeze(0)
        span_exp = span_cont.unsqueeze(1).unsqueeze(2)
        mask_weights = (1.0 - dist / span_exp).clamp(min=0.0)
        mask_weights = mask_weights.unsqueeze(0).expand(B, -1, -1, -1)
        mask_weights = mask_weights.masked_fill(~mask, 0.0)

        # apply and softmax
        scores = scores * mask_weights
        attn   = F.softmax(scores, dim=-1)
        return attn

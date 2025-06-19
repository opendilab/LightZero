"""
Gaussian Adaptive Attention Mechanism (GAAM) for Transformers.
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

class GAAM(Attention):
    """
    Implements Gaussian Adaptive Attention Mechanism for transformers.

    Each head learns a mean offset (mu_p_raw) and a variance (sigma_p) for a Gaussian
    over relative positions, which modulates the attention scores multiplicatively.
    """
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__(config)
        assert config.embed_dim % config.num_heads == 0, \
            "Embedding dimension must be divisible by number of heads."

        self.config = config
        self.num_heads = config.num_heads
        self.head_dim = config.embed_dim // config.num_heads
        self.max_len = config.max_tokens

        # projections
        self.key   = nn.Linear(config.embed_dim, config.embed_dim)
        self.query = nn.Linear(config.embed_dim, config.embed_dim)
        self.value = nn.Linear(config.embed_dim, config.embed_dim)

        # dropouts + output projection
        self.attn_drop  = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.proj       = nn.Linear(config.embed_dim, config.embed_dim)

        # learnable Gaussian parameters per head (stored in softplus domain)
        inv_softplus = lambda x: math.log(math.expm1(x))
        init_sigma = getattr(config, 'init_adaptive_sigma', 1.0)
        init_mu    = getattr(config, 'init_adaptive_mu', 0.0)
        # variance parameter per head
        self.sigma_p    = nn.Parameter(torch.full((self.num_heads,), inv_softplus(init_sigma)))
        # raw mean parameter per head (will be softplus-clamped)
        self.mu_p_raw   = nn.Parameter(torch.full((self.num_heads,), inv_softplus(init_mu)))

        # precompute full causal mask
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
        device  = x.device

        # project queries, keys, values
        q = self.query(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x)  .view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # apply rotary embeddings if enabled
        if getattr(self.config, 'rotary_emb', False) and freqs_cis is not None:
            q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)

        # update key/value cache if provided
        if kv_cache is not None:
            kv_cache.update(k, v)
            k, v = kv_cache.get()  # shape: (B, nh, L+T, head_dim)
            L = k.shape[2] - T
        else:
            L = 0
        total_len = L + T

        # raw attention scores
        scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))  # (B, nh, T, total_len)

        # build causal/stale mask
        base = self.causal_mask[L:L+T, :L+T]
        if valid_ctx_len is not None:
            m = torch.zeros(B, T, total_len, dtype=torch.bool, device=device)
            for i in range(B):
                valid = int(valid_ctx_len[i].item())
                stale = L - valid
                sub   = base.clone()
                if stale > 0:
                    sub[:, :stale] = False
                m[i] = sub
            mask = m.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        else:
            mask = base.unsqueeze(0).unsqueeze(1).expand(B, self.num_heads, -1, -1)

        # Gaussian adaptive weights (relative-position bias)
        sigmas = F.softplus(self.sigma_p)                           # (nh,)
        mus    = F.softplus(self.mu_p_raw).clamp(max=self.max_len)   # (nh,)
        qpos   = torch.arange(L, L+T, device=device).unsqueeze(1)     # (T,1)
        kpos   = torch.arange(0, total_len, device=device).unsqueeze(0)  # (1,total_len)
        dist   = (qpos - kpos).abs()                                  # (T, total_len)
        # expand to heads
        dist     = dist.unsqueeze(0).expand(self.num_heads, -1, -1)  # (nh,T,total_len)
        mu       = mus.unsqueeze(1).unsqueeze(2)                     # (nh,1,1)
        sigma    = sigmas.unsqueeze(1).unsqueeze(2)                  # (nh,1,1)
        gaussian = torch.exp(-((dist - mu)**2) / (2 * sigma**2))     # (nh,T,total_len)
        gaussian = gaussian.unsqueeze(0).expand(B, -1, -1, -1)        # (B,nh,T,total_len)

        # apply Gaussian weights and mask
        scores = scores * gaussian
        scores = scores.masked_fill(~mask, float('-inf'))

        # softmax and dropout
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)

        # compute output
        y = attn @ v  # (B,nh,T,head_dim)
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
        Compute the Gaussian-adaptive attention map without updating cache.
        Returns weights of shape (B, nh, T, total_len).
        """
        B, T, C = x.shape
        device  = x.device
        q = self.query(x).view(B, T, self.num_heads, self.head_dim).transpose(1,2)

        if kv_cache is not None:
            k, _ = kv_cache.get()
            L = k.shape[2] - T
        else:
            k = self.key(x).view(B, T, self.num_heads, self.head_dim).transpose(1,2)
            L = 0
        total_len = L + T

        scores = (q @ k.transpose(-2,-1)) * (1.0 / math.sqrt(self.head_dim))

        base = self.causal_mask[L:L+T, :L+T]
        if valid_context_lengths is not None:
            m = torch.zeros(B, T, total_len, dtype=torch.bool, device=device)
            for i in range(B):
                valid = int(valid_context_lengths[i].item())
                stale = L - valid
                sub   = base.clone()
                if stale > 0:
                    sub[:, :stale] = False
                m[i] = sub
            mask = m.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        else:
            mask = base.unsqueeze(0).unsqueeze(1).expand(B, self.num_heads, T, total_len)

        # Gaussian mask
        sigmas = F.softplus(self.sigma_p)
        mus    = F.softplus(self.mu_p_raw).clamp(max=self.max_len)
        qpos   = torch.arange(L, L+T, device=device).unsqueeze(1)
        kpos   = torch.arange(0, total_len, device=device).unsqueeze(0)
        dist   = (qpos - kpos).abs().unsqueeze(0).expand(self.num_heads, -1, -1)
        mu       = mus.unsqueeze(1).unsqueeze(2)
        sigma    = sigmas.unsqueeze(1).unsqueeze(2)
        gaussian = torch.exp(-((dist - mu)**2) / (2 * sigma**2))
        gaussian = gaussian.unsqueeze(0).expand(B, -1, -1, -1)

        scores = scores * gaussian
        scores = scores.masked_fill(~mask, float('-inf'))
        return F.softmax(scores, dim=-1)

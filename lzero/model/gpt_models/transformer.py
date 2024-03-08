"""
Credits to https://github.com/karpathy/minGPT
"""

from dataclasses import dataclass
import math
from typing import Optional

from einops import rearrange
import torch
import torch.nn as nn
from torch.nn import functional as F

from .kv_caching import KeysValues, KVCache
from line_profiler import line_profiler


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

    @property
    def max_tokens(self):
        return self.tokens_per_block * self.max_blocks


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.drop = nn.Dropout(config.embed_pdrop)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.num_layers)])
        self.ln_f = nn.LayerNorm(config.embed_dim)

    def generate_empty_keys_values(self, n: int, max_tokens: int) -> KeysValues:
        device = self.ln_f.weight.device  # Assumption that all submodules are on the same device
        return KeysValues(n, self.config.num_heads, max_tokens, self.config.embed_dim, self.config.num_layers, device)
        # return KeysValues(n, self.config.num_heads, 50, self.config.embed_dim, self.config.num_layers, device)

    def forward(self, sequences: torch.Tensor, past_keys_values: Optional[KeysValues] = None) -> torch.Tensor:
        assert past_keys_values is None or len(past_keys_values) == len(self.blocks)
        x = self.drop(sequences)
        for i, block in enumerate(self.blocks):
            x = block(x, None if past_keys_values is None else past_keys_values[i])

        x = self.ln_f(x)
        return x


# class Block(nn.Module):
#     def __init__(self, config: TransformerConfig) -> None:
#         super().__init__()
#         self.ln1 = nn.LayerNorm(config.embed_dim)
#         self.ln2 = nn.LayerNorm(config.embed_dim)
#         self.attn = SelfAttention(config)
#         self.mlp = nn.Sequential(
#             nn.Linear(config.embed_dim, 4 * config.embed_dim),
#             nn.GELU(),
#             nn.Linear(4 * config.embed_dim, config.embed_dim),
#             nn.Dropout(config.resid_pdrop),
#         )

#     def forward(self, x: torch.Tensor, past_keys_values: Optional[KeysValues] = None) -> torch.Tensor:
#         x_attn = self.attn(self.ln1(x), past_keys_values)
#         x = x + x_attn
#         x = x + self.mlp(self.ln2(x))
#         return x

from ding.torch_utils.network import GRUGatingUnit

class Block(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        # TODO
        self.gru_gating = False
        # self.gru_gating = True
        self.gru_bias  = 2.
        if self.gru_gating is True:
            self.gate1 = GRUGatingUnit(config.embed_dim, self.gru_bias)
            self.gate2 = GRUGatingUnit(config.embed_dim, self.gru_bias)

        self.ln1 = nn.LayerNorm(config.embed_dim)
        self.ln2 = nn.LayerNorm(config.embed_dim)
        self.attn = SelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.embed_dim, 4 * config.embed_dim),
            nn.GELU(),
            nn.Linear(4 * config.embed_dim, config.embed_dim),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x: torch.Tensor, past_keys_values: Optional[KeysValues] = None) -> torch.Tensor:
        x_attn = self.attn(self.ln1(x), past_keys_values)
        # x = x + x_attn
        x = self.gate1(x, x_attn) if self.gru_gating else x + x_attn
        # x = x + self.mlp(self.ln2(x))
        x = self.gate2(x, self.mlp(self.ln2(x))) if self.gru_gating else x + self.mlp(self.ln2(x))

        return x


class SelfAttention(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        assert config.embed_dim % config.num_heads == 0
        assert config.attention in ('causal', 'block_causal')
        self.config = config
        self.num_heads = config.num_heads

        self.key = nn.Linear(config.embed_dim, config.embed_dim)
        self.query = nn.Linear(config.embed_dim, config.embed_dim)
        self.value = nn.Linear(config.embed_dim, config.embed_dim)

        # key, query, value projections for all heads, but in a batch
        # self.c_attn = nn.Linear(config.embed_dim, 3 * config.embed_dim)

        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.proj = nn.Linear(config.embed_dim, config.embed_dim)

        causal_mask = torch.tril(torch.ones(config.max_tokens, config.max_tokens))
        block_causal_mask = torch.max(causal_mask, torch.block_diag(*[torch.ones(config.tokens_per_block, config.tokens_per_block) for _ in range(config.max_blocks)]))
        self.register_buffer('mask', causal_mask if config.attention == 'causal' else block_causal_mask)


    #@profile
    def forward(self, x: torch.Tensor, kv_cache: Optional[KVCache] = None) -> torch.Tensor:
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        if kv_cache is not None:
            b, nh, L, c = kv_cache.shape
            assert nh == self.num_heads and b == B and c * nh == C
        else:
            L = 0

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # q, k, v  = self.c_attn(x).split(self.config.embed_dim, dim=2)
        # k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) # (B, nh, T, hs)
        # q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) # (B, nh, T, hs)
        # v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) # (B, nh, T, hs)

        q = self.query(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)   # (B, nh, T, hs)
        k = self.key(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)     # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)   # (B, nh, T, hs)

        if kv_cache is not None:
            kv_cache.update(k, v)
            k, v = kv_cache.get()

        # method1: manual implementation of attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[L:L + T, :L + T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # TODO
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)

        # method2: efficient attention using Flash Attention CUDA kernels
        # 手动实现的掩码区域
        # manual_attn_mask = self.mask[L:L + T, :L + T]
        # # # https://github.com/pytorch/pytorch/blob/main/torch/nn/functional.py#L5243
        # manual_attn_mask = manual_attn_mask.masked_fill(manual_attn_mask == 0, float('-inf'))
        # manual_attn_mask = manual_attn_mask.masked_fill(manual_attn_mask != 0, 0)
        # y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=manual_attn_mask, dropout_p=self.config.attn_pdrop if self.training else 0, is_causal=False)
        
        y = rearrange(y, 'b h t e -> b t (h e)')
        y = self.resid_drop(self.proj(y))

        return y

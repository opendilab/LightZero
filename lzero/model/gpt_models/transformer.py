"""
Credits to https://github.com/karpathy/minGPT
"""

from dataclasses import dataclass
import math
import copy
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

    def forward(self, sequences: torch.Tensor, past_keys_values: Optional[KeysValues] = None, valid_context_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert past_keys_values is None or len(past_keys_values) == len(self.blocks)
        x = self.drop(sequences)
        for i, block in enumerate(self.blocks):
            x = block(x, None if past_keys_values is None else past_keys_values[i], valid_context_lengths)

        x = self.ln_f(x)
        return x


from ding.torch_utils.network import GRUGatingUnit

class Block(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        # TODO
        self.gru_gating = config.gru_gating
        # self.gru_gating = False
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

    def forward(self, x: torch.Tensor, past_keys_values: Optional[KeysValues] = None, valid_context_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        x_attn = self.attn(self.ln1(x), past_keys_values, valid_context_lengths)
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

        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.proj = nn.Linear(config.embed_dim, config.embed_dim)

        causal_mask = torch.tril(torch.ones(config.max_tokens, config.max_tokens))
        block_causal_mask = torch.max(causal_mask, torch.block_diag(*[torch.ones(config.tokens_per_block, config.tokens_per_block) for _ in range(config.max_blocks)]))
        self.register_buffer('mask', causal_mask if config.attention == 'causal' else block_causal_mask)


    def forward(self, x: torch.Tensor, kv_cache: Optional[KeysValues] = None, valid_context_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.size()
        if kv_cache is not None:
            b, nh, L, c = kv_cache.shape
            assert nh == self.num_heads and b == B and c * nh == C
        else:
            L = 0

        q = self.query(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)   # (B, nh, T, hs)
        k = self.key(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)     # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)   # (B, nh, T, hs)

        if kv_cache is not None:
            kv_cache.update(k, v)
            k, v = kv_cache.get()

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # # # # 处理mask
        # if valid_context_lengths is not None:
        #     # 最终得到的  mask.shape: (B, T, L + T)
        #     # 其中L是context长度，T是当前输入的长度, valid_context_lengths是context中有效的长度，位于context的后面valid_context_lengths长的一段
        #     mask = torch.ones(B, T, L + T, device=att.device)
        #     # 对每个样本,根据其有效长度,将无效的部分设为0
        #     for i in range(B):
        #         mask[i] = copy.deepcopy(self.mask[L:L + T, :L + T])
        #         if L - valid_context_lengths[i]>0:
        #             mask[i, :, :(L - valid_context_lengths[i])] = 0
        # else:
        #     # mask.shape: (B, nh, T, L + T)
        #     mask = self.mask[L:L + T, :L + T]
        # att.shape: (B, nh, T, L + T)
        # att = att.masked_fill(mask == 0, float('-inf'))

        if valid_context_lengths is not None:
            # 最终得到的  mask.shape: (B, T, L + T)
            # 其中L是context长度，T是当前输入的长度, valid_context_lengths是context中有效的长度，位于context的后面valid_context_lengths长的一段
            # mask = torch.ones(B, T, L + T, device=att.device)
            mask = torch.zeros(B, T, L + T, device=att.device)
            # 对每个样本,根据其有效长度,将无效的部分设为0
            for i in range(B):
                mask[i] = self.mask[L:L + T, :L + T].clone() # 不需要.clone()吗
                mask[i, :, :(L - valid_context_lengths[i])] = 0  # 无效的部分设为0
            # 将mask的维度调整为与att的后两个维度相同
            # (B, T, L + T) -> (B, 1, T, L + T) -> (B, nh, T, L + T)
            mask = mask.unsqueeze(1).expand(-1, att.size(1), -1, -1)
        else:
            # mask.shape: (T, L + T) 
            mask = self.mask[L:L + T, :L + T]

        # att.shape: (B, nh, T, L + T)
        att = att.masked_fill(mask == 0, float('-inf'))

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        y = rearrange(y, 'b h t e -> b t (h e)')
        y = self.resid_drop(self.proj(y))

        return y

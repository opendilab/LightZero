"""
<<<<<<< HEAD
The following code is modified from https://github.com/karpathy/nanoGPT.
=======
Modified from https://github.com/karpathy/nanoGPT

在原 transformer.py 基础上增加 LoRA 微调相关代码，
并通过传入配置参数控制 LoRA 微调的模块（默认是 attention 中的 k, q, v, proj 和 feed_forward）
保持原有代码的可扩展性。
>>>>>>> dev-multitask-clean
"""

import numpy as np
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn as nn
from torch.nn import functional as F
from ding.torch_utils.network import GRUGatingUnit
from einops import rearrange

from .kv_caching import KeysValues
from .moe import MoeLayer, MultiplicationFeedForward
from line_profiler import line_profiler
from lzero.model.common import SimNorm


#############################################
# 新增：LoRA 微调相关代码
#############################################
class LoRALinear(nn.Module):
    """
    LoRA 适配器包装的线性层。

    原理：
      使用冻结的原始 nn.Linear 层，并添加两个小型低秩矩阵，
      计算公式为：y = x @ W^T + scaling * ((drop(x) @ A^T) @ B^T)
      其中 A 和 B 为低秩参数，scaling = lora_alpha / r.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r if r > 0 else 1.0
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()

        # 原始权重（冻结参数，不更新）
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

        # 低秩矩阵参数（仅在 r > 0 时添加）
        if r > 0:
            # A 将 in_features 映射到低秩 r；B 从低秩 r 映射回 out_features
            self.lora_A = nn.Parameter(torch.randn(r, in_features) * 0.01)
            self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        else:
            self.lora_A = None
            self.lora_B = None

        # 冻结原始权重参数，保证仅更新 LoRA 参数
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 原始线性输出（冻结部分）
        result = F.linear(x, self.weight, self.bias)
        # 如启用了 LoRA，则加上低秩部分
        if self.r > 0:
            lora_out = F.linear(self.lora_dropout(x), self.lora_A)  # (…, r)
            lora_out = F.linear(lora_out, self.lora_B)                # (…, out_features)
            result = result + self.scaling * lora_out
        return result


def _maybe_wrap_linear(linear: nn.Linear, config, module_label: str) -> nn.Module:
    """
    辅助函数：当 config.lora_r > 0 且 module_label 存在于 config.lora_target_modules 时，
    将传入的线性层替换为 LoRALinear，并复制原始权重数据。

    module_label 的取值含义由上层逻辑定义，例如：
      - 若 module_label 为 "attn"，表示在 SelfAttention 中替换 k, q, v, proj 等层。
      - 若 module_label 为 "feed_forward"，表示在 Transformer Block 的 MLP 中替换线性层。
    """
    if config.lora_r > 0 and module_label in config.lora_target_modules:
        new_linear = LoRALinear(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=(linear.bias is not None),
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout
        )
        new_linear.weight.data.copy_(linear.weight.data)
        if linear.bias is not None:
            new_linear.bias.data.copy_(linear.bias.data)
        return new_linear
    else:
        return linear

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

    # LoRA 参数：
    lora_r: int = 0
    lora_alpha: int = 1
    lora_dropout: float = 0.0
    # 指定哪些模块应用 LoRA，默认：attention 中的 k, q, v, proj 和 feed_forward 层（当非 moe 模型时）
    lora_target_modules: list = None

    # Register Token 相关
    task_embed_option: str = "none"
    register_token_num: int = 4
    register_token_shared: bool = True

    # 其它配置项
    gru_gating: bool = False
    moe_in_transformer: bool = False
    multiplication_moe_in_transformer: bool = False
    num_experts_of_moe_in_transformer: int = 1

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

    def __init__(self, config: TransformerConfig, task_embed=None) -> None:
        super().__init__()
        self.config = config
        self.drop = nn.Dropout(config.embed_pdrop)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.num_layers)])
        self.ln_f = nn.LayerNorm(config.embed_dim)

<<<<<<< HEAD
        if self.config.rotary_emb:
            freqs_cis = precompute_freqs_cis(
                self.config.embed_dim // self.config.num_heads,
                self.config.max_seq_len * 2,
                self.config.rope_theta,
            )
            self.register_buffer("freqs_cis", freqs_cis)
=======
        self.task_embed = task_embed
        self.task_embed_option = self.config.task_embed_option  # Strategy for task embeddings
        self.register_token_shared = True

        # TODO: 共享模式下，所有任务使用同一参数

        if self.task_embed_option == "register_task_embed":
            self.use_register_token = True # TODO
            # Register token setup
            self.register_token_num = config.register_token_num if hasattr(config, "register_token_num") else 4

            # 判断是否采用共享模式
            self.register_token_shared = getattr(config, "register_token_shared", True)
            if self.register_token_shared:
                # print(f'self.register_token_shared:{self.register_token_shared}')
                # print(f'='*20)
                # 共享模式：所有任务使用同一个 register_tokens 参数，形状为 (register_token_num, embed_dim)
                self.register_tokens = nn.Parameter(torch.empty(self.register_token_num, config.embed_dim))
                nn.init.xavier_uniform_(self.register_tokens)
            else:
                # 非共享模式：依赖外部传入的 task_embed 模块来生成 task embedding，
                # 并通过 SimNorm 归一化后复制出 register token
                self.task_embed = task_embed  # 外部传入的模块，如 nn.Embedding
                self.sim_norm = SimNorm(simnorm_dim=config.embed_dim) # Normalization for task embeddings

        else:
            self.use_register_token = False # TODO
   

    def add_register_tokens(self, sequences: torch.Tensor, task_id: int) -> torch.Tensor:
        """
        将 register_token_num 个 Register Token 拼接到序列最前面。

        Arguments:
            - sequences (:obj:`torch.Tensor`): (B, T, C)
            - task_id (:obj:`int`): 当前任务的 ID

        Returns:
            - new_sequences (:obj:`torch.Tensor`): (B, T + register_token_num, C)
        """
        B = sequences.size(0)
        device = sequences.device

        if self.register_token_shared:
            # 共享模式：直接使用同一组 register_tokens 参数
            # register_tokens 形状为 (register_token_num, embed_dim)
            register_tokens = self.register_tokens  
            register_tokens = register_tokens.unsqueeze(0).expand(B, -1, -1)  # 形状 (B, register_token_num, embed_dim)
        else:
            # 非共享模式：依靠 task_embed 动态生成 task embedding，然后复制出 register tokens
            task_embedding = self.task_embed(torch.tensor([task_id], device=device))  # (1, embed_dim)
            task_embedding = self.sim_norm(task_embedding.view(1, -1)).view(-1)         # (embed_dim,)
            register_tokens = task_embedding.unsqueeze(0).expand(self.register_token_num, -1)  # (register_token_num, embed_dim)
            register_tokens = register_tokens.unsqueeze(0).expand(B, -1, -1)  # (B, register_token_num, embed_dim)

        new_sequences = torch.cat([sequences, register_tokens], dim=1)  # 在序列末尾拼接 register tokens (B, register_token_num + T, C)
        return new_sequences

    def remove_register_tokens_from_kv(self, past_keys_values: KeysValues) -> None:
        """
        移除所有层 KV 中最前面的 register_token_num 个 token，用于在 forward() 结束时调用。
        """
        if past_keys_values is None:
            return
        past_keys_values.remove_register_tokens(self.register_token_num)
>>>>>>> dev-multitask-clean

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

<<<<<<< HEAD
    def forward(self, sequences: torch.Tensor, past_keys_values: Optional[KeysValues] = None,
                valid_context_lengths: Optional[torch.Tensor] = None, start_pos: int = 0) -> torch.Tensor:
=======

    #@profile
    def forward(
        self,
        sequences: torch.Tensor,         # (B, T, C)
        past_keys_values: Optional[KeysValues] = None,
        valid_context_lengths: Optional[torch.Tensor] = None,
        task_id: int = 0
    ) -> torch.Tensor:
>>>>>>> dev-multitask-clean
        """
        Forward pass of the Transformer model.

        Arguments:
<<<<<<< HEAD
            - sequences (:obj:`torch.Tensor`): Input tensor of shape (batch_size, seq_length, embed_dim).
            - past_keys_values (:obj:`Optional[KeysValues]`): Precomputed keys and values for faster generation (default: None).
            - valid_context_lengths (:obj:`Optional[torch.Tensor]`): Valid lengths of context for masking (default: None).
            - start_pos (:obj:`int`): Starting position for rotary embeddings (default: 0).
=======
            - sequences (:obj:`torch.Tensor`): (B, T, C)
            - past_keys_values (:obj:`Optional[KeysValues]`): 缓存，用于推理时加速
            - valid_context_lengths (:obj:`Optional[torch.Tensor]`): 某些场景下可用的有效上下文长度
            - task_id (:obj:`int`): 任务 ID
>>>>>>> dev-multitask-clean

        Returns:
            - 输出张量 (B, T + register_token_num, C) 或 (B, T, C)，视是否添加 Register Token 而定
        """
<<<<<<< HEAD
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
=======
        # 若使用 Register Token，则将其拼到序列最前面
        # 训练阶段和推理阶段都统一处理
        if self.use_register_token:
            sequences = self.add_register_tokens(sequences, task_id)

        # 接入 dropout
        x = self.drop(sequences)

        # 逐层调用
        for i, block in enumerate(self.blocks):
            x = block(x,
                      None if past_keys_values is None else past_keys_values[i],
                      valid_context_lengths)

        # 最后层 LN
>>>>>>> dev-multitask-clean
        x = self.ln_f(x)

        # 如果 past_keys_values 不为 None，说明是推理阶段，此时我们需要把 KV 缓存中
        # 尾部多加的 Register Token 移除，以保证外键信息一致，不用修改外部逻辑
        # if self.use_register_token and (past_keys_values is not None):
        if self.use_register_token:
            self.remove_register_tokens_from_kv(past_keys_values)

        # TODO
        if self.use_register_token:
            # import ipdb; ipdb.set_trace()
            x = x[:, :-self.register_token_num, :]

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

        if config.moe_in_transformer:
            # 创Create multiple independent MLP instances
            self.experts = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(config.embed_dim, 4 * config.embed_dim),
                    nn.GELU(approximate='tanh'),
                    nn.Linear(4 * config.embed_dim, config.embed_dim),
                    nn.Dropout(config.resid_pdrop),
                ) for _ in range(config.num_experts_of_moe_in_transformer)
            ])
            self.feed_forward = MoeLayer(
                experts=self.experts,
                gate=nn.Linear(config.embed_dim, config.num_experts_of_moe_in_transformer, bias=False),
                num_experts_per_tok=1,
            )
            
            print("="*20)
            print(f'use moe in feed_forward of transformer, num of expert: {config.num_experts_of_moe_in_transformer}')
            print("="*20)
        elif config.multiplication_moe_in_transformer:
            # Create multiple FeedForward instances for multiplication-based MoE
            self.experts = nn.ModuleList([
                MultiplicationFeedForward(config) for _ in range(config.num_experts_of_moe_in_transformer)
            ])

            self.feed_forward = MoeLayer(
                experts=self.experts,
                gate=nn.Linear(config.embed_dim, config.num_experts_of_moe_in_transformer, bias=False),
                num_experts_per_tok=1,
            )

            print("="*20)
            print(f'use multiplication moe in feed_forward of transformer, num of expert: {config.num_experts_of_moe_in_transformer}')
            print("="*20)
        else:
            # self.feed_forward = nn.Sequential(
            #     nn.Linear(config.embed_dim, 4 * config.embed_dim),
            #     nn.GELU(approximate='tanh'),
            #     nn.Linear(4 * config.embed_dim, config.embed_dim),
            #     nn.Dropout(config.resid_pdrop),
            # )
            # 普通的 MLP，若在 feed_forward 上启用 LoRA，则对其中线性层进行包装
            self.feed_forward = nn.Sequential(
                _maybe_wrap_linear(nn.Linear(config.embed_dim, 4 * config.embed_dim), config, "feed_forward"),
                nn.GELU(approximate='tanh'),
                _maybe_wrap_linear(nn.Linear(4 * config.embed_dim, config.embed_dim), config, "feed_forward"),
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
            x = self.gate2(x, self.feed_forward(self.ln2(x)))
        else:
            x = x + x_attn
            x = x + self.feed_forward(self.ln2(x))

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

        self.task_embed_option = self.config.task_embed_option
        if self.task_embed_option == "register_task_embed":
            self.use_register_token = True # TODO
            # Register token setup
            self.register_token_num = config.register_token_num if hasattr(config, "register_token_num") else 4
        else:
            self.use_register_token = False # TODO

        self.num_heads = config.num_heads

        if config.lora_r > 0 and ("attn" in config.lora_target_modules):
            self.key = _maybe_wrap_linear(nn.Linear(config.embed_dim, config.embed_dim), config, "attn")
            self.query = _maybe_wrap_linear(nn.Linear(config.embed_dim, config.embed_dim), config, "attn")
            self.value = _maybe_wrap_linear(nn.Linear(config.embed_dim, config.embed_dim), config, "attn")
            self.proj = _maybe_wrap_linear(nn.Linear(config.embed_dim, config.embed_dim), config, "attn")
        else:
            self.key = nn.Linear(config.embed_dim, config.embed_dim)
            self.query = nn.Linear(config.embed_dim, config.embed_dim)
            self.value = nn.Linear(config.embed_dim, config.embed_dim)
            self.proj = nn.Linear(config.embed_dim, config.embed_dim)

        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)

        if self.use_register_token: # ======= TODO ========
            causal_mask = torch.tril(torch.ones(config.max_tokens+self.register_token_num*5, config.max_tokens+self.register_token_num*5))
        else:
            causal_mask = torch.tril(torch.ones(config.max_tokens, config.max_tokens))

        self.register_buffer('mask', causal_mask)

    #@profile
    def forward(self, x: torch.Tensor, kv_cache: Optional[KeysValues] = None,
<<<<<<< HEAD
                valid_context_lengths: Optional[torch.Tensor] = None,  freqs_cis: torch.Tensor = None) -> torch.Tensor:
=======
                valid_context_lengths: Optional[torch.Tensor] = None, ) -> torch.Tensor:
>>>>>>> dev-multitask-clean
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
            try:
                assert nh == self.num_heads and b == B and c * nh == C, "Cache dimensions do not match input dimensions."
            except Exception as e:
                print('debug')
        else:
            L = 0

        q = self.query(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)  # (B, num_heads, T, head_size)
        k = self.key(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)  # (B, num_heads, T, head_size)
        v = self.value(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)  # (B, num_heads, T, head_size)
        
        if self.config.rotary_emb:
            q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)

        if kv_cache is not None:
            # import ipdb; ipdb.set_trace()
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

        # import ipdb; ipdb.set_trace()

        # Adjust mask for register tokens if applicable
        if self.use_register_token and self.register_token_num > 0:
            # Allow all positions to attend to the last `register_token_num` tokens
            register_mask = mask.clone()  # (T, L + T)
            register_mask[-self.register_token_num:, :] = 1  # Allow register tokens to see all positions
            register_mask[:, -self.register_token_num:] = 1  # Allow all positions to see register tokens
            mask = register_mask

            if kv_cache is not None:
                # =============TODO=============
                # import ipdb; ipdb.set_trace()
                b, nh, new_L, c = kv_cache.shape # new_L可能小于L + T
                mask = mask[:,-new_L:]
            # else:
            #     import ipdb; ipdb.set_trace()


        # att.shape: (B, num_heads, T, L + T)
        att = att.masked_fill(mask == 0, float('-inf'))

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        # import ipdb; ipdb.set_trace()
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
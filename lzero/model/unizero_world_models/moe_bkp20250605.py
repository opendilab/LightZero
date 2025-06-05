import dataclasses
from typing import List

import torch
import torch.nn.functional as F
from simple_parsing.helpers import Serializable
from torch import nn

from .transformer import _maybe_wrap_linear

# _maybe_wrap_linear(nn.Linear(config.embed_dim, 4 * config.embed_dim), config, "feed_forward")

# https://github.com/mistralai/mistral-inference/blob/main/src/mistral_inference/moe.py
# https://github.com/mistralai/mistral-inference/blob/main/src/mistral_inference/transformer_layers.py#L149
# Modified from https://github.com/mistralai/mistral-inference/blob/main/src/mistral_inference/transformer.py#L108
class MultiplicationFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.moe_use_lora:
            self.w1 = _maybe_wrap_linear(nn.Linear(config.embed_dim, 4 * config.embed_dim, bias=False), config, "feed_forward")
            self.w2 = _maybe_wrap_linear(nn.Linear(4 * config.embed_dim, config.embed_dim, bias=False), config, "feed_forward")
            self.w3 = _maybe_wrap_linear(nn.Linear(config.embed_dim, 4 * config.embed_dim, bias=False), config, "feed_forward")
        else:
            self.w1 = nn.Linear(config.embed_dim, 4 * config.embed_dim, bias=False)
            self.w2 = nn.Linear(4 * config.embed_dim, config.embed_dim, bias=False)
            self.w3 = nn.Linear(config.embed_dim, 4 * config.embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))  # type: ignore

@dataclasses.dataclass
class MoeArgs(Serializable):
    num_experts: int
    num_experts_per_tok: int


class MoELayer(nn.Module):
    """
    Mixture-of-Experts (MoE) 层的实现，参考了如下的设计：
    
    - 根据输入 x 的形状先展平为二维张量（[batch_size, dim]）
    - 使用门控网络（gate）为每个 token 计算各专家的 logits，并选出前 k 个专家（k = num_experts_per_tok）
    - 对于选中的每个专家，对应 token 调用该专家的前向传播，将专家计算结果乘以门控权重后累积
    - 可选支持共享专家分支 shared_expert 对所有 token 做统一处理
    - 最后恢复输入的原始形状返回 
    
    Attributes:
        dim (int): 输入特征的维度
        num_experts (int): 专家数量
        num_experts_per_tok (int): 每个 token 激活的专家个数
        gate (nn.Module): 门控模块，用于生成专家路由 logits
        experts (nn.ModuleList): 专家模块列表
        shared_expert (nn.Module or None): 用于所有 token 的共享专家分支（如果配置了 n_shared_experts）
    """
    def __init__(self, config, experts: List[nn.Module], gate: nn.Module, num_experts_per_tok: int = 1):
        super().__init__()
        self.dim = config.embed_dim
        self.num_experts = len(experts)
        self.num_experts_per_tok = num_experts_per_tok
        self.gate = gate
        self.experts = nn.ModuleList(experts)
        
        # 如果配置中指定了共享专家数量，则构建共享专家分支
        if hasattr(config, "n_shared_experts") and config.n_shared_experts > 0:
            self.shared_expert = nn.Sequential(
                nn.Linear(self.dim, config.n_shared_experts * (4 * self.dim)),
                nn.GELU(),
                nn.Linear(config.n_shared_experts * (4 * self.dim), self.dim)
            )
        else:
            self.shared_expert = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 保存原始形状后将 x reshape 为二维张量： [batch_size * seq_len, dim]
        original_shape = x.size()
        x = x.view(-1, self.dim)
        
        # 计算门控 logits，shape 为 [N, num_experts]，N 为 token 数量
        gate_logits = self.gate(x)
        # 选取每个 token 得分最高的 k 个专家
        weights, indices = torch.topk(gate_logits, self.num_experts_per_tok, dim=1)
        # 对选中的 logits 做 softmax，获得归一化权重
        weights = F.softmax(weights, dim=1).to(x.dtype)
        
        # 初始化存放专家计算输出的张量
        expert_output = torch.zeros_like(x)
        
        # 遍历所有专家，对被该专家选择的 token 分支进行计算
        for expert_id in range(self.num_experts):
            # 通过 where 找到 indices 中等于当前 expert_id 的 token 索引
            batch_idx, expert_tok_idx = torch.where(indices == expert_id)
            if batch_idx.numel() == 0:
                continue
            token_subset = x[batch_idx]  # 选中的 token，形状 [num_tokens, dim]
            # 调用当前专家模块计算输出
            output_expert = self.experts[expert_id](token_subset)
            # 获取对应 token 的权重，注意 weights 的形状为 [N, num_experts_per_tok]
            token_weights = weights[batch_idx, expert_tok_idx].unsqueeze(-1)
            expert_output[batch_idx] += output_expert * token_weights

        # 如果使用了共享专家分支，则加上其输出
        if self.shared_expert is not None:
            shared_output = self.shared_expert(x)
            output = expert_output + shared_output
        else:
            output = expert_output

        # 恢复原始形状后返回结果
        return output.view(original_shape)
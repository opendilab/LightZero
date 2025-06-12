import dataclasses
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from simple_parsing.helpers import Serializable

# -------------------------------------------------
# 一些辅助：自动检测分布式环境
# -------------------------------------------------
# if torch.distributed.is_available() and torch.distributed.is_initialized():
#     world_size = torch.distributed.get_world_size()
#     rank = torch.distributed.get_rank()
# else:
#     world_size = 1
#     rank = 0
world_size = 1
rank = 0

# -------------------------------------------------
# 配置
# -------------------------------------------------
@dataclasses.dataclass
class MoeConfig(Serializable):
    embed_dim: int
    num_experts_total: int = 8              # 总路由专家数
    num_experts_per_tok: int = 1            # 每个 token 激活的专家数 (Top-k)
    moe_inter_dim: int = None               # 隐层维度，如为 None 则取 4 * embed_dim
    num_shared_experts: int = 1            # 可选共享专家数量（所有 token 都会经过）
    # ——兼容原配置——
    resid_pdrop: float = 0.0                # dropout 给 Transformer 用
    num_experts_of_moe_in_transformer: int = 8


# -------------------------------------------------
# Expert
# -------------------------------------------------
class Expert(nn.Module):
    """
    乘法前馈专家:  w2( silu(w1(x)) * w3(x) )
    """
    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, inter_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# -------------------------------------------------
# Gate
# -------------------------------------------------
class TopKGate(nn.Module):
    """
    返回 (weights, indices)
      • weights:  [batch⋯ , k]
      • indices:  [batch⋯ , k]
    """
    def __init__(self, dim: int, num_experts: int, k: int):
        super().__init__()
        self.k = k
        self.proj = nn.Linear(dim, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.proj(x)                       # [..., E]
        weights, indices = torch.topk(logits, self.k, dim=-1)  # (..., k)
        weights = F.softmax(weights, dim=-1, dtype=x.dtype)
        return weights, indices                     # 同参考实现返回顺序


# -------------------------------------------------
# MoE Layer
# -------------------------------------------------
class MoELayer(nn.Module):
    """
    • 按参考实现构造，支持多机自动 All-Reduce
    • 当 world_size==1 时退化为单机
    """
    def __init__(self, cfg: MoeConfig):
        super().__init__()

        self.dim = cfg.embed_dim
        self.n_routed_experts = cfg.num_experts_total
        # assert self.n_routed_experts % world_size == 0, \
        #     f"num_experts_total({self.n_routed_experts}) 必须能被 world_size({world_size}) 整除"

        # ——本地专家范围——
        self.n_local_experts = self.n_routed_experts // world_size
        self.expert_start = rank * self.n_local_experts
        self.expert_end   = self.expert_start + self.n_local_experts

        self.n_activated_experts = cfg.num_experts_per_tok
        inter_dim = cfg.moe_inter_dim or 4 * self.dim

        # Gate
        self.gate = TopKGate(self.dim, self.n_routed_experts, self.n_activated_experts)

        # 路由专家：只有本 rank 的专家才真正实例化
        experts: List[nn.Module | None] = []
        for idx in range(self.n_routed_experts):
            if self.expert_start <= idx < self.expert_end:
                experts.append(Expert(self.dim, inter_dim))
            else:
                experts.append(None)   # 占位，便于下标一致
        self.experts = nn.ModuleList([e for e in experts if e is not None])  # register 仅本地专家

        # 共享专家（可选）
        if cfg.num_shared_experts > 0:
            self.shared_experts = nn.Sequential(
                nn.Linear(self.dim, cfg.num_shared_experts * inter_dim, bias=False),
                nn.GELU(),
                nn.Linear(cfg.num_shared_experts * inter_dim, self.dim, bias=False)
            )
        else:
            self.shared_experts = None

    # -------------------------------------------------
    # 前向：与参考实现保持一致
    # -------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: shape = [*, dim] 或 [B, T, dim]
        """
        original_shape = x.shape
        x_flat = x.view(-1, self.dim)                           # [N, dim]

        # -------- Routing --------
        weights, indices = self.gate(x_flat)                    # [N, k]
        y = torch.zeros_like(x_flat)                            # 聚合结果

        # 每个专家对应路由到的样本数
        counts = torch.bincount(indices.flatten(),
                                 minlength=self.n_routed_experts).tolist()

        # 遍历本地专家
        for local_idx, global_idx in enumerate(
                range(self.expert_start, self.expert_end)):
            if counts[global_idx] == 0:
                continue
            expert = self.experts[local_idx]                    # 实例化过的
            # 找到路由到该专家的 sample
            sample_rows, nth = torch.where(indices == global_idx)
            # 取权重
            sample_weights = weights[sample_rows, nth][:, None]  # [m, 1]
            # 计算并加权
            y[sample_rows] += sample_weights * expert(x_flat[sample_rows])

        # -------- 共享专家（可选）--------
        if self.shared_experts is not None:
            y += self.shared_experts(x_flat)

        # -------- 多机 All-Reduce --------
        if world_size > 1:
            torch.distributed.all_reduce(y)

        return y.view(original_shape)
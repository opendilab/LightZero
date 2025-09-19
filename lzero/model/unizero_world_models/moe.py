import dataclasses
from typing import List

import torch
import torch.nn.functional as F
from simple_parsing.helpers import Serializable
from torch import nn
import torch.distributed as dist
from lzero.model.unizero_world_models.transformer import _maybe_wrap_linear

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
        self.config=config

        # 如果配置中指定了共享专家数量，则构建共享专家分支
        if hasattr(config, "n_shared_experts") and config.n_shared_experts > 0:
            self.shared_expert = nn.Sequential(
                nn.Linear(self.dim, config.n_shared_experts * (4 * self.dim)),
                nn.GELU(),
                nn.Linear(config.n_shared_experts * (4 * self.dim), self.dim)
            )
        else:
            self.shared_expert = None
        
        # GPU memory expert selection statistics collector - multi-granularity sliding windows
        self.device = next(iter(experts)).w1.weight.device if experts else torch.device('cuda')
        
        # Sliding window configuration
        self.window_sizes = {
            'immediate': 100,    # Immediate statistics (last 100 steps)
            'short': 1000,       # Short-term statistics (last 1000 steps)
            'medium': 10000,     # Medium-term statistics (last 10000 steps)
            'long': 100000       # Long-term statistics (last 100000 steps)
        }
        
        # GPU statistics buffer: task_id -> {window_type -> [expert selection history]}
        self.expert_stats_gpu = {}
        self.step_count = 0


    def forward(self, x: torch.Tensor, task_id: int = None) -> torch.Tensor:
        # 保存原始形状后将 x reshape 为二维张量： [batch_size * seq_len, dim]
        original_shape = x.size()
        x = x.view(-1, self.dim)
        expert_output=x
        if self.num_experts!=0:
            # 计算门控 logits，shape 为 [N, num_experts]，N 为 token 数量
            gate_logits = self.gate(x)
            # 选取每个 token 得分最高的 k 个专家
            weights, indices = torch.topk(gate_logits, self.num_experts_per_tok, dim=1)
            # 对选中的 logits 做 softmax，获得归一化权重
            weights = F.softmax(weights, dim=1).to(x.dtype)
            
            if self.training and task_id is not None:
                self._collect_expert_selection_stats(task_id, indices)

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
    
    def _collect_expert_selection_stats(self, task_id: int, indices: torch.Tensor):
        """
        Overview:
            Collect expert selection statistics in GPU memory using multi-granularity sliding windows.
            Maintains separate rolling buffers for different time window sizes to track expert usage patterns.
        Arguments:
            - task_id (:obj:`int`): The identifier of the current task.
            - indices (:obj:`torch.Tensor`): Expert indices selected by the router for the current batch.
        Shapes:
            - indices: :math:`(N, k)` where N is batch size and k is number of experts per token.
        Examples:
            >>> # Collect stats for task 0 with expert indices
            >>> indices = torch.tensor([[0, 2], [1, 3]])  # batch_size=2, k=2
            >>> moe_layer._collect_expert_selection_stats(task_id=0, indices=indices)
        """
        self.step_count += 1
        
        if task_id not in self.expert_stats_gpu:
            self.expert_stats_gpu[task_id] = {}
            for window_type in self.window_sizes.keys():
                self.expert_stats_gpu[task_id][window_type] = torch.zeros(
                    self.window_sizes[window_type], 
                    self.num_experts, 
                    dtype=torch.float32, 
                    device=self.device
                )
        
        # Calculate expert selection frequency for current batch
        indices_flat = indices.flatten()  # [N*k]
        expert_counts = torch.zeros(self.num_experts, device=self.device, dtype=torch.float32)
        for expert_id in range(self.num_experts):
            expert_counts[expert_id] = (indices_flat == expert_id).sum().float()
        
        # Update sliding windows for all granularities
        for window_type, window_size in self.window_sizes.items():
            buffer = self.expert_stats_gpu[task_id][window_type]
            # Sliding window: new data goes to the end, old data moves forward
            buffer[:-1] = buffer[1:].clone()
            buffer[-1] = expert_counts
    
    def get_expert_selection_stats(self, task_id: int = None):
        """
        Overview:
            Get multi-granularity expert selection frequency statistics.
            Simplified version that directly returns current data without complex aggregation.
        Arguments:
            - task_id (:obj:`int`, optional): The identifier of the specific task. If None, returns stats for all tasks.
        Returns:
            - stats (:obj:`dict`): Dictionary containing expert selection statistics.
                                  Structure: {task_id: {window_type: {frequencies, total_counts, total_selections, data_points}}}
        Examples:
            >>> # Get stats for all tasks
            >>> all_stats = moe_layer.get_expert_selection_stats()
            >>> # Get stats for specific task
            >>> task_stats = moe_layer.get_expert_selection_stats(task_id=0)
        """
        if task_id is None:
            # Return statistics for all tasks
            all_stats = {}
            for tid in self.expert_stats_gpu.keys():
                all_stats[tid] = self._compute_task_stats(tid)
            return all_stats
        else:
            # Return statistics for specified task
            return self._compute_task_stats(task_id)
    
    def _compute_task_stats(self, task_id: int):
        """
        Overview:
            Compute multi-granularity statistics for a specified task.
            Processes expert selection data across different time window granularities.
        Arguments:
            - task_id (:obj:`int`): The identifier of the task to compute statistics for.
        Returns:
            - stats (:obj:`dict`): Dictionary containing computed statistics for each window type.
                                  Structure: {window_type: {frequencies, total_counts, total_selections, data_points}}
        Shapes:
            - frequencies: :math:`(num\_experts,)` normalized selection frequencies per expert.
            - total_counts: :math:`(num\_experts,)` absolute selection counts per expert.
        Examples:
            >>> # Compute stats for task 0
            >>> task_stats = moe_layer._compute_task_stats(task_id=0)
            >>> immediate_freq = task_stats['immediate']['frequencies']
        """
        if task_id not in self.expert_stats_gpu:
            return {}
        
        stats = {}
        for window_type, buffer in self.expert_stats_gpu[task_id].items():
            # Simplified version: directly average all existing data, ignoring whether window is full
            # buffer shape: [window_size, num_experts]
            total_counts = buffer.sum(dim=0)  # [num_experts]
            total_selections = total_counts.sum()
            
            if total_selections > 0:
                frequencies = total_counts / total_selections
            else:
                frequencies = torch.zeros(self.num_experts, device=self.device)
            
            stats[window_type] = {
                'frequencies': frequencies,  # Keep tensor format
                'total_counts': total_counts,  # Keep tensor format  
                'total_selections': total_selections.item(),
                'data_points': min(self.step_count, self.window_sizes[window_type])
            }
        
        return stats
    
    def reset_expert_selection_stats(self):
        """
        Overview:
            Reset expert selection statistics by clearing all accumulated data.
            Clears GPU memory buffers and resets step counter to initial state.
        Examples:
            >>> # Reset all expert selection statistics
            >>> moe_layer.reset_expert_selection_stats()
        """
        self.expert_stats_gpu.clear()
        self.step_count = 0

class MoELayerOptimized(nn.Module):
    """
    Overview:
        Optimized MoE layer that maintains interface consistency with original MoELayer.
        Provides end-to-end forward pass with O(N_token + ΣE_i) complexity,
        where ΣE_i is the total number of tokens actually processed by all experts.
    Interfaces:
        - __init__: Initialize the optimized MoE layer with experts and gating mechanism.
        - forward: Perform optimized forward pass through the MoE layer.
    """
    def __init__(self, config, experts: List[nn.Module], gate: nn.Module,
                 num_experts_per_tok: int = 1):
        """
        Overview:
            Initialize the optimized MoE layer with configuration, experts, and gating mechanism.
            Sets up expert modules, routing gate, and optional shared experts.
        Arguments:
            - config (:obj:`object`): Configuration object containing model parameters like embed_dim and n_shared_experts.
            - experts (:obj:`List[nn.Module]`): List of expert neural network modules.
            - gate (:obj:`nn.Module`): Gating network for routing tokens to experts.
            - num_experts_per_tok (:obj:`int`, optional): Number of experts to select per token. Default is 1.
        Examples:
            >>> experts = [nn.Linear(512, 512) for _ in range(8)]
            >>> gate = nn.Linear(512, 8)
            >>> moe_layer = MoELayerOptimized(config, experts, gate, num_experts_per_tok=2)
        """
        super().__init__()
        self.dim = config.embed_dim
        self.num_experts = len(experts)
        self.num_experts_per_tok = num_experts_per_tok
        self.gate = gate
        self.experts = nn.ModuleList(experts)

        self.use_shared = getattr(config, "n_shared_experts", 0) > 0
        if self.use_shared:
            self.shared_expert = nn.Sequential(
                nn.Linear(self.dim, config.n_shared_experts * (4 * self.dim)),
                nn.GELU(),
                nn.Linear(config.n_shared_experts * (4 * self.dim), self.dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Overview:
            Perform optimized forward pass through the MoE layer.
            Routes tokens to appropriate experts and combines their outputs efficiently.
        Arguments:
            - x (:obj:`torch.Tensor`): Input tensor containing token embeddings.
        Returns:
            - output (:obj:`torch.Tensor`): Processed tensor after expert routing and combination.
        Shapes:
            - x: :math:`(B, T, D)` where B is batch size, T is sequence length, D is embedding dimension.
            - output: :math:`(B, T, D)` same shape as input.
        Examples:
            >>> x = torch.randn(2, 10, 512)  # batch_size=2, seq_len=10, embed_dim=512
            >>> output = moe_layer.forward(x)
            >>> print(output.shape)  # torch.Size([2, 10, 512])
        """          # [B, T, D]
        B, T, D = x.shape
        x_flat = x.reshape(-1, D)                                # [N, D]; N = B*T

        # -------- 1. Routing ----------
        gate_logits = self.gate(x_flat)                          # [N, E]
        weights, topk_idx = torch.topk(
            gate_logits, self.num_experts_per_tok, dim=1
        )                                                        # [N, k]

        weights = F.softmax(weights, dim=1).to(x.dtype)          # [N, k]

        # ---- 2. Flatten token-expert pairs ----
        N, k = weights.shape
        flat_token_idx  = torch.arange(N, device=x.device).repeat_interleave(k)  # [N*k]
        flat_expert_idx = topk_idx.reshape(-1)                                    # [N*k]
        flat_weight     = weights.reshape(-1, 1)                                  # [N*k, 1]
        flat_input      = x_flat[flat_token_idx]                                  # [N*k, D]

        # ---- 3. Group by expert ----
        sort_order      = torch.argsort(flat_expert_idx)                          # [N*k]
        flat_expert_idx = flat_expert_idx[sort_order]
        flat_token_idx  = flat_token_idx[sort_order]
        flat_weight     = flat_weight[sort_order]
        flat_input      = flat_input[sort_order]

        # Sample count for each expert
        counts = torch.bincount(flat_expert_idx, minlength=self.num_experts)      # [E]

        # Prepare output buffer
        out_buffer = torch.zeros_like(flat_input)                                 # [N*k, D]

        # ---- 4. Process each expert sequentially ----
        ptr = 0
        for eid, num in enumerate(counts.tolist()):
            if num == 0:
                continue
            seg = slice(ptr, ptr + num)
            out_buffer[seg] = self.experts[eid](flat_input[seg])
            ptr += num

        # ---- 5. Weight and scatter back to tokens ----
        out_buffer.mul_(flat_weight)                                              # inplace weighting
        token_output = torch.zeros_like(x_flat)                                   # [N, D]
        token_output.index_add_(0, flat_token_idx, out_buffer)

        # ---- 6. Shared experts (if any) ----
        if self.use_shared:
            token_output.add_(self.shared_expert(x_flat))

        return token_output.reshape(B, T, D)
import dataclasses
from typing import List

import torch
import torch.nn.functional as F
from simple_parsing.helpers import Serializable
from torch import nn
from .transformer import _maybe_wrap_linear

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
    Mixture-of-Experts (MoE) layer.
    Design:
        - Flatten input to 2D [N, dim] where N = batch_size * seq_len.
        - A gating module produces logits over experts for each token.
        - Select top-k experts per token (k = num_experts_per_tok), softmax the
        selected logits to get normalized weights, and combine expert outputs
        weighted by those gate weights.
        - Optionally add a shared expert branch applied to all tokens.
        - Finally, restore the original shape.
    Attributes:
        dim (int): Input feature dimension.
        num_experts (int): Number of experts.
        num_experts_per_tok (int): Top-k experts activated per token.
        gate (nn.Module): Gating module that outputs logits of shape [N, num_experts].
        experts (nn.ModuleList): List of expert modules.
        shared_expert (nn.Module or None): Optional shared expert used for all tokens
            when `config.n_shared_experts > 0`.
    """
    def __init__(self, config, experts: List[nn.Module], gate: nn.Module, num_experts_per_tok: int = 1):
        super().__init__()
        self.dim = config.embed_dim
        self.num_experts = len(experts)
        self.dim = config.embed_dim
        self.num_experts = len(experts)
        self.num_experts_per_tok = num_experts_per_tok
        self.gate = gate
        self.experts = nn.ModuleList(experts)
        
        # Optional shared expert branch
        if hasattr(config, "n_shared_experts") and config.n_shared_experts > 0:
            self.shared_expert = nn.Sequential(
                nn.Linear(self.dim, config.n_shared_experts * (4 * self.dim)),
                nn.GELU(),
                nn.Linear(config.n_shared_experts * (4 * self.dim), self.dim)
            )
        else:
            self.shared_expert = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Save shape and flatten to [N, dim]
        original_shape = x.size()
        x = x.view(-1, self.dim)
        
        # Gate logits: [N, num_experts]
        gate_logits = self.gate(x)
        # Top-k experts per token
        weights, indices = torch.topk(gate_logits, self.num_experts_per_tok, dim=1)
        weights = F.softmax(weights, dim=1).to(x.dtype)
        # Accumulate expert outputs
        expert_output = torch.zeros_like(x)
        # For each expert, gather the tokens routed to it
        for expert_id in range(self.num_experts):
            batch_idx, expert_tok_idx = torch.where(indices == expert_id)
            if batch_idx.numel() == 0:
                continue
            token_subset = x[batch_idx]  # [num_tokens_routed, dim]
            output_expert = self.experts[expert_id](token_subset)
            # Get the corresponding token weights; note that `weights` has shape [N, num_experts_per_tok]
            token_weights = weights[batch_idx, expert_tok_idx].unsqueeze(-1)
            expert_output[batch_idx] += output_expert * token_weights

        # If a shared expert branch is configured, add its output
        if self.shared_expert is not None:
            shared_output = self.shared_expert(x)
            output = expert_output + shared_output
        else:
            output = expert_output

        # Restore the original shape and return the result
        return output.view(original_shape)
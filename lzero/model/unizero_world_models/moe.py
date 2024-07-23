import dataclasses
from typing import List

import torch
import torch.nn.functional as F
from simple_parsing.helpers import Serializable
from torch import nn

# Modified from https://github.com/mistralai/mistral-inference/blob/main/src/mistral_inference/transformer.py#L108
class MultiplicationFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.w1 = nn.Linear(config.embed_dim, 4 * config.embed_dim, bias=False)
        self.w2 = nn.Linear(4 * config.embed_dim, config.embed_dim, bias=False)
        self.w3 = nn.Linear(config.embed_dim, 4 * config.embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))  # type: ignore

@dataclasses.dataclass
class MoeArgs(Serializable):
    num_experts: int
    num_experts_per_tok: int


class MoeLayer(nn.Module):
    def __init__(self, experts: List[nn.Module], gate: nn.Module, num_experts_per_tok=1):
        super().__init__()
        assert len(experts) > 0
        self.experts = nn.ModuleList(experts)
        self.gate = gate
        self.num_experts_per_tok = num_experts_per_tok

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # if len(self.experts) == 1:
        #     # 只有一个专家时，直接使用该专家
        #     return self.experts[0](inputs)

        gate_logits = self.gate(inputs)
        weights, selected_experts = torch.topk(gate_logits, self.num_experts_per_tok)
        weights = F.softmax(weights, dim=1, dtype=torch.float).to(inputs.dtype)
        results = torch.zeros_like(inputs)
        for i, expert in enumerate(self.experts):
            # batch_idx, nth_expert = torch.where(selected_experts == i)
            # results[batch_idx] += weights[batch_idx, nth_expert, None] * expert(inputs[batch_idx])
            batch_idx, token_idx, nth_expert = torch.where(selected_experts == i)
            results[batch_idx, token_idx] += weights[batch_idx, token_idx, nth_expert][:, None] * expert(inputs[batch_idx, token_idx])
        return results
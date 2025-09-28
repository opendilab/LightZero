import dataclasses
from typing import List, Any

import torch
import torch.nn.functional as F
from simple_parsing.helpers import Serializable
from torch import nn

from lzero.model.unizero_world_models.transformer import _maybe_wrap_linear

# Note: The following lines are examples of how _maybe_wrap_linear might be used.
# _maybe_wrap_linear(nn.Linear(config.embed_dim, 4 * config.embed_dim), config, "feed_forward")

# This implementation is inspired by the following sources:
# https://github.com/mistralai/mistral-inference/blob/main/src/mistral_inference/moe.py
# https://github.com/mistralai/mistral-inference/blob/main/src/mistral_inference/transformer_layers.py#L149
# Modified from https://github.com/mistralai/mistral-inference/blob/main/src/mistral_inference/transformer.py#L108


class MultiplicationFeedForward(nn.Module):
    """
    Overview:
        Implements the SwiGLU (Swish-Gated Linear Unit) feed-forward layer, a variant of a transformer feed-forward network
        that uses element-wise multiplication of two linear projections, one of which is passed through a SiLU activation.
        This is often expressed as: FFN_SwiGLU(x) = (SiLU(x @ W1) * (x @ W3)) @ W2.
    """

    def __init__(self, config: Any) -> None:
        """
        Overview:
            Initializes the MultiplicationFeedForward layer.
        Arguments:
            - config (:obj:`Any`): A configuration object containing model hyperparameters.
                It is expected to have `embed_dim` (int) and `moe_use_lora` (bool).
        """
        super().__init__()
        hidden_dim = 4 * config.embed_dim
        if config.moe_use_lora:
            self.w1 = _maybe_wrap_linear(nn.Linear(config.embed_dim, hidden_dim, bias=False), config, "feed_forward")
            self.w2 = _maybe_wrap_linear(nn.Linear(hidden_dim, config.embed_dim, bias=False), config, "feed_forward")
            self.w3 = _maybe_wrap_linear(nn.Linear(config.embed_dim, hidden_dim, bias=False), config, "feed_forward")
        else:
            self.w1 = nn.Linear(config.embed_dim, hidden_dim, bias=False)
            self.w2 = nn.Linear(hidden_dim, config.embed_dim, bias=False)
            self.w3 = nn.Linear(config.embed_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Overview:
            Performs the forward pass of the SwiGLU layer.
        Arguments:
            - x (:obj:`torch.Tensor`): The input tensor.
        Returns:
            - torch.Tensor: The output tensor after applying the SwiGLU transformation.
        """
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))


@dataclasses.dataclass
class MoeArgs(Serializable):
    """
    Overview:
        Dataclass for storing Mixture-of-Experts (MoE) configuration arguments.
    """
    num_experts: int  # The total number of experts in the MoE layer.
    num_experts_per_tok: int  # The number of experts to route each token to (k).


class MoELayer(nn.Module):
    """
    Overview:
        A straightforward implementation of a Mixture-of-Experts (MoE) layer.
        This version iterates through each expert and processes the tokens routed to it.
        While clear and easy to understand, it can be less efficient than vectorized approaches.

        The process is as follows:
        1. The input tensor `x` is flattened from [B, T, D] to [N, D], where N = B * T.
        2. A gating network calculates logits for each token to determine expert assignment.
        3. For each token, the top-k experts are selected based on the logits.
        4. The layer iterates through each expert, gathers all tokens assigned to it,
           and computes their outputs.
        5. The outputs are weighted by the gating scores and summed up.
        6. An optional shared expert can be applied to all tokens.
        7. The final tensor is reshaped to its original shape [B, T, D].

    Attributes:
        - dim (:obj:`int`): The dimension of the input features.
        - num_experts (:obj:`int`): The total number of experts.
        - num_experts_per_tok (:obj:`int`): The number of experts activated per token (top-k).
        - gate (:obj:`nn.Module`): The gating network that produces routing logits.
        - experts (:obj:`nn.ModuleList`): A list of expert networks.
        - shared_expert (:obj:`nn.Module` or `None`): An optional shared expert applied to all tokens.
    """

    def __init__(self, config: Any, experts: List[nn.Module], gate: nn.Module, num_experts_per_tok: int = 1) -> None:
        """
        Overview:
            Initializes the MoELayer.
        Arguments:
            - config (:obj:`Any`): A configuration object. Expected to have `embed_dim` and optionally `n_shared_experts`.
            - experts (:obj:`List[nn.Module]`): A list of PyTorch modules representing the experts.
            - gate (:obj:`nn.Module`): The gating module for routing tokens.
            - num_experts_per_tok (:obj:`int`): The number of experts to use for each token.
        """
        super().__init__()
        self.dim = config.embed_dim
        self.num_experts = len(experts)
        self.num_experts_per_tok = num_experts_per_tok
        self.gate = gate
        self.experts = nn.ModuleList(experts)

        # If specified in the config, create a shared expert branch.
        if hasattr(config, "n_shared_experts") and config.n_shared_experts > 0:
            # TODO: The architecture of the shared expert could be made more configurable.
            self.shared_expert = nn.Sequential(
                nn.Linear(self.dim, config.n_shared_experts * (4 * self.dim)),
                nn.GELU(),
                nn.Linear(config.n_shared_experts * (4 * self.dim), self.dim)
            )
        else:
            self.shared_expert = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Overview:
            Performs the forward pass for the MoE layer.
        Arguments:
            - x (:obj:`torch.Tensor`): The input tensor of shape [batch_size, seq_len, dim].
        Returns:
            - torch.Tensor: The output tensor with the same shape as the input.
        """
        # Store original shape and flatten input to 2D: [batch_size * seq_len, dim]
        original_shape = x.size()
        x = x.view(-1, self.dim)

        # Compute gate logits, shape: [num_tokens, num_experts]
        gate_logits = self.gate(x)
        # Select top-k experts for each token.
        weights, indices = torch.topk(gate_logits, self.num_experts_per_tok, dim=1)
        # Normalize the weights of selected experts using softmax.
        weights = F.softmax(weights, dim=1).to(x.dtype)

        # Initialize the output tensor for expert computations.
        expert_output = torch.zeros_like(x)

        # Iterate over each expert to compute outputs for the tokens routed to it.
        for expert_id in range(self.num_experts):
            # Find the tokens that have this expert in their top-k list.
            batch_idx, expert_tok_idx = torch.where(indices == expert_id)
            if batch_idx.numel() == 0:
                continue

            # Select the subset of tokens for the current expert.
            token_subset = x[batch_idx]  # Shape: [num_tokens_for_expert, dim]
            # Compute the output from the current expert.
            output_expert = self.experts[expert_id](token_subset)
            # Get the corresponding weights for these tokens.
            token_weights = weights[batch_idx, expert_tok_idx].unsqueeze(-1)
            # Apply weights and accumulate the output.
            expert_output[batch_idx] += output_expert * token_weights

        # If a shared expert exists, add its output.
        if self.shared_expert is not None:
            shared_output = self.shared_expert(x)
            output = expert_output + shared_output
        else:
            output = expert_output

        # Restore the original tensor shape and return.
        return output.view(original_shape)


class MoELayerOptimized(nn.Module):
    """
    Overview:
        An optimized implementation of the Mixture-of-Experts (MoE) layer that maintains the same API as `MoELayer`.
        This version avoids loops over experts by using a vectorized scatter-gather approach, which is significantly
        more efficient on modern hardware. The forward pass complexity is O(N_tokens + ΣE_i), where ΣE_i is the
        total number of tokens processed across all experts.

    The process is as follows:
        1. **Routing**: Get top-k experts and their weights for each token.
        2. **Flattening**: Create a flat list of (token_index, expert_index, weight) tuples.
        3. **Sorting**: Sort these tuples by expert_index. This groups all tokens destined for the same expert together.
        4. **Batch Forward**: Process the tokens for each expert in a single, contiguous batch, avoiding Python loops.
        5. **Weighted Scatter**: Apply gating weights to the expert outputs and scatter-add them back to a buffer
           indexed by the original token positions.
        6. **Shared Expert**: If configured, add the output from the shared expert.
        7. **Reshape**: Reshape the final output tensor to its original 3D shape.
    """

    def __init__(self, config: Any, experts: List[nn.Module], gate: nn.Module, num_experts_per_tok: int = 1) -> None:
        """
        Overview:
            Initializes the MoELayerOptimized.
        Arguments:
            - config (:obj:`Any`): A configuration object. Expected to have `embed_dim` and optionally `n_shared_experts`.
            - experts (:obj:`List[nn.Module]`): A list of PyTorch modules representing the experts.
            - gate (:obj:`nn.Module`): The gating module for routing tokens.
            - num_experts_per_tok (:obj:`int`): The number of experts to use for each token.
        """
        super().__init__()
        self.dim = config.embed_dim
        self.num_experts = len(experts)
        self.num_experts_per_tok = num_experts_per_tok
        self.gate = gate
        self.experts = nn.ModuleList(experts)

        self.use_shared = getattr(config, "n_shared_experts", 0) > 0
        if self.use_shared:
            # TODO: The architecture of the shared expert could be made more configurable.
            self.shared_expert = nn.Sequential(
                nn.Linear(self.dim, config.n_shared_experts * (4 * self.dim)),
                nn.GELU(),
                nn.Linear(config.n_shared_experts * (4 * self.dim), self.dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Overview:
            Performs the optimized forward pass for the MoE layer.
        Arguments:
            - x (:obj:`torch.Tensor`): The input tensor of shape [B, T, D].
        Returns:
            - torch.Tensor: The output tensor with the same shape as the input.
        """
        B, T, D = x.shape
        x_flat = x.reshape(-1, D)  # [N, D]; N = B*T

        # 1. Routing: Get top-k experts and weights.
        gate_logits = self.gate(x_flat)  # [N, E]
        weights, topk_idx = torch.topk(gate_logits, self.num_experts_per_tok, dim=1)  # [N, k]
        weights = F.softmax(weights, dim=1).to(x.dtype)  # [N, k]

        # 2. Flatten token-expert pairs.
        N, k = weights.shape
        flat_token_idx = torch.arange(N, device=x.device).repeat_interleave(k)  # [N*k]
        flat_expert_idx = topk_idx.reshape(-1)  # [N*k]
        flat_weight = weights.reshape(-1, 1)  # [N*k, 1]
        flat_input = x_flat[flat_token_idx]  # [N*k, D]

        # 3. Sort by expert index to group tokens for batch processing.
        sort_order = torch.argsort(flat_expert_idx)  # [N*k]
        flat_expert_idx = flat_expert_idx[sort_order]
        flat_token_idx = flat_token_idx[sort_order]
        flat_weight = flat_weight[sort_order]
        flat_input = flat_input[sort_order]

        # Count how many tokens each expert will process.
        counts = torch.bincount(flat_expert_idx, minlength=self.num_experts)  # [E]

        # Prepare output buffer.
        out_buffer = torch.zeros_like(flat_input)  # [N*k, D]

        # 4. Perform forward pass for each expert on its batch of tokens.
        ptr = 0
        for eid, num in enumerate(counts.tolist()):
            if num == 0:
                continue
            seg = slice(ptr, ptr + num)
            out_buffer[seg] = self.experts[eid](flat_input[seg])
            ptr += num

        # 5. Apply weights and scatter-add results back to token-indexed buffer.
        out_buffer.mul_(flat_weight)  # In-place multiplication by weights.
        token_output = torch.zeros_like(x_flat)  # [N, D]
        token_output.index_add_(0, flat_token_idx, out_buffer)

        # 6. Add shared expert output if it exists.
        if self.use_shared:
            token_output.add_(self.shared_expert(x_flat))

        return token_output.reshape(B, T, D)
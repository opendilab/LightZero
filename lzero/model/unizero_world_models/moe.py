import dataclasses
from typing import List, Optional

import torch
import torch.nn.functional as F
from simple_parsing.helpers import Serializable
from torch import nn

# Assume lzero.model.unizero_world_models.transformer._maybe_wrap_linear exists
# from lzero.model.unizero_world_models.transformer import _maybe_wrap_linear
def _maybe_wrap_linear(linear_layer: nn.Module, config: 'MoEConfig', name: str) -> nn.Module:
    """A placeholder for the actual _maybe_wrap_linear function."""
    # This function is assumed to wrap a linear layer, e.g., for applying LoRA.
    # The actual implementation is external to this snippet.
    return linear_layer


@dataclasses.dataclass
class MoEConfig(Serializable):
    """
    Overview:
        Configuration for the Mixture-of-Experts (MoE) model components.

    Arguments:
        - embed_dim (:obj:`int`): The embedding dimension for the input and output tensors.
        - num_experts (:obj:`int`): The total number of experts in the MoE layer.
        - num_experts_per_tok (:obj:`int`): The number of experts to route each token to (the 'k' in Top-k routing).
        - moe_use_lora (:obj:`bool`): Whether to wrap linear layers with LoRA wrappers. Defaults to False.
        - n_shared_experts (:obj:`int`): The number of shared experts to be applied to all tokens. Defaults to 0.
    """
    embed_dim: int
    num_experts: int
    num_experts_per_tok: int = 1
    moe_use_lora: bool = False
    n_shared_experts: int = 0


class MultiplicationFeedForward(nn.Module):
    """
    Overview:
        A feed-forward network layer implementing the SwiGLU variant.
        This architecture is defined as: FFN(x) = W_2(SiLU(W_1(x)) * W_3(x)).
        It is commonly used in modern transformer models.

    References:
        - https://github.com/mistralai/mistral-inference/blob/main/src/mistral_inference/transformer.py#L108
    """

    def __init__(self, config: MoEConfig):
        """
        Overview:
            Initializes the MultiplicationFeedForward layer.
        Arguments:
            - config (:obj:`MoEConfig`): The configuration object containing model dimensions and settings.
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
            Performs the forward pass of the SwiGLU-variant feed-forward network.
        Arguments:
            - x (:obj:`torch.Tensor`): The input tensor of shape [batch_size, seq_len, embed_dim].
        Returns:
            - (:obj:`torch.Tensor`): The output tensor of shape [batch_size, seq_len, embed_dim].
        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoELayer(nn.Module):
    """
    Overview:
        An efficient, vectorized implementation of a Mixture-of-Experts (MoE) layer.
        This layer routes each token to a subset of experts (Top-k routing) and combines their
        outputs. The implementation is designed to be highly efficient on parallel hardware
        by avoiding loops and using vectorized operations. An optional shared expert can
        be applied to all tokens.

    Algorithm:
        1.  **Routing**: A gating network computes logits for each expert. Top-k experts are selected for each token.
        2.  **Dispatch**: Token-expert assignments are flattened and sorted by expert ID. This groups all tokens
            destined for the same expert into contiguous blocks.
        3.  **Expert Computation**: Each expert processes its assigned batch of tokens in a single forward pass.
        4.  **Combine & Scatter**: The outputs from the experts are weighted by the gate probabilities and
            scattered back to their original token positions.
        5.  **Shared Expert**: If configured, a shared expert's output is added to the result.

    References:
        - https://github.com/mistralai/mistral-inference/blob/main/src/mistral_inference/moe.py
    """

    def __init__(self, config: MoEConfig, experts: List[nn.Module], gate: nn.Module):
        """
        Overview:
            Initializes the MoE layer.
        Arguments:
            - config (:obj:`MoEConfig`): The configuration object for the MoE layer.
            - experts (:obj:`List[nn.Module]`): A list of expert neural network modules.
            - gate (:obj:`nn.Module`): The gating network that computes routing logits.
        """
        super().__init__()
        self.dim = config.embed_dim
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok

        self.gate = gate
        self.experts = nn.ModuleList(experts)

        self.shared_expert: Optional[nn.Module] = None
        if config.n_shared_experts > 0:
            # Create a shared expert FFN if configured
            self.shared_expert = nn.Sequential(
                nn.Linear(self.dim, config.n_shared_experts * (4 * self.dim)),
                nn.GELU(),
                nn.Linear(config.n_shared_experts * (4 * self.dim), self.dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Overview:
            Performs the forward pass of the MoE layer.
        Arguments:
            - x (:obj:`torch.Tensor`): Input tensor of shape `[batch_size, seq_len, embed_dim]`.
        Returns:
            - (:obj:`torch.Tensor`): Output tensor of the same shape as the input.
        """
        batch_size, seq_len, dim = x.shape
        x_flat = x.view(-1, dim)  # Shape: [N, D], where N = B * T

        # 1. --- Routing ---
        # Compute routing logits and select top-k experts for each token.
        gate_logits = self.gate(x_flat)  # Shape: [N, E]
        weights, topk_indices = torch.topk(gate_logits, self.num_experts_per_tok, dim=1)  # Shape: [N, k]
        weights = F.softmax(weights, dim=1, dtype=torch.float).to(x.dtype)  # Shape: [N, k]

        # 2. --- Flatten token-expert assignments ---
        # Create a flat list of (token_index, expert_index) pairs for efficient processing.
        num_tokens, k = weights.shape
        flat_token_indices = torch.arange(num_tokens, device=x.device).repeat_interleave(k)  # Shape: [N*k]
        flat_expert_indices = topk_indices.reshape(-1)  # Shape: [N*k]
        flat_weights = weights.reshape(-1, 1)  # Shape: [N*k, 1]
        flat_inputs = x_flat[flat_token_indices]  # Shape: [N*k, D]

        # 3. --- Dispatch tokens to experts by sorting ---
        # Sort by expert index to group tokens for the same expert together.
        sort_order = torch.argsort(flat_expert_indices)
        sorted_expert_indices = flat_expert_indices[sort_order]
        sorted_token_indices = flat_token_indices[sort_order]
        sorted_weights = flat_weights[sort_order]
        sorted_inputs = flat_inputs[sort_order]

        # 4. --- Batched expert computation ---
        # Process tokens for each expert in a single batch.
        expert_counts = torch.bincount(sorted_expert_indices, minlength=self.num_experts)  # Shape: [E]
        output_buffer = torch.zeros_like(sorted_inputs)  # Shape: [N*k, D]

        ptr = 0
        for expert_id, count in enumerate(expert_counts.tolist()):
            if count == 0:
                continue
            
            # Select the slice of tokens for the current expert.
            segment = slice(ptr, ptr + count)
            # Run the expert on its batch of tokens.
            output_buffer[segment] = self.experts[expert_id](sorted_inputs[segment])
            ptr += count

        # 5. --- Combine outputs and scatter back ---
        # Weight the outputs and add them back to the original token positions.
        output_buffer.mul_(sorted_weights)  # In-place weighting
        
        token_output = torch.zeros_like(x_flat)  # Shape: [N, D]
        token_output.index_add_(0, sorted_token_indices, output_buffer)

        # 6. --- Add shared expert output (if any) ---
        if self.shared_expert is not None:
            token_output.add_(self.shared_expert(x_flat))

        return token_output.view(batch_size, seq_len, dim)
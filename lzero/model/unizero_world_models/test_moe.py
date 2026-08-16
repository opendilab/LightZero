"""
test_moe.py

Overview:
    A test script to verify the functional equivalence between a standard Transformer's feed-forward network (FFN)
    and a Mixture-of-Experts (MoE) layer configured with a single expert. This script demonstrates that
    the MoE layer correctly specializes to a standard FFN when num_experts is 1, ensuring backward
    compatibility and correct routing logic.
"""
import dataclasses
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclasses.dataclass
class TransformerConfig:
    """
    Overview:
        Configuration for the Transformer block and its potential MoE layer.

    Arguments:
        - embed_dim (int): The embedding dimension for the model.
        - resid_pdrop (float): The dropout probability for the residual connections.
        - moe_in_transformer (bool): If True, use an MoE layer for the feed-forward part. Otherwise, use a standard MLP.
        - num_experts (int): The total number of experts in the MoE layer.
        - num_experts_per_tok (int): The number of experts to route each token to (top-k routing).
    """
    embed_dim: int = 64
    resid_pdrop: float = 0.1
    moe_in_transformer: bool = False
    num_experts: int = 1
    num_experts_per_tok: int = 1


class MoELayer(nn.Module):
    """
    Overview:
        An efficient, vectorized implementation of a Mixture-of-Experts (MoE) layer.
        This layer routes each token to a subset of experts (Top-k routing) and combines their
        outputs using a weighted sum. The implementation is highly optimized for parallel
        computation on hardware like GPUs.
    """

    def __init__(self, experts: List[nn.Module], gate: nn.Module, num_experts_per_tok: int):
        """
        Overview:
            Initializes the MoE layer.
        Arguments:
            - experts (List[nn.Module]): A list of expert neural network modules.
            - gate (nn.Module): The gating network that computes routing logits.
            - num_experts_per_tok (int): The number of experts to route each token to.
        """
        super().__init__()
        assert len(experts) > 0, "The list of experts cannot be empty."
        self.experts = nn.ModuleList(experts)
        self.gate = gate
        self.num_experts = len(experts)
        self.num_experts_per_tok = num_experts_per_tok

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Overview:
            Performs the forward pass of the MoE layer.
        Arguments:
            - x (torch.Tensor): Input tensor of shape `[batch_size, seq_len, embed_dim]`.
        Returns:
            - (torch.Tensor): Output tensor of the same shape as the input.
        """
        batch_size, seq_len, dim = x.shape
        x_flat = x.view(-1, dim)

        gate_logits = self.gate(x_flat)
        weights, topk_indices = torch.topk(gate_logits, self.num_experts_per_tok, dim=1)
        weights = F.softmax(weights, dim=1, dtype=torch.float).to(x.dtype)

        num_tokens = x_flat.shape[0]
        flat_token_indices = torch.arange(num_tokens, device=x.device).repeat_interleave(self.num_experts_per_tok)
        flat_expert_indices = topk_indices.view(-1)
        
        sort_order = torch.argsort(flat_expert_indices)
        sorted_expert_indices = flat_expert_indices[sort_order]
        sorted_token_indices = flat_token_indices[sort_order]
        
        expert_inputs = x_flat[sorted_token_indices]
        sorted_weights = weights.view(-1, 1)[sort_order]
        
        expert_counts = torch.bincount(sorted_expert_indices, minlength=self.num_experts)
        output_buffer = torch.zeros_like(expert_inputs)
        
        ptr = 0
        for i, count in enumerate(expert_counts.tolist()):
            if count == 0:
                continue
            segment = slice(ptr, ptr + count)
            output_buffer[segment] = self.experts[i](expert_inputs[segment])
            ptr += count
        
        # --- FIX: Simplified and corrected scattering logic ---
        # Weight the outputs and directly add them to the correct token's position.
        weighted_outputs = output_buffer * sorted_weights
        
        token_output = torch.zeros_like(x_flat)
        # Use `sorted_token_indices` to add the results back to their original token positions.
        token_output.index_add_(0, sorted_token_indices, weighted_outputs)

        return token_output.view(batch_size, seq_len, dim)


class TransformerBlock(nn.Module):
    """
    Overview:
        A simplified Transformer block that contains a feed-forward network (FFN).
        The FFN can be either a standard MLP or a Mixture-of-Experts (MoE) layer,
        controlled by the configuration.
    """
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(config.embed_dim, 4 * config.embed_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(4 * config.embed_dim, config.embed_dim),
            nn.Dropout(config.resid_pdrop),
        )
        
        if config.moe_in_transformer:
            experts = [self.mlp for _ in range(config.num_experts)]
            self.feed_forward = MoELayer(
                experts=experts,
                gate=nn.Linear(config.embed_dim, config.num_experts, bias=False),
                num_experts_per_tok=config.num_experts_per_tok,
            )
            print("=" * 40)
            print("TransformerBlock initialized with MoE layer.")
            print("=" * 40)
        else:
            self.feed_forward = self.mlp
            print("-" * 40)
            print("TransformerBlock initialized with standard MLP.")
            print("-" * 40)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.feed_forward(x)


def test_transformer_block_equivalence():
    """
    Overview:
        Tests that an MoE layer with a single expert produces an output identical
        to that of a standard MLP layer, given that they share the same weights.
    """
    torch.manual_seed(42)
    
    embed_dim = 64
    batch_size = 10
    seq_len = 5
    
    config_mlp = TransformerConfig(embed_dim=embed_dim, moe_in_transformer=False)
    config_moe = TransformerConfig(embed_dim=embed_dim, moe_in_transformer=True, num_experts=1, num_experts_per_tok=1)

    # --- FIX: Ensure identical weights for a fair comparison ---
    # 1. Create the standard MLP block first.
    transformer_block_mlp = TransformerBlock(config_mlp)

    # 2. Create the MoE block.
    transformer_block_moe = TransformerBlock(config_moe)

    # 3. CRITICAL: Load the MLP's weights into the MoE's expert MLP.
    # This guarantees that the underlying expert has the exact same weights as the standalone MLP.
    transformer_block_moe.mlp.load_state_dict(transformer_block_mlp.mlp.state_dict())
    
    # Also, for a perfect match, the gate should be initialized to a state
    # that it doesn't affect the output scaling. We can manually set its weights.
    # In a single-expert case, softmax ensures the weight is 1, so this is not strictly
    # necessary, but it's good practice for more complex tests.
    
    inputs = torch.randn(batch_size, seq_len, embed_dim)
    
    print("\nRunning forward pass for standard MLP block...")
    output_mlp = transformer_block_mlp(inputs)
    
    print("\nRunning forward pass for MoE block...")
    output_moe = transformer_block_moe(inputs)

    is_close = torch.allclose(output_moe, output_mlp, atol=1e-6)
    mse_difference = F.mse_loss(output_moe, output_mlp).item()
    
    print("\n" + "=" * 25 + " TEST RESULTS " + "=" * 25)
    print(f"Outputs are close: {is_close}")
    print(f"Mean Squared Error (MSE) between outputs: {mse_difference:.10f}")
    
    assert is_close, "Test failed: Outputs of single-expert MoE and MLP are not identical."
    print("\n✅ Test Passed: Single-expert MoE layer behaves identically to a standard MLP.")
    print("=" * 64 + "\n")


if __name__ == "__main__":
    test_transformer_block_equivalence()
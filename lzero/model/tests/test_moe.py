"""
test_moe.py

Overview:
    A pytest test suite to verify the functional equivalence between a standard Transformer's feed-forward network (FFN)
    and a Mixture-of-Experts (MoE) layer configured with a single expert. This test demonstrates that
    the MoE layer correctly specializes to a standard FFN when num_experts is 1, ensuring backward
    compatibility and correct routing logic.
"""
import dataclasses
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from lzero.model.unizero_world_models.moe import MoELayer


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
        - moe_use_lora (bool): Whether to use LoRA in the MoE layer.
        - n_shared_experts (int): Number of shared experts (optional).
    """
    embed_dim: int = 64
    resid_pdrop: float = 0.1
    moe_in_transformer: bool = False
    num_experts: int = 1
    num_experts_per_tok: int = 1
    moe_use_lora: bool = False
    n_shared_experts: int = 0


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
            # Create experts - in the single expert case, we share the same MLP
            experts = [self.mlp for _ in range(config.num_experts)]
            gate = nn.Linear(config.embed_dim, config.num_experts, bias=False)

            # Use MoELayer from moe.py (note the different signature)
            self.feed_forward = MoELayer(
                config=config,
                experts=experts,
                gate=gate,
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


class TestMoELayer:
    """Test suite for MoE layer functionality."""

    @pytest.fixture
    def embed_dim(self):
        """Embedding dimension for tests."""
        return 64

    @pytest.fixture
    def batch_size(self):
        """Batch size for tests."""
        return 10

    @pytest.fixture
    def seq_len(self):
        """Sequence length for tests."""
        return 5

    def test_single_expert_moe_equivalence(self, embed_dim, batch_size, seq_len):
        """
        Test that an MoE layer with a single expert produces an output identical
        to that of a standard MLP layer, given that they share the same weights.
        """
        torch.manual_seed(42)

        config_mlp = TransformerConfig(embed_dim=embed_dim, moe_in_transformer=False)
        config_moe = TransformerConfig(
            embed_dim=embed_dim,
            moe_in_transformer=True,
            num_experts=1,
            num_experts_per_tok=1,
            moe_use_lora=False,
            n_shared_experts=0
        )

        # 1. Create the standard MLP block first.
        transformer_block_mlp = TransformerBlock(config_mlp)

        # 2. Create the MoE block.
        transformer_block_moe = TransformerBlock(config_moe)

        # 3. CRITICAL: Load the MLP's weights into the MoE's expert MLP.
        # This guarantees that the underlying expert has the exact same weights as the standalone MLP.
        transformer_block_moe.mlp.load_state_dict(transformer_block_mlp.mlp.state_dict())

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

        assert is_close, f"Test failed: Outputs of single-expert MoE and MLP are not identical. MSE: {mse_difference}"
        print("\n✅ Test Passed: Single-expert MoE layer behaves identically to a standard MLP.")
        print("=" * 64 + "\n")

    def test_moe_output_shape(self, embed_dim, batch_size, seq_len):
        """
        Test that MoE layer preserves the input shape.
        """
        torch.manual_seed(42)

        config_moe = TransformerConfig(
            embed_dim=embed_dim,
            moe_in_transformer=True,
            num_experts=4,
            num_experts_per_tok=2,
            moe_use_lora=False,
            n_shared_experts=0
        )

        transformer_block_moe = TransformerBlock(config_moe)
        inputs = torch.randn(batch_size, seq_len, embed_dim)

        output = transformer_block_moe(inputs)

        assert output.shape == inputs.shape, \
            f"Expected output shape {inputs.shape}, but got {output.shape}"
        print(f"✅ Test Passed: MoE layer preserves input shape: {inputs.shape}")

    def test_moe_with_multiple_experts(self, embed_dim, batch_size, seq_len):
        """
        Test that MoE layer works correctly with multiple experts.
        """
        torch.manual_seed(42)

        num_experts = 8
        num_experts_per_tok = 2

        config_moe = TransformerConfig(
            embed_dim=embed_dim,
            moe_in_transformer=True,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            moe_use_lora=False,
            n_shared_experts=0
        )

        transformer_block_moe = TransformerBlock(config_moe)
        inputs = torch.randn(batch_size, seq_len, embed_dim)

        output = transformer_block_moe(inputs)

        assert output.shape == inputs.shape, \
            f"Expected output shape {inputs.shape}, but got {output.shape}"
        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert not torch.isinf(output).any(), "Output contains Inf values"

        print(f"✅ Test Passed: MoE layer with {num_experts} experts and top-{num_experts_per_tok} routing works correctly")


if __name__ == "__main__":
    if PYTEST_AVAILABLE:
        pytest.main([__file__, "-v", "-s"])
    else:
        # Run tests directly without pytest
        print("Pytest not available. Running tests directly...\n")
        test_suite = TestMoELayer()

        # Set up fixtures
        embed_dim = 64
        batch_size = 10
        seq_len = 5

        print("\n" + "=" * 60)
        print("Test 1: Single Expert MoE Equivalence")
        print("=" * 60)
        test_suite.test_single_expert_moe_equivalence(embed_dim, batch_size, seq_len)

        print("\n" + "=" * 60)
        print("Test 2: MoE Output Shape")
        print("=" * 60)
        test_suite.test_moe_output_shape(embed_dim, batch_size, seq_len)

        print("\n" + "=" * 60)
        print("Test 3: MoE with Multiple Experts")
        print("=" * 60)
        test_suite.test_moe_with_multiple_experts(embed_dim, batch_size, seq_len)

        print("\n" + "=" * 60)
        print("All tests passed! ✅")
        print("=" * 60)

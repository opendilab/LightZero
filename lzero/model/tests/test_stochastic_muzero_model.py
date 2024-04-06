import torch
import pytest
from torch import nn
from lzero.model.stochastic_muzero_model import ChanceEncoder

# Initialize a ChanceEncoder instance for testing
@pytest.fixture
def encoder():
    return ChanceEncoder((3, 32, 32), 4)

def test_ChanceEncoder(encoder):
    # Create a dummy tensor for testing
    x_and_last_x = torch.randn(1, 6, 32, 32)

    # Forward pass through the encoder
    chance_encoding_t, chance_onehot_t = encoder(x_and_last_x)

    # Check the output shapes
    assert chance_encoding_t.shape == (1, 4)
    assert chance_onehot_t.shape == (1, 4)

    # Check that chance_onehot_t is indeed one-hot
    assert torch.all((chance_onehot_t == 0) | (chance_onehot_t == 1))
    assert torch.all(torch.sum(chance_onehot_t, dim=1) == 1)
    
def test_ChanceEncoder_gradients_chance_encoding(encoder):
    # Create a dummy tensor for testing
    x_and_last_x = torch.randn(1, 6, 32, 32)

    # Forward pass through the encoder
    chance_encoding_t, chance_onehot_t = encoder(x_and_last_x)

    # Create a dummy target tensor for a simple loss function
    target = torch.randn(1, 4)

    # Use mean squared error as a simple loss function
    loss = nn.MSELoss()(chance_encoding_t, target)

    # Backward pass
    loss.backward()

    # Check if gradients are computed
    for param in encoder.parameters():
        assert param.grad is not None

    # Check if gradients have the correct shape
    for param in encoder.parameters():
        assert param.grad.shape == param.shape

def test_ChanceEncoder_gradients_chance_onehot_t(encoder):
    # Create a dummy tensor for testing
    x_and_last_x = torch.randn(1, 6, 32, 32)

    # Forward pass through the encoder
    chance_encoding_t, chance_onehot_t = encoder(x_and_last_x)

    # Create a dummy target tensor for a simple loss function
    target = torch.randn(1, 4)

    # Use mean squared error as a simple loss function
    loss = nn.MSELoss()(chance_onehot_t, target)

    # Backward pass
    loss.backward()

    # Check if gradients are computed
    for param in encoder.parameters():
        assert param.grad is not None

    # Check if gradients have the correct shape
    for param in encoder.parameters():
        assert param.grad.shape == param.shape

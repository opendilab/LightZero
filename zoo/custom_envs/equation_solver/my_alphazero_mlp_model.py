"""
A custom MLP-based AlphaZero model for vector inputs.
This model is designed for tasks where the state is a simple vector (e.g., [3, -2, 1, 10, ...]).
We support two observation shape formats:
  - A flat vector, e.g. (41,)
  - A pseudo-image, e.g. (1, 41, 1)
In either case, the input is flattened and passed through an MLP to produce both policy logits and a value.
"""

from typing import Tuple, Optional, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
from ding.torch_utils import MLP  # DI-engine's MLP utility
from lzero.model.common import MLP_V2
from ding.utils import MODEL_REGISTRY, SequenceType

@MODEL_REGISTRY.register('AlphaZeroMLPModel')
class AlphaZeroMLPModel(nn.Module):
    def __init__(
        self,
        observation_shape: SequenceType = (41,), 
        action_space_size: int = 50,
        categorical_distribution: bool = False,
        activation: Optional[nn.Module] = nn.ReLU(inplace=True),
        hidden_sizes: SequenceType = [128, 128],
        last_linear_layer_init_zero: bool = True,
    ):
        """
        Args:
            observation_shape: Expected shape of observations.
                Can be a flat vector, e.g. (41,), or a pseudo-image, e.g. (1, 41, 1).
            action_space_size: Number of discrete actions.
            categorical_distribution: If True, use a categorical representation for value.
            activation: Activation function.
            hidden_sizes: List of hidden layer sizes for the shared MLP.
            last_linear_layer_init_zero: Whether to initialize the last layer of the heads with zeros.
        """
        super(AlphaZeroMLPModel, self).__init__()
        self.categorical_distribution = categorical_distribution
        self.observation_shape = observation_shape
        self.value_support_size = 601 if self.categorical_distribution else 1
        self.last_linear_layer_init_zero = last_linear_layer_init_zero

        # Determine the input dimension based on the observation shape.
        if len(observation_shape) == 1:
            self.input_dim = observation_shape[0]
        elif len(observation_shape) == 3:
            self.input_dim = observation_shape[0] * observation_shape[1] * observation_shape[2]
        else:
            raise ValueError(f"Unsupported observation_shape: {observation_shape}")

        self.action_space_size = action_space_size

        # Shared representation network: a simple MLP.
        self.representation_network = MLP_V2(
            in_channels=self.input_dim,
            hidden_channels=hidden_sizes,
            out_channels=hidden_sizes[-1],
            activation=activation,
            norm_type='LN',
            output_activation=False,
            output_norm=True,
        )

        # Policy head: maps shared representation to action logits.
        self.policy_head = MLP_V2(
            in_channels=hidden_sizes[-1],
            hidden_channels=hidden_sizes,
            out_channels=action_space_size,
            activation=activation,
            norm_type='LN',
            output_activation=False,
            output_norm=False,
            last_linear_layer_init_zero=last_linear_layer_init_zero,
        )

        # Value head: maps shared representation to a scalar value.
        self.value_head = MLP_V2(
            in_channels=hidden_sizes[-1],
            hidden_channels=hidden_sizes,
            out_channels=self.value_support_size,
            activation=activation,
            norm_type='LN',
            output_activation=False,
            output_norm=False,
            last_linear_layer_init_zero=last_linear_layer_init_zero,
        )

    def forward(self, state_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            state_batch: Tensor of shape (B, C, H, W) or (B, L) if flat.
        Returns:
            Tuple (policy_logits, value) where:
              - policy_logits: (B, action_space_size)
              - value: (B, 1) (if not using categorical distribution)
        """
        B = state_batch.size(0)
        # If the state is 4D, flatten; if it's 2D, it's already flat.
        if state_batch.dim() == 4:
            x = state_batch.view(B, -1)
        else:
            x = state_batch
        rep = self.representation_network(x)
        policy_logits = self.policy_head(rep)
        value = self.value_head(rep)
        if not self.categorical_distribution:
            value = value.unsqueeze(-1)
        return policy_logits, value

    def compute_policy_value(self, state_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the softmax probabilities and state value.
        """
        logits, value = self.forward(state_batch)
        prob = F.softmax(logits, dim=-1)
        return prob, value

    def compute_logp_value(self, state_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the log-softmax probabilities and state value.
        """
        logits, value = self.forward(state_batch)
        log_prob = F.log_softmax(logits, dim=-1)
        return log_prob, value

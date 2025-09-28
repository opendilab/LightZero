"""
Modified from https://github.com/CompVis/taming-transformers
This module provides an autoencoder-style tokenizer for encoding observations into latent embeddings and decoding them back.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F


class LossWithIntermediateLosses:
    """
    Overview:
        A helper class to manage a total loss value alongside a dictionary of its constituent, named loss components.
        This is primarily used for detailed logging.
    """

    def __init__(self, **kwargs: torch.Tensor) -> None:
        """
        Overview:
            Initializes the loss object.
        Arguments:
            - kwargs (:obj:`torch.Tensor`): Keyword arguments where keys are loss names and values are the corresponding loss tensors.
        """
        # The total loss, which can be used for backpropagation.
        self.loss_total: torch.Tensor = sum(kwargs.values())
        # A dictionary holding the scalar values of intermediate losses, detached from the computation graph.
        self.intermediate_losses: Dict[str, float] = {k: v.item() for k, v in kwargs.items()}

    def __truediv__(self, value: float) -> "LossWithIntermediateLosses":
        """
        Overview:
            Overloads the division operator to scale all loss components by a scalar value.
            This is useful for operations like averaging over batch size or gradient accumulation steps.
        Arguments:
            - value (:obj:`float`): The scalar value to divide the losses by.
        Returns:
            - LossWithIntermediateLosses: The same instance with updated loss values.
        """
        if not isinstance(value, (int, float)) or value == 0:
            raise ValueError(f"Division is only supported for a non-zero scalar, but got {value}.")

        self.loss_total = self.loss_total / value
        for k in self.intermediate_losses:
            self.intermediate_losses[k] /= value
        return self


@dataclass
class TokenizerEncoderOutput:
    """
    Overview:
        A data structure to hold the various outputs from a VQ-VAE style encoder,
        including continuous and quantized latent representations, and discrete tokens.
    """
    # Continuous latent representation from the encoder.
    z: torch.FloatTensor
    # Quantized latent representation.
    z_quantized: torch.FloatTensor
    # Discrete integer tokens corresponding to the codebook entries.
    tokens: torch.LongTensor


class Tokenizer(nn.Module):
    """
    Overview:
        An autoencoder model that encodes high-dimensional observations (like images or state vectors)
        into low-dimensional latent embeddings and decodes them back. It can also compute reconstruction
        and perceptual losses. This implementation does not include the quantization step (Vector Quantization)
        but serves as the encoder-decoder backbone.
    """

    def __init__(
            self,
            encoder: nn.Module,
            decoder: nn.Module,
            with_lpips: bool = False,
            obs_type: str = 'image'
    ) -> None:
        """
        Overview:
            Initializes the Tokenizer (Autoencoder).
        Arguments:
            - encoder (:obj:`nn.Module`): The network responsible for encoding observations into latent embeddings. It can be a single module or an nn.ModuleList for multi-task scenarios.
            - decoder (:obj:`nn.Module`): The network responsible for decoding latent embeddings back into observations.
            - with_lpips (:obj:`bool`): If True, initializes the LPIPS model to compute perceptual loss. Defaults to False.
            - obs_type (:obj:`str`): The type of observation, e.g., 'image' or 'vector'. This can inform model architecture choices. Defaults to 'image'.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder_network = decoder
        self.obs_type = obs_type
        self.lpips: Optional[nn.Module] = None
        if with_lpips:
            # Lazily import LPIPS as it's an optional dependency.
            from lzero.model.unizero_world_models.lpips import LPIPS
            self.lpips = LPIPS().eval()

    def encode_to_obs_embeddings(self, x: torch.Tensor, task_id: int = 0) -> torch.Tensor:
        """
        Overview:
            Encodes a batch of observations into latent embeddings, handling various input shapes and multi-task encoders.
        Arguments:
            - x (:obj:`torch.Tensor`): The input tensor of observations. Shape can be (B, E), (B, T, E), (B, C, H, W), or (B, T, C, H, W).
            - task_id (:obj:`int`): The identifier for the task, used to select the correct encoder from an nn.ModuleList in multi-task settings. Defaults to 0.
        Returns:
            - torch.Tensor: The encoded latent embeddings with a consistent shape of (B, 1, E), where B is the effective batch size.
        """
        # Step 1: Select the appropriate encoder module.
        # This handles both single-task (a single nn.Module) and multi-task (an nn.ModuleList) scenarios.
        if isinstance(self.encoder, nn.ModuleList):
            if not 0 <= task_id < len(self.encoder):
                raise ValueError(
                    f"Provided task_id {task_id} is invalid for the encoder list of size {len(self.encoder)}."
                )
            encoder_module = self.encoder[task_id]
        else:
            encoder_module = self.encoder

        # Step 2: Pre-process and reshape the input tensor based on its dimensions.
        # The goal is to transform the input into a 2D or 4D tensor that the encoder can process.
        original_shape = x.shape
        if len(original_shape) == 5:  # Batch of sequences of images: (B, T, C, H, W)
            # Flatten the batch and time dimensions to create a batch of images.
            x = x.contiguous().view(-1, *original_shape[-3:])  # Shape: (B*T, C, H, W)
        elif len(original_shape) == 3:  # Batch of sequences of vectors: (B, T, E)
            # Flatten the batch and time dimensions to create a batch of vectors.
            x = x.contiguous().view(-1, original_shape[-1])  # Shape: (B*T, E)
        # Note: 2D (B, E) and 4D (B, C, H, W) inputs are processed directly without reshaping.

        # Step 3: Pass the processed tensor through the encoder.
        obs_embeddings = encoder_module(x)
        if len(obs_embeddings.shape) != 2:
            raise RuntimeError(
                f"Encoder output was expected to be 2D (batch, embedding_dim), but got shape {obs_embeddings.shape}."
            )

        # Step 4: Reshape the output to a consistent sequence format (B', 1, E).
        # The '1' represents a sequence length of one, making it compatible with sequence models.
        obs_embeddings = rearrange(obs_embeddings, 'b e -> b 1 e')

        return obs_embeddings

    def decode_to_obs(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Overview:
            Decodes a batch of latent embeddings back into the observation space.
        Arguments:
            - embeddings (:obj:`torch.Tensor`): The latent embeddings to decode.
        Returns:
            - torch.Tensor: The reconstructed observations.
        """
        return self.decoder_network(embeddings)

    @staticmethod
    def reconstruction_loss(original_images: torch.Tensor, reconstructed_images: torch.Tensor) -> torch.Tensor:
        """
        Overview:
            Calculates the reconstruction loss between original and reconstructed observations.
            It uses L2 (MSE) loss for vector-based observations and L1 (MAE) loss for image-based observations.
        Arguments:
            - original_images (:obj:`torch.Tensor`): The ground-truth observations.
            - reconstructed_images (:obj:`torch.Tensor`): The observations reconstructed by the decoder.
        Returns:
            - torch.Tensor: A scalar tensor representing the computed reconstruction loss.
        """
        if len(original_images.shape) == 2:
            # Use Mean Squared Error (L2 loss) for vector-based observations.
            return F.mse_loss(reconstructed_images, original_images)
        else:
            # Use Mean Absolute Error (L1 loss) for image-based observations, which is often more robust to outliers.
            return torch.abs(original_images - reconstructed_images).mean()

    def perceptual_loss(self, original_images: torch.Tensor, reconstructed_images: torch.Tensor) -> torch.Tensor:
        """
        Overview:
            Calculates the perceptual loss (LPIPS) between original and reconstructed images.
            This loss is designed to better align with human perception of image similarity.
        Arguments:
            - original_images (:obj:`torch.Tensor`): The ground-truth images.
            - reconstructed_images (:obj:`torch.Tensor`): The images reconstructed by the decoder.
        Returns:
            - torch.Tensor: A scalar tensor representing the computed perceptual loss.
        """
        if self.lpips is None:
            raise RuntimeError("LPIPS model was not initialized. Please set `with_lpips=True` during Tokenizer instantiation.")
        return torch.mean(self.lpips(original_images, reconstructed_images))

    def __repr__(self) -> str:
        """
        Overview:
            Provides a string representation of the Tokenizer module.
        """
        return f"Tokenizer(obs_type='{self.obs_type}', with_lpips={self.lpips is not None})"
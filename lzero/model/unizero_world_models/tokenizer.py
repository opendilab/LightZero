"""
Modified from https://github.com/CompVis/taming-transformers
"""

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F

from lzero.model.unizero_world_models.lpips import LPIPS


class LossWithIntermediateLosses:
    def __init__(self, **kwargs):
        """Initialize with various loss components."""
        self.loss_total = sum(kwargs.values())
        self.intermediate_losses = {k: v.item() for k, v in kwargs.items()}

    def __truediv__(self, value):
        """Divide all loss components by a given value."""
        for k, v in self.intermediate_losses.items():
            self.intermediate_losses[k] = v / value
        self.loss_total = self.loss_total / value
        return self


@dataclass
class TokenizerEncoderOutput:
    z: torch.FloatTensor
    z_quantized: torch.FloatTensor
    tokens: torch.LongTensor


class Tokenizer(nn.Module):
    """
    Overview:
        Tokenizer model that encodes and decodes observations.
    """
    def __init__(self, encoder=None, decoder_network=None, with_lpips: bool = False) -> None:
        """Initialize the Tokenizer.

        Arguments:
            encoder (nn.Module, optional): Encoder network. Defaults to None.
            decoder_network (nn.Module, optional): Decoder network. Defaults to None.
            with_lpips (bool, optional): Whether to use LPIPS for perceptual loss. Defaults to False.
        """
        super().__init__()
        self.lpips = LPIPS().eval() if with_lpips else None
        self.encoder = encoder
        self.decoder_network = decoder_network


    def encode_to_obs_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Encode observations to embeddings.

        Arguments:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Encoded embeddings.
        """
        shape = x.shape
        if len(shape) == 2:
            obs_embeddings = self.encoder(x)
            obs_embeddings = rearrange(obs_embeddings, 'b e -> b 1 (e)')
        elif len(shape) == 3:
            x = x.contiguous().view(-1, shape[-1])
            obs_embeddings = self.encoder(x)
            obs_embeddings = rearrange(obs_embeddings, 'b e -> b 1 (e)')
        elif len(shape) == 4:
            obs_embeddings = self.encoder(x)
            obs_embeddings = rearrange(obs_embeddings, 'b e -> b 1 e')
        elif len(shape) == 5:
            x = x.contiguous().view(-1, *shape[-3:])
            obs_embeddings = self.encoder(x)
            obs_embeddings = rearrange(obs_embeddings, 'b e -> b 1 e')
        return obs_embeddings

    def decode_to_obs(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Decode embeddings to observations.

        Arguments:
            embeddings (torch.Tensor): Input embeddings.

        Returns:
            torch.Tensor: Decoded observations.
        """
        return self.decoder_network(embeddings)

    # the following methods are used for the discrete latent state case, not used in the current implementation of UniZero
    def compute_loss(self, batch, **kwargs: Any) -> LossWithIntermediateLosses:
        """Compute the loss for a given batch.

        Arguments:
            batch (dict): Batch of data.

        Returns:
            LossWithIntermediateLosses: Computed losses.
        """
        if len(batch['observations'][0, 0].shape) == 1:
            # Observations are 1-dimensional vectors.
            original_shape = list(batch['observations'].shape)
            desired_shape = original_shape + [64, 64]
            expanded_observations = batch['observations'].unsqueeze(-1).unsqueeze(-1).expand(*desired_shape)
            batch['observations'] = expanded_observations

        assert self.lpips is not None
        observations = self.preprocess_input(rearrange(batch['observations'], 'b t c h w -> (b t) c h w'))
        z, z_quantized, reconstructions = self(observations, should_preprocess=False, should_postprocess=False)

        # Codebook loss
        beta = 1.0
        commitment_loss = (z.detach() - z_quantized).pow(2).mean() + beta * (z - z_quantized.detach()).pow(2).mean()
        # L1 loss
        reconstruction_loss = torch.abs(observations - reconstructions).mean()
        # Perceptual loss using LPIPS
        perceptual_loss = torch.mean(self.lpips(observations, reconstructions))

        return LossWithIntermediateLosses(
            commitment_loss=commitment_loss,
            reconstruction_loss=reconstruction_loss,
            perceptual_loss=perceptual_loss
        )

    def reconstruction_loss(self, original_images: torch.Tensor, reconstructed_images: torch.Tensor) -> torch.Tensor:
        """Calculate the reconstruction loss.

        Arguments:
            original_images (torch.Tensor): Original images.
            reconstructed_images (torch.Tensor): Reconstructed images.

        Returns:
            torch.Tensor: Computed reconstruction loss.
        """
        if len(original_images.shape) == 2:
            # For memory environment vector observations
            loss = F.mse_loss(original_images, reconstructed_images)  # L2 loss
        else:
            # For Atari image environment
            loss = torch.abs(original_images - reconstructed_images).mean()  # L1 loss
        return loss

    def perceptual_loss(self, original_images: torch.Tensor, reconstructed_images: torch.Tensor) -> torch.Tensor:
        """Calculate the perceptual loss using LPIPS.

        Arguments:
            original_images (torch.Tensor): Original images.
            reconstructed_images (torch.Tensor): Reconstructed images.

        Returns:
            torch.Tensor: Computed perceptual loss.
        """
        return torch.mean(self.lpips(original_images, reconstructed_images))

    @torch.no_grad()
    def encode_decode(self, x: torch.Tensor, should_preprocess: bool = False, should_postprocess: bool = False) -> torch.Tensor:
        """Encode and decode a tensor.

        Arguments:
            x (torch.Tensor): Input tensor.
            should_preprocess (bool, optional): Whether to preprocess the input. Defaults to False.
            should_postprocess (bool, optional): Whether to postprocess the output. Defaults to False.

        Returns:
            torch.Tensor: Decoded tensor.
        """
        z_q = self.encode(x, should_preprocess).z_quantized
        return self.decode(z_q, should_postprocess)

    def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        """Preprocess input tensor from [0, 1] to [-1, 1].

        Arguments:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Preprocessed tensor.
        """
        return x.mul(2).sub(1)

    def postprocess_output(self, y: torch.Tensor) -> torch.Tensor:
        """Postprocess output tensor from [-1, 1] to [0, 1].

        Arguments:
            y (torch.Tensor): Output tensor.

        Returns:
            torch.Tensor: Postprocessed tensor.
        """
        return y.add(1).div(2)

    def __repr__(self) -> str:
        return "Tokenizer"
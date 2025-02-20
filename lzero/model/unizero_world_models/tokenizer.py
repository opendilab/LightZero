"""
Modified from https://github.com/CompVis/taming-transformers
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F


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
    def __init__(self, encoder=None, decoder_network=None, with_lpips: bool = False, obs_type=None) -> None:
        """Initialize the Tokenizer.

        Arguments:
            encoder (nn.Module, optional): Encoder network. Defaults to None.
            decoder_network (nn.Module, optional): Decoder network. Defaults to None.
            with_lpips (bool, optional): Whether to use LPIPS for perceptual loss. Defaults to False.
        """
        super().__init__()
        if with_lpips:
            from lzero.model.unizero_world_models.lpips import LPIPS
            self.lpips = LPIPS().eval()
        else:
            self.lpips = None

        self.encoder = encoder
        self.decoder_network = decoder_network
        self.obs_type = obs_type

    def encode_to_obs_embeddings(self, x: torch.Tensor, task_id = None) -> torch.Tensor:
        """
        Encode observations to embeddings.

        Arguments:
            - x (torch.Tensor): Input tensor of shape (B, ...).

        Returns:
            - torch.Tensor: Encoded embeddings of shape (B, 1, E).
        """
        shape = x.shape
        # TODO: ======
        if task_id is None:
            # for compatibility with multitask setting
            task_id = 0
        else:
            # task_id = 0  # one share encoder
            task_id = task_id  # TODO: one encoder per task
        # print(f'='*20)
        # print(f'x.shape:{x.shape}')
        # print(f'self.encoder:{self.encoder}')

        # Process input tensor based on its dimensionality
        if len(shape) == 2:
            # Case when input is 2D (B, E)
            # obs_embeddings = self.encoder[task_id](x)
            obs_embeddings = self.encoder(x, task_id)  # TODO:

            obs_embeddings = rearrange(obs_embeddings, 'b e -> b 1 e')
        elif len(shape) == 3:
            # Case when input is 3D (B, T, E)
            x = x.contiguous().view(-1, shape[-1])  # Flatten the last two dimensions (B * T, E)
            # obs_embeddings = self.encoder[task_id](x)
            obs_embeddings = self.encoder(x,task_id)  # TODO:

            obs_embeddings = rearrange(obs_embeddings, 'b e -> b 1 e')
        elif len(shape) == 4:
            # Case when input is 4D (B, C, H, W)
            if self.obs_type == 'vector':
                obs_embeddings = self.encoder(x, task_id=task_id)  # TODO: for dmc multitask
            elif self.obs_type == 'image':
                obs_embeddings = self.encoder[0](x) # TODO: for atari/memory env
                # obs_embeddings = self.encoder(x) # TODO: for atari/memory env single-task

            obs_embeddings = rearrange(obs_embeddings, 'b e -> b 1 e')
        elif len(shape) == 5:
            # Case when input is 5D (B, T, C, H, W)
            x = x.contiguous().view(-1, *shape[-3:])  # Flatten the first two dimensions (B * T, C, H, W)
            if self.obs_type == 'vector':
                obs_embeddings = self.encoder[task_id](x)
            elif self.obs_type == 'image':
                obs_embeddings = self.encoder[0](x) # TODO: for atari/memory env 
                # obs_embeddings = self.encoder(x) # TODO: for atari/memory env single-task

            obs_embeddings = rearrange(obs_embeddings, 'b e -> b 1 e')
        else:
            raise ValueError(f"Invalid input shape: {shape}")

        return obs_embeddings

    def decode_to_obs(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Decode embeddings to observations.

        Arguments:
            embeddings (:obj:`torch.Tensor`): Input embeddings.

        Returns:
            torch.Tensor: Decoded observations.
        """
        return self.decoder_network(embeddings)

    @staticmethod
    def reconstruction_loss(original_images: torch.Tensor, reconstructed_images: torch.Tensor) -> torch.Tensor:
        """Calculate the reconstruction loss.

        Arguments:
            - original_images (:obj:`torch.Tensor`): Original images.
            - reconstructed_images (:obj:`torch.Tensor`): Reconstructed images.

        Returns:
            - torch.Tensor: Computed reconstruction loss.
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
            original_images (:obj:`torch.Tensor`): Original images.
            reconstructed_images (:obj:`torch.Tensor`): Reconstructed images.

        Returns:
            torch.Tensor: Computed perceptual loss.
        """
        return torch.mean(self.lpips(original_images, reconstructed_images))

    def __repr__(self) -> str:
        return "Tokenizer"
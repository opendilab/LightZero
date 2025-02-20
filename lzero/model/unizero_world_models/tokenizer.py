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
    def __init__(self, encoder=None, decoder_network=None, with_lpips: bool = False, projection: list = None) -> None:
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
        if projection is None:
            self.projection_layer = nn.Identity()
        else:
            self.projection_layer = nn.Linear(projection[0], projection[1])

    def encode_to_obs_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode observations to embeddings.

        Arguments:
            x (torch.Tensor): Input tensor of shape (B, ...).

        Returns:
            torch.Tensor: Encoded embeddings of shape (B, 1, E).
        """
        # NOTE: only for Jerico env
        # x = x.long()  # 确保输入为长整型
        shape = x.shape
        # print(f"Max index in x: {x.max()}")
        # print(f"Min index in x: {x.min()}")
        # Process input tensor based on its dimensionality
        if len(shape) == 2:
            # Case when input is 2D (B, E)
            obs_embeddings = self.encoder(x)
            obs_embeddings = rearrange(obs_embeddings, 'b e -> b 1 e')
        elif len(shape) == 3:
            # Case when input is 3D (B, T, E)
            # print(f"x:{x}")
            try:
                x = x.contiguous().view(-1, shape[-1])  # Flatten the last two dimensions (B * T, E)
            except Exception as e:
                print(f"x:{x}")
                print(' =='*20)
                print(e)

            obs_embeddings = self.encoder(x)
            obs_embeddings = rearrange(obs_embeddings, 'b e -> b 1 e')
        elif len(shape) == 4:
            # Case when input is 4D (B, C, H, W)
            obs_embeddings = self.encoder(x)
            obs_embeddings = rearrange(obs_embeddings, 'b e -> b 1 e')
        elif len(shape) == 5:
            # Case when input is 5D (B, T, C, H, W)
            x = x.contiguous().view(-1, *shape[-3:])  # Flatten the first two dimensions (B * T, C, H, W)
            obs_embeddings = self.encoder(x)
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

    def lm_reconstruction_loss(self, labels: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        total_dims = 1
        for i in labels.shape:
            total_dims *= i
        logits = logits.reshape(total_dims, -1)
        labels = labels.reshape(total_dims)
        loss = F.cross_entropy(logits, labels)
        return loss

    def decode_to_language_logits(self, embeddings: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        # embeddings: [B, T, H] -> [B * T, 1, H]
        embeddings = embeddings.reshape(embeddings.shape[0] * embeddings.shape[1], 1, -1)
        # target_ids: [B, T, L] -> [B * T, L]
        target_ids = target_ids.reshape(target_ids.shape[0] * target_ids.shape[1], -1)

        # For each decision transformer token (encoding for one observation),
        # the embedding serves as the initial hidden state for t5 to decode.
        # Hence, the sequence dimension can be paralleled, i.e. should be merged to the batch dimension.
        embeddings = self.projection_layer(embeddings)
        outputs = self.decoder_network(
            input_ids=target_ids,
            encoder_hidden_states=embeddings,
        )
        logits = self.decoder_network.lm_head(outputs.last_hidden_state)
        return logits

    def __repr__(self) -> str:
        return "Tokenizer"
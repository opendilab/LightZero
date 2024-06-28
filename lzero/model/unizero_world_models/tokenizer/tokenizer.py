"""
Credits to https://github.com/CompVis/taming-transformers
"""

from dataclasses import dataclass
from typing import Any

from einops import rearrange
import torch
import torch.nn as nn

from lzero.model.unizero_world_models.lpips import LPIPS


class LossWithIntermediateLosses:
    def __init__(self, **kwargs):
        self.loss_total = sum(kwargs.values())
        self.intermediate_losses = {k: v.item() for k, v in kwargs.items()}

    def __truediv__(self, value):
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
    def __init__(self, encoder=None, decoder_network=None, with_lpips: bool = True, ) -> None:
        super().__init__()
        self.lpips = LPIPS().eval() if with_lpips else None
        self.encoder = encoder
        self.decoder_network = decoder_network

    def compute_loss(self, batch, **kwargs: Any) -> LossWithIntermediateLosses:
        if len(batch['observations'][0, 0].shape) == 3:
            # obs is a 3-dimensional image
            pass
        elif len(batch['observations'][0, 0].shape) == 1:
            # print('obs is a 1-dimensional vector.')
            # TODO()
            # obs is a 1-dimensional vector
            original_shape = list(batch['observations'].shape)
            desired_shape = original_shape + [64, 64]
            expanded_observations = batch['observations'].unsqueeze(-1).unsqueeze(-1)
            expanded_observations = expanded_observations.expand(*desired_shape)
            batch['observations'] = expanded_observations

        assert self.lpips is not None
        observations = self.preprocess_input(rearrange(batch['observations'], 'b t c h w -> (b t) c h w'))
        z, z_quantized, reconstructions = self(observations, should_preprocess=False, should_postprocess=False)

        # Codebook loss. Notes:
        # - beta position is different from taming and identical to original VQVAE paper
        # - VQVAE uses 0.25 by default
        beta = 1.0
        commitment_loss = (z.detach() - z_quantized).pow(2).mean() + beta * (z - z_quantized.detach()).pow(2).mean()
        # L1 loss
        reconstruction_loss = torch.abs(observations - reconstructions).mean()
        # for atari pong
        perceptual_loss = torch.mean(self.lpips(observations, reconstructions))
        # TODO: NOTE only for cartpole
        # perceptual_loss = torch.zeros_like(reconstruction_loss)

        return LossWithIntermediateLosses(commitment_loss=commitment_loss, reconstruction_loss=reconstruction_loss,
                                          perceptual_loss=perceptual_loss)

    # @profile
    def encode_to_obs_embeddings(self, x: torch.Tensor):
        shape = x.shape  # (..., C, H, W)
        if len(shape) == 2:
            # x shape (4,4)
            obs_embeddings = self.encoder(x)
            obs_embeddings = rearrange(obs_embeddings, 'b e -> b 1 (e)')  # (4,1,256)
        elif len(shape) == 3:
            # x shape (32,5,4)
            x = x.contiguous().view(-1, shape[-1])  # (32,5,4) -> (160, 4)
            obs_embeddings = self.encoder(x)  # (160, 4) -> (160, 256)
            obs_embeddings = rearrange(obs_embeddings, 'b e -> b 1 (e)')  # ()

        if len(shape) == 4:
            # x shape (4,3,64,64)
            obs_embeddings = self.encoder(x)  # (4,3,64,64) -> (4,64,4,4)
            # obs_embeddings = rearrange(obs_embeddings, 'b c h w -> b 1 (c h w)')  # (4,1,1024)
            obs_embeddings = rearrange(obs_embeddings, 'b e -> b 1 e')  # (4,1,256) # TODO

        elif len(shape) == 5:
            # x shape (32,5,3,64,64)
            x = x.contiguous().view(-1, *shape[-3:])  # (32,5,3,64,64) -> (160,3,64,64)
            obs_embeddings = self.encoder(x)  # (160,3,64,64) -> (160,64,4,4)
            # obs_embeddings = rearrange(obs_embeddings, 'b c h w -> b 1 (c h w)')  # (160,1,1024)
            obs_embeddings = rearrange(obs_embeddings, 'b e -> b 1 e')  # (4,1,256) # TODO

        return obs_embeddings

    # @profile
    def decode_to_obs(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.decoder_network(embeddings)

    # @profile
    def reconstruction_loss(self, original_images: torch.Tensor, reconstructed_images: torch.Tensor) -> torch.Tensor:
        # Mean Squared Error (MSE) is commonly used as a reconstruction loss
        if len(original_images.shape) == 2:  # TODO
            # for memory env vector obs
            loss = nn.MSELoss()(original_images, reconstructed_images)  # L2 loss
        else:
            # for atari image env
            loss = torch.abs(original_images - reconstructed_images).mean()  # L1 loss
        return loss

    # @profile
    def perceptual_loss(self, original_images: torch.Tensor, reconstructed_images: torch.Tensor) -> torch.Tensor:
        # Mean Squared Error (MSE) is commonly used as a reconstruction loss
        perceptual_loss = torch.mean(self.lpips(original_images, reconstructed_images))
        return perceptual_loss

    @torch.no_grad()
    def encode_decode(self, x: torch.Tensor, should_preprocess: bool = False,
                      should_postprocess: bool = False) -> torch.Tensor:
        z_q = self.encode(x, should_preprocess).z_quantized
        return self.decode(z_q, should_postprocess)

    def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        """x is supposed to be channels first and in [0, 1]"""
        # [0,1] -> [-1, 1]
        return x.mul(2).sub(1)

    def postprocess_output(self, y: torch.Tensor) -> torch.Tensor:
        """y is supposed to be channels first and in [-1, 1]"""
        # [-1, 1] -> [0,1]
        return y.add(1).div(2)

    def __repr__(self) -> str:
        return "tokenizer"

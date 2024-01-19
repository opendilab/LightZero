"""
Credits to https://github.com/CompVis/taming-transformers
"""

from dataclasses import dataclass
from typing import Any, Tuple

from einops import rearrange
import torch
import torch.nn as nn

# from dataset import Batch
from .lpips import LPIPS
from .nets import Encoder, Decoder
# from utils import LossWithIntermediateLosses


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
    def __init__(self, vocab_size: int, embed_dim: int, encoder: Encoder, decoder: Decoder, with_lpips: bool = True, representation_network = None, decoder_network =None) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.encoder = encoder
        self.pre_quant_conv = torch.nn.Conv2d(encoder.config.z_channels, embed_dim, 1)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, decoder.config.z_channels, 1)
        self.decoder = decoder
        self.embedding.weight.data.uniform_(-1.0 / vocab_size, 1.0 / vocab_size)
        self.lpips = LPIPS().eval() if with_lpips else None
        self.representation_network = representation_network
        self.decoder_network = decoder_network


    def __repr__(self) -> str:
        return "tokenizer"

    def forward(self, x: torch.Tensor, should_preprocess: bool = False, should_postprocess: bool = False) -> Tuple[torch.Tensor]:
        outputs = self.encode(x, should_preprocess)
        decoder_input = outputs.z + (outputs.z_quantized - outputs.z).detach()
        reconstructions = self.decode(decoder_input, should_postprocess)
        return outputs.z, outputs.z_quantized, reconstructions

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
        # IRIS original code
        observations = self.preprocess_input(rearrange(batch['observations'], 'b t c h w -> (b t) c h w'))
        # TODO
        # observations = rearrange(batch['observations'], 'b t c h w -> (b t) c h w')

        z, z_quantized, reconstructions = self(observations, should_preprocess=False, should_postprocess=False)

        # Codebook loss. Notes:
        # - beta position is different from taming and identical to original VQVAE paper
        # - VQVAE uses 0.25 by default
        beta = 1.0
        commitment_loss = (z.detach() - z_quantized).pow(2).mean() + beta * (z - z_quantized.detach()).pow(2).mean()
        # L1 loss
        reconstruction_loss = torch.abs(observations - reconstructions).mean()
        # TODO: for atari pong
        perceptual_loss = torch.mean(self.lpips(observations, reconstructions))
        # TODO: NOTE only for cartpole
        perceptual_loss = torch.zeros_like(reconstruction_loss)

        return LossWithIntermediateLosses(commitment_loss=commitment_loss, reconstruction_loss=reconstruction_loss, perceptual_loss=perceptual_loss)

    def encode(self, x: torch.Tensor, should_preprocess: bool = False) -> TokenizerEncoderOutput:
        if should_preprocess:
            x = self.preprocess_input(x)
        
        # # only for debug
        # observations = x
        # z, z_quantized, reconstructions = self(observations, should_preprocess=False, should_postprocess=False)
        # reconstruction_loss = torch.abs(observations - reconstructions).mean()
        # perceptual_loss = torch.mean(self.lpips(observations, reconstructions))
        # rec_img = self.postprocess_output(reconstructions)
        # # only for debug

        shape = x.shape  # (..., C, H, W)
        x = x.view(-1, *shape[-3:])
        z = self.encoder(x)
        z = self.pre_quant_conv(z)
        b, e, h, w = z.shape
        z_flattened = rearrange(z, 'b e h w -> (b h w) e')
        dist_to_embeddings = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + torch.sum(self.embedding.weight**2, dim=1) - 2 * torch.matmul(z_flattened, self.embedding.weight.t())

        tokens = dist_to_embeddings.argmin(dim=-1)
        z_q = rearrange(self.embedding(tokens), '(b h w) e -> b e h w', b=b, e=e, h=h, w=w).contiguous()

        # Reshape to original
        z = z.reshape(*shape[:-3], *z.shape[1:])
        z_q = z_q.reshape(*shape[:-3], *z_q.shape[1:])
        tokens = tokens.reshape(*shape[:-3], -1)

        return TokenizerEncoderOutput(z, z_q, tokens)

    def encode_to_obs_embeddings(self, x: torch.Tensor, should_preprocess: bool = False):
        # if should_preprocess:
        #     x = self.preprocess_input(x)
        
        # # only for debug
        # observations = x
        # z, z_quantized, reconstructions = self(observations, should_preprocess=False, should_postprocess=False)
        # reconstruction_loss = torch.abs(observations - reconstructions).mean()
        # perceptual_loss = torch.mean(self.lpips(observations, reconstructions))
        # rec_img = self.postprocess_output(reconstructions)
        # # only for debug

        shape = x.shape  # (..., C, H, W)
        # x = x.view(-1, *shape[-3:])

        #===============
        # z = self.encoder(x)
        # z = self.pre_quant_conv(z)
        # b, e, h, w = z.shape

        # z_flattened = rearrange(z, 'b e h w -> (b h w) e')
        # dist_to_embeddings = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + torch.sum(self.embedding.weight**2, dim=1) - 2 * torch.matmul(z_flattened, self.embedding.weight.t())

        # tokens = dist_to_embeddings.argmin(dim=-1)
        # z_q = rearrange(self.embedding(tokens), '(b h w) e -> b e h w', b=b, e=e, h=h, w=w).contiguous()

        # Reshape to original
        # z = z.reshape(*shape[:-3], *z.shape[1:])
        # return z

        # z_q = z_q.reshape(*shape[:-3], *z_q.shape[1:])
        # tokens = tokens.reshape(*shape[:-3], -1)

        # obs_embeddings = z.reshape(*shape[:-3], -1, e)
        # obs_embeddings = rearrange(z, 'b e h w -> b (h w) e')  # 3, 16, 64  TODO: K=16
        # obs_embeddings = rearrange(z, 'b e h w -> b 1 (h w e)')  # 3, 16, 64 TODO: K=1
        #===============

        #===============
        if len(shape) == 2:
            # x shape (4,4)
            obs_embeddings = self.representation_network(x)
            obs_embeddings = rearrange(obs_embeddings, 'b e -> b 1 (e)')  # (4,1,256)
        elif len(shape) == 3:
            # x shape (32,5,4)
            x = x.contiguous().view(-1, shape[-1]) # (32,5,4) -> (160, 4)
            obs_embeddings = self.representation_network(x) # (160, 4) -> (160, 256)
            obs_embeddings = rearrange(obs_embeddings, 'b e -> b 1 (e)')  # ()

        if len(shape) == 4:
            # x shape (4,3,64,64)
            obs_embeddings = self.representation_network(x) # (4,3,64,64) -> (4,64,4,4)
            # obs_embeddings = rearrange(obs_embeddings, 'b c h w -> b 1 (c h w)')  # (4,1,1024)
            obs_embeddings = rearrange(obs_embeddings, 'b e -> b 1 e')  # (4,1,256) # TODO

        elif len(shape) == 5:
            # x shape (32,5,3,64,64)
            x = x.contiguous().view(-1, *shape[-3:]) # (32,5,3,64,64) -> (160,3,64,64)
            obs_embeddings = self.representation_network(x) # (160,3,64,64) -> (160,64,4,4)
            # obs_embeddings = rearrange(obs_embeddings, 'b c h w -> b 1 (c h w)')  # (160,1,1024)
            obs_embeddings = rearrange(obs_embeddings, 'b e -> b 1 e')  # (4,1,256) # TODO


        return obs_embeddings

    def decode_to_obs(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.decoder_network(embeddings)


    def reconstruction_loss(self, original_images: torch.Tensor, reconstructed_images: torch.Tensor) -> torch.Tensor:
        # Mean Squared Error (MSE) is commonly used as a reconstruction loss
        # loss = nn.MSELoss()(original_images, reconstructed_images) # L1 loss
        loss = torch.abs(original_images - reconstructed_images).mean()
        return loss



    def perceptual_loss(self, original_images: torch.Tensor, reconstructed_images: torch.Tensor) -> torch.Tensor:
        # Mean Squared Error (MSE) is commonly used as a reconstruction loss
        perceptual_loss = torch.mean(self.lpips(original_images, reconstructed_images))
        return perceptual_loss



    def decode(self, z_q: torch.Tensor, should_postprocess: bool = False) -> torch.Tensor:
        shape = z_q.shape  # (..., E, h, w)
        z_q = z_q.view(-1, *shape[-3:])
        z_q = self.post_quant_conv(z_q)
        rec = self.decoder(z_q)
        rec = rec.reshape(*shape[:-3], *rec.shape[1:])
        if should_postprocess:
            rec = self.postprocess_output(rec)
        return rec

    @torch.no_grad()
    def encode_decode(self, x: torch.Tensor, should_preprocess: bool = False, should_postprocess: bool = False) -> torch.Tensor:
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

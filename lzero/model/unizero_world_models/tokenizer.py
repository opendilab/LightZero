"""
Modified from https://github.com/CompVis/taming-transformers
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F
from typing import Optional, List
from transformers.modeling_outputs import BaseModelOutput

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
        Can operate on visual or textual data, supporting optional LPIPS perceptual loss.
        It optionally includes a linear projection layer and can be paired with a decoder tokenizer.
    """
    def __init__(self, encoder=None, decoder_network=None, decoder_network_tokenizer=None, with_lpips: bool = False, projection: list = None) -> None:
        """Initialize the Tokenizer.

        Arguments:
            encoder (nn.Module, optional): Encoder network to transform raw inputs into embeddings.
            decoder_network (nn.Module, optional): Decoder network used for observation reconstruction or text generation.
            decoder_network_tokenizer (PreTrainedTokenizer, optional): Tokenizer compatible with the decoder network (e.g., T5 tokenizer).
            with_lpips (bool, optional): If True, enable perceptual loss computation via LPIPS. Defaults to False.
            projection (list[int], optional): If provided, defines a linear projection layer from projection[0] → projection[1]. 
                                              If None, an identity layer is used.
        """
        super().__init__()
        if with_lpips:
            from lzero.model.unizero_world_models.lpips import LPIPS
            self.lpips = LPIPS().eval()
        else:
            self.lpips = None

        self.encoder = encoder
        self.decoder_network = decoder_network
        self.decoder_network_tokenizer = decoder_network_tokenizer 

        if projection is None:
            self.projection_layer = nn.Identity()
        else:
            self.projection_layer = nn.Linear(projection[0], projection[1])


    def decode_to_plain_text(self, x) -> str:
        """
        Decode the input tensor to plain text.

        Arguments:
            x (torch.Tensor): Input tensor of shape (B, ...).

        Returns:
            str: Decoded plain text.
        """
        # Convert the input tensor to a numpy array and decode it
        return self.encoder.tokenizer.batch_decode(x, skip_special_tokens=True)

    def encode_to_obs_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode observations to embeddings.

        Arguments:
            x (torch.Tensor): Input tensor of shape (B, ...).

        Returns:
            torch.Tensor: Encoded embeddings of shape (B, 1, E).
        """
        shape = x.shape
        # Process input tensor based on its dimensionality
        if len(shape) == 2:
            # Case when input is 2D (B, E)
            obs_embeddings = self.encoder(x)
            obs_embeddings = rearrange(obs_embeddings, 'b e -> b 1 e')
        elif len(shape) == 3:
            # Case when input is 3D (B, T, E)
            x = x.contiguous().view(-1, shape[-1])  # Flatten the last two dimensions (B * T, E)
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

    def decode_to_reconstruction_outputs(self, embeddings: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        """
        Overview:
            This function takes input embeddings and corresponding target token IDs,
            then uses a seq2seq decoder (like T5) to reconstruct the original text.
            It handles reshaping, retokenization, projection, and calls the decoder 
            to compute the reconstruction loss and logits.
        Arguments:
            embeddings (torch.Tensor): Input embeddings of shape (B, E), (B, L, E), or (B*T, 1, E).
            target_ids (torch.Tensor): Ground-truth token IDs of shape (B, L) or (B*T, L).
        Returns:
            torch.Tensor: Decoder output including loss, logits, hidden states (if return_dict=True).
        """
        if embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(1)
        elif embeddings.dim() == 3:
            B,T,E = embeddings.shape
            embeddings = embeddings.reshape(B*T,1,E)
            target_ids = target_ids.reshape(B*T, -1)

        # Instead of using raw target_ids, convert them to plain text and re-tokenize using the decoder's tokenizer.
        # This guarantees alignment with the decoder's vocabulary, special tokens, and tokenization rules.
        text_list = self.decode_to_plain_text(target_ids)
        t5_target_ids = self.decoder_network_tokenizer(text_list, 
                                                       padding="max_length",
                                                       truncation=True, 
                                                       max_length=512, 
                                                       return_tensors="pt")
        labels = t5_target_ids.input_ids
        labels[labels == self.decoder_network_tokenizer.pad_token_id] = -100 

        embeddings = self.projection_layer(embeddings)     # (B', 1, E) -> (B', 1, E'), B' = B*T
        encoder_outputs_tuple = BaseModelOutput(last_hidden_state=embeddings)
        encoder_attention_mask = torch.ones(
            embeddings.size(0), embeddings.size(1),
            device=embeddings.device, dtype=torch.long
        )

        labels = labels.to(embeddings.device)

        outputs = self.decoder_network(encoder_outputs=encoder_outputs_tuple,
                                       attention_mask=encoder_attention_mask,
                                       labels=labels,
                                       return_dict=True)
        
        return outputs
    
    def decode_to_plain_text_for_decoder(
            self, embeddings: torch.Tensor,
            max_length: int = 512
        ) -> List[List[int]]:
        """
        Overview:
            This function decodes latent embeddings into plain text using the decoder's generate method.
            It includes projection, prepares encoder outputs and attention mask, and performs autoregressive decoding.
        Arguments:
            embeddings (torch.Tensor): Latent embeddings, shape (B, E) or (B, L, E).
            max_length (int, optional): Max token length for generation. Defaults to 512.
        Returns:
            List[List[int]]: List of decoded strings, one per input in batch.
        """
        
        # Set decoder_network and projection_layer to evaluation mode to disable dropout and other training-specific behaviors.
        self.decoder_network.eval()
        self.projection_layer.eval()

        # If embeddings is not a Tensor, convert it to a torch.Tensor.
        if not isinstance(embeddings, torch.Tensor):
            embeddings = torch.tensor(embeddings, dtype=torch.float32)
        
        # Attempt to retrieve the device information from decoder_network; if unavailable, fall back to the model’s parameters.
        try:
            device = self.decoder_network.device
        except AttributeError:
            device = next(self.decoder_network.parameters()).device
            
        embeddings = embeddings.to(device)

        with torch.no_grad(): 
            if embeddings.dim() == 2:
                embeddings = embeddings.unsqueeze(1)

            embeddings = self.projection_layer(embeddings)

            encoder_outputs_tuple = BaseModelOutput(last_hidden_state=embeddings)
            encoder_attention_mask = torch.ones(
                embeddings.size(0), embeddings.size(1),
                device=device, dtype=torch.long
            )

            # Use the decoder's generate() method to autoregressively decode text from the input embeddings.
            # The projected embeddings serve as encoder outputs in a typical encoder-decoder architecture,
            # where the decoder attends to them via cross-attention at each step until max_length or EOS is reached.
            generated_t5_ids = self.decoder_network.generate(
                encoder_outputs=encoder_outputs_tuple,
                attention_mask=encoder_attention_mask,
                max_length=max_length
            )

            # Convert the generated output to a list of strings on CPU, skipping special tokens.
            generated_text = self.decoder_network_tokenizer.batch_decode(
                generated_t5_ids, skip_special_tokens=True)

            assert len(generated_text) == 1, f"Expected 1 generated text, got {len(generated_text)}"
            
            return generated_text[0]

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
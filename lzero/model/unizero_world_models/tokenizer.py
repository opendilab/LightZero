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
from typing import Optional, List
from transformers.modeling_outputs import BaseModelOutput

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

        if self.encoder_option == 'legacy':  # T5 decoder
            # Instead of using raw target_ids, convert them to plain text and re-tokenize using the decoder's tokenizer.
            # This guarantees alignment with the decoder's vocabulary, special tokens, and tokenization rules.
            text_list = self.encoder.tokenizer.batch_decode(target_ids, skip_special_tokens=True)
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

        elif self.encoder_option == 'qwen':
            hidden = self.projection_layer(embeddings)  
            lm = self.decoder_network.pretrained_model
            # Get a reference parameter for device/dtype info
            param = next(lm.parameters()) 

            try:
                # Retrieve the input embedding layer of the language model
                input_embedding_layer = lm.get_input_embeddings()
            except:
                raise ValueError('Error... Could not retrieve input embedding layer from the decoder network.')
            
            # Convert target token IDs into embeddings using the LM's input embedding layer
            target_embeds = input_embedding_layer(target_ids)

            # Concatenate the projected hidden embeddings (prompt) with target embeddings
            # hidden: (B, 1, D), target_embeds: (B, L, D) → inputs_embeds: (B, 1+L, D)
            inputs_embeds = torch.cat([hidden, target_embeds.detach()], dim=1)

            inputs_embeds = inputs_embeds.to(device=param.device, dtype=param.dtype)

            prompt_attention_mask = torch.ones(hidden.size(0), 1, device=param.device, dtype=torch.long)
            target_attention_mask = (target_ids != self.decoder_network.tokenizer.pad_token_id).to(device=param.device, dtype=torch.long)
            # Concatenate prompt mask and target mask along sequence length
            attention_mask = torch.cat([prompt_attention_mask, target_attention_mask], dim=1)
            # Construct labels: for the prompt part, use -100 (ignored by loss function)
            prompt_labels = torch.full((hidden.size(0), 1), -100, device=param.device, dtype=torch.long)

            # Copy target token IDs as labels, masking pad positions with -100
            labels = target_ids.clone().to(param.device)
            labels[labels == self.decoder_network.tokenizer.pad_token_id] = -100

            final_labels = torch.cat([prompt_labels, labels], dim=1)

            outputs = lm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=final_labels,
                return_dict=True
            )

            return outputs
    
    def decode_to_plain_text(
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
            if self.encoder_option == 'legacy': # T5 decoder
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
            
            elif self.encoder_option == 'qwen':
                return self.decoder_network.decode(embeddings=embeddings, max_length=max_length)

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
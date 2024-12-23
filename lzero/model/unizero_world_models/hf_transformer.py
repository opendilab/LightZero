from typing import Optional

import torch
from transformers import AutoModelForCausalLM
from transformers.utils.cache_utils import DynamicCache

from .kv_caching import KeysValues


def kv2dc(cache: KeysValues):
    res = DynamicCache()
    for kv_cache in cache:
        k_tensor = kv_cache._k_cache.get()
        v_tensor = kv_cache._v_cache.get()
        res.key_cache.append(k_tensor)
        res.value_cache.append(v_tensor)
    return res


class HuggingfaceTransformer(AutoModelForCausalLM):
    def from_pretrained(self, lzero_config, *args, **kwargs):
        self.lzero_config = lzero_config
        super(HuggingfaceTransformer, self).from_pretrained(*args, **kwargs)

    def generate_empty_keys_values(self, n: int, max_tokens: int) -> KeysValues:
        """
        Generate a placeholder for keys and values.

        Arguments:
            - n (:obj:`int`): Batch size.
            - max_tokens (:obj:`int`): Maximum number of tokens in the sequence.

        Returns:
            - KeysValues: An object containing empty keys and values.
        """
        device = self.lzero_config.device  # Assumption: All submodules are on the same device
        return KeysValues(n, self.lzero_config.num_heads, max_tokens,
                          self.lzero_config.embed_dim, self.lzero_config.num_layers, device)

    def forward(self, sequences: torch.Tensor, past_keys_values: Optional[KeysValues] = None,
                valid_context_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the Transformer model.

        Arguments:
            - sequences (:obj:`torch.Tensor`): Input tensor of shape (batch_size, seq_length, embed_dim).
            - past_keys_values (:obj:`Optional[KeysValues]`): Precomputed keys and values for faster generation (default: None).
            - valid_context_lengths (:obj:`Optional[torch.Tensor]`): Valid lengths of context for masking (default: None).

        Returns:
            - torch.Tensor: Output tensor of shape (batch_size, seq_length, embed_dim).
        """
        assert past_keys_values is None or len(past_keys_values) == len(self.blocks)
        if past_keys_values is not None:
            kv_cache = kv2dc(past_keys_values)
            use_cache = True
        else:
            kv_cache = None
            use_cache = False

        B, T, _ = sequences.shape
        attention_mask = torch.arange(T).expand(B, T) >= (T - valid_context_lengths.unsqueeze(1))
        # print(valid_context_lengths.shape)
        # print(attention_mask.shape)
        # print(sequences.shape)
        # assert False

        output = super.forward(
            attention_mask=attention_mask,
            past_key_values=kv_cache,
            inputs_embeds=sequences,
            use_cache=use_cache
        )

        return output.logits[:, -1, :]

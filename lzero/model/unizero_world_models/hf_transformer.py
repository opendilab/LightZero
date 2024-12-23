from typing import Optional

import torch
from transformers import LlamaForCausalLM
from transformers.cache_utils import DynamicCache

from .kv_caching import KeysValues


def kv2dc(cache: KeysValues):
    res = DynamicCache()
    for kv_cache in cache:
        k_tensor = kv_cache._k_cache.get()
        v_tensor = kv_cache._v_cache.get()
        res.key_cache.append(k_tensor)
        res.value_cache.append(v_tensor)
    return res


def update_kv(cache: KeysValues, new_cache: DynamicCache):
    for i in range(len(new_cache.key_cache)):
        cache[i].update(new_cache.key_cache[-1], new_cache.value_cache[-1])


class HuggingfaceLlamaTransformer(LlamaForCausalLM):
    @classmethod
    def from_pretrained(cls, lzero_config, *args, **kwargs):
        # Add custom logic here
        model = super(HuggingfaceLlamaTransformer, cls).from_pretrained(*args, **kwargs)
        model.lzero_config = lzero_config
        return model

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

    def _get_positional_embedding(self, layer, attn_type, pos_emb) -> torch.Tensor:
        """
         Helper function to get positional embedding for a given layer and attention type.

         Arguments:
         - layer (:obj:`int`): Layer index.
         - attn_type (:obj:`str`): Attention type, either 'key' or 'value'.

         Returns:
         - torch.Tensor: The positional embedding tensor.
         """
        if attn_type == 'key':
            module_name = 'k_proj'
        elif attn_type == 'value':
            module_name = 'v_proj'
        elif attn_type == 'query':
            module_name = 'q_proj'
        else:
            assert False
        attn_func = getattr(self.model.layers[layer].self_attn, module_name)
        return attn_func(pos_emb.weight)

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
        assert past_keys_values is None or len(past_keys_values) == len(self.model.layers)
        if past_keys_values is not None:
            kv_cache = kv2dc(past_keys_values)
            use_cache = True
        else:
            kv_cache = None
            use_cache = False

        B, T, _ = sequences.shape
        if valid_context_lengths is not None:
            attention_mask = torch.arange(T).expand(B, T) >= (T - valid_context_lengths.unsqueeze(1))
        else:
            attention_mask = torch.ones_like(sequences)
        # print(valid_context_lengths.shape)
        # print(attention_mask.shape)
        # print(sequences.shape)
        # assert False

        output = self.model.forward(
            attention_mask=attention_mask,
            past_key_values=kv_cache,
            inputs_embeds=sequences,
            use_cache=use_cache
        )

        update_kv(past_keys_values, kv_cache)
        return output.logits[:, -1, :]

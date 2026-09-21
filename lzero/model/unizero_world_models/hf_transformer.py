from typing import Optional

import torch
from transformers import Qwen2ForCausalLM
from transformers.cache_utils import DynamicCache

from .kv_caching import KeysValues


def kv2dc(cache: KeysValues) -> DynamicCache:
    legacy_cache = tuple((kv_cache._k_cache.get(), kv_cache._v_cache.get()) for kv_cache in cache)
    return DynamicCache.from_legacy_cache(legacy_cache)


def update_kv(cache: KeysValues, new_cache: DynamicCache) -> None:
    for i, (key_cache, value_cache) in enumerate(new_cache.to_legacy_cache()):
        cache[i].update(key_cache[:, :, -1:, :], value_cache[:, :, -1:, :])


class HuggingfaceQwenTransformer(Qwen2ForCausalLM):
    """Qwen2 backbone adapter exposing the minimal UniZero transformer interface."""

    @classmethod
    def from_pretrained(cls, lzero_config, *args, **kwargs):
        model = super(HuggingfaceQwenTransformer, cls).from_pretrained(*args, **kwargs)
        model.lzero_config = lzero_config
        return model

    def generate_empty_keys_values(self, n: int, max_tokens: int) -> KeysValues:
        device = torch.device(self.lzero_config.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            device = torch.device("cpu")
        return KeysValues(
            n,
            self.lzero_config.num_heads,
            max_tokens,
            self.lzero_config.embed_dim,
            self.lzero_config.num_layers,
            device,
            self.lzero_config.hidden_size,
        )

    def _get_positional_embedding(self, layer: int, attn_type: str, pos_emb) -> torch.Tensor:
        if attn_type == 'key':
            module_name = 'k_proj'
        elif attn_type == 'value':
            module_name = 'v_proj'
        elif attn_type == 'query':
            module_name = 'q_proj'
        else:
            raise ValueError(f"Unsupported attention projection type: {attn_type}")
        attn_func = getattr(self.model.layers[layer].self_attn, module_name)
        return attn_func(pos_emb.weight)

    def forward(
            self,
            sequences: torch.Tensor,
            past_keys_values: Optional[KeysValues] = None,
            valid_context_lengths: Optional[torch.Tensor] = None,
            start_pos: int = 0,
    ) -> torch.Tensor:
        assert past_keys_values is None or len(past_keys_values) == len(self.model.layers)
        if past_keys_values is not None:
            kv_cache = kv2dc(past_keys_values)
            use_cache = True
        else:
            kv_cache = None
            use_cache = False

        batch_size, seq_len, _ = sequences.shape
        if valid_context_lengths is not None:
            position = torch.arange(seq_len, device=sequences.device).expand(batch_size, seq_len)
            attention_mask = position >= (seq_len - valid_context_lengths.to(sequences.device).unsqueeze(1))
        else:
            attention_mask = torch.ones(batch_size, seq_len, device=sequences.device, dtype=torch.long)

        output = self.model.forward(
            attention_mask=attention_mask,
            past_key_values=kv_cache,
            inputs_embeds=sequences,
            use_cache=use_cache,
        )

        if kv_cache is not None:
            update_kv(past_keys_values, kv_cache)
        return output.last_hidden_state

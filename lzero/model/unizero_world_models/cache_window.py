"""Exact KV-window rebuild helpers shared by UniZero world models."""

from typing import Dict, NamedTuple, Optional

import torch

from .kv_caching import KeysValues


class ExactCacheResetBatch(NamedTuple):
    """Batched exact-window rebuild and its mapping back to input samples."""

    cache: Optional[KeysValues]
    offsets: Dict[int, int]
    retained_contexts: Dict[int, torch.Tensor]


class CacheWindowMixin:
    """Shared primitives for exact absolute-position KV-window rebuilding."""

    def _prepare_exact_cache_resets(
            self, latent_state: torch.Tensor, effective_sizes
    ) -> ExactCacheResetBatch:
        """Rebuild every overflowing sample in one transformer call."""
        reset_indices = [
            index for index, size in enumerate(effective_sizes)
            if size >= self.context_length - 1
        ]
        if not reset_indices:
            return ExactCacheResetBatch(None, {}, {})

        self.exact_kv_reset_batches += 1
        self.exact_kv_reset_samples += len(reset_indices)
        reset_cache = self.transformer.generate_empty_keys_values(
            n=len(reset_indices), max_tokens=self.context_length
        )
        retained_contexts = {}
        if self.rebuild_kv_window_from_tokens:
            if len(self.keys_values_wm_token_context_list) != latent_state.size(0):
                raise RuntimeError(
                    'Raw token contexts are missing for an overflowing KV-cache batch.'
                )
            keep_tokens = self.context_length - 3
            for index in reset_indices:
                history = self.keys_values_wm_token_context_list[index]
                if history.size(0) < keep_tokens:
                    raise RuntimeError(
                        f'KV cache reports {effective_sizes[index]} tokens but raw history '
                        f'contains only {history.size(0)}; cannot rebuild exactly.'
                    )
                retained_contexts[index] = history[-keep_tokens:]
            reset_sequences = torch.stack([
                retained_contexts[index] for index in reset_indices
            ])
        else:
            reset_sequences = latent_state[reset_indices]

        transformer_start_pos = 0
        if not self.config.rotary_emb:
            positions = torch.arange(reset_sequences.size(1), device=self.device)
            reset_sequences = (
                reset_sequences + self._lookup_position_embeddings(positions)
            )
            transformer_start_pos = None
        # Only the K/V side effect is required; prediction heads would add
        # substantial MCTS-time work and all of their outputs would be dropped.
        self.transformer(
            reset_sequences,
            past_keys_values=reset_cache,
            start_pos=transformer_start_pos,
        )
        return ExactCacheResetBatch(
            reset_cache,
            {index: offset for offset, index in enumerate(reset_indices)},
            retained_contexts,
        )

    def _copy_exact_cache_reset(self, reset_batch: ExactCacheResetBatch, index: int):
        """Copy one sample from a batched rebuild into an independent cache."""
        destination_cache = self.transformer.generate_empty_keys_values(
            n=1, max_tokens=self.context_length
        )
        source_offset = reset_batch.offsets[index]
        for source_layer, destination_layer in zip(
            reset_batch.cache._keys_values,
            destination_cache._keys_values,
        ):
            destination_layer._k_cache._cache.copy_(
                source_layer._k_cache._cache[source_offset:source_offset + 1]
            )
            destination_layer._v_cache._cache.copy_(
                source_layer._v_cache._cache[source_offset:source_offset + 1]
            )
            destination_layer._k_cache._size = source_layer._k_cache._size
            destination_layer._v_cache._size = source_layer._v_cache._size
        return destination_cache

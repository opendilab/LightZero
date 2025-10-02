# -*- coding: utf-8 -*-
"""
This script is a refactored version of the key-value caching mechanism from:
https://github.com/eloialonso/iris/blob/main/src/models/kv_caching.py

The optimization focuses on improving clarity, documentation, and adherence to modern coding standards
while strictly preserving the original functionality and external API.
"""
from typing import Tuple, Optional

import numpy as np
import torch


class AssignWithoutInplaceCheck(torch.autograd.Function):
    """
    Overview:
        A custom autograd function to perform an in-place-like assignment on a tensor slice
        without triggering PyTorch's version counter checks. This is useful for updating
        buffers or caches within a computation graph.

    Reference:
        Inspired by discussions on the PyTorch forums, such as:
        https://discuss.pytorch.org/t/disable-in-place-correctness-version-check-any-other-workaround/90738/4

    .. warning::
        This function is unsafe if the same slice of the input tensor is overwritten
        multiple times, as it can lead to incorrect gradient calculations.
    """

    @staticmethod
    def _get_slice(dim: int, start: int, stop: int) -> Tuple[slice, ...]:
        """
        Overview:
            Creates a slice tuple for indexing a tensor at a specific dimension.
        Arguments:
            - dim (:obj:`int`): The dimension to slice along.
            - start (:obj:`int`): The starting index for the slice.
            - stop (:obj:`int`): The ending index for the slice.
        Returns:
            - slice_tuple (:obj:`Tuple[slice, ...]`): A tuple of slice objects for indexing.
        """
        return (slice(None),) * dim + (slice(start, stop),)

    @staticmethod
    def forward(
            ctx,
            input_tensor: torch.Tensor,
            value: torch.Tensor,
            dim: int,
            start: int,
            stop: int
    ) -> torch.Tensor:
        """
        Overview:
            The forward pass assigns the `value` tensor to a slice of the `input_tensor`.
        Arguments:
            - ctx: The context object for storing information for the backward pass.
            - input_tensor (:obj:`torch.Tensor`): The tensor to be modified.
            - value (:obj:`torch.Tensor`): The tensor to assign to the slice.
            - dim (:obj:`int`): The dimension along which to perform the assignment.
            - start (:obj:`int`): The starting index of the slice.
            - stop (:obj:`int`): The ending index of the slice.
        Returns:
            - modified_tensor (:obj:`torch.Tensor`): The `input_tensor` after modification.
        """
        ctx.dim = dim
        ctx.start = start
        ctx.stop = stop
        # Directly modify the data of the input tensor to bypass version checks.
        input_tensor.data[AssignWithoutInplaceCheck._get_slice(dim, start, stop)] = value
        return input_tensor

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        """
        Overview:
            The backward pass computes gradients for the inputs of the forward pass.
        Arguments:
            - ctx: The context object with saved information from the forward pass.
            - grad_output (:obj:`torch.Tensor`): The gradient of the output tensor.
        Returns:
            - grad_input_tensor (:obj:`torch.Tensor`): The gradient with respect to `input_tensor`.
            - grad_value (:obj:`torch.Tensor`): The gradient with respect to `value`.
            - None, None, None: Gradients for `dim`, `start`, and `stop`, which are not needed.
        """
        # The gradient for the original input tensor is the same as the output gradient.
        grad_input_tensor = grad_output
        # The gradient for the value tensor is the slice of the output gradient.
        grad_value = grad_output[AssignWithoutInplaceCheck._get_slice(ctx.dim, ctx.start, ctx.stop)]
        return grad_input_tensor, grad_value, None, None, None


class Cache:
    """
    Overview:
        A cache for storing a single type of intermediate tensor (e.g., keys or values)
        in a Transformer-like model. It handles dynamic updates and size management.
    """

    def __init__(self, num_samples: int, num_heads: int, max_tokens: int, embed_dim: int, device: torch.device) -> None:
        """
        Overview:
            Initializes the cache.
        Arguments:
            - num_samples (:obj:`int`): The number of samples (batch size) to cache.
            - num_heads (:obj:`int`): The number of attention heads.
            - max_tokens (:obj:`int`): The maximum number of tokens the cache can hold.
            - embed_dim (:obj:`int`): The total dimension of the embeddings.
            - device (:obj:`torch.device`): The device on which to store the cache tensor.
        """
        if embed_dim % num_heads != 0:
            raise ValueError(f"Embedding dimension ({embed_dim}) must be divisible by the number of heads ({num_heads}).")

        self._num_samples = num_samples
        self._num_heads = num_heads
        self._max_tokens = max_tokens
        self._head_dim = embed_dim // num_heads
        self._device = device

        self._cache: torch.Tensor = self._create_cache_tensor(self._num_samples)
        self._size: int = 0
        self.reset()

    def _create_cache_tensor(self, num_samples: int) -> torch.Tensor:
        """
        Overview:
            Creates an empty tensor with the correct shape and device for the cache.
        Arguments:
            - num_samples (:obj:`int`): The number of samples for which to create the cache.
        Returns:
            - empty_cache (:obj:`torch.Tensor`): An uninitialized tensor for the cache.
        """
        return torch.empty(
            num_samples, self._num_heads, self._max_tokens, self._head_dim, device=self._device
        )  # Shape: (B, nh, T, hs)

    @property
    def shape(self) -> Tuple[int, int, int, int]:
        """
        Overview:
            Gets the effective shape of the cache's content.
        Returns:
            - shape (:obj:`Tuple[int, int, int, int]`): A tuple representing (num_samples, num_heads, current_size, head_dim).
        """
        return self._num_samples, self._num_heads, self._size, self._head_dim

    def reset(self) -> None:
        """
        Overview:
            Resets the cache to an empty state.
        """
        self._cache = self._create_cache_tensor(self._num_samples)
        self._size = 0

    def prune(self, mask: np.ndarray) -> None:
        """
        Overview:
            Prunes the cache along the sample dimension using a boolean mask.
        Arguments:
            - mask (:obj:`np.ndarray`): A 1D boolean array where `True` indicates which samples to keep.
        """
        if not (mask.ndim == 1 and mask.shape[0] == self._num_samples):
            raise ValueError("Mask must be a 1D numpy array with length equal to the number of samples.")
        self._cache = self._cache[mask]
        self._num_samples = self._cache.shape[0]

    def get(self) -> torch.Tensor:
        """
        Overview:
            Retrieves the current contents of the cache.
        Returns:
            - cache_content (:obj:`torch.Tensor`): A tensor containing the valid data in the cache.
        """
        return self._cache[:, :, :self._size, :]

    def update(self, x: torch.Tensor, tokens: int) -> None:
        """
        Overview:
            Updates the cache with new tensor values. If the cache is full, it discards the oldest
            tokens to make space.
        Arguments:
            - x (:obj:`torch.Tensor`): The new tensor data to add to the cache.
            - tokens (:obj:`int`): The number of tokens being added (sequence length of `x`).
        """
        required_capacity = self._size + tokens

        # If the new tokens exceed the cache's maximum capacity, shift existing data to make room.
        if required_capacity > self._max_tokens:
            shift_amount = required_capacity - self._max_tokens

            # This logic is crucial for models like MuZero where tokens are added in (state, action) pairs.
            # To maintain the integrity of these pairs, an even number of tokens must be discarded.
            if shift_amount % 2 != 0:
                shift_amount += 1

            if shift_amount >= self._size:
                # If the required shift is larger than the current cache size, it's more efficient to reset.
                self._cache.zero_()
                self._size = 0
            else:
                # Shift the existing cache content to the left, discarding the oldest tokens.
                self._cache[:, :, :self._size - shift_amount, :] = self._cache[:, :, shift_amount:self._size, :]
                self._size -= shift_amount
                # NOTE: Shifting the cache invalidates absolute positional embeddings.
                # The parent model must handle positional encoding adjustments. For example, if positional
                # embeddings are calculated based on `prev_steps`, this shift means `prev_steps` may no
                # longer correspond to the true start, potentially causing discontinuities.

        # Use the custom autograd function to assign the new data without inplace errors.
        self._cache = AssignWithoutInplaceCheck.apply(
            self._cache, x, 2, self._size, self._size + tokens
        )
        self._size += tokens


class KVCache:
    """
    Overview:
        A container for a pair of caches: one for keys (K) and one for values (V),
        typically used in a single attention layer of a Transformer.
    """

    def __init__(self, num_samples: int, num_heads: int, max_tokens: int, embed_dim: int, device: torch.device) -> None:
        """
        Overview:
            Initializes the Key-Value cache pair.
        Arguments:
            - num_samples (:obj:`int`): The number of samples (batch size) to cache.
            - num_heads (:obj:`int`): The number of attention heads.
            - max_tokens (:obj:`int`): The maximum number of tokens the cache can hold.
            - embed_dim (:obj:`int`): The total dimension of the embeddings.
            - device (:obj:`torch.device`): The device on which to store the cache tensors.
        """
        self._k_cache = Cache(num_samples, num_heads, max_tokens, embed_dim, device)
        self._v_cache = Cache(num_samples, num_heads, max_tokens, embed_dim, device)

    @property
    def shape(self) -> Tuple[int, int, int, int]:
        """
        Overview:
            Gets the effective shape of the key cache's content.
        Returns:
            - shape (:obj:`Tuple[int, int, int, int]`): Shape of the key cache (num_samples, num_heads, current_size, head_dim).
        """
        return self._k_cache.shape

    def reset(self) -> None:
        """
        Overview:
            Resets both the key and value caches to their empty states.
        """
        self._k_cache.reset()
        self._v_cache.reset()

    def prune(self, mask: np.ndarray) -> None:
        """
        Overview:
            Prunes both key and value caches based on a boolean mask.
        Arguments:
            - mask (:obj:`np.ndarray`): A 1D boolean array indicating which samples to keep.
        """
        self._k_cache.prune(mask)
        self._v_cache.prune(mask)

    def get(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Overview:
            Retrieves the current contents of the key and value caches.
        Returns:
            - key_cache (:obj:`torch.Tensor`): The current contents of the key cache.
            - value_cache (:obj:`torch.Tensor`): The current contents of the value cache.
        """
        return self._k_cache.get(), self._v_cache.get()

    def update(self, k: torch.Tensor, v: torch.Tensor) -> None:
        """
        Overview:
            Updates both key and value caches with new tensors.
        Arguments:
            - k (:obj:`torch.Tensor`): The new key tensor to add.
            - v (:obj:`torch.Tensor`): The new value tensor to add.
        """
        # The number of tokens is inferred from the sequence dimension (dim 2).
        num_tokens = k.size(2)
        self._k_cache.update(k, num_tokens)
        self._v_cache.update(v, num_tokens)


class KeysValues:
    """
    Overview:
        Manages a collection of KVCache objects, one for each layer in a Transformer model.
    """

    def __init__(
            self,
            num_samples: int,
            num_heads: int,
            max_tokens: int,
            embed_dim: int,
            num_layers: int,
            device: torch.device
    ) -> None:
        """
        Overview:
            Initializes KV caches for all layers.
        Arguments:
            - num_samples (:obj:`int`): The number of samples (batch size).
            - num_heads (:obj:`int`): The number of attention heads.
            - max_tokens (:obj:`int`): The maximum number of tokens per cache.
            - embed_dim (:obj:`int`): The dimension of the embeddings.
            - num_layers (:obj:`int`): The number of layers in the Transformer model.
            - device (:obj:`torch.device`): The device for storing cache tensors.
        """
        self._keys_values = tuple([
            KVCache(num_samples, num_heads, max_tokens, embed_dim, device) for _ in range(num_layers)
        ])

    def __getitem__(self, layer_index: int) -> KVCache:
        """
        Overview:
            Retrieves the KVCache for a specific layer.
        Arguments:
            - layer_index (:obj:`int`): The index of the layer.
        Returns:
            - kv_cache (:obj:`KVCache`): The key-value cache for the specified layer.
        """
        return self._keys_values[layer_index]

    def __len__(self) -> int:
        """
        Overview:
            Gets the number of layers.
        Returns:
            - num_layers (:obj:`int`): The number of layers being managed.
        """
        return len(self._keys_values)

    @property
    def size(self) -> int:
        """
        Overview:
            Gets the current number of tokens stored in the caches.
        Returns:
            - size (:obj:`int`): The number of tokens in the cache (assumes all layers have the same size).
        """
        # All layer caches are synchronized, so we can check the size of the first one.
        if not self._keys_values:
            return 0
        return self._keys_values[0].shape[2]

    def reset(self) -> None:
        """
        Overview:
            Resets the KV caches for all layers.
        """
        for kv_cache in self._keys_values:
            kv_cache.reset()

    def prune(self, mask: np.ndarray) -> None:
        """
        Overview:
            Prunes the KV caches for all layers based on a mask.
        Arguments:
            - mask (:obj:`np.ndarray`): A boolean mask indicating which samples to keep.
        """
        for kv_cache in self._keys_values:
            kv_cache.prune(mask)

    def remove_register_tokens(self, register_token_num: int) -> None:
        """
        Overview:
            Removes the last `register_token_num` tokens from the active view of the cache
            in each layer by adjusting the internal size pointer. This does not delete the data
            but makes it invisible to subsequent `get` and `update` calls.
            This is typically called after an inference step that used temporary tokens
            (e.g., register tokens) to ensure they are not part of the ongoing context.
        Arguments:
            - register_token_num (:obj:`int`): The number of tokens to remove from the end of the cache view.
        """
        if register_token_num <= 0:
            return
        for kv_cache in self._keys_values:
            # Decrement the size pointer for both K and V caches.
            kv_cache._k_cache._size = max(0, kv_cache._k_cache._size - register_token_num)
            kv_cache._v_cache._size = max(0, kv_cache._v_cache._size - register_token_num)
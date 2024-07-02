# Modified from https://github.com/eloialonso/iris/blob/main/src/models/kv_caching.py

from typing import Tuple

import numpy as np
import torch


class Cache:
    def __init__(self, num_samples: int, num_heads: int, max_tokens: int, embed_dim: int, device: torch.device) -> None:
        """
        Overview:
            Cache for storing intermediate results in a transformer model.
        Arguments:
            - num_samples (:obj:`int`): The number of samples to cache.
            - num_heads (:obj:`int`): The number of attention heads.
            - max_tokens (:obj:`int`): The maximum number of tokens.
            - embed_dim (:obj:`int`): The dimension of the embeddings.
            - device (:obj:`torch.device`): The device on which to store the cache.
        """
        assert embed_dim % num_heads == 0
        self._n, self._cache, self._size = num_samples, None, None
        self._reset = lambda n: torch.empty(n, num_heads, max_tokens, embed_dim // num_heads,
                                            device=device)  # (B, nh, T, hs)
        self.reset()

    @property
    def shape(self) -> Tuple[int, int, int, int]:
        """
        Overview:
            Get the shape of the cache.
        Returns:
            - shape (:obj:`Tuple[int, int, int, int]`): The shape of the cache.
        """
        n, num_heads, _, head_dim = self._cache.shape
        return n, num_heads, self._size, head_dim

    def reset(self) -> None:
        """
        Overview:
            Reset the cache to its initial state.
        """
        self._cache = self._reset(self._n)
        self._size = 0

    def prune(self, mask: np.ndarray) -> None:
        """
        Overview:
            Prune the cache based on a mask.
        Arguments:
            - mask (:obj:`np.ndarray`): A boolean mask indicating which samples to keep.
        """
        assert mask.ndim == 1 and mask.shape[0] == self.shape[0]
        self._cache = self._cache[mask]
        self._n = self._cache.shape[0]

    def get(self) -> torch.Tensor:
        """
        Overview:
            Get the current contents of the cache.
        Returns:
            - cache (:obj:`torch.Tensor`): The current contents of the cache.
        """
        return self._cache[:, :, :self._size, :]

    def update(self, x: torch.Tensor) -> None:
        """
        Overview:
            Update the cache with new values.
        Arguments:
            - x (:obj:`torch.Tensor`): The new values to update the cache with.
        """
        assert (x.ndim == self._cache.ndim) and all([x.size(i) == self._cache.size(i) for i in (0, 1, 3)])
        assert self._size + x.size(2) <= self._cache.shape[2]  # TODO
        self._cache = AssignWithoutInplaceCheck.apply(self._cache, x, 2, self._size, self._size + x.size(2))
        self._size += x.size(2)


class KVCache:
    def __init__(self, n: int, num_heads: int, max_tokens: int, embed_dim: int, device: torch.device) -> None:
        """
        Overview:
            Cache for storing key and value tensors in a transformer model.
        Arguments:
            - n (:obj:`int`): The number of samples to cache.
            - num_heads (:obj:`int`): The number of attention heads.
            - max_tokens (:obj:`int`): The maximum number of tokens.
            - embed_dim (:obj:`int`): The dimension of the embeddings.
            - device (:obj:`torch.device`): The device on which to store the cache.
        """
        self._k_cache = Cache(n, num_heads, max_tokens, embed_dim, device)
        self._v_cache = Cache(n, num_heads, max_tokens, embed_dim, device)

    @property
    def shape(self) -> Tuple[int, int, int, int]:
        """
        Overview:
            Get the shape of the key cache.
        Returns:
            - shape (:obj:`Tuple[int, int, int, int]`): The shape of the key cache.
        """
        return self._k_cache.shape

    def reset(self) -> None:
        """
        Overview:
            Reset both key and value caches to their initial states.
        """
        self._k_cache.reset()
        self._v_cache.reset()

    def prune(self, mask: np.ndarray) -> None:
        """
        Overview:
            Prune both key and value caches based on a mask.
        Arguments:
            - mask (:obj:`np.ndarray`): A boolean mask indicating which samples to keep.
        """
        self._k_cache.prune(mask)
        self._v_cache.prune(mask)

    def get(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Overview:
            Get the current contents of the key and value caches.
        Returns:
            - key_cache (:obj:`torch.Tensor`): The current contents of the key cache.
            - value_cache (:obj:`torch.Tensor`): The current contents of the value cache.
        """
        return self._k_cache.get(), self._v_cache.get()

    def update(self, k: torch.Tensor, v: torch.Tensor):
        """
        Overview:
            Update both key and value caches with new values.
        Arguments:
            - k (:obj:`torch.Tensor`): The new values to update the key cache with.
            - v (:obj:`torch.Tensor`): The new values to update the value cache with.
        """
        self._k_cache.update(k)
        self._v_cache.update(v)


class KeysValues:
    def __init__(self, n: int, num_heads: int, max_tokens: int, embed_dim: int, num_layers: int, device: torch.device) -> None:
        """
        Overview:
            Class for managing multiple layers of key and value caches in a transformer model.
        Arguments:
            - n (:obj:`int`): The number of samples to cache.
            - num_heads (:obj:`int`): The number of attention heads.
            - max_tokens (:obj:`int`): The maximum number of tokens.
            - embed_dim (:obj:`int`): The dimension of the embeddings.
            - num_layers (:obj:`int`): The number of layers in the transformer model.
            - device (:obj:`torch.device`): The device on which to store the caches.
        """
        self._keys_values = tuple([KVCache(n, num_heads, max_tokens, embed_dim, device) for _ in range(num_layers)])

    def __getitem__(self, key: int) -> KVCache:
        """
        Overview:
            Get the key and value cache for a specific layer.
        Arguments:
            - key (:obj:`int`): The layer index.
        Returns:
            - kv_cache (:obj:`KVCache`): The key and value cache for the specified layer.
        """
        return self._keys_values[key]

    def __len__(self):
        """
        Overview:
            Get the number of layers in the transformer model.
        Returns:
            - length (:obj:`int`): The number of layers.
        """
        return len(self._keys_values)

    @property
    def size(self):
        """
        Overview:
            Get the size of the tokens in the cache.
        Returns:
            - size (:obj:`int`): The size of the tokens in the cache.
        """
        # TODO
        return self._keys_values[0].shape[2]
        # print([self._keys_values[layer].shape[2] for layer in range(len(self._keys_values))])

    def reset(self) -> None:
        """
        Overview:
            Reset all key and value caches to their initial states.
        """
        for kv_cache in self._keys_values:
            kv_cache.reset()

    def prune(self, mask: np.ndarray) -> None:
        """
        Overview:
            Prune all key and value caches based on a mask.
        Arguments:
            - mask (:obj:`np.ndarray`): A boolean mask indicating which samples to keep.
        """
        for kv_cache in self._keys_values:
            kv_cache.prune(mask)


class AssignWithoutInplaceCheck(torch.autograd.Function):
    """
    Overview:
        Custom autograd function to perform in-place assignment without triggering version checks.
    Inspired from:
        https://discuss.pytorch.org/t/disable-in-place-correctness-version-check-any-other-workaround/90738/4
    Warning:
        Do not use it to overwrite a slice twice.
    """

    @staticmethod
    def get_slice(dim: int, start: int, stop: int) -> Tuple[slice]:
        """
        Overview:
            Get the slice object for the given dimension and range.
        Arguments:
            - dim (:obj:`int`): The dimension along which to slice.
            - start (:obj:`int`): The start index of the slice.
            - stop (:obj:`int`): The stop index of the slice.
        Returns:
            - slice (:obj:`Tuple[slice]`): The slice object.
        """
        return tuple([slice(None), ] * dim + [slice(start, stop)])

    @staticmethod
    def forward(ctx, input: torch.Tensor, value: torch.Tensor, dim: int, start: int, stop: int) -> torch.Tensor:
        """
        Overview:
            Forward pass of the custom autograd function.
        Arguments:
            - ctx: The context object to store information for backward computation.
            - input (:obj:`torch.Tensor`): The input tensor to be modified.
            - value (:obj:`torch.Tensor`): The value tensor to assign to the input.
            - dim (:obj:`int`): The dimension along which to assign the value.
            - start (:obj:`int`): The start index of the assignment.
            - stop (:obj:`int`): The stop index of the assignment.
        Returns:
            - output (:obj:`torch.Tensor`): The modified input tensor.
        """
        ctx.dim = dim
        ctx.start = start
        ctx.stop = stop
        input.data[AssignWithoutInplaceCheck.get_slice(dim, start, stop)] = value
        return input

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Overview:
            Backward pass of the custom autograd function.
        Arguments:
            - ctx: The context object storing information from forward computation.
            - grad_out (:obj:`torch.Tensor`): The gradient of the output tensor.
        Returns:
            - grad_input (:obj:`torch.Tensor`): The gradient of the input tensor.
            - grad_value (:obj:`torch.Tensor`): The gradient of the value tensor.
        """
        return grad_out, grad_out[AssignWithoutInplaceCheck.get_slice(ctx.dim, ctx.start, ctx.stop)], None, None, None
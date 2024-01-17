import math
from typing import List

import torch
import torch.nn as nn

class Slicer(nn.Module):
    def __init__(self, max_blocks: int, block_mask: torch.Tensor) -> None:
        super().__init__()
        self.block_size = block_mask.size(0)
        self.num_kept_tokens = block_mask.sum().long().item()
        kept_indices = torch.where(block_mask)[0].repeat(max_blocks)
        offsets = torch.arange(max_blocks).repeat_interleave(self.num_kept_tokens)
        self.register_buffer('indices', kept_indices + block_mask.size(0) * offsets)
        self.cache: Dict[str, torch.Tensor] = {}
        self.precompute_slices()

    def precompute_slices(self) -> None:
        for num_steps in range(self.block_size*20):
            for prev_steps in range(self.block_size*20):
                cache_key = f"{num_steps}_{prev_steps}"
                total_steps = num_steps + prev_steps
                num_blocks = math.ceil(total_steps / self.block_size)
                indices = self.indices[:num_blocks * self.num_kept_tokens]
                result = indices[torch.logical_and(prev_steps <= indices, indices < total_steps)] - prev_steps
                self.cache[cache_key] = result

    def compute_slice(self, num_steps: int, prev_steps: int = 0) -> torch.Tensor:
        cache_key = f"{num_steps}_{prev_steps}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        else:
            # Handle the case where cache_key is not in self.cache
            # You could return a default value, raise an exception, or compute the result on the fly
            # For example, to raise an exception:
            raise ValueError(f"Cache key {cache_key} not found in precomputed slices")

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class Head(Slicer):
    def __init__(self, max_blocks: int, block_mask: torch.Tensor, head_module: nn.Module) -> None:
        super().__init__(max_blocks, block_mask)
        assert isinstance(head_module, nn.Module)
        self.head_module = head_module

    def forward(self, x: torch.Tensor, num_steps: int, prev_steps: int) -> torch.Tensor:
        x_sliced = x[:, self.compute_slice(num_steps, prev_steps)]  # x is (B, T, E)
        return self.head_module(x_sliced)


class Embedder(nn.Module):
    def __init__(self, max_blocks: int, block_masks: List[torch.Tensor], embedding_tables: List[nn.Embedding]) -> None:
        super().__init__()
        assert len(block_masks) == len(embedding_tables)
        assert (sum(block_masks) == 1).all()  # block mask are a partition of a block
        self.embedding_dim = embedding_tables[0].embedding_dim
        assert all([e.embedding_dim == self.embedding_dim for e in embedding_tables])
        self.embedding_tables = embedding_tables
        self.slicers = [Slicer(max_blocks, block_mask) for block_mask in block_masks]

    def forward(self, tokens: torch.Tensor, num_steps: int, prev_steps: int) -> torch.Tensor:
        assert tokens.ndim == 2  # x is (B, T)
        output = torch.zeros(*tokens.size(), self.embedding_dim, device=tokens.device)
        for slicer, emb in zip(self.slicers, self.embedding_tables):
            s = slicer.compute_slice(num_steps, prev_steps)
            output[:, s] = emb(tokens[:, s])
        return output
import torch
import pytest
from slicer import Slicer, Head, Embedder


def test_slicer_1():
    max_blocks = 20
    all_but_last_obs_tokens_pattern = torch.ones(17)
    all_but_last_obs_tokens_pattern[-2] = 0  # 1,...,0,1
    block_mask = all_but_last_obs_tokens_pattern

    slicer = Slicer(max_blocks, block_mask)

    # Test slice computation
    slice_ = slicer.compute_slice(5, 2)
    assert torch.equal(slice_, torch.tensor([0, 1, 2, 3, 4]))

    # Test caching
    cache_key = (5, 2)
    assert cache_key in slicer.cache
    assert torch.equal(slice_, slicer.cache[cache_key])


def test_slicer_2():
    max_blocks = 20
    act_tokens_pattern = torch.zeros(17)  # 17
    act_tokens_pattern[-1] = 1  # 0,...,0,1
    block_mask = act_tokens_pattern

    slicer = Slicer(max_blocks, block_mask)

    # Test slice computation
    slice_ = slicer.compute_slice(5, 2)
    assert torch.equal(slice_, torch.tensor([]))

    # Test caching
    cache_key = (5, 2)
    assert cache_key in slicer.cache
    assert torch.equal(slice_, slicer.cache[cache_key])


def test_head():
    max_blocks = 20

    all_but_last_obs_tokens_pattern = torch.ones(17)
    all_but_last_obs_tokens_pattern[-2] = 0  # 1,...,0,1
    block_mask = all_but_last_obs_tokens_pattern

    head_module = torch.nn.Linear(5, 2)
    head = Head(max_blocks, block_mask, head_module)

    x = torch.randn((2, 20, 5))
    output = head(x, 5, 2)
    assert output.shape == (2, 5, 2)


def test_embedder():
    max_blocks = 20

    act_tokens_pattern = torch.zeros(17)  # 17
    act_tokens_pattern[-1] = 1  # [0,...,0,1]
    obs_tokens_pattern = 1 - act_tokens_pattern  # [1,...,1,0]
    block_masks = [act_tokens_pattern, obs_tokens_pattern]

    embedding_tables = [torch.nn.Embedding(128, 3), torch.nn.Embedding(128, 3)]
    embedder = Embedder(max_blocks, block_masks, embedding_tables)

    tokens = torch.randint(0, 128, (2, 20))
    output = embedder(tokens, 5, 2)
    assert output.shape == (2, 20, 3)


if __name__ == "__main__":
    test_slicer_1()
    test_slicer_2()
    test_head()
    test_embedder()


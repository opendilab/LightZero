import torch
import pytest
from slicer import Slicer, Head, Embedder


def test_slicer_time():
    max_blocks = 20
    act_tokens_pattern = torch.zeros(17)  # 17
    act_tokens_pattern[-1] = 1   # 0,...,0,1
    block_mask = act_tokens_pattern

    slicer = Slicer(max_blocks, block_mask)

    import timeit
    start_time = timeit.default_timer()
    # code you want to evaluate
    # Test slice computation
    slice_ = slicer.compute_slice(num_steps=5,  prev_steps=2)
    
    end_time = timeit.default_timer()
    execution_time = end_time - start_time
    print(f"Executed the function in {execution_time} seconds")

    assert torch.equal(slice_, torch.tensor([]))
    
    # Test caching
    cache_key = (5, 2)
    assert cache_key in slicer.cache
    assert torch.equal(slice_, slicer.cache[cache_key])



if __name__ == "__main__":
    test_slicer_time()


import pytest
import torch
from lzero.policy.scaling_transform import inverse_scalar_transform, InverseScalarTransform


@pytest.mark.unittest
def test_scaling_transform():
    import time
    logit = torch.randn(16, 601)
    start = time.time()
    output_1 = inverse_scalar_transform(logit, 300)
    print('t1', time.time() - start)
    handle = InverseScalarTransform(300)
    start = time.time()
    output_2 = handle(logit)
    print('t2', time.time() - start)
    assert output_1.shape == output_2.shape == (16, 1)
    assert (output_1 == output_2).all()
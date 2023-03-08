import pytest
import torch

from lzero.policy.utlis import negative_cosine_similarity


@pytest.mark.unittest
class TestUtils():

    def test_negative_cosine_similarity(self):
        batch_size = 256
        dim = 512
        x1 = torch.randn(batch_size, dim)
        x2 = torch.randn(batch_size, dim)
        output = negative_cosine_similarity(x1, x2)
        assert output.shape == (batch_size, )
        assert ((output >= -1) & (output <= 1)).all()

        x1 = torch.randn(batch_size, dim)
        positive_factor = torch.randint(1, 100, [1])
        output_positive = negative_cosine_similarity(x1, positive_factor.float() * x1)
        assert output_positive.shape == (batch_size, )
        # assert (output_negative == -1).all()  # is not True, because of the numerical precision
        assert ((output_positive - (-1)) < 1e-6).all()

        negative_factor = - torch.randint(1, 100, [1])
        output_negative = negative_cosine_similarity(x1, negative_factor.float() * x1)
        assert output_negative .shape == (batch_size, )
        # assert (output_negative == 1).all()
        # assert (output_negative == 1).all()  # is not True, because of the numerical precision
        assert ((output_positive - 1) < 1e-6).all()


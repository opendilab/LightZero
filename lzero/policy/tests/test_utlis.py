import pytest
import torch
import numpy as np

from lzero.policy.utils import negative_cosine_similarity, to_torch_float_tensor


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

    def test_to_torch_float_tensor(self):
        device = 'cpu'
        mask_batch_np, target_value_prefix_np, target_value_np, target_policy_np, weights_np = np.random.randn(4,
                                                                                                               5), np.random.randn(
            4, 5), np.random.randn(4, 5), np.random.randn(4, 5), np.random.randn(4, 5)
        data_list_np = [mask_batch_np, target_value_prefix_np.astype('float64'), target_value_np.astype('float64'),
                        target_policy_np, weights_np]
        [mask_batch_func, target_value_prefix_func, target_value_func, target_policy_func,
         weights_func] = to_torch_float_tensor(data_list_np,
                                               device)
        mask_batch_2 = torch.from_numpy(mask_batch_np).to(device).float()
        target_value_prefix_2 = torch.from_numpy(target_value_prefix_np.astype('float64')).to(device).float()
        target_value_2 = torch.from_numpy(target_value_np.astype('float64')).to(device).float()
        target_policy_2 = torch.from_numpy(target_policy_np).to(device).float()
        weights_2 = torch.from_numpy(weights_np).to(device).float()

        assert (mask_batch_func == mask_batch_2).all() and (
                    target_value_prefix_func == target_value_prefix_2).all() and (
                           target_value_func == target_value_2).all() and (
                           target_policy_func == target_policy_2).all() and (weights_func == weights_2).all()

import numpy as np
import pytest
import torch

from lzero.mcts.utils import to_torch_float_tensor


@pytest.mark.unittest
class TestUtils():

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

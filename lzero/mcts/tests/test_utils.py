import numpy as np
import pytest
import torch

from lzero.mcts.utils import to_torch_float_tensor,  get_augmented_data


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

    def test_get_augmented_data(self):
        num_of_data = 100
        board_size = 15
        state = np.random.randint(0, 3, (board_size, board_size, 3), dtype=np.uint8)
        mcts_prob = np.random.randn(board_size, board_size)
        winner = np.random.randint(0, 2, 1, dtype=np.uint8)
        play_data = [{'state': state, 'mcts_prob': mcts_prob, 'winner': winner} for _ in range(num_of_data)]

        extented_data = get_augmented_data(board_size, play_data)
        assert len(extented_data) == num_of_data * 8
        # TODO(pu): extented data shape is not same as original data?
        # assert extented_data[0]['state'].shape == state.shape
        assert extented_data[0]['state'].flatten().shape == state.flatten().shape
        assert extented_data[0]['mcts_prob'].shape == mcts_prob.flatten().shape
        assert extented_data[0]['winner'].shape == winner.shape
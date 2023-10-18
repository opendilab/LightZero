import numpy as np
import pytest
import torch
import torch.nn.functional as F

from lzero.policy.utils import negative_cosine_similarity, to_torch_float_tensor, visualize_avg_softmax, \
    calculate_topk_accuracy, plot_topk_accuracy, compare_argmax, plot_argmax_distribution


# We use the pytest.mark.unittest decorator to mark this class for unit testing.
@pytest.mark.unittest
class TestVisualizationFunctions:

    def test_visualize_avg_softmax(self):
        """
        This test checks whether the visualize_avg_softmax function correctly
        computes the average softmax probabilities and visualizes them.
        """

        # We initialize the input parameters.
        batch_size = 256
        num_classes = 10
        logits = torch.randn(batch_size, num_classes)

        # We call the visualize_avg_softmax function.
        visualize_avg_softmax(logits)

        # This function does not return anything, it only creates a plot.
        # Therefore, we can only visually inspect the plot to check if it is correct.

    def test_calculate_topk_accuracy(self):
        """
        This test checks whether the calculate_topk_accuracy function correctly
        computes the top-k accuracy.
        """

        # We initialize the input parameters.
        batch_size = 256
        num_classes = 10
        logits = torch.randn(batch_size, num_classes)
        true_labels = torch.randint(0, num_classes, [batch_size])
        true_one_hot = F.one_hot(true_labels, num_classes)
        top_k = 5

        # We call the calculate_topk_accuracy function.
        match_percentage = calculate_topk_accuracy(logits, true_one_hot, top_k)

        # We check if the match percentage is a float and within the range [0, 100].
        assert isinstance(match_percentage, float)
        assert 0 <= match_percentage <= 100

    def test_plot_topk_accuracy(self):
        """
        This test checks whether the plot_topk_accuracy function correctly
        plots the top-k accuracy for different values of k.
        """

        # We initialize the input parameters.
        batch_size = 256
        num_classes = 10
        logits = torch.randn(batch_size, num_classes)
        true_labels = torch.randint(0, num_classes, [batch_size])
        true_one_hot = F.one_hot(true_labels, num_classes)
        top_k_values = range(1, 6)

        # We call the plot_topk_accuracy function.
        plot_topk_accuracy(logits, true_one_hot, top_k_values)

        # This function does not return anything, it only creates a plot.
        # Therefore, we can only visually inspect the plot to check if it is correct.

    def test_compare_argmax(self):
        """
        This test checks whether the compare_argmax function correctly
        plots the comparison of argmax values.
        """

        # We initialize the input parameters.
        batch_size = 256
        num_classes = 10
        logits = torch.randn(batch_size, num_classes)
        true_labels = torch.randint(0, num_classes, [batch_size])
        chance_one_hot = F.one_hot(true_labels, num_classes)

        # We call the compare_argmax function.
        compare_argmax(logits, chance_one_hot)

        # This function does not return anything, it only creates a plot.
        # Therefore, we can only visually inspect the plot to check if it is correct.

    def test_plot_argmax_distribution(self):
        """
        This test checks whether the plot_argmax_distribution function correctly
        plots the distribution of argmax values.
        """

        # We initialize the input parameters.
        batch_size = 256
        num_classes = 10
        true_labels = torch.randint(0, num_classes, [batch_size])
        true_chance_one_hot = F.one_hot(true_labels, num_classes)

        # We call the plot_argmax_distribution function.
        plot_argmax_distribution(true_chance_one_hot)

        # This function does not return anything, it only creates a plot.
        # Therefore, we can only visually inspect the plot to check if it is correct.


# We use the pytest.mark.unittest decorator to mark this class for unit testing.
@pytest.mark.unittest
class TestUtils():

    # This function tests the negative_cosine_similarity function.
    # This function computes the negative cosine similarity between two vectors.
    def test_negative_cosine_similarity(self):
        # We initialize the input parameters.
        batch_size = 256
        dim = 512
        x1 = torch.randn(batch_size, dim)
        x2 = torch.randn(batch_size, dim)

        # We call the negative_cosine_similarity function.
        output = negative_cosine_similarity(x1, x2)

        # We check if the output shape is as expected.
        assert output.shape == (batch_size, )

        # We check if all elements of the output are between -1 and 1.
        assert ((output >= -1) & (output <= 1)).all()

        # We test a special case where the two input vectors are in the same direction.
        # In this case, the cosine similarity should be -1.
        x1 = torch.randn(batch_size, dim)
        positive_factor = torch.randint(1, 100, [1])
        output_positive = negative_cosine_similarity(x1, positive_factor.float() * x1)
        assert output_positive.shape == (batch_size, )
        assert ((output_positive - (-1)) < 1e-6).all()

        # We test another special case where the two input vectors are in opposite directions.
        # In this case, the cosine similarity should be 1.
        negative_factor = -torch.randint(1, 100, [1])
        output_negative = negative_cosine_similarity(x1, negative_factor.float() * x1)
        assert output_negative.shape == (batch_size, )
        assert ((output_positive - 1) < 1e-6).all()

    def test_to_torch_float_tensor(self):
        device = 'cpu'
        mask_batch_np, target_value_prefix_np, target_value_np, target_policy_np, weights_np = np.random.randn(
            4, 5
        ), np.random.randn(4, 5), np.random.randn(4, 5), np.random.randn(4, 5), np.random.randn(4, 5)
        data_list_np = [
            mask_batch_np,
            target_value_prefix_np.astype('float32'),
            target_value_np.astype('float32'), target_policy_np, weights_np
        ]
        [mask_batch_func, target_value_prefix_func, target_value_func, target_policy_func,
         weights_func] = to_torch_float_tensor(data_list_np, device)
        mask_batch_2 = torch.from_numpy(mask_batch_np).to(device).float()
        target_value_prefix_2 = torch.from_numpy(target_value_prefix_np.astype('float32')).to(device).float()
        target_value_2 = torch.from_numpy(target_value_np.astype('float32')).to(device).float()
        target_policy_2 = torch.from_numpy(target_policy_np).to(device).float()
        weights_2 = torch.from_numpy(weights_np).to(device).float()

        assert (mask_batch_func == mask_batch_2).all() and (target_value_prefix_func == target_value_prefix_2).all(
        ) and (target_value_func == target_value_2).all() and (target_policy_func == target_policy_2
                                                               ).all() and (weights_func == weights_2).all()

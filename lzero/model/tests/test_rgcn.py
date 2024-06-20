import torch
import torch.nn as nn
from torch.nn import functional as F
from itertools import product
import unittest
from lzero.model.common_gcn import RGCNLayer


class TestRGCNLayer(unittest.TestCase):

    def setUp(self):
        self.robot_state_dim = 10
        self.human_state_dim = 10
        self.similarity_function = 'embedded_gaussian'
        self.batch_size = 4
        self.num_nodes = 5  # Suppose 5 robots and 5 humans

        # Create a RGCNLayer object
        self.rgcn_layer = RGCNLayer(
            robot_state_dim=self.robot_state_dim,
            human_state_dim=self.human_state_dim,
            similarity_function=self.similarity_function,
            num_layer=2,
            X_dim=32,
            layerwise_graph=False,
            skip_connection=True
        )

        # Creating dummy inputs
        self.state = {
            'robot_state': torch.randn(self.batch_size, self.num_nodes, self.robot_state_dim),
            'human_state': torch.randn(self.batch_size, self.num_nodes, self.human_state_dim)
        }

    def test_forward_shape(self):
        # Forward pass
        output = self.rgcn_layer(self.state)
        expected_shape = (self.batch_size, self.num_nodes * 2, 32)  # Since final_state_dim is set to X_dim
        self.assertEqual(output.shape, expected_shape, "Output shape is incorrect.")

    def test_similarity_function(self):
        # Check if the similarity matrix computation is working as expected
        # This only checks for one similarity function due to space constraints
        if self.similarity_function == 'embedded_gaussian':
            X = torch.randn(self.batch_size, self.num_nodes * 2, 32)
            A = self.rgcn_layer.compute_similarity_matrix(X)
            self.assertEqual(
                A.shape, (self.batch_size, self.num_nodes * 2, self.num_nodes * 2),
                "Similarity matrix shape is incorrect."
            )
            self.assertTrue(torch.all(A >= 0) and torch.all(A <= 1), "Similarity matrix values should be normalized.")


# Running the tests
if __name__ == '__main__':
    unittest.main()

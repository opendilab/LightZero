import torch
import numpy as np
from torch import nn
from lzero.model.common_gcn import RepresentationNetworkGCN, RGCNLayer

# ...

class TestLightZeroEnvWrapper:

    # ...
    def test_representation_network_gcn_with_dict_obs(self):
        robot_state_dim = 10
        human_state_dim = 5
        robot_num = 3
        human_num = 2
        hidden_channels = 64
        layer_num = 2
        activation = nn.ReLU(inplace=True)
        last_linear_layer_init_zero = True
        norm_type = 'BN'

        representation_network = RepresentationNetworkGCN(
            robot_state_dim=robot_state_dim,
            human_state_dim=human_state_dim,
            robot_num=robot_num,
            human_num=human_num,
            hidden_channels=hidden_channels,
            layer_num=layer_num,
            activation=activation,
            last_linear_layer_init_zero=last_linear_layer_init_zero,
            norm_type=norm_type,
        )

        # Create dummy input
        batch_size = 4
        x = {
            'robot_state': torch.randn(batch_size, robot_num, robot_state_dim),
            'human_state': torch.randn(batch_size, human_num, human_state_dim)
        }

        # Forward pass
        output = representation_network(x)

        # Check output shape
        assert output.shape == (batch_size, hidden_channels)

        # Check output type
        assert isinstance(output, torch.Tensor)

        # Check intermediate shape
        assert representation_network.rgcn(x).shape == (batch_size, robot_num + human_num, hidden_channels)

        # Check intermediate type
        assert isinstance(representation_network.rgcn(x), torch.Tensor)

    def test_representation_network_gcn_with_2d_array_obs(self):
        robot_state_dim = 10
        human_state_dim = 10    # 2d_array_obs, so the dimensions must be the same
        robot_num = 3
        human_num = 2
        hidden_channels = 64
        layer_num = 2
        activation = nn.ReLU(inplace=True)
        last_linear_layer_init_zero = True
        norm_type = 'BN'

        representation_network = RepresentationNetworkGCN(
            robot_state_dim=robot_state_dim,
            human_state_dim=human_state_dim,
            robot_num=robot_num,
            human_num=human_num,
            hidden_channels=hidden_channels,
            layer_num=layer_num,
            activation=activation,
            last_linear_layer_init_zero=last_linear_layer_init_zero,
            norm_type=norm_type,
        )

        # Create dummy input
        batch_size = 4
        x = torch.randn(batch_size, robot_num + human_num, robot_state_dim)

        # Forward pass
        output = representation_network(x)

        # Check output shape
        assert output.shape == (batch_size, hidden_channels)

        # Check output type
        assert isinstance(output, torch.Tensor)

        # Check intermediate shape
        assert representation_network.rgcn(x).shape == (batch_size, robot_num + human_num, hidden_channels)

        # Check intermediate type
        assert isinstance(representation_network.rgcn(x), torch.Tensor)

if __name__ == '__main__':
    test = TestLightZeroEnvWrapper()
    test.test_representation_network_gcn_with_dict_obs()
    test.test_representation_network_gcn_with_2d_array_obs()
    print("All tests passed.")
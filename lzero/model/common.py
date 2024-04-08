"""
Overview:
    In this Python file, we provide a collection of reusable model templates designed to streamline the development
    process for various custom algorithms. By utilizing these pre-built model templates, users can quickly adapt and
    customize their custom algorithms, ensuring efficient and effective development.
    BTW, users can refer to the unittest of these model templates to learn how to use them.
"""
import math
from typing import Optional, Tuple
from dataclasses import dataclass
import logging
import itertools
import numpy as np
import torch
import torch.nn as nn
from ding.torch_utils import MLP, ResBlock
from ding.utils import SequenceType


# use dataclass to make the output of network more convenient to use
@dataclass
class EZNetworkOutput:
    # output format of the EfficientZero model
    value: torch.Tensor
    value_prefix: torch.Tensor
    policy_logits: torch.Tensor
    latent_state: torch.Tensor
    reward_hidden_state: Tuple[torch.Tensor]


@dataclass
class MZNetworkOutput:
    # output format of the MuZero model
    value: torch.Tensor
    reward: torch.Tensor
    policy_logits: torch.Tensor
    latent_state: torch.Tensor


class DownSample(nn.Module):
            
    def __init__(self, observation_shape: SequenceType, out_channels: int, activation: nn.Module = nn.ReLU(inplace=True),
                 norm_type: Optional[str] = 'BN',
                 ) -> None:
        """
        Overview:
            Define downSample convolution network. Encode the observation into hidden state.
            This network is often used in video games like Atari. In board games like go and chess,
            we don't need this module.
        Arguments:
            - observation_shape (:obj:`SequenceType`): The shape of observation space, e.g. [C, W, H]=[12, 96, 96]
                for video games like atari, RGB 3 channel times stack 4 frames.
            - out_channels (:obj:`int`): The output channels of output hidden state.
            - activation (:obj:`nn.Module`): The activation function used in network, defaults to nn.ReLU(). \
                Use the inplace operation to speed up.
            - norm_type (:obj:`Optional[str]`): The normalization type used in network, defaults to 'BN'. 
        """
        super().__init__()
        assert norm_type in ['BN', 'LN'], "norm_type must in ['BN', 'LN']"

        self.conv1 = nn.Conv2d(
            observation_shape[0],
            out_channels // 2,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,  # disable bias for better convergence
        )
        if norm_type == 'BN':
            self.norm1 = nn.BatchNorm2d(out_channels // 2)
        elif norm_type == 'LN':
            self.norm1 = nn.LayerNorm([out_channels // 2, observation_shape[-2] // 2, observation_shape[-1] // 2])

        self.resblocks1 = nn.ModuleList(
            [
                ResBlock(
                    in_channels=out_channels // 2,
                    activation=activation,
                    norm_type='BN',
                    res_type='basic',
                    bias=False
                ) for _ in range(1)
            ]
        )
        self.downsample_block = ResBlock(
            in_channels=out_channels // 2,
            out_channels=out_channels,
            activation=activation,
            norm_type='BN',
            res_type='downsample',
            bias=False
        )
        self.resblocks2 = nn.ModuleList(
            [
                ResBlock(
                    in_channels=out_channels, activation=activation, norm_type='BN', res_type='basic', bias=False
                ) for _ in range(1)
            ]
        )
        self.pooling1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.resblocks3 = nn.ModuleList(
            [
                ResBlock(
                    in_channels=out_channels, activation=activation, norm_type='BN', res_type='basic', bias=False
                ) for _ in range(1)
            ]
        )
        self.pooling2 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Shapes:
            - x (:obj:`torch.Tensor`): :math:`(B, C_in, W, H)`, where B is batch size, C_in is channel, W is width, \
                H is height.
            - output (:obj:`torch.Tensor`): :math:`(B, C_out, W_, H_)`, where B is batch size, C_out is channel, W_ is \
                output width, H_ is output height.
        """
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)

        for block in self.resblocks1:
            x = block(x)
        x = self.downsample_block(x)
        for block in self.resblocks2:
            x = block(x)
        x = self.pooling1(x)
        for block in self.resblocks3:
            x = block(x)
        output = self.pooling2(x)
        return output


class RepresentationNetwork(nn.Module):

    def __init__(
            self,
            observation_shape: SequenceType = (12, 96, 96),
            num_res_blocks: int = 1,
            num_channels: int = 64,
            downsample: bool = True,
            activation: nn.Module = nn.ReLU(inplace=True),
            norm_type: str = 'BN',
    ) -> None:
        """
        Overview:
            Representation network used in MuZero and derived algorithms. Encode the 2D image obs into hidden state.
        Arguments:
            - observation_shape (:obj:`SequenceType`): The shape of observation space, e.g. [C, W, H]=[12, 96, 96]
                for video games like atari, RGB 3 channel times stack 4 frames.
            - num_res_blocks (:obj:`int`): The number of residual blocks.
            - num_channels (:obj:`int`): The channel of output hidden state.
            - downsample (:obj:`bool`): Whether to do downsampling for observations in ``representation_network``, \
                defaults to True. This option is often used in video games like Atari. In board games like go, \
                we don't need this module.
            - activation (:obj:`nn.Module`): The activation function used in network, defaults to nn.ReLU(). \
                Use the inplace operation to speed up.
            - norm_type (:obj:`str`): The type of normalization in networks. defaults to 'BN'.
        """
        super().__init__()
        assert norm_type in ['BN', 'LN'], "norm_type must in ['BN', 'LN']"

        self.downsample = downsample
        if self.downsample:
            self.downsample_net = DownSample(
                observation_shape,
                num_channels,
                activation=activation,
                norm_type=norm_type,
            )
        else:
            self.conv = nn.Conv2d(observation_shape[0], num_channels, kernel_size=3, stride=1, padding=1, bias=False)

            if norm_type == 'BN':
                self.norm = nn.BatchNorm2d(num_channels)
            elif norm_type == 'LN':
                if downsample:
                    self.norm = nn.LayerNorm([num_channels, math.ceil(observation_shape[-2] / 16), math.ceil(observation_shape[-1] / 16)])
                else:
                    self.norm = nn.LayerNorm([num_channels, observation_shape[-2], observation_shape[-1]])
            
        self.resblocks = nn.ModuleList(
            [
                ResBlock(
                    in_channels=num_channels, activation=activation, norm_type='BN', res_type='basic', bias=False
                ) for _ in range(num_res_blocks)
            ]
        )
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Shapes:
            - x (:obj:`torch.Tensor`): :math:`(B, C_in, W, H)`, where B is batch size, C_in is channel, W is width, \
                H is height.
            - output (:obj:`torch.Tensor`): :math:`(B, C_out, W_, H_)`, where B is batch size, C_out is channel, W_ is \
                output width, H_ is output height.
        """
        if self.downsample:
            x = self.downsample_net(x)
        else:
            x = self.conv(x)
            x = self.norm(x)
            x = self.activation(x)

        for block in self.resblocks:
            x = block(x)
        return x

    def get_param_mean(self) -> float:
        """
        Overview:
            Get the mean of parameters in the network for debug and visualization.
        Returns:
            - mean (:obj:`float`): The mean of parameters in the network.
        """
        mean = []
        for name, param in self.named_parameters():
            mean += np.abs(param.detach().cpu().numpy().reshape(-1)).tolist()
        mean = sum(mean) / len(mean)
        return mean


class RepresentationNetworkMLP(nn.Module):

    def __init__(
            self,
            observation_shape: int,
            hidden_channels: int = 64,
            layer_num: int = 2,
            activation: Optional[nn.Module] = nn.ReLU(inplace=True),
            last_linear_layer_init_zero: bool = True,
            norm_type: Optional[str] = 'BN',
    ) -> torch.Tensor:
        """
        Overview:
            Representation network used in MuZero and derived algorithms. Encode the vector obs into latent state \
                with Multi-Layer Perceptron (MLP).
        Arguments:
            - observation_shape (:obj:`int`): The shape of vector observation space, e.g. N = 10.
            - num_res_blocks (:obj:`int`): The number of residual blocks.
            - hidden_channels (:obj:`int`): The channel of output hidden state.
            - downsample (:obj:`bool`): Whether to do downsampling for observations in ``representation_network``, \
                defaults to True. This option is often used in video games like Atari. In board games like go, \
                we don't need this module.
            - activation (:obj:`nn.Module`): The activation function used in network, defaults to nn.ReLU(). \
                Use the inplace operation to speed up.
            - last_linear_layer_init_zero (:obj:`bool`): Whether to initialize the last linear layer with zeros, \
                which can provide stable zero outputs in the beginning, defaults to True.
            - norm_type (:obj:`str`): The type of normalization in networks. defaults to 'BN'.
        """
        super().__init__()
        self.fc_representation = MLP(
            in_channels=observation_shape,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            layer_num=layer_num,
            activation=activation,
            norm_type=norm_type,
            # don't use activation and norm in the last layer of representation network is important for convergence.
            output_activation=False,
            output_norm=False,
            # last_linear_layer_init_zero=True is beneficial for convergence speed.
            last_linear_layer_init_zero=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Shapes:
            - x (:obj:`torch.Tensor`): :math:`(B, N)`, where B is batch size, N is the length of vector observation.
            - output (:obj:`torch.Tensor`): :math:`(B, hidden_channels)`, where B is batch size.
        """
        return self.fc_representation(x)

class RGCNLayer(nn.Module):
    """
    Overview:
        Relational graph convolutional network layer.
    """
    def __init__(
            self,
            robot_state_dim,
            human_state_dim,
            similarity_function,
            num_layer = 2,
            X_dim = 32,
            layerwise_graph = False,
            skip_connection = True,
            wr_dims = [64, 32],     # the last dim should equal to X_dim
            wh_dims = [64, 32],     # the last dim should equal to X_dim
            final_state_dim = 32,   # should equal to X_dim
            norm_type= 'BN',
            last_linear_layer_init_zero=True,
            activation: Optional[nn.Module] = nn.ReLU(inplace=True),
            ):
        super().__init__()

        # design choice
        # 'gaussian', 'embedded_gaussian', 'cosine', 'cosine_softmax', 'concatenation'
        self.similarity_function = similarity_function
        self.robot_state_dim = robot_state_dim
        self.human_state_dim = human_state_dim
        self.num_layer = num_layer
        self.X_dim = X_dim
        self.layerwise_graph = layerwise_graph
        self.skip_connection = skip_connection

        logging.info('Similarity_func: {}'.format(self.similarity_function))
        logging.info('Layerwise_graph: {}'.format(self.layerwise_graph))
        logging.info('Skip_connection: {}'.format(self.skip_connection))
        logging.info('Number of layers: {}'.format(self.num_layer))

        self.w_r = MLP(
            in_channels=robot_state_dim,
            hidden_channels=wr_dims[0],
            out_channels=wr_dims[1],
            layer_num=num_layer,
            activation=activation,
            norm_type=norm_type,
            last_linear_layer_init_zero=last_linear_layer_init_zero,
            )  # inputs,64,32
        self.w_h = MLP(
            in_channels=human_state_dim,
            hidden_channels=wh_dims[0],
            out_channels=wh_dims[1],
            layer_num=num_layer,
            activation=activation,
            norm_type=norm_type,
            last_linear_layer_init_zero=last_linear_layer_init_zero,
            )  # inputs,64,32

        if self.similarity_function == 'embedded_gaussian':
            self.w_a = nn.Parameter(torch.randn(self.X_dim, self.X_dim))
        elif self.similarity_function == 'concatenation':
            # TODO: fix the dim size
            self.w_a = MLP(
                in_channels=2 * X_dim,
                hidden_channels=2 * X_dim,
                out_channels=1,
                layer_num=1,
                )

        embedding_dim = self.X_dim
        self.Ws = torch.nn.ParameterList()
        for i in range(self.num_layer):
            if i == 0:
                self.Ws.append(nn.Parameter(torch.randn(self.X_dim, embedding_dim)))
            elif i == self.num_layer - 1:
                self.Ws.append(nn.Parameter(torch.randn(embedding_dim, final_state_dim)))
            else:
                self.Ws.append(nn.Parameter(torch.randn(embedding_dim, embedding_dim)))

        # TODO: for visualization
        self.A = None

    def compute_similarity_matrix(self, X):
        if self.similarity_function == 'embedded_gaussian':
            A = torch.matmul(torch.matmul(X, self.w_a), X.permute(0, 2, 1))
            normalized_A = nn.functional.softmax(A, dim=2)
        elif self.similarity_function == 'gaussian':
            A = torch.matmul(X, X.permute(0, 2, 1))
            normalized_A = nn.functional.softmax(A, dim=2)
        elif self.similarity_function == 'cosine':
            A = torch.matmul(X, X.permute(0, 2, 1))
            magnitudes = torch.norm(A, dim=2, keepdim=True)
            norm_matrix = torch.matmul(magnitudes, magnitudes.permute(0, 2, 1))
            normalized_A = torch.div(A, norm_matrix)
        elif self.similarity_function == 'cosine_softmax':
            A = torch.matmul(X, X.permute(0, 2, 1))
            magnitudes = torch.norm(A, dim=2, keepdim=True)
            norm_matrix = torch.matmul(magnitudes, magnitudes.permute(0, 2, 1))
            normalized_A = nn.functional.softmax(torch.div(A, norm_matrix), dim=2)
        elif self.similarity_function == 'concatenation':
            indices = [pair for pair in itertools.product(list(range(X.size(1))), repeat=2)]
            selected_features = torch.index_select(X, dim=1, index=torch.LongTensor(indices).reshape(-1))
            pairwise_features = selected_features.reshape((-1, X.size(1) * X.size(1), X.size(2) * 2))
            A = self.w_a(pairwise_features).reshape(-1, X.size(1), X.size(1))
            normalized_A = A
        elif self.similarity_function == 'squared':
            A = torch.matmul(X, X.permute(0, 2, 1))
            squared_A = A * A
            normalized_A = squared_A / torch.sum(squared_A, dim=2, keepdim=True)
        elif self.similarity_function == 'equal_attention':
            normalized_A = (torch.ones(X.size(1), X.size(1)) / X.size(1)).expand(X.size(0), X.size(1), X.size(1))
        elif self.similarity_function == 'diagonal':
            normalized_A = (torch.eye(X.size(1), X.size(1))).expand(X.size(0), X.size(1), X.size(1))
        else:
            raise NotImplementedError

        return normalized_A

    def forward(self, state):
        robot_states = state['robot_state']
        human_states = state['human_state']

        # compute feature matrix X
        robot_state_embedings = self.w_r(robot_states)  # batch x num x embedding_dim
        human_state_embedings = self.w_h(human_states)
        X = torch.cat([robot_state_embedings, human_state_embedings], dim=1)

        # compute matrix A
        if not self.layerwise_graph:
            normalized_A = self.compute_similarity_matrix(X)
            self.A = normalized_A[0, :, :].data.cpu().numpy()  # total_num x total_num

        # next_H = H = X

        H = X.contiguous().clone()
        next_H = H.contiguous().clone()  # batch x total_num x embedding_dim
        for i in range(self.num_layer):  # 2
            if self.layerwise_graph:  # False
                A = self.compute_similarity_matrix(H)
                next_H = nn.functional.relu(torch.matmul(torch.matmul(A, H), self.Ws[i]))
            else:  # (A x H) x W_i
                next_H = nn.functional.relu(torch.matmul(torch.matmul(normalized_A, H), self.Ws[i]))

            if self.skip_connection:
                # next_H += H
                next_H = next_H + H
            H = next_H.contiguous().clone()

        return next_H

class RepresentationNetworkGCN(nn.Module):

    def __init__(
            self,
            robot_observation_shape: tuple,
            human_observation_shape: tuple,
            hidden_channels: int = 64,
            layer_num: int = 2,
            activation: Optional[nn.Module] = nn.ReLU(inplace=True),
            last_linear_layer_init_zero: bool = True,
            norm_type: Optional[str] = 'BN',
    ) -> torch.Tensor:
        """
        Overview:
            
        Arguments:
            - robot_observation_shape (:obj:`tuple`): The shape of robot observation space, e.g. (2, 4).
            - human_observation_shape (:obj:`tuple`): The shape of human observation space, e.g. (59, 4).
            - hidden_channels (:obj:`int`): The channel of output hidden state.
            - activation (:obj:`nn.Module`): The activation function used in network, defaults to nn.ReLU(). \
                Use the inplace operation to speed up.
            - last_linear_layer_init_zero (:obj:`bool`): Whether to initialize the last linear layer with zeros, \
                which can provide stable zero outputs in the beginning, defaults to True.
            - norm_type (:obj:`str`): The type of normalization in networks. defaults to 'BN'.
        """
        super().__init__()
        # self.fc_representation = MLP(
        #     in_channels=observation_shape,
        #     hidden_channels=hidden_channels,
        #     out_channels=hidden_channels,
        #     layer_num=layer_num,
        #     activation=activation,
        #     norm_type=norm_type,
        #     # don't use activation and norm in the last layer of representation network is important for convergence.
        #     output_activation=False,
        #     output_norm=False,
        #     # last_linear_layer_init_zero=True is beneficial for convergence speed.
        #     last_linear_layer_init_zero=True,
        # )
        self.rgl_representation = RGCNLayer(
            robot_state_dim=robot_observation_shape[-1],
            human_state_dim=human_observation_shape[-1],
            similarity_function='embedded_gaussian',
            num_layer=layer_num,
            X_dim=hidden_channels,
            layerwise_graph=False,
            skip_connection=True,
            wr_dims=[64, hidden_channels],
            wh_dims=[64, hidden_channels],
            final_state_dim=hidden_channels,
            # for mlp
            norm_type=norm_type,
            last_linear_layer_init_zero=last_linear_layer_init_zero,
            activation=activation,
        )

    def forward(self, x: dict) -> torch.Tensor:
        """
        Shapes:
            - x (:obj:`torch.Tensor`): :math:`(B, N)`, where B is batch size, N is the length of vector observation.
            - output (:obj:`torch.Tensor`): :math:`(B, hidden_channels)`, where B is batch size.
        """
        return self.rgl_representation(x)

class PredictionNetwork(nn.Module):

    def __init__(
            self,
            observation_shape: SequenceType,
            action_space_size: int,
            num_res_blocks: int,
            num_channels: int,
            value_head_channels: int,
            policy_head_channels: int,
            fc_value_layers: int,
            fc_policy_layers: int,
            output_support_size: int,
            flatten_output_size_for_value_head: int,
            flatten_output_size_for_policy_head: int,
            downsample: bool = False,
            last_linear_layer_init_zero: bool = True,
            activation: nn.Module = nn.ReLU(inplace=True),
            norm_type: Optional[str] = 'BN',
    ) -> None:
        """
        Overview:
            The definition of policy and value prediction network, which is used to predict value and policy by the
            given latent state.
        Arguments:
            - observation_shape (:obj:`SequenceType`): The shape of observation space, e.g. (C, H, W) for image.
            - action_space_size: (:obj:`int`): Action space size, usually an integer number for discrete action space.
            - num_res_blocks (:obj:`int`): The number of res blocks in AlphaZero model.
            - num_channels (:obj:`int`): The channels of hidden states.
            - value_head_channels (:obj:`int`): The channels of value head.
            - policy_head_channels (:obj:`int`): The channels of policy head.
            - fc_value_layers (:obj:`SequenceType`): The number of hidden layers used in value head (MLP head).
            - fc_policy_layers (:obj:`SequenceType`): The number of hidden layers used in policy head (MLP head).
            - output_support_size (:obj:`int`): The size of categorical value output.
            - self_supervised_learning_loss (:obj:`bool`): Whether to use self_supervised_learning related networks \
            - flatten_output_size_for_value_head (:obj:`int`): The size of flatten hidden states, i.e. the input size \
                of the value head.
            - flatten_output_size_for_policy_head (:obj:`int`): The size of flatten hidden states, i.e. the input size \
                of the policy head.
            - downsample (:obj:`bool`): Whether to do downsampling for observations in ``representation_network``.
            - last_linear_layer_init_zero (:obj:`bool`): Whether to use zero initializations for the last layer of \
                dynamics/prediction mlp, default sets it to True.
            - activation (:obj:`Optional[nn.Module]`): Activation function used in network, which often use in-place \
                operation to speedup, e.g. ReLU(inplace=True).
            - norm_type (:obj:`str`): The type of normalization in networks. defaults to 'BN'.
        """
        super(PredictionNetwork, self).__init__()
        assert norm_type in ['BN', 'LN'], "norm_type must in ['BN', 'LN']"

        self.resblocks = nn.ModuleList(
            [
                ResBlock(
                    in_channels=num_channels, activation=activation, norm_type='BN', res_type='basic', bias=False
                ) for _ in range(num_res_blocks)
            ]
        )

        self.conv1x1_value = nn.Conv2d(num_channels, value_head_channels, 1)
        self.conv1x1_policy = nn.Conv2d(num_channels, policy_head_channels, 1)
        
        if norm_type == 'BN':
            self.norm_value = nn.BatchNorm2d(value_head_channels)
            self.norm_policy = nn.BatchNorm2d(policy_head_channels)
        elif norm_type == 'LN':
            if downsample:
                self.norm_value = nn.LayerNorm([value_head_channels, math.ceil(observation_shape[-2] / 16), math.ceil(observation_shape[-1] / 16)])
                self.norm_policy = nn.LayerNorm([policy_head_channels, math.ceil(observation_shape[-2] / 16), math.ceil(observation_shape[-1] / 16)])
            else:
                self.norm_value = nn.LayerNorm([value_head_channels, observation_shape[-2], observation_shape[-1]])
                self.norm_policy = nn.LayerNorm([policy_head_channels, observation_shape[-2], observation_shape[-1]])
        
        self.flatten_output_size_for_value_head = flatten_output_size_for_value_head
        self.flatten_output_size_for_policy_head = flatten_output_size_for_policy_head
        self.activation = activation

        self.fc_value = MLP(
            in_channels=self.flatten_output_size_for_value_head,
            hidden_channels=fc_value_layers[0],
            out_channels=output_support_size,
            layer_num=len(fc_value_layers) + 1,
            activation=self.activation,
            norm_type=norm_type,
            output_activation=False,
            output_norm=False,
            # last_linear_layer_init_zero=True is beneficial for convergence speed.
            last_linear_layer_init_zero=last_linear_layer_init_zero
        )
        self.fc_policy = MLP(
            in_channels=self.flatten_output_size_for_policy_head,
            hidden_channels=fc_policy_layers[0],
            out_channels=action_space_size,
            layer_num=len(fc_policy_layers) + 1,
            activation=self.activation,
            norm_type=norm_type,
            output_activation=False,
            output_norm=False,
            # last_linear_layer_init_zero=True is beneficial for convergence speed.
            last_linear_layer_init_zero=last_linear_layer_init_zero
        )

    def forward(self, latent_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Overview:
            Forward computation of the prediction network.
        Arguments:
            - latent_state (:obj:`torch.Tensor`): input tensor with shape (B, latent_state_dim).
        Returns:
            - policy (:obj:`torch.Tensor`): policy tensor with shape (B, action_space_size).
            - value (:obj:`torch.Tensor`): value tensor with shape (B, output_support_size).
        """
        for res_block in self.resblocks:
            latent_state = res_block(latent_state)

        value = self.conv1x1_value(latent_state)
        value = self.norm_value(value)
        value = self.activation(value)

        policy = self.conv1x1_policy(latent_state)
        policy = self.norm_policy(policy)
        policy = self.activation(policy)

        value = value.reshape(-1, self.flatten_output_size_for_value_head)
        policy = policy.reshape(-1, self.flatten_output_size_for_policy_head)

        value = self.fc_value(value)
        policy = self.fc_policy(policy)
        return policy, value


class PredictionNetworkMLP(nn.Module):

    def __init__(
            self,
            action_space_size,
            num_channels,
            common_layer_num: int = 2,
            fc_value_layers: SequenceType = [32],
            fc_policy_layers: SequenceType = [32],
            output_support_size: int = 601,
            last_linear_layer_init_zero: bool = True,
            activation: Optional[nn.Module] = nn.ReLU(inplace=True),
            norm_type: Optional[str] = 'BN',
    ):
        """
        Overview:
            The definition of policy and value prediction network with Multi-Layer Perceptron (MLP),
            which is used to predict value and policy by the given latent state.
        Arguments:
            - action_space_size: (:obj:`int`): Action space size, usually an integer number. For discrete action \
                space, it is the number of discrete actions.
            - num_channels (:obj:`int`): The channels of latent states.
            - fc_value_layers (:obj:`SequenceType`): The number of hidden layers used in value head (MLP head).
            - fc_policy_layers (:obj:`SequenceType`): The number of hidden layers used in policy head (MLP head).
            - output_support_size (:obj:`int`): The size of categorical value output.
            - last_linear_layer_init_zero (:obj:`bool`): Whether to use zero initializations for the last layer of \
                dynamics/prediction mlp, default sets it to True.
            - activation (:obj:`Optional[nn.Module]`): Activation function used in network, which often use in-place \
                operation to speedup, e.g. ReLU(inplace=True).
            - norm_type (:obj:`str`): The type of normalization in networks. defaults to 'BN'.
        """
        super().__init__()
        self.num_channels = num_channels

        # ******* common backbone ******
        self.fc_prediction_common = MLP(
            in_channels=self.num_channels,
            hidden_channels=self.num_channels,
            out_channels=self.num_channels,
            layer_num=common_layer_num,
            activation=activation,
            norm_type=norm_type,
            output_activation=True,
            output_norm=True,
            # last_linear_layer_init_zero=False is important for convergence
            last_linear_layer_init_zero=False,
        )

        # ******* value and policy head ******
        self.fc_value_head = MLP(
            in_channels=self.num_channels,
            hidden_channels=fc_value_layers[0],
            out_channels=output_support_size,
            layer_num=len(fc_value_layers) + 1,
            activation=activation,
            norm_type=norm_type,
            output_activation=False,
            output_norm=False,
            # last_linear_layer_init_zero=True is beneficial for convergence speed.
            last_linear_layer_init_zero=last_linear_layer_init_zero
        )
        self.fc_policy_head = MLP(
            in_channels=self.num_channels,
            hidden_channels=fc_policy_layers[0],
            out_channels=action_space_size,
            layer_num=len(fc_policy_layers) + 1,
            activation=activation,
            norm_type=norm_type,
            output_activation=False,
            output_norm=False,
            # last_linear_layer_init_zero=True is beneficial for convergence speed.
            last_linear_layer_init_zero=last_linear_layer_init_zero
        )

    def forward(self, latent_state: torch.Tensor):
        """
        Overview:
            Forward computation of the prediction network.
        Arguments:
            - latent_state (:obj:`torch.Tensor`): input tensor with shape (B, latent_state_dim).
        Returns:
            - policy (:obj:`torch.Tensor`): policy tensor with shape (B, action_space_size).
            - value (:obj:`torch.Tensor`): value tensor with shape (B, output_support_size).
        """
        x_prediction_common = self.fc_prediction_common(latent_state)

        value = self.fc_value_head(x_prediction_common)
        policy = self.fc_policy_head(x_prediction_common)
        return policy, value

from typing import Optional, Tuple, Dict
import logging
import itertools

import torch
import torch.nn as nn
from ding.torch_utils import MLP
from ding.utils import MODEL_REGISTRY, SequenceType

from .utils import renormalize, get_params_mean, get_dynamic_mean, get_reward_mean


class RGCNLayer(nn.Module):
    """
    Overview:
        Relational graph convolutional network layer.
    """

    def __init__(
        self,
        robot_num: int,
        human_num: int,
        robot_state_dim,
        human_state_dim,
        similarity_function,
        num_layer=2,
        X_dim=32,
        layerwise_graph=False,
        skip_connection=True,
        wr_dims=[64, 32],  # the last dim should equal to X_dim
        wh_dims=[64, 32],  # the last dim should equal to X_dim
        final_state_dim=32,  # should equal to X_dim
        norm_type=None,
        last_linear_layer_init_zero=True,
        activation: Optional[nn.Module] = nn.ReLU(inplace=True),
    ):
        super().__init__()

        # design choice
        # 'gaussian', 'embedded_gaussian', 'cosine', 'cosine_softmax', 'concatenation'
        self.similarity_function = similarity_function
        self.robot_num = robot_num
        self.human_num = human_num
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
        state = state.to(self.w_r[0].weight.dtype)
        if isinstance(state, dict):
            robot_states = state['robot_state']
            human_states = state['human_state']
        elif isinstance(state, torch.Tensor):
            if state.dim() == 3:
                # state shape:(B, stack_num*(robot_num+human_num), state_dim)
                stack_num = state.size(1) // (self.robot_num + self.human_num)
                # robot_states shape:(B, stack_num*robot_num, state_dim)
                robot_states = state[:, :stack_num * self.robot_num, :]
                # human_states shape:(B, stack_num*human_num, state_dim)
                human_states = state[:, stack_num * self.robot_num:, :]
            elif state.dim() == 2:
                # state shape:(B, stack_num*(robot_num+human_num)*state_dim)
                stack_num = state.size(1) // ((self.robot_num + self.human_num) * self.robot_state_dim)
                assert stack_num == 1, "stack_num should be 1 for 1-dim-array obs"
                # robot_states shape:(B, stack_num*robot_num, state_dim)
                robot_states = state[:, :stack_num * self.robot_num *
                                     self.robot_state_dim].reshape(-1, self.robot_num, self.robot_state_dim)
                # human_states shape:(B, stack_num*human_num, state_dim)
                human_states = state[:, stack_num * self.robot_num *
                                     self.robot_state_dim:].reshape(-1, self.human_num, self.human_state_dim)

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
        robot_state_dim: int,
        human_state_dim: int,
        robot_num: int,
        human_num: int,
        hidden_channels: int = 64,
        layer_num: int = 2,
        activation: Optional[nn.Module] = nn.ReLU(inplace=True),
        last_linear_layer_init_zero: bool = True,
        norm_type: Optional[str] = 'BN',
    ) -> torch.Tensor:
        """
        Overview:
            Representation network used in MuZero and derived algorithms. 
        Arguments:
            - robot_state_dim (:obj:`int`): The dimension of robot state.
            - human_state_dim (:obj:`int`): The dimension of human state.
            - robot_num (:obj:`int`): The number of robots.
            - human_num (:obj:`int`): The number of humans.
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
        self.robot_state_dim = robot_state_dim
        self.human_state_dim = human_state_dim
        self.hidden_channels = hidden_channels
        self.similarity_function = 'embedded_gaussian'
        self.robot_num = robot_num
        self.human_num = human_num
        self.rgcn = RGCNLayer(
            robot_num=self.robot_num,
            human_num=self.human_num,
            robot_state_dim=self.robot_state_dim,
            human_state_dim=self.human_state_dim,
            similarity_function=self.similarity_function,
            num_layer=2,
            X_dim=hidden_channels,
            final_state_dim=hidden_channels,
            wr_dims=[hidden_channels, hidden_channels],  # TODO: check dim
            wh_dims=[hidden_channels, hidden_channels],
            layerwise_graph=False,
            skip_connection=True,
            norm_type=None,
        )
        mlp_input_shape = (robot_num + human_num) * hidden_channels
        self.fc_representation = MLP(
            in_channels=mlp_input_shape,
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

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Shapes:
            - x (:obj:`torch.Tensor`): :math:`(B, N)`, where B is batch size, N is the length of vector observation.
            - output (:obj:`torch.Tensor`): :math:`(B, hidden_channels)`, where B is batch size.
        """
        gcn_embedding = self.rgcn(x)
        gcn_embedding = gcn_embedding.view(gcn_embedding.shape[0], -1)  # (B,M,N) -> (B,M*N)
        return self.fc_representation(gcn_embedding)

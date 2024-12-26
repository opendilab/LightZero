from typing import Optional, Tuple

import torch
import torch.nn as nn
from ding.utils import MODEL_REGISTRY, SequenceType

from .common import RepresentationNetworkMLP, PredictionNetworkMLP
from .muzero_model_mlp import DynamicsNetwork
from .stochastic_muzero_model import StochasticMuZeroModel, ChanceEncoder
from .utils import renormalize


@MODEL_REGISTRY.register('StochasticMuZeroModelMLP')
class StochasticMuZeroModelMLP(StochasticMuZeroModel):

    def __init__(
            self,
            observation_shape: int = 2,
            action_space_size: int = 6,
            chance_space_size: int = 2,
            latent_state_dim: int = 256,
            fc_reward_layers: SequenceType = [32],
            fc_value_layers: SequenceType = [32],
            fc_policy_layers: SequenceType = [32],
            reward_support_size: int = 601,
            value_support_size: int = 601,
            proj_hid: int = 1024,
            proj_out: int = 1024,
            pred_hid: int = 512,
            pred_out: int = 1024,
            self_supervised_learning_loss: bool = False,
            categorical_distribution: bool = True,
            activation: Optional[nn.Module] = nn.ReLU(inplace=True),
            last_linear_layer_init_zero: bool = True,
            state_norm: bool = False,
            discrete_action_encoding_type: str = 'one_hot',
            norm_type: Optional[str] = 'BN',
            res_connection_in_dynamics: bool = False,
            *args,
            **kwargs
    ):
        """
        Overview:
            The definition of the network model of Stochastic, which is a generalization version for 1D vector obs.  \
            The networks are mainly built on fully connected layers. \
            The representation network is an MLP network which maps the raw observation to a latent state. \
            The dynamics network is an MLP network which predicts the next latent state, and reward given the current latent state and action. \
            The prediction network is an MLP network which predicts the value and policy given the current latent state.
        Arguments:
            - observation_shape (:obj:`int`): Observation space shape, e.g. 8 for Lunarlander.
            - action_space_size: (:obj:`int`): Action space size, usually an integer number for discrete action space.
            - action_space_size: (:obj:`int`): Action space size, e.g. 4 for Lunarlander.
            - latent_state_dim (:obj:`int`): The dimension of latent state, such as 256.
            - fc_reward_layers (:obj:`SequenceType`): The number of hidden layers of the reward head (MLP head).
            - fc_value_layers (:obj:`SequenceType`): The number of hidden layers used in value head (MLP head).
            - fc_policy_layers (:obj:`SequenceType`): The number of hidden layers used in policy head (MLP head).
            - reward_support_size (:obj:`int`): The size of categorical reward output
            - value_support_size (:obj:`int`): The size of categorical value output.
            - proj_hid (:obj:`int`): The size of projection hidden layer.
            - proj_out (:obj:`int`): The size of projection output layer.
            - pred_hid (:obj:`int`): The size of prediction hidden layer.
            - pred_out (:obj:`int`): The size of prediction output layer.
            - self_supervised_learning_loss (:obj:`bool`): Whether to use self_supervised_learning related networks in Stochastic model, default set it to False.
            - categorical_distribution (:obj:`bool`): Whether to use discrete support to represent categorical distribution for value, reward/value_prefix.
            - activation (:obj:`Optional[nn.Module]`): Activation function used in network, which often use in-place \
                operation to speedup, e.g. ReLU(inplace=True).
            - last_linear_layer_init_zero (:obj:`bool`): Whether to use zero initializations for the last layer of value/policy mlp, default sets it to True.
            - state_norm (:obj:`bool`): Whether to use normalization for latent states, default sets it to True.
            - discrete_action_encoding_type (:obj:`str`): The encoding type of discrete action, which can be 'one_hot' or 'not_one_hot'.
            - norm_type (:obj:`str`): The type of normalization in networks. defaults to 'BN'.
            - res_connection_in_dynamics (:obj:`bool`): Whether to use residual connection for dynamics network, default set it to False.
        """
        super(StochasticMuZeroModelMLP, self).__init__()
        self.categorical_distribution = categorical_distribution
        if not self.categorical_distribution:
            self.reward_support_size = 1
            self.value_support_size = 1
        else:
            self.reward_support_size = reward_support_size
            self.value_support_size = value_support_size

        self.action_space_size = action_space_size
        self.chance_space_size = chance_space_size

        self.continuous_action_space = False
        # The dim of action space. For discrete action space, it is 1.
        # For continuous action space, it is the dimension of continuous action.
        self.action_space_dim = action_space_size if self.continuous_action_space else 1
        assert discrete_action_encoding_type in ['one_hot', 'not_one_hot'], discrete_action_encoding_type
        self.discrete_action_encoding_type = discrete_action_encoding_type
        if self.continuous_action_space:
            self.action_encoding_dim = action_space_size
        else:
            if self.discrete_action_encoding_type == 'one_hot':
                self.action_encoding_dim = action_space_size
            elif self.discrete_action_encoding_type == 'not_one_hot':
                self.action_encoding_dim = 1

        self.latent_state_dim = latent_state_dim
        self.proj_hid = proj_hid
        self.proj_out = proj_out
        self.pred_hid = pred_hid
        self.pred_out = pred_out
        self.self_supervised_learning_loss = self_supervised_learning_loss
        self.last_linear_layer_init_zero = last_linear_layer_init_zero
        self.state_norm = state_norm
        self.res_connection_in_dynamics = res_connection_in_dynamics

        self.representation_network = RepresentationNetworkMLP(
            observation_shape=observation_shape, hidden_channels=self.latent_state_dim, norm_type=norm_type
        )
        # TODO(pu): different input data type for chance_encoder
        # here, the input is two concatenated frames
        self.chance_encoder = ChanceEncoder(observation_shape * 2, chance_space_size, encoder_backbone_type='mlp')
        self.dynamics_network = DynamicsNetwork(
            action_encoding_dim=self.chance_space_size,
            num_channels=self.latent_state_dim + self.chance_space_size,
            common_layer_num=2,
            fc_reward_layers=fc_reward_layers,
            output_support_size=self.reward_support_size,
            last_linear_layer_init_zero=self.last_linear_layer_init_zero,
            norm_type=norm_type,
            res_connection_in_dynamics=self.res_connection_in_dynamics,
        )

        self.prediction_network = PredictionNetworkMLP(
            action_space_size=action_space_size,
            num_channels=latent_state_dim,
            fc_value_layers=fc_value_layers,
            fc_policy_layers=fc_policy_layers,
            output_support_size=self.value_support_size,
            last_linear_layer_init_zero=self.last_linear_layer_init_zero,
            norm_type=norm_type
        )

        self.afterstate_dynamics_network = AfterstateDynamicsNetwork(
            action_encoding_dim=self.action_encoding_dim,
            num_channels=self.latent_state_dim + self.action_encoding_dim,
            common_layer_num=2,
            fc_reward_layers=fc_reward_layers,
            output_support_size=self.reward_support_size,
            last_linear_layer_init_zero=self.last_linear_layer_init_zero,
            norm_type=norm_type,
            res_connection_in_dynamics=self.res_connection_in_dynamics,
        )
        self.afterstate_prediction_network = AfterstatePredictionNetworkMLP(
            chance_space_size=chance_space_size,
            num_channels=latent_state_dim,
            fc_value_layers=fc_value_layers,
            fc_policy_layers=fc_policy_layers,
            output_support_size=self.value_support_size,
            last_linear_layer_init_zero=self.last_linear_layer_init_zero,
            norm_type=norm_type
        )

        if self.self_supervised_learning_loss:
            # self_supervised_learning_loss related network proposed in EfficientZero
            self.projection_input_dim = latent_state_dim

            self.projection = nn.Sequential(
                nn.Linear(self.projection_input_dim, self.proj_hid), nn.BatchNorm1d(self.proj_hid), activation,
                nn.Linear(self.proj_hid, self.proj_hid), nn.BatchNorm1d(self.proj_hid), activation,
                nn.Linear(self.proj_hid, self.proj_out), nn.BatchNorm1d(self.proj_out)
            )
            self.prediction_head = nn.Sequential(
                nn.Linear(self.proj_out, self.pred_hid),
                nn.BatchNorm1d(self.pred_hid),
                activation,
                nn.Linear(self.pred_hid, self.pred_out),
            )

    def _dynamics(self, latent_state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Overview:
            Concatenate ``latent_state`` and ``action`` and use the dynamics network to predict ``next_latent_state`` \
            ``reward`` and ``next_reward_hidden_state``.
        Arguments:
            - latent_state (:obj:`torch.Tensor`): The encoding latent state of input state.
            - reward_hidden_state (:obj:`Tuple[torch.Tensor]`): The input hidden state of LSTM about reward.
            - action (:obj:`torch.Tensor`): The predicted action to rollout.
        Returns:
            - next_latent_state (:obj:`torch.Tensor`): The predicted latent state of the next timestep.
            - next_reward_hidden_state (:obj:`Tuple[torch.Tensor]`): The output hidden state of LSTM about reward.
            - reward (:obj:`torch.Tensor`): The predicted reward for input state.
        Shapes:
            - latent_state (:obj:`torch.Tensor`): :math:`(B, H)`, where B is batch_size, H is the dimension of latent state.
            - action (:obj:`torch.Tensor`): :math:`(B, )`, where B is batch_size.
            - next_latent_state (:obj:`torch.Tensor`): :math:`(B, H)`, where B is batch_size, H is the dimension of latent state.
            - reward (:obj:`torch.Tensor`): :math:`(B, reward_support_size)`, where B is batch_size.
        """
        # NOTE: the discrete action encoding type is important for some environments

        # Stack latent_state with the one hot encoded action
        if len(action.shape) == 1:
            # (batch_size, ) -> (batch_size, 1)
            # e.g.,  torch.Size([8]) ->  torch.Size([8, 1])
            action = action.unsqueeze(-1)

        # transform action to one-hot encoding.
        # action_one_hot shape: (batch_size, action_space_size), e.g., (8, 4)
        action_one_hot = torch.zeros(action.shape[0], self.chance_space_size, device=action.device)
        # transform action to torch.int64
        action = action.long()
        action_one_hot.scatter_(1, action, 1)
        action_encoding = action_one_hot

        action_encoding = action_encoding.to(latent_state.device).float()
        # state_action_encoding shape: (batch_size, latent_state[1] + action_dim]) or
        # (batch_size, latent_state[1] + action_space_size]) depending on the discrete_action_encoding_type.
        state_action_encoding = torch.cat((latent_state, action_encoding), dim=1)

        next_latent_state, reward = self.dynamics_network(state_action_encoding)

        if not self.state_norm:
            return next_latent_state, reward
        else:
            next_latent_state_normalized = renormalize(next_latent_state)
            return next_latent_state_normalized, reward

    def _afterstate_dynamics(self, latent_state: torch.Tensor, action: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor]:
        """
        Overview:
            Concatenate ``latent_state`` and ``action`` and use the dynamics network to predict ``next_latent_state`` \
            ``reward`` and ``next_reward_hidden_state``.
        Arguments:
            - latent_state (:obj:`torch.Tensor`): The encoding latent state of input state.
            - reward_hidden_state (:obj:`Tuple[torch.Tensor]`): The input hidden state of LSTM about reward.
            - action (:obj:`torch.Tensor`): The predicted action to rollout.
        Returns:
            - next_latent_state (:obj:`torch.Tensor`): The predicted latent state of the next timestep.
            - next_reward_hidden_state (:obj:`Tuple[torch.Tensor]`): The output hidden state of LSTM about reward.
            - reward (:obj:`torch.Tensor`): The predicted reward for input state.
        Shapes:
            - latent_state (:obj:`torch.Tensor`): :math:`(B, H)`, where B is batch_size, H is the dimension of latent state.
            - action (:obj:`torch.Tensor`): :math:`(B, )`, where B is batch_size.
            - next_latent_state (:obj:`torch.Tensor`): :math:`(B, H)`, where B is batch_size, H is the dimension of latent state.
            - reward (:obj:`torch.Tensor`): :math:`(B, reward_support_size)`, where B is batch_size.
        """
        # NOTE: the discrete action encoding type is important for some environments

        # discrete action space
        if self.discrete_action_encoding_type == 'one_hot':
            # Stack latent_state with the one hot encoded action
            if len(action.shape) == 1:
                # (batch_size, ) -> (batch_size, 1)
                # e.g.,  torch.Size([8]) ->  torch.Size([8, 1])
                action = action.unsqueeze(-1)

            # transform action to one-hot encoding.
            # action_one_hot shape: (batch_size, action_space_size), e.g., (8, 4)
            action_one_hot = torch.zeros(action.shape[0], self.action_space_size, device=action.device)
            # transform action to torch.int64
            action = action.long()
            action_one_hot.scatter_(1, action, 1)
            action_encoding = action_one_hot
        elif self.discrete_action_encoding_type == 'not_one_hot':
            action_encoding = action / self.action_space_size
            if len(action_encoding.shape) == 1:
                # (batch_size, ) -> (batch_size, 1)
                # e.g.,  torch.Size([8]) ->  torch.Size([8, 1])
                action_encoding = action_encoding.unsqueeze(-1)

        action_encoding = action_encoding.to(latent_state.device).float()
        # state_action_encoding shape: (batch_size, latent_state[1] + action_dim]) or
        # (batch_size, latent_state[1] + action_space_size]) depending on the discrete_action_encoding_type.
        state_action_encoding = torch.cat((latent_state, action_encoding), dim=1)

        next_latent_state, reward = self.afterstate_dynamics_network(state_action_encoding)

        if not self.state_norm:
            return next_latent_state, reward
        else:
            next_latent_state_normalized = renormalize(next_latent_state)
            return next_latent_state_normalized, reward

    def project(self, latent_state: torch.Tensor, with_grad=True) -> torch.Tensor:
        """
        Overview:
            Project the latent state to a lower dimension to calculate the self-supervised loss, which is \
            proposed in EfficientZero. For more details, please refer to the paper ``Exploring Simple Siamese Representation Learning``.
        Arguments:
            - latent_state (:obj:`torch.Tensor`): The encoding latent state of input state.
            - with_grad (:obj:`bool`): Whether to calculate gradient for the projection result.
        Returns:
            - proj (:obj:`torch.Tensor`): The result embedding vector of projection operation.
        Shapes:
            - latent_state (:obj:`torch.Tensor`): :math:`(B, H)`, where B is batch_size, H is the dimension of latent state.
            - proj (:obj:`torch.Tensor`): :math:`(B, projection_output_dim)`, where B is batch_size.

        Examples:
            >>> latent_state = torch.randn(256, 64)
            >>> output = self.project(latent_state)
            >>> output.shape # (256, 1024)
        """
        proj = self.projection(latent_state)

        if with_grad:
            # with grad, use prediction_head
            return self.prediction_head(proj)
        else:
            return proj.detach()


AfterstateDynamicsNetwork = DynamicsNetwork


class AfterstatePredictionNetworkMLP(PredictionNetworkMLP):

    def __init__(
            self,
            chance_space_size,
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
            The definition of policy and value prediction network with Multi-Layer Perceptron (MLP), \
            which is used to predict value and policy by the given latent state.
        Arguments:
            - chance_space_size: (:obj:`int`): Chance space size, usually an integer number. For discrete action \
                space, it is the number of discrete chance outcomes.
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
        super(AfterstatePredictionNetworkMLP, self).__init__(chance_space_size, num_channels, common_layer_num,
                                                             fc_value_layers, fc_policy_layers, output_support_size,
                                                             last_linear_layer_init_zero
                                                             , activation, norm_type)

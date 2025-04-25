from typing import Optional, Tuple

import torch
import torch.nn as nn
from ding.torch_utils import MLP
from ding.utils import MODEL_REGISTRY, SequenceType
from numpy import ndarray

from .common import EZNetworkOutput, RepresentationNetworkMLP, PredictionNetworkMLP
from .utils import renormalize, get_params_mean, get_dynamic_mean, get_reward_mean


@MODEL_REGISTRY.register('EfficientZeroModelMLP')
class EfficientZeroModelMLP(nn.Module):

    def __init__(
        self,
        observation_shape: int = 2,
        action_space_size: int = 6,
        lstm_hidden_size: int = 512,
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
        self_supervised_learning_loss: bool = True,
        categorical_distribution: bool = True,
        last_linear_layer_init_zero: bool = True,
        state_norm: bool = False,
        activation: Optional[nn.Module] = nn.ReLU(inplace=True),
        norm_type: Optional[str] = 'BN',
        discrete_action_encoding_type: str = 'one_hot',
        res_connection_in_dynamics: bool = False,
        *args,
        **kwargs,
    ):
        """
        Overview:
            The definition of the network model of EfficientZero, which is a generalization version for 1D vector obs.
            The networks are mainly built on fully connected layers.
            Sampled EfficientZero model consists of a representation network, a dynamics network and a prediction network.
            The representation network is an MLP network which maps the raw observation to a latent state.
            The dynamics network is an MLP+LSTM network which predicts the next latent state, reward_hidden_state and value_prefix given the current latent state and action.
            The prediction network is an MLP network which predicts the value and policy given the current latent state.
        Arguments:
            - observation_shape (:obj:`int`): Observation space shape, e.g. 8 for Lunarlander.
            - action_space_size: (:obj:`int`): Action space size, e.g. 4 for Lunarlander.
            - lstm_hidden_size (:obj:`int`): The hidden size of LSTM in dynamics network to predict value_prefix.
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
            - self_supervised_learning_loss (:obj:`bool`): Whether to use self_supervised_learning related networks in Sampled EfficientZero model, default set it to False.
            - categorical_distribution (:obj:`bool`): Whether to use discrete support to represent categorical distribution for value, reward/value_prefix.
            - last_linear_layer_init_zero (:obj:`bool`): Whether to use zero initializations for the last layer of value/policy mlp, default sets it to True.
            - state_norm (:obj:`bool`): Whether to use normalization for latent states, default sets it to True.
            - activation (:obj:`Optional[nn.Module]`): Activation function used in network, which often use in-place \
                operation to speedup, e.g. ReLU(inplace=True).
            - discrete_action_encoding_type (:obj:`str`): The type of encoding for discrete action. Default sets it to 'one_hot'. options = {'one_hot', 'not_one_hot'}
            - norm_type (:obj:`str`): The type of normalization in networks. defaults to 'BN'.
            - res_connection_in_dynamics (:obj:`bool`): Whether to use residual connection for dynamics network, default set it to False.
        """
        super(EfficientZeroModelMLP, self).__init__()
        if not categorical_distribution:
            self.reward_support_size = 1
            self.value_support_size = 1
        else:
            self.reward_support_size = reward_support_size
            self.value_support_size = value_support_size

        self.action_space_size = action_space_size
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

        self.lstm_hidden_size = lstm_hidden_size
        self.proj_hid = proj_hid
        self.proj_out = proj_out
        self.pred_hid = pred_hid
        self.pred_out = pred_out
        self.self_supervised_learning_loss = self_supervised_learning_loss
        self.last_linear_layer_init_zero = last_linear_layer_init_zero
        self.state_norm = state_norm
        self.res_connection_in_dynamics = res_connection_in_dynamics

        self.representation_network = RepresentationNetworkMLP(
            observation_shape=observation_shape, hidden_channels=latent_state_dim, norm_type=norm_type
        )

        self.dynamics_network = DynamicsNetworkMLP(
            action_encoding_dim=self.action_encoding_dim,
            num_channels=latent_state_dim + self.action_encoding_dim,
            common_layer_num=2,
            lstm_hidden_size=lstm_hidden_size,
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

    def initial_inference(self, obs: torch.Tensor) -> EZNetworkOutput:
        """
        Overview:
            Initial inference of EfficientZero model, which is the first step of the EfficientZero model.
            To perform the initial inference, we first use the representation network to obtain the "latent_state" of the observation.
            Then we use the prediction network to predict the "value" and "policy_logits" of the "latent_state", and
            also prepare the zeros-like ``reward_hidden_state`` for the next step of the EfficientZero model.
        Arguments:
            - obs (:obj:`torch.Tensor`): The 1D vector observation data.
        Returns (EZNetworkOutput):
            - value (:obj:`torch.Tensor`): The output value of input state to help policy improvement and evaluation.
            - value_prefix (:obj:`torch.Tensor`): The predicted prefix sum of value for input state. \
                In initial inference, we set it to zero vector.
            - policy_logits (:obj:`torch.Tensor`): The output logit to select discrete action.
            - latent_state (:obj:`torch.Tensor`): The encoding latent state of input state.
            - reward_hidden_state (:obj:`Tuple[torch.Tensor]`): The hidden state of LSTM about reward. In initial inference, \
                we set it to the zeros-like hidden state (H and C).
        Shapes:
            - obs (:obj:`torch.Tensor`): :math:`(B, obs_shape)`, where B is batch_size.
            - value (:obj:`torch.Tensor`): :math:`(B, value_support_size)`, where B is batch_size.
            - value_prefix (:obj:`torch.Tensor`): :math:`(B, reward_support_size)`, where B is batch_size.
            - policy_logits (:obj:`torch.Tensor`): :math:`(B, action_dim)`, where B is batch_size.
            - latent_state (:obj:`torch.Tensor`): :math:`(B, H)`, where B is batch_size, H is the dimension of latent state.
            - reward_hidden_state (:obj:`Tuple[torch.Tensor]`): The shape of each element is :math:`(1, B, lstm_hidden_size)`, where B is batch_size.
        """
        batch_size = obs.size(0)
        latent_state = self._representation(obs)
        policy_logits, value = self._prediction(latent_state)
        # zero initialization for reward hidden states
        # (hn, cn), each element shape is (layer_num=1, batch_size, lstm_hidden_size)
        reward_hidden_state = (
            torch.zeros(1, batch_size,
                        self.lstm_hidden_size).to(obs.device), torch.zeros(1, batch_size,
                                                                           self.lstm_hidden_size).to(obs.device)
        )
        return EZNetworkOutput(value, [0. for _ in range(batch_size)], policy_logits, latent_state, reward_hidden_state)

    def recurrent_inference(
            self, latent_state: torch.Tensor, reward_hidden_state: torch.Tensor, action: torch.Tensor
    ) -> EZNetworkOutput:
        """
        Overview:
            Recurrent inference of EfficientZero model, which is the rollout step of the EfficientZero model.
            To perform the recurrent inference, we first use the dynamics network to predict ``next_latent_state``,
            ``reward_hidden_state``, ``value_prefix`` by the given current ``latent_state`` and ``action``.
             We then use the prediction network to predict the ``value`` and ``policy_logits``.
        Arguments:
            - latent_state (:obj:`torch.Tensor`): The encoding latent state of input state.
            - reward_hidden_state (:obj:`Tuple[torch.Tensor]`): The input hidden state of LSTM about reward.
            - action (:obj:`torch.Tensor`): The predicted action to rollout.
        Returns (EZNetworkOutput):
            - value (:obj:`torch.Tensor`): The output value of input state to help policy improvement and evaluation.
            - value_prefix (:obj:`torch.Tensor`): The predicted prefix sum of value for input state.
            - policy_logits (:obj:`torch.Tensor`): The output logit to select discrete action.
            - next_latent_state (:obj:`torch.Tensor`): The predicted next latent state.
            - reward_hidden_state (:obj:`Tuple[torch.Tensor]`): The output hidden state of LSTM about reward.
        Shapes:
            - action (:obj:`torch.Tensor`): :math:`(B, )`, where B is batch_size.
            - value (:obj:`torch.Tensor`): :math:`(B, value_support_size)`, where B is batch_size.
            - value_prefix (:obj:`torch.Tensor`): :math:`(B, reward_support_size)`, where B is batch_size.
            - policy_logits (:obj:`torch.Tensor`): :math:`(B, action_dim)`, where B is batch_size.
            - latent_state (:obj:`torch.Tensor`): :math:`(B, H)`, where B is batch_size, H is the dimension of latent state.
            - next_latent_state (:obj:`torch.Tensor`): :math:`(B, H)`, where B is batch_size, H is the dimension of latent state.
            - reward_hidden_state (:obj:`Tuple[torch.Tensor]`): The shape of each element is :math:`(1, B, lstm_hidden_size)`, where B is batch_size.
        """
        next_latent_state, reward_hidden_state, value_prefix = self._dynamics(latent_state, reward_hidden_state, action)
        policy_logits, value = self._prediction(next_latent_state)
        return EZNetworkOutput(value, value_prefix, policy_logits, next_latent_state, reward_hidden_state)

    def _representation(self, observation: torch.Tensor) -> Tuple[torch.Tensor]:
        """
         Overview:
             Use the representation network to encode the observations into latent state.
         Arguments:
             - obs (:obj:`torch.Tensor`): The 1D vector  observation data.
         Returns:
             - latent_state (:obj:`torch.Tensor`): The encoding latent state of input state.
         Shapes:
             - obs (:obj:`torch.Tensor`): :math:`(B, obs_shape)`, where B is batch_size.
            - latent_state (:obj:`torch.Tensor`): :math:`(B, H)`, where B is batch_size, H is the dimension of latent state.
         """
        latent_state = self.representation_network(observation)
        if self.state_norm:
            latent_state = renormalize(latent_state)
        return latent_state

    def _prediction(self, latent_state: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Overview:
            Use the representation network to encode the observations into latent state.
        Arguments:
            - obs (:obj:`torch.Tensor`): The 1D vector observation data.
        Returns:
            - policy_logits (:obj:`torch.Tensor`): The output logit to select discrete action.
            - value (:obj:`torch.Tensor`): The output value of input state to help policy improvement and evaluation.
        Shapes:
            - latent_state (:obj:`torch.Tensor`): :math:`(B, H)`, where B is batch_size, H is the dimension of latent state.
            - policy_logits (:obj:`torch.Tensor`): :math:`(B, action_dim)`, where B is batch_size.
            - value (:obj:`torch.Tensor`): :math:`(B, value_support_size)`, where B is batch_size.
        """
        policy_logits, value = self.prediction_network(latent_state)
        return policy_logits, value

    def _dynamics(self, latent_state: torch.Tensor, reward_hidden_state: Tuple,
                  action: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor], torch.Tensor]:
        """
        Overview:
            Concatenate ``latent_state`` and ``action`` and use the dynamics network to predict ``next_latent_state``
            ``value_prefix`` and ``next_reward_hidden_state``.
        Arguments:
            - latent_state (:obj:`torch.Tensor`): The encoding latent state of input state.
            - reward_hidden_state (:obj:`Tuple[torch.Tensor]`): The input hidden state of LSTM about reward.
            - action (:obj:`torch.Tensor`): The predicted action to rollout.
        Returns:
            - next_latent_state (:obj:`torch.Tensor`): The predicted latent state of the next timestep.
            - next_reward_hidden_state (:obj:`Tuple[torch.Tensor]`): The output hidden state of LSTM about reward.
            - value_prefix (:obj:`torch.Tensor`): The predicted prefix sum of value for input state.
        Shapes:
            - latent_state (:obj:`torch.Tensor`): :math:`(B, H)`, where B is batch_size, H is the dimension of latent state.
            - action (:obj:`torch.Tensor`): :math:`(B, )`, where B is batch_size.
            - next_latent_state (:obj:`torch.Tensor`): :math:`(B, H)`, where B is batch_size, H is the dimension of latent state.
            - value_prefix (:obj:`torch.Tensor`): :math:`(B, reward_support_size)`, where B is batch_size.
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

        # NOTE: the key difference with MuZero
        next_latent_state, next_reward_hidden_state, value_prefix = self.dynamics_network(
            state_action_encoding, reward_hidden_state
        )

        if self.state_norm:
            next_latent_state = renormalize(next_latent_state)
        return next_latent_state, next_reward_hidden_state, value_prefix

    def project(self, latent_state: torch.Tensor, with_grad=True):
        """
        Overview:
            Project the latent state to a lower dimension to calculate the self-supervised loss, which is proposed in EfficientZero.
            For more details, please refer to the paper ``Exploring Simple Siamese Representation Learning``.
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

    def get_params_mean(self) -> float:
        return get_params_mean(self)


class DynamicsNetworkMLP(nn.Module):

    def __init__(
        self,
        action_encoding_dim: int = 2,
        num_channels: int = 64,
        common_layer_num: int = 2,
        fc_reward_layers: SequenceType = [32],
        output_support_size: int = 601,
        lstm_hidden_size: int = 512,
        last_linear_layer_init_zero: bool = True,
        activation: Optional[nn.Module] = nn.ReLU(inplace=True),
        norm_type: Optional[str] = 'BN',
        res_connection_in_dynamics: bool = False,
    ):
        """
        Overview:
            The definition of dynamics network in EfficientZero algorithm, which is used to predict next latent state
            value_prefix and reward_hidden_state by the given current latent state and action.
            The networks are mainly built on fully connected layers.
        Arguments:
            - action_encoding_dim (:obj:`int`): The dimension of action encoding.
            - num_channels (:obj:`int`): The num of channels in latent states.
            - common_layer_num (:obj:`int`): The number of common layers in dynamics network.
            - fc_reward_layers (:obj:`SequenceType`): The number of hidden layers of the reward head (MLP head).
            - output_support_size (:obj:`int`): The size of categorical reward output.
            - lstm_hidden_size (:obj:`int`): The hidden size of lstm in dynamics network.
            - last_linear_layer_init_zero (:obj:`bool`): Whether to use zero initializationss for the last layer of value/policy head, default sets it to True.
            - activation (:obj:`Optional[nn.Module]`): Activation function used in network, which often use in-place \
                operation to speedup, e.g. ReLU(inplace=True).
            - norm_type (:obj:`str`): The type of normalization in networks. defaults to 'BN'.
            - res_connection_in_dynamics (:obj:`bool`): Whether to use residual connection in dynamics network.
        """
        super().__init__()
        assert num_channels > action_encoding_dim, f'num_channels:{num_channels} <= action_encoding_dim:{action_encoding_dim}'

        self.num_channels = num_channels
        self.action_encoding_dim = action_encoding_dim
        self.latent_state_dim = self.num_channels - self.action_encoding_dim
        self.lstm_hidden_size = lstm_hidden_size
        self.activation = activation
        self.res_connection_in_dynamics = res_connection_in_dynamics

        if self.res_connection_in_dynamics:
            self.fc_dynamics_1 = MLP(
                in_channels=self.num_channels,
                hidden_channels=self.latent_state_dim,
                layer_num=common_layer_num,
                out_channels=self.latent_state_dim,
                activation=activation,
                norm_type=norm_type,
                output_activation=True,
                output_norm=True,
                # last_linear_layer_init_zero=False is important for convergence
                last_linear_layer_init_zero=False,
            )
            self.fc_dynamics_2 = MLP(
                in_channels=self.latent_state_dim,
                hidden_channels=self.latent_state_dim,
                layer_num=common_layer_num,
                out_channels=self.latent_state_dim,
                activation=activation,
                norm_type=norm_type,
                output_activation=True,
                output_norm=True,
                # last_linear_layer_init_zero=False is important for convergence
                last_linear_layer_init_zero=False,
            )
        else:
            self.fc_dynamics = MLP(
                in_channels=self.num_channels,
                hidden_channels=self.latent_state_dim,
                layer_num=common_layer_num,
                out_channels=self.latent_state_dim,
                activation=activation,
                norm_type=norm_type,
                output_activation=True,
                output_norm=True,
                # last_linear_layer_init_zero=False is important for convergence
                last_linear_layer_init_zero=False,
            )

        # input_shape: （sequence_length，batch_size，input_size)
        # output_shape: (sequence_length, batch_size, hidden_size)
        self.lstm = nn.LSTM(input_size=self.latent_state_dim, hidden_size=self.lstm_hidden_size)

        self.fc_reward_head = MLP(
            in_channels=self.lstm_hidden_size,
            hidden_channels=fc_reward_layers[0],
            layer_num=2,
            out_channels=output_support_size,
            activation=self.activation,
            norm_type=norm_type,
            output_activation=False,
            output_norm=False,
            last_linear_layer_init_zero=last_linear_layer_init_zero
        )

    def forward(self, state_action_encoding: torch.Tensor, reward_hidden_state):
        """
        Overview:
            Forward computation of the dynamics network. Predict next latent state given current state_action_encoding and reward hidden state.
        Arguments:
            - state_action_encoding (:obj:`torch.Tensor`): The state-action encoding, which is the concatenation of \
                    latent state and action encoding, with shape (batch_size, num_channels, height, width).
            - reward_hidden_state (:obj:`Tuple[torch.Tensor, torch.Tensor]`): The input hidden state of LSTM about reward.
        Returns:
            - next_latent_state (:obj:`torch.Tensor`): The next latent state, with shape (batch_size, latent_state_dim).
            - next_reward_hidden_state (:obj:`torch.Tensor`): The input hidden state of LSTM about reward.
            - value_prefix (:obj:`torch.Tensor`): The predicted prefix sum of value for input state.
        """
        if self.res_connection_in_dynamics:
            # take the state encoding (latent_state), state_action_encoding[:, -self.action_encoding_dim]
            # is action encoding
            latent_state = state_action_encoding[:, :-self.action_encoding_dim]
            x = self.fc_dynamics_1(state_action_encoding)
            # the residual link: add state encoding to the state_action encoding
            next_latent_state = x + latent_state
            next_latent_state_ = self.fc_dynamics_2(next_latent_state)
        else:
            next_latent_state = self.fc_dynamics(state_action_encoding)
            next_latent_state_ = next_latent_state

        next_latent_state_unsqueeze = next_latent_state_.unsqueeze(0)
        value_prefix, next_reward_hidden_state = self.lstm(next_latent_state_unsqueeze, reward_hidden_state)
        value_prefix = self.fc_reward_head(value_prefix.squeeze(0))

        return next_latent_state, next_reward_hidden_state, value_prefix

    def get_dynamic_mean(self) -> float:
        return get_dynamic_mean(self)

    def get_reward_mean(self) -> Tuple[ndarray, float]:
        return get_reward_mean(self)

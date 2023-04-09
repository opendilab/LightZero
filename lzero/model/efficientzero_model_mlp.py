from typing import Optional, Tuple

import torch
import torch.nn as nn
from ding.torch_utils import MLP
from ding.utils import MODEL_REGISTRY, SequenceType

from .common import EZNetworkOutput, RepresentationNetworkMLP
from .utils import renormalize, get_params_mean, get_dynamic_mean, get_reward_mean


@MODEL_REGISTRY.register('EfficientZeroModelMLP')
class EfficientZeroModelMLP(nn.Module):

    def __init__(
            self,
            observation_shape: int = 2,
            action_space_size: int = 6,
            latent_state_dim: int = 256,
            categorical_distribution: bool = True,
            activation: Optional[nn.Module] = nn.ReLU(inplace=True),
            last_linear_layer_init_zero: bool = True,
            state_norm: bool = False,
            lstm_hidden_size: int = 512,
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
    ):
        """
        Overview:
            EfficientZero network.
        Arguments:
            - observation_shape (:obj:`int`): Observation space shape, e.g. 2.
            - action_space_size: (:obj:`int`): Action space size, such as 6.
            - latent_state_dim (:obj:`int`): Latent state dimension, such as 256.
            - categorical_distribution (:obj:`bool`): Whether to use discrete support to represent categorical distribution for value, reward/value_prefix.
            - activation (:obj:`Optional[nn.Module]`): the activation in EfficientZero model.
            - last_linear_layer_init_zero (:obj:`bool`): Whether to use zero initialization for the last layer of value/policy mlp, default set it to True.
            - state_norm (:obj:`bool`): Whether to use normalization for hidden states, default set it to True.
            - fc_reward_layers (:obj:`SequenceType`): hidden layers of the reward prediction head (MLP head).
            - fc_value_layers (:obj:`SequenceType`): hidden layers of the value prediction head (MLP head).
            - fc_policy_layers (:obj:`SequenceType`): hidden layers of the policy prediction head (MLP head).
            - reward_support_size (:obj:`int`): dim of reward output
            - value_support_size (:obj:`int`): dim of value output.
            - proj_hid (:obj:`int`): dim of projection hidden layer.
            - proj_out (:obj:`int`): dim of projection output layer.
            - pred_hid (:obj:`int`):dim of prediction hidden layer.
            - pred_out (:obj:`int`): dim of prediction output layer.
            - self_supervised_learning_loss (:obj:`bool`): Whether to use self_supervised_learning related networks in EfficientZero model, default set it to False.
        """
        super(EfficientZeroModelMLP, self).__init__()
        self.action_space_size = action_space_size
        self.lstm_hidden_size = lstm_hidden_size
        self.categorical_distribution = categorical_distribution
        self.self_supervised_learning_loss = self_supervised_learning_loss
        if not self.categorical_distribution:
            self.reward_support_size = 1
            self.value_support_size = 1
        else:
            self.reward_support_size = reward_support_size
            self.value_support_size = value_support_size

        # for discrete action space, we use one-hot encoding
        self.action_encoding_dim = self.action_space_size

        self.proj_hid = proj_hid
        self.proj_out = proj_out
        self.pred_hid = pred_hid
        self.pred_out = pred_out
        self.last_linear_layer_init_zero = last_linear_layer_init_zero
        self.state_norm = state_norm
        self.action_space_size = action_space_size

        self.representation_network = RepresentationNetworkMLP(observation_shape=observation_shape,
                                                               hidden_channels=latent_state_dim)

        self.dynamics_network = DynamicsNetwork(
            action_space_size=action_space_size,
            in_channels=latent_state_dim + self.action_encoding_dim,
            common_layer_num=2,
            lstm_hidden_size=lstm_hidden_size,
            fc_reward_layers=fc_reward_layers,
            output_support_size=self.reward_support_size,
            last_linear_layer_init_zero=self.last_linear_layer_init_zero,
        )

        self.prediction_network = PredictionNetwork(
            action_space_size=action_space_size,
            in_channels=latent_state_dim,
            fc_value_layers=fc_value_layers,
            fc_policy_layers=fc_policy_layers,
            output_support_size=self.value_support_size,
            last_linear_layer_init_zero=self.last_linear_layer_init_zero,
        )

        if self.self_supervised_learning_loss:
            # projection
            self.projection_input_dim = latent_state_dim

            self.projection = nn.Sequential(
                nn.Linear(self.projection_input_dim, self.proj_hid), nn.BatchNorm1d(self.proj_hid),
                activation, nn.Linear(self.proj_hid, self.proj_hid), nn.BatchNorm1d(self.proj_hid),
                activation, nn.Linear(self.proj_hid, self.proj_out), nn.BatchNorm1d(self.proj_out)
            )
            self.projection_head = nn.Sequential(
                nn.Linear(self.proj_out, self.pred_hid),
                nn.BatchNorm1d(self.pred_hid),
                activation,
                nn.Linear(self.pred_hid, self.pred_out),
            )

    def initial_inference(self, obs: torch.Tensor) -> EZNetworkOutput:
        """
         Overview:
             To perform the initial inference, we first use the representation network to obtain the "latent_state" of the observation.
             We then use the prediction network to predict the "value" and "policy_logits" of the "latent_state".
         Arguments:
             - obs (:obj:`torch.Tensor`): (batch_size, num_channel, obs_shape[1], obs_shape[2]), e.g. (1,64,6,6).
             - reward_hidden_state (:obj:`torch.Tensor`): (batch_size, 1, 1) e.g. (1, 1, 1).
             - action (:obj:`torch.Tensor`): (batch_size, action_dim).
        Returns:
             EZNetworkOutput
                - value (:obj:`torch.Tensor`): (batch_size, 1).
                - policy_logits (:obj:`torch.Tensor`): (batch_size, action_dim).
                - latent_state (:obj:`torch.Tensor`): (batch_size, 1, 1) e.g. (1, 1, 1).
                - reward_hidden (:obj:`torch.Tensor`): (batch_size, 1, 1) e.g. (1, 1, 1).
         """
        num = obs.size(0)
        latent_state = self._representation(obs)
        policy_logits, value = self._prediction(latent_state)
        # zero initialization for reward hidden states
        reward_hidden = (torch.zeros(1, num, self.lstm_hidden_size).to(obs.device), torch.zeros(1, num, self.lstm_hidden_size).to(obs.device))

        return EZNetworkOutput(value, [0. for _ in range(num)], policy_logits, latent_state, reward_hidden)

    def recurrent_inference(
            self, latent_state: torch.Tensor, reward_hidden: torch.Tensor, action: torch.Tensor
    ) -> EZNetworkOutput:
        """
         Overview:
             To perform the recurrent inference, we first use the dynamics network to predict ``next_latent_state``, ``reward_hidden_state``, ``value_prefix``
             given current ``latent_state`` and ``action``.
             We then use the prediction network to predict the "value" and "policy_logits" of the "latent_state".
         Arguments:
             - obs (:obj:`torch.Tensor`): (batch_size, num_channel, obs_shape[1], obs_shape[2]), e.g. (1,64,6,6).
             - reward_hidden_state (:obj:`torch.Tensor`): (batch_size, 1, 1) e.g. (1, 1, 1).
             - action (:obj:`torch.Tensor`): (batch_size, action_dim).
        Returns:
             EZNetworkOutput
                - value (:obj:`torch.Tensor`): (batch_size, 1).
                - value_prefix (:obj:`torch.Tensor`): (batch_size, 1).
                - policy_logits (:obj:`torch.Tensor`): (batch_size, action_dim).
                - latent_state (:obj:`torch.Tensor`): (batch_size, 1, 1) e.g. (1, 1, 1).
                - reward_hidden (:obj:`torch.Tensor`): (batch_size, 1, 1) e.g. (1, 1, 1).
         """
        latent_state, reward_hidden, value_prefix = self._dynamics(latent_state, reward_hidden, action)
        policy_logits, value = self._prediction(latent_state)
        return EZNetworkOutput(value, value_prefix, policy_logits, latent_state, reward_hidden)

    def _representation(self, observation: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Overview:
            Representation network. Encode the observations into latent state.
        Arguments:
             - observation (:obj:`torch.Tensor`): (batch_size, num_channel, obs_shape[1], obs_shape[2]), e.g. (1,64,6,6).
        Returns:
            - latent_state (:obj:`torch.Tensor`): (batch_size, 1, 1) e.g. (1, 1, 1).
        """
        latent_state = self.representation_network(observation)
        if not self.state_norm:
            return latent_state
        else:
            latent_state_normalized = renormalize(latent_state)
            return latent_state_normalized

    def _prediction(self, latent_state: torch.Tensor) -> Tuple[torch.Tensor]:
        """
         Overview:
             use the prediction network to predict the "value" and "policy_logits" of the "latent_state".
         Arguments:
            - latent_state (:obj:`torch.Tensor`): (batch_size, 1, 1) e.g. (1, 1, 1).
        Returns:
            - policy_logits (:obj:`torch.Tensor`): (batch_size, action_dim).
            - value (:obj:`torch.Tensor`): (batch_size, 1).
         """
        policy_logits, value = self.prediction_network(latent_state)
        return policy_logits, value

    def _dynamics(self, latent_state: torch.Tensor, reward_hidden_state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor]:
        """
         Overview:
             Dynamics function. Predict ``next_latent_state``, ``reward_hidden_state``, ``value_prefix``
             given current ``latent_state`` and ``action``.
         Arguments:
             - latent_state (:obj:`torch.Tensor`): (batch_size, num_channel, obs_shape[1], obs_shape[2]), e.g. (1,64,6,6).
             - reward_hidden_state (:obj:`torch.Tensor`): (batch_size, 1, 1) e.g. (1, 1, 1).
             - action (:obj:`torch.Tensor`): (batch_size, action_dim).
        Returns:
            - next_latent_state (:obj:`torch.Tensor`): (batch_size, 1, 1) e.g. (1, 1, 1).
            - next_reward_hidden_state (:obj:`torch.Tensor`): (batch_size, 1, 1) e.g. (1, 1, 1).
            - value_prefix (:obj:`torch.Tensor`): (batch_size, 1).
         """
        # TODO: action encoding
        # discrete action space
        # Stack latent_state with a game specific one hot encoded action
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

        state_action_encoding = torch.cat((latent_state, action_encoding), dim=1)

        # NOTE: the key difference with MuZero
        next_latent_state, next_reward_hidden_state, value_prefix = self.dynamics_network(state_action_encoding, reward_hidden_state)

        if not self.state_norm:
            return next_latent_state, next_reward_hidden_state, value_prefix
        else:
            next_latent_state_normalized = renormalize(next_latent_state)
            return next_latent_state_normalized, next_reward_hidden_state, value_prefix

    def project(self, latent_state: torch.Tensor, with_grad=True):
        """
        Overview:
            only used when ``self.self_supervised_learning_loss=True``.
            Please refer to paper ``Exploring Simple Siamese Representation Learning`` for details.
        """
        proj = self.projection(latent_state)

        # with grad, use proj_head
        if with_grad:
            return self.projection_head(proj)
        else:
            return proj.detach()

    def get_params_mean(self):
        return get_params_mean(self)


class DynamicsNetwork(nn.Module):

    def __init__(
            self,
            action_space_size: int = 2,
            in_channels: int = 64,
            common_layer_num: int = 2,
            lstm_hidden_size: int = 512,
            fc_reward_layers: SequenceType = [32],
            output_support_size: int = 601,
            last_linear_layer_init_zero: bool = True,
            activation: Optional[nn.Module] = nn.ReLU(inplace=True),
    ):
        """
        Overview:
            Dynamics network. Predict next hidden state given current hidden state and action.
        Arguments:
            - action_space_size (:obj:`int`): action space size.
            - in_channels (:obj:`int`): channels of hidden states.
            - common_layer_num (:obj:`int`): number of common layers.
            - lstm_hidden_size (:obj:`int`): dim of lstm hidden state in dynamics network.
            - fc_reward_layers (:obj:`list`):  hidden layers of the reward prediction head (MLP head).
            - output_support_size (:obj:`int`): dim of reward output
            - last_linear_layer_init_zero (:obj:`bool`): Whether to use zero initialization for the last layer of value/policy mlp, default set it to True.
            - activation (:obj:`Optional[nn.Module]`): the activation in Dynamics network.
        """
        super().__init__()
        self.in_channels = in_channels
        # for discrete action space, we use one-hot encoding
        self.action_encoding_dim = action_space_size
        self.latent_state_dim = self.in_channels - self.action_encoding_dim

        self.lstm_hidden_size = lstm_hidden_size

        self.fc_dynamics = MLP(
            in_channels=self.in_channels,
            hidden_channels=self.latent_state_dim,
            layer_num=common_layer_num,
            out_channels=self.latent_state_dim,
            activation=activation,
            norm_type='BN',
            output_activation=nn.Identity(),
            last_linear_layer_init_zero=last_linear_layer_init_zero
        )

        # lstm input: (batch_size, seq_len, input_size)
        # lstm output: (batch_size, seq_len, hidden_size)
        self.lstm = nn.LSTM(input_size=self.latent_state_dim, hidden_size=self.lstm_hidden_size)

        self.fc_reward_head = MLP(
            in_channels=self.lstm_hidden_size,
            hidden_channels=fc_reward_layers[0],
            layer_num=2,
            out_channels=output_support_size,
            activation=activation,
            norm_type='BN',
            output_activation=nn.Identity(),
            last_linear_layer_init_zero=last_linear_layer_init_zero
        )

    def forward(self, state_action_encoding: torch.Tensor, reward_hidden_state):
        """
        Overview:
            Forward function. Predict next latent state given current latent state, action and reward hidden state.
        Arguments:
            - state_action_encoding (:obj:`torch.Tensor`): current latent state and action encoding.
            - reward_hidden_state (:obj:`torch.Tensor`): reward hidden state. We use LSTM to predict value prefix.
        Returns:
            - state (:obj:`torch.Tensor`): next latent state.
            - reward_hidden_state (:obj:`torch.Tensor`): next reward hidden state.
            - value_prefix (:obj:`torch.Tensor`): value prefix.
        """
        state = self.fc_dynamics(state_action_encoding)
        state_unsqueeze = state.unsqueeze(0)
        value_prefix, reward_hidden_state = self.lstm(state_unsqueeze, reward_hidden_state)
        value_prefix = self.fc_reward_head(value_prefix.squeeze(0))

        return state, reward_hidden_state, value_prefix

    def get_dynamic_mean(self):
        return get_dynamic_mean(self)

    def get_reward_mean(self):
        return get_reward_mean(self)


class PredictionNetwork(nn.Module):

    def __init__(
            self,
            action_space_size,
            in_channels,
            common_layer_num: int = 2,
            fc_value_layers: SequenceType = [32],
            fc_policy_layers: SequenceType = [32],
            output_support_size: int = 601,
            last_linear_layer_init_zero: bool = True,
            activation: Optional[nn.Module] = nn.ReLU(inplace=True),
    ):
        """
        Overview:
            Prediction network. Predict the value and policy given hidden state.
        Arguments:
            - action_space_size: (:obj:`int`): Action space size, such as 6.
            - in_channels: (:obj:`int`): channels of hidden states.
            - common_layer_num (:obj:`int`): number of common layers.
            - fc_value_layers (:obj:`SequenceType`): hidden layers of the value prediction head (MLP head).
            - fc_policy_layers (:obj:`SequenceType`): hidden layers of the policy prediction head (MLP head).
            - output_support_size (:obj:`int`): dim of value output.
            - last_linear_layer_init_zero (:obj:`bool`): Whether to use zero initialization for the last layer of value/policy mlp, default set it to True.
            - activation (:obj:`Optional[nn.Module]`): the activation in Prediction network.
        """
        super().__init__()
        self.in_channels = in_channels

        self.fc_prediction_common = MLP(
            in_channels=self.in_channels,
            hidden_channels=self.in_channels,
            out_channels=self.in_channels,
            layer_num=common_layer_num,
            activation=activation,
            norm_type='BN',
            output_activation=nn.Identity(),
            last_linear_layer_init_zero=last_linear_layer_init_zero
        )

        self.fc_value_head = MLP(
            in_channels=self.in_channels,
            hidden_channels=fc_value_layers[0],
            out_channels=output_support_size,
            layer_num=2,
            activation=activation,
            norm_type='BN',
            output_activation=nn.Identity(),
            last_linear_layer_init_zero=last_linear_layer_init_zero
        )
        self.fc_policy_head = MLP(
            in_channels=self.in_channels,
            hidden_channels=fc_policy_layers[0],
            out_channels=action_space_size,
            layer_num=2,
            activation=activation,
            norm_type='BN',
            output_activation=nn.Identity(),
            last_linear_layer_init_zero=last_linear_layer_init_zero
        )

    def forward(self, x: torch.Tensor):
        """
        Overview:
            Forward computation of the prediction network.
        Arguments:
            - x (:obj:`torch.Tensor`): input tensor with shape (B, in_channels).
        Returns:
            - policy (:obj:`torch.Tensor`): policy tensor with shape (B, action_space_size).
            - value (:obj:`torch.Tensor`): value tensor with shape (B, output_support_size).
        """
        x_prediction_common = self.fc_prediction_common(x)

        value = self.fc_value_head(x_prediction_common)
        policy = self.fc_policy_head(x_prediction_common)
        return policy, value

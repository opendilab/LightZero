from typing import Optional, Tuple

import torch
import torch.nn as nn
from ding.model.common import ReparameterizationHead
from ding.torch_utils import MLP
from ding.utils import MODEL_REGISTRY, SequenceType

from .common import MZNetworkOutput, RepresentationNetworkMLP
from .muzero_model_mlp import DynamicsNetwork
from .utils import renormalize


@MODEL_REGISTRY.register('SampledMuZeroModelMLP')
class SampledMuZeroModelMLP(nn.Module):

    def __init__(
            self,
            observation_shape: int = 2,
            action_space_size: int = 6,
            latent_state_dim: int = 256,
            fc_reward_layers: SequenceType = [256],
            fc_value_layers: SequenceType = [256],
            fc_policy_layers: SequenceType = [256],
            reward_support_size: int = 601,
            value_support_size: int = 601,
            proj_hid: int = 1024,
            proj_out: int = 1024,
            pred_hid: int = 512,
            pred_out: int = 1024,
            self_supervised_learning_loss: bool = True,
            categorical_distribution: bool = True,
            activation: Optional[nn.Module] = nn.GELU(approximate='tanh'),
            last_linear_layer_init_zero: bool = True,
            state_norm: bool = False,
            # ==============================================================
            # specific sampled related config
            # ==============================================================
            continuous_action_space: bool = False,
            num_of_sampled_actions: int = 6,
            sigma_type='conditioned',
            fixed_sigma_value: float = 0.3,
            bound_type: str = None,
            norm_type: str = 'LN',
            discrete_action_encoding_type: str = 'one_hot',
            res_connection_in_dynamics: bool = True,
            *args,
            **kwargs,
    ):
        """
        Overview:
            The definition of the network model of Sampled MuZero, which is a generalization version for 1D vector obs.
            The networks are mainly built on fully connected layers.
            Sampled MuZero model consists of a representation network, a dynamics network and a prediction network.
            The representation network is an MLP network which maps the raw observation to a latent state.
            The dynamics network is an MLP+LSTM network which predicts the next latent state, reward_hidden_state and value_prefix given the current latent state and action.
            The prediction network is an MLP network which predicts the value and policy given the current latent state.
        Arguments:
            - observation_shape (:obj:`int`): Observation space shape, e.g. 8 for Lunarlander.
            - action_space_size: (:obj:`int`): Action space size, which is an integer number. For discrete action space, it is the num of discrete actions, \
                e.g. 4 for Lunarlander. For continuous action space, it is the dimension of the continuous action, e.g. 4 for bipedalwalker.
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
            - self_supervised_learning_loss (:obj:`bool`): Whether to use self_supervised_learning related networks in Sampled MuZero model, default set it to False.
            - categorical_distribution (:obj:`bool`): Whether to use discrete support to represent categorical distribution for value, reward/value_prefix.
            - activation (:obj:`Optional[nn.Module]`): Activation function used in network, which often use in-place \
                operation to speedup, e.g. ReLU(inplace=True).
            - last_linear_layer_init_zero (:obj:`bool`): Whether to use zero initializations for the last layer of value/policy mlp, default sets it to True.
            - state_norm (:obj:`bool`): Whether to use normalization for latent states, default sets it to True.
            # ==============================================================
            # specific sampled related config
            # ==============================================================
            - continuous_action_space (:obj:`bool`): The type of action space. default set it to False.
            - num_of_sampled_actions (:obj:`int`): the number of sampled actions, i.e. the K in original Sampled MuZero paper.
            # see ``ReparameterizationHead`` in ``ding.model.common.head`` for more details about the following arguments.
            - sigma_type (:obj:`str`): the type of sigma in policy head of prediction network, options={'conditioned', 'fixed'}.
            - fixed_sigma_value (:obj:`float`): the fixed sigma value in policy head of prediction network,
            - bound_type (:obj:`str`): The type of bound in networks.  Default sets it to None.
            - norm_type (:obj:`str`): The type of normalization in networks. default set it to 'BN'.
            - discrete_action_encoding_type (:obj:`str`): The type of encoding for discrete action. Default sets it to 'one_hot'. options = {'one_hot', 'not_one_hot'}
            - res_connection_in_dynamics (:obj:`bool`): Whether to use residual connection for dynamics network, default set it to False.
        """
        super(SampledMuZeroModelMLP, self).__init__()
        if not categorical_distribution:
            self.reward_support_size = 1
            self.value_support_size = 1
        else:
            self.reward_support_size = reward_support_size
            self.value_support_size = value_support_size

        self.continuous_action_space = continuous_action_space
        self.observation_shape = observation_shape
        self.action_space_size = action_space_size
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
        self.fc_reward_layers = fc_reward_layers
        self.fc_value_layers = fc_value_layers
        self.fc_policy_layers = fc_policy_layers
        self.proj_hid = proj_hid
        self.proj_out = proj_out
        self.pred_hid = pred_hid
        self.pred_out = pred_out

        self.last_linear_layer_init_zero = last_linear_layer_init_zero
        self.state_norm = state_norm
        self.self_supervised_learning_loss = self_supervised_learning_loss

        self.sigma_type = sigma_type
        self.fixed_sigma_value = fixed_sigma_value
        self.bound_type = bound_type
        self.norm_type = norm_type
        self.num_of_sampled_actions = num_of_sampled_actions
        self.res_connection_in_dynamics = res_connection_in_dynamics
        self.activation = activation

        self.representation_network = RepresentationNetworkMLP(
            observation_shape=self.observation_shape, hidden_channels=self.latent_state_dim, activation=self.activation,
            norm_type=self.norm_type
        )

        self.dynamics_network = DynamicsNetwork(
            action_encoding_dim=self.action_encoding_dim,
            num_channels=self.latent_state_dim + self.action_encoding_dim,
            common_layer_num=2,
            fc_reward_layers=self.fc_reward_layers,
            output_support_size=self.reward_support_size,
            last_linear_layer_init_zero=self.last_linear_layer_init_zero,
            activation=self.activation,
            norm_type=self.norm_type,
            res_connection_in_dynamics=self.res_connection_in_dynamics,
        )

        self.prediction_network = PredictionNetworkMLP(
            continuous_action_space=self.continuous_action_space,
            action_space_size=self.action_space_size,
            num_channels=self.latent_state_dim,
            fc_value_layers=self.fc_value_layers,
            fc_policy_layers=self.fc_policy_layers,
            output_support_size=self.value_support_size,
            last_linear_layer_init_zero=self.last_linear_layer_init_zero,
            activation=self.activation,
            sigma_type=self.sigma_type,
            fixed_sigma_value=self.fixed_sigma_value,
            bound_type=self.bound_type,
            norm_type=self.norm_type,
        )

        if self.self_supervised_learning_loss:
            # self_supervised_learning_loss related network proposed in EfficientZero
            self.projection_input_dim = latent_state_dim
            self.projection = nn.Sequential(
                nn.Linear(self.projection_input_dim, self.proj_hid), nn.BatchNorm1d(self.proj_hid), self.activation,
                nn.Linear(self.proj_hid, self.proj_hid), nn.BatchNorm1d(self.proj_hid), self.activation,
                nn.Linear(self.proj_hid, self.proj_out), nn.BatchNorm1d(self.proj_out)
            )
            self.prediction_head = nn.Sequential(
                nn.Linear(self.proj_out, self.pred_hid),
                nn.BatchNorm1d(self.pred_hid),
                self.activation,
                nn.Linear(self.pred_hid, self.pred_out),
            )

    def initial_inference(self, obs: torch.Tensor) -> MZNetworkOutput:
        """
         Overview:
            Initial inference of SampledMuZero model, which is the first step of the SampledMuZero model.
            To perform the initial inference, we first use the representation network to obtain the "latent_state" of the observation.
            Then we use the prediction network to predict the "value" and "policy_logits" of the "latent_state", and
            also prepare the zeros-like ``reward_hidden_state`` for the next step of the Sampled MuZero model.
        Arguments:
            - obs (:obj:`torch.Tensor`): The 1D vector observation data.
        Returns (MZNetworkOutput):
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
        return MZNetworkOutput(
            value,
            [0. for _ in range(batch_size)],
            policy_logits,
            latent_state,
        )

    def recurrent_inference(self, latent_state: torch.Tensor, action: torch.Tensor) -> MZNetworkOutput:
        """
        Overview:
            Recurrent inference of MuZero model, which is the rollout step of the MuZero model.
            To perform the recurrent inference, we first use the dynamics network to predict ``next_latent_state``,
            ``reward`` by the given current ``latent_state`` and ``action``.
             We then use the prediction network to predict the ``value`` and ``policy_logits``.
        Arguments:
            - latent_state (:obj:`torch.Tensor`): The encoding latent state of input obs.
            - action (:obj:`torch.Tensor`): The predicted action to rollout.
        Returns (MZNetworkOutput):
            - value (:obj:`torch.Tensor`): The output value of input state to help policy improvement and evaluation.
            - reward (:obj:`torch.Tensor`): The predicted reward for input state.
            - policy_logits (:obj:`torch.Tensor`): The output logit to select discrete action.
            - next_latent_state (:obj:`torch.Tensor`): The predicted next latent state.
        Shapes:
            - action (:obj:`torch.Tensor`): :math:`(B, )`, where B is batch_size.
            - value (:obj:`torch.Tensor`): :math:`(B, value_support_size)`, where B is batch_size.
            - reward (:obj:`torch.Tensor`): :math:`(B, reward_support_size)`, where B is batch_size.
            - policy_logits (:obj:`torch.Tensor`): :math:`(B, action_dim)`, where B is batch_size.
            - latent_state (:obj:`torch.Tensor`): :math:`(B, H)`, where B is batch_size, H is the dimension of latent state.
            - next_latent_state (:obj:`torch.Tensor`): :math:`(B, H)`, where B is batch_size, H is the dimension of latent state.
        """
        next_latent_state, reward = self._dynamics(latent_state, action)
        policy_logits, value = self._prediction(next_latent_state)
        return MZNetworkOutput(value, reward, policy_logits, next_latent_state)

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

    def _prediction(self, latent_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
        policy, value = self.prediction_network(latent_state)
        return policy, value

    def _dynamics(self, latent_state: torch.Tensor, action: torch.Tensor) -> Tuple[
        torch.Tensor, Tuple[torch.Tensor], torch.Tensor]:
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

        if not self.continuous_action_space:
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
                    # e.g., torch.Size([8]) ->  torch.Size([8, 1])
                    action_encoding = action_encoding.unsqueeze(-1)
        else:
            # continuous action space
            if len(action.shape) == 1:
                # (batch_size, ) -> (batch_size, 1)
                # e.g., torch.Size([8]) ->  torch.Size([8, 1])
                action = action.unsqueeze(-1)
            elif len(action.shape) == 3:
                # (batch_size, action_dim, 1) -> (batch_size,  action_dim)
                # e.g., torch.Size([8, 2, 1]) ->  torch.Size([8, 2])
                action = action.squeeze(-1)

            action_encoding = action

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

    def project(self, latent_state: torch.Tensor, with_grad=True) -> torch.Tensor:
        """
        Overview:
            Project the latent state to a lower dimension to calculate the self-supervised loss, which is proposed in MuZero.
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


class PredictionNetworkMLP(nn.Module):

    def __init__(
            self,
            continuous_action_space,
            action_space_size,
            num_channels,
            common_layer_num: int = 2,
            fc_value_layers: SequenceType = [32],
            fc_policy_layers: SequenceType = [32],
            output_support_size: int = 601,
            last_linear_layer_init_zero: bool = True,
            activation: Optional[nn.Module] = nn.GELU(approximate='tanh'),
            # ==============================================================
            # specific sampled related config
            # ==============================================================
            sigma_type='conditioned',
            fixed_sigma_value: float = 0.3,
            bound_type: str = None,
            norm_type: str = 'LN',
    ):
        """
        Overview:
            The definition of policy and value prediction network, which is used to predict value and policy by the
            given latent state.
            The networks are mainly built on fully connected layers.
        Arguments:
            - continuous_action_space (:obj:`bool`): The type of action space. default set it to False.
            - action_space_size: (:obj:`int`): Action space size, usually an integer number. For discrete action \
                space, it is the number of discrete actions. For continuous action space, it is the dimension of \
                continuous action.
            - num_channels (:obj:`int`): The num of channels in latent states.
            - num_res_blocks (:obj:`int`): The number of res blocks.
            - fc_value_layers (:obj:`SequenceType`): hidden layers of the value prediction head (MLP head).
            - fc_policy_layers (:obj:`SequenceType`): hidden layers of the policy prediction head (MLP head).
            - output_support_size (:obj:`int`): dim of value output.
            - last_linear_layer_init_zero (:obj:`bool`): Whether to use zero initializations for the last layer of value/policy mlp, default sets it to True.
            # ==============================================================
            # specific sampled related config
            # ==============================================================
            # see ``ReparameterizationHead`` in ``ding.model.common.head`` for more details about thee following arguments.
            - sigma_type (:obj:`str`): the type of sigma in policy head of prediction network, options={'conditioned', 'fixed'}.
            - fixed_sigma_value (:obj:`float`): the fixed sigma value in policy head of prediction network,
            - bound_type (:obj:`str`): The type of bound in networks.  default set it to None.
            - norm_type (:obj:`str`): The type of normalization in networks. default set it to 'BN'.
        """
        super().__init__()
        self.num_channels = num_channels
        self.continuous_action_space = continuous_action_space
        self.norm_type = norm_type
        self.sigma_type = sigma_type
        self.fixed_sigma_value = fixed_sigma_value
        self.bound_type = bound_type
        self.action_space_size = action_space_size
        if self.continuous_action_space:
            self.action_encoding_dim = self.action_space_size
        else:
            self.action_encoding_dim = 1

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
            layer_num=2,
            activation=activation,
            norm_type=norm_type,
            output_activation=False,
            output_norm=False,
            # last_linear_layer_init_zero=True is beneficial for convergence speed.
            last_linear_layer_init_zero=last_linear_layer_init_zero
        )

        # sampled related core code
        if self.continuous_action_space:
            self.fc_policy_head = ReparameterizationHead(
                input_size=self.num_channels,
                output_size=action_space_size,
                layer_num=2,
                sigma_type=self.sigma_type,
                fixed_sigma_value=self.fixed_sigma_value,
                activation=nn.ReLU(),
                norm_type=None,
                bound_type=self.bound_type
            )
        else:
            self.fc_policy_head = MLP(
                in_channels=self.num_channels,
                hidden_channels=fc_policy_layers[0],
                out_channels=action_space_size,
                layer_num=2,
                activation=activation,
                norm_type=self.norm_type,
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
             - latent_state (:obj:`torch.Tensor`): input tensor with shape (B, in_channels).
        Returns:
             - policy (:obj:`torch.Tensor`): policy tensor. If action space is discrete, shape is (B, action_space_size).
                If action space is continuous, shape is (B, action_space_size * 2).
             - value (:obj:`torch.Tensor`): value tensor with shape (B, output_support_size).
        """
        x_prediction_common = self.fc_prediction_common(latent_state)
        value = self.fc_value_head(x_prediction_common)

        # sampled related core code
        policy = self.fc_policy_head(x_prediction_common)
        if self.continuous_action_space:
            policy = torch.cat([policy['mu'], policy['sigma']], dim=-1)

        return policy, value

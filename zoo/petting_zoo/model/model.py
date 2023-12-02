import torch
import torch.nn as nn
from ding.model.common import FCEncoder
from ding.torch_utils import MLP, ResBlock

from lzero.model.common import RepresentationNetworkMLP
from typing import Optional, Tuple
import torch
import torch.nn as nn
from ding.torch_utils import MLP
from ding.utils import MODEL_REGISTRY, SequenceType
from lzero.model.utils import get_dynamic_mean, get_reward_mean
from numpy import ndarray

class PettingZooEncoder(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.agent_num = cfg.policy.model.agent_num
        self.agent_obs_shape = cfg.policy.model.agent_obs_shape
        self.global_obs_shape = cfg.policy.model.global_obs_shape
        self.agent_encoder  = RepresentationNetworkMLP(observation_shape=self.agent_obs_shape,
                                                       hidden_channels=256, 
                                                       norm_type='BN')
        
        self.global_encoder = RepresentationNetworkMLP(observation_shape=self.global_obs_shape, 
                                                       hidden_channels=256,
                                                       norm_type='BN')

    def forward(self, x):
        # agent
        agent_state = x['agent_state']
        agent_state = self.agent_encoder(agent_state)
        # agent_state = agent_state.reshape(batch_size, agent_num, -1)
        # global
        global_state = x['global_state']
        global_state = self.global_encoder(global_state)
        # global_state = global_state.reshape(batch_size, agent_num, -1)
        return (agent_state, global_state)

class PettingZooPrediction(nn.Module):

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
        self.fc_prediction_agent_common = MLP(
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

        self.fc_prediction_global_common = MLP(
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
        agent_state, global_state = latent_state
        global_state_common = self.fc_prediction_global_common(global_state)
        agent_state_common = self.fc_prediction_agent_common(agent_state)

        value = self.fc_value_head(global_state_common)
        policy = self.fc_policy_head(agent_state_common)
        return policy, value


class PettingZooMZDynamics(nn.Module):

    def __init__(
        self,
        action_encoding_dim: int = 2,
        num_channels: int = 64,
        common_layer_num: int = 2,
        fc_reward_layers: SequenceType = [32],
        output_support_size: int = 601,
        last_linear_layer_init_zero: bool = True,
        activation: Optional[nn.Module] = nn.ReLU(inplace=True),
        norm_type: Optional[str] = 'BN',
        res_connection_in_dynamics: bool = False,
    ):
        """
        Overview:
            The definition of dynamics network in MuZero algorithm, which is used to predict next latent state
            reward by the given current latent state and action.
            The networks are mainly built on fully connected layers.
        Arguments:
            - action_encoding_dim (:obj:`int`): The dimension of action encoding.
            - num_channels (:obj:`int`): The num of channels in latent states.
            - common_layer_num (:obj:`int`): The number of common layers in dynamics network.
            - fc_reward_layers (:obj:`SequenceType`): The number of hidden layers of the reward head (MLP head).
            - output_support_size (:obj:`int`): The size of categorical reward output.
            - last_linear_layer_init_zero (:obj:`bool`): Whether to use zero initializations for the last layer of value/policy mlp, default sets it to True.
            - activation (:obj:`Optional[nn.Module]`): Activation function used in network, which often use in-place \
                operation to speedup, e.g. ReLU(inplace=True).
            - norm_type (:obj:`str`): The type of normalization in networks. defaults to 'BN'.
            - res_connection_in_dynamics (:obj:`bool`): Whether to use residual connection in dynamics network.
        """
        super().__init__()
        self.num_channels = num_channels
        self.action_encoding_dim = action_encoding_dim
        self.latent_state_dim = self.num_channels - self.action_encoding_dim

        self.res_connection_in_dynamics = res_connection_in_dynamics
        if self.res_connection_in_dynamics:
            self.agent_fc_dynamics_1 = MLP(
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
            self.agent_fc_dynamics_2 = MLP(
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
            self.global_fc_dynamics_1 = MLP(
                in_channels=1024,
                hidden_channels=self.latent_state_dim,
                layer_num=common_layer_num,
                out_channels=256,
                activation=activation,
                norm_type=norm_type,
                output_activation=True,
                output_norm=True,
                # last_linear_layer_init_zero=False is important for convergence
                last_linear_layer_init_zero=False,
            )
            self.global_fc_dynamics_2 = MLP(
                in_channels=256,
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
            self.fc_alignment = MLP(
                in_channels=512,
                hidden_channels=self.latent_state_dim,
                out_channels=self.latent_state_dim,
                layer_num=common_layer_num,
                activation=activation,
                norm_type=norm_type,
                output_activation=True,
                output_norm=True,
                # last_linear_layer_init_zero=False is important for convergence
                last_linear_layer_init_zero=False,
            )
        else:
            self.agent_fc_dynamics = MLP(
                in_channels=261,
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
            self.global_fc_dynamics = MLP(
                in_channels=517,
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

        self.fc_reward_head = MLP(
            in_channels=self.latent_state_dim,
            hidden_channels=fc_reward_layers[0],
            layer_num=2,
            out_channels=output_support_size,
            activation=activation,
            norm_type=norm_type,
            output_activation=False,
            output_norm=False,
            last_linear_layer_init_zero=last_linear_layer_init_zero
        )

    def forward(self, state_action_encoding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Overview:
            Forward computation of the dynamics network. Predict the next latent state given current latent state and action.
        Arguments:
            - state_action_encoding (:obj:`torch.Tensor`): The state-action encoding, which is the concatenation of \
                    latent state and action encoding, with shape (batch_size, num_channels, height, width).
        Returns:
            - next_latent_state (:obj:`torch.Tensor`): The next latent state, with shape (batch_size, latent_state_dim).
            - reward (:obj:`torch.Tensor`): The predicted reward for input state.
        """
        if self.res_connection_in_dynamics:
            # # take the state encoding (e.g. latent_state),
            # # state_action_encoding[:, -self.action_encoding_dim:] is action encoding
            # latent_state = state_action_encoding[:, :-self.action_encoding_dim]
            # x = self.fc_dynamics_1(state_action_encoding)
            # # the residual link: add the latent_state to the state_action encoding
            # next_latent_state = x + latent_state
            # next_latent_state_encoding = self.fc_dynamics_2(next_latent_state)

            agent_state_action_encoding, global_latent_state = state_action_encoding
            batch_size = agent_state_action_encoding.shape[0] // 3
            # agent
            agent_latent_state = agent_state_action_encoding[:, :-self.action_encoding_dim]
            x = self.agent_fc_dynamics_1(agent_state_action_encoding)
            next_agent_latent_state = x + agent_latent_state
            # next_agent_latent_state_encoding = self.agent_fc_dynamics_2(next_agent_latent_state)
            # global
            next_agent_latent_state_tmp = next_agent_latent_state.reshape(batch_size, -1)
            global_latent_state = global_latent_state[::3, ]
            global_latent_state = torch.cat((next_agent_latent_state_tmp, global_latent_state), dim=1)
            next_global_latent_state = self.global_fc_dynamics_1(global_latent_state)  # x 512
            next_global_latent_state = next_global_latent_state.unsqueeze(1).expand(-1, 3, -1).reshape(-1, 256)
            next_global_latent_state_encoding = self.global_fc_dynamics_2(next_global_latent_state)
        else:
            agent_state_action_encoding, global_state_action_encoding = state_action_encoding
            next_agent_latent_state = self.agent_fc_dynamics(agent_state_action_encoding)
            next_global_latent_state = self.global_fc_dynamics(global_state_action_encoding)
            next_global_latent_state_encoding = next_global_latent_state

        reward = self.fc_reward_head(next_global_latent_state_encoding)

        return (next_agent_latent_state, next_global_latent_state), reward

    def get_dynamic_mean(self) -> float:
        return get_dynamic_mean(self)

    def get_reward_mean(self) -> float:
        return get_reward_mean(self)

class PettingZooEZDynamics(nn.Module):

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
            self.agent_fc_dynamics_1 = MLP(
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
            self.agent_fc_dynamics_2 = MLP(
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
            self.global_fc_dynamics_1 = MLP(
                in_channels=517,
                hidden_channels=self.latent_state_dim,
                layer_num=common_layer_num,
                out_channels=512,
                activation=activation,
                norm_type=norm_type,
                output_activation=True,
                output_norm=True,
                # last_linear_layer_init_zero=False is important for convergence
                last_linear_layer_init_zero=False,
            )
            self.global_fc_dynamics_2 = MLP(
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
            self.fc_alignment = MLP(
                in_channels=512,
                hidden_channels=self.latent_state_dim,
                out_channels=self.latent_state_dim,
                layer_num=common_layer_num,
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
            # # take the state encoding (latent_state), state_action_encoding[:, -self.action_encoding_dim]
            # # is action encoding
            # latent_state = state_action_encoding[:, :-self.action_encoding_dim]
            # x = self.fc_dynamics_1(state_action_encoding)
            # # the residual link: add state encoding to the state_action encoding
            # next_latent_state = x + latent_state
            # next_latent_state_ = self.fc_dynamics_2(next_latent_state)

            agent_state_action_encoding, global_state_action_encoding = state_action_encoding
            # agent
            agent_latent_state = agent_state_action_encoding[:, :-self.action_encoding_dim]
            x = self.agent_fc_dynamics_1(agent_state_action_encoding)
            next_agent_latent_state = x + agent_latent_state
            next_agent_latent_state_encoding = self.agent_fc_dynamics_2(next_agent_latent_state)
            # global
            global_latent_state = global_state_action_encoding[:, :-self.action_encoding_dim] #
            x = self.global_fc_dynamics_1(global_state_action_encoding)  # x 512
            next_global_latent_state = x + global_latent_state
            next_global_latent_state = self.fc_alignment(next_global_latent_state)
            next_latent_state_ = self.global_fc_dynamics_2(next_global_latent_state)
        else:
            next_latent_state = self.fc_dynamics(state_action_encoding)
            next_latent_state_ = next_latent_state

        next_latent_state_unsqueeze = next_latent_state_.unsqueeze(0)
        value_prefix, next_reward_hidden_state = self.lstm(next_latent_state_unsqueeze, reward_hidden_state)
        value_prefix = self.fc_reward_head(value_prefix.squeeze(0))

        return (next_agent_latent_state, next_global_latent_state), next_reward_hidden_state, value_prefix

    def get_dynamic_mean(self) -> float:
        return get_dynamic_mean(self)

    def get_reward_mean(self) -> Tuple[ndarray, float]:
        return get_reward_mean(self)
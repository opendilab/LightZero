import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from ding.model.common import ReparameterizationHead
from ding.torch_utils import MLP, ResBlock
from ding.utils import MODEL_REGISTRY, SequenceType

from .common import EZNetworkOutput, RepresentationNetwork
from .utils import renormalize, get_dynamic_mean, get_reward_mean, get_params_mean


@MODEL_REGISTRY.register('SampledEfficientZeroModel')
class SampledEfficientZeroModel(nn.Module):

    def __init__(
        self,
        observation_shape: SequenceType = (12, 96, 96),
        action_space_size: int = 6,
        representation_network_type: str = 'conv_res_blocks',
        categorical_distribution: bool = True,
        activation: Optional[nn.Module] = nn.ReLU(inplace=True),
        representation_network: nn.Module = None,
        batch_norm_momentum: float = 0.1,
        last_linear_layer_init_zero: bool = True,
        state_norm: bool = False,
        downsample: bool = False,
        num_res_blocks: int = 1,
        num_channels: int = 64,
        lstm_hidden_size: int = 512,
        # the following model para. is usually fixed
        reward_head_channels: int = 16,
        value_head_channels: int = 16,
        policy_head_channels: int = 16,
        fc_reward_layers: SequenceType = [32],
        fc_value_layers: SequenceType = [32],
        fc_policy_layers: SequenceType = [32],
        reward_support_size: int = 601,
        value_support_size: int = 601,
        proj_hid: int = 1024,
        proj_out: int = 1024,
        pred_hid: int = 512,
        pred_out: int = 1024,
        # the above model para. is usually fixed
        self_supervised_learning_loss: bool = False,
        # ==============================================================
        # specific sampled related config
        # ==============================================================
        continuous_action_space: bool = False,
        num_of_sampled_actions: int = 6,
        sigma_type='conditioned',
        fixed_sigma_value: float = 0.3,
        bound_type: str = None,
        norm_type: str = 'BN',
        *args,
        **kwargs,
    ):
        """
        Overview:
            Sampled EfficientZero network
        Arguments:
            - observation_shape (:obj:`SequenceType`): Observation space shape, e.g. [C, W, H]=[12, 96, 96].
            - action_space_size: (:obj:`int`): Action space size, such as 6.
            - representation_network_type (:obj:`Optional[str]`): The type of representation_network in Sampled EfficientZero model. options={'conv_res_blocks', 'identity'}
            - categorical_distribution (:obj:`bool`): Whether to use discrete support to represent categorical distribution for value, reward/value_prefix.
            - activation (:obj:`Optional[nn.Module]`): the activation in Sampled EfficientZero model.
            - representation_network (:obj:`nn.Module`): the user-defined representation_network.
            - batch_norm_momentum (:obj:`float`):  Momentum of BN
            - last_linear_layer_init_zero (:obj:`bool`): Whether to use zero initialization for the last layer of value/policy mlp, default set it to True.
            - state_norm (:obj:`bool`): Whether to use normalization for hidden states, default set it to True.
            - downsample (:obj:`bool`): Whether to do downsampling for observations in ``representation_network``, default set it to True. \
                But sometimes, we do not need, e.g. board games.
            - num_res_blocks (:obj:`int`): number of res blocks in Sampled EfficientZero model.
            - num_channels (:obj:`int`): channels of hidden states.
            - lstm_hidden_size (:obj:`int`): dim of lstm hidden state in dynamics network.
            - reward_head_channels (:obj:`int`): channels of reward head.
            - value_head_channels (:obj:`int`): channels of value head.
            - policy_head_channels (:obj:`int`): channels of policy head.
            - fc_reward_layers (:obj:`SequenceType`): hidden layers of the reward prediction head (MLP head).
            - fc_value_layers (:obj:`SequenceType`): hidden layers of the value prediction head (MLP head).
            - fc_policy_layers (:obj:`SequenceType`): hidden layers of the policy prediction head (MLP head).
            - reward_support_size (:obj:`int`): dim of reward output
            - value_support_size (:obj:`int`): dim of value output.
            - proj_hid (:obj:`int`): dim of projection hidden layer.
            - proj_out (:obj:`int`): dim of projection output layer.
            - pred_hid (:obj:`int`):dim of prediction hidden layer.
            - pred_out (:obj:`int`): dim of prediction output layer.
            - self_supervised_learning_loss (:obj:`bool`): Whether to use self_supervised_learning related networks in MuZero model, default set it to False.
            # ==============================================================
            # specific sampled related config
            # ==============================================================
            - continuous_action_space (:obj:`bool`): The type of action space. default set it to False.
            - num_of_sampled_actions (:obj:`int`): the number of sampled actions, i.e. the K in original Sampled MuZero paper.
            # see ``ReparameterizationHead`` in ``ding.model.cmmon.head`` for more details about thee following arguments.
            - sigma_type (:obj:`str`): the type of sigma in policy head of prediction network, options={'conditioned', 'fixed'}.
            - fixed_sigma_value (:obj:`float`): the fixed sigma value in policy head of prediction network,
            - bound_type (:obj:`str`): The type of bound in networks.  default set it to None.
            - norm_type (:obj:`str`): The type of normalization in networks. default set it to 'BN'.
        """
        super(SampledEfficientZeroModel, self).__init__()
        self.representation_network_type = representation_network_type
        assert self.representation_network_type in ['identity', 'conv_res_blocks']
        self.sigma_type = sigma_type
        self.fixed_sigma_value = fixed_sigma_value
        self.bound_type = bound_type
        self.norm_type = norm_type

        self.continuous_action_space = continuous_action_space
        self.num_of_sampled_actions = num_of_sampled_actions
        self.action_space_size = action_space_size

        self.self_supervised_learning_loss = self_supervised_learning_loss
        self.categorical_distribution = categorical_distribution
        if not self.categorical_distribution:
            self.reward_support_size = 1
            self.value_support_size = 1
        else:
            self.reward_support_size = reward_support_size
            self.value_support_size = value_support_size

        self.proj_hid = proj_hid
        self.proj_out = proj_out
        self.pred_hid = pred_hid
        self.pred_out = pred_out
        self.last_linear_layer_init_zero = last_linear_layer_init_zero
        self.state_norm = state_norm
        self.representation_network = representation_network
        self.downsample = downsample
        self.lstm_hidden_size = lstm_hidden_size

        if isinstance(observation_shape, int) or len(observation_shape) == 1:
            # vector obs input, e.g. classical control ad box2d environments
            # to be compatible with LightZero model/policy, transform to shape: [C, W, H]
            observation_shape = [1, observation_shape, 1]

        flatten_output_size_for_reward_head = (
            (reward_head_channels * math.ceil(observation_shape[1] / 16) *
             math.ceil(observation_shape[2] / 16)) if downsample else
            (reward_head_channels * observation_shape[1] * observation_shape[2])
        )

        flatten_output_size_for_value_head = (
            (value_head_channels * math.ceil(observation_shape[1] / 16) *
             math.ceil(observation_shape[2] / 16)) if downsample else
            (value_head_channels * observation_shape[1] * observation_shape[2])
        )

        flatten_output_size_for_policy_head = (
            (policy_head_channels * math.ceil(observation_shape[1] / 16) *
             math.ceil(observation_shape[2] / 16)) if downsample else
            (policy_head_channels * observation_shape[1] * observation_shape[2])
        )

        if self.representation_network is None:
            if self.representation_network_type == 'identity':
                self.representation_network = nn.Identity()
            elif self.representation_network_type == 'conv_res_blocks':
                self.representation_network = RepresentationNetwork(
                    observation_shape,
                    num_res_blocks,
                    num_channels,
                    downsample,
                    momentum=batch_norm_momentum,
                    norm_type=norm_type,
                )
        else:
            self.representation_network = self.representation_network

        if self.representation_network_type == 'identity':
            self.dynamics_network = DynamicsNetwork(
                num_res_blocks,
                observation_shape[0] + 1,  # in_channels=observation_shape[0]
                1,
                reward_head_channels,
                fc_reward_layers,
                self.reward_support_size,
                flatten_output_size_for_reward_head,
                lstm_hidden_size=lstm_hidden_size,
                momentum=batch_norm_momentum,
                last_linear_layer_init_zero=self.last_linear_layer_init_zero,
                norm_type=norm_type,
            )
            self.prediction_network = PredictionNetwork(
                self.continuous_action_space,
                action_space_size,
                num_res_blocks,
                observation_shape[0],  # in_channels
                num_channels,
                value_head_channels,
                policy_head_channels,
                fc_value_layers,
                fc_policy_layers,
                self.value_support_size,
                flatten_output_size_for_value_head,
                flatten_output_size_for_policy_head,
                momentum=batch_norm_momentum,
                last_linear_layer_init_zero=self.last_linear_layer_init_zero,
                sigma_type=self.sigma_type,
                fixed_sigma_value=self.fixed_sigma_value,
                bound_type=self.bound_type,
                norm_type=self.norm_type,
            )
        else:
            if self.continuous_action_space:
                self.dynamics_network = DynamicsNetwork(
                    num_res_blocks,
                    num_channels + self.action_space_size,
                    self.action_space_size,
                    reward_head_channels,
                    fc_reward_layers,
                    self.reward_support_size,
                    flatten_output_size_for_reward_head,
                    lstm_hidden_size=lstm_hidden_size,
                    momentum=batch_norm_momentum,
                    last_linear_layer_init_zero=self.last_linear_layer_init_zero,
                    norm_type=self.norm_type,
                )
            else:
                self.dynamics_network = DynamicsNetwork(
                    num_res_blocks,
                    num_channels + 1,
                    1,
                    reward_head_channels,
                    fc_reward_layers,
                    self.reward_support_size,
                    flatten_output_size_for_reward_head,
                    lstm_hidden_size=lstm_hidden_size,
                    momentum=batch_norm_momentum,
                    last_linear_layer_init_zero=self.last_linear_layer_init_zero,
                    norm_type=self.norm_type,
                )
            self.prediction_network = PredictionNetwork(
                self.continuous_action_space,
                action_space_size,
                num_res_blocks,
                None,  # in_channels
                num_channels,
                value_head_channels,
                policy_head_channels,
                fc_value_layers,
                fc_policy_layers,
                self.value_support_size,
                flatten_output_size_for_value_head,
                flatten_output_size_for_policy_head,
                momentum=batch_norm_momentum,
                last_linear_layer_init_zero=self.last_linear_layer_init_zero,
                sigma_type=self.sigma_type,
                fixed_sigma_value=self.fixed_sigma_value,
                bound_type=self.bound_type,
                norm_type=self.norm_type,
            )

        if self.self_supervised_learning_loss:
            # projection
            if self.representation_network_type == 'identity':
                self.projection_input_dim = observation_shape[0] * observation_shape[1] * observation_shape[2]
            else:
                if self.downsample:
                    # for atari, due to downsample
                    # observation_shape=(12, 96, 96),  # stack=4
                    # 3 * 96/16 * 96/16 = 3*6*6 = 108
                    self.projection_input_dim = num_channels * math.ceil(observation_shape[1] / 16
                                                                         ) * math.ceil(observation_shape[2] / 16)
                else:
                    self.projection_input_dim = num_channels * observation_shape[1] * observation_shape[2]

            self.projection = nn.Sequential(
                nn.Linear(self.projection_input_dim, self.proj_hid), nn.BatchNorm1d(self.proj_hid), activation,
                nn.Linear(self.proj_hid, self.proj_hid), nn.BatchNorm1d(self.proj_hid), activation,
                nn.Linear(self.proj_hid, self.proj_out), nn.BatchNorm1d(self.proj_out)
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
             SEZNetworkOutput
                - value (:obj:`torch.Tensor`): (batch_size, 1).
                - policy_logits (:obj:`torch.Tensor`): (batch_size, action_dim).
                - latent_state (:obj:`torch.Tensor`): (batch_size, 1, 1) e.g. (1, 1, 1).
                - reward_hidden (:obj:`torch.Tensor`): (batch_size, 1, 1) e.g. (1, 1, 1).
         """
        num = obs.size(0)
        hidden_state = self._representation(obs)
        policy_logits, value = self._prediction(hidden_state)
        # zero initialization for reward hidden states
        reward_hidden = (torch.zeros(1, num, self.lstm_hidden_size).to(obs.device), torch.zeros(1, num, self.lstm_hidden_size).to(obs.device))
        return EZNetworkOutput(value, [0. for _ in range(num)], policy_logits, hidden_state, reward_hidden)

    def recurrent_inference(
            self, hidden_state: torch.Tensor, reward_hidden: torch.Tensor, action: torch.Tensor
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
             SEZNetworkOutput
                - value (:obj:`torch.Tensor`): (batch_size, 1).
                - value_prefix (:obj:`torch.Tensor`): (batch_size, 1).
                - policy_logits (:obj:`torch.Tensor`): (batch_size, action_dim).
                - latent_state (:obj:`torch.Tensor`): (batch_size, 1, 1) e.g. (1, 1, 1).
                - reward_hidden (:obj:`torch.Tensor`): (batch_size, 1, 1) e.g. (1, 1, 1).
         """
        hidden_state, reward_hidden, value_prefix = self._dynamics(hidden_state, reward_hidden, action)
        policy_logits, value = self._prediction(hidden_state)
        return EZNetworkOutput(value, value_prefix, policy_logits, hidden_state, reward_hidden)

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
        policy, value = self.prediction_network(latent_state)
        return policy, value

    def _representation(self, observation: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Overview:
            Representation network. Encode the observations into latent state.
        Arguments:
             - observation (:obj:`torch.Tensor`): (batch_size, num_channel, obs_shape[1], obs_shape[2]), e.g. (1,64,6,6).
        Returns:
            - latent_state (:obj:`torch.Tensor`): (batch_size, 1, 1) e.g. (1, 1, 1).
        """
        if len(observation.shape) == 1:
            # vector obs input, e.g. classical control ad box2d environments
            # to be compatible with LightZero model/policy, shape: [C, W, H]
            observation = observation.reshape(1, observation.shape[0], 1)
        latent_state = self.representation_network(observation)
        if not self.state_norm:
            return latent_state
        else:
            latent_state_normalized = renormalize(latent_state)
            return latent_state_normalized

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
        if not self.continuous_action_space:
            # discrete action space
            # stack latent_state with a game specific one hot encoded action
            #  action_one_hot (batch_siize, 1, obs_shape[1], obs_shape[2]), e.g. (4,1,6,6)
            action_one_hot = (
                torch.ones((
                    latent_state.shape[0],
                    1,
                    latent_state.shape[2],
                    latent_state.shape[3],
                )).to(action.device).float()
            )
            if len(action.shape) == 2:
                # (batch_size, action_dim) -> (batch_size, action_dim, 1)
                # e.g.,  torch.Size([4, 1]) ->  torch.Size([4, 1, 1])
                action = action.unsqueeze(-1)
            elif len(action.shape) == 1:
                # (action_dim) -> (1, action_dim, 1)
                # e.g.,  torch.Size([4, 1]) ->  torch.Size([4, 1, 1])
                action = action.unsqueeze(0).unsqueeze(-1)

            # action[:, 0, None, None] shape: (4, 1, 1, 1)
            action_one_hot = (action[:, 0, None, None] * action_one_hot / self.action_space_size)
            state_action_encoding = torch.cat((latent_state, action_one_hot), dim=1)

        else:  # continuous action space
            action_one_hot = (
                torch.ones((
                    latent_state.shape[0],
                    1,
                    latent_state.shape[2],
                    latent_state.shape[3],
                )).to(action.device).float()
            )

            if len(action.shape) == 2:
                # (batch_size, action_dim) -> (batch_size, action_dim, 1)
                action = action.unsqueeze(-1)

            # len(action.shape)==3:
            # action: (8,2,1)
            # action_one_hot (batch_siize, 1, obs_shape[1], obs_shape[2]) : (8,1,8,1)
            # action[:, 0, None, None]: 8,1,1,1
            # action_embedding: 8,2,8,1
            try:
                action_embedding = torch.cat(
                    [action[:, dim, None, None] * action_one_hot for dim in range(self.action_space_size)], dim=1
                )
            except Exception as error:
                print(error)
                print(action.shape, action_one_hot.shape)

            state_action_encoding = torch.cat((latent_state, action_embedding), dim=1)

        try:
            next_latent_state, next_reward_hidden_state, value_prefix = self.dynamics_network(
                state_action_encoding, reward_hidden_state
            )
        except Exception as error:
            print(error)
            print(latent_state.shape, action_embedding.shape)
            print(state_action_encoding.shape)

        if not self.state_norm:
            return next_latent_state, next_reward_hidden_state, value_prefix
        else:
            next_latent_state_normalized = renormalize(next_latent_state)
            return next_latent_state_normalized, next_reward_hidden_state, value_prefix

    def project(self, hidden_state: torch.Tensor, with_grad=True):
        """
        Overview:
            only used when ``self.self_supervised_learning_loss=True``.
            Please refer to paper ``Exploring Simple Siamese Representation Learning`` for details.
        # only the branch of proj + pred can share the gradients
        Examples:
            # for lunarlander:
            # observation_shape = (4, 8, 1),  # stack=4
            # self.projection_input_dim = 64*8*1
            # hidden_state.shape: (batch_size, num_channel, obs_shape[1], obs_shape[2])  256,64,8,1
            # 256,64,8,1 -> 256,64*8*1

            # for atari:
            # observation_shape = (12, 96, 96),  # 3,96,96 stack=4
            # self.projection_input_dim = 3*6*6 = 108
            # hidden_state.shape: (batch_size, num_channel, obs_shape[1]/16, obs_shape[2]/16)  256,64,96/16,96/16 = 256,64,6,6
            # 256, 64, 6, 6 -> 256,64*6*6
        """

        # hidden_state.shape[0] = batch_size
        hidden_state = hidden_state.reshape(hidden_state.shape[0], -1)

        proj = self.projection(hidden_state)

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
        num_res_blocks,
        num_channels,
        action_space_size,
        reward_head_channels,
        fc_reward_layers,
        output_support_size,
        flatten_output_size_for_reward_head,
        lstm_hidden_size,
        momentum: float = 0.1,
        last_linear_layer_init_zero: bool = True,
        activation: Optional[nn.Module] = nn.ReLU(inplace=True),
        norm_type: str = 'BN',
    ):
        """
        Overview:
            Dynamics network. Predict next hidden states, reward_hidden_state, and value_prefix given current state and action.
        Arguments:
            - num_res_blocks (:obj:`int`): number of res blocks.
            - num_channels (:obj:`int`): channels of hidden states.
            - fc_reward_layers (:obj:`list`):  hidden layers of the reward prediction head (MLP head)
            - output_support_size (:obj:`int`): dim of reward output
            - flatten_output_size_for_reward_head (:obj:`int`): dim of flatten hidden states
            - lstm_hidden_size (:obj:`int`): dim of lstm hidden state in dynamics network.
            - last_linear_layer_init_zero (:obj:`bool`): Whether to use zero initialization for the last layer of value/policy mlp, default set it to True.
            - activation (:obj:`Optional[nn.Module]`): the activation in Dynamics network.
            - norm_type (:obj:`str`): The type of normalization in networks. default set it to 'BN'.
        """
        super().__init__()
        self.num_channels = num_channels

        self.norm_type = norm_type
        self.lstm_hidden_size = lstm_hidden_size
        self.action_space_size = action_space_size
        assert num_channels > self.action_space_size, f'num_channels:{num_channels} <= action_space_size:{self.action_space_size}'
        self.conv = nn.Conv2d(
            num_channels, num_channels - self.action_space_size, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(num_channels - self.action_space_size, momentum=momentum)
        self.resblocks = nn.ModuleList(
            [
                ResBlock(
                    in_channels=num_channels - self.action_space_size,
                    activation=activation,
                    norm_type=self.norm_type,
                    res_type='basic',
                    bias=False
                ) for _ in range(num_res_blocks)
            ]
        )

        self.reward_resblocks = nn.ModuleList(
            [
                ResBlock(
                    in_channels=num_channels - self.action_space_size,
                    activation=activation,
                    norm_type=self.norm_type,
                    res_type='basic',
                    bias=False
                ) for _ in range(num_res_blocks)
            ]
        )

        self.conv1x1_reward = nn.Conv2d(num_channels - self.action_space_size, reward_head_channels, 1)
        self.bn_reward = nn.BatchNorm2d(reward_head_channels, momentum=momentum)
        self.flatten_output_size_for_reward_head = flatten_output_size_for_reward_head
        self.lstm = nn.LSTM(input_size=self.flatten_output_size_for_reward_head, hidden_size=self.lstm_hidden_size)
        self.bn_value_prefix = nn.BatchNorm1d(self.lstm_hidden_size, momentum=momentum)
        self.fc = MLP(
            in_channels=self.lstm_hidden_size,
            hidden_channels=fc_reward_layers[0],
            out_channels=output_support_size,
            layer_num=len(fc_reward_layers) + 1,
            activation=activation,
            norm_type=self.norm_type,
            output_activation=nn.Identity(),
            output_norm_type=None,
            last_linear_layer_init_zero=last_linear_layer_init_zero
        )
        self.activation = activation

    def forward(self, x: torch.Tensor, reward_hidden_state: torch.Tensor):
        # take the state encoding
        state = x[:, :-self.action_space_size, :, :]
        x = self.conv(x)
        x = self.bn(x)

        x += state
        x = self.activation(x)

        for block in self.resblocks:
            x = block(x)
        state = x

        x = self.conv1x1_reward(x)
        x = self.bn_reward(x)
        x = self.activation(x)

        x = x.contiguous().view(-1, self.flatten_output_size_for_reward_head).unsqueeze(0)
        value_prefix, reward_hidden_state = self.lstm(x, reward_hidden_state)
        value_prefix = value_prefix.squeeze(0)
        value_prefix = self.bn_value_prefix(value_prefix)
        value_prefix = self.activation(value_prefix)
        value_prefix = self.fc(value_prefix)

        return state, reward_hidden_state, value_prefix

    def get_dynamic_mean(self):
        return get_dynamic_mean(self)

    def get_reward_mean(self):
        return get_reward_mean(self)


class PredictionNetwork(nn.Module):

    def __init__(
        self,
        continuous_action_space,
        action_space_size,
        num_res_blocks,
        in_channels,
        num_channels,
        value_head_channels,
        policy_head_channels,
        fc_value_layers,
        fc_policy_layers,
        output_support_size,
        flatten_output_size_for_value_head,
        flatten_output_size_for_policy_head,
        momentum: float = 0.1,
        last_linear_layer_init_zero: bool = True,
        activation: Optional[nn.Module] = nn.ReLU(inplace=True),
        # ==============================================================
        # specific sampled related config
        # ==============================================================
        sigma_type='conditioned',
        fixed_sigma_value: float = 0.3,
        bound_type: str = None,
        norm_type: str = 'BN',
    ):
        """
        Overview:
            Prediction network. predict the value and policy given hidden states
        Arguments:
            - continuous_action_space (:obj:`bool`): The type of action space. default set it to False.
            - action_space_size: (:obj:`int`): Action space size, such as 6.
            - num_res_blocks (:obj:`int`): number of res blocks in model.
            - in_channels (:obj:`int`): channels of input, if None, then in_channels = num_channels
            - num_channels (:obj:`int`): channels of hidden states.
            - value_head_channels (:obj:`int`): channels of value head.
            - policy_head_channels (:obj:`int`): channels of policy head.
            - fc_value_layers (:obj:`SequenceType`): hidden layers of the value prediction head (MLP head).
            - fc_policy_layers (:obj:`SequenceType`): hidden layers of the policy prediction head (MLP head).
            - output_support_size (:obj:`int`): dim of value output.
            - flatten_output_size_for_value_head (:obj:`int`): dim of flatten hidden states.
            - flatten_output_size_for_policy_head (:obj:`int`): dim of flatten hidden states.
            - last_linear_layer_init_zero (:obj:`bool`): Whether to use zero initialization for the last layer of value/policy mlp, default set it to True.
            # ==============================================================
            # specific sampled related config
            # ==============================================================
            # see ``ReparameterizationHead`` in ``ding.model.cmmon.head`` for more details about thee following arguments.
            - sigma_type (:obj:`str`): the type of sigma in policy head of prediction network, options={'conditioned', 'fixed'}.
            - fixed_sigma_value (:obj:`float`): the fixed sigma value in policy head of prediction network,
            - bound_type (:obj:`str`): The type of bound in networks.  default set it to None.
            - norm_type (:obj:`str`): The type of normalization in networks. default set it to 'BN'.
        """
        super().__init__()
        self.in_channels = in_channels

        self.continuous_action_space = continuous_action_space
        self.norm_type = norm_type
        self.sigma_type = sigma_type
        self.fixed_sigma_value = fixed_sigma_value
        self.bound_type = bound_type

        if self.in_channels is not None:
            self.conv_input = nn.Conv2d(in_channels, num_channels, 1)

        self.resblocks = nn.ModuleList(
            [
                ResBlock(
                    in_channels=num_channels,
                    activation=activation,
                    norm_type=self.norm_type,
                    res_type='basic',
                    bias=False
                ) for _ in range(num_res_blocks)
            ]
        )

        self.conv1x1_value = nn.Conv2d(num_channels, value_head_channels, 1)
        self.conv1x1_policy = nn.Conv2d(num_channels, policy_head_channels, 1)
        self.bn_value = nn.BatchNorm2d(value_head_channels, momentum=momentum)
        self.bn_policy = nn.BatchNorm2d(policy_head_channels, momentum=momentum)
        self.flatten_output_size_for_value_head = flatten_output_size_for_value_head
        self.flatten_output_size_for_policy_head = flatten_output_size_for_policy_head
        # TODO(pu)
        self.fc_value = MLP(
            in_channels=self.flatten_output_size_for_value_head,
            hidden_channels=fc_value_layers[0],
            out_channels=output_support_size,
            layer_num=len(fc_value_layers) + 1,
            activation=activation,
            norm_type=self.norm_type,
            output_activation=nn.Identity(),
            output_norm_type=None,
            last_linear_layer_init_zero=last_linear_layer_init_zero
        )

        ######################
        # sampled related core code
        ######################

        if self.continuous_action_space:
            try:
                self.sampled_fc_policy = ReparameterizationHead(
                    input_size=self.flatten_output_size_for_policy_head,  # 256,
                    output_size=action_space_size,
                    layer_num=len(fc_policy_layers) + 1,
                    sigma_type=self.sigma_type,
                    fixed_sigma_value=self.fixed_sigma_value,
                    activation=nn.ReLU(),
                    norm_type=None,
                    bound_type=self.bound_type  # TODO(pu)
                )
            except:
                self.sampled_fc_policy = ReparameterizationHead(
                    hidden_size=self.flatten_output_size_for_policy_head,  # 256,
                    output_size=action_space_size,
                    layer_num=len(fc_policy_layers) + 1,
                    sigma_type=self.sigma_type,
                    fixed_sigma_value=self.fixed_sigma_value,
                    activation=nn.ReLU(),
                    norm_type=None,
                    bound_type=self.bound_type  # TODO(pu)
                )
        else:
            self.sampled_fc_policy = MLP(
                in_channels=self.flatten_output_size_for_policy_head,
                hidden_channels=fc_policy_layers[0],
                out_channels=action_space_size,
                layer_num=len(fc_policy_layers) + 1,
                activation=activation,
                norm_type=self.norm_type,
                output_activation=nn.Identity(),
                output_norm_type=None,
                last_linear_layer_init_zero=last_linear_layer_init_zero
            )

        self.activation = activation

    def forward(self, x: torch.Tensor):
        if self.in_channels is not None:
            x = self.conv_input(x)

        for block in self.resblocks:
            x = block(x)
        value = self.conv1x1_value(x)
        value = self.bn_value(value)
        value = self.activation(value)

        policy = self.conv1x1_policy(x)
        policy = self.bn_policy(policy)
        policy = self.activation(policy)

        # print(value.shape, value)
        # value = value.view(-1, self.flatten_output_size_for_value_head)
        # policy = policy.view(-1, self.flatten_output_size_for_policy_head)
        # TODO
        value = value.reshape(-1, self.flatten_output_size_for_value_head)
        policy = policy.reshape(-1, self.flatten_output_size_for_policy_head)
        value = self.fc_value(value)
        # policy = self.fc_policy(policy)

        ######################
        # sampled related core code
        ######################

        #  {'mu': mu, 'sigma': sigma}
        policy = self.sampled_fc_policy(policy)

        # print("policy['mu']", policy['mu'].max(), policy['mu'].min(), policy['mu'].std())
        # print("policy['sigma']", policy['sigma'].max(), policy['sigma'].min(), policy['sigma'].std())
        if self.continuous_action_space:
            policy = torch.cat([policy['mu'], policy['sigma']], dim=-1)

        return policy, value


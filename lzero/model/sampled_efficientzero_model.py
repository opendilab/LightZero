"""
The following code is adapted from https://github.com/YeWR/EfficientZero/blob/main/config/atari/model.py
"""

import math
from typing import Optional

import torch
import torch.nn as nn
from ding.model.common import ReparameterizationHead
from ding.torch_utils import MLP, ResBlock
from ding.utils import MODEL_REGISTRY, SequenceType

from .common import EZNetworkOutput, RepresentationNetwork
from .utils import renormalize, get_dynamic_mean, get_reward_mean, get_params_mean


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
        lstm_hidden_size=64,
        momentum=0.1,
        last_linear_layer_init_zero=False,
        norm_type='BN',
    ):
        """
        Overview:
            Dynamics network. Predict next hidden states, reward_hidden_state, and value_prefix given current states and actions
        Arguments:
            - num_res_blocks (:obj:int): number of res blocks
            - num_channels (:obj:int): channels of hidden states
            - fc_reward_layers (:obj:list):  hidden layers of the reward prediction head (MLP head)
            - output_support_size (:obj:int): dim of reward output
            - flatten_output_size_for_reward_head (:obj:int): dim of flatten hidden states
            - lstm_hidden_size (:obj:int): dim of lstm hidden
            - last_linear_layer_init_zero (:obj:bool): if True -> zero initialization for the last layer of reward mlp
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
                    activation=torch.nn.ReLU(inplace=True),
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
                    activation=torch.nn.ReLU(inplace=True),
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
        # TODO(pu)
        self.fc = MLP(
            in_channels=self.lstm_hidden_size,
            hidden_channels=fc_reward_layers[0],
            out_channels=output_support_size,
            layer_num=len(fc_reward_layers) + 1,
            activation=nn.ReLU(inplace=True),
            norm_type=self.norm_type,
            output_activation=nn.Identity(),
            output_norm_type=None,
            last_linear_layer_init_zero=last_linear_layer_init_zero
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x, reward_hidden_state):
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

        # RuntimeError: view size is not compatible with input tensor size and stride (at least one dimension spans
        # across two contiguous subspaces) 100,2,3,3 -> 1,10,180
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


# predict the value and policy given hidden states
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
        momentum=0.1,
        last_linear_layer_init_zero=False,
        sigma_type='fixed',
        fixed_sigma_value=0.3,
        bound_type=None,
        norm_type='BN',
    ):
        """Prediction network
        Parameters
        ----------
        action_space_size: int
            action space
        num_res_blocks: int
            number of res blocks
        in_channels: int
            channels of input
        num_channels: int
            channels of hidden states
        value_head_channels: int
            channels of value head
        policy_head_channels: int
            channels of policy head
        fc_value_layers: list
            hidden layers of the value prediction head (MLP head)
        fc_policy_layers: list
            hidden layers of the policy prediction head (MLP head)
        output_support_size: int
            dim of value output
        flatten_output_size_for_value_head: int
            dim of flatten hidden states
        flatten_output_size_for_policy_head: int
            dim of flatten hidden states
        last_linear_layer_init_zero: bool
            True -> zero initialization for the last layer of value/policy mlp
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
                    activation=torch.nn.ReLU(inplace=True),
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
            activation=nn.ReLU(inplace=True),
            norm_type=self.norm_type,
            output_activation=nn.Identity(),
            output_norm_type=None,
            last_linear_layer_init_zero=last_linear_layer_init_zero
        )

        ######################
        # sampled related code
        ######################

        if self.continuous_action_space:
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
                activation=nn.ReLU(inplace=True),
                norm_type=self.norm_type,
                output_activation=nn.Identity(),
                output_norm_type=None,
                last_linear_layer_init_zero=last_linear_layer_init_zero
            )

        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
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
        # sampled related code
        ######################

        #  {'mu': mu, 'sigma': sigma}
        policy = self.sampled_fc_policy(policy)

        # print("policy['mu']", policy['mu'].max(), policy['mu'].min(), policy['mu'].std())
        # print("policy['sigma']", policy['sigma'].max(), policy['sigma'].min(), policy['sigma'].std())
        if self.continuous_action_space:
            policy = torch.cat([policy['mu'], policy['sigma']], dim=-1)

        return policy, value


@MODEL_REGISTRY.register('SampledEfficientZeroModel')
class SampledEfficientZeroModel(nn.Module):

    def __init__(
        self,
        observation_shape: SequenceType = (12, 96, 96),
        action_space_size: int = 6,
        continuous_action_space: bool = False,
        num_of_sampled_actions: int = 6,
        num_res_blocks: int = 1,
        num_channels: int = 64,
        reward_head_channels: int = 16,
        value_head_channels: int = 16,
        policy_head_channels: int = 16,
        fc_reward_layers: SequenceType = [32],
        fc_value_layers: SequenceType = [32],
        fc_policy_layers: SequenceType = [32],
        reward_support_size: int = 601,
        value_support_size: int = 601,
        downsample: bool = True,
        representation_model_type: str = 'conv_res_blocks',
        representation_model: nn.Module = None,
        batch_norm_momentum=0.1,
        proj_hid: int = 1024,
        proj_out: int = 1024,
        pred_hid: int = 512,
        pred_out: int = 1024,
        lstm_hidden_size: int = 512,
        last_linear_layer_init_zero: bool = True,
        state_norm: bool = False,
        categorical_distribution: bool = True,
        activation: Optional[nn.Module] = nn.ReLU(inplace=True),
        # ==============================================================
        # specific sampled related config for continuous action space
        # ==============================================================
        sigma_type='conditioned',
        fixed_sigma_value=0.3,
        bound_type=None,
        norm_type: str = 'BN',
        *args,
        **kwargs,
    ):
        """
        Overview:
            EfficientZero network
        Arguments:
            - representation_model_type
            - observation_shape: tuple or list. shape of observations: [C, W, H]
            - action_space_size: (:obj:`int`): . action space
            - num_res_blocks (:obj:`int`):  number of res blocks
            - num_channels (:obj:`int`): channels of hidden states
            - reward_head_channels (:obj:`int`): channels of reward head
            - value_head_channels (:obj:`int`): channels of value head
            - policy_head_channels (:obj:`int`): channels of policy head
            - fc_reward_layers (:obj:`list`):  hidden layers of the reward prediction head (MLP head)
            - fc_value_layers (:obj:`list`):  hidden layers of the value prediction head (MLP head)
            - fc_policy_layers (:obj:`list`):  hidden layers of the policy prediction head (MLP head)
            - reward_support_size (:obj:`int`): dim of reward output
            - value_support_size (:obj:`int`): dim of value output
            - downsample (:obj:`bool`): True -> do downsampling for observations. (For board games, do not need)
            - lstm_hidden_size (:obj:`int`):  dim of lstm hidden
            - batch_norm_momentum (:obj:`float`):  Momentum of BN
            - proj_hid (:obj:`int`): dim of projection hidden layer
            - proj_out (:obj:`int`): dim of projection output layer
            - pred_hid (:obj:`int`):dim of projection head (prediction) hidden layer
            - pred_out (:obj:`int`): dim of projection head (prediction) output layer
            - last_linear_layer_init_zero (:obj:`bool`): True -> zero initialization for the last layer of value/policy mlp
            - state_norm (:obj:`bool`):  True -> normalization for hidden states
            - categorical_distribution (:obj:`bool`): whether to use discrete support to represent categorical distribution for value, reward/value_prefix
        """
        super(SampledEfficientZeroModel, self).__init__()
        self.sigma_type = sigma_type
        self.fixed_sigma_value = fixed_sigma_value
        self.bound_type = bound_type
        self.norm_type = norm_type

        self.continuous_action_space = continuous_action_space
        self.num_of_sampled_actions = num_of_sampled_actions
        self.action_space_size = action_space_size

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
        self.representation_model_type = representation_model_type
        self.representation_model = representation_model
        self.downsample = downsample
        self.lstm_hidden_size = lstm_hidden_size

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

        if self.representation_model is None:
            if self.representation_model_type == 'identity':
                self.representation_network = nn.Identity()
            elif self.representation_model_type == 'conv_res_blocks':
                self.representation_network = RepresentationNetwork(
                    observation_shape,
                    num_res_blocks,
                    num_channels,
                    downsample,
                    momentum=batch_norm_momentum,
                    norm_type=norm_type,
                )
        else:
            self.representation_network = self.representation_model

        if self.representation_model_type == 'identity':
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

        # projection
        if self.representation_model_type == 'identity':
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
            nn.Linear(self.projection_input_dim, self.proj_hid), nn.BatchNorm1d(self.proj_hid), nn.ReLU(inplace=True),
            nn.Linear(self.proj_hid, self.proj_hid), nn.BatchNorm1d(self.proj_hid), nn.ReLU(inplace=True),
            nn.Linear(self.proj_hid, self.proj_out), nn.BatchNorm1d(self.proj_out)
        )
        self.projection_head = nn.Sequential(
            nn.Linear(self.proj_out, self.pred_hid),
            nn.BatchNorm1d(self.pred_hid),
            nn.ReLU(inplace=True),
            nn.Linear(self.pred_hid, self.pred_out),
        )

    def initial_inference(self, obs) -> EZNetworkOutput:
        num = obs.size(0)
        hidden_state = self.representation(obs)
        policy_logits, value = self.prediction(hidden_state)
        # zero initialization for reward hidden states
        reward_hidden = (torch.zeros(1, num, self.lstm_hidden_size), torch.zeros(1, num, self.lstm_hidden_size))
        return EZNetworkOutput(value, [0. for _ in range(num)], policy_logits, hidden_state, reward_hidden)

    def recurrent_inference(
            self, hidden_state: torch.Tensor, reward_hidden: torch.Tensor, action: torch.Tensor
    ) -> EZNetworkOutput:
        hidden_state, reward_hidden, value_prefix = self.dynamics(hidden_state, reward_hidden, action)
        policy_logits, value = self.prediction(hidden_state)
        return EZNetworkOutput(value, value_prefix, policy_logits, hidden_state, reward_hidden)

    def prediction(self, encoded_state):
        policy, value = self.prediction_network(encoded_state)
        return policy, value

    def representation(self, observation):
        encoded_state = self.representation_network(observation)
        if not self.state_norm:
            return encoded_state
        else:
            encoded_state_normalized = renormalize(encoded_state)
            return encoded_state_normalized

    def dynamics(self, encoded_state, reward_hidden_state, action):
        """
        Overview:
        :param encoded_state: (batch_size, num_channel, obs_shape[1], obs_shape[2]), e.g. (1,64,6,6)
        :param reward_hidden_state: (batch_size, 1, 1) e.g. (1, 1, 1)
        :param action: (batch_size, action_dim)
        :return:
        """
        if not self.continuous_action_space:
            # discrete action space
            # stack encoded_state with a game specific one hot encoded action
            #  action_one_hot (batch_siize, 1, obs_shape[1], obs_shape[2]), e.g. (4,1,6,6)
            action_one_hot = (
                torch.ones((
                    encoded_state.shape[0],
                    1,
                    encoded_state.shape[2],
                    encoded_state.shape[3],
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
            state_action_encoding = torch.cat((encoded_state, action_one_hot), dim=1)

        else:  # continuous action space
            action_one_hot = (
                torch.ones((
                    encoded_state.shape[0],
                    1,
                    encoded_state.shape[2],
                    encoded_state.shape[3],
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

            state_action_encoding = torch.cat((encoded_state, action_embedding), dim=1)

        try:
            next_encoded_state, reward_hidden_state, value_prefix = self.dynamics_network(
                state_action_encoding, reward_hidden_state
            )
        except Exception as error:
            print(error)
            print(encoded_state.shape, action_embedding.shape)
            print(state_action_encoding.shape)

        if not self.state_norm:
            return next_encoded_state, reward_hidden_state, value_prefix
        else:
            next_encoded_state_normalized = renormalize(next_encoded_state)
            return next_encoded_state_normalized, reward_hidden_state, value_prefix

    def project(self, hidden_state, with_grad=True):
        # only the branch of proj + pred can share the gradients

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

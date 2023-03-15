import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from ding.torch_utils import MLP, ResBlock
from ding.utils import MODEL_REGISTRY, SequenceType

from .common import MZNetworkOutput, RepresentationNetwork
from .utils import renormalize, get_params_mean, get_dynamic_mean, get_reward_mean


@MODEL_REGISTRY.register('MuZeroModel')
class MuZeroModel(nn.Module):

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
        downsample: bool = True,
        num_res_blocks: int = 1,
        num_channels: int = 64,
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
        *args,
        **kwargs
    ):
        """
        Overview:
            MuZero network.
        Arguments:
            - observation_shape (:obj:`SequenceType`): Observation space shape, e.g. [C, W, H]=[12, 96, 96].
            - action_space_size: (:obj:`int`): Action space size, such as 6.
            - representation_network_type (:obj:`Optional[str]`): The type of representation_network in MuZero model. options={'conv_res_blocks', 'identity'}
            - categorical_distribution (:obj:`bool`): Whether to use discrete support to represent categorical distribution for value, reward/value_prefix.
            - activation (:obj:`Optional[nn.Module]`): the activation in MuZero model.
            - representation_network (:obj:`nn.Module`): the user-defined representation_network.
            - batch_norm_momentum (:obj:`float`):  Momentum of BN
            - last_linear_layer_init_zero (:obj:`bool`): Whether to use zero initialization for the last layer of value/policy mlp, default set it to True.
            - state_norm (:obj:`bool`): Whether to use normalization for hidden states, default set it to True.
            - downsample (:obj:`bool`): Whether to do downsampling for observations in ``representation_network``, default set it to True. \
                But sometimes, we do not need, e.g. board games.
            - num_res_blocks (:obj:`int`): number of res blocks in MuZero model.
            - num_channels (:obj:`int`): channels of hidden states.
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
        """
        super(MuZeroModel, self).__init__()
        self.representation_network_type = representation_network_type
        assert self.representation_network_type in ['identity', 'conv_res_blocks']
        self.categorical_distribution = categorical_distribution
        self.self_supervised_learning_loss = self_supervised_learning_loss
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

        self.action_space_size = action_space_size
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
                )
        else:
            self.representation_network = self.representation_network

        if self.representation_network_type == 'identity':
            self.dynamics_network = DynamicsNetwork(
                num_res_blocks,
                observation_shape[0] + 1,  # in_channels=observation_shape[0]
                reward_head_channels,
                fc_reward_layers,
                self.reward_support_size,
                flatten_output_size_for_reward_head,
                momentum=batch_norm_momentum,
                last_linear_layer_init_zero=self.last_linear_layer_init_zero,
            )
            self.prediction_network = PredictionNetwork(
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
            )
        else:
            self.dynamics_network = DynamicsNetwork(
                num_res_blocks,
                num_channels + 1,
                reward_head_channels,
                fc_reward_layers,
                self.reward_support_size,
                flatten_output_size_for_reward_head,
                momentum=batch_norm_momentum,
                last_linear_layer_init_zero=self.last_linear_layer_init_zero,
            )
            self.prediction_network = PredictionNetwork(
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

    def initial_inference(self, obs: torch.Tensor) -> MZNetworkOutput:
        num = obs.size(0)
        hidden_state = self.representation(obs)
        policy_logits, value = self.prediction(hidden_state)
        return MZNetworkOutput(
            value,
            [0. for _ in range(num)],
            policy_logits,
            hidden_state,
        )

    def recurrent_inference(self, hidden_state: torch.Tensor, action: torch.Tensor) -> MZNetworkOutput:
        hidden_state, reward = self.dynamics(hidden_state, action)
        policy_logits, value = self.prediction(hidden_state)
        return MZNetworkOutput(value, reward, policy_logits, hidden_state)

    def prediction(self, encoded_state: torch.Tensor) -> Tuple[torch.Tensor]:
        policy, value = self.prediction_network(encoded_state)
        return policy, value

    def representation(self, observation: torch.Tensor) -> Tuple[torch.Tensor]:
        encoded_state = self.representation_network(observation)
        if not self.state_norm:
            return encoded_state
        else:
            encoded_state_normalized = renormalize(encoded_state)
            return encoded_state_normalized

    def dynamics(self, encoded_state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor]:
        """
         Overview:
             Dynamics function. Predict ``next_encoded_state``, ``reward``
             given current ``encoded_state`` and ``action``.
         Arguments:
             - encoded_state (:obj:`torch.Tensor`): (batch_size, num_channel, obs_shape[1], obs_shape[2]), e.g. (1,64,6,6).
             - action (:obj:`torch.Tensor`): (batch_size, action_dim).
         """

        # Stack encoded_state with a game specific one hot encoded action
        action_one_hot = (
            torch.ones((
                encoded_state.shape[0],
                1,
                encoded_state.shape[2],
                encoded_state.shape[3],
            )).to(action.device).float()
        )
        if len(action.shape) == 1:
            # (batch_size, ) -> (batch_size, 1)
            # e.g.,  torch.Size([4]) ->  torch.Size([4, 1])
            action = action.unsqueeze(-1)

        # action shape: (batch_size, 1)
        # action[:, :, None, None] shape:  (batch_size, 1, 1, 1)
        action_one_hot = (action[:, :, None, None] * action_one_hot / self.action_space_size)

        x = torch.cat((encoded_state, action_one_hot), dim=1)
        next_encoded_state, reward = self.dynamics_network(x)
        if not self.state_norm:
            return next_encoded_state, reward
        else:
            next_encoded_state_normalized = renormalize(next_encoded_state)
            return next_encoded_state_normalized, reward

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
        reward_head_channels,
        fc_reward_layers,
        output_support_size,
        flatten_output_size_for_reward_head,
        momentum: float = 0.1,
        last_linear_layer_init_zero: bool = True,
        activation: Optional[nn.Module] = nn.ReLU(inplace=True),
    ):
        """
        Overview:
            Dynamics network. Predict next hidden state given current hidden state and action.
        Arguments:
            - num_res_blocks (:obj:`int`): number of res blocks.
            - num_channels (:obj:`int`): channels of hidden states.
            - fc_reward_layers (:obj:`list`):  hidden layers of the reward prediction head (MLP head)
            - output_support_size (:obj:`int`): dim of reward output
            - flatten_output_size_for_reward_head (:obj:`int`): dim of flatten hidden states
            - lstm_hidden_size (:obj:`int`): dim of lstm hidden state in dynamics network.
            - last_linear_layer_init_zero (:obj:`bool`): Whether to use zero initialization for the last layer of value/policy mlp, default set it to True.
            - activation (:obj:`Optional[nn.Module]`): the activation in Dynamics network.
        """
        super().__init__()
        self.num_channels = num_channels

        self.conv = nn.Conv2d(num_channels, num_channels - 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(num_channels - 1, momentum=momentum)
        self.resblocks = nn.ModuleList(
            [
                ResBlock(
                    in_channels=num_channels - 1,
                    activation=activation,
                    norm_type='BN',
                    res_type='basic',
                    bias=False
                ) for _ in range(num_res_blocks)
            ]
        )

        self.conv1x1_reward = nn.Conv2d(num_channels - 1, reward_head_channels, 1)
        self.bn_reward = nn.BatchNorm2d(reward_head_channels, momentum=momentum)
        self.flatten_output_size_for_reward_head = flatten_output_size_for_reward_head
        # TODO(pu)
        self.fc = MLP(
            self.flatten_output_size_for_reward_head,
            hidden_channels=fc_reward_layers[0],
            layer_num=len(fc_reward_layers) + 1,
            out_channels=output_support_size,
            activation=activation,
            norm_type='BN',
            output_activation=nn.Identity(),
            output_norm_type=None,
            last_linear_layer_init_zero=last_linear_layer_init_zero
        )
        self.activation = activation

    def forward(self, x: torch.Tensor):
        # take the state encoding,  x[:, -1, :, :] is action encoding
        state = x[:, :-1, :, :]
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

        x = x.view(-1, self.flatten_output_size_for_reward_head)
        reward = self.fc(x)

        return state, reward

    def get_dynamic_mean(self):
        return get_dynamic_mean(self)

    def get_reward_mean(self):
        return get_reward_mean(self)


class PredictionNetwork(nn.Module):

    def __init__(
        self,
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
        activation=nn.ReLU(inplace=True),
    ):
        """
        Overview:
            Prediction network. Predict the value and policy given hidden state.
        Arguments:
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
        """
        super().__init__()
        self.in_channels = in_channels
        if self.in_channels is not None:
            self.conv_input = nn.Conv2d(in_channels, num_channels, 1)

        self.resblocks = nn.ModuleList(
            [
                ResBlock(
                    in_channels=num_channels,
                    activation=activation,
                    norm_type='BN',
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
        self.fc_value = MLP(
            in_channels=self.flatten_output_size_for_value_head,
            hidden_channels=fc_value_layers[0],
            out_channels=output_support_size,
            layer_num=len(fc_value_layers) + 1,
            activation=activation,
            norm_type='BN',
            output_activation=nn.Identity(),
            output_norm_type=None,
            last_linear_layer_init_zero=last_linear_layer_init_zero
        )
        self.fc_policy = MLP(
            in_channels=self.flatten_output_size_for_policy_head,
            hidden_channels=fc_policy_layers[0],
            out_channels=action_space_size,
            layer_num=len(fc_policy_layers) + 1,
            activation=activation,
            norm_type='BN',
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

        value = value.reshape(-1, self.flatten_output_size_for_value_head)
        policy = policy.reshape(-1, self.flatten_output_size_for_policy_head)

        value = self.fc_value(value)
        policy = self.fc_policy(policy)
        return policy, value



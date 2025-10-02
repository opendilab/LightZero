import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from ding.utils import MODEL_REGISTRY, SequenceType

from .common import MZNetworkOutput, RepresentationNetwork
from .muzero_model import DynamicsNetwork
from .sampled_efficientzero_model import PredictionNetwork
from .utils import renormalize


@MODEL_REGISTRY.register('SampledMuZeroModel')
class SampledMuZeroModel(nn.Module):

    def __init__(
            self,
            observation_shape: SequenceType = (12, 96, 96),
            action_space_size: int = 6,
            num_res_blocks: int = 1,
            num_channels: int = 64,
            latent_state_dim: int = 256,
            reward_head_channels: int = 16,
            value_head_channels: int = 16,
            policy_head_channels: int = 16,
            reward_head_hidden_channels: SequenceType = [256],
            value_head_hidden_channels: SequenceType = [256],
            policy_head_hidden_channels: SequenceType = [256],
            reward_support_range: SequenceType =(-300., 301., 1.),
            value_support_range: SequenceType =(-300., 301., 1.),
            proj_hid: int = 1024,
            proj_out: int = 1024,
            pred_hid: int = 512,
            pred_out: int = 1024,
            self_supervised_learning_loss: bool = True,
            categorical_distribution: bool = True,
            activation: Optional[nn.Module] = nn.GELU(approximate='tanh'),
            last_linear_layer_init_zero: bool = True,
            state_norm: bool = False,
            downsample: bool = False,
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
            use_sim_norm: bool = False,
            *args,
            **kwargs,
    ):
        """
        Overview:
            The definition of the network model of Sampled MuZero for 3D-image obs.
        Arguments:
            - observation_shape (:obj:`int`): Observation space shape, e.g. 8 for Lunarlander.
            - action_space_size: (:obj:`int`): Action space size, which is an integer number. For discrete action space, it is the num of discrete actions, \
                e.g. 4 for Lunarlander. For continuous action space, it is the dimension of the continuous action, e.g. 4 for bipedalwalker.
            - latent_state_dim (:obj:`int`): The dimension of latent state, such as 256.
            - reward_head_hidden_channels (:obj:`SequenceType`): The number of hidden layers of the reward head (MLP head).
            - value_head_hidden_channels (:obj:`SequenceType`): The number of hidden layers used in value head (MLP head).
            - policy_head_hidden_channels (:obj:`SequenceType`): The number of hidden layers used in policy head (MLP head).
            - reward_support_range (:obj:`SequenceType`): The range of categorical reward output
            - value_support_range (:obj:`SequenceType`): The range of categorical value output.
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
        super(SampledMuZeroModel, self).__init__()
        self.categorical_distribution = categorical_distribution
        if self.categorical_distribution:
            self.reward_support_size = len(torch.arange(*reward_support_range))
            self.value_support_size = len(torch.arange(*value_support_range))
        else:
            self.reward_support_size = 1
            self.value_support_size = 1

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
        self.reward_head_hidden_channels = reward_head_hidden_channels
        self.value_head_hidden_channels = value_head_hidden_channels
        self.policy_head_hidden_channels = policy_head_hidden_channels
        self.proj_hid = proj_hid
        self.proj_out = proj_out
        self.pred_hid = pred_hid
        self.pred_out = pred_out

        self.last_linear_layer_init_zero = last_linear_layer_init_zero
        self.state_norm = state_norm
        self.downsample = downsample

        self.self_supervised_learning_loss = self_supervised_learning_loss

        self.sigma_type = sigma_type
        self.fixed_sigma_value = fixed_sigma_value
        self.bound_type = bound_type
        self.norm_type = norm_type
        self.num_of_sampled_actions = num_of_sampled_actions
        self.res_connection_in_dynamics = res_connection_in_dynamics
        self.activation = activation


        if observation_shape[1] == 96:
            latent_size = math.ceil(observation_shape[1] / 16) * math.ceil(observation_shape[2] / 16)
        elif observation_shape[1] == 84:
            latent_size = math.ceil(observation_shape[1] / 14) * math.ceil(observation_shape[2] / 14)
        elif observation_shape[1] == 64:
            latent_size = math.ceil(observation_shape[1] / 8) * math.ceil(observation_shape[2] / 8)
        else:
            raise ValueError("Invalid observation shape, only support 64, 84, 96 for now.")

        flatten_input_size_for_reward_head = (
            (reward_head_channels * latent_size) if downsample else
            (reward_head_channels * observation_shape[1] * observation_shape[2])
        )
        flatten_input_size_for_value_head = (
            (value_head_channels * latent_size) if downsample else
            (value_head_channels * observation_shape[1] * observation_shape[2])
        )
        flatten_input_size_for_policy_head = (
            (policy_head_channels * latent_size) if downsample else
            (policy_head_channels * observation_shape[1] * observation_shape[2])
        )

        self.representation_network = RepresentationNetwork(
            observation_shape,
            num_res_blocks,
            num_channels,
            downsample,
            norm_type=self.norm_type,
            use_sim_norm=use_sim_norm,
        )

        self.dynamics_network = DynamicsNetwork(
            observation_shape,
            self.action_encoding_dim,
            num_res_blocks,
            num_channels + self.action_encoding_dim,
            reward_head_channels,
            reward_head_hidden_channels,
            self.reward_support_size,
            flatten_input_size_for_reward_head,
            downsample,
            last_linear_layer_init_zero=self.last_linear_layer_init_zero,
            activation=activation,
            norm_type=norm_type
        )

        self.prediction_network = PredictionNetwork(
            observation_shape,
            self.continuous_action_space,
            action_space_size,
            num_res_blocks,
            num_channels,
            value_head_channels,
            policy_head_channels,
            value_head_hidden_channels,
            policy_head_hidden_channels,
            self.value_support_size,
            flatten_input_size_for_value_head,
            flatten_input_size_for_policy_head,
            downsample,
            last_linear_layer_init_zero=self.last_linear_layer_init_zero,
            sigma_type=self.sigma_type,
            fixed_sigma_value=self.fixed_sigma_value,
            bound_type=self.bound_type,
            norm_type=self.norm_type,
        )

        if self.self_supervised_learning_loss:
            if self.downsample:
                # In Atari, if the observation_shape is set to (12, 96, 96), which indicates the original shape of
                # (3,96,96), and frame_stack_num is 4. Due to downsample, the encoding of observation (latent_state) is
                # (64, 96/16, 96/16), where 64 is the number of channels, 96/16 is the size of the latent state. Thus,
                # self.projection_input_dim = 64 * 96/16 * 96/16 = 64*6*6 = 2304
                self.projection_input_dim = num_channels * latent_size
            else:
                self.projection_input_dim = num_channels * observation_shape[1] * observation_shape[2]

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
                # (batch_size,) -> (batch_size, action_dim=1, 1, 1)
                # e.g., torch.Size([8]) -> torch.Size([8, 1, 1, 1])
                action = action.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            elif len(action.shape) == 2:
                # (batch_size, action_dim) -> (batch_size, action_dim, 1, 1)
                # e.g., torch.Size([8, 2]) ->  torch.Size([8, 2, 1, 1])
                action = action.unsqueeze(-1).unsqueeze(-1)
            elif len(action.shape) == 3:
                # (batch_size, action_dim, 1) -> (batch_size, action_dim)
                # e.g., torch.Size([8, 2, 1]) ->  torch.Size([8, 2, 1, 1])
                action = action.unsqueeze(-1)

            action_encoding_tmp = action
            action_encoding = action_encoding_tmp.expand(
                latent_state.shape[0], self.action_space_size, latent_state.shape[2], latent_state.shape[3]
            )

        action_encoding = action_encoding.to(latent_state.device).float()
        # state_action_encoding shape: (batch_size, latent_state[1] + action_dim, latent_state[2], latent_state[3]) or
        # (batch_size, latent_state[1] + action_space_size, latent_state[2], latent_state[3]) depending on the discrete_action_encoding_type.
        state_action_encoding = torch.cat((latent_state, action_encoding), dim=1)

        next_latent_state, reward = self.dynamics_network(state_action_encoding)

        if not self.state_norm:
            return next_latent_state, reward
        else:
            next_latent_state_normalized = renormalize(next_latent_state)
            return next_latent_state_normalized, reward

    def project(self, latent_state: torch.Tensor, with_grad: bool = True) -> torch.Tensor:
        """
        Overview:
            Project the latent state to a lower dimension to calculate the self-supervised loss, which is involved in
            MuZero algorithm in EfficientZero.
            For more details, please refer to the paper ``Exploring Simple Siamese Representation Learning``.
        Arguments:
            - latent_state (:obj:`torch.Tensor`): The encoding latent state of input state.
            - with_grad (:obj:`bool`): Whether to calculate gradient for the projection result.
        Returns:
            - proj (:obj:`torch.Tensor`): The result embedding vector of projection operation.
        Shapes:
            - latent_state (:obj:`torch.Tensor`): :math:`(B, H_, W_)`, where B is batch_size, H_ is the height of \
                latent state, W_ is the width of latent state.
            - proj (:obj:`torch.Tensor`): :math:`(B, projection_output_dim)`, where B is batch_size.

        Examples:
            >>> latent_state = torch.randn(256, 64, 6, 6)
            >>> output = self.project(latent_state)
            >>> output.shape # (256, 1024)

        .. note::
            for Atari:
            observation_shape = (12, 96, 96),  # original shape is (3,96,96), frame_stack_num=4
            if downsample is True, latent_state.shape: (batch_size, num_channel, obs_shape[1] / 16, obs_shape[2] / 16)
            i.e., (256, 64, 96 / 16, 96 / 16) = (256, 64, 6, 6)
            latent_state reshape: (256, 64, 6, 6) -> (256,64*6*6) = (256, 2304)
            # self.projection_input_dim = 64*6*6 = 2304
            # self.projection_output_dim = 1024
        """
        latent_state = latent_state.reshape(latent_state.shape[0], -1)
        proj = self.projection(latent_state)

        if with_grad:
            # with grad, use prediction_head
            return self.prediction_head(proj)
        else:
            return proj.detach()


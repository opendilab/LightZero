import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from ding.model.common import ReparameterizationHead
from ding.torch_utils import MLP, ResBlock
from ding.utils import MODEL_REGISTRY, SequenceType

from .common import EZNetworkOutput, RepresentationNetwork
from .efficientzero_model import DynamicsNetwork
from .utils import renormalize, get_params_mean


# use ModelRegistry to register the model, for more details about ModelRegistry, please refer to DI-engine's document.
@MODEL_REGISTRY.register('SampledEfficientZeroModel')
class SampledEfficientZeroModel(nn.Module):

    def __init__(
            self,
            observation_shape: SequenceType = (12, 96, 96),
            action_space_size: int = 6,
            num_res_blocks: int = 1,
            num_channels: int = 64,
            lstm_hidden_size: int = 512,
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
            self_supervised_learning_loss: bool = True,
            categorical_distribution: bool = True,
            activation: Optional[nn.Module] = nn.ReLU(inplace=True),
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
            norm_type: str = 'BN',
            discrete_action_encoding_type: str = 'one_hot',
            *args,
            **kwargs,
    ):
        """
        Overview:
            The definition of the network model of Sampled EfficientZero, which is a generalization version for 2D image obs.
            The networks are mainly build on convolution residual blocks and fully connected layers.
            Sampled EfficientZero model consists of a representation network, a dynamics network and a prediction network.
            The representation network is an MLP network which maps the raw observation to a latent state.
            The dynamics network is an MLP+LSTM network which predicts the next latent state, reward_hidden_state and value_prefix given the current latent state and action.
            The prediction network is an MLP network which predicts the value and policy given the current latent state.
        Arguments:
            - observation_shape (:obj:`SequenceType`): Observation space shape, e.g. [C, W, H]=[12, 96, 96] for Atari.
            - action_space_size: (:obj:`int`): Action space size, which is an integer number. For discrete action space, it is the num of discrete actions, \
                e.g. 4 for Lunarlander. For continuous action space, it is the dimension of the continuous action, e.g. 4 for bipedalwalker.
            - num_res_blocks (:obj:`int`): The number of res blocks in Sampled EfficientZero model.
            - num_channels (:obj:`int`): The channels of hidden states.
            - lstm_hidden_size (:obj:`int`): dim of lstm hidden state in dynamics network.
            - reward_head_channels (:obj:`int`): The channels of reward head.
            - value_head_channels (:obj:`int`): The channels of value head.
            - policy_head_channels (:obj:`int`): The channels of policy head.
            - fc_reward_layers (:obj:`SequenceType`): The number of hidden layers of the reward head (MLP head).
            - fc_value_layers (:obj:`SequenceType`): The number of hidden layers used in value head (MLP head).
            - fc_policy_layers (:obj:`SequenceType`): The number of hidden layers used in policy head (MLP head).
            - reward_support_size (:obj:`int`): The size of categorical reward output
            - value_support_size (:obj:`int`): The size of categorical value output.
            - proj_hid (:obj:`int`): The size of projection hidden layer.
            - proj_out (:obj:`int`): The size of projection output layer.
            - pred_hid (:obj:`int`): The size of prediction hidden layer.
            - pred_out (:obj:`int`): The size of prediction output layer.
            - self_supervised_learning_loss (:obj:`bool`): Whether to use self_supervised_learning related networks in model, default set it to False.
            - categorical_distribution (:obj:`bool`): Whether to use discrete support to represent categorical distribution \
                for value, reward/value_prefix.
            - activation (:obj:`Optional[nn.Module]`): Activation function used in network, which often use in-place \
                operation to speedup, e.g. ReLU(inplace=True).
            - last_linear_layer_init_zero (:obj:`bool`): Whether to use zero initialization for the last layer of \
                value/policy mlp, default set it to True.
            - state_norm (:obj:`bool`): Whether to use normalization for hidden states, default set it to True.
            - downsample (:obj:`bool`): Whether to do downsampling for observations in ``representation_network``, \
                defaults to True. This option is often used in video games like Atari. In board games like go, \
                we don't need this module.
            # ==============================================================
            # specific sampled related config
            # ==============================================================
            - continuous_action_space (:obj:`bool`): The type of action space. default set it to False.
            - num_of_sampled_actions (:obj:`int`): the number of sampled actions, i.e. the K in original Sampled MuZero paper.
            # Please see ``ReparameterizationHead`` in ``ding.model.common.head`` for more details about the following arguments.
            - sigma_type (:obj:`str`): the type of sigma in policy head of prediction network, options={'conditioned', 'fixed'}.
            - fixed_sigma_value (:obj:`float`): the fixed sigma value in policy head of prediction network,
            - bound_type (:obj:`str`): The type of bound in networks, default set it to None.
            - norm_type (:obj:`str`): The type of normalization in networks, default set it to 'BN'.
            - discrete_action_encoding_type (:obj:`str`): The type of encoding for discrete action. default set it to 'one_hot'. options = {'one_hot', 'not_one_hot'}
        """
        super(SampledEfficientZeroModel, self).__init__()
        if isinstance(observation_shape, int) or len(observation_shape) == 1:
            # for vector obs input, e.g. classical control and box2d environments
            # to be compatible with LightZero model/policy, transform to shape: [C, W, H]
            observation_shape = [1, observation_shape, 1]
        if not categorical_distribution:
            self.reward_support_size = 1
            self.value_support_size = 1
        else:
            self.reward_support_size = reward_support_size
            self.value_support_size = value_support_size

        self.continuous_action_space = continuous_action_space
        self.action_space_size = action_space_size
        # The dim of action space. For discrete action space, it's 1.
        # For continuous action space, it is the dim of action.
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

        self.last_linear_layer_init_zero = last_linear_layer_init_zero
        self.state_norm = state_norm
        self.downsample = downsample
        self.self_supervised_learning_loss = self_supervised_learning_loss

        self.sigma_type = sigma_type
        self.fixed_sigma_value = fixed_sigma_value
        self.bound_type = bound_type
        self.norm_type = norm_type
        self.num_of_sampled_actions = num_of_sampled_actions

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

        self.representation_network = RepresentationNetwork(
            observation_shape,
            num_res_blocks,
            num_channels,
            downsample,
            norm_type=self.norm_type,
        )

        self.dynamics_network = DynamicsNetwork(
            observation_shape,
            self.action_encoding_dim,
            num_res_blocks,
            num_channels + self.action_encoding_dim,
            reward_head_channels,
            fc_reward_layers,
            self.reward_support_size,
            flatten_output_size_for_reward_head,
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
            fc_value_layers,
            fc_policy_layers,
            self.value_support_size,
            flatten_output_size_for_value_head,
            flatten_output_size_for_policy_head,
            downsample,
            last_linear_layer_init_zero=self.last_linear_layer_init_zero,
            sigma_type=self.sigma_type,
            fixed_sigma_value=self.fixed_sigma_value,
            bound_type=self.bound_type,
            norm_type=self.norm_type,
        )

        if self.self_supervised_learning_loss:
            # self_supervised_learning_loss related network proposed in EfficientZero
            if self.downsample:
                # In Atari, if the observation_shape is set to (12, 96, 96), which indicates the original shape of
                # (3,96,96), and frame_stack_num is 4. Due to downsample, the encoding of observation (latent_state) is
                # (64, 96/16, 96/16), where 64 is the number of channels, 96/16 is the size of the latent state. Thus,
                # self.projection_input_dim = 64 * 96/16 * 96/16 = 64*6*6 = 2304
                self.projection_input_dim = num_channels * math.ceil(observation_shape[1] / 16
                                                                     ) * math.ceil(observation_shape[2] / 16)
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

    def initial_inference(self, obs: torch.Tensor) -> EZNetworkOutput:
        """
         Overview:
            Initial inference of SampledEfficientZero model, which is the first step of the SampledEfficientZero model.
             To perform the initial inference, we first use the representation network to obtain the "latent_state" of the observation.
             Then we use the prediction network to predict the "value" and "policy_logits" of the "latent_state", and
            also prepare the zeros-like ``reward_hidden_state`` for the next step of the SampledEfficientZero model.
        Arguments:
            - obs (:obj:`torch.Tensor`): The 2D image observation data.
        Returns (EZNetworkOutput):
            - value (:obj:`torch.Tensor`): The output value of input state to help policy improvement and evaluation.
            - value_prefix (:obj:`torch.Tensor`): The predicted prefix sum of value for input state. \
                In initial inference, we set it to zero vector.
            - policy_logits (:obj:`torch.Tensor`): The output logit to select discrete action.
            - latent_state (:obj:`torch.Tensor`): The encoding latent state of input state.
            - reward_hidden_state (:obj:`Tuple[torch.Tensor]`): The hidden state of LSTM about reward. In initial inference, \
                we set it to the zeros-like hidden state (H and C).
        Shapes:
            - obs (:obj:`torch.Tensor`): :math:`(B, num_channel, obs_shape[1], obs_shape[2])`, where B is batch_size.
            - value (:obj:`torch.Tensor`): :math:`(B, value_support_size)`, where B is batch_size.
            - value_prefix (:obj:`torch.Tensor`): :math:`(B, reward_support_size)`, where B is batch_size.
            - policy_logits (:obj:`torch.Tensor`): :math:`(B, action_dim)`, where B is batch_size.
            - latent_state (:obj:`torch.Tensor`): :math:`(B, H_, W_)`, where B is batch_size, H_ is the height of \
                latent state, W_ is the width of latent state.
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
            Recurrent inference of Sampled EfficientZero model, which is the rollout step of the Sampled EfficientZero model.
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
            - latent_state (:obj:`torch.Tensor`): The encoding latent state of input state.
            - next_latent_state (:obj:`torch.Tensor`): The predicted next latent state.
            - reward_hidden_state (:obj:`Tuple[torch.Tensor]`): The output hidden state of LSTM about reward.
        Shapes:
            - obs (:obj:`torch.Tensor`): :math:`(B, num_channel, obs_shape[1], obs_shape[2])`, where B is batch_size.
            - action (:obj:`torch.Tensor`): :math:`(B, )`, where B is batch_size.
            - value (:obj:`torch.Tensor`): :math:`(B, value_support_size)`, where B is batch_size.
            - value_prefix (:obj:`torch.Tensor`): :math:`(B, reward_support_size)`, where B is batch_size.
            - policy_logits (:obj:`torch.Tensor`): :math:`(B, action_dim)`, where B is batch_size.
            - latent_state (:obj:`torch.Tensor`): :math:`(B, H_, W_)`, where B is batch_size, H_ is the height of \
                latent state, W_ is the width of latent state.
            - next_latent_state (:obj:`torch.Tensor`): :math:`(B, H_, W_)`, where B is batch_size, H_ is the height of \
                latent state, W_ is the width of latent state.
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
            - obs (:obj:`torch.Tensor`): The 2D image observation data.
        Returns:
            - latent_state (:obj:`torch.Tensor`): The encoding latent state of input state.
        Shapes:
            - obs (:obj:`torch.Tensor`): :math:`(B, num_channel, obs_shape[1], obs_shape[2])`, where B is batch_size.
            - latent_state (:obj:`torch.Tensor`): :math:`(B, H_, W_)`, where B is batch_size, H_ is the height of \
                latent state, W_ is the width of latent state.
        """
        latent_state = self.representation_network(observation)
        if self.state_norm:
            latent_state = renormalize(latent_state)
        return latent_state

    def _prediction(self, latent_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Overview:
             use the prediction network to predict the "value" and "policy_logits" of the "latent_state".
        Arguments:
            - latent_state (:obj:`torch.Tensor`): The encoding latent state of input obs.
        Returns:
            - policy_logits (:obj:`torch.Tensor`): The output logit to select discrete action.
            - value (:obj:`torch.Tensor`): The output value of input state to help policy improvement and evaluation.
        Shapes:
            - latent_state (:obj:`torch.Tensor`): :math:`(B, H_, W_)`, where B is batch_size, H_ is the height of \
                latent state, W_ is the width of latent state.
            - policy_logits (:obj:`torch.Tensor`): :math:`(B, action_dim)`, where B is batch_size.
            - value (:obj:`torch.Tensor`): :math:`(B, value_support_size)`, where B is batch_size.
        """
        return self.prediction_network(latent_state)

    def _dynamics(self, latent_state: torch.Tensor, reward_hidden_state: Tuple[torch.Tensor],
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
            - latent_state (:obj:`torch.Tensor`): :math:`(B, H_, W_)`, where B is batch_size, H_ is the height of \
                latent state, W_ is the width of latent state.
            - action (:obj:`torch.Tensor`): :math:`(B, )`, where B is batch_size.
            - next_latent_state (:obj:`torch.Tensor`): :math:`(B, H_, W_)`, where B is batch_size, H_ is the height of \
                latent state, W_ is the width of latent state.
            - value_prefix (:obj:`torch.Tensor`): :math:`(B, reward_support_size)`, where B is batch_size.
        """
        # NOTE: the discrete action encoding type is important for some environments

        if not self.continuous_action_space:
            # discrete action space
            if self.discrete_action_encoding_type == 'one_hot':
                # Stack latent_state with the one hot encoded action.
                # The final action_encoding shape is (batch_size, action_space_size, latent_state[2], latent_state[3]), e.g. (8, 2, 4, 1).
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

                action_encoding_tmp = action_one_hot.unsqueeze(-1).unsqueeze(-1)
                action_encoding = action_encoding_tmp.expand(
                    latent_state.shape[0], self.action_space_size, latent_state.shape[2], latent_state.shape[3]
                )

            elif self.discrete_action_encoding_type == 'not_one_hot':
                # Stack latent_state with the normalized encoded action.
                # The final action_encoding shape is (batch_size, 1, latent_state[2], latent_state[3]), e.g. (8, 1, 4, 1).
                if len(action.shape) == 2:
                    # (batch_size, action_dim=1) -> (batch_size, 1, 1, 1)
                    # e.g.,  torch.Size([8, 1]) ->  torch.Size([8, 1, 1, 1])
                    action = action.unsqueeze(-1).unsqueeze(-1)
                elif len(action.shape) == 1:
                    # (batch_size,) -> (batch_size, 1, 1, 1)
                    # e.g.,  -> torch.Size([8, 1, 1, 1])
                    action = action.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

                action_encoding = action.expand(
                    latent_state.shape[0], 1, latent_state.shape[2], latent_state.shape[3]
                ) / self.action_space_size
        else:
            # continuous action space
            if len(action.shape) == 2:
                # (batch_size, action_dim) -> (batch_size, action_dim, 1, 1)
                # e.g.,  torch.Size([8, 2]) ->  torch.Size([8, 2, 1, 1])
                action = action.unsqueeze(-1).unsqueeze(-1)
            elif len(action.shape) == 1:
                # (batch_size,) -> (batch_size, action_dim=1, 1, 1)
                # e.g.,  -> torch.Size([8, 2, 1, 1])
                action = action.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

            action_encoding_tmp = action
            action_encoding = action_encoding_tmp.expand(
                latent_state.shape[0], self.action_space_size, latent_state.shape[2], latent_state.shape[3]
            )

        action_encoding = action_encoding.to(latent_state.device).float()
        # state_action_encoding shape: (batch_size, latent_state[1] + action_dim, latent_state[2], latent_state[3]) or
        # (batch_size, latent_state[1] + action_space_size, latent_state[2], latent_state[3]) depending on the discrete_action_encoding_type.
        state_action_encoding = torch.cat((latent_state, action_encoding), dim=1)
        next_latent_state, next_reward_hidden_state, value_prefix = self.dynamics_network(
            state_action_encoding, reward_hidden_state
        )
        if not self.state_norm:
            return next_latent_state, next_reward_hidden_state, value_prefix
        else:
            next_latent_state_normalized = renormalize(next_latent_state)
            return next_latent_state_normalized, next_reward_hidden_state, value_prefix

    def project(self, latent_state: torch.Tensor, with_grad=True) -> torch.Tensor:
        """
        Overview:
            Project the latent state to a lower dimension to calculate the self-supervised loss, which is proposed in EfficientZero.
            For more details, please refer to paper ``Exploring Simple Siamese Representation Learning``.
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

    def get_params_mean(self):
        return get_params_mean(self)


class PredictionNetwork(nn.Module):

    def __init__(
            self,
            observation_shape: SequenceType,
            continuous_action_space,
            action_space_size,
            num_res_blocks,
            num_channels,
            value_head_channels,
            policy_head_channels,
            fc_value_layers,
            fc_policy_layers,
            output_support_size,
            flatten_output_size_for_value_head,
            flatten_output_size_for_policy_head,
            downsample: bool = False,
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
            The definition of policy and value prediction network, which is used to predict value and policy by the
            given latent state.
            The networks are mainly build on res_conv_blocks and fully connected layers.
        Arguments:
            - observation_shape (:obj:`SequenceType`): The shape of observation space, e.g. (C, H, W) for image.
            - continuous_action_space (:obj:`bool`): The type of action space. default set it to False.
            - action_space_size: (:obj:`int`): Action space size, usually an integer number. For discrete action \
                space, it is the number of discrete actions. For continuous action space, it is the dimension of \
                continuous action.
            - num_res_blocks (:obj:`int`): number of res blocks in model.
            - num_channels (:obj:`int`): channels of hidden states.
            - value_head_channels (:obj:`int`): channels of value head.
            - policy_head_channels (:obj:`int`): channels of policy head.
            - fc_value_layers (:obj:`SequenceType`): hidden layers of the value prediction head (MLP head).
            - fc_policy_layers (:obj:`SequenceType`): hidden layers of the policy prediction head (MLP head).
            - output_support_size (:obj:`int`): dim of value output.
            - flatten_output_size_for_value_head (:obj:`int`): dim of flatten hidden states.
            - flatten_output_size_for_policy_head (:obj:`int`): dim of flatten hidden states.
            - downsample (:obj:`bool`): Whether to do downsampling for observations in ``representation_network``.
            - last_linear_layer_init_zero (:obj:`bool`): Whether to use zero initialization for the last layer of value/policy mlp, default set it to True.
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
        self.continuous_action_space = continuous_action_space
        self.flatten_output_size_for_value_head = flatten_output_size_for_value_head
        self.flatten_output_size_for_policy_head = flatten_output_size_for_policy_head
        self.norm_type = norm_type
        self.sigma_type = sigma_type
        self.fixed_sigma_value = fixed_sigma_value
        self.bound_type = bound_type
        self.activation = activation

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

        if norm_type == 'BN':
            self.norm_value = nn.BatchNorm2d(value_head_channels)
            self.norm_policy = nn.BatchNorm2d(policy_head_channels)
        elif norm_type == 'LN':
            if downsample:
                self.norm_value = nn.LayerNorm(
                    [value_head_channels, math.ceil(observation_shape[-2] / 16), math.ceil(observation_shape[-1] / 16)])
                self.norm_policy = nn.LayerNorm([policy_head_channels, math.ceil(observation_shape[-2] / 16),
                                                 math.ceil(observation_shape[-1] / 16)])
            else:
                self.norm_value = nn.LayerNorm([value_head_channels, observation_shape[-2], observation_shape[-1]])
                self.norm_policy = nn.LayerNorm([policy_head_channels, observation_shape[-2], observation_shape[-1]])

        self.fc_value_head = MLP(
            in_channels=self.flatten_output_size_for_value_head,
            hidden_channels=fc_value_layers[0],
            out_channels=output_support_size,
            layer_num=len(fc_value_layers) + 1,
            activation=activation,
            norm_type=self.norm_type,
            output_activation=False,
            output_norm=False,
            last_linear_layer_init_zero=last_linear_layer_init_zero
        )

        # sampled related core code
        if self.continuous_action_space:
            self.fc_policy_head = ReparameterizationHead(
                input_size=self.flatten_output_size_for_policy_head,
                output_size=action_space_size,
                layer_num=len(fc_policy_layers) + 1,
                sigma_type=self.sigma_type,
                fixed_sigma_value=self.fixed_sigma_value,
                activation=nn.ReLU(),
                norm_type=None,
                bound_type=self.bound_type
            )
        else:
            self.fc_policy_head = MLP(
                in_channels=self.flatten_output_size_for_policy_head,
                hidden_channels=fc_policy_layers[0],
                out_channels=action_space_size,
                layer_num=len(fc_policy_layers) + 1,
                activation=activation,
                norm_type=self.norm_type,
                output_activation=False,
                output_norm=False,
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

        for res_block in self.resblocks:
            latent_state = res_block(latent_state)
        value = self.conv1x1_value(latent_state)
        value = self.norm_value(value)
        value = self.activation(value)

        policy = self.conv1x1_policy(latent_state)
        policy = self.norm_policy(policy)
        policy = self.activation(policy)

        value = value.reshape(-1, self.flatten_output_size_for_value_head)
        policy = policy.reshape(-1, self.flatten_output_size_for_policy_head)
        value = self.fc_value_head(value)

        # sampled related core code
        policy = self.fc_policy_head(policy)

        if self.continuous_action_space:
            policy = torch.cat([policy['mu'], policy['sigma']], dim=-1)

        return policy, value

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from ding.torch_utils import MLP, ResBlock
from ding.utils import MODEL_REGISTRY, SequenceType

from lzero.model.muzero_model import MuZeroModel
from .common import MZNetworkOutput, RepresentationNetwork, PredictionNetwork, FeatureAndGradientHook, SimNorm


# use ModelRegistry to register the model, for more details about ModelRegistry, please refer to DI-engine's document.
@MODEL_REGISTRY.register('MuZeroContextModel')
class MuZeroContextModel(MuZeroModel):

    def __init__(
        self,
        observation_shape: SequenceType = (12, 96, 96),
        action_space_size: int = 6,
        num_res_blocks: int = 1,
        num_channels: int = 64,
        reward_head_channels: int = 16,
        value_head_channels: int = 16,
        policy_head_channels: int = 16,
        reward_head_hidden_channels: SequenceType = [32],
        value_head_hidden_channels: SequenceType = [32],
        policy_head_hidden_channels: SequenceType = [32],
        reward_support_size: int = 601,
        value_support_size: int = 601,
        proj_hid: int = 1024,
        proj_out: int = 1024,
        pred_hid: int = 512,
        pred_out: int = 1024,
        self_supervised_learning_loss: bool = False,
        categorical_distribution: bool = True,
        activation: nn.Module = nn.ReLU(inplace=True),
        last_linear_layer_init_zero: bool = True,
        state_norm: bool = False,
        downsample: bool = False,
        norm_type: Optional[str] = 'BN',
        discrete_action_encoding_type: str = 'one_hot',
        context_length_init: int = 5,
        use_sim_norm: bool = False,
        analysis_sim_norm: bool = False,
        *args,
        **kwargs
    ):
        """
        Overview:
            The definition of the model for MuZero w/ Context, a variant of MuZero.
            This variant retains the same training settings as MuZero but diverges during inference
            by employing a k-step recursively predicted latent representation at the root node,
            proposed in the UniZero paper https://arxiv.org/abs/2406.10667.
        Arguments:
            - observation_shape (:obj:`SequenceType`): Observation space shape, e.g. [C, W, H]=[12, 96, 96] for Atari.
            - action_space_size: (:obj:`int`): Action space size, usually an integer number for discrete action space.
            - num_res_blocks (:obj:`int`): The number of res blocks in AlphaZero model.
            - num_channels (:obj:`int`): The channels of hidden states.
            - reward_head_channels (:obj:`int`): The channels of reward head.
            - value_head_channels (:obj:`int`): The channels of value head.
            - policy_head_channels (:obj:`int`): The channels of policy head.
            - reward_head_hidden_channels (:obj:`SequenceType`): The number of hidden layers of the reward head (MLP head).
            - value_head_hidden_channels (:obj:`SequenceType`): The number of hidden layers used in value head (MLP head).
            - policy_head_hidden_channels (:obj:`SequenceType`): The number of hidden layers used in policy head (MLP head).
            - reward_support_size (:obj:`int`): The size of categorical reward output
            - value_support_size (:obj:`int`): The size of categorical value output.
            - proj_hid (:obj:`int`): The size of projection hidden layer.
            - proj_out (:obj:`int`): The size of projection output layer.
            - pred_hid (:obj:`int`): The size of prediction hidden layer.
            - pred_out (:obj:`int`): The size of prediction output layer.
            - self_supervised_learning_loss (:obj:`bool`): Whether to use self_supervised_learning related networks \
                in MuZero model, default set it to False.
            - categorical_distribution (:obj:`bool`): Whether to use discrete support to represent categorical \
                distribution for value and reward.
            - activation (:obj:`Optional[nn.Module]`): Activation function used in network, which often use in-place \
                operation to speedup, e.g. ReLU(inplace=True).
            - last_linear_layer_init_zero (:obj:`bool`): Whether to use zero initializations for the last layer of \
                dynamics/prediction mlp, default sets it to True.
            - state_norm (:obj:`bool`): Whether to use normalization for hidden states, default set it to False.
            - downsample (:obj:`bool`): Whether to do downsampling for observations in ``representation_network``, \
                defaults to True. This option is often used in video games like Atari. In board games like go, \
                we don't need this module.
            - norm_type (:obj:`str`): The type of normalization in networks. defaults to 'BN'.
            - discrete_action_encoding_type (:obj:`str`): The type of encoding for discrete action. Default sets it to 'one_hot'. options = {'one_hot', 'not_one_hot'}
        """
        super(MuZeroContextModel, self).__init__()

        self.timestep = 0
        self.context_length_init = context_length_init  # NOTE

        if isinstance(observation_shape, int) or len(observation_shape) == 1:
            # for vector obs input, e.g. classical control and box2d environments
            # to be compatible with LightZero model/policy, transform to shape: [C, W, H]
            observation_shape = [1, observation_shape, 1]

        self.categorical_distribution = categorical_distribution
        if self.categorical_distribution:
            self.reward_support_size = reward_support_size
            self.value_support_size = value_support_size
        else:
            self.reward_support_size = 1
            self.value_support_size = 1

        self.action_space_size = action_space_size
        print('action_space_size:', action_space_size)
        assert discrete_action_encoding_type in ['one_hot', 'not_one_hot'], discrete_action_encoding_type
        self.discrete_action_encoding_type = discrete_action_encoding_type
        if self.discrete_action_encoding_type == 'one_hot':
            self.action_encoding_dim = action_space_size
        elif self.discrete_action_encoding_type == 'not_one_hot':
            self.action_encoding_dim = 1
        self.proj_hid = proj_hid
        self.proj_out = proj_out
        self.pred_hid = pred_hid
        self.pred_out = pred_out
        self.self_supervised_learning_loss = self_supervised_learning_loss
        self.last_linear_layer_init_zero = last_linear_layer_init_zero
        self.state_norm = state_norm
        self.downsample = downsample
        self.analysis_sim_norm = analysis_sim_norm

        if observation_shape[1] == 96:
            latent_size = math.ceil(observation_shape[1] / 16) * math.ceil(observation_shape[2] / 16)
        elif observation_shape[1] == 64:
            latent_size = math.ceil(observation_shape[1] / 8) * math.ceil(observation_shape[2] / 8)

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
            activation=activation,
            norm_type=norm_type,
            embedding_dim=768,
            group_size=8,
            use_sim_norm=use_sim_norm,  # NOTE
        )

        # ====== for analysis ======
        if self.analysis_sim_norm:
            self.encoder_hook = FeatureAndGradientHook()
            self.encoder_hook.setup_hooks(self.representation_network)

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
            norm_type=norm_type,
            embedding_dim=768,
            group_size=8,
            use_sim_norm=use_sim_norm,  # NOTE
        )
        self.prediction_network = PredictionNetwork(
            observation_shape,
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
            activation=activation,
            norm_type=norm_type
        )

        if self.self_supervised_learning_loss:
            # projection used in EfficientZero
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

    def initial_inference(self, obs: torch.Tensor, action_batch=None, current_obs_batch=None) -> MZNetworkOutput:
        """
        Overview:
            Initial inference of MuZero model, which is the first step of the MuZero model.
            To perform the initial inference, we first use the representation network to obtain the ``latent_state``.
            Then we use the prediction network to predict ``value`` and ``policy_logits`` of the ``latent_state``.
        Arguments:
            - obs (:obj:`torch.Tensor`): The 2D image observation data.
        Returns (MZNetworkOutput):
            - value (:obj:`torch.Tensor`): The output value of input state to help policy improvement and evaluation.
            - reward (:obj:`torch.Tensor`): The predicted reward of input state and selected action. \
                In initial inference, we set it to zero vector.
            - policy_logits (:obj:`torch.Tensor`): The output logit to select discrete action.
            - latent_state (:obj:`torch.Tensor`): The encoding latent state of input state.
        Shapes:
            - obs (:obj:`torch.Tensor`): :math:`(B, num_channel, obs_shape[1], obs_shape[2])`, where B is batch_size.
            - value (:obj:`torch.Tensor`): :math:`(B, value_support_size)`, where B is batch_size.
            - reward (:obj:`torch.Tensor`): :math:`(B, reward_support_size)`, where B is batch_size.
            - policy_logits (:obj:`torch.Tensor`): :math:`(B, action_dim)`, where B is batch_size.
            - latent_state (:obj:`torch.Tensor`): :math:`(B, H_, W_)`, where B is batch_size, H_ is the height of \
                latent state, W_ is the width of latent state.
         """
        batch_size = obs.size(0)

        if self.training or action_batch is None:
            # train phase
            self.latent_state = self._representation(obs)
            self.timestep = 0
        else:
            # collect/eval phase
            if action_batch is not None and max(action_batch) == -1:  # The first step of an episode
                self.latent_state = self._representation(current_obs_batch)
            else:
                action_batch = torch.from_numpy(np.array(action_batch)).to(self.latent_state.device)
                self.recurrent_inference(self.latent_state, action_batch)  # update self.latent_state
                if self.timestep % self.context_length_init == 0:
                    # print(f'self.timestep:{self.timestep}, reset latent_state')
                    # TODO: the method that use the recent context, rather than the hard reset
                    self.latent_state = self._representation(current_obs_batch)
        policy_logits, value = self._prediction(self.latent_state)
        self.timestep += 1
        return MZNetworkOutput(
            value,
            [0. for _ in range(batch_size)],
            policy_logits,
            self.latent_state,
        )

    def recurrent_inference(self, latent_state: torch.Tensor, action: torch.Tensor) -> MZNetworkOutput:
        """
        Overview:
            Recurrent inference of MuZero model, which is the rollout step of the MuZero model.
            To perform the recurrent inference, we first use the dynamics network to predict ``next_latent_state``,
            ``reward``, by the given current ``latent_state`` and ``action``.
            We then use the prediction network to predict the ``value`` and ``policy_logits`` of the current
            ``latent_state``.
        Arguments:
            - latent_state (:obj:`torch.Tensor`): The encoding latent state of input state.
            - action (:obj:`torch.Tensor`): The predicted action to rollout.
        Returns (MZNetworkOutput):
            - value (:obj:`torch.Tensor`): The output value of input state to help policy improvement and evaluation.
            - reward (:obj:`torch.Tensor`): The predicted reward of input state and selected action.
            - policy_logits (:obj:`torch.Tensor`): The output logit to select discrete action.
            - latent_state (:obj:`torch.Tensor`): The encoding latent state of input state.
            - next_latent_state (:obj:`torch.Tensor`): The predicted next latent state.
        Shapes:
            - obs (:obj:`torch.Tensor`): :math:`(B, num_channel, obs_shape[1], obs_shape[2])`, where B is batch_size.
            - action (:obj:`torch.Tensor`): :math:`(B, )`, where B is batch_size.
            - value (:obj:`torch.Tensor`): :math:`(B, value_support_size)`, where B is batch_size.
            - reward (:obj:`torch.Tensor`): :math:`(B, reward_support_size)`, where B is batch_size.
            - policy_logits (:obj:`torch.Tensor`): :math:`(B, action_dim)`, where B is batch_size.
            - latent_state (:obj:`torch.Tensor`): :math:`(B, H_, W_)`, where B is batch_size, H_ is the height of \
                latent state, W_ is the width of latent state.
            - next_latent_state (:obj:`torch.Tensor`): :math:`(B, H_, W_)`, where B is batch_size, H_ is the height of \
                latent state, W_ is the width of latent state.
         """
        next_latent_state, reward = self._dynamics(latent_state, action)
        policy_logits, value = self._prediction(next_latent_state)
        self.latent_state = next_latent_state  # NOTE: update latent_state
        return MZNetworkOutput(value, reward, policy_logits, next_latent_state)


class DynamicsNetwork(nn.Module):

    def __init__(
        self,
        observation_shape: SequenceType,
        action_encoding_dim: int = 2,
        num_res_blocks: int = 1,
        num_channels: int = 64,
        reward_head_channels: int = 64,
        reward_head_hidden_channels: SequenceType = [32],
        output_support_size: int = 601,
        flatten_input_size_for_reward_head: int = 64,
        downsample: bool = False,
        last_linear_layer_init_zero: bool = True,
        activation: Optional[nn.Module] = nn.ReLU(inplace=True),
        norm_type: Optional[str] = 'BN',
        embedding_dim: int = 256,
        group_size: int = 8,
        use_sim_norm: bool = False,
    ):
        """
        Overview:
            The definition of dynamics network in MuZero algorithm, which is used to predict next latent state and
            reward given current latent state and action.
        Arguments:
            - observation_shape (:obj:`SequenceType`): The shape of input observation, e.g., (12, 96, 96).
            - action_encoding_dim (:obj:`int`): The dimension of action encoding.
            - num_res_blocks (:obj:`int`): The number of res blocks in AlphaZero model.
            - num_channels (:obj:`int`): The channels of input, including obs and action encoding.
            - reward_head_channels (:obj:`int`): The channels of reward head.
            - reward_head_hidden_channels (:obj:`SequenceType`): The number of hidden layers of the reward head (MLP head).
            - output_support_size (:obj:`int`): The size of categorical reward output.
            - flatten_input_size_for_reward_head (:obj:`int`): The flatten size of output for reward head, i.e., \
                the input size of reward head.
            - downsample (:obj:`bool`): Whether to downsample the input observation, default set it to False.
            - last_linear_layer_init_zero (:obj:`bool`): Whether to use zero initializationss for the last layer of \
                reward mlp, default sets it to True.
            - activation (:obj:`Optional[nn.Module]`): Activation function used in network, which often use in-place \
                operation to speedup, e.g. ReLU(inplace=True).
            - norm_type (:obj:`str`): The type of normalization in networks. defaults to 'BN'.
        """
        super().__init__()
        assert norm_type in ['BN', 'LN'], "norm_type must in ['BN', 'LN']"
        assert num_channels > action_encoding_dim, f'num_channels:{num_channels} <= action_encoding_dim:{action_encoding_dim}'

        self.num_channels = num_channels
        self.flatten_input_size_for_reward_head = flatten_input_size_for_reward_head

        self.action_encoding_dim = action_encoding_dim
        self.conv = nn.Conv2d(num_channels, num_channels - self.action_encoding_dim, kernel_size=3, stride=1, padding=1, bias=False)
        
        if norm_type == 'BN':
            self.norm_common = nn.BatchNorm2d(num_channels - self.action_encoding_dim)
        elif norm_type == 'LN':
            if downsample:
                self.norm_common = nn.LayerNorm([num_channels - self.action_encoding_dim, math.ceil(observation_shape[-2] / 16), math.ceil(observation_shape[-1] / 16)])
            else:
                self.norm_common = nn.LayerNorm([num_channels - self.action_encoding_dim, observation_shape[-2], observation_shape[-1]])
            
        self.resblocks = nn.ModuleList(
            [
                ResBlock(
                    in_channels=num_channels - self.action_encoding_dim, activation=activation, norm_type='BN', res_type='basic', bias=False
                ) for _ in range(num_res_blocks)
            ]
        )

        self.conv1x1_reward = nn.Conv2d(num_channels - self.action_encoding_dim, reward_head_channels, 1)

        if norm_type == 'BN':
            self.norm_reward = nn.BatchNorm2d(reward_head_channels)
        elif norm_type == 'LN':
            if downsample:
                self.norm_reward = nn.LayerNorm([reward_head_channels, math.ceil(observation_shape[-2] / 16), math.ceil(observation_shape[-1] / 16)])
            else:
                self.norm_reward = nn.LayerNorm([reward_head_channels, observation_shape[-2], observation_shape[-1]])

        self.fc_reward_head = MLP(
            self.flatten_input_size_for_reward_head,
            hidden_channels=reward_head_hidden_channels[0],
            layer_num=len(reward_head_hidden_channels) + 1,
            out_channels=output_support_size,
            activation=activation,
            norm_type=norm_type,
            output_activation=False,
            output_norm=False,
            last_linear_layer_init_zero=last_linear_layer_init_zero
        )
        self.activation = activation
        self.use_sim_norm = use_sim_norm
        if self.use_sim_norm:
            self.sim_norm = SimNorm(simnorm_dim=group_size)

    def forward(self, state_action_encoding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
         Overview:
            Forward computation of the dynamics network. Predict the next latent state given current latent state and action.
         Arguments:
             - state_action_encoding (:obj:`torch.Tensor`): The state-action encoding, which is the concatenation of \
                    latent state and action encoding, with shape (batch_size, num_channels, height, width).
         Returns:
             - next_latent_state (:obj:`torch.Tensor`): The next latent state, with shape (batch_size, num_channels, \
                    height, width).
            - reward (:obj:`torch.Tensor`): The predicted reward, with shape (batch_size, output_support_size).
         """
        # take the state encoding, state_action_encoding[:, -self.action_encoding_dim:, :, :] is action encoding
        state_encoding = state_action_encoding[:, :-self.action_encoding_dim:, :, :]
        x = self.conv(state_action_encoding)
        x = self.norm_common(x)

        # the residual link: add state encoding to the state_action encoding
        x += state_encoding
        x = self.activation(x)

        for block in self.resblocks:
            x = block(x)
        next_latent_state = x

        x = self.conv1x1_reward(next_latent_state)
        x = self.norm_reward(x)
        x = self.activation(x)
        x = x.view(-1, self.flatten_input_size_for_reward_head)

        # use the fully connected layer to predict reward
        reward = self.fc_reward_head(x)

        if self.use_sim_norm:
            next_latent_state = self.sim_norm(next_latent_state)

        return next_latent_state, reward

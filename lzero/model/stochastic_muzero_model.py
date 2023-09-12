from typing import Optional, Tuple

import math
import torch
import torch.nn as nn
from ding.torch_utils import MLP, ResBlock
from ding.utils import MODEL_REGISTRY, SequenceType

from .common import MZNetworkOutput, RepresentationNetwork, PredictionNetwork
from .utils import renormalize, get_params_mean, get_dynamic_mean, get_reward_mean


# use ModelRegistry to register the model, for more details about ModelRegistry, please refer to DI-engine's document.
@MODEL_REGISTRY.register('StochasticMuZeroModel')
class StochasticMuZeroModel(nn.Module):

    def __init__(
            self,
            observation_shape: SequenceType = (12, 96, 96),
            action_space_size: int = 6,
            chance_space_size: int = 2,
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
            *args,
            **kwargs
    ):
        """
        Overview:
            The definition of the neural network model used in Stochastic MuZero,
            which is proposed in the paper https://openreview.net/pdf?id=X6D9bAHhBQ1.
            Stochastic MuZero model consists of a representation network, a dynamics network and a prediction network.
            The networks are built on convolution residual blocks and fully connected layers.
        Arguments:
            - observation_shape (:obj:`SequenceType`): Observation space shape, e.g. [C, W, H]=[12, 96, 96] for Atari.
            - action_space_size: (:obj:`int`): Action space size, usually an integer number for discrete action space.
            - chance_space_size: (:obj:`int`): Chance space size, the action space for decision node, usually an integer
                number for discrete action space.
            - num_res_blocks (:obj:`int`): The number of res blocks in AlphaZero model.
            - num_channels (:obj:`int`): The channels of hidden states.
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
            - self_supervised_learning_loss (:obj:`bool`): Whether to use self_supervised_learning related networks \
                in Stochastic MuZero model, default set it to False.
            - categorical_distribution (:obj:`bool`): Whether to use discrete support to represent categorical \
                distribution for value and reward.
            - activation (:obj:`Optional[nn.Module]`): Activation function used in network, which often use in-place \
                operation to speedup, e.g. ReLU(inplace=True).
            - last_linear_layer_init_zero (:obj:`bool`): Whether to use zero initialization for the last layer of \
                dynamics/prediction mlp, default set it to True.
            - state_norm (:obj:`bool`): Whether to use normalization for hidden states, default set it to False.
            - downsample (:obj:`bool`): Whether to do downsampling for observations in ``representation_network``, \
                defaults to True. This option is often used in video games like Atari. In board games like go, \
                we don't need this module.
        """
        super(StochasticMuZeroModel, self).__init__()
        self.categorical_distribution = categorical_distribution
        if self.categorical_distribution:
            self.reward_support_size = reward_support_size
            self.value_support_size = value_support_size
        else:
            self.reward_support_size = 1
            self.value_support_size = 1

        self.action_space_size = action_space_size
        self.chance_space_size = chance_space_size

        self.proj_hid = proj_hid
        self.proj_out = proj_out
        self.pred_hid = pred_hid
        self.pred_out = pred_out
        self.self_supervised_learning_loss = self_supervised_learning_loss
        self.last_linear_layer_init_zero = last_linear_layer_init_zero
        self.state_norm = state_norm
        self.downsample = downsample

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
        )

        self.chance_encoder = ChanceEncoder(
            observation_shape, chance_space_size
        )
        self.dynamics_network = DynamicsNetwork(
            num_res_blocks,
            num_channels + 1,
            reward_head_channels,
            fc_reward_layers,
            self.reward_support_size,
            flatten_output_size_for_reward_head,
            last_linear_layer_init_zero=self.last_linear_layer_init_zero,
        )
        self.prediction_network = PredictionNetwork(
            observation_shape,
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
            last_linear_layer_init_zero=self.last_linear_layer_init_zero,
        )

        self.afterstate_dynamics_network = AfterstateDynamicsNetwork(
            num_res_blocks,
            num_channels + 1,
            reward_head_channels,
            fc_reward_layers,
            self.reward_support_size,
            flatten_output_size_for_reward_head,
            last_linear_layer_init_zero=self.last_linear_layer_init_zero,
        )
        self.afterstate_prediction_network = AfterstatePredictionNetwork(
            chance_space_size,
            num_res_blocks,
            num_channels,
            value_head_channels,
            policy_head_channels,
            fc_value_layers,
            fc_policy_layers,
            self.value_support_size,
            flatten_output_size_for_value_head,
            flatten_output_size_for_policy_head,
            last_linear_layer_init_zero=self.last_linear_layer_init_zero,
        )

        if self.self_supervised_learning_loss:
            # projection used in EfficientZero
            if self.downsample:
                # In Atari, if the observation_shape is set to (12, 96, 96), which indicates the original shape of
                # (3,96,96), and frame_stack_num is 4. Due to downsample, the encoding of observation (latent_state) is
                # (64, 96/16, 96/16), where 64 is the number of channels, 96/16 is the size of the latent state. Thus,
                # self.projection_input_dim = 64 * 96/16 * 96/16 = 64*6*6 = 2304
                ceil_size = math.ceil(observation_shape[1] / 16) * math.ceil(observation_shape[2] / 16)
                self.projection_input_dim = num_channels * ceil_size
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
            Initial inference of Stochastic MuZero model, which is the first step of the Stochastic MuZero model.
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
        latent_state = self._representation(obs)
        policy_logits, value = self._prediction(latent_state)
        return MZNetworkOutput(
            value,
            [0. for _ in range(batch_size)],
            policy_logits,
            latent_state,
        )

    def recurrent_inference(self, state: torch.Tensor, option: torch.Tensor,
                            afterstate: bool = False) -> MZNetworkOutput:
        """
        Overview:
            Recurrent inference of Stochastic MuZero model, which is the rollout step of the Stochastic MuZero model.
            To perform the recurrent inference, we first use the dynamics network to predict ``next_latent_state``,
            ``reward``, by the given current ``latent_state`` and ``action``.
            We then use the prediction network to predict the ``value`` and ``policy_logits`` of the current
            ``latent_state``.
        Arguments:
            - state (:obj:`torch.Tensor`): The encoding latent state of input state or the afterstate.
            - option (:obj:`torch.Tensor`):  The action to rollout or the chance to predict next latent state.
            - afterstate (:obj:`bool`): Whether to use afterstate prediction network to predict next latent state.
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

        if afterstate:
            # state is afterstate, option is chance
            next_latent_state, reward = self._dynamics(state, option)
            policy_logits, value = self._prediction(next_latent_state)
            return MZNetworkOutput(value, reward, policy_logits, next_latent_state)
        else:
            # state is latent_state, option is action
            next_afterstate, reward = self._afterstate_dynamics(state, option)
            policy_logits, value = self._afterstate_prediction(next_afterstate)
            return MZNetworkOutput(value, reward, policy_logits, next_afterstate)

    def _representation(self, observation: torch.Tensor) -> torch.Tensor:
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

    def chance_encode(self, observation: torch.Tensor):
        output = self.chance_encoder(observation)
        return output

    def _prediction(self, latent_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Overview:
            Use the prediction network to predict ``policy_logits`` and ``value``.
        Arguments:
            - latent_state (:obj:`torch.Tensor`): The encoding latent state of input state.
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

    def _afterstate_prediction(self, afterstate: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Overview:
            Use the prediction network to predict ``policy_logits`` and ``value``.
        Arguments:
            - latent_state (:obj:`torch.Tensor`): The encoding latent state of input state.
        Returns:
            - policy_logits (:obj:`torch.Tensor`): The output logit to select discrete action.
            - value (:obj:`torch.Tensor`): The output value of input state to help policy improvement and evaluation.
        Shapes:
            - latent_state (:obj:`torch.Tensor`): :math:`(B, H_, W_)`, where B is batch_size, H_ is the height of \
                latent state, W_ is the width of latent state.
            - policy_logits (:obj:`torch.Tensor`): :math:`(B, action_dim)`, where B is batch_size.
            - value (:obj:`torch.Tensor`): :math:`(B, value_support_size)`, where B is batch_size.
        """
        return self.afterstate_prediction_network(afterstate)

    def _dynamics(self, latent_state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Overview:
            Concatenate ``latent_state`` and ``action`` and use the dynamics network to predict ``next_latent_state``
            and ``reward``.
        Arguments:
            - latent_state (:obj:`torch.Tensor`): The encoding latent state of input state.
            - action (:obj:`torch.Tensor`): The predicted action to rollout.
        Returns:
            - next_latent_state (:obj:`torch.Tensor`): The predicted latent state of the next timestep.
            - reward (:obj:`torch.Tensor`): The predicted reward of the current latent state and selected action.
        Shapes:
            - latent_state (:obj:`torch.Tensor`): :math:`(B, H_, W_)`, where B is batch_size, H_ is the height of \
                latent state, W_ is the width of latent state.
            - action (:obj:`torch.Tensor`): :math:`(B, )`, where B is batch_size.
            - next_latent_state (:obj:`torch.Tensor`): :math:`(B, H_, W_)`, where B is batch_size, H_ is the height of \
                latent state, W_ is the width of latent state.
            - reward (:obj:`torch.Tensor`): :math:`(B, reward_support_size)`, where B is batch_size.
        """
        # NOTE: the discrete action encoding type is important for some environments

        # discrete action space
        # the final action_encoding shape is (batch_size, 1, latent_state[2], latent_state[3]), e.g. (8, 1, 4, 1).
        action_encoding = (
            torch.ones((
                latent_state.shape[0],
                1,
                latent_state.shape[2],
                latent_state.shape[3],
            )).to(action.device).float()
        )
        if len(action.shape) == 2:
            # (batch_size, action_dim) -> (batch_size, action_dim, 1)
            # e.g.,  torch.Size([8, 1]) ->  torch.Size([8, 1, 1])
            action = action.unsqueeze(-1)
        elif len(action.shape) == 1:
            # (batch_size,) -> (batch_size, action_dim=1, 1)
            # e.g.,  -> torch.Size([8, 1]) ->  torch.Size([8, 1, 1])
            action = action.unsqueeze(-1).unsqueeze(-1)

        # action[:, 0, None, None] shape:  (batch_size, action_dim, 1, 1) e.g. (8, 1, 1, 1)
        # the final action_encoding shape: (batch_size, 1, latent_state[2], latent_state[3]) e.g. (8, 1, 4, 1),
        # where each element is normalized as action[i]/action_space_size
        action_encoding = (action[:, 0, None, None] * action_encoding / self.chance_space_size)

        # state_action_encoding shape: (batch_size, latent_state[1] + 1, latent_state[2], latent_state[3])
        state_action_encoding = torch.cat((latent_state, action_encoding), dim=1)

        next_latent_state, reward = self.dynamics_network(state_action_encoding)
        if self.state_norm:
            next_latent_state = renormalize(next_latent_state)
        return next_latent_state, reward

    def _afterstate_dynamics(self, latent_state: torch.Tensor, action: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor]:
        """
        Overview:
            Concatenate ``latent_state`` and ``action`` and use the dynamics network to predict ``next_latent_state``
            and ``reward``.
        Arguments:
            - latent_state (:obj:`torch.Tensor`): The encoding latent state of input state.
            - action (:obj:`torch.Tensor`): The predicted action to rollout.
        Returns:
            - next_latent_state (:obj:`torch.Tensor`): The predicted latent state of the next timestep.
            - reward (:obj:`torch.Tensor`): The predicted reward of the current latent state and selected action.
        Shapes:
            - latent_state (:obj:`torch.Tensor`): :math:`(B, H_, W_)`, where B is batch_size, H_ is the height of \
                latent state, W_ is the width of latent state.
            - action (:obj:`torch.Tensor`): :math:`(B, )`, where B is batch_size.
            - next_latent_state (:obj:`torch.Tensor`): :math:`(B, H_, W_)`, where B is batch_size, H_ is the height of \
                latent state, W_ is the width of latent state.
            - reward (:obj:`torch.Tensor`): :math:`(B, reward_support_size)`, where B is batch_size.
        """
        # NOTE: the discrete action encoding type is important for some environments

        # discrete action space
        # the final action_encoding shape is (batch_size, 1, latent_state[2], latent_state[3]), e.g. (8, 1, 4, 1).
        action_encoding = (
            torch.ones((
                latent_state.shape[0],
                1,
                latent_state.shape[2],
                latent_state.shape[3],
            )).to(action.device).float()
        )
        if len(action.shape) == 2:
            # (batch_size, action_dim) -> (batch_size, action_dim, 1)
            # e.g.,  torch.Size([8, 1]) ->  torch.Size([8, 1, 1])
            action = action.unsqueeze(-1)
        elif len(action.shape) == 1:
            # (batch_size,) -> (batch_size, action_dim=1, 1)
            # e.g.,  -> torch.Size([8, 1]) ->  torch.Size([8, 1, 1])
            action = action.unsqueeze(-1).unsqueeze(-1)

        # action[:, 0, None, None] shape:  (batch_size, action_dim, 1, 1) e.g. (8, 1, 1, 1)
        # the final action_encoding shape: (batch_size, 1, latent_state[2], latent_state[3]) e.g. (8, 1, 4, 1),
        # where each element is normalized as action[i]/action_space_size
        action_encoding = (action[:, 0, None, None] * action_encoding / self.action_space_size)

        # state_action_encoding shape: (batch_size, latent_state[1] + 1, latent_state[2], latent_state[3])
        state_action_encoding = torch.cat((latent_state, action_encoding), dim=1)

        next_latent_state, reward = self.afterstate_dynamics_network(state_action_encoding)
        if self.state_norm:
            next_latent_state = renormalize(next_latent_state)
        return next_latent_state, reward

    def project(self, latent_state: torch.Tensor, with_grad: bool = True) -> torch.Tensor:
        """
        Overview:
            Project the latent state to a lower dimension to calculate the self-supervised loss, which is involved in
            in EfficientZero.
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

    def get_params_mean(self) -> float:
        return get_params_mean(self)


class DynamicsNetwork(nn.Module):

    def __init__(
            self,
            num_res_blocks: int,
            num_channels: int,
            reward_head_channels: int,
            fc_reward_layers: SequenceType,
            output_support_size: int,
            flatten_output_size_for_reward_head: int,
            last_linear_layer_init_zero: bool = True,
            activation: Optional[nn.Module] = nn.ReLU(inplace=True),
    ):
        """
        Overview:
            The definition of dynamics network in Stochastic MuZero algorithm, which is used to predict next latent state and
            reward given current latent state and action.
        Arguments:
            - num_res_blocks (:obj:`int`): The number of res blocks in AlphaZero model.
            - num_channels (:obj:`int`): The channels of input, including obs and action encoding.
            - reward_head_channels (:obj:`int`): The channels of reward head.
            - fc_reward_layers (:obj:`SequenceType`): The number of hidden layers of the reward head (MLP head).
            - output_support_size (:obj:`int`): The size of categorical reward output.
            - flatten_output_size_for_reward_head (:obj:`int`): The flatten size of output for reward head, i.e., \
                the input size of reward head.
            - last_linear_layer_init_zero (:obj:`bool`): Whether to use zero initialization for the last layer of \
                reward mlp, default set it to True.
            - activation (:obj:`Optional[nn.Module]`): Activation function used in network, which often use in-place \
                operation to speedup, e.g. ReLU(inplace=True).
        """
        super().__init__()
        self.num_channels = num_channels
        self.flatten_output_size_for_reward_head = flatten_output_size_for_reward_head

        self.conv = nn.Conv2d(num_channels, num_channels - 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(num_channels - 1)
        self.resblocks = nn.ModuleList(
            [
                ResBlock(
                    in_channels=num_channels - 1, activation=activation, norm_type='BN', res_type='basic', bias=False
                ) for _ in range(num_res_blocks)
            ]
        )

        self.conv1x1_reward = nn.Conv2d(num_channels - 1, reward_head_channels, 1)
        self.bn_reward = nn.BatchNorm2d(reward_head_channels)
        self.fc_reward_head = MLP(
            self.flatten_output_size_for_reward_head,
            hidden_channels=fc_reward_layers[0],
            layer_num=len(fc_reward_layers) + 1,
            out_channels=output_support_size,
            activation=activation,
            norm_type='BN',
            output_activation=False,
            output_norm=False,
            last_linear_layer_init_zero=last_linear_layer_init_zero
        )
        self.activation = activation

    def forward(self, state_action_encoding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
         Overview:
            Forward computation of the dynamics network. Predict next latent state given current latent state and action.
         Arguments:
             - state_action_encoding (:obj:`torch.Tensor`): The state-action encoding, which is the concatenation of \
                    latent state and action encoding, with shape (batch_size, num_channels, height, width).
         Returns:
             - next_latent_state (:obj:`torch.Tensor`): The next latent state, with shape (batch_size, num_channels, \
                    height, width).
            - reward (:obj:`torch.Tensor`): The predicted reward, with shape (batch_size, output_support_size).
         """
        # take the state encoding (latent_state),  state_action_encoding[:, -1, :, :] is action encoding
        latent_state = state_action_encoding[:, :-1, :, :]
        x = self.conv(state_action_encoding)
        x = self.bn(x)

        # the residual link: add state encoding to the state_action encoding
        x += latent_state
        x = self.activation(x)

        for block in self.resblocks:
            x = block(x)
        next_latent_state = x

        x = self.conv1x1_reward(next_latent_state)
        x = self.bn_reward(x)
        x = self.activation(x)
        x = x.view(-1, self.flatten_output_size_for_reward_head)

        # use the fully connected layer to predict reward
        reward = self.fc_reward_head(x)

        return next_latent_state, reward

    def get_dynamic_mean(self) -> float:
        return get_dynamic_mean(self)

    def get_reward_mean(self) -> float:
        return get_reward_mean(self)


# TODO(pu): customize different afterstate dynamics network
AfterstateDynamicsNetwork = DynamicsNetwork


class AfterstatePredictionNetwork(nn.Module):
    def __init__(
            self,
            action_space_size: int,
            num_res_blocks: int,
            num_channels: int,
            value_head_channels: int,
            policy_head_channels: int,
            fc_value_layers: int,
            fc_policy_layers: int,
            output_support_size: int,
            flatten_output_size_for_value_head: int,
            flatten_output_size_for_policy_head: int,
            last_linear_layer_init_zero: bool = True,
            activation: nn.Module = nn.ReLU(inplace=True),
    ) -> None:
        """
        Overview:
            The definition of afterstate policy and value prediction network, which is used to predict value and policy by the
            given afterstate.
        Arguments:
            - action_space_size: (:obj:`int`): Action space size, usually an integer number for discrete action space.
            - num_res_blocks (:obj:`int`): The number of res blocks in AlphaZero model.
            - num_channels (:obj:`int`): The channels of hidden states.
            - value_head_channels (:obj:`int`): The channels of value head.
            - policy_head_channels (:obj:`int`): The channels of policy head.
            - fc_value_layers (:obj:`SequenceType`): The number of hidden layers used in value head (MLP head).
            - fc_policy_layers (:obj:`SequenceType`): The number of hidden layers used in policy head (MLP head).
            - output_support_size (:obj:`int`): The size of categorical value output.
            - self_supervised_learning_loss (:obj:`bool`): Whether to use self_supervised_learning related networks \
            - flatten_output_size_for_value_head (:obj:`int`): The size of flatten hidden states, i.e. the input size \
                of the value head.
            - flatten_output_size_for_policy_head (:obj:`int`): The size of flatten hidden states, i.e. the input size \
                of the policy head.
            - last_linear_layer_init_zero (:obj:`bool`): Whether to use zero initialization for the last layer of \
                dynamics/prediction mlp, default set it to True.
            - activation (:obj:`Optional[nn.Module]`): Activation function used in network, which often use in-place \
                operation to speedup, e.g. ReLU(inplace=True).
        """
        super(AfterstatePredictionNetwork, self).__init__()
        self.resblocks = nn.ModuleList(
            [
                ResBlock(in_channels=num_channels, activation=activation, norm_type='BN', res_type='basic', bias=False)
                for _ in range(num_res_blocks)
            ]
        )
        self.conv1x1_value = nn.Conv2d(num_channels, value_head_channels, 1)
        self.conv1x1_policy = nn.Conv2d(num_channels, policy_head_channels, 1)
        self.bn_value = nn.BatchNorm2d(value_head_channels)
        self.bn_policy = nn.BatchNorm2d(policy_head_channels)
        self.flatten_output_size_for_value_head = flatten_output_size_for_value_head
        self.flatten_output_size_for_policy_head = flatten_output_size_for_policy_head
        self.activation = activation

        self.fc_value = MLP(
            in_channels=self.flatten_output_size_for_value_head,
            hidden_channels=fc_value_layers[0],
            out_channels=output_support_size,
            layer_num=len(fc_value_layers) + 1,
            activation=self.activation,
            norm_type='BN',
            output_activation=False,
            output_norm=False,
            last_linear_layer_init_zero=last_linear_layer_init_zero
        )
        self.fc_policy = MLP(
            in_channels=self.flatten_output_size_for_policy_head,
            hidden_channels=fc_policy_layers[0],
            out_channels=action_space_size,
            layer_num=len(fc_policy_layers) + 1,
            activation=self.activation,
            norm_type='BN',
            output_activation=False,
            output_norm=False,
            last_linear_layer_init_zero=last_linear_layer_init_zero
        )

    def forward(self, afterstate: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Overview:
            Forward computation of the afterstate prediction network.
        Arguments:
            - afterstate (:obj:`torch.Tensor`): input tensor with shape (B, afterstate_dim).
        Returns:
            - afterstate_policy_logits (:obj:`torch.Tensor`): policy tensor with shape (B, action_space_size).
            - afterstate_value (:obj:`torch.Tensor`): value tensor with shape (B, output_support_size).
        """
        for res_block in self.resblocks:
            afterstate = res_block(afterstate)

        value = self.conv1x1_value(afterstate)
        value = self.bn_value(value)
        value = self.activation(value)

        policy = self.conv1x1_policy(afterstate)
        policy = self.bn_policy(policy)
        policy = self.activation(policy)

        value = value.reshape(-1, self.flatten_output_size_for_value_head)
        policy = policy.reshape(-1, self.flatten_output_size_for_policy_head)

        afterstate_value = self.fc_value(value)
        afterstate_policy_logits = self.fc_policy(policy)
        return afterstate_policy_logits, afterstate_value


class ChanceEncoderBackbone(nn.Module):
    """
    Overview:
        The definition of chance encoder backbone network, \
        which is used to encode the (image) observation into a latent space.
    Arguments:
        - input_dimensions (:obj:`tuple`): The dimension of observation space.
        - chance_encoding_dim (:obj:`int`): The dimension of chance encoding.
    """

    def __init__(self, input_dimensions, chance_encoding_dim=4):
        super(ChanceEncoderBackbone, self).__init__()
        self.conv1 = nn.Conv2d(input_dimensions[0] * 2, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * input_dimensions[1] * input_dimensions[2], 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, chance_encoding_dim)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.shape[0], -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ChanceEncoderBackboneMLP(nn.Module):
    """
    Overview:
        The definition of chance encoder backbone network, \
        which is used to encode the (vector) observation into a latent space.
    Arguments:
        - input_dimensions (:obj:`tuple`): The dimension of observation space.
        - chance_encoding_dim (:obj:`int`): The dimension of chance encoding.
    """

    def __init__(self, input_dimensions, chance_encoding_dim=4):
        super(ChanceEncoderBackboneMLP, self).__init__()
        self.fc1 = nn.Linear(input_dimensions, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, chance_encoding_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ChanceEncoder(nn.Module):

    def __init__(self, input_dimensions, action_dimension, encoder_backbone_type='conv'):
        super().__init__()
        # Specify the action space for the model
        self.action_space = action_dimension
        if encoder_backbone_type == 'conv':
            # Define the encoder, which transforms observations into a latent space
            self.encoder = ChanceEncoderBackbone(input_dimensions, action_dimension)
        elif encoder_backbone_type == 'mlp':
            self.encoder = ChanceEncoderBackboneMLP(input_dimensions, action_dimension)
        else:
            raise ValueError('Encoder backbone type not supported')

        # Using the Straight Through Estimator method for backpropagation
        self.onehot_argmax = StraightThroughEstimator()

    def forward(self, observations):
        """
        Overview:
            Forward method for the ChanceEncoder. This method takes an observation \
            and applies the encoder to transform it to a latent space. Then applies the \
            StraightThroughEstimator to this encoding. \

            References: Planning in Stochastic Environments with a Learned Model (ICLR 2022), page 5,
            Chance Outcomes section.
        Arguments:
            - observations (:obj:`torch.Tensor`): Observation tensor.
        Returns:
            - chance (:obj:`torch.Tensor`): Transformed tensor after applying one-hot argmax.
            - chance_encoding (:obj:`torch.Tensor`): Encoding of the input observation tensor.
        """
        # Apply the encoder to the observation
        chance_encoding = self.encoder(observations)
        # Apply one-hot argmax to the encoding
        chance_onehot = self.onehot_argmax(chance_encoding)
        return chance_encoding, chance_onehot


class StraightThroughEstimator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        Overview:
            Forward method for the StraightThroughEstimator. This applies the one-hot argmax \
            function to the input tensor.
        Arguments:
            - x (:obj:`torch.Tensor`): Input tensor.
        Returns:
            - (:obj:`torch.Tensor`): Transformed tensor after applying one-hot argmax.
        """
        # Apply one-hot argmax to the input
        x = OnehotArgmax.apply(x)
        return x


class OnehotArgmax(torch.autograd.Function):
    """
    Overview:
        Custom PyTorch function for one-hot argmax. This function transforms the input tensor \
        into a one-hot tensor where the index with the maximum value in the original tensor is \
        set to 1 and all other indices are set to 0. It allows gradients to flow to the encoder \
        during backpropagation.

        For more information, refer to: \
        https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
    """

    @staticmethod
    def forward(ctx, input):
        """
        Overview:
            Forward method for the one-hot argmax function. This method transforms the input \
            tensor into a one-hot tensor.
        Arguments:
            - ctx (:obj:`context`): A context object that can be used to stash information for
            backward computation.
            - input (:obj:`torch.Tensor`): Input tensor.
        Returns:
            - (:obj:`torch.Tensor`): One-hot tensor.
        """
        # Transform the input tensor to a one-hot tensor
        return torch.zeros_like(input).scatter_(-1, torch.argmax(input, dim=-1, keepdim=True), 1.)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Overview:
            Backward method for the one-hot argmax function. This method allows gradients \
            to flow to the encoder during backpropagation.
        Arguments:
            - ctx (:obj:`context`):  A context object that was stashed in the forward pass.
            - grad_output (:obj:`torch.Tensor`): The gradient of the output tensor.
        Returns:
            - (:obj:`torch.Tensor`): The gradient of the input tensor.
        """
        return grad_output

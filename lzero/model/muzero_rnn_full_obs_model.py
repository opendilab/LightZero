import copy
import math
from typing import Optional, Tuple, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from ding.torch_utils import MLP, ResBlock
from ding.utils import MODEL_REGISTRY, SequenceType

from .common import EZNetworkOutputV2, RepresentationNetwork, PredictionHiddenNetwork, FeatureAndGradientHook
from .utils import renormalize, get_params_mean, SimNorm


# use ModelRegistry to register the model, for more details about ModelRegistry, please refer to DI-engine's document.
@MODEL_REGISTRY.register('MuZeroRNNFullobsModel')
class MuZeroRNNFullobsModel(nn.Module):

    def __init__(
            self,
            observation_shape: SequenceType = (12, 96, 96),
            action_space_size: int = 6,
            rnn_hidden_size: int = 512,
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
            self_supervised_learning_loss: bool = True,
            categorical_distribution: bool = True,
            last_linear_layer_init_zero: bool = True,
            state_norm: bool = False,
            downsample: bool = False,
            activation: Optional[nn.Module] = nn.ReLU(inplace=True),
            norm_type: Optional[str] = 'BN',
            discrete_action_encoding_type: str = 'one_hot',
            context_length_init: int = 5,
            use_sim_norm: bool = False,
            analysis_sim_norm: bool = False,
            collector_env_num: int = 8,
            *args,
            **kwargs
    ) -> None:
        """
        Overview:
            The definition of the network model of EfficientZero, which is a generalization version for 2D image obs.
            The networks are built on convolution residual blocks and fully connected layers.
            EfficientZero model which consists of a representation network, a dynamics network and a prediction network.
        Arguments:
            - observation_shape (:obj:`SequenceType`): Observation space shape, e.g. [C, W, H]=[12, 96, 96] for Atari.
            - action_space_size: (:obj:`int`): Action space size, usually an integer number for discrete action space.
            - rnn_hidden_size (:obj:`int`): The hidden size of LSTM in dynamics network to predict reward.
            - num_res_blocks (:obj:`int`): The number of res blocks in EfficientZero model.
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
            - categorical_distribution (:obj:`bool`): Whether to use discrete support to represent categorical \
                distribution for value and reward/reward.
            - last_linear_layer_init_zero (:obj:`bool`): Whether to use zero initializationss for the last layer of \
                dynamics/prediction mlp, default sets it to True.
            - state_norm (:obj:`bool`): Whether to use normalization for hidden states, default set it to False.
            - downsample (:obj:`bool`): Whether to do downsampling for observations in ``representation_network``, \
                defaults to True. This option is often used in video games like Atari. In board games like go, \
                we don't need this module.
            - activation (:obj:`Optional[nn.Module]`): Activation function used in network, which often use in-place \
                operation to speedup, e.g. ReLU(inplace=True).
            - norm_type (:obj:`str`): The type of normalization in networks. defaults to 'BN'.
            - discrete_action_encoding_type (:obj:`str`): The type of encoding for discrete action. Default sets it to 'one_hot'. 
                options = {'one_hot', 'not_one_hot'}
        """
        super(MuZeroRNNFullobsModel, self).__init__()
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

        self.action_space_size = action_space_size
        assert discrete_action_encoding_type in ['one_hot', 'not_one_hot'], discrete_action_encoding_type
        self.discrete_action_encoding_type = discrete_action_encoding_type
        if self.discrete_action_encoding_type == 'one_hot':
            self.action_encoding_dim = action_space_size
        elif self.discrete_action_encoding_type == 'not_one_hot':
            self.action_encoding_dim = 1
        self.rnn_hidden_size = rnn_hidden_size
        self.proj_hid = proj_hid
        self.proj_out = proj_out
        self.pred_hid = pred_hid
        self.pred_out = pred_out
        self.last_linear_layer_init_zero = last_linear_layer_init_zero
        self.state_norm = state_norm
        self.downsample = downsample
        self.self_supervised_learning_loss = self_supervised_learning_loss
        self.norm_type = norm_type
        self.activation = activation
        self.analysis_sim_norm = analysis_sim_norm
        self.env_num = collector_env_num

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
            activation=self.activation,
            norm_type=self.norm_type,
            embedding_dim=768,
            group_size=8,
            use_sim_norm=use_sim_norm,  # NOTE
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
            rnn_hidden_size,
            last_linear_layer_init_zero=self.last_linear_layer_init_zero,
            activation=activation,
            norm_type=norm_type,
            embedding_dim=768,
            group_size=8,
            use_sim_norm=use_sim_norm,  # NOTE
            res_connection_in_dynamics=True,
        )

        # ====== for analysis ======
        if self.analysis_sim_norm:
            self.encoder_hook = FeatureAndGradientHook()
            self.encoder_hook.setup_hooks(self.representation_network)

        self.prediction_network = PredictionHiddenNetwork(
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
            downsample,
            last_linear_layer_init_zero=self.last_linear_layer_init_zero,
            activation=self.activation,
            norm_type=self.norm_type,
            gru_hidden_size=self.rnn_hidden_size

        )

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

        # self.latent_state_flatten_dim = 64 * 8 * 8  # 4096
        self.projection_input_dim = 64 * 8 * 8  # 4096

        if self.self_supervised_learning_loss:
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

        self.timestep = 0
        self.context_length_init = context_length_init  # TODO
        self.last_ready_env_id = None

    def initial_inference(self, last_obs: torch.Tensor, last_action=None, current_obs=None, ready_env_id=None,
                          last_ready_env_id=None) -> 'EZNetworkOutputV2':
        """
        Perform initial inference based on the phase (training or evaluation/collect).

        Args:
            last_obs (torch.Tensor): The last observation tensor.
            last_action: The last action taken.
            current_obs: The current observation tensor.
            ready_env_id: The ready environment ID.
            last_ready_env_id: The last ready environment ID.

        Returns:
            EZNetworkOutputV2: The output object containing value, policy logits, and latent states.
        """
        if self.training or last_action is None:
            # ===================== Training phase  ======================
            batch_size = last_obs.shape[0]
            self.timestep = 0
            self.current_latent_state = self._representation(last_obs)

            # Initialize hidden state
            self.world_model_latent_history_init_complete = torch.zeros(1, batch_size, self.rnn_hidden_size).to(
                last_obs.device)

            # Compute prediction
            policy_logits, value = self._prediction(self.current_latent_state,
                                                    self.world_model_latent_history_init_complete)
            # NOTE: need to pass the gradient
            selected_world_model_latent_history = self.world_model_latent_history_init_complete
        else:
            #  ===================== Inference phase at Evaluation/Collect  =====================
            batch_size = current_obs.shape[0]

            if last_action is not None and max(last_action) == -1:
                # First step of an episode
                self.current_latent_state = self._representation(current_obs)
                self.world_model_latent_history_init_complete = torch.zeros(1, self.env_num, self.rnn_hidden_size).to(
                    last_obs.device)
                self.last_latent_state = self.current_latent_state
            else:
                # The second to last steps of an episode
                last_action = torch.from_numpy(np.array(last_action)).to(self.current_latent_state.device)
                self.last_latent_state = self._representation(last_obs)  # NOTE: note it's last_obs

                if len(last_ready_env_id) == self.env_num:
                    _, self.world_model_latent_history_init_complete, _ = self._dynamics(self.last_latent_state,
                                                                                         self.world_model_latent_history_init_complete,
                                                                                         last_action)
                else:
                    last_index_tensor = torch.tensor(list(last_ready_env_id))
                    self.world_model_latent_history_init = copy.deepcopy(
                        self.world_model_latent_history_init_complete[:, last_index_tensor, :])
                    _, self.world_model_latent_history_init, _ = self._dynamics(self.last_latent_state,
                                                                                self.world_model_latent_history_init,
                                                                                last_action)
                    self.world_model_latent_history_init_complete[:, last_index_tensor, :] = self.world_model_latent_history_init

                self.current_latent_state = self._representation(current_obs)

                if self.timestep % self.context_length_init == 0:
                    # TODO: use recent context recent
                    self.world_model_latent_history_init_complete = torch.zeros(1, self.env_num,
                                                                                self.rnn_hidden_size).to(
                        last_obs.device)

            if len(ready_env_id) == self.env_num:
                selected_world_model_latent_history = copy.deepcopy(self.world_model_latent_history_init_complete)
                policy_logits, value = self._prediction(self.current_latent_state, selected_world_model_latent_history)
            else:
                # the ready_env_id is not complete, need to select the corresponding latent history
                index_tensor = torch.tensor(list(ready_env_id))
                selected_world_model_latent_history = copy.deepcopy(self.world_model_latent_history_init_complete[:, index_tensor, :])
                policy_logits, value = self._prediction(self.current_latent_state, selected_world_model_latent_history)

        self.timestep += 1
        return EZNetworkOutputV2(value, [0. for _ in range(batch_size)], policy_logits, self.current_latent_state, None,
                               selected_world_model_latent_history)

    def recurrent_inference(
            self,
            latent_state: torch.Tensor,
            world_model_latent_history: Tuple[torch.Tensor],
            action: torch.Tensor,
            next_latent_state: Optional[Tuple[torch.Tensor]] = None,
            ready_env_id: Optional[int] = None
    ) -> EZNetworkOutputV2:
        """
        Perform recurrent inference to predict the next latent state, reward, and policy logits.

        Args:
            latent_state (torch.Tensor): The current latent state tensor.
            world_model_latent_history (Tuple[torch.Tensor]): The history of latent states from the world model.
            action (torch.Tensor): The action tensor.
            next_latent_state (Optional[Tuple[torch.Tensor]], optional): The next latent state tensor if available. Defaults to None.
            ready_env_id (Optional[int], optional): ID of the ready environment. Defaults to None.

        Returns:
            EZNetworkOutputV2: An object containing value, reward, policy logits, next latent state,
                             predicted next latent state, and updated world model latent history.
        """

        # Use the dynamics model to predict the next latent state and reward
        predict_next_latent_state, world_model_latent_history, reward = self._dynamics(
            latent_state, world_model_latent_history, action
        )

        # Determine which latent state to use for prediction
        inference_latent_state = next_latent_state if next_latent_state is not None else predict_next_latent_state

        # Use the prediction model to get policy logits and value
        policy_logits, value = self._prediction(inference_latent_state, world_model_latent_history)

        # If next_latent_state is provided, use it; otherwise, use the predicted next latent state
        return EZNetworkOutputV2(value, reward, policy_logits, next_latent_state, predict_next_latent_state, world_model_latent_history)

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

    def _prediction(self, latent_state: torch.Tensor, world_model_latent_history: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor]:
        """
        Overview:
             use the prediction network to predict the "value" and "policy_logits" of the "latent_state".
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
        return self.prediction_network(latent_state, world_model_latent_history)

    def _dynamics(self, latent_state: torch.Tensor, world_model_latent_history: Tuple[torch.Tensor],
                  action: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor], torch.Tensor]:
        """
        Overview:
            Concatenate ``latent_state`` and ``action`` and use the dynamics network to predict ``next_latent_state``
            ``reward`` and ``next_world_model_latent_history``.
        Arguments:
            - latent_state (:obj:`torch.Tensor`): The encoding latent state of input state.
            - world_model_latent_history (:obj:`Tuple[torch.Tensor]`): The input hidden state of LSTM about reward.
            - action (:obj:`torch.Tensor`): The predicted action to rollout.
        Returns:
            - next_latent_state (:obj:`torch.Tensor`): The predicted latent state of the next timestep.
            - next_world_model_latent_history (:obj:`Tuple[torch.Tensor]`): The output hidden state of LSTM about reward.
            - reward (:obj:`torch.Tensor`): The predicted prefix sum of value for input state.
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

        # state_action_encoding shape: (batch_size, latent_state[1] + action_dim, latent_state[2], latent_state[3]) or
        # (batch_size, latent_state[1] + action_space_size, latent_state[2], latent_state[3]) depending on the discrete_action_encoding_type.
        state_action_encoding = torch.cat((latent_state, action_encoding), dim=1)

        if state_action_encoding.shape[0] != world_model_latent_history.shape[1]:
            print('debug')

        # NOTE: the key difference between EfficientZero and MuZero
        next_latent_state, next_world_model_latent_history, reward = self.dynamics_network(
            state_action_encoding, world_model_latent_history
        )

        if self.state_norm:
            next_latent_state = renormalize(next_latent_state)
        return next_latent_state, next_world_model_latent_history, reward

    def project(self, latent_state: torch.Tensor, with_grad: bool = True) -> torch.Tensor:
        """
        Overview:
            Project the latent state to a lower dimension to calculate the self-supervised loss, which is proposed in EfficientZero.
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

    def get_params_mean(self) -> float:
        return get_params_mean(self)


class DynamicsNetwork(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int],
        action_encoding_dim: int = 2,
        num_res_blocks: int = 1,
        num_channels: int = 64,
        reward_head_channels: int = 64,
        fc_reward_layers: Sequence[int] = [32],
        output_support_size: int = 601,
        flatten_output_size_for_reward_head: int = 64,
        downsample: bool = False,
        rnn_hidden_size: int = 512,
        last_linear_layer_init_zero: bool = True,
        activation: Optional[nn.Module] = nn.ReLU(inplace=True),
        norm_type: Optional[str] = 'BN',
        embedding_dim: int = 256,
        group_size: int = 8,
        use_sim_norm: bool = False,
        res_connection_in_dynamics: bool = True,
    ):
        """
        Define the Dynamics Network for predicting the next latent state, reward,
        and reward hidden state based on the current state and action.

        Args:
            observation_shape (Sequence[int]): Shape of the input observation, e.g., (12, 96, 96).
            action_encoding_dim (int): Dimension of the action encoding.
            num_res_blocks (int): Number of residual blocks in the EfficientZero model.
            num_channels (int): Number of channels in the latent state.
            reward_head_channels (int): Number of channels in the reward head.
            fc_reward_layers (Sequence[int]): Hidden layers in the reward head MLP.
            output_support_size (int): Size of the output for reward classification.
            flatten_output_size_for_reward_head (int): Flattened output size for the reward head.
            downsample (bool): Whether to downsample the input observation. Default is False.
            rnn_hidden_size (int): Hidden size of the LSTM in the dynamics network.
            last_linear_layer_init_zero (bool): Whether to initialize the last reward MLP layer to zero. Default is True.
            activation (Optional[nn.Module]): Activation function used in the network. Default is ReLU(inplace=True).
            norm_type (Optional[str]): Type of normalization used in the network. Default is 'BN'.
            embedding_dim (int): Embedding dimension if using SimNorm.
            group_size (int): Group size for SimNorm.
            use_sim_norm (bool): Whether to use SimNorm. Default is False.
            res_connection_in_dynamics (bool): Whether to use residual connections in the dynamics. Default is True.
        """
        super().__init__()
        assert norm_type in ['BN', 'LN'], "Normalization type must be 'BN' or 'LN'"
        assert num_channels > action_encoding_dim, f'Number of channels:{num_channels} <= action encoding dimension:{action_encoding_dim}'

        self.action_encoding_dim = action_encoding_dim
        self.num_channels = num_channels
        self.rnn_hidden_size = rnn_hidden_size
        self.flatten_output_size_for_reward_head = flatten_output_size_for_reward_head

        self.num_channels_of_latent_state = num_channels - self.action_encoding_dim
        self.activation = activation

        # 1x1 convolution to adjust the number of channels
        self.conv = nn.Conv2d(num_channels, self.num_channels_of_latent_state, kernel_size=1, stride=1, bias=False)

        # Choose between BatchNorm or LayerNorm
        if norm_type == 'BN':
            self.norm_common = nn.BatchNorm2d(self.num_channels_of_latent_state)
        elif norm_type == 'LN':
            self.norm_common = nn.LayerNorm(
                [self.num_channels_of_latent_state,
                 math.ceil(observation_shape[-2] / (16 if downsample else 1)),
                 math.ceil(observation_shape[-1] / (16 if downsample else 1))]
            )

        # Residual Blocks
        self.resblocks = nn.ModuleList([
            ResBlock(
                in_channels=self.num_channels_of_latent_state,
                activation=self.activation,
                norm_type=norm_type,
                res_type='basic',
                bias=False
            ) for _ in range(num_res_blocks)
        ])

        # Reward prediction residual blocks
        self.reward_resblocks = nn.ModuleList([
            ResBlock(
                in_channels=self.num_channels_of_latent_state,
                activation=self.activation,
                norm_type=norm_type,
                res_type='basic',
                bias=False
            ) for _ in range(num_res_blocks)
        ])

        # 1x1 convolution to adjust the number of reward head channels
        self.conv1x1_reward = nn.Conv2d(self.num_channels_of_latent_state, reward_head_channels, 1)

        # Choose normalization before LSTM
        if norm_type == 'BN':
            self.norm_before_lstm = nn.BatchNorm2d(reward_head_channels)
        elif norm_type == 'LN':
            self.norm_before_lstm = nn.LayerNorm(
                [reward_head_channels,
                 math.ceil(observation_shape[-2] / (16 if downsample else 1)),
                 math.ceil(observation_shape[-1] / (16 if downsample else 1))]
            )

        # Reward head MLP
        self.fc_reward_head = MLP(
            self.rnn_hidden_size,
            hidden_channels=fc_reward_layers[0],
            layer_num=len(fc_reward_layers) + 1,
            out_channels=output_support_size,
            activation=activation,
            norm_type=norm_type,
            output_activation=False,
            output_norm=False,
            last_linear_layer_init_zero=last_linear_layer_init_zero
        )

        self.latent_state_dim = self.flatten_output_size_for_reward_head

        self.gru = nn.GRU(input_size=self.latent_state_dim, hidden_size=self.rnn_hidden_size, num_layers=1, batch_first=True)

        # Compute output dimensions and shapes based on whether downsampling is used
        if downsample:
            ceil_size = math.ceil(observation_shape[1] / 16) * math.ceil(observation_shape[2] / 16)
            self.output_dim = self.num_channels_of_latent_state * ceil_size
            self.output_shape = (
                self.num_channels_of_latent_state,
                math.ceil(observation_shape[1] / 16),
                math.ceil(observation_shape[2] / 16)
            )
        else:
            self.output_dim = self.num_channels_of_latent_state * observation_shape[1] * observation_shape[2]
            self.output_shape = (self.num_channels_of_latent_state, observation_shape[1], observation_shape[2])

        # Flatten dimension of the latent state
        self.latent_state_flatten_dim = 64 * 8 * 8

        # Linear layer to adjust dimensions
        self.linear_common = nn.Linear(self.latent_state_flatten_dim, self.latent_state_dim, bias=False)

        # Dynamics head MLP
        self.fc_dynamics_head = MLP(
            self.rnn_hidden_size,
            hidden_channels=self.rnn_hidden_size,
            layer_num=2,
            out_channels=self.latent_state_flatten_dim,
            activation=activation,
            norm_type=norm_type,
            output_activation=True,
            output_norm=True,
            last_linear_layer_init_zero=False  # Important for convergence
        )

        self.res_connection_in_dynamics = res_connection_in_dynamics
        self.use_sim_norm = use_sim_norm

        # If using SimNorm
        if self.use_sim_norm:
            self.embedding_dim = embedding_dim
            self.last_linear = nn.Linear(self.latent_state_flatten_dim, self.embedding_dim, bias=False)
            init.kaiming_normal_(self.last_linear.weight, mode='fan_out', nonlinearity='relu')
            self.sim_norm = SimNorm(simnorm_dim=group_size)

    def forward(
        self,
        state_action_encoding: torch.Tensor,
        dynamics_hidden_state: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Forward pass for the Dynamics Network. Predict the next latent state,
        the next dynamics hidden state, and the reward based on the current state-action encoding.

        Args:
            state_action_encoding (torch.Tensor): State-action encoding, a concatenation of the latent state and action encoding.
            dynamics_hidden_state (Tuple[torch.Tensor, torch.Tensor]): Hidden state for the LSTM related to reward.

        Returns:
            next_latent_state (torch.Tensor): Next latent state.
            next_dynamics_hidden_state (Tuple[torch.Tensor, torch.Tensor]): Next hidden state for the LSTM.
            reward (torch.Tensor): Predicted reward.
        """
        # Extract latent state
        latent_state = state_action_encoding[:, :-self.action_encoding_dim, :, :]

        # Adjust channels and normalize
        x = self.conv(state_action_encoding)
        x = self.norm_common(x)
        x += latent_state  # Residual connection
        x = self.activation(x)

        # Pass through residual blocks
        for block in self.resblocks:
            x = block(x)

        # Reshape and transform
        x = self.linear_common(x.view(-1, self.latent_state_flatten_dim))
        x = self.activation(x).unsqueeze(1)

        # ==================== GRU backbone ==================
        # Pass through GRU
        gru_outputs, next_dynamics_hidden_state = self.gru(x, dynamics_hidden_state)

        # Predict reward
        reward = self.fc_reward_head(gru_outputs.squeeze(1))

        # Predict next latent state
        next_latent_state_encoding = self.fc_dynamics_head(gru_outputs.squeeze(1))

        # Residual connection
        if self.res_connection_in_dynamics:
            next_latent_state = next_latent_state_encoding.view(latent_state.shape) + latent_state
        else:
            next_latent_state = next_latent_state_encoding.view(latent_state.shape)

        # Apply SimNorm if used
        if self.use_sim_norm:
            next_latent_state = self.sim_norm(next_latent_state)

        return next_latent_state, next_dynamics_hidden_state, reward



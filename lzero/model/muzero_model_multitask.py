from typing import Optional, Tuple, Sequence, List

import math
import torch
import torch.nn as nn
from ding.torch_utils import MLP, ResBlock
from ding.utils import MODEL_REGISTRY, SequenceType
from numpy import ndarray

# The following imports are assumed to be from the same project directory.
# To maintain API consistency, their internal logic is not modified.
from .common import MZNetworkOutput, RepresentationNetwork, PredictionNetwork, FeatureAndGradientHook
from .utils import renormalize, get_params_mean, get_dynamic_mean, get_reward_mean


@MODEL_REGISTRY.register('MuZeroMTModel')
class MuZeroMTModel(nn.Module):
    """
    Overview:
        The Multi-Task MuZero model, which is a variant of the original MuZero model adapted for multi-task learning.
        This model features a shared representation network and dynamics network, but utilizes separate, task-specific
        prediction networks. This architecture allows the model to learn shared dynamics while specializing its
        policy and value predictions for each individual task.
    """
    # Default configuration for the model.
    # This structure is recommended over using cfg.get('key', default_value) inside the code.
    config = dict(
        observation_shape=(12, 96, 96),
        action_space_size=6,
        num_res_blocks=1,
        num_channels=64,
        reward_head_channels=16,
        value_head_channels=16,
        policy_head_channels=16,
        fc_reward_layers=[32],
        fc_value_layers=[32],
        fc_policy_layers=[32],
        reward_support_size=601,
        value_support_size=601,
        proj_hid=1024,
        proj_out=1024,
        pred_hid=512,
        pred_out=1024,
        self_supervised_learning_loss=False,
        categorical_distribution=True,
        activation=nn.ReLU(inplace=True),
        last_linear_layer_init_zero=True,
        state_norm=False,
        downsample=False,
        norm_type='BN',
        discrete_action_encoding_type='one_hot',
        analysis_sim_norm=False,
        task_num=1,
    )

    def __init__(
        self,
        observation_shape: SequenceType = (12, 96, 96),
        action_space_size: int = 6,
        num_res_blocks: int = 1,
        num_channels: int = 64,
        reward_head_channels: int = 16,
        value_head_channels: int = 16,
        policy_head_channels: int = 16,
        fc_reward_layers: List[int] = [32],
        fc_value_layers: List[int] = [32],
        fc_policy_layers: List[int] = [32],
        reward_support_size: int = 601,
        value_support_size: int = 601,
        proj_hid: int = 1024,
        proj_out: int = 1024,
        pred_hid: int = 512,
        pred_out: int = 1024,
        self_supervised_learning_loss: bool = False,
        categorical_distribution: bool = True,
        activation: Optional[nn.Module] = None,
        last_linear_layer_init_zero: bool = True,
        state_norm: bool = False,
        downsample: bool = False,
        norm_type: Optional[str] = 'BN',
        discrete_action_encoding_type: str = 'one_hot',
        analysis_sim_norm: bool = False,
        task_num: int = 1,
        *args,
        **kwargs
    ) -> None:
        """
        Overview:
            Constructor for the MuZeroMTModel.
        Arguments:
            - observation_shape (:obj:`SequenceType`): The shape of the input observation, e.g., (12, 96, 96).
            - action_space_size (:obj:`int`): The size of the action space, applicable for discrete action spaces.
            - num_res_blocks (:obj:`int`): The number of residual blocks in the representation, dynamics, and prediction networks.
            - num_channels (:obj:`int`): The number of channels in the latent state.
            - reward_head_channels (:obj:`int`): The number of channels in the reward head.
            - value_head_channels (:obj:`int`): The number of channels in the value head.
            - policy_head_channels (:obj:`int`): The number of channels in the policy head.
            - fc_reward_layers (:obj:`List[int]`): The hidden layer sizes of the reward MLP.
            - fc_value_layers (:obj:`List[int]`): The hidden layer sizes of the value MLP.
            - fc_policy_layers (:obj:`List[int]`): The hidden layer sizes of the policy MLP.
            - reward_support_size (:obj:`int`): The support size for categorical reward distribution.
            - value_support_size (:obj:`int`): The support size for categorical value distribution.
            - proj_hid (:obj:`int`): The hidden size of the projection network for SSL.
            - proj_out (:obj:`int`): The output size of the projection network for SSL.
            - pred_hid (:obj:`int`): The hidden size of the prediction head for SSL.
            - pred_out (:obj:`int`): The output size of the prediction head for SSL.
            - self_supervised_learning_loss (:obj:`bool`): Whether to use self-supervised learning loss.
            - categorical_distribution (:obj:`bool`): Whether to use categorical distribution for value and reward.
            - activation (:obj:`Optional[nn.Module]`): The activation function to use. Defaults to nn.ReLU(inplace=True).
            - last_linear_layer_init_zero (:obj:`bool`): Whether to initialize the last linear layer to zero.
            - state_norm (:obj:`bool`): Whether to apply re-normalization to the latent state.
            - downsample (:obj:`bool`): Whether to downsample the observation image.
            - norm_type (:obj:`Optional[str]`): The type of normalization to use, either 'BN' (BatchNorm) or 'LN' (LayerNorm).
            - discrete_action_encoding_type (:obj:`str`): The encoding type for discrete actions, 'one_hot' or 'not_one_hot'.
            - analysis_sim_norm (:obj:`bool`): A flag for analysis, enables hooks for SimNorm analysis.
            - task_num (:obj:`int`): The total number of tasks for the multi-task setup.
        """
        super(MuZeroMTModel, self).__init__()
        if activation is None:
            activation = nn.ReLU(inplace=True)

        # --- Store configuration ---
        self.action_space_size = action_space_size
        self.categorical_distribution = categorical_distribution
        self.self_supervised_learning_loss = self_supervised_learning_loss
        self.state_norm = state_norm
        self.downsample = downsample
        self.task_num = task_num
        self.discrete_action_encoding_type = discrete_action_encoding_type

        if self.categorical_distribution:
            self.reward_support_size = reward_support_size
            self.value_support_size = value_support_size
        else:
            self.reward_support_size = 1
            self.value_support_size = 1
            
        # --- Prepare observation shape and action encoding dimension ---
        if isinstance(observation_shape, int) or len(observation_shape) == 1:
            # For 1D vector observations (e.g., classic control), wrap them into a 2D image-like format [C, W, H]
            # to be compatible with the convolutional networks.
            observation_shape = (1, observation_shape[0], 1) if isinstance(observation_shape, tuple) else (1, observation_shape, 1)

        if self.discrete_action_encoding_type == 'one_hot':
            self.action_encoding_dim = self.action_space_size
        elif self.discrete_action_encoding_type == 'not_one_hot':
            self.action_encoding_dim = 1
        else:
            raise ValueError(f"Unsupported discrete_action_encoding_type: {self.discrete_action_encoding_type}")

        latent_size = self._get_latent_size(observation_shape, self.downsample)

        # --- Initialize Network Components ---

        # 1. Shared Representation Network
        self.representation_network = RepresentationNetwork(
            observation_shape=observation_shape,
            num_res_blocks=num_res_blocks,
            num_channels=num_channels,
            downsample=self.downsample,
            activation=activation,
            norm_type=norm_type
        )

        # 2. Shared Dynamics Network
        self.dynamics_network = DynamicsNetwork(
            observation_shape=observation_shape,
            action_encoding_dim=self.action_encoding_dim,
            num_res_blocks=num_res_blocks,
            num_channels=num_channels + self.action_encoding_dim,
            reward_head_channels=reward_head_channels,
            fc_reward_layers=fc_reward_layers,
            output_support_size=self.reward_support_size,
            flatten_output_size_for_reward_head=reward_head_channels * latent_size,
            downsample=self.downsample,
            last_linear_layer_init_zero=last_linear_layer_init_zero,
            activation=activation,
            norm_type=norm_type
        )

        # 3. Task-Specific Prediction Networks
        self.prediction_networks = nn.ModuleList([
            PredictionNetwork(
                observation_shape=observation_shape,
                action_space_size=self.action_space_size,
                num_res_blocks=num_res_blocks,
                num_channels=num_channels,
                value_head_channels=value_head_channels,
                policy_head_channels=policy_head_channels,
                fc_value_layers=fc_value_layers,
                fc_policy_layers=fc_policy_layers,
                output_support_size=self.value_support_size,
                flatten_output_size_for_value_head=value_head_channels * latent_size,
                flatten_output_size_for_policy_head=policy_head_channels * latent_size,
                downsample=self.downsample,
                last_linear_layer_init_zero=last_linear_layer_init_zero,
                activation=activation,
                norm_type=norm_type
            ) for _ in range(self.task_num)
        ])

        # 4. Optional Self-Supervised Learning (SSL) Components
        if self.self_supervised_learning_loss:
            self.projection_network = nn.Sequential(
                nn.Linear(num_channels * latent_size, proj_hid),
                nn.BatchNorm1d(proj_hid),
                activation,
                nn.Linear(proj_hid, proj_hid),
                nn.BatchNorm1d(proj_hid),
                activation,
                nn.Linear(proj_hid, proj_out),
                nn.BatchNorm1d(proj_out)
            )
            self.prediction_head = nn.Sequential(
                nn.Linear(proj_out, pred_hid),
                nn.BatchNorm1d(pred_hid),
                activation,
                nn.Linear(pred_hid, pred_out),
            )
            
        # 5. Optional Hook for Analysis
        if analysis_sim_norm:
            self.encoder_hook = FeatureAndGradientHook()
            self.encoder_hook.setup_hooks(self.representation_network)

    @staticmethod
    def _get_latent_size(observation_shape: SequenceType, downsample: bool) -> int:
        """
        Overview:
            Helper function to calculate the flattened size of the latent space based on observation shape and downsampling.
        Arguments:
            - observation_shape (:obj:`SequenceType`): The shape of the input observation.
            - downsample (:obj:`bool`): Whether downsampling is enabled.
        Returns:
            - int: The flattened size (height * width) of the latent space.
        """
        if downsample:
            # With downsampling, the spatial dimensions are reduced by a factor of 16 (2^4).
            return math.ceil(observation_shape[-2] / 16) * math.ceil(observation_shape[-1] / 16)
        else:
            return observation_shape[-2] * observation_shape[-1]

    def initial_inference(self, obs: torch.Tensor, task_id: int = 0) -> MZNetworkOutput:
        """
        Overview:
            Performs the initial inference from a raw observation. It encodes the observation into a latent state
            and then uses the task-specific prediction network to compute the policy and value.
        Arguments:
            - obs (:obj:`torch.Tensor`): The raw observation tensor.
            - task_id (:obj:`int`): The identifier for the current task, used to select the correct prediction network.
        Returns:
            - MZNetworkOutput: A dataclass containing the predicted value, reward (initially zero), policy logits, and latent state.
        Shapes:
            - obs (:obj:`torch.Tensor`): :math:`(B, C, H, W)`, where B is batch size.
            - task_id (:obj:`int`): Scalar.
            - Return.value: :math:`(B, value_support_size)`.
            - Return.reward: :math:`(B, reward_support_size)`.
            - Return.policy_logits: :math:`(B, action_space_size)`.
            - Return.latent_state: :math:`(B, num_channels, H', W')`.
        """
        batch_size = obs.size(0)
        latent_state = self.representation_network(obs)
        if self.state_norm:
            latent_state = renormalize(latent_state)
            
        # Select the prediction network based on the task ID.
        assert 0 <= task_id < self.task_num, f"Task ID {task_id} is out of range [0, {self.task_num-1}]"
        prediction_net = self.prediction_networks[task_id]
        policy_logits, value = prediction_net(latent_state)

        return MZNetworkOutput(
            value=value,
            reward=[0. for _ in range(batch_size)],  # Initial reward is always zero.
            policy_logits=policy_logits,
            latent_state=latent_state,
        )

    def recurrent_inference(self, latent_state: torch.Tensor, action: torch.Tensor, task_id: int = 0) -> MZNetworkOutput:
        """
        Overview:
            Performs recurrent inference from a latent state and an action. It uses the dynamics network to predict
            the next latent state and reward, and then uses the task-specific prediction network to compute the
            policy and value for the next state.
        Arguments:
            - latent_state (:obj:`torch.Tensor`): The current latent state.
            - action (:obj:`torch.Tensor`): The action taken in the current state.
            - task_id (:obj:`int`): The identifier for the current task.
        Returns:
            - MZNetworkOutput: A dataclass containing the predicted value, reward, policy logits, and the next latent state.
        Shapes:
            - latent_state (:obj:`torch.Tensor`): :math:`(B, num_channels, H', W')`.
            - action (:obj:`torch.Tensor`): :math:`(B, )`.
            - task_id (:obj:`int`): Scalar.
            - Return.value: :math:`(B, value_support_size)`.
            - Return.reward: :math:`(B, reward_support_size)`.
            - Return.policy_logits: :math:`(B, action_space_size)`.
            - Return.latent_state: :math:`(B, num_channels, H', W')`.
        """
        next_latent_state, reward = self._dynamics(latent_state, action)

        if self.state_norm:
            next_latent_state = renormalize(next_latent_state)
            
        # Select the prediction network based on the task ID.
        assert 0 <= task_id < self.task_num, f"Task ID {task_id} is out of range [0, {self.task_num-1}]"
        prediction_net = self.prediction_networks[task_id]
        policy_logits, value = prediction_net(next_latent_state)

        return MZNetworkOutput(value, reward, policy_logits, next_latent_state)

    def _dynamics(self, latent_state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Overview:
            Applies the dynamics function by concatenating the latent state with the encoded action and passing it
            through the dynamics network to predict the next latent state and reward.
        Arguments:
            - latent_state (:obj:`torch.Tensor`): The encoding latent state of the input state.
            - action (:obj:`torch.Tensor`): The action to rollout.
        Returns:
            - Tuple[torch.Tensor, torch.Tensor]: A tuple containing the predicted next latent state and reward.
        Shapes:
            - latent_state (:obj:`torch.Tensor`): :math:`(B, C, H', W')`.
            - action (:obj:`torch.Tensor`): :math:`(B, )`.
            - next_latent_state (:obj:`torch.Tensor`): :math:`(B, C, H', W')`.
            - reward (:obj:`torch.Tensor`): :math:`(B, reward_support_size)`.
        """
        # Encode the action and expand it to match the spatial dimensions of the latent state.
        if self.discrete_action_encoding_type == 'one_hot':
            # Convert action indices to one-hot vectors.
            action_one_hot = F.one_hot(action.long(), num_classes=self.action_space_size).float()
            # Reshape for broadcasting: (B, A) -> (B, A, 1, 1)
            action_encoding_tmp = action_one_hot.unsqueeze(-1).unsqueeze(-1)
            # Expand to (B, A, H', W')
            action_encoding = action_encoding_tmp.expand(
                latent_state.shape[0], self.action_space_size, latent_state.shape[2], latent_state.shape[3]
            )
        elif self.discrete_action_encoding_type == 'not_one_hot':
            # Encode action as a single channel, normalized by action space size.
            # Reshape for broadcasting: (B,) -> (B, 1, 1, 1)
            action_encoding_tmp = action.float().view(-1, 1, 1, 1)
            # Normalize and expand to (B, 1, H', W')
            action_encoding = action_encoding_tmp / self.action_space_size
            action_encoding = action_encoding.expand(
                latent_state.shape[0], 1, latent_state.shape[2], latent_state.shape[3]
            )

        # Concatenate latent state and action encoding along the channel dimension.
        state_action_encoding = torch.cat((latent_state, action_encoding), dim=1)

        # Predict next state and reward.
        next_latent_state, reward = self.dynamics_network(state_action_encoding)
        
        if self.state_norm:
            next_latent_state = renormalize(next_latent_state)
            
        return next_latent_state, reward
    
    def project(self, latent_state: torch.Tensor, with_grad: bool = True) -> torch.Tensor:
        """
        Overview:
            Projects the latent state into a different space for self-supervised learning (e.g., BYOL, SimSiam).
            This involves a projection network and an optional prediction head.
        Arguments:
            - latent_state (:obj:`torch.Tensor`): The latent state to project.
            - with_grad (:obj:`bool`): If False, detach the output of the projection network to stop gradients.
                                      This is typically used for the target network in SSL.
        Returns:
            - torch.Tensor: The projected (and possibly predicted) representation.
        """
        if not self.self_supervised_learning_loss:
            raise NotImplementedError("The 'project' method requires 'self_supervised_learning_loss' to be enabled.")
        
        # Flatten the latent state from (B, C, H, W) to (B, C*H*W).
        latent_state = latent_state.reshape(latent_state.shape[0], -1)
        
        proj = self.projection_network(latent_state)
        
        if with_grad:
            # Return the output of the prediction head, with gradients flowing.
            return self.prediction_head(proj)
        else:
            # Return the output of the projection network, detached from the graph.
            return proj.detach()

    def get_params_mean(self) -> float:
        """
        Overview:
            Computes the mean of all model parameters. Useful for debugging and monitoring training.
        Returns:
            - float: The mean value of all parameters.
        """
        return get_params_mean(self)


class DynamicsNetwork(nn.Module):
    """
    Overview:
        The dynamics network of the MuZero model. It takes a state-action encoding as input and predicts
        the next latent state and the reward for the transition. This network is shared across all tasks
        in the multi-task setup.
    """

    def __init__(
        self,
        observation_shape: SequenceType,
        action_encoding_dim: int = 2,
        num_res_blocks: int = 1,
        num_channels: int = 64,
        reward_head_channels: int = 64,
        fc_reward_layers: List[int] = [32],
        output_support_size: int = 601,
        flatten_output_size_for_reward_head: int = 64,
        downsample: bool = False,
        last_linear_layer_init_zero: bool = True,
        activation: Optional[nn.Module] = None,
        norm_type: Optional[str] = 'BN',
    ) -> None:
        """
        Overview:
            Constructor for the DynamicsNetwork.
        Arguments:
            - observation_shape (:obj:`SequenceType`): The shape of the original input observation.
            - action_encoding_dim (:obj:`int`): The dimension of the encoded action.
            - num_res_blocks (:obj:`int`): The number of residual blocks.
            - num_channels (:obj:`int`): The number of channels in the input (latent_state + action_encoding).
            - reward_head_channels (:obj:`int`): The number of channels for the reward head's convolutional layer.
            - fc_reward_layers (:obj:`List[int]`): The hidden layer sizes of the reward MLP.
            - output_support_size (:obj:`int`): The support size for the categorical reward distribution.
            - flatten_output_size_for_reward_head (:obj:`int`): The flattened input size for the reward MLP.
            - downsample (:obj:`bool`): Whether downsampling is used, affecting LayerNorm shapes.
            - last_linear_layer_init_zero (:obj:`bool`): Whether to initialize the last linear layer to zero.
            - activation (:obj:`Optional[nn.Module]`): The activation function. Defaults to nn.ReLU(inplace=True).
            - norm_type (:obj:`Optional[str]`): The type of normalization, 'BN' or 'LN'.
        """
        super().__init__()
        if activation is None:
            activation = nn.ReLU(inplace=True)
            
        assert norm_type in ['BN', 'LN'], f"norm_type must be 'BN' or 'LN', but got {norm_type}"
        # The input channels to the first conv layer is num_channels, which includes the original latent channels
        # and the action encoding channels. The output should be the number of channels for the latent state.
        latent_channels = num_channels - action_encoding_dim
        assert latent_channels > 0, f"num_channels ({num_channels}) must be greater than action_encoding_dim ({action_encoding_dim})"

        self.action_encoding_dim = action_encoding_dim
        self.activation = activation

        # Convolutional layer to process the combined state-action encoding.
        self.conv = nn.Conv2d(num_channels, latent_channels, kernel_size=3, stride=1, padding=1, bias=False)

        # Normalization layer for the main path.
        if norm_type == 'BN':
            self.norm_common = nn.BatchNorm2d(latent_channels)
        elif norm_type == 'LN':
            if downsample:
                ln_shape = [latent_channels, math.ceil(observation_shape[-2] / 16), math.ceil(observation_shape[-1] / 16)]
            else:
                ln_shape = [latent_channels, observation_shape[-2], observation_shape[-1]]
            self.norm_common = nn.LayerNorm(ln_shape)
            
        # A series of residual blocks to deepen the network.
        self.resblocks = nn.ModuleList(
            [ResBlock(in_channels=latent_channels, activation=activation, norm_type='BN', res_type='basic', bias=False)
             for _ in range(num_res_blocks)]
        )

        # --- Reward Head ---
        # 1x1 convolution to create an input for the reward MLP.
        self.conv1x1_reward = nn.Conv2d(latent_channels, reward_head_channels, 1)
        
        # Normalization for the reward head.
        if norm_type == 'BN':
            self.norm_reward = nn.BatchNorm2d(reward_head_channels)
        elif norm_type == 'LN':
            if downsample:
                ln_shape_reward = [reward_head_channels, math.ceil(observation_shape[-2] / 16), math.ceil(observation_shape[-1] / 16)]
            else:
                ln_shape_reward = [reward_head_channels, observation_shape[-2], observation_shape[-1]]
            self.norm_reward = nn.LayerNorm(ln_shape_reward)

        # MLP to predict the reward value from the processed features.
        self.fc_reward_head = MLP(
            in_channels=flatten_output_size_for_reward_head,
            hidden_channels=fc_reward_layers[0],
            out_channels=output_support_size,
            layer_num=len(fc_reward_layers) + 1,
            activation=activation,
            norm_type=norm_type,
            output_activation=False,
            output_norm=False,
            last_linear_layer_init_zero=last_linear_layer_init_zero
        )

    def forward(self, state_action_encoding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Overview:
            Forward pass for the dynamics network.
        Arguments:
            - state_action_encoding (:obj:`torch.Tensor`): The concatenated latent state and action encoding.
        Returns:
            - Tuple[torch.Tensor, torch.Tensor]: A tuple containing the next latent state and the predicted reward.
        Shapes:
            - state_action_encoding (:obj:`torch.Tensor`): :math:`(B, C_latent + C_action, H', W')`.
            - next_latent_state (:obj:`torch.Tensor`): :math:`(B, C_latent, H', W')`.
            - reward (:obj:`torch.Tensor`): :math:`(B, output_support_size)`.
        """
        # The original latent state is part of the input, used for the residual connection.
        state_encoding = state_action_encoding[:, : -self.action_encoding_dim, :, :]
        
        # Main path for predicting the next latent state.
        x = self.conv(state_action_encoding)
        x = self.norm_common(x)
        
        # Add residual connection from the original latent state.
        x += state_encoding
        x = self.activation(x)

        for block in self.resblocks:
            x = block(x)
        next_latent_state = x

        # --- Reward Prediction Path ---
        # Process the next latent state to predict the reward.
        reward_x = self.conv1x1_reward(next_latent_state)
        reward_x = self.norm_reward(reward_x)
        reward_x = self.activation(reward_x)
        # Flatten the features before passing to the MLP.
        reward_x = reward_x.view(reward_x.shape[0], -1)
        reward = self.fc_reward_head(reward_x)

        return next_latent_state, reward

    def get_dynamic_mean(self) -> float:
        """
        Overview:
            Computes the mean of parameters in the dynamics-related layers (conv and resblocks).
        Returns:
            - float: The mean value of dynamics parameters.
        """
        return get_dynamic_mean(self)

    def get_reward_mean(self) -> Tuple[ndarray, float]:
        """
        Overview:
            Computes the mean of parameters and the last layer bias in the reward head.
        Returns:
            - Tuple[ndarray, float]: A tuple containing the mean of the last layer's weights and its bias.
        """
        return get_reward_mean(self)
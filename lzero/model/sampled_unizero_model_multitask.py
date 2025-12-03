from typing import Optional, List, Sequence

import torch
import torch.nn as nn
from ding.torch_utils import MLP
from ding.utils import MODEL_REGISTRY
from easydict import EasyDict

from .common import MZNetworkOutput, RepresentationNetworkUniZero, LatentDecoder, \
    FeatureAndGradientHook, SimNorm
from .unizero_world_models.tokenizer import Tokenizer
from .unizero_world_models.world_model_multitask import WorldModelMT

class RepresentationNetworkMLPMT(nn.Module):
    """
    Overview:
        A multi-task representation network that encodes vector observations into a latent state
        using a Multi-Layer Perceptron (MLP). It supports task-specific encoders and an optional
        shared projection layer to map representations into a common embedding space.
    """

    def __init__(
            self,
            observation_shape_list: List[int],
            hidden_channels: int = 64,
            layer_num: int = 2,
            activation: nn.Module = nn.GELU(approximate='tanh'),
            norm_type: Optional[str] = 'BN',
            embedding_dim: int = 256,
            group_size: int = 8,
            use_shared_projection: bool = False,
            shared_projection_dim: Optional[int] = None,
            final_norm_option_in_encoder: str = 'LayerNorm',  # TODO: Further investigate norm options
    ) -> None:
        """
        Arguments:
            - observation_shape_list (:obj:`List[int]`): A list of observation feature dimensions, one for each task.
            - hidden_channels (:obj:`int`): The number of hidden channels in the task-specific MLPs.
            - layer_num (:obj:`int`): The number of layers in each MLP.
            - activation (:obj:`nn.Module`): The activation function to use in the MLPs. Defaults to nn.GELU(approximate='tanh').
            - norm_type (:obj:`str`): The type of normalization to use within the MLPs. Defaults to 'BN'.
            - embedding_dim (:obj:`int`): The dimension of the final output embedding.
            - group_size (:obj:`int`): The group size for SimNorm if it is used.
            - use_shared_projection (:obj:`bool`): Whether to use a shared projection layer after task-specific encoding. Defaults to False.
            - shared_projection_dim (:obj:`Optional[int]`): The dimension of the shared projection layer. If None, it defaults to `hidden_channels`.
            - final_norm_option_in_encoder (:obj:`str`): The final normalization layer type ('LayerNorm' or 'SimNorm'). Defaults to 'LayerNorm'.
        """
        super().__init__()
        self.env_num = len(observation_shape_list)
        self.use_shared_projection = use_shared_projection
        self.hidden_channels = hidden_channels
        self.shared_projection_dim = shared_projection_dim or hidden_channels
        self.embedding_dim = embedding_dim
        self.final_norm_option_in_encoder = final_norm_option_in_encoder

        # Task-specific representation networks
        self.fc_representation = nn.ModuleList([
            MLP(
                in_channels=obs_shape,
                hidden_channels=hidden_channels,
                out_channels=hidden_channels,
                layer_num=layer_num,
                activation=activation,
                norm_type=norm_type,
                # No activation or norm in the last layer is important for convergence.
                output_activation=False,
                output_norm=False,
                # Initializing the last linear layer to zero can be beneficial for convergence speed.
                last_linear_layer_init_zero=True,
            )
            for obs_shape in observation_shape_list
        ])

        # Final normalization layer before projection
        if self.final_norm_option_in_encoder == 'LayerNorm':
            self.final_norm = nn.LayerNorm(self.embedding_dim, eps=1e-5)
        elif self.final_norm_option_in_encoder == 'SimNorm':
            self.final_norm = SimNorm(simnorm_dim=group_size)
        else:
            raise ValueError(f"Unsupported final_norm_option_in_encoder: {self.final_norm_option_in_encoder}")

        # Optional shared projection layer
        if self.use_shared_projection:
            self.shared_projection = nn.Linear(hidden_channels, self.shared_projection_dim)
            # Using SimNorm for the shared space projection
            self.projection_norm = SimNorm(simnorm_dim=group_size)

    def forward(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        """
        Shapes:
            - x (:obj:`torch.Tensor`): The input tensor of shape :math:`(B, N)`, where B is the batch size and N is the length of the vector observation.
            - task_id (:obj:`int`): The identifier for the current task, used to select the appropriate encoder.
            - output (:obj:`torch.Tensor`): The output latent state. Its shape is :math:`(B, embedding_dim)` if shared projection is not used, otherwise :math:`(B, shared_projection_dim)`.
        """
        # Encode observation using the task-specific MLP
        x = self.fc_representation[task_id](x)
        # Apply final normalization
        x = self.final_norm(x)

        # Apply the shared projection layer if enabled
        if self.use_shared_projection:
            x = self.shared_projection(x)
            x = self.projection_norm(x)
        return x


@MODEL_REGISTRY.register('SampledUniZeroMTModel')
class SampledUniZeroMTModel(nn.Module):
    """
    Overview:
        The main model for Sampled UniZero in a multi-task setting. It integrates a representation
        network, a tokenizer, and a world model to perform initial and recurrent inference,
        which are essential for MuZero-style planning algorithms. The model is designed to handle
        both vector and image-based observations across multiple tasks.
    """

    def __init__(
            self,
            observation_shape_list: List[Sequence],
            action_space_size_list: List[int],
            num_res_blocks: int = 1,
            num_channels: int = 64,
            activation: nn.Module = nn.GELU(approximate='tanh'),
            downsample: bool = True,
            norm_type: Optional[str] = 'LN',
            world_model_cfg: EasyDict = None,
            *args,
            **kwargs
    ):
        """
        Arguments:
            - observation_shape_list (:obj:`List[Sequence]`): A list of observation space shapes for each task (e.g., `[C, W, H]` for images or `[D]` for vectors).
            - action_space_size_list (:obj:`List[int]`): A list of action space sizes for each task.
            - num_res_blocks (:obj:`int`): The number of residual blocks in the image representation network.
            - num_channels (:obj:`int`): The number of channels in the hidden states of the image representation network.
            - activation (:obj:`nn.Module`): The activation function used throughout the network.
            - downsample (:obj:`bool`): Whether to downsample observations in the image representation network.
            - norm_type (:obj:`str`): The type of normalization to use in networks. Defaults to 'LN'.
            - world_model_cfg (:obj:`EasyDict`): A single configuration object for the world model, shared across all tasks.
        """
        super(SampledUniZeroMTModel, self).__init__()
        self.task_num = len(observation_shape_list)
        self.activation = activation
        self.downsample = downsample

        # Determine the embedding dimension for observations and actions
        if world_model_cfg.task_embed_option == "concat_task_embed":
            obs_act_embed_dim = world_model_cfg.embed_dim - world_model_cfg.task_embed_dim if hasattr(world_model_cfg, "task_embed_dim") else 96
        else:
            obs_act_embed_dim = world_model_cfg.embed_dim

        world_model_cfg.norm_type = norm_type
        assert world_model_cfg.max_tokens == 2 * world_model_cfg.max_blocks, \
            'max_tokens should be 2 * max_blocks, as each timestep consists of an observation and an action token.'

        # Initialize networks based on observation type
        if world_model_cfg.obs_type == 'vector':
            # A single representation network capable of handling multiple tasks via task_id
            self.representation_network = RepresentationNetworkMLPMT(
                observation_shape_list=observation_shape_list,
                hidden_channels=obs_act_embed_dim,
                layer_num=2,
                activation=self.activation,
                norm_type=norm_type,
                embedding_dim=obs_act_embed_dim,
                group_size=world_model_cfg.group_size,
                use_shared_projection=world_model_cfg.use_shared_projection,
                final_norm_option_in_encoder=world_model_cfg.final_norm_option_in_encoder,
            )
            self.tokenizer = Tokenizer(encoder=self.representation_network, decoder_network=None, with_lpips=False)
            self.world_model = WorldModelMT(config=world_model_cfg, tokenizer=self.tokenizer)

        elif world_model_cfg.obs_type == 'image':
            self.representation_network = nn.ModuleList()
            # TODO: Currently uses a single shared encoder for all image-based tasks.
            # This can be extended to support multiple independent encoders if needed.
            for _ in range(1):
                self.representation_network.append(RepresentationNetworkUniZero(
                    observation_shape_list[0],  # Assuming shared encoder uses the shape of the first task
                    num_res_blocks,
                    num_channels,
                    self.downsample,
                    activation=self.activation,
                    norm_type=norm_type,
                    embedding_dim=obs_act_embed_dim,
                    group_size=world_model_cfg.group_size,
                    final_norm_option_in_encoder=world_model_cfg.final_norm_option_in_encoder,
                ))
            # TODO: The world model and tokenizer for the 'image' case should be initialized here.
            # self.tokenizer = Tokenizer(...)
            # self.world_model = WorldModelMT(...)

        # Print model parameter counts for verification
        print(f'{sum(p.numel() for p in self.world_model.parameters())} parameters in agent.world_model')
        print('==' * 20)
        print(f'{sum(p.numel() for p in self.world_model.transformer.parameters())} parameters in agent.world_model.transformer')
        if hasattr(self.tokenizer, 'encoder') and self.tokenizer.encoder is not None:
             print(f'{sum(p.numel() for p in self.tokenizer.encoder.parameters())} parameters in agent.tokenizer.encoder')
        print('==' * 20)

    def initial_inference(self, obs_batch: torch.Tensor, action_batch: Optional[torch.Tensor] = None, current_obs_batch: Optional[torch.Tensor] = None, task_id: Optional[int] = None) -> MZNetworkOutput:
        """
        Overview:
            Performs the initial inference step of the UniZero model. It takes an observation
            and produces a latent state, a value prediction, and an initial policy.
        Arguments:
            - obs_batch (:obj:`torch.Tensor`): The initial batch of observations.
            - action_batch (:obj:`Optional[torch.Tensor]`): An optional batch of actions.
            - current_obs_batch (:obj:`Optional[torch.Tensor]`): An optional batch of current observations.
            - task_id (:obj:`Optional[int]`): The identifier for the current task.
        Returns (MZNetworkOutput):
            An object containing the predicted value, initial reward (zero), policy logits, and latent state.
        Shapes:
            - obs_batch (:obj:`torch.Tensor`): :math:`(B, ...)` where B is the batch size.
            - value (:obj:`torch.Tensor`): :math:`(B, value_support_size)`.
            - reward (:obj:`torch.Tensor`): :math:`(B, reward_support_size)`.
            - policy_logits (:obj:`torch.Tensor`): :math:`(B, action_dim)`.
            - latent_state (:obj:`torch.Tensor`): :math:`(B, embedding_dim)`.
        """
        batch_size = obs_batch.size(0)
        obs_act_dict = {'obs': obs_batch, 'action': action_batch, 'current_obs': current_obs_batch}
        _, obs_token, logits_rewards, logits_policy, logits_value = self.world_model.forward_initial_inference(obs_act_dict, task_id=task_id)

        latent_state = obs_token
        policy_logits = logits_policy.squeeze(1)
        value = logits_value.squeeze(1)

        return MZNetworkOutput(
            value=value,
            reward=[0. for _ in range(batch_size)],  # Initial reward is always zero
            policy_logits=policy_logits,
            latent_state=latent_state,
        )

    def recurrent_inference(self, state_action_history: torch.Tensor, simulation_index: int = 0, search_depth: List[int] = [], task_id: int = 0) -> MZNetworkOutput:
        """
        Overview:
            Performs the recurrent inference step (the dynamics function). Given a history of
            latent states and actions, it predicts the next latent state, reward, value, and policy.
        Arguments:
            - state_action_history (:obj:`torch.Tensor`): A history of states and actions.
            - simulation_index (:obj:`int`): The index of the current simulation step in MCTS.
            - search_depth (:obj:`List[int]`): The indices of latent states in the current search path.
            - task_id (:obj:`int`): The identifier for the current task.
        Returns (MZNetworkOutput):
            An object containing the predicted value, reward, policy logits, and the next latent state.
        Shapes:
            - state_action_history (:obj:`torch.Tensor`): :math:`(B, L, D)`, where L is sequence length.
            - value (:obj:`torch.Tensor`): :math:`(B, value_support_size)`.
            - reward (:obj:`torch.Tensor`): :math:`(B, reward_support_size)`.
            - policy_logits (:obj:`torch.Tensor`): :math:`(B, action_dim)`.
            - next_latent_state (:obj:`torch.Tensor`): :math:`(B, embedding_dim)`.
        """
        _, logits_observations, logits_rewards, logits_policy, logits_value = self.world_model.forward_recurrent_inference(
            state_action_history, simulation_index, search_depth, task_id=task_id)

        next_latent_state = logits_observations
        reward = logits_rewards.squeeze(1)
        policy_logits = logits_policy.squeeze(1)
        value = logits_value.squeeze(1)

        return MZNetworkOutput(value, reward, policy_logits, next_latent_state)
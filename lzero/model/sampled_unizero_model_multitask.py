from typing import Optional, List, Sequence

import torch
import torch.nn as nn
from ding.torch_utils import MLP
from ding.utils import MODEL_REGISTRY, get_rank
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
            # FIXED: Tokenizer parameter name is 'decoder', not 'decoder_network'
            self.tokenizer = Tokenizer(encoder=self.representation_network, decoder=None, with_lpips=False)
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

        # --- Log parameter counts for analysis ---
        self._log_model_parameters(world_model_cfg.obs_type)

    def _log_model_parameters(self, obs_type: str) -> None:
        """
        Overview:
            Logs detailed parameter counts for all model components with a comprehensive breakdown.
            Includes encoder, transformer, prediction heads, and other components.
            This version is adapted for multi-task models.
        Arguments:
            - obs_type (:obj:`str`): The type of observation ('vector' or 'image').
        """
        # Only print from rank 0 to avoid duplicate logs in DDP
        if get_rank() != 0:
            return

        print('=' * 80)
        print('MODEL PARAMETER STATISTICS (Multi-Task)'.center(80))
        print('=' * 80)

        # --- Total Model Parameters ---
        total_params = sum(p.numel() for p in self.parameters())
        total_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'\n{"TOTAL MODEL":<40} {total_params:>15,} parameters')
        print(f'{"  └─ Trainable":<40} {total_trainable:>15,} parameters')
        print(f'{"  └─ Frozen":<40} {total_params - total_trainable:>15,} parameters')
        print(f'{"  └─ Number of Tasks":<40} {self.task_num:>15,}')

        # --- World Model Components ---
        print(f'\n{"-" * 80}')
        print(f'{"WORLD MODEL BREAKDOWN":<40}')
        print(f'{"-" * 80}')

        wm_params = sum(p.numel() for p in self.world_model.parameters())
        wm_trainable = sum(p.numel() for p in self.world_model.parameters() if p.requires_grad)
        print(f'{"World Model Total":<40} {wm_params:>15,} parameters')
        print(f'{"  └─ Trainable":<40} {wm_trainable:>15,} parameters ({100*wm_trainable/wm_params:.1f}%)')

        # --- Encoder ---
        if hasattr(self.tokenizer, 'encoder') and self.tokenizer.encoder is not None:
            encoder_params = sum(p.numel() for p in self.tokenizer.encoder.parameters())
            encoder_trainable = sum(p.numel() for p in self.tokenizer.encoder.parameters() if p.requires_grad)
            print(f'\n{"1. ENCODER (Tokenizer)":<40} {encoder_params:>15,} parameters')
            print(f'{"  └─ Trainable":<40} {encoder_trainable:>15,} parameters ({100*encoder_trainable/encoder_params:.1f}%)')

            # For multi-task encoder, show per-task breakdown
            if isinstance(self.tokenizer.encoder, nn.ModuleList):
                print(f'{"  └─ Multi-Task Encoders":<40} {len(self.tokenizer.encoder):>15,} tasks')
                for i, enc in enumerate(self.tokenizer.encoder):
                    task_params = sum(p.numel() for p in enc.parameters())
                    print(f'{"    ├─ Task " + str(i):<38} {task_params:>15,} parameters')
            elif hasattr(self.tokenizer.encoder, 'fc_representation'):
                # RepresentationNetworkMLPMT case
                print(f'{"  └─ Task-Specific Encoders":<40} {len(self.tokenizer.encoder.fc_representation):>15,} tasks')
                for i, enc in enumerate(self.tokenizer.encoder.fc_representation):
                    task_params = sum(p.numel() for p in enc.parameters())
                    print(f'{"    ├─ Task " + str(i):<38} {task_params:>15,} parameters')

                # Show shared projection if exists
                if hasattr(self.tokenizer.encoder, 'use_shared_projection') and self.tokenizer.encoder.use_shared_projection:
                    shared_proj_params = sum(p.numel() for p in self.tokenizer.encoder.shared_projection.parameters())
                    print(f'{"  └─ Shared Projection Layer":<40} {shared_proj_params:>15,} parameters')

        # --- Transformer Backbone ---
        transformer_params = sum(p.numel() for p in self.world_model.transformer.parameters())
        transformer_trainable = sum(p.numel() for p in self.world_model.transformer.parameters() if p.requires_grad)
        print(f'\n{"2. TRANSFORMER BACKBONE":<40} {transformer_params:>15,} parameters')
        print(f'{"  └─ Trainable":<40} {transformer_trainable:>15,} parameters ({100*transformer_trainable/transformer_params:.1f}%)')

        # --- Prediction Heads (Detailed Breakdown) ---
        print(f'\n{"3. PREDICTION HEADS":<40}')

        # Access head_dict from world_model
        if hasattr(self.world_model, 'head_dict'):
            head_dict = self.world_model.head_dict

            # Calculate total heads parameters
            total_heads_params = sum(p.numel() for module in head_dict.values() for p in module.parameters())
            total_heads_trainable = sum(p.numel() for module in head_dict.values() for p in module.parameters() if p.requires_grad)
            print(f'{"  Total (All Heads)":<40} {total_heads_params:>15,} parameters')
            print(f'{"  └─ Trainable":<40} {total_heads_trainable:>15,} parameters ({100*total_heads_trainable/total_heads_params:.1f}%)')

            # Breakdown by head type
            head_names_map = {
                'head_policy_multi_task': 'Policy Head (Multi-Task)',
                'head_value_multi_task': 'Value Head (Multi-Task)',
                'head_rewards_multi_task': 'Reward Head (Multi-Task)',
                'head_observations_multi_task': 'Next Latent Head (Multi-Task)'
            }

            print(f'\n{"  Breakdown by Head Type:":<40}')
            for head_key, head_name in head_names_map.items():
                if head_key in head_dict:
                    head_module = head_dict[head_key]
                    head_params = sum(p.numel() for p in head_module.parameters())
                    head_trainable = sum(p.numel() for p in head_module.parameters() if p.requires_grad)

                    # Count number of task-specific heads (for ModuleList)
                    if isinstance(head_module, nn.ModuleList):
                        num_heads = len(head_module)
                        params_per_head = head_params // num_heads if num_heads > 0 else 0
                        print(f'{"    ├─ " + head_name:<38} {head_params:>15,} parameters')
                        print(f'{"      └─ " + f"{num_heads} task-specific heads":<38} {params_per_head:>15,} params/head')
                    else:
                        print(f'{"    ├─ " + head_name:<38} {head_params:>15,} parameters')
                        print(f'{"      └─ Shared across tasks":<38}')

        # --- Positional & Task Embeddings ---
        print(f'\n{"4. EMBEDDINGS":<40}')

        if hasattr(self.world_model, 'pos_emb'):
            pos_emb_params = sum(p.numel() for p in self.world_model.pos_emb.parameters())
            pos_emb_trainable = sum(p.numel() for p in self.world_model.pos_emb.parameters() if p.requires_grad)
            print(f'{"  ├─ Positional Embedding":<40} {pos_emb_params:>15,} parameters')
            if pos_emb_trainable == 0:
                print(f'{"    └─ (Frozen)":<40}')

        if hasattr(self.world_model, 'task_emb') and self.world_model.task_emb is not None:
            task_emb_params = sum(p.numel() for p in self.world_model.task_emb.parameters())
            task_emb_trainable = sum(p.numel() for p in self.world_model.task_emb.parameters() if p.requires_grad)
            print(f'{"  ├─ Task Embedding":<40} {task_emb_params:>15,} parameters')
            print(f'{"    └─ Trainable":<40} {task_emb_trainable:>15,} parameters')
            print(f'{"    └─ Num Embeddings":<40} {self.task_num:>15,} tasks')

        if hasattr(self.world_model, 'act_embedding_table'):
            act_emb_params = sum(p.numel() for p in self.world_model.act_embedding_table.parameters())
            act_emb_trainable = sum(p.numel() for p in self.world_model.act_embedding_table.parameters() if p.requires_grad)
            print(f'{"  └─ Action Embedding":<40} {act_emb_params:>15,} parameters')
            print(f'{"    └─ Trainable":<40} {act_emb_trainable:>15,} parameters')

        # --- Decoder (if applicable) ---
        if hasattr(self.tokenizer, 'decoder_network') and self.tokenizer.decoder_network is not None:
            print(f'\n{"5. DECODER":<40}')
            decoder_params = sum(p.numel() for p in self.tokenizer.decoder_network.parameters())
            decoder_trainable = sum(p.numel() for p in self.tokenizer.decoder_network.parameters() if p.requires_grad)
            print(f'{"  Decoder Network":<40} {decoder_params:>15,} parameters')
            print(f'{"  └─ Trainable":<40} {decoder_trainable:>15,} parameters')

            if hasattr(self.tokenizer, 'lpips') and self.tokenizer.lpips is not None:
                lpips_params = sum(p.numel() for p in self.tokenizer.lpips.parameters())
                print(f'{"  LPIPS Loss Network":<40} {lpips_params:>15,} parameters')

                # Calculate world model params excluding decoder and LPIPS
                params_without_decoder = wm_params - decoder_params - lpips_params
                print(f'\n{"  World Model (exc. Decoder & LPIPS)":<40} {params_without_decoder:>15,} parameters')

        # --- Summary Table ---
        print(f'\n{"=" * 80}')
        print(f'{"SUMMARY":<40}')
        print(f'{"=" * 80}')
        print(f'{"Component":<30} {"Total Params":>15} {"Trainable":>15} {"% of Total":>15}')
        print(f'{"-" * 80}')

        components = []

        if hasattr(self.tokenizer, 'encoder') and self.tokenizer.encoder is not None:
            encoder_params = sum(p.numel() for p in self.tokenizer.encoder.parameters())
            encoder_trainable = sum(p.numel() for p in self.tokenizer.encoder.parameters() if p.requires_grad)
            components.append(("Encoder", encoder_params, encoder_trainable))

        components.append(("Transformer", transformer_params, transformer_trainable))

        if hasattr(self.world_model, 'head_dict'):
            components.append(("Prediction Heads", total_heads_params, total_heads_trainable))

        for name, total, trainable in components:
            pct = 100 * total / total_params if total_params > 0 else 0
            print(f'{name:<30} {total:>15,} {trainable:>15,} {pct:>14.1f}%')

        print(f'{"=" * 80}')
        print(f'{"TOTAL":<30} {total_params:>15,} {total_trainable:>15,} {"100.0%":>15}')
        print(f'{"=" * 80}\n')

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
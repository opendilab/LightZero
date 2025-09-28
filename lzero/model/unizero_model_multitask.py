from typing import Optional, Sequence, Dict, Any, List

import torch
import torch.nn as nn
from ding.utils import MODEL_REGISTRY, SequenceType
from easydict import EasyDict

from .common import MZNetworkOutput, RepresentationNetworkUniZero, RepresentationNetworkMLP, LatentDecoder, \
    VectorDecoderForMemoryEnv, LatentEncoderForMemoryEnv, LatentDecoderForMemoryEnv, FeatureAndGradientHook
from .unizero_world_models.tokenizer import Tokenizer
from .unizero_world_models.world_model_multitask import WorldModelMT
from .vit import ViT, ViTConfig


@MODEL_REGISTRY.register('UniZeroMTModel')
class UniZeroMTModel(nn.Module):
    """
    Overview:
        The main model for UniZero, a multi-task agent based on a scalable latent world model.
        This class orchestrates the representation network, world model, and prediction heads.
        It provides two primary interfaces:
        - `initial_inference`: Encodes an observation to produce an initial latent state and predictions (value, policy).
        - `recurrent_inference`: Simulates dynamics by taking a history of latent states and actions to predict the next
          latent state, reward, value, and policy.
    """

    def __init__(
            self,
            observation_shape: SequenceType = (4, 64, 64),
            action_space_size: int = 6,
            num_res_blocks: int = 1,
            num_channels: int = 64,
            activation: nn.Module = nn.GELU(approximate='tanh'),
            downsample: bool = True,
            norm_type: str = 'BN',
            world_model_cfg: EasyDict = None,
            task_num: int = 1,
            *args: Any,
            **kwargs: Any
    ) -> None:
        """
        Overview:
            Initializes the UniZeroMTModel, setting up the representation network, tokenizer, and world model
            based on the provided configuration.
        Arguments:
            - observation_shape (:obj:`SequenceType`): The shape of the input observation, e.g., (C, H, W).
            - action_space_size (:obj:`int`): The size of the discrete action space.
            - num_res_blocks (:obj:`int`): The number of residual blocks in the ResNet-based representation network.
            - num_channels (:obj:`int`): The number of channels in the ResNet-based representation network.
            - activation (:obj:`nn.Module`): The activation function to use throughout the network.
            - downsample (:obj:`bool`): Whether to downsample the observation in the representation network.
            - norm_type (:obj:`str`): The type of normalization to use, e.g., 'BN' for BatchNorm.
            - world_model_cfg (:obj:`EasyDict`): Configuration for the world model and its components.
            - task_num (:obj:`int`): The number of tasks for multi-task learning.
        """
        super().__init__()
        print(f'========== Initializing UniZeroMTModel (num_res_blocks: {num_res_blocks}, num_channels: {num_channels}) ==========')

        # --- Basic attribute setup ---
        self.task_num = task_num
        self.activation = activation
        self.downsample = downsample
        world_model_cfg.norm_type = norm_type

        # NOTE: The action_space_size passed as an argument is immediately overridden.
        # This might be intentional for specific experiments but is not a general practice.
        self.action_space_size = 18

        assert world_model_cfg.max_tokens == 2 * world_model_cfg.max_blocks, \
            "max_tokens should be 2 * max_blocks, as each timestep consists of an observation and an action token."

        # --- Determine embedding dimensions ---
        if world_model_cfg.task_embed_option == "concat_task_embed":
            task_embed_dim = world_model_cfg.get("task_embed_dim", 32)  # Default task_embed_dim to 32 if not specified
            obs_act_embed_dim = world_model_cfg.embed_dim - task_embed_dim
        else:
            obs_act_embed_dim = world_model_cfg.embed_dim

        # --- Initialize model components based on observation type ---
        obs_type = world_model_cfg.obs_type
        if obs_type == 'vector':
            self._init_vector_components(world_model_cfg, obs_act_embed_dim)
        elif obs_type == 'image':
            self._init_image_components(world_model_cfg, observation_shape, num_res_blocks, num_channels, obs_act_embed_dim)
        elif obs_type == 'image_memory':
            self._init_image_memory_components(world_model_cfg)
        else:
            raise ValueError(f"Unsupported observation type: {obs_type}")

        # --- Initialize world model and tokenizer ---
        self.world_model = WorldModelMT(config=world_model_cfg, tokenizer=self.tokenizer)

        # --- Log parameter counts for analysis ---
        self._log_model_parameters(obs_type)

    def _init_vector_components(self, world_model_cfg: EasyDict, obs_act_embed_dim: int) -> None:
        """Initializes components for 'vector' observation type."""
        self.representation_network = RepresentationNetworkMLP(
            observation_shape=world_model_cfg.observation_shape,
            hidden_channels=obs_act_embed_dim,
            layer_num=2,
            activation=self.activation,
            group_size=world_model_cfg.group_size,
        )
        # TODO: This is currently specific to MemoryEnv. Generalize if needed.
        self.decoder_network = VectorDecoderForMemoryEnv(embedding_dim=world_model_cfg.embed_dim, output_shape=25)
        self.tokenizer = Tokenizer(
            encoder=self.representation_network,
            decoder_network=self.decoder_network,
            with_lpips=False,
            obs_type=world_model_cfg.obs_type
        )

    def _init_image_components(self, world_model_cfg: EasyDict, observation_shape: SequenceType, num_res_blocks: int,
                               num_channels: int, obs_act_embed_dim: int) -> None:
        """Initializes components for 'image' observation type."""
        self.representation_network = nn.ModuleList()
        encoder_type = world_model_cfg.encoder_type

        # NOTE: Using a single shared encoder. The original code used a loop `for _ in range(1):`.
        # To support N independent encoders, this logic would need to be modified.
        if encoder_type == "resnet":
            encoder = RepresentationNetworkUniZero(
                observation_shape=observation_shape,
                num_res_blocks=num_res_blocks,
                num_channels=num_channels,
                downsample=self.downsample,
                activation=self.activation,
                norm_type=world_model_cfg.norm_type,
                embedding_dim=obs_act_embed_dim,
                group_size=world_model_cfg.group_size,
                final_norm_option_in_encoder=world_model_cfg.final_norm_option_in_encoder,
            )
            self.representation_network.append(encoder)
        elif encoder_type == "vit":
            vit_configs = {
                'small': {'dim': 768, 'depth': 6, 'heads': 6, 'mlp_dim': 2048},
                'base': {'dim': 768, 'depth': 12, 'heads': 12, 'mlp_dim': 3072},
                'large': {'dim': 1024, 'depth': 24, 'heads': 16, 'mlp_dim': 4096},
            }
            vit_size = 'base' if self.task_num > 8 else 'small'
            selected_vit_config = vit_configs[vit_size]

            vit_params = {
                'image_size': observation_shape[1],
                'patch_size': 8,
                'num_classes': obs_act_embed_dim,
                'dropout': 0.1,
                'emb_dropout': 0.1,
                'final_norm_option_in_encoder': world_model_cfg.final_norm_option_in_encoder,
                'lora_config': world_model_cfg,
                **selected_vit_config
            }
            vit_config = ViTConfig(**vit_params)
            encoder = ViT(config=vit_config)
            
            self.representation_network.append(encoder)
        else:
            raise ValueError(f"Unsupported encoder type for image observations: {encoder_type}")

        # For image observations, the decoder is currently not used for reconstruction during training.
        self.decoder_network = None
        self.tokenizer = Tokenizer(
            encoder=self.representation_network,
            decoder_network=self.decoder_network,
            with_lpips=False,
            obs_type=world_model_cfg.obs_type
        )
        if world_model_cfg.analysis_sim_norm:
            self.encoder_hook = FeatureAndGradientHook()
            self.encoder_hook.setup_hooks(self.representation_network)

    def _init_image_memory_components(self, world_model_cfg: EasyDict) -> None:
        """Initializes components for 'image_memory' observation type."""
        # TODO: The 'concat_task_embed' option needs to be fully implemented for this obs_type.
        self.representation_network = LatentEncoderForMemoryEnv(
            image_shape=(3, 5, 5),
            embedding_size=world_model_cfg.embed_dim,
            channels=[16, 32, 64],
            kernel_sizes=[3, 3, 3],
            strides=[1, 1, 1],
            activation=self.activation,
            group_size=world_model_cfg.group_size,
        )
        self.decoder_network = LatentDecoderForMemoryEnv(
            image_shape=(3, 5, 5),
            embedding_size=world_model_cfg.embed_dim,
            channels=[64, 32, 16],
            kernel_sizes=[3, 3, 3],
            strides=[1, 1, 1],
            activation=self.activation,
        )
        self.tokenizer = Tokenizer(
            encoder=self.representation_network,
            decoder_network=self.decoder_network,
            with_lpips=True,
            obs_type=world_model_cfg.obs_type
        )
        if world_model_cfg.analysis_sim_norm:
            self.encoder_hook = FeatureAndGradientHook()
            self.encoder_hook.setup_hooks(self.representation_network)

    def _log_model_parameters(self, obs_type: str) -> None:
        """Logs the parameter counts of the main model components."""
        print('--------------------------------------------------')
        print(f'{sum(p.numel() for p in self.world_model.parameters()):,} parameters in world_model')
        print(f'{sum(p.numel() for p in self.world_model.transformer.parameters()):,} parameters in world_model.transformer')
        print(f'{sum(p.numel() for p in self.tokenizer.encoder.parameters()):,} parameters in tokenizer.encoder')

        if obs_type in ['vector', 'image_memory'] and self.tokenizer.decoder_network is not None:
            print(f'{sum(p.numel() for p in self.tokenizer.decoder_network.parameters()):,} parameters in tokenizer.decoder_network')
            if obs_type == 'image_memory':
                 # Calculate parameters excluding decoder and LPIPS for a specific comparison point.
                 params_without_decoder = sum(p.numel() for p in self.world_model.parameters()) - \
                                          sum(p.numel() for p in self.tokenizer.decoder_network.parameters()) - \
                                          sum(p.numel() for p in self.tokenizer.lpips.parameters())
                 print(f'{params_without_decoder:,} parameters in world_model (excluding decoder and lpips)')
        print('--------------------------------------------------')

    def initial_inference(self, obs_batch: torch.Tensor, action_batch: Optional[torch.Tensor] = None,
                          current_obs_batch: Optional[torch.Tensor] = None, task_id: Optional[Any] = None) -> MZNetworkOutput:
        """
        Overview:
            Performs the initial inference step of the model, corresponding to the representation function `h` in MuZero.
            It takes an observation and produces a latent state and initial predictions.
        Arguments:
            - obs_batch (:obj:`torch.Tensor`): A batch of initial observations.
            - action_batch (:obj:`Optional[torch.Tensor]`): A batch of actions (if available, context-dependent).
            - current_obs_batch (:obj:`Optional[torch.Tensor]`): A batch of current observations (if different from obs_batch).
            - task_id (:obj:`Optional[Any]`): Identifier for the current task in a multi-task setting.
        Returns:
            - MZNetworkOutput: An object containing the predicted value, policy logits, and the initial latent state.
              The reward is set to a zero tensor, as it's not predicted at the initial step.
        """
        batch_size = obs_batch.size(0)
        obs_act_dict = {'obs': obs_batch, 'action': action_batch, 'current_obs': current_obs_batch}

        _, obs_token, logits_rewards, logits_policy, logits_value = self.world_model.forward_initial_inference(
            obs_act_dict, task_id=task_id
        )

        # The world model returns tokens and logits; map them to the standard MZNetworkOutput format.
        latent_state = obs_token
        policy_logits = logits_policy.squeeze(1)
        value = logits_value.squeeze(1)

        return MZNetworkOutput(
            value=value,
            reward=torch.zeros(batch_size, device=value.device),  # Reward is 0 at initial inference
            policy_logits=policy_logits,
            latent_state=latent_state,
        )

    def recurrent_inference(self, state_action_history: torch.Tensor, simulation_index: int = 0,
                            search_depth: List = [], task_id: Optional[Any] = None) -> MZNetworkOutput:
        """
        Overview:
            Performs a recurrent inference step, corresponding to the dynamics function `g` and prediction
            function `f` in MuZero. It predicts the next latent state, reward, policy, and value based on a
            history of latent states and actions.
        Arguments:
            - state_action_history (:obj:`torch.Tensor`): A tensor representing the history of latent states and actions.
            - simulation_index (:obj:`int`): The index of the current simulation step within MCTS.
            - search_depth (:obj:`List`): Information about the search depth, used for positional embeddings.
            - task_id (:obj:`Optional[Any]`): Identifier for the current task in a multi-task setting.
        Returns:
            - MZNetworkOutput: An object containing the predicted value, reward, policy logits, and the next latent state.
        """
        _, logits_observations, logits_rewards, logits_policy, logits_value = self.world_model.forward_recurrent_inference(
            state_action_history, simulation_index, search_depth, task_id=task_id
        )

        # Map the world model outputs to the standard MZNetworkOutput format.
        next_latent_state = logits_observations
        reward = logits_rewards.squeeze(1)
        policy_logits = logits_policy.squeeze(1)
        value = logits_value.squeeze(1)

        return MZNetworkOutput(
            value=value,
            reward=reward,
            policy_logits=policy_logits,
            latent_state=next_latent_state,
        )
import collections
import logging
import math
import os
from typing import Any, Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from ding.utils import get_rank
from einops import rearrange
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Patch
from sklearn.manifold import TSNE

from lzero.model.common import SimNorm
from lzero.model.unizero_world_models.world_model import WorldModel
from lzero.model.utils import (
    calculate_dormant_ratio,
    calculate_effective_rank,
    compute_average_weight_magnitude,
)

from .slicer import Head
from .tokenizer import Tokenizer
from .transformer import Transformer, TransformerConfig
from .utils import LossWithIntermediateLosses, WorldModelOutput, hash_state, init_weights

# Set the logging level for the root logger
logging.getLogger().setLevel(logging.DEBUG)


class WorldModelMT(WorldModel):
    """
    Overview:
        The WorldModel class for the multi-task UniZero model. It is responsible for
        predicting the next latent state, reward, policy, and value based on the
        current latent state and action. This model is a scalable latent world model
        composed of three main parts: a tokenizer, a transformer, and prediction heads.
    """

    def __init__(self, config: TransformerConfig, tokenizer: Tokenizer) -> None:
        """
        Overview:
            Initializes the multi-task WorldModel.
        Arguments:
            - config (:obj:`TransformerConfig`): The configuration object for the transformer and world model.
            - tokenizer (:obj:`Tokenizer`): The tokenizer for encoding observations.
        """
        super().__init__(config, tokenizer)
        self.tokenizer = tokenizer
        self.config = config

        self.continuous_action_space = self.config.continuous_action_space
        self.task_num = config.task_num
        self.env_num = self.config.env_num

        # TODO: Investigate sharing the encoder across all 26 games and scaling its gradient.
        # if not self.continuous_action_space:
        #     # Share encoder for Atari games.
        #     encoder_index = 0
        #     encoder = self.tokenizer.encoder[encoder_index]
        #     # Register a hook for all parameters of the encoder to scale gradients.
        #     for p in encoder.parameters():
        #         p.register_hook(self._scale_grad)

        # Whether to share prediction heads across tasks.
        self.share_head = config.share_head

        self.device = torch.device('cuda' if torch.cuda.is_available() and self.config.device != 'cpu' else 'cpu')
        print(f"self.device: {self.device}")

        # Positional embedding layer.
        self.pos_emb = nn.Embedding(config.max_tokens, self.config.embed_dim, device=self.device)
        print(f"self.pos_emb.weight.device: {self.pos_emb.weight.device}")

        # Task embedding setup.
        self.use_task_embed = config.use_task_embed
        self.task_embed_option = self.config.task_embed_option
        self.task_embed_dim = config.task_embed_dim if hasattr(config, "task_embed_dim") else 96
        self.register_token_num = config.register_token_num if hasattr(config, "register_token_num") else 4

        if self.task_embed_option == "register_task_embed":
            # When using "register_task_embed", the positional encoding is not adjusted.
            # Use a non-trainable, zero-initialized nn.Embedding for positional embeddings.
            self.pos_emb = nn.Embedding(config.max_tokens, self.config.embed_dim, device=self.device)
            nn.init.constant_(self.pos_emb.weight, 0.0)  # Initialize with all zeros.
            self.pos_emb.weight.requires_grad = False    # Disable updates.

        # Precompute positional embedding differences for efficient inference.
        self.precompute_pos_emb_diff_kv()

        self.sim_norm = SimNorm(simnorm_dim=self.config.group_size)

        # Configure embedding dimensions based on the task embedding strategy.
        if self.task_embed_option == "concat_task_embed":
            # TODO: Currently, with "concat_task_embed", self.pos_emb needs to be fixed at 0.
            self.task_emb = nn.Embedding(self.task_num, self.task_embed_dim, max_norm=1)  # TDMPC2 suggests max_norm=1.
            self.obs_act_embed_dim = config.embed_dim - self.task_embed_dim
            self.register_token_num = 0
        elif self.task_embed_option == "register_task_embed":
            self.task_emb = nn.Embedding(self.task_num, config.embed_dim, max_norm=1)
            self.obs_act_embed_dim = config.embed_dim
        elif self.task_embed_option == "add_task_embed":
            self.task_emb = nn.Embedding(self.task_num, config.embed_dim, max_norm=1)
            self.obs_act_embed_dim = config.embed_dim
        else:
            self.task_emb = None
            self.obs_act_embed_dim = config.embed_dim
            self.register_token_num = 0

        self.transformer = Transformer(self.config, self.task_emb)

        # --- Analysis and Logging Setup ---
        self.analysis_dormant_ratio_interval = self.config.get('analysis_dormant_ratio_interval', 100)
        self._analysis_step_counter = 0
        self.do_analysis = self.config.analysis_dormant_ratio_weight_rank

        self.analysis_tsne = self.config.get('analysis_tsne', False)
        if self.analysis_tsne:
            self.env_id_list = self.config.env_id_list
            # Automatically generate short names for environments.
            self.env_short_names = {
                env_id: env_id.replace('NoFrameskip-v4', '')
                for env_id in self.config.env_id_list
            }
            # Color mapping to ensure each task has a fixed color.
            self.num_tasks = len(self.env_id_list)
            self.colors = self._generate_colors(self.num_tasks)

        # --- Prediction Head Initialization ---
        self.head_policy_multi_task = nn.ModuleList()
        self.head_value_multi_task = nn.ModuleList()
        self.head_rewards_multi_task = nn.ModuleList()
        self.head_observations_multi_task = nn.ModuleList()

        self.num_experts_in_moe_head = config.num_experts_in_moe_head
        self.use_normal_head = config.use_normal_head
        self.use_moe_head = config.use_moe_head
        self.use_softmoe_head = config.use_softmoe_head

        self.to(self.device)

        # Initialize configuration parameters from the config object.
        self._initialize_config_parameters()
        self._initialize_patterns()

        self.hidden_size = config.embed_dim // config.num_heads

        # Initialize action embedding table based on action space type.
        if self.continuous_action_space:
            self.act_embedding_table = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(config.action_space_size_list[task_id], self.obs_act_embed_dim, device=self.device, bias=False),
                    SimNorm(simnorm_dim=self.group_size)
                ) for task_id in range(self.task_num)
            ])
        else:
            # For discrete action space.
            self.act_embedding_table = nn.Embedding(config.action_space_size, self.obs_act_embed_dim, device=self.device)
            print(f"self.act_embedding_table.weight.device: {self.act_embedding_table.weight.device}")
            print(f'=' * 20)
            print(f"self.obs_act_embed_dim: {self.obs_act_embed_dim}")
            print(f'=' * 20)

        assert self.num_experts_in_moe_head > 0
        if self.use_normal_head:
            self.final_norm_option_in_obs_head = getattr(config, 'final_norm_option_in_obs_head', 'LayerNorm')
            print('We use normal head')
            for task_id in range(self.task_num):
                if self.continuous_action_space:
                    self.sigma_type = self.config.sigma_type
                    self.bound_type = self.config.bound_type
                    head_policy = self._create_head_cont(self.value_policy_tokens_pattern, self.config.action_space_size_list[task_id])
                else:
                    head_policy = self._create_head(self.value_policy_tokens_pattern, self.action_space_size)

                if not self.share_head or task_id == 0:
                    self.head_policy_multi_task.append(head_policy)

                head_value = self._create_head(self.value_policy_tokens_pattern, self.support_size)
                if not self.share_head or task_id == 0:
                    self.head_value_multi_task.append(head_value)

                head_rewards = self._create_head(self.act_tokens_pattern, self.support_size)
                if not self.share_head or task_id == 0:
                    self.head_rewards_multi_task.append(head_rewards)

                head_observations = self._create_head(
                    self.all_but_last_latent_state_pattern,
                    self.config.embed_dim,
                    self._get_final_norm(self.final_norm_option_in_obs_head)  # Use the specified normalization method.
                )
                if not self.share_head or task_id == 0:
                    self.head_observations_multi_task.append(head_observations)

        elif self.use_softmoe_head:
            print(f'We use softmoe head, self.num_experts_in_moe_head is {self.num_experts_in_moe_head}')
            self.soft_moe_instances = {}
            self.create_head_modules_softmoe()
            self.head_policy_multi_task.append(self.head_policy)
            self.head_value_multi_task.append(self.head_value)
            self.head_rewards_multi_task.append(self.head_rewards)
            self.head_observations_multi_task.append(self.head_observations)
        elif self.use_moe_head:
            print(f'We use moe head, self.num_experts_in_moe_head is {self.num_experts_in_moe_head}')
            self.moe_instances = {}
            self.create_head_modules_moe()
            self.head_policy_multi_task.append(self.head_policy)
            self.head_value_multi_task.append(self.head_value)
            self.head_rewards_multi_task.append(self.head_rewards)
            self.head_observations_multi_task.append(self.head_observations)

        # Group all head modules into a ModuleDict for easier management.
        self.head_dict = nn.ModuleDict({
            name: module for name, module in self.named_children()
            if name.startswith("head_") and name.endswith("_multi_task")
        })
        print("=" * 20)
        print(f"self.head_dict:{self.head_dict}")

        # Apply weight initialization. The order of initialization is important.
        self.apply(lambda module: init_weights(module, norm_type=self.config.norm_type))
        self._initialize_last_layer_mt()

        # --- Cache and State Initialization ---
        self._initialize_cache_structures()
        self._initialize_projection_input_dim()
        self._initialize_statistics()
        self._initialize_transformer_keys_values()

        self.latent_recon_loss = torch.tensor(0., device=self.device)
        self.perceptual_loss = torch.tensor(0., device=self.device)

        # 先设置为game_segment_length，以保持self.shared_pool_init_infer都是有效的kv
        # TODO: 非常重要，应该改为和segment_length一样
        self.shared_pool_size_init = int(self.config.game_segment_length)  # NOTE: Will having too many cause incorrect retrieval of the kv cache?

        self.shared_pool_size_recur = int(self.num_simulations*self.env_num)
        self.shared_pool_recur_infer = [None] * self.shared_pool_size_recur
        self.shared_pool_index = 0

        # For init_infer, it only needs to retain the results of the most recent step.
        # NOTE: A large pool size might cause incorrect retrieval of the kv cache.
        self.shared_pool_init_infer = [[None] * self.shared_pool_size_init for _ in range(self.env_num)]
        self.shared_pool_index_init_envs = [0 for _ in range(self.env_num)]

        # For wm (world model) forward passes during training.
        self.shared_pool_size_wm = int(self.env_num)
        self.shared_pool_wm = [None] * self.shared_pool_size_wm
        self.shared_pool_index_wm = 0

        self.reanalyze_phase = False
        self._rank = get_rank()

    def _scale_grad(self, grad: torch.Tensor) -> torch.Tensor:
        """
        Overview:
            Scales the gradient. This hook is registered to encoder parameters
            to stabilize multi-task training.
        Arguments:
            - grad (:obj:`torch.Tensor`): The original gradient.
        Returns:
            - (:obj:`torch.Tensor`): The scaled gradient.
        """
        # Scale by 1/sqrt(k) for a conservative approach, where k is the number of tasks.
        return grad / math.sqrt(self.task_num)

    def _generate_colors(self, num_colors: int) -> list:
        """
        Overview:
            Generates a list of unique colors for visualization purposes,
            suitable for a large number of categories.
        Arguments:
            - num_colors (:obj:`int`): The desired number of unique colors.
        Returns:
            - (:obj:`list`): A list of colors.
        """
        # Concatenate multiple discrete colormaps from matplotlib to get more colors.
        color_maps = ['tab20', 'tab20b', 'tab20c']
        colors = []
        for cmap_name in color_maps:
            cmap = plt.get_cmap(cmap_name)
            colors.extend([cmap(i) for i in range(cmap.N)])
            if len(colors) >= num_colors:
                break
        # Generate additional colors if needed.
        if len(colors) < num_colors:
            additional_colors = plt.cm.get_cmap('hsv', num_colors - len(colors))
            colors.extend([additional_colors(i) for i in range(num_colors - len(colors))])
        return colors[:num_colors]

    def _initialize_config_parameters(self) -> None:
        """Initializes model attributes from the configuration object."""
        self.policy_entropy_weight = self.config.policy_entropy_weight
        self.predict_latent_loss_type = self.config.predict_latent_loss_type
        self.group_size = self.config.group_size
        self.num_groups = self.config.embed_dim // self.group_size
        self.obs_type = self.config.obs_type
        self.embed_dim = self.config.embed_dim
        self.num_heads = self.config.num_heads
        self.gamma = self.config.gamma
        self.context_length = self.config.context_length
        self.dormant_threshold = self.config.dormant_threshold
        self.analysis_dormant_ratio_weight_rank = self.config.analysis_dormant_ratio_weight_rank
        self.num_observations_tokens = self.config.tokens_per_block - 1
        self.latent_recon_loss_weight = self.config.latent_recon_loss_weight
        self.perceptual_loss_weight = self.config.perceptual_loss_weight
        self.support_size = self.config.support_size
        self.action_space_size = self.config.action_space_size
        self.max_cache_size = self.config.max_cache_size
        self.num_layers = self.config.num_layers

    def _initialize_patterns(self) -> None:
        """Initializes patterns (masks) for selecting specific tokens for prediction heads."""
        self.all_but_last_latent_state_pattern = torch.ones(self.config.tokens_per_block)
        self.all_but_last_latent_state_pattern[-2] = 0
        self.act_tokens_pattern = torch.zeros(self.config.tokens_per_block)
        self.act_tokens_pattern[-1] = 1
        self.value_policy_tokens_pattern = torch.zeros(self.config.tokens_per_block)
        self.value_policy_tokens_pattern[-2] = 1

    def _get_final_norm(self, norm_option: str) -> nn.Module:
        """Returns the specified normalization module."""
        if norm_option == 'LayerNorm':
            return nn.LayerNorm(self.config.embed_dim, eps=1e-5)
        elif norm_option == 'SimNorm':
            return SimNorm(simnorm_dim=self.config.group_size)
        else:
            raise ValueError(f"Unsupported final_norm_option_in_obs_head: {norm_option}")

    def _create_head(self, block_mask: torch.Tensor, output_dim: int, norm_layer: Optional[nn.Module] = None) -> Head:
        """Creates a standard prediction head."""
        modules = [
            nn.LayerNorm(self.config.embed_dim),  # <-- 核心优化！ # TODO
            nn.Linear(self.config.embed_dim, self.config.embed_dim),
            nn.LayerNorm(self.config.embed_dim),      # 2. <-- 新增！稳定内部激活
            nn.GELU(approximate='tanh'),
            nn.Linear(self.config.embed_dim, output_dim)
        ]
        if norm_layer:
            modules.append(norm_layer)
        return Head(
            max_blocks=self.config.max_blocks,
            block_mask=block_mask,
            head_module=nn.Sequential(*modules)
        )

    def _create_head_moe(self, block_mask: torch.Tensor, output_dim: int, norm_layer: Optional[nn.Module] = None, moe: Optional[nn.Module] = None) -> Head:
        """Creates a prediction head with a Mixture-of-Experts (MoE) layer."""
        modules = [
            nn.LayerNorm(self.config.embed_dim),  # <-- 核心优化！ # TODO
            moe,
            nn.Linear(self.config.embed_dim, output_dim)
        ]
        if norm_layer:
            modules.append(norm_layer)
        return Head(
            max_blocks=self.config.max_blocks,
            block_mask=block_mask,
            head_module=nn.Sequential(*modules)
        )

    def get_moe(self, name: str) -> nn.Module:
        """Gets or creates a MoE instance by name."""
        from .moe import MoELayer, MultiplicationFeedForward

        if name not in self.moe_instances:
            # Create multiple FeedForward instances for multiplication-based MoE.
            experts = nn.ModuleList([
                MultiplicationFeedForward(self.config) for _ in range(self.config.num_experts_of_moe_in_transformer)
            ])
            self.moe_instances[name] = MoELayer(
                experts=experts,
                gate=nn.Linear(self.config.embed_dim, self.config.num_experts_of_moe_in_transformer, bias=False),
                num_experts_per_tok=1,
            )
        return self.moe_instances[name]

    def create_head_modules_moe(self) -> None:
        """Creates all MoE prediction head modules."""
        self.head_rewards = self._create_head_moe(self.act_tokens_pattern, self.support_size, moe=self.get_moe("rewards_moe"))
        self.head_observations = self._create_head_moe(self.all_but_last_latent_state_pattern, self.embed_dim, norm_layer=self.sim_norm, moe=self.get_moe("observations_moe"))
        self.head_policy = self._create_head_moe(self.value_policy_tokens_pattern, self.action_space_size, moe=self.get_moe("policy_moe"))
        self.head_value = self._create_head_moe(self.value_policy_tokens_pattern, self.support_size, moe=self.get_moe("value_moe"))

    def _create_head_softmoe(self, block_mask: torch.Tensor, output_dim: int, norm_layer: Optional[nn.Module] = None, soft_moe: Optional[nn.Module] = None) -> Head:
        """Creates a prediction head with a Soft-MoE layer."""
        modules = [
            soft_moe,
            nn.Linear(self.config.embed_dim, output_dim)
        ]
        if norm_layer:
            modules.append(norm_layer)
        return Head(
            max_blocks=self.config.max_blocks,
            block_mask=block_mask,
            head_module=nn.Sequential(*modules)
        )

    def get_soft_moe(self, name: str) -> nn.Module:
        """Gets or creates a Soft-MoE instance by name."""
        from soft_moe_pytorch import DynamicSlotsSoftMoE as SoftMoE
        if name not in self.soft_moe_instances:
            self.soft_moe_instances[name] = SoftMoE(
                dim=self.embed_dim,
                num_experts=self.num_experts_in_moe_head,
                geglu=True
            )
        return self.soft_moe_instances[name]

    def create_head_modules_softmoe(self) -> None:
        """Creates all Soft-MoE prediction head modules."""
        self.head_rewards = self._create_head_softmoe(self.act_tokens_pattern, self.support_size, soft_moe=self.get_soft_moe("rewards_soft_moe"))
        self.head_observations = self._create_head_softmoe(self.all_but_last_latent_state_pattern, self.config.embed_dim, norm_layer=self.sim_norm, soft_moe=self.get_soft_moe("observations_soft_moe"))
        self.head_policy = self._create_head_softmoe(self.value_policy_tokens_pattern, self.action_space_size, soft_moe=self.get_soft_moe("policy_soft_moe"))
        self.head_value = self._create_head_softmoe(self.value_policy_tokens_pattern, self.support_size, soft_moe=self.get_soft_moe("value_soft_moe"))

    def _initialize_last_layer_mt(self) -> None:
        """Initializes the last linear layer of prediction heads to zero for training stability."""
        last_linear_layer_init_zero = True
        print(f'world_model_mt.py:self.task_num:{self.task_num}')
        if last_linear_layer_init_zero:
            if self.continuous_action_space:
                # For continuous actions, policy head might have a different initialization strategy.
                module_to_initialize = self.head_value_multi_task + self.head_rewards_multi_task + self.head_observations_multi_task
            else:
                module_to_initialize = self.head_policy_multi_task + self.head_value_multi_task + self.head_rewards_multi_task + self.head_observations_multi_task

            for head in module_to_initialize:
                for layer in reversed(head.head_module):
                    if isinstance(layer, nn.Linear):
                        nn.init.zeros_(layer.weight)
                        if layer.bias is not None:
                            nn.init.zeros_(layer.bias)
                        break

    def _initialize_cache_structures(self) -> None:
        """Initializes cache structures for storing past keys and values during inference."""
        # self.past_kv_cache_recurrent_infer = collections.OrderedDict()
        # self.past_kv_cache_init_infer_envs = [collections.OrderedDict() for _ in range(self.env_num)]
        
        self.past_kv_cache_recurrent_infer = {}
        self.pool_idx_to_key_map_recur_infer = [None] * self.shared_pool_size_recur
        self.past_kv_cache_init_infer_envs = [{} for _ in range(self.env_num)]
        # 辅助数据结构，用于反向查找：pool_index -> key
        self.pool_idx_to_key_map_init_envs = [[None] * self.shared_pool_size_init for _ in range(self.env_num)]
        
        self.keys_values_wm_list = []
        self.keys_values_wm_size_list = []

    def _initialize_projection_input_dim(self) -> None:
        """Initializes the input dimension for the projection based on observation tokenization."""
        if self.num_observations_tokens == 16:
            self.projection_input_dim = 128
        elif self.num_observations_tokens == 1:
            if self.task_embed_option in ["concat_task_embed", "register_task_embed", "add_task_embed"]:
                self.projection_input_dim = self.config.embed_dim
                if self.task_embed_option == "concat_task_embed":
                    self.projection_input_dim -= self.task_embed_dim
            else:
                self.projection_input_dim = self.config.embed_dim

    def _initialize_statistics(self) -> None:
        """Initializes counters for cache hit rates and other statistics."""
        self.hit_count = 0
        self.total_query_count = 0
        self.length_largethan_maxminus5_context_cnt = 0
        self.length_largethan_maxminus7_context_cnt = 0
        self.root_hit_cnt = 0
        self.root_total_query_cnt = 0

    def _initialize_transformer_keys_values(self) -> None:
        """Initializes empty key-value cache structures for the transformer."""
        self.keys_values_wm_single_env = self.transformer.generate_empty_keys_values(n=1, max_tokens=self.context_length)
        self.keys_values_wm = self.transformer.generate_empty_keys_values(n=self.env_num, max_tokens=self.context_length)

    def precompute_pos_emb_diff_kv(self) -> None:
        """
        Overview:
            Precomputes positional embedding differences for keys and values. This is an
            optimization to speed up KV cache updates during recurrent inference by avoiding
            re-computation of positional embeddings.
        """
        if self.context_length <= 2:
            return  # No context to precompute for.

        # Precompute positional embedding matrices for all layers.
        self.positional_embedding_k = [self._get_positional_embedding(layer, 'key') for layer in range(self.config.num_layers)]
        self.positional_embedding_v = [self._get_positional_embedding(layer, 'value') for layer in range(self.config.num_layers)]

        # Precompute all possible positional embedding differences.
        self.pos_emb_diff_k = []
        self.pos_emb_diff_v = []

        for layer in range(self.config.num_layers):
            layer_pos_emb_diff_k = {}
            layer_pos_emb_diff_v = {}

            # This is for the case when context window is full and we shift it.
            # TODO: Generalize for different start/end points if necessary.
            for start in [2]:
                for end in [self.context_length - 1]:
                    original_pos_emb_k = self.positional_embedding_k[layer][:, :, start:end, :]
                    new_pos_emb_k = self.positional_embedding_k[layer][:, :, :end - start, :]
                    layer_pos_emb_diff_k[(start, end)] = new_pos_emb_k - original_pos_emb_k

                    original_pos_emb_v = self.positional_embedding_v[layer][:, :, start:end, :]
                    new_pos_emb_v = self.positional_embedding_v[layer][:, :, :end - start, :]
                    layer_pos_emb_diff_v[(start, end)] = new_pos_emb_v - original_pos_emb_v

            self.pos_emb_diff_k.append(layer_pos_emb_diff_k)
            self.pos_emb_diff_v.append(layer_pos_emb_diff_v)

    def _get_positional_embedding(self, layer: int, attn_type: str) -> torch.Tensor:
        """
        Overview:
            Helper function to get positional embedding for a given layer and attention type.
        Arguments:
            - layer (:obj:`int`): The layer index.
            - attn_type (:obj:`str`): The attention type, either 'key' or 'value'.
        Returns:
            - (:obj:`torch.Tensor`): The positional embedding tensor, detached from the graph.
        """
        # TODO: Review the use of detach(). It's used here to prevent gradients from flowing back
        # through the positional embeddings during this pre-computation phase.
        attn_func = getattr(self.transformer.blocks[layer].attn, attn_type)
        pos_emb = attn_func(self.pos_emb.weight).view(
            1, self.config.max_tokens, self.num_heads, self.embed_dim // self.num_heads
        ).transpose(1, 2)
        return pos_emb.to(self.device).detach()

    def forward(
        self,
        obs_embeddings_or_act_tokens: Dict[str, Union[torch.Tensor, tuple]],
        past_keys_values: Optional[torch.Tensor] = None,
        kvcache_independent: bool = False,
        is_init_infer: bool = True,
        valid_context_lengths: Optional[torch.Tensor] = None,
        task_id: int = 0
    ) -> WorldModelOutput:
        """
        Overview:
            Main forward pass for the world model. It processes either observation embeddings,
            action tokens, or a combination of both, and passes them through the transformer
            to generate predictions.
        Arguments:
            - obs_embeddings_or_act_tokens (:obj:`Dict`): A dictionary containing input tensors.
                Can be 'obs_embeddings', 'act_tokens', or 'obs_embeddings_and_act_tokens'.
            - past_keys_values (:obj:`Optional[torch.Tensor]`): The KV cache from previous steps.
            - kvcache_independent (:obj:`bool`): Whether to use independent KV caching per item in the batch.
            - is_init_infer (:obj:`bool`): Flag indicating if this is an initial inference step.
            - valid_context_lengths (:obj:`Optional[torch.Tensor]`): Tensor of valid context lengths for each item.
            - task_id (:obj:`int`): The ID of the current task.
        Returns:
            - (:obj:`WorldModelOutput`): An object containing the transformer output and logits for
                observations, rewards, policy, and value.
        """
        if self.use_task_embed:
            self.task_embeddings = self.task_emb(torch.tensor(task_id, device=self.device))
            self.task_embeddings = self.sim_norm(self.task_embeddings.view(1, -1)).view(-1)
        else:
            # Use a zero tensor if task embeddings are disabled.
            self.task_embeddings = torch.zeros(self.config.embed_dim, device=self.device)

        prev_steps = 0 if past_keys_values is None else past_keys_values.size
        if kvcache_independent:
            prev_steps = torch.tensor([0 if past_keys_values is None else past_kv.size for past_kv in past_keys_values], device=self.device)

        if is_init_infer:
            valid_context_lengths = None

        # --- Branch 1: Inference Phase (Collect/Eval) - Process observation embeddings ---
        if 'obs_embeddings' in obs_embeddings_or_act_tokens:
            obs_embeddings = obs_embeddings_or_act_tokens['obs_embeddings']
            if len(obs_embeddings.shape) == 2:
                obs_embeddings = obs_embeddings.unsqueeze(1)

            # Apply task embeddings based on the chosen strategy.
            if self.task_embed_option == "add_task_embed":
                obs_embeddings = obs_embeddings + self.task_embeddings
            elif self.task_embed_option == "concat_task_embed":
                if is_init_infer and not self.reanalyze_phase:
                    # Concatenate task embeddings only during initial inference.
                    task_emb_expanded = self.task_embeddings.view(1, 1, -1).expand(obs_embeddings.shape[0], obs_embeddings.shape[1], -1)
                    obs_embeddings = torch.cat([obs_embeddings, task_emb_expanded], dim=-1)

            num_steps = obs_embeddings.size(1)
            sequences = self._add_position_embeddings(obs_embeddings, prev_steps, num_steps, kvcache_independent, is_init_infer, valid_context_lengths)

        # --- Branch 2: Inference Phase (Collect/Eval) - Process action tokens ---
        elif 'act_tokens' in obs_embeddings_or_act_tokens:
            act_tokens = obs_embeddings_or_act_tokens['act_tokens']
            if self.continuous_action_space:
                num_steps = 1
                act_tokens = act_tokens.float()
                if len(act_tokens.shape) == 2:
                    act_tokens = act_tokens.unsqueeze(1)
            else:
                if len(act_tokens.shape) == 3:
                    act_tokens = act_tokens.squeeze(1)
                num_steps = act_tokens.size(1)

            # Get action embeddings from the task-specific or shared table.
            if self.task_num >= 1 and self.continuous_action_space:
                act_embeddings = self.act_embedding_table[task_id](act_tokens)
            else:
                act_embeddings = self.act_embedding_table(act_tokens)

            # Apply task embeddings.
            if self.task_embed_option == "concat_task_embed":
                task_emb_expanded = self.task_embeddings.view(1, 1, -1).expand(act_embeddings.shape[0], act_embeddings.shape[1], -1)
                act_embeddings = torch.cat([act_embeddings, task_emb_expanded], dim=-1)

            sequences = self._add_position_embeddings(act_embeddings, prev_steps, num_steps, kvcache_independent, is_init_infer, valid_context_lengths)

        # --- Branch 3: Training Phase - Process combined observation embeddings and action tokens ---
        else:
            if self.continuous_action_space:
                sequences, num_steps = self._process_obs_act_combined_cont(obs_embeddings_or_act_tokens, prev_steps, task_id=task_id)
            else:
                sequences, num_steps = self._process_obs_act_combined(obs_embeddings_or_act_tokens, prev_steps)

        # Pass sequences through the transformer.
        x = self._transformer_pass(sequences, past_keys_values, kvcache_independent, valid_context_lengths, task_id=task_id)

        # Generate logits using shared, task-specific, or MoE heads.
        head_index = 0 if self.share_head else task_id
        if self.use_moe_head or self.use_softmoe_head:
            logits_observations = self.head_observations(x, num_steps=num_steps, prev_steps=prev_steps)
            logits_rewards = self.head_rewards(x, num_steps=num_steps, prev_steps=prev_steps)
            logits_policy = self.head_policy(x, num_steps=num_steps, prev_steps=prev_steps)
            logits_value = self.head_value(x, num_steps=num_steps, prev_steps=prev_steps)
        else:
            logits_observations = self.head_observations_multi_task[head_index](x, num_steps=num_steps, prev_steps=prev_steps)
            logits_rewards = self.head_rewards_multi_task[head_index](x, num_steps=num_steps, prev_steps=prev_steps)
            logits_policy = self.head_policy_multi_task[head_index](x, num_steps=num_steps, prev_steps=prev_steps)
            logits_value = self.head_value_multi_task[head_index](x, num_steps=num_steps, prev_steps=prev_steps)

        return WorldModelOutput(x, logits_observations, logits_rewards, None, logits_policy, logits_value)

    def _add_position_embeddings(
        self,
        embeddings: torch.Tensor,
        prev_steps: Union[int, torch.Tensor],
        num_steps: int,
        kvcache_independent: bool,
        is_init_infer: bool,
        valid_context_lengths: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Overview:
            Adds positional embeddings to the input embeddings.
        Arguments:
            - embeddings (:obj:`torch.Tensor`): Input embeddings.
            - prev_steps (:obj:`Union[int, torch.Tensor]`): Number of previous steps in the cache.
            - num_steps (:obj:`int`): Number of new steps being added.
            - kvcache_independent (:obj:`bool`): Flag for independent KV caching.
            - is_init_infer (:obj:`bool`): Flag for initial inference.
            - valid_context_lengths (:obj:`Optional[torch.Tensor]`): Valid context lengths for each sequence.
        Returns:
            - (:obj:`torch.Tensor`): Embeddings with added positional information.
        """
        if kvcache_independent:
            steps_indices = prev_steps.unsqueeze(1) + torch.arange(num_steps, device=embeddings.device)
            position_embeddings = self.pos_emb(steps_indices)
            return embeddings + position_embeddings
        else:
            if is_init_infer:
                # For initial inference, positions are sequential from the previous step count.
                pos_indices = prev_steps + torch.arange(num_steps, device=self.device)
                return embeddings + self.pos_emb(pos_indices)
            else:
                # For recurrent steps, use valid_context_lengths to get correct positions.
                valid_context_lengths = torch.tensor(self.keys_values_wm_size_list_current, device=self.device)
                pos_indices = valid_context_lengths.unsqueeze(1) + torch.arange(num_steps, device=self.device)
                position_embeddings = self.pos_emb(pos_indices)
                return embeddings + position_embeddings

    def _process_obs_act_combined_cont(self, obs_embeddings_or_act_tokens: dict, prev_steps: int, task_id: int = 0) -> Tuple[torch.Tensor, int]:
        """
        Overview:
            Processes and combines observation embeddings and continuous action tokens for training.
        Arguments:
            - obs_embeddings_or_act_tokens (:obj:`dict`): Dictionary with 'obs_embeddings_and_act_tokens'.
            - prev_steps (:obj:`int`): Number of previous steps.
            - task_id (:obj:`int`): The current task ID.
        Returns:
            - (:obj:`Tuple[torch.Tensor, int]`): A tuple of the combined sequence tensor and the number of steps.
        """
        obs_embeddings, act_tokens = obs_embeddings_or_act_tokens['obs_embeddings_and_act_tokens']
        if len(obs_embeddings.shape) == 3:
            obs_embeddings = obs_embeddings.view(act_tokens.shape[0], act_tokens.shape[1], self.num_observations_tokens, -1)

        num_steps = int(obs_embeddings.size(1) * (obs_embeddings.size(2) + 1))
        act_tokens = act_tokens.float()
        if len(act_tokens.shape) == 2:
            act_tokens = act_tokens.unsqueeze(-1)

        act_embeddings = self.act_embedding_table[task_id](act_tokens)

        B, L, K, E_obs = obs_embeddings.size()
        obs_act_embeddings = torch.empty(B, L * (K + 1), self.config.embed_dim, device=self.device)

        if self.task_embed_option == "concat_task_embed":
            task_emb_expanded = self.task_embeddings.view(1, 1, -1).expand(B, 1, -1)

        for i in range(L):
            obs = obs_embeddings[:, i, :, :]
            if self.task_embed_option == "add_task_embed":
                obs = obs + self.task_embeddings
            elif self.task_embed_option == "concat_task_embed":
                obs = torch.cat([obs, task_emb_expanded.expand(B, K, -1)], dim=-1)

            act = act_embeddings[:, i, :].unsqueeze(1)
            if self.task_embed_option == "concat_task_embed":
                act = torch.cat([act, task_emb_expanded], dim=-1)

            obs_act = torch.cat([obs, act], dim=1)
            obs_act_embeddings[:, i * (K + 1):(i + 1) * (K + 1), :] = obs_act

        pos_indices = prev_steps + torch.arange(num_steps, device=self.device)
        return obs_act_embeddings + self.pos_emb(pos_indices), num_steps

    def _process_obs_act_combined(self, obs_embeddings_or_act_tokens: dict, prev_steps: int, task_id: int = 0) -> Tuple[torch.Tensor, int]:
        """
        Overview:
            Processes and combines observation embeddings and discrete action tokens for training.
        Arguments:
            - obs_embeddings_or_act_tokens (:obj:`dict`): Dictionary with 'obs_embeddings_and_act_tokens'.
            - prev_steps (:obj:`int`): Number of previous steps.
            - task_id (:obj:`int`): The current task ID.
        Returns:
            - (:obj:`Tuple[torch.Tensor, int]`): A tuple of the combined sequence tensor and the number of steps.
        """
        obs_embeddings, act_tokens = obs_embeddings_or_act_tokens['obs_embeddings_and_act_tokens']
        if len(obs_embeddings.shape) == 3:
            obs_embeddings = obs_embeddings.view(act_tokens.shape[0], act_tokens.shape[1], self.num_observations_tokens, -1)

        num_steps = int(obs_embeddings.size(1) * (obs_embeddings.size(2) + 1))
        act_embeddings = self.act_embedding_table(act_tokens)

        B, L, K, E_obs = obs_embeddings.size()
        obs_act_embeddings = torch.empty(B, L * (K + 1), self.config.embed_dim, device=self.device)

        if self.task_embed_option == "concat_task_embed":
            task_emb_expanded = self.task_embeddings.view(1, 1, -1).expand(B, 1, -1)

        for i in range(L):
            obs = obs_embeddings[:, i, :, :]
            if self.task_embed_option == "add_task_embed":
                obs = obs + self.task_embeddings
            elif self.task_embed_option == "concat_task_embed":
                obs = torch.cat([obs, task_emb_expanded.expand(B, K, -1)], dim=-1)

            act = act_embeddings[:, i, 0, :].unsqueeze(1)
            if self.task_embed_option == "concat_task_embed":
                act = torch.cat([act, task_emb_expanded], dim=-1)

            obs_act = torch.cat([obs, act], dim=1)
            obs_act_embeddings[:, i * (K + 1):(i + 1) * (K + 1), :] = obs_act

        pos_indices = prev_steps + torch.arange(num_steps, device=self.device)
        return obs_act_embeddings + self.pos_emb(pos_indices), num_steps

    def _transformer_pass(
        self,
        sequences: torch.Tensor,
        past_keys_values: Optional[torch.Tensor],
        kvcache_independent: bool,
        valid_context_lengths: Optional[torch.Tensor],
        task_id: int = 0
    ) -> torch.Tensor:
        """
        Overview:
            Passes sequences through the transformer, handling different KV cache modes.
        Arguments:
            - sequences (:obj:`torch.Tensor`): Input sequences.
            - past_keys_values (:obj:`Optional[torch.Tensor]`): The KV cache from previous steps.
            - kvcache_independent (:obj:`bool`): Flag for independent KV caching.
            - valid_context_lengths (:obj:`Optional[torch.Tensor]`): Tensor of valid context lengths.
            - task_id (:obj:`int`): The current task ID.
        Returns:
            - (:obj:`torch.Tensor`): The output from the transformer.
        """
        if kvcache_independent:
            x = [
                self.transformer(sequences[k].unsqueeze(0), past_kv, valid_context_lengths=valid_context_lengths[k].unsqueeze(0))
                for k, past_kv in enumerate(past_keys_values)
            ]
            return torch.cat(x, dim=0)
        else:
            return self.transformer(sequences, past_keys_values, valid_context_lengths=valid_context_lengths)

    @torch.no_grad()
    def reset_for_initial_inference(self, obs_act_dict: dict, task_id: int = 0) -> Tuple[WorldModelOutput, torch.Tensor]:
        """
        Overview:
            Resets the model state for the beginning of an episode or a new inference sequence.
            It processes the initial observations and actions to create the first latent state
            and populate the KV cache.
        Arguments:
            - obs_act_dict (:obj:`dict`): A dictionary containing 'obs', 'action', and 'current_obs'.
            - task_id (:obj:`int`): The ID of the current task.
        Returns:
            - (:obj:`Tuple[WorldModelOutput, torch.Tensor]`): A tuple containing the world model output
                and the initial latent state.
        """
        if self.use_task_embed:
            self.task_embeddings = self.task_emb(torch.tensor(task_id, device=self.device))
            self.task_embeddings = self.sim_norm(self.task_embeddings.view(1, -1)).view(-1)
        else:
            self.task_embeddings = torch.zeros(self.config.embed_dim, device=self.device)

        batch_obs = obs_act_dict['obs']
        batch_action = obs_act_dict['action']
        batch_current_obs = obs_act_dict['current_obs']

        obs_embeddings = self.tokenizer.encode_to_obs_embeddings(batch_obs, task_id=task_id)

        if batch_current_obs is not None:
            # --- Collect and Evaluation Phase ---
            current_obs_embeddings = self.tokenizer.encode_to_obs_embeddings(batch_current_obs, task_id=task_id)

            # The latent state is the combination of observation embedding and task embedding.
            if self.use_task_embed:
                if self.task_embed_option == "add_task_embed":
                    self.latent_state = current_obs_embeddings + self.task_embeddings
                elif self.task_embed_option == "concat_task_embed":
                    task_emb_expanded = self.task_embeddings.view(1, 1, -1).expand(current_obs_embeddings.shape[0], current_obs_embeddings.shape[1], -1)
                    self.latent_state = torch.cat([current_obs_embeddings, task_emb_expanded], dim=-1)
                else: # "register_task_embed" or other cases
                    self.latent_state = current_obs_embeddings
            else:
                self.latent_state = current_obs_embeddings

            outputs_wm = self.wm_forward_for_initial_inference(obs_embeddings, batch_action, current_obs_embeddings, task_id=task_id)
        else:
            # --- Training Phase (for calculating target values) ---
            if self.use_task_embed:
                if self.task_embed_option == "add_task_embed":
                    self.latent_state = obs_embeddings + self.task_embeddings
                elif self.task_embed_option == "concat_task_embed":
                    task_emb_expanded = self.task_embeddings.view(1, 1, -1).expand(obs_embeddings.shape[0], obs_embeddings.shape[1], -1)
                    self.latent_state = torch.cat([obs_embeddings, task_emb_expanded], dim=-1)
                else:
                    self.latent_state = obs_embeddings
            else:
                self.latent_state = obs_embeddings

            outputs_wm = self.wm_forward_for_initial_inference(obs_embeddings, batch_action, None, task_id=task_id)

        return outputs_wm, self.latent_state


    #@profile
    @torch.no_grad()
    def wm_forward_for_initial_inference(self, last_obs_embeddings: torch.LongTensor,
                                                             batch_action=None,
                                                             current_obs_embeddings=None, task_id = 0) -> torch.FloatTensor:
        """
        Refresh key-value pairs with the initial latent state for inference.

        Arguments:
            - latent_state (:obj:`torch.LongTensor`): The latent state embeddings.
            - batch_action (optional): Actions taken.
            - current_obs_embeddings (optional): Current observation embeddings.
        Returns:
            - torch.FloatTensor: The outputs from the world model.
        """
        n, num_observations_tokens, _ = last_obs_embeddings.shape
        if n <= self.env_num and current_obs_embeddings is not None:
            # ================ Collect and Evaluation Phase ================
            if current_obs_embeddings is not None:
                if self.continuous_action_space:
                    first_step_flag = not isinstance(batch_action[0], np.ndarray)
                else:
                    first_step_flag = max(batch_action) == -1
                if first_step_flag:
                    # First step in an episode
                    self.keys_values_wm = self.transformer.generate_empty_keys_values(n=current_obs_embeddings.shape[0],
                                                                                      max_tokens=self.context_length)
                    # print(f"current_obs_embeddings.device: {current_obs_embeddings.device}")
                    outputs_wm = self.forward({'obs_embeddings': current_obs_embeddings},
                                              past_keys_values=self.keys_values_wm, is_init_infer=True, task_id=task_id)
                    
                    if self.use_task_embed and self.task_embed_option in ["concat_task_embed", "add_task_embed"]:
                        # Copy and store keys_values_wm for a single environment
                        self.update_cache_context(self.latent_state, is_init_infer=True)
                    else:
                        # Copy and store keys_values_wm for a single environment
                        self.update_cache_context(current_obs_embeddings, is_init_infer=True)
                else:
                    # Assume latest_state is the new latent_state, containing information from ready_env_num environments
                    ready_env_num = current_obs_embeddings.shape[0]
                    self.keys_values_wm_list = []
                    self.keys_values_wm_size_list = []
                    for i in range(ready_env_num):
                        # Retrieve latent state for a single environment
                        state_single_env = last_obs_embeddings[i]
                        # Compute hash value using latent state for a single environment
                        cache_key = hash_state(
                            state_single_env.view(-1).cpu().numpy())  # last_obs_embeddings[i] is torch.Tensor

                        # Retrieve cached value
                        cache_index = self.past_kv_cache_init_infer_envs[i].get(cache_key)
                        if cache_index is not None:
                            matched_value = self.shared_pool_init_infer[i][cache_index]
                        else:
                            matched_value = None

                        self.root_total_query_cnt += 1
                        if matched_value is not None:
                            # If a matching value is found, add it to the list
                            self.root_hit_cnt += 1
                            # deepcopy is needed because forward modifies matched_value in place
                            self.keys_values_wm_list.append(self.custom_copy_kv_cache_to_shared_wm(matched_value))
                            self.keys_values_wm_size_list.append(matched_value.size)
                        else:
                            # Reset using zero values
                            self.keys_values_wm_single_env = self.transformer.generate_empty_keys_values(n=1, max_tokens=self.context_length)
                            outputs_wm = self.forward({'obs_embeddings': state_single_env.unsqueeze(0)},
                                                      past_keys_values=self.keys_values_wm_single_env,
                                                      is_init_infer=True, task_id=task_id)
                            self.keys_values_wm_list.append(self.keys_values_wm_single_env)
                            self.keys_values_wm_size_list.append(1)

                    # Input self.keys_values_wm_list, output self.keys_values_wm
                    self.keys_values_wm_size_list_current = self.trim_and_pad_kv_cache(is_init_infer=True)

                    batch_action = batch_action[:ready_env_num]
                    # if ready_env_num < self.env_num:
                    #     print(f'init inference ready_env_num: {ready_env_num} < env_num: {self.env_num}')
                    if self.continuous_action_space:
                        act_tokens = torch.from_numpy(np.array(batch_action)).to(last_obs_embeddings.device).unsqueeze(1)
                    else:
                        act_tokens = torch.from_numpy(np.array(batch_action)).to(last_obs_embeddings.device).unsqueeze(-1)
                    outputs_wm = self.forward({'act_tokens': act_tokens}, past_keys_values=self.keys_values_wm,
                                              is_init_infer=True, task_id=task_id)

                    outputs_wm = self.forward({'obs_embeddings': current_obs_embeddings},
                                              past_keys_values=self.keys_values_wm, is_init_infer=True, task_id=task_id)

                    # Copy and store keys_values_wm for a single environment
                    if self.use_task_embed and self.task_embed_option in ["concat_task_embed", "add_task_embed"]:
                        # Copy and store keys_values_wm for a single environment
                        self.update_cache_context(self.latent_state, is_init_infer=True)
                    else:
                        # import ipdb; ipdb.set_trace()
                        # Copy and store keys_values_wm for a single environment
                        self.update_cache_context(current_obs_embeddings, is_init_infer=True)

        elif batch_action is not None and current_obs_embeddings is None:
        # elif n > self.env_num and batch_action is not None and current_obs_embeddings is None:
            # ================ calculate the target value in Train phase ================
            # [192, 16, 64] -> [32, 6, 16, 64]
            last_obs_embeddings = last_obs_embeddings.contiguous().view(batch_action.shape[0], -1, num_observations_tokens,
                                                          self.obs_act_embed_dim)  # (BL, K) for unroll_step=1

            last_obs_embeddings = last_obs_embeddings[:, :-1, :]
            batch_action = torch.from_numpy(batch_action).to(last_obs_embeddings.device)
            
            if self.continuous_action_space:
                act_tokens = batch_action
            else:
                act_tokens = rearrange(batch_action, 'b l -> b l 1')

            # select the last timestep for each sample
            # This will select the last column while keeping the dimensions unchanged, and the target policy/value in the final step itself is not used.
            last_steps_act = act_tokens[:, -1:, :]
            act_tokens = torch.cat((act_tokens, last_steps_act), dim=1)

            outputs_wm = self.forward({'obs_embeddings_and_act_tokens': (last_obs_embeddings, act_tokens)}, task_id=task_id)
            
            # select the last timestep for each sample
            last_steps_value = outputs_wm.logits_value[:, -1:, :]
            outputs_wm.logits_value = torch.cat((outputs_wm.logits_value, last_steps_value), dim=1)

            last_steps_policy = outputs_wm.logits_policy[:, -1:, :]
            outputs_wm.logits_policy = torch.cat((outputs_wm.logits_policy, last_steps_policy), dim=1)

            # Reshape your tensors
            # outputs_wm.logits_value.shape (B, H, 101) = (B*H, 101)
            outputs_wm.logits_value = rearrange(outputs_wm.logits_value, 'b t e -> (b t) e')
            outputs_wm.logits_policy = rearrange(outputs_wm.logits_policy, 'b t e -> (b t) e')

        return outputs_wm


    #@profile
    @torch.no_grad()
    def forward_initial_inference(self, obs_act_dict, task_id = 0):
        """
        Perform initial inference based on the given observation-action dictionary.

        Arguments:
            - obs_act_dict (:obj:`dict`): Dictionary containing observations and actions.
        Returns:
            - tuple: A tuple containing output sequence, latent state, logits rewards, logits policy, and logits value.
        """
        # UniZero has context in the root node
        outputs_wm, latent_state = self.reset_for_initial_inference(obs_act_dict, task_id=task_id)
        self.past_kv_cache_recurrent_infer.clear()

        return (outputs_wm.output_sequence, latent_state, outputs_wm.logits_rewards,
                outputs_wm.logits_policy, outputs_wm.logits_value)

    #@profile
    @torch.no_grad()
    def forward_recurrent_inference(self, state_action_history, simulation_index=0,
                                    latent_state_index_in_search_path=[], task_id = 0):
        """
        Perform recurrent inference based on the state-action history.

        Arguments:
            - state_action_history (:obj:`list`): List containing tuples of state and action history.
            - simulation_index (:obj:`int`, optional): Index of the current simulation. Defaults to 0.
            - latent_state_index_in_search_path (:obj:`list`, optional): List containing indices of latent states in the search path. Defaults to [].
        Returns:
            - tuple: A tuple containing output sequence, updated latent state, reward, logits policy, and logits value.
        """
        latest_state, action = state_action_history[-1]
        ready_env_num = latest_state.shape[0]

        self.keys_values_wm_list = []
        self.keys_values_wm_size_list = []
        self.keys_values_wm_size_list = self.retrieve_or_generate_kvcache(latest_state, ready_env_num, simulation_index, task_id=task_id)

        latent_state_list = []
        if not self.continuous_action_space:
            token = action.reshape(-1, 1)
        else:
            token = action.reshape(-1, self.config.action_space_size_list[task_id])

        # ======= Print statistics for debugging =============
        # min_size = min(self.keys_values_wm_size_list)
        # if min_size >= self.config.max_tokens - 5:
        #     self.length_largethan_maxminus5_context_cnt += len(self.keys_values_wm_size_list)
        # if min_size >= self.config.max_tokens - 7:
        #     self.length_largethan_maxminus7_context_cnt += len(self.keys_values_wm_size_list)
        # if self.total_query_count > 0 and self.total_query_count % 10000 == 0:
        #     self.hit_freq = self.hit_count / self.total_query_count
        #     print('total_query_count:', self.total_query_count)
        #     length_largethan_maxminus5_context_cnt_ratio = self.length_largethan_maxminus5_context_cnt / self.total_query_count
        #     print('recurrent largethan_maxminus5_context:', self.length_largethan_maxminus5_context_cnt)
        #     print('recurrent largethan_maxminus5_context_ratio:', length_largethan_maxminus5_context_cnt_ratio)
        #     length_largethan_maxminus7_context_cnt_ratio = self.length_largethan_maxminus7_context_cnt / self.total_query_count
        #     print('recurrent largethan_maxminus7_context_ratio:', length_largethan_maxminus7_context_cnt_ratio)
        #     print('recurrent largethan_maxminus7_context:', self.length_largethan_maxminus7_context_cnt)

        # Trim and pad kv_cache
        self.keys_values_wm_size_list = self.trim_and_pad_kv_cache(is_init_infer=False)
        self.keys_values_wm_size_list_current = self.keys_values_wm_size_list

        for k in range(2):
            # action_token obs_token
            if k == 0:
                obs_embeddings_or_act_tokens = {'act_tokens': token}
            else:
                obs_embeddings_or_act_tokens = {'obs_embeddings': token}

            # Perform forward pass
            outputs_wm = self.forward(
                obs_embeddings_or_act_tokens,
                past_keys_values=self.keys_values_wm,
                kvcache_independent=False,
                is_init_infer=False,
                task_id = task_id
            )

            self.keys_values_wm_size_list_current = [i + 1 for i in self.keys_values_wm_size_list_current]

            if k == 0:
                reward = outputs_wm.logits_rewards  # (B,)

            if k < self.num_observations_tokens:
                token = outputs_wm.logits_observations
                if len(token.shape) != 3:
                    token = token.unsqueeze(1)  # (8,1024) -> (8,1,1024)
                # print(f'token.shape:{token.shape}')

                latent_state_list.append(token)

        del self.latent_state  # Very important to minimize cuda memory usage
        self.latent_state = torch.cat(latent_state_list, dim=1)  # (B, K)

        self.update_cache_context(
            self.latent_state,
            is_init_infer=False,
            simulation_index=simulation_index,
            latent_state_index_in_search_path=latent_state_index_in_search_path
        )

        return (outputs_wm.output_sequence, self.latent_state, reward, outputs_wm.logits_policy, outputs_wm.logits_value)

    def trim_and_pad_kv_cache(self, is_init_infer=True) -> list:
        """
        Adjusts the key-value cache for each environment to ensure they all have the same size.

        In a multi-environment setting, the key-value cache (kv_cache) for each environment is stored separately.
        During recurrent inference, the kv_cache sizes may vary across environments. This method pads each kv_cache
        to match the largest size found among them, facilitating batch processing in the transformer forward pass.

        Arguments:
            - is_init_infer (:obj:`bool`): Indicates if this is an initial inference. Default is True.
        Returns:
            - list: Updated sizes of the key-value caches.
        """
        # Find the maximum size among all key-value caches
        max_size = max(self.keys_values_wm_size_list)

        # Iterate over each layer of the transformer
        for layer in range(self.num_layers):
            kv_cache_k_list = []
            kv_cache_v_list = []

            # Enumerate through each environment's key-value pairs
            for idx, keys_values in enumerate(self.keys_values_wm_list):
                k_cache = keys_values[layer]._k_cache._cache
                v_cache = keys_values[layer]._v_cache._cache

                effective_size = self.keys_values_wm_size_list[idx]
                pad_size = max_size - effective_size

                # If padding is required, trim the end and pad the beginning of the cache
                if pad_size > 0:
                    k_cache_trimmed = k_cache[:, :, :-pad_size, :]
                    v_cache_trimmed = v_cache[:, :, :-pad_size, :]
                    k_cache_padded = F.pad(k_cache_trimmed, (0, 0, pad_size, 0), "constant", 0)
                    v_cache_padded = F.pad(v_cache_trimmed, (0, 0, pad_size, 0), "constant", 0)
                else:
                    k_cache_padded = k_cache
                    v_cache_padded = v_cache

                kv_cache_k_list.append(k_cache_padded)
                kv_cache_v_list.append(v_cache_padded)

            # Stack the caches along a new dimension and remove any extra dimensions
            self.keys_values_wm._keys_values[layer]._k_cache._cache = torch.stack(kv_cache_k_list, dim=0).squeeze(1)
            self.keys_values_wm._keys_values[layer]._v_cache._cache = torch.stack(kv_cache_v_list, dim=0).squeeze(1)

            # Update the cache size to the maximum size
            self.keys_values_wm._keys_values[layer]._k_cache._size = max_size
            self.keys_values_wm._keys_values[layer]._v_cache._size = max_size

        return self.keys_values_wm_size_list

    #@profile
    def update_cache_context(self, latent_state, is_init_infer=True, simulation_index=0,
                             latent_state_index_in_search_path=[], valid_context_lengths=None):
        """
        Update the cache context with the given latent state.

        Arguments:
            - latent_state (:obj:`torch.Tensor`): The latent state tensor.
            - is_init_infer (:obj:`bool`): Flag to indicate if this is the initial inference.
            - simulation_index (:obj:`int`): Index of the simulation.
            - latent_state_index_in_search_path (:obj:`list`): List of indices in the search path.
            - valid_context_lengths (:obj:`list`): List of valid context lengths.
        """
        if self.context_length <= 2:
            # No context to update if the context length is less than or equal to 2.
            return
        for i in range(latent_state.size(0)):
            # ============ Iterate over each environment ============
            cache_key = hash_state(latent_state[i].view(-1).cpu().numpy())  # latent_state[i] is torch.Tensor
            
            context_length = self.context_length

            if not is_init_infer:
                # ============ Internal Node ============
                # Retrieve KV from global KV cache self.keys_values_wm to single environment KV cache self.keys_values_wm_single_env, ensuring correct positional encoding
                current_max_context_length = max(self.keys_values_wm_size_list_current)
                trim_size = current_max_context_length - self.keys_values_wm_size_list_current[i]
                for layer in range(self.num_layers):
                    # ============ Apply trimming and padding to each layer of kv_cache ============
                    # cache shape [batch_size, num_heads, sequence_length, features]
                    k_cache_current = self.keys_values_wm._keys_values[layer]._k_cache._cache[i]
                    v_cache_current = self.keys_values_wm._keys_values[layer]._v_cache._cache[i]

                    if trim_size > 0:
                        # Trim invalid leading zeros as per effective length
                        # Remove the first trim_size zero kv items
                        k_cache_trimmed = k_cache_current[:, trim_size:, :]
                        v_cache_trimmed = v_cache_current[:, trim_size:, :]
                        # If effective length < current_max_context_length, pad the end of cache with 'trim_size' zeros
                        k_cache_padded = F.pad(k_cache_trimmed, (0, 0, 0, trim_size), "constant",
                                               0)  # Pad with 'trim_size' zeros at end of cache
                        v_cache_padded = F.pad(v_cache_trimmed, (0, 0, 0, trim_size), "constant", 0)
                    else:
                        k_cache_padded = k_cache_current
                        v_cache_padded = v_cache_current

                    # Update cache of self.keys_values_wm_single_env
                    self.keys_values_wm_single_env._keys_values[layer]._k_cache._cache = k_cache_padded.unsqueeze(0)
                    self.keys_values_wm_single_env._keys_values[layer]._v_cache._cache = v_cache_padded.unsqueeze(0)
                    # Update size of self.keys_values_wm_single_env
                    self.keys_values_wm_single_env._keys_values[layer]._k_cache._size = \
                        self.keys_values_wm_size_list_current[i]
                    self.keys_values_wm_single_env._keys_values[layer]._v_cache._size = \
                        self.keys_values_wm_size_list_current[i]

                    # ============ NOTE: Very Important ============
                    if self.keys_values_wm_single_env._keys_values[layer]._k_cache._size >= context_length - 1:
                        # import ipdb; ipdb.set_trace()

                        # Keep only the last self.context_length-3 timesteps of context
                        # For memory environments, training is for H steps, recurrent_inference might exceed H steps
                        # Assuming cache dimension is [batch_size, num_heads, sequence_length, features]
                        k_cache_current = self.keys_values_wm_single_env._keys_values[layer]._k_cache._cache
                        v_cache_current = self.keys_values_wm_single_env._keys_values[layer]._v_cache._cache

                        # Remove the first 2 steps, keep the last self.context_length-3 steps
                        k_cache_trimmed = k_cache_current[:, :, 2:context_length - 1, :].squeeze(0)
                        v_cache_trimmed = v_cache_current[:, :, 2:context_length - 1, :].squeeze(0)

                        # Index pre-computed positional encoding differences
                        # import ipdb; ipdb.set_trace()
                        pos_emb_diff_k = self.pos_emb_diff_k[layer][(2, context_length - 1)]
                        pos_emb_diff_v = self.pos_emb_diff_v[layer][(2, context_length - 1)]
                        # ============ NOTE: Very Important ============
                        # Apply positional encoding correction to k and v
                        k_cache_trimmed += pos_emb_diff_k.squeeze(0)
                        v_cache_trimmed += pos_emb_diff_v.squeeze(0)

                        # Pad the last 3 steps along the third dimension with zeros
                        # F.pad parameters (0, 0, 0, 3) specify padding amounts for each dimension: (left, right, top, bottom). For 3D tensor, they correspond to (dim2 left, dim2 right, dim1 left, dim1 right).
                        padding_size = (0, 0, 0, 3)
                        k_cache_padded = F.pad(k_cache_trimmed, padding_size, 'constant', 0)
                        v_cache_padded = F.pad(v_cache_trimmed, padding_size, 'constant', 0)
                        # Update single environment cache
                        self.keys_values_wm_single_env._keys_values[layer]._k_cache._cache = k_cache_padded.unsqueeze(0)
                        self.keys_values_wm_single_env._keys_values[layer]._v_cache._cache = v_cache_padded.unsqueeze(0)

                        self.keys_values_wm_single_env._keys_values[layer]._k_cache._size = context_length - 3
                        self.keys_values_wm_single_env._keys_values[layer]._v_cache._size = context_length - 3

            else:
                # ============ Root Node ============
                # Retrieve KV from global KV cache self.keys_values_wm to single environment KV cache self.keys_values_wm_single_env, ensuring correct positional encoding

                for layer in range(self.num_layers):
                    # ============ Apply trimming and padding to each layer of kv_cache ============

                    if self.keys_values_wm._keys_values[layer]._k_cache._size < context_length - 1:  # Keep only the last self.context_length-1 timesteps of context
                        self.keys_values_wm_single_env._keys_values[layer]._k_cache._cache = self.keys_values_wm._keys_values[layer]._k_cache._cache[i].unsqueeze(0)  # Shape torch.Size([2, 100, 512])
                        self.keys_values_wm_single_env._keys_values[layer]._v_cache._cache = self.keys_values_wm._keys_values[layer]._v_cache._cache[i].unsqueeze(0)
                        self.keys_values_wm_single_env._keys_values[layer]._k_cache._size = self.keys_values_wm._keys_values[layer]._k_cache._size
                        self.keys_values_wm_single_env._keys_values[layer]._v_cache._size = self.keys_values_wm._keys_values[layer]._v_cache._size
                    else:
                        # import ipdb; ipdb.set_trace()

                        # Assuming cache dimension is [batch_size, num_heads, sequence_length, features]
                        k_cache_current = self.keys_values_wm._keys_values[layer]._k_cache._cache[i]
                        v_cache_current = self.keys_values_wm._keys_values[layer]._v_cache._cache[i]

                        # Remove the first 2 steps, keep the last self.context_length-3 steps
                        k_cache_trimmed = k_cache_current[:, 2:context_length - 1, :]
                        v_cache_trimmed = v_cache_current[:, 2:context_length - 1, :]

                        # Index pre-computed positional encoding differences
                        pos_emb_diff_k = self.pos_emb_diff_k[layer][(2, context_length - 1)]
                        pos_emb_diff_v = self.pos_emb_diff_v[layer][(2, context_length - 1)]
                        # ============ NOTE: Very Important ============
                        # Apply positional encoding correction to k and v
                        k_cache_trimmed += pos_emb_diff_k.squeeze(0)
                        v_cache_trimmed += pos_emb_diff_v.squeeze(0)

                        # Pad the last 3 steps along the third dimension with zeros
                        # F.pad parameters (0, 0, 0, 3) specify padding amounts for each dimension: (left, right, top, bottom). For 3D tensor, they correspond to (dim2 left, dim2 right, dim1 left, dim1 right).
                        padding_size = (0, 0, 0, 3)
                        k_cache_padded = F.pad(k_cache_trimmed, padding_size, 'constant', 0)
                        v_cache_padded = F.pad(v_cache_trimmed, padding_size, 'constant', 0)
                        # Update cache of self.keys_values_wm_single_env
                        self.keys_values_wm_single_env._keys_values[layer]._k_cache._cache = k_cache_padded.unsqueeze(0)
                        self.keys_values_wm_single_env._keys_values[layer]._v_cache._cache = v_cache_padded.unsqueeze(0)
                        # Update size of self.keys_values_wm_single_env
                        self.keys_values_wm_single_env._keys_values[layer]._k_cache._size = context_length - 3
                        self.keys_values_wm_single_env._keys_values[layer]._v_cache._size = context_length - 3

            # ORIGNAL
            # if is_init_infer:
            #     # Store the latest key-value cache for initial inference
            #     cache_index = self.custom_copy_kv_cache_to_shared_init_envs(self.keys_values_wm_single_env, i)
            #     self.past_kv_cache_init_infer_envs[i][cache_key] = cache_index
            # else:
            #     # Store the latest key-value cache for recurrent inference
            #     cache_index = self.custom_copy_kv_cache_to_shared_recur(self.keys_values_wm_single_env)
            #     self.past_kv_cache_recurrent_infer[cache_key] = cache_index


            if is_init_infer:
                # TODO
                # ==================== 主动淘汰修复逻辑 ====================
                # 1. 获取即将被覆写的物理索引
                index_to_write = self.shared_pool_index_init_envs[i]
                # 2. 使用辅助列表查找该索引上存储的旧的 key
                old_key_to_evict = self.pool_idx_to_key_map_init_envs[i][index_to_write]
                # 3. 如果存在旧 key，就从主 cache map 中删除它
                if old_key_to_evict is not None:
                    # 确保要删除的键确实存在，避免意外错误
                    if old_key_to_evict in self.past_kv_cache_init_infer_envs[i]:
                        del self.past_kv_cache_init_infer_envs[i][old_key_to_evict]

                # 现在可以安全地写入新数据了
                cache_index = self.custom_copy_kv_cache_to_shared_init_envs(self.keys_values_wm_single_env, i)

                # 4. 在主 cache map 和辅助列表中同时更新新的映射关系
                self.past_kv_cache_init_infer_envs[i][cache_key] = cache_index
                self.pool_idx_to_key_map_init_envs[i][index_to_write] = cache_key
            else:
                # ==================== RECURRENT INFER FIX ====================
                # 1. 获取即将被覆写的物理索引
                index_to_write = self.shared_pool_index
                # 2. 使用辅助列表查找该索引上存储的旧的 key
                old_key_to_evict = self.pool_idx_to_key_map_recur_infer[index_to_write]
                # 3. 如果存在旧 key，就从主 cache map 中删除它
                if old_key_to_evict is not None:
                    if old_key_to_evict in self.past_kv_cache_recurrent_infer:
                        del self.past_kv_cache_recurrent_infer[old_key_to_evict]

                # 4. 现在可以安全地写入新数据了
                cache_index = self.custom_copy_kv_cache_to_shared_recur(self.keys_values_wm_single_env)

                # 5. 在主 cache map 和辅助列表中同时更新新的映射关系
                self.past_kv_cache_recurrent_infer[cache_key] = cache_index
                self.pool_idx_to_key_map_recur_infer[index_to_write] = cache_key

    #@profile
    def retrieve_or_generate_kvcache(self, latent_state: list, ready_env_num: int,
                                     simulation_index: int = 0, task_id = 0) -> list:
        """
        Retrieves or generates key-value caches for each environment based on the latent state.

        For each environment, this method either retrieves a matching cache from the predefined
        caches if available, or generates a new cache if no match is found. The method updates
        the internal lists with these caches and their sizes.

        Arguments:
            - latent_state (:obj:`list`): List of latent states for each environment.
            - ready_env_num (:obj:`int`): Number of environments ready for processing.
            - simulation_index (:obj:`int`, optional): Index for simulation tracking. Default is 0.
        Returns:
            - list: Sizes of the key-value caches for each environment.
        """
        for i in range(ready_env_num):
            self.total_query_count += 1
            state_single_env = latent_state[i]  # latent_state[i] is np.array
            cache_key = hash_state(state_single_env)

            if self.reanalyze_phase:
                # TODO: check if this is correct
                matched_value = None
            else:
                # Try to retrieve the cached value from past_kv_cache_init_infer_envs
                cache_index = self.past_kv_cache_init_infer_envs[i].get(cache_key)
                if cache_index is not None:
                    matched_value = self.shared_pool_init_infer[i][cache_index]
                else:
                    matched_value = None

                # If not found, try to retrieve from past_kv_cache_recurrent_infer
                # if matched_value is None:
                #     matched_value = self.shared_pool_recur_infer[self.past_kv_cache_recurrent_infer.get(cache_key)]

                # ==================== TODO ====================
                # 步骤 2: 仅当在 init_infer 中未找到时，才尝试从 recurrent_infer 缓存中查找
                if matched_value is None:
                    # 2.1 安全地从字典中获取索引，它可能返回 None
                    recur_cache_index = self.past_kv_cache_recurrent_infer.get(cache_key)
                    # 2.2 只有在索引有效（不是 None）的情况下，才使用它来从物理池中检索值
                    if recur_cache_index is not None:
                        matched_value = self.shared_pool_recur_infer[recur_cache_index]

                    if recur_cache_index is None:
                        print(f"[CACHE MISS]  Not found for key={cache_key} in recurrent infer. Generating new cache.")

            if matched_value is not None:
                # If a matching cache is found, add it to the lists
                self.hit_count += 1
                # Perform a deep copy because the transformer's forward pass might modify matched_value in-place
                self.keys_values_wm_list.append(self.custom_copy_kv_cache_to_shared_wm(matched_value))
                self.keys_values_wm_size_list.append(matched_value.size)
            else:
                # If no matching cache is found, generate a new one using zero reset
                self.keys_values_wm_single_env = self.transformer.generate_empty_keys_values(
                    n=1, max_tokens=self.context_length
                )
                self.forward(
                    {'obs_embeddings': torch.from_numpy(state_single_env).unsqueeze(0).to(self.device)},
                    past_keys_values=self.keys_values_wm_single_env, is_init_infer=True, task_id=task_id
                    )
                self.keys_values_wm_list.append(self.keys_values_wm_single_env)
                self.keys_values_wm_size_list.append(1)

        return self.keys_values_wm_size_list

    def plot_embeddings(
        self,
        tsne_results: np.ndarray,
        task_ids: np.ndarray,
        observations: Union[np.ndarray, torch.Tensor],
        samples_per_task: int = 5,
        save_dir: str = 'tsne_plots_26games'
    ) -> None:
        """
        Overview:
            Generates a t-SNE visualization plot and annotates it with a specified number of
            randomly selected observation images for each task.

        Arguments:
            - tsne_results (:obj:`np.ndarray`): The t-SNE dimensionality reduction results (N x 2 array).
            - task_ids (:obj:`np.ndarray`): An array of environment task IDs, used for coloring the points (N array).
            - observations (:obj:`Union[np.ndarray, torch.Tensor]`): The corresponding observation samples (N x C x H x W tensor or array).
            - samples_per_task (:obj:`int`): The number of samples to select for image annotation per task. Defaults to 5.
            - save_dir (:obj:`str`): The directory path where the plot will be saved. Defaults to 'tsne_plots_26games'.
        """
        # Create the save directory if it doesn't exist.
        os.makedirs(save_dir, exist_ok=True)
        print(f"[INFO] Save directory created or already exists: {save_dir}")

        # Create the t-SNE plot.
        print("[INFO] Starting to draw the t-SNE scatter plot...")
        plt.figure(figsize=(18, 10))  # Increase figure width to accommodate the legend on the right.

        # Scatter plot of the t-SNE results.
        scatter = plt.scatter(
            tsne_results[:, 0],
            tsne_results[:, 1],
            c=[self.colors[tid] for tid in task_ids],
            alpha=0.6,
            edgecolor='w',
            linewidth=0.5
        )

        # Create a custom legend for the tasks.
        legend_elements = []
        for idx, env_id in enumerate(self.env_id_list):
            short_name = self.env_short_names.get(env_id, env_id)
            color = self.colors[idx]
            legend_elements.append(
                Patch(facecolor=color, edgecolor='w', label=f"{idx}: {short_name}")
            )
        
        # Place the legend on the right side of the plot, with each item on a new line.
        plt.legend(
            handles=legend_elements,
            title="Environment IDs",
            loc='center left',
            bbox_to_anchor=(1, 0.5),  # Position the legend in the center-right of the plot area.
            fontsize=10,
            title_fontsize=12,
            ncol=1,
            frameon=False  # Remove the legend border for a cleaner look.
        )

        # Set the title and axis labels.
        plt.title("t-SNE of Latent States across Environments", fontsize=16)
        plt.xlabel("t-SNE Dimension 1", fontsize=14)
        plt.ylabel("t-SNE Dimension 2", fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)
        print(f"[INFO] t-SNE scatter plot completed with {len(tsne_results)} points.")

        # Select a specified number of samples per task for image annotation.
        print(f"[INFO] Starting to select {samples_per_task} samples per task for image annotation...")
        for task_id in range(len(self.env_id_list)):
            # Find all indices for the current task.
            task_indices = np.where(task_ids == task_id)[0]
            if len(task_indices) == 0:
                print(f"[WARNING] No samples found for task ID {task_id}.")
                continue
            
            # If the number of samples is less than required, select all of them.
            if len(task_indices) < samples_per_task:
                selected_indices = task_indices
                print(f"[INFO] Task ID {task_id} has fewer samples ({len(task_indices)}) than required ({samples_per_task}). Selecting all.")
            else:
                selected_indices = np.random.choice(task_indices, size=samples_per_task, replace=False)
                print(f"[INFO] Randomly selecting {samples_per_task} samples for task ID {task_id} for annotation.")

            for idx in selected_indices:
                img = observations[idx]
                if isinstance(img, torch.Tensor):
                    img = img.cpu().numpy()
                
                # Handle channel-first (C, H, W) format for grayscale or RGB images.
                if img.shape[0] == 1 or img.shape[0] == 3:
                    img = np.transpose(img, (1, 2, 0))
                else:
                    raise ValueError(f"Unsupported image shape: {img.shape}")
        
                # Normalize the image to the [0, 1] range for correct display.
                img_min, img_max = img.min(), img.max()
                if img_max - img_min > 1e-5:
                    img = (img - img_min) / (img_max - img_min)
                else:
                    img = np.zeros_like(img)
        
                imagebox = OffsetImage(img, zoom=0.5)
                ab = AnnotationBbox(
                    imagebox,
                    (tsne_results[idx, 0], tsne_results[idx, 1]),
                    frameon=False,
                    pad=0.3
                )
                plt.gca().add_artist(ab)
                print(f"[INFO] Added image annotation: Task ID {task_id}, point index {idx}, t-SNE coords ({tsne_results[idx, 0]:.2f}, {tsne_results[idx, 1]:.2f})")

        # Adjust layout to prevent the legend from being cut off.
        plt.tight_layout(rect=[0, 0, 0.9, 1])  # Reserve space for the legend on the right.

        # Save the figure in both PNG and PDF formats with high resolution.
        save_path_png = os.path.join(save_dir, 'tsne_plot.png')
        save_path_pdf = os.path.join(save_dir, 'tsne_plot.pdf')
        plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
        plt.savefig(save_path_pdf, dpi=300, bbox_inches='tight')
        print(f"[INFO] t-SNE visualization plot saved to: {save_path_png} and {save_path_pdf}")
        plt.close()

    @torch.no_grad()
    def gather_and_plot(
        self,
        local_embeddings: torch.Tensor,
        local_task_ids: torch.Tensor,
        local_observations: torch.Tensor
    ) -> None:
        """
        Overview:
            Gathers embeddings, task IDs, and observations from all distributed processes.
            On the main process (rank 0), it performs t-SNE and plots the results.

        Arguments:
            - local_embeddings (:obj:`torch.Tensor`): The embedding tensor from the current process.
            - local_task_ids (:obj:`torch.Tensor`): The task ID tensor from the current process.
            - local_observations (:obj:`torch.Tensor`): The observation tensor from the current process.
        """
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        
        # Prepare lists to receive CUDA tensors from all processes.
        embeddings_list = [torch.zeros_like(local_embeddings) for _ in range(world_size)]
        task_ids_list = [torch.zeros_like(local_task_ids) for _ in range(world_size)]
        
        # Prepare a list to receive CPU objects (observations) from all processes.
        observations_list = [None for _ in range(world_size)]

        try:
            # Gather CUDA tensors: embeddings and task_ids.
            dist.all_gather(embeddings_list, local_embeddings)
            dist.all_gather(task_ids_list, local_task_ids)
            
            # Gather CPU objects: observations (must be moved to CPU and converted first).
            local_observations_cpu = local_observations.cpu().numpy().tolist()
            dist.all_gather_object(observations_list, local_observations_cpu)
        except RuntimeError as e:
            print(f"Rank {rank}: all_gather failed with error: {e}")
            return
        
        if rank == 0:
            # Concatenate all embeddings and task_ids on the main process.
            all_embeddings = torch.cat(embeddings_list, dim=0).cpu().numpy()
            all_task_ids = torch.cat(task_ids_list, dim=0).cpu().numpy()
            
            # Concatenate all observations.
            all_observations_list = []
            for obs in observations_list:
                all_observations_list.extend(obs)
            all_observations = np.array(all_observations_list)

            print(f"Shape of all_embeddings: {all_embeddings.shape}")
            all_embeddings = all_embeddings.reshape(-1, all_embeddings.shape[-1])
            print(f"Shape of all_observations: {all_observations.shape}")
            all_observations = all_observations.reshape(-1,  *all_observations.shape[-3:])
        
            # Perform t-SNE dimensionality reduction.
            tsne = TSNE(n_components=2, random_state=42)
            tsne_results = tsne.fit_transform(all_embeddings)
        
            # Plot and save the resulting image.
            self.plot_embeddings(tsne_results, all_task_ids, all_observations, save_dir=f'tsne_plots_{self.num_tasks}games')

    #@profile
    def compute_loss(self, batch, target_tokenizer: Tokenizer = None, inverse_scalar_transform_handle=None, task_id = 0, **kwargs: Any) -> LossWithIntermediateLosses:
        # Encode observations into latent state representations
        obs_embeddings = self.tokenizer.encode_to_obs_embeddings(batch['observations'], task_id=task_id)

        if self.analysis_tsne:
            # =========== tsne analysis ===========
            if not obs_embeddings.is_cuda:
                obs_embeddings = obs_embeddings.cuda()
            obs_embeddings = obs_embeddings.contiguous()
            local_embeddings = obs_embeddings.detach()
            local_task_ids = torch.full((local_embeddings.size(0),), task_id, dtype=torch.long, device=local_embeddings.device)
            local_observations = batch['observations'].detach().cpu()
            self.gather_and_plot(local_embeddings, local_task_ids, local_observations)
            
        # ========= logging for analysis =========
        if self.analysis_dormant_ratio_weight_rank:
            self._analysis_step_counter += 1
            self.do_analysis = (
                self.analysis_dormant_ratio_weight_rank          # 总开关
                and self._analysis_step_counter % self.analysis_dormant_ratio_interval == 0
            )

        # ========= logging for analysis =========
        if self.do_analysis:
            # Calculate dormant ratio of the encoder
            shape = batch['observations'].shape  # (..., C, H, W)
            inputs = batch['observations'].contiguous().view(-1, *shape[-3:])  # (32,5,3,64,64) -> (160,3,64,64)
            if self.continuous_action_space:
                encoder_index = task_id
            else:
                encoder_index = 0
            dormant_ratio_encoder_dict = calculate_dormant_ratio(self.tokenizer.encoder[encoder_index], inputs.detach(),
                                                    dormant_threshold=self.dormant_threshold)

            dormant_ratio_encoder = dormant_ratio_encoder_dict['global']

            avg_weight_mag_encoder = compute_average_weight_magnitude(self.tokenizer.encoder[encoder_index])
            avg_weight_mag_transformer = compute_average_weight_magnitude(self.transformer)
            avg_weight_mag_head = compute_average_weight_magnitude(self.head_dict)

            e_rank_last_linear = calculate_effective_rank(self.tokenizer.encoder[encoder_index], inputs, representation_layer_name="last_linear")
            try:
                e_rank_sim_norm = calculate_effective_rank(self.tokenizer.encoder[encoder_index], inputs, representation_layer_name="final_norm")
            except Exception as e:
                e_rank_sim_norm = torch.tensor(0.)
                
            for kv_cache_dict_env in self.past_kv_cache_init_infer_envs:
                kv_cache_dict_env.clear()
            self.past_kv_cache_recurrent_infer.clear()
            self.keys_values_wm_list.clear()
            torch.cuda.empty_cache()
        else:
            dormant_ratio_encoder = torch.tensor(0.)
            avg_weight_mag_encoder = torch.tensor(0.)
            avg_weight_mag_transformer = torch.tensor(0.)
            avg_weight_mag_head = torch.tensor(0.)
            e_rank_last_linear = torch.tensor(0.)
            e_rank_sim_norm = torch.tensor(0.)
            # dormant_ratio_encoder   = None


        # Calculate the L2 norm of the latent state roots
        latent_state_l2_norms = torch.norm(obs_embeddings, p=2, dim=2).mean()

        if self.obs_type == 'image':
            # Reconstruct observations from latent state representations
            # reconstructed_images = self.tokenizer.decode_to_obs(obs_embeddings)

            #  ========== for visualization ==========
            # Uncomment the lines below for visual analysis
            # original_images, reconstructed_images = batch['observations'], reconstructed_images
            # target_policy = batch['target_policy']
            # ==== for value priority ====
            # target_predict_value = inverse_scalar_transform_handle(batch['target_value'].reshape(-1, 101)).reshape(
            #     batch['observations'].shape[0], batch['observations'].shape[1], 1)
            # true_rewards = inverse_scalar_transform_handle(batch['rewards'].reshape(-1, 101)).reshape(
            #     batch['observations'].shape[0], batch['observations'].shape[1], 1)
            #  ========== for visualization ==========

            # Calculate reconstruction loss and perceptual loss
            # latent_recon_loss = self.tokenizer.reconstruction_loss(batch['observations'].reshape(-1, 3, 64, 64), reconstructed_images) # NOTE: for stack=1
            # perceptual_loss = self.tokenizer.perceptual_loss(batch['observations'].reshape(-1, 3, 64, 64), reconstructed_images) # NOTE: for stack=1
            latent_recon_loss = torch.tensor(0., device=batch['observations'].device,
                                             dtype=batch['observations'].dtype)
            perceptual_loss = torch.tensor(0., device=batch['observations'].device,
                                           dtype=batch['observations'].dtype)

        elif self.obs_type == 'vector':
            perceptual_loss = torch.tensor(0., device=batch['observations'].device,
                                           dtype=batch['observations'].dtype)

            # Reconstruct observations from latent state representations
            # reconstructed_images = self.tokenizer.decode_to_obs(obs_embeddings.reshape(-1, self.embed_dim))
            # # Calculate reconstruction loss
            # latent_recon_loss = self.tokenizer.reconstruction_loss(batch['observations'].reshape(-1, 25),
            #                                                        reconstructed_images)
            latent_recon_loss = torch.tensor(0., device=batch['observations'].device,
                                             dtype=batch['observations'].dtype)

        elif self.obs_type == 'image_memory':
            # Reconstruct observations from latent state representations
            # reconstructed_images = self.tokenizer.decode_to_obs(obs_embeddings)
            # original_images, reconstructed_images = batch['observations'], reconstructed_images

            #  ========== for visualization ==========
            # Uncomment the lines below for visual analysis
            # target_policy = batch['target_policy']
            # target_predict_value = inverse_scalar_transform_handle(batch['target_value'].reshape(-1, 101)).reshape(
            #     batch['observations'].shape[0], batch['observations'].shape[1], 1)
            # true_rewards = inverse_scalar_transform_handle(batch['rewards'].reshape(-1, 101)).reshape(
            #     batch['observations'].shape[0], batch['observations'].shape[1], 1)
            #  ========== for visualization ==========

            # Calculate reconstruction loss and perceptual loss
            # latent_recon_loss = self.tokenizer.reconstruction_loss(batch['observations'].reshape(-1, 3, 5, 5),
            #                                                        reconstructed_images)

            latent_recon_loss = torch.tensor(0., device=batch['observations'].device,
                                             dtype=batch['observations'].dtype)
            perceptual_loss = torch.tensor(0., device=batch['observations'].device,
                                           dtype=batch['observations'].dtype)

        # Action tokens
        if self.continuous_action_space:
            act_tokens = batch['actions']
        else:
            act_tokens = rearrange(batch['actions'], 'b l -> b l 1')

        # Forward pass to obtain predictions for observations, rewards, and policies
        outputs = self.forward({'obs_embeddings_and_act_tokens': (obs_embeddings, act_tokens)}, task_id=task_id)

        if self.config.use_priority:
            # ==================== START MODIFICATION 5 ====================
            # Calculate value_priority, similar to MuZero.
            with torch.no_grad():
                # 1. Get the predicted value logits for the first step of the sequence (t=0).
                # The shape is (B, support_size).
                predicted_value_logits_step0 = outputs.logits_value[:, 0, :]

                # 2. Convert the categorical prediction to a scalar value.
                # The shape becomes (B, 1).
                predicted_scalar_value_step0 = inverse_scalar_transform_handle(predicted_value_logits_step0)

                # 3. Get the target scalar value for the first step from the batch.
                # The shape is (B, num_unroll_steps), so we take the first column.
                target_scalar_value_step0 = batch['scalar_target_value'][:, 0]

                # 4. Calculate the L1 loss (absolute difference) between prediction and target.
                # This is the priority. We use reduction='none' to get per-sample priorities.
                value_priority = F.l1_loss(predicted_scalar_value_step0.squeeze(-1), target_scalar_value_step0, reduction='none')
            # ===================== END MODIFICATION 5 =====================
        else:
            value_priority = torch.tensor(0.)

        # ========= logging for analysis =========
        # if self.analysis_dormant_ratio_weight_rank:
        if self.do_analysis:
            # Calculate dormant ratio of the world model
            dormant_ratio_world_model = calculate_dormant_ratio(self, {
                'obs_embeddings_and_act_tokens': (obs_embeddings.detach(), act_tokens.detach())},
                                                          dormant_threshold=self.dormant_threshold)
            dormant_ratio_transformer = dormant_ratio_world_model['transformer']
            dormant_ratio_head = dormant_ratio_world_model['head']
            for kv_cache_dict_env in self.past_kv_cache_init_infer_envs:
                kv_cache_dict_env.clear()
            self.past_kv_cache_recurrent_infer.clear()
            self.keys_values_wm_list.clear()
            torch.cuda.empty_cache()
        else:
            dormant_ratio_transformer = torch.tensor(0.)
            dormant_ratio_head = torch.tensor(0.)

        #  ========== for visualization ==========
        # Uncomment the lines below for visualization
        # predict_policy = outputs.logits_policy
        # predict_policy = F.softmax(outputs.logits_policy, dim=-1)
        # predict_value = inverse_scalar_transform_handle(outputs.logits_value.reshape(-1, 101)).reshape(batch['observations'].shape[0], batch['observations'].shape[1], 1)
        # predict_rewards = inverse_scalar_transform_handle(outputs.logits_rewards.reshape(-1, 101)).reshape(batch['observations'].shape[0], batch['observations'].shape[1], 1)
        # import pdb; pdb.set_trace()
        # visualize_reward_value_img_policy(original_images, reconstructed_images, target_predict_value, true_rewards, target_policy, predict_value, predict_rewards, predict_policy, not_plot_timesteps=[], suffix='pong_H10_H4_0613')
        
        # visualize_reward_value_img_policy(original_images, reconstructed_images, target_predict_value, true_rewards, target_policy, predict_value, predict_rewards, predict_policy, not_plot_timesteps=list(np.arange(4,60)), suffix='visual_match_memlen1-60-15/one_success_episode')
        # visualize_reward_value_img_policy(original_images, reconstructed_images, target_predict_value, true_rewards, target_policy, predict_value, predict_rewards, predict_policy, not_plot_timesteps=list(np.arange(4,60)), suffix='visual_match_memlen1-60-15/one_fail_episode')
        #  ========== for visualization ==========

        # For training stability, use target_tokenizer to compute the true next latent state representations
        with torch.no_grad():
            target_obs_embeddings = target_tokenizer.encode_to_obs_embeddings(batch['observations'], task_id=task_id)

        # Compute labels for observations, rewards, and ends
        labels_observations, labels_rewards, labels_ends = self.compute_labels_world_model(target_obs_embeddings,
                                                                                           batch['rewards'],
                                                                                           batch['ends'],
                                                                                           batch['mask_padding'])

        # Reshape the logits and labels for observations
        logits_observations = rearrange(outputs.logits_observations[:, :-1], 'b t o -> (b t) o')
        labels_observations = labels_observations.reshape(-1, self.projection_input_dim)

        if self.use_task_embed and self.task_embed_option == "concat_task_embed":
            # Expand task embeddings to match the sequence shape
            self.task_embeddings = self.task_emb(torch.tensor(task_id, device=self.device))  # NOTE: TODO
            self.task_embeddings = self.sim_norm(self.task_embeddings.view(1,-1)).view(-1) # TODO
            task_emb_expanded = self.task_embeddings.expand(labels_observations.shape[0], -1)
            labels_observations = torch.cat([labels_observations, task_emb_expanded.detach()], dim=-1) # NOTE: detach()

        # Compute prediction loss for observations. Options: MSE and Group KL
        if self.predict_latent_loss_type == 'mse':
            # MSE loss, directly compare logits and labels
            loss_obs = torch.nn.functional.mse_loss(logits_observations, labels_observations, reduction='none').mean(
                -1)
        elif self.predict_latent_loss_type == 'group_kl':
            # Group KL loss, group features and calculate KL divergence within each group
            batch_size, num_features = logits_observations.shape
            epsilon = 1e-6
            logits_reshaped = logits_observations.reshape(batch_size, self.num_groups, self.group_size) + epsilon
            labels_reshaped = labels_observations.reshape(batch_size, self.num_groups, self.group_size) + epsilon

            loss_obs = F.kl_div(logits_reshaped.log(), labels_reshaped, reduction='none').sum(dim=-1).mean(dim=-1)

            #  ========== for debugging ==========
            # assert not torch.isnan(logits_reshaped).any(), "logits_reshaped contains NaN values"
            # assert not torch.isnan(labels_reshaped).any(), "labels_reshaped contains NaN values"
            # print('loss_obs:', loss_obs.mean())
            # for name, param in self.tokenizer.encoder.named_parameters():
            #     print('name, param.mean(), param.std():', name, param.mean(), param.std())
            # logits_grad = torch.autograd.grad(loss_obs.mean(), logits_observations, retain_graph=True)[0]
            # print(f"logits_grad (min, max, mean): {logits_grad.min()}, {logits_grad.max()}, {logits_grad.mean()}")

        # Apply mask to loss_obs
        mask_padding_expanded = batch['mask_padding'][:, 1:].contiguous().view(-1)
        loss_obs = (loss_obs * mask_padding_expanded)

        # Compute labels for policy and value
        labels_policy, labels_value = self.compute_labels_world_model_value_policy(batch['target_value'],
                                                                                   batch['target_policy'],
                                                                                   batch['mask_padding'])

        # Compute losses for rewards, policy, and value
        loss_rewards = self.compute_cross_entropy_loss(outputs, labels_rewards, batch, element='rewards')

        if not self.continuous_action_space:
            loss_policy, orig_policy_loss, policy_entropy = self.compute_cross_entropy_loss(outputs, labels_policy,
                                                                                            batch,
                                                                                            element='policy')
        else:
            # NOTE: for continuous action space
            if self.config.policy_loss_type == 'simple':
                orig_policy_loss, policy_entropy_loss, target_policy_entropy, target_sampled_actions, mu, sigma = self._calculate_policy_loss_cont_simple(
                    outputs, batch)
            else:
                orig_policy_loss, policy_entropy_loss, target_policy_entropy, target_sampled_actions, mu, sigma = self._calculate_policy_loss_cont(
                    outputs, batch, task_id=task_id)

            loss_policy = orig_policy_loss + self.policy_entropy_weight * policy_entropy_loss
            policy_entropy = - policy_entropy_loss

        loss_value = self.compute_cross_entropy_loss(outputs, labels_value, batch, element='value')

        # Compute timesteps
        timesteps = torch.arange(batch['actions'].shape[1], device=batch['actions'].device)
        # Compute discount coefficients for each timestep
        discounts = self.gamma ** timesteps

        if batch['mask_padding'].sum() == 0:
            assert False, "mask_padding is all zeros"

        # Group losses into first step, middle step, and last step
        first_step_losses = {}
        middle_step_losses = {}
        last_step_losses = {}
        # batch['mask_padding'] indicates mask status for future H steps, exclude masked losses to maintain accurate mean statistics
        # Group losses for each loss item
        for loss_name, loss_tmp in zip(
                ['loss_obs', 'loss_rewards', 'loss_value', 'loss_policy', 'orig_policy_loss', 'policy_entropy'],
                [loss_obs, loss_rewards, loss_value, loss_policy, orig_policy_loss, policy_entropy]
        ):
            if loss_name == 'loss_obs':
                seq_len = batch['actions'].shape[1] - 1
                # Get the corresponding mask_padding
                mask_padding = batch['mask_padding'][:, 1:seq_len]
            else:
                seq_len = batch['actions'].shape[1]
                # Get the corresponding mask_padding
                mask_padding = batch['mask_padding'][:, :seq_len]

            # Adjust loss shape to (batch_size, seq_len)
            loss_tmp = loss_tmp.view(-1, seq_len)

            # First step loss
            first_step_mask = mask_padding[:, 0]
            first_step_losses[loss_name] = loss_tmp[:, 0][first_step_mask].mean()

            # Middle step loss
            middle_step_index = seq_len // 2
            middle_step_mask = mask_padding[:, middle_step_index]
            middle_step_losses[loss_name] = loss_tmp[:, middle_step_index][middle_step_mask].mean()

            # Last step loss
            last_step_mask = mask_padding[:, -1]
            last_step_losses[loss_name] = loss_tmp[:, -1][last_step_mask].mean()

        # Discount reconstruction loss and perceptual loss
        discounted_latent_recon_loss = latent_recon_loss
        discounted_perceptual_loss = perceptual_loss

        # Calculate overall discounted loss
        discounted_loss_obs = (loss_obs.view(-1, batch['actions'].shape[1] - 1) * discounts[1:]).sum()/ batch['mask_padding'][:,1:].sum()
        discounted_loss_rewards = (loss_rewards.view(-1, batch['actions'].shape[1]) * discounts).sum()/ batch['mask_padding'].sum()
        discounted_loss_value = (loss_value.view(-1, batch['actions'].shape[1]) * discounts).sum()/ batch['mask_padding'].sum()
        discounted_loss_policy = (loss_policy.view(-1, batch['actions'].shape[1]) * discounts).sum()/ batch['mask_padding'].sum()
        discounted_orig_policy_loss = (orig_policy_loss.view(-1, batch['actions'].shape[1]) * discounts).sum()/ batch['mask_padding'].sum()
        discounted_policy_entropy = (policy_entropy.view(-1, batch['actions'].shape[1]) * discounts).sum()/ batch['mask_padding'].sum()

        if self.continuous_action_space:
            return LossWithIntermediateLosses(
                latent_recon_loss_weight=self.latent_recon_loss_weight,
                perceptual_loss_weight=self.perceptual_loss_weight,
                continuous_action_space=True,
                loss_obs=discounted_loss_obs,
                loss_rewards=discounted_loss_rewards,
                loss_value=discounted_loss_value,
                loss_policy=discounted_loss_policy,
                latent_recon_loss=discounted_latent_recon_loss,
                perceptual_loss=discounted_perceptual_loss,
                orig_policy_loss=discounted_orig_policy_loss,
                policy_entropy=discounted_policy_entropy,
                first_step_losses=first_step_losses,
                middle_step_losses=middle_step_losses,
                last_step_losses=last_step_losses,
                dormant_ratio_encoder=dormant_ratio_encoder,
                dormant_ratio_transformer=dormant_ratio_transformer,
                dormant_ratio_head=dormant_ratio_head,
                avg_weight_mag_encoder = avg_weight_mag_encoder,
                avg_weight_mag_transformer = avg_weight_mag_transformer,
                avg_weight_mag_head = avg_weight_mag_head,
                e_rank_last_linear = e_rank_last_linear,
                e_rank_sim_norm = e_rank_sim_norm,
                latent_state_l2_norms=latent_state_l2_norms,
                policy_mu=mu,
                policy_sigma=sigma,
                target_sampled_actions=target_sampled_actions,
                value_priority=value_priority,

            )
        else:
            return LossWithIntermediateLosses(
                latent_recon_loss_weight=self.latent_recon_loss_weight,
                perceptual_loss_weight=self.perceptual_loss_weight,
                continuous_action_space=False,
                loss_obs=discounted_loss_obs,
                loss_rewards=discounted_loss_rewards,
                loss_value=discounted_loss_value,
                loss_policy=discounted_loss_policy,
                latent_recon_loss=discounted_latent_recon_loss,
                perceptual_loss=discounted_perceptual_loss,
                orig_policy_loss=discounted_orig_policy_loss,
                policy_entropy=discounted_policy_entropy,
                first_step_losses=first_step_losses,
                middle_step_losses=middle_step_losses,
                last_step_losses=last_step_losses,
                dormant_ratio_encoder=dormant_ratio_encoder,
                dormant_ratio_transformer=dormant_ratio_transformer,
                dormant_ratio_head=dormant_ratio_head,
                avg_weight_mag_encoder = avg_weight_mag_encoder,
                avg_weight_mag_transformer = avg_weight_mag_transformer,
                avg_weight_mag_head = avg_weight_mag_head,
                e_rank_last_linear = e_rank_last_linear,
                e_rank_sim_norm = e_rank_sim_norm,
                latent_state_l2_norms=latent_state_l2_norms,
                value_priority=value_priority,

            )

    #@profile
    def compute_cross_entropy_loss(self, outputs, labels, batch, element='rewards'):
        # Assume outputs is an object with logits attributes like 'rewards', 'policy', and 'value'.
        # labels is a target tensor for comparison. batch is a dictionary with a mask indicating valid timesteps.

        logits = getattr(outputs, f'logits_{element}')

        # Reshape your tensors
        logits = rearrange(logits, 'b t e -> (b t) e')
        labels = labels.reshape(-1, labels.shape[-1])  # Assume labels initially have shape [batch, time, dim]

        # Reshape your mask. True indicates valid data.
        mask_padding = rearrange(batch['mask_padding'], 'b t -> (b t)')

        # Compute cross-entropy loss
        loss = -(torch.log_softmax(logits, dim=1) * labels).sum(1)
        loss = (loss * mask_padding)

        # if torch.isnan(loss).any():
        #     raise ValueError(f"NaN detected in outputs for batch {batch} and element '{element}'")

        if element == 'policy':
            # Compute policy entropy loss
            policy_entropy = self.compute_policy_entropy_loss(logits, mask_padding)
            # Combine losses with specified weight
            combined_loss = loss - self.policy_entropy_weight * policy_entropy
            return combined_loss, loss, policy_entropy

        return loss

    #@profile
    def compute_policy_entropy_loss(self, logits, mask):
        # Compute entropy of the policy
        probs = torch.softmax(logits, dim=1)
        log_probs = torch.log_softmax(logits, dim=1)
        entropy = -(probs * log_probs).sum(1)
        # Apply mask and return average entropy loss
        entropy_loss = (entropy * mask)
        return entropy_loss

    #@profile
    def compute_labels_world_model(self, obs_embeddings: torch.Tensor, rewards: torch.Tensor, ends: torch.Tensor,
                                   mask_padding: torch.BoolTensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # assert torch.all(ends.sum(dim=1) <= 1)  # Each sequence sample should have at most one 'done' flag
        mask_fill = torch.logical_not(mask_padding)

        # Prepare observation labels
        labels_observations = obs_embeddings.contiguous().view(rewards.shape[0], -1, self.projection_input_dim)[:, 1:]

        # Fill the masked areas of rewards
        mask_fill_rewards = mask_fill.unsqueeze(-1).expand_as(rewards)
        labels_rewards = rewards.masked_fill(mask_fill_rewards, -100)

        # Fill the masked areas of ends
        # labels_ends = ends.masked_fill(mask_fill, -100)

        # return labels_observations, labels_rewards.reshape(-1, self.support_size), labels_ends.reshape(-1)
        return labels_observations, labels_rewards.view(-1, self.support_size), None

    #@profile
    def compute_labels_world_model_value_policy(self, target_value: torch.Tensor, target_policy: torch.Tensor,
                                                mask_padding: torch.BoolTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Compute labels for value and policy predictions. """
        mask_fill = torch.logical_not(mask_padding)

        # Fill the masked areas of policy
        mask_fill_policy = mask_fill.unsqueeze(-1).expand_as(target_policy)
        labels_policy = target_policy.masked_fill(mask_fill_policy, -100)

        # Fill the masked areas of value
        mask_fill_value = mask_fill.unsqueeze(-1).expand_as(target_value)
        labels_value = target_value.masked_fill(mask_fill_value, -100)

        if self.continuous_action_space:
            return None, labels_value.reshape(-1, self.support_size)
        else:
            return labels_policy.reshape(-1, self.action_space_size), labels_value.reshape(-1, self.support_size)

    #@profile
    def clear_caches(self):
        """
        Clears the caches of the world model.
        """
        for kv_cache_dict_env in self.past_kv_cache_init_infer_envs:
            kv_cache_dict_env.clear()
        self.past_kv_cache_recurrent_infer.clear()
        self.keys_values_wm_list.clear()

        print(f'rank {self._rank} Cleared {self.__class__.__name__} past_kv_cache.')

    def __repr__(self) -> str:
        return "transformer-based latent world_model of UniZero"

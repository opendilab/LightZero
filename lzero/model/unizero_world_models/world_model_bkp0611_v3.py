import copy
import logging
from typing import Any, Tuple
from typing import Optional, Union, Dict
from typing import Optional

logging.getLogger().setLevel(logging.DEBUG)
from einops import rearrange
import torch.nn.functional as F
from sklearn.manifold import TSNE
from .kv_caching import KeysValues
from .slicer import Head
from .tokenizer import Tokenizer
from .transformer import Transformer, TransformerConfig
from .utils import LossWithIntermediateLosses, init_weights
from lzero.model.utils import cal_dormant_ratio
import os
from PIL import ImageDraw
import numpy as np
from PIL import Image
import torchvision
import matplotlib.pyplot as plt
from .utils import SimNorm, WorldModelOutput, quantize_state, to_device_for_kvcache
from .visualize_utils import visualize_reward_value_img_policy, visualize_reconstruction_v3, save_as_image_with_timestep, plot_latent_tsne_each_and_all_for_pong, plot_latent_tsne_each_and_all_for_visualmatch
import torch
import torch.nn as nn
import collections


class WorldModel(nn.Module):
    def __init__(self, act_vocab_size: int, config: TransformerConfig, tokenizer) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.act_vocab_size = act_vocab_size
        self.config = config
        self.transformer = Transformer(self.config)

        # Initialize configuration parameters
        self._initialize_config_parameters()

        # Initialize patterns for block masks
        self._initialize_patterns()

        # Position embedding
        self.pos_emb = nn.Embedding(config.max_tokens, config.embed_dim)
        self.precompute_pos_emb_diff_kv()

        # Initialize action embedding table
        self.act_embedding_table = nn.Embedding(act_vocab_size, config.embed_dim)

        # Head modules
        self.head_rewards = self._create_head(self.act_tokens_pattern, self.support_size)
        self.head_observations = self._create_head(self.all_but_last_latent_state_pattern, self.obs_per_embdding_dim,
                                                   self.sim_norm)
        self.head_policy = self._create_head(self.value_policy_tokens_pattern, self.action_shape)
        self.head_value = self._create_head(self.value_policy_tokens_pattern, self.support_size)

        # Apply weight initialization
        self.apply(init_weights)
        self._initialize_last_layer()

        # Cache structures
        self._initialize_cache_structures()

        # Projection input dimension
        self._initialize_projection_input_dim()

        # Hit count and query count statistics
        self._initialize_statistics()

        # Initialize keys and values for transformer
        self._initialize_transformer_keys_values()

    def _initialize_config_parameters(self) -> None:
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
        self.analysis_dormant_ratio = self.config.analysis_dormant_ratio
        self.num_observations_tokens = self.config.tokens_per_block - 1
        self.latent_recon_loss_weight = self.config.latent_recon_loss_weight
        self.perceptual_loss_weight = self.config.perceptual_loss_weight
        self.device = self.config.device
        self.support_size = self.config.support_size
        self.action_shape = self.config.action_shape
        self.max_cache_size = self.config.max_cache_size
        self.env_num = self.config.env_num
        self.num_layers = self.config.num_layers
        self.obs_per_embdding_dim = self.config.embed_dim
        self.sim_norm = SimNorm(simnorm_dim=self.group_size)

    def _initialize_patterns(self) -> None:
        self.all_but_last_latent_state_pattern = torch.ones(self.config.tokens_per_block)
        self.all_but_last_latent_state_pattern[-2] = 0
        self.act_tokens_pattern = torch.zeros(self.config.tokens_per_block)
        self.act_tokens_pattern[-1] = 1
        self.value_policy_tokens_pattern = torch.zeros(self.config.tokens_per_block)
        self.value_policy_tokens_pattern[-2] = 1

    def _create_head(self, block_mask: torch.Tensor, output_dim: int, norm_layer=None) -> Head:
        modules = [
            nn.Linear(self.config.embed_dim, self.config.embed_dim),
            nn.GELU(),
            nn.Linear(self.config.embed_dim, output_dim)
        ]
        if norm_layer:
            modules.append(norm_layer)
        return Head(
            max_blocks=self.config.max_blocks,
            block_mask=block_mask,
            head_module=nn.Sequential(*modules)
        )

    def _initialize_last_layer(self) -> None:
        last_linear_layer_init_zero = True
        if last_linear_layer_init_zero:
            for head in [self.head_value, self.head_rewards, self.head_observations]:
                for layer in reversed(head.head_module):
                    if isinstance(layer, nn.Linear):
                        nn.init.zeros_(layer.weight)
                        if layer.bias is not None:
                            nn.init.zeros_(layer.bias)
                        break

    def _initialize_cache_structures(self) -> None:
        """Initialize cache structures for past keys and values."""
        self.past_kv_cache_recurrent_infer = collections.OrderedDict()
        self.past_kv_cache_init_infer = collections.OrderedDict()
        self.past_kv_cache_init_infer_envs = [collections.OrderedDict() for _ in range(self.env_num)]
        self.keys_values_wm_list = []
        self.keys_values_wm_size_list = []

    def _initialize_projection_input_dim(self) -> None:
        """Initialize the projection input dimension based on the number of observation tokens."""
        if self.num_observations_tokens == 16:
            self.projection_input_dim = 128
        elif self.num_observations_tokens == 1:
            self.projection_input_dim = self.obs_per_embdding_dim

    def _initialize_statistics(self) -> None:
        """Initialize counters for hit count and query count statistics."""
        self.hit_count = 0
        self.total_query_count = 0
        self.length_largethan_maxminus5_context_cnt = 0
        self.length_largethan_maxminus7_context_cnt = 0
        self.root_hit_cnt = 0
        self.root_total_query_cnt = 0

    def _initialize_transformer_keys_values(self) -> None:
        """Initialize keys and values for the transformer."""
        self.keys_values_wm_single_env = self.transformer.generate_empty_keys_values(n=1,
                                                                                     max_tokens=self.context_length)
        self.keys_values_wm = self.transformer.generate_empty_keys_values(n=self.env_num,
                                                                          max_tokens=self.context_length)

    def precompute_pos_emb_diff_kv(self):
        if self.context_length <= 2:
            # If context length is 2 or less, no context is present
            return

        # Precompute positional embedding matrices for inference in collect/eval stages, not for training
        self.positional_embedding_k = [
            self._get_positional_embedding(layer, 'key')
            for layer in range(self.config.num_layers)
        ]
        self.positional_embedding_v = [
            self._get_positional_embedding(layer, 'value')
            for layer in range(self.config.num_layers)
        ]

        # Precompute all possible positional embedding differences
        self.pos_emb_diff_k = []
        self.pos_emb_diff_v = []

        for layer in range(self.config.num_layers):
            layer_pos_emb_diff_k = {}
            layer_pos_emb_diff_v = {}

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

    def _get_positional_embedding(self, layer, attn_type):
        """
        Helper function to get positional embedding for a given layer and attention type.
        Args:
            layer (int): Layer index.
            attn_type (str): Attention type, either 'key' or 'value'.
        Returns:
            torch.Tensor: The positional embedding tensor.
        """
        attn_func = getattr(self.transformer.blocks[layer].attn, attn_type)
        return attn_func(self.pos_emb.weight).view(
            1, self.config.max_tokens, self.num_heads, self.embed_dim // self.num_heads
        ).transpose(1, 2).detach()

    def forward(self, obs_embeddings_or_act_tokens: Dict[str, Union[torch.Tensor, tuple]],
                past_keys_values: Optional[torch.Tensor] = None,
                kvcache_independent: bool = False, is_init_infer: bool = True,
                valid_context_lengths: Optional[torch.Tensor] = None) -> WorldModelOutput:
        """
        Forward pass for the model.

        Args:
            obs_embeddings_or_act_tokens (dict): Dictionary containing observation embeddings or action tokens.
            past_keys_values (Optional[KeysValues]): Previous keys and values for transformer.
            kvcache_independent (bool): Whether to use independent key-value caching.
            is_init_infer (bool): Initialize inference.
            valid_context_lengths (Optional[torch.Tensor]): Valid context lengths.

        Returns:
            WorldModelOutput: Model output containing logits for observations, rewards, policy, and value.
        """
        # Determine previous steps based on key-value caching method
        if kvcache_independent:
            prev_steps = torch.tensor([0 if past_keys_values is None else past_kv.size for past_kv in past_keys_values],
                                      device=self.device)
        else:
            prev_steps = 0 if past_keys_values is None else past_keys_values.size

        # Reset valid_context_lengths during initial inference
        if is_init_infer:
            valid_context_lengths = None

        # Process observation embeddings
        if 'obs_embeddings' in obs_embeddings_or_act_tokens:
            obs_embeddings = obs_embeddings_or_act_tokens['obs_embeddings']
            if len(obs_embeddings.shape) == 2:
                obs_embeddings = obs_embeddings.unsqueeze(1)
            num_steps = obs_embeddings.size(1)
            sequences = self._add_position_embeddings(obs_embeddings, prev_steps, num_steps, kvcache_independent,
                                                      is_init_infer, valid_context_lengths)

        # Process action tokens
        elif 'act_tokens' in obs_embeddings_or_act_tokens:
            act_tokens = obs_embeddings_or_act_tokens['act_tokens']
            if len(act_tokens.shape) == 3:
                act_tokens = act_tokens.squeeze(1)
            num_steps = act_tokens.size(1)
            act_embeddings = self.act_embedding_table(act_tokens)
            sequences = self._add_position_embeddings(act_embeddings, prev_steps, num_steps, kvcache_independent,
                                                      is_init_infer, valid_context_lengths)

        # Process combined observation embeddings and action tokens
        else:
            sequences, num_steps = self._process_obs_act_combined(obs_embeddings_or_act_tokens, prev_steps)

        # Pass sequences through transformer
        x = self._transformer_pass(sequences, past_keys_values, kvcache_independent, valid_context_lengths)

        # Generate logits
        logits_observations = self.head_observations(x, num_steps=num_steps, prev_steps=prev_steps)
        logits_rewards = self.head_rewards(x, num_steps=num_steps, prev_steps=prev_steps)
        logits_policy = self.head_policy(x, num_steps=num_steps, prev_steps=prev_steps)
        logits_value = self.head_value(x, num_steps=num_steps, prev_steps=prev_steps)

        # logits_ends is None
        return WorldModelOutput(x, logits_observations, logits_rewards, None, logits_policy, logits_value)

    def _add_position_embeddings(self, embeddings, prev_steps, num_steps, kvcache_independent, is_init_infer,
                                 valid_context_lengths):
        """
        Add position embeddings to the input embeddings.

        Args:
            embeddings (torch.Tensor): Input embeddings.
            prev_steps (torch.Tensor): Previous steps.
            num_steps (int): Number of steps.
            kvcache_independent (bool): Whether to use independent key-value caching.
            is_init_infer (bool): Initialize inference.
            valid_context_lengths (torch.Tensor): Valid context lengths.

        Returns:
            torch.Tensor: Embeddings with position information added.
        """
        if kvcache_independent:
            steps_indices = prev_steps + torch.arange(num_steps, device=embeddings.device)
            position_embeddings = self.pos_emb(steps_indices).view(-1, num_steps, embeddings.shape[-1])
            return embeddings + position_embeddings
        else:
            if is_init_infer:
                return embeddings + self.pos_emb(prev_steps + torch.arange(num_steps, device=embeddings.device))
            else:
                valid_context_lengths = torch.tensor(self.keys_values_wm_size_list_current, device=self.device)
                position_embeddings = self.pos_emb(
                    valid_context_lengths + torch.arange(num_steps, device=embeddings.device)).unsqueeze(1)
                return embeddings + position_embeddings

    def _process_obs_act_combined(self, obs_embeddings_or_act_tokens, prev_steps):
        """
        Process combined observation embeddings and action tokens.

        Args:
            obs_embeddings_or_act_tokens (dict): Dictionary containing combined observation embeddings and action tokens.
            prev_steps (torch.Tensor): Previous steps.

        Returns:
            torch.Tensor: Combined observation and action embeddings with position information added.
        """
        obs_embeddings, act_tokens = obs_embeddings_or_act_tokens['obs_embeddings_and_act_tokens']
        if len(obs_embeddings.shape) == 3:
            obs_embeddings = obs_embeddings.view(act_tokens.shape[0], act_tokens.shape[1], self.num_observations_tokens,
                                                 -1)

        num_steps = int(obs_embeddings.size(1) * (obs_embeddings.size(2) + 1))
        act_embeddings = self.act_embedding_table(act_tokens)

        B, L, K, E = obs_embeddings.size()
        obs_act_embeddings = torch.empty(B, L * (K + 1), E, device=obs_embeddings.device)

        for i in range(L):
            obs = obs_embeddings[:, i, :, :]
            act = act_embeddings[:, i, 0, :].unsqueeze(1)
            obs_act = torch.cat([obs, act], dim=1)
            obs_act_embeddings[:, i * (K + 1):(i + 1) * (K + 1), :] = obs_act

        return obs_act_embeddings + self.pos_emb(prev_steps + torch.arange(num_steps, device=obs_embeddings.device)), num_steps

    def _transformer_pass(self, sequences, past_keys_values, kvcache_independent, valid_context_lengths):
        """
        Pass sequences through the transformer.

        Args:
            sequences (torch.Tensor): Input sequences.
            past_keys_values (Optional[torch.Tensor]): Previous keys and values for transformer.
            kvcache_independent (bool): Whether to use independent key-value caching.
            valid_context_lengths (torch.Tensor): Valid context lengths.

        Returns:
            torch.Tensor: Transformer output.
        """
        if kvcache_independent:
            x = [self.transformer(sequences[k].unsqueeze(0), past_kv,
                                  valid_context_lengths=valid_context_lengths[k].unsqueeze(0)) for k, past_kv in
                 enumerate(past_keys_values)]
            return torch.cat(x, dim=0)
        else:
            return self.transformer(sequences, past_keys_values, valid_context_lengths=valid_context_lengths)

    @torch.no_grad()
    def reset_from_initial_observations(self, obs_act_dict: torch.FloatTensor) -> torch.FloatTensor:
        """
        Reset the model state based on initial observations and actions.

        Args:
            obs_act_dict (torch.FloatTensor): A dictionary containing 'obs', 'action', and 'current_obs'.

        Returns:
            torch.FloatTensor: The outputs from the world model and the latent state.
        """
        # Extract observations, actions, and current observations from the dictionary.
        if isinstance(obs_act_dict, dict):
            observations = obs_act_dict['obs']
            buffer_action = obs_act_dict['action']
            current_obs = obs_act_dict['current_obs']

        # Encode observations to latent embeddings.
        obs_embeddings = self.tokenizer.encode_to_obs_embeddings(observations)

        if current_obs is not None:
            # ================ Collect and Evaluation Phase ================
            # Encode current observations to latent embeddings
            current_obs_embeddings = self.tokenizer.encode_to_obs_embeddings(current_obs)
            self.latent_state = current_obs_embeddings
            outputs_wm = self.refresh_kvs_with_initial_latent_state_for_init_infer(obs_embeddings, buffer_action, current_obs_embeddings)
        else:
            # ================ calculate the target value in Train phase ================
            self.latent_state = obs_embeddings
            outputs_wm = self.refresh_kvs_with_initial_latent_state_for_init_infer(obs_embeddings, buffer_action, None)

        return outputs_wm, self.latent_state

    @torch.no_grad()
    def refresh_kvs_with_initial_latent_state_for_init_infer(self, latent_state: torch.LongTensor,
                                                             buffer_action=None,
                                                             current_obs_embeddings=None) -> torch.FloatTensor:
        """
        Refresh key-value pairs with the initial latent state for inference.

        Args:
            latent_state (torch.LongTensor): The latent state embeddings.
            buffer_action (optional): Actions taken.
            current_obs_embeddings (optional): Current observation embeddings.

        Returns:
            torch.FloatTensor: The outputs from the world model.
        """
        n, num_observations_tokens, _ = latent_state.shape
        if n <= self.env_num:
            # ================ Collect and Evaluation Phase ================
            if current_obs_embeddings is not None:
                if max(buffer_action) == -1:
                    # First step in an episode
                    self.keys_values_wm = self.transformer.generate_empty_keys_values(n=n,
                                                                                      max_tokens=self.context_length)
                    outputs_wm = self.forward({'obs_embeddings': current_obs_embeddings},
                                              past_keys_values=self.keys_values_wm, is_init_infer=True)

                    # Copy and store keys_values_wm for a single environment
                    self.update_cache_context(current_obs_embeddings, is_init_infer=True)
                else:
                    # Assume latest_state is the new latent_state, containing information from ready_env_num environments
                    ready_env_num = current_obs_embeddings.shape[0]
                    self.keys_values_wm_list = []
                    self.keys_values_wm_size_list = []
                    for i in range(ready_env_num):
                        # Retrieve latent state for a single environment
                        state_single_env = latent_state[i]
                        quantized_state = state_single_env.detach().cpu().numpy()
                        # Compute hash value using quantized state
                        cache_key = quantize_state(quantized_state)
                        # Retrieve cached value
                        matched_value = self.past_kv_cache_init_infer_envs[i].get(cache_key)

                        self.root_total_query_cnt += 1
                        if matched_value is not None:
                            # If a matching value is found, add it to the list
                            self.root_hit_cnt += 1
                            # deepcopy is needed because forward modifies matched_value in place
                            self.keys_values_wm_list.append(
                                copy.deepcopy(to_device_for_kvcache(matched_value, 'cuda')))
                            self.keys_values_wm_size_list.append(matched_value.size)
                        else:
                            # Reset using zero values
                            self.keys_values_wm_single_env = self.transformer.generate_empty_keys_values(n=1,
                                                                                                         max_tokens=self.context_length)
                            outputs_wm = self.forward({'obs_embeddings': state_single_env.unsqueeze(0)},
                                                      past_keys_values=self.keys_values_wm_single_env,
                                                      is_init_infer=True)
                            self.keys_values_wm_list.append(self.keys_values_wm_single_env)
                            self.keys_values_wm_size_list.append(1)

                    # Input self.keys_values_wm_list, output self.keys_values_wm
                    self.keys_values_wm_size_list_current = self.trim_and_pad_kv_cache(is_init_infer=True)

                    buffer_action = buffer_action[:ready_env_num]
                    # if ready_env_num < self.env_num:
                    #     print(f'init inference ready_env_num: {ready_env_num} < env_num: {self.env_num}')
                    act_tokens = torch.from_numpy(np.array(buffer_action)).to(latent_state.device).unsqueeze(-1)
                    outputs_wm = self.forward({'act_tokens': act_tokens}, past_keys_values=self.keys_values_wm,
                                              is_init_infer=True)

                    outputs_wm = self.forward({'obs_embeddings': current_obs_embeddings},
                                              past_keys_values=self.keys_values_wm, is_init_infer=True)

                    # Copy and store keys_values_wm for a single environment
                    self.update_cache_context(current_obs_embeddings, is_init_infer=True)

        # elif n > self.env_num and buffer_action is not None and current_obs_embeddings is None:
        elif buffer_action is not None and current_obs_embeddings is None:
            # ================ calculate the target value in Train phase ================
            # [192, 16, 64] -> [32, 6, 16, 64]
            latent_state = latent_state.contiguous().view(buffer_action.shape[0], -1, num_observations_tokens,
                                                          self.obs_per_embdding_dim)  # (BL, K) for unroll_step=1

            latent_state = latent_state[:, :-1, :]
            buffer_action = torch.from_numpy(buffer_action).to(latent_state.device)
            act_tokens = rearrange(buffer_action, 'b l -> b l 1')

            # select the last timestep for each sample
            # This will select the last column while keeping the dimensions unchanged, and the target policy/value in the final step itself is not used.
            last_steps_act = act_tokens[:, -1:, :]
            act_tokens = torch.cat((act_tokens, last_steps_act), dim=1)

            outputs_wm = self.forward({'obs_embeddings_and_act_tokens': (latent_state, act_tokens)})

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

    @torch.no_grad()
    def forward_initial_inference(self, obs_act_dict):
        """
        Perform initial inference based on the given observation-action dictionary.

        Args:
            obs_act_dict (dict): Dictionary containing observations and actions.

        Returns:
            tuple: A tuple containing output sequence, latent state, logits rewards,
                   logits policy, and logits value.
        """
        # Unizero has context in the root node
        outputs_wm, latent_state = self.reset_from_initial_observations(obs_act_dict)
        self.past_kv_cache_recurrent_infer.clear()

        return (outputs_wm.output_sequence, latent_state, outputs_wm.logits_rewards,
                outputs_wm.logits_policy, outputs_wm.logits_value)


    @torch.no_grad()
    def forward_recurrent_inference(self, state_action_history, simulation_index=0,
                                    latent_state_index_in_search_path=[]):
        """
        Perform recurrent inference based on the state-action history.

        Args:
            state_action_history (list): List containing tuples of state and action history.
            simulation_index (int, optional): Index of the current simulation. Defaults to 0.
            latent_state_index_in_search_path (list, optional): List containing indices of latent states in the search path. Defaults to [].

        Returns:
            tuple: A tuple containing output sequence, updated latent state, reward,
                   logits policy, and logits value.
        """
        latest_state, action = state_action_history[-1]
        ready_env_num = latest_state.shape[0]

        self.keys_values_wm_list = []
        self.keys_values_wm_size_list = []
        self.keys_values_wm_size_list = self.retrieve_or_generate_kvcache(latest_state, ready_env_num, simulation_index)

        latent_state_list = []
        token = action.reshape(-1, 1)

        min_size = min(self.keys_values_wm_size_list)
        if min_size >= self.config.max_tokens - 5:
            self.length_largethan_maxminus5_context_cnt += len(self.keys_values_wm_size_list)
        if min_size >= self.config.max_tokens - 7:
            self.length_largethan_maxminus7_context_cnt += len(self.keys_values_wm_size_list)

        # Print statistics
        if self.total_query_count > 0 and self.total_query_count % 10000 == 0:
            self.hit_freq = self.hit_count / self.total_query_count
            print('total_query_count:', self.total_query_count)
            length_largethan_maxminus5_context_cnt_ratio = self.length_largethan_maxminus5_context_cnt / self.total_query_count
            print('recurrent largethan_maxminus5_context:', self.length_largethan_maxminus5_context_cnt)
            print('recurrent largethan_maxminus5_context_ratio:', length_largethan_maxminus5_context_cnt_ratio)
            length_largethan_maxminus7_context_cnt_ratio = self.length_largethan_maxminus7_context_cnt / self.total_query_count
            print('recurrent largethan_maxminus7_context_ratio:', length_largethan_maxminus7_context_cnt_ratio)
            print('recurrent largethan_maxminus7_context:', self.length_largethan_maxminus7_context_cnt)

        # Trim and pad kv_cache
        self.keys_values_wm_size_list = self.trim_and_pad_kv_cache(is_init_infer=False)
        self.keys_values_wm_size_list_current = self.keys_values_wm_size_list

        for k in range(2):
            # action_token obs_token, ..., obs_token  1+1
            if k == 0:
                obs_embeddings_or_act_tokens = {'act_tokens': token}
            else:
                obs_embeddings_or_act_tokens = {'obs_embeddings': token}

            # Perform forward pass
            outputs_wm = self.forward(
                obs_embeddings_or_act_tokens,
                past_keys_values=self.keys_values_wm,
                kvcache_independent=False,
                is_init_infer=False
            )

            self.keys_values_wm_size_list_current = [i + 1 for i in self.keys_values_wm_size_list_current]

            if k == 0:
                reward = outputs_wm.logits_rewards  # (B,)

            if k < self.num_observations_tokens:
                token = outputs_wm.logits_observations
                if len(token.shape) != 3:
                    token = token.unsqueeze(1)  # (8,1024) -> (8,1,1024)
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

    def trim_and_pad_kv_cache(self, is_init_infer=True):
        """
        Adjusts the key-value cache for each environment to ensure they all have the same size.

        In a multi-environment setting, the key-value cache (kv_cache) for each environment is stored separately.
        During recurrent inference, the kv_cache sizes may vary across environments. This method pads each kv_cache
        to match the largest size found among them, facilitating batch processing in the transformer forward pass.

        Parameters:
        is_init_infer (bool): Indicates if this is an initial inference. Default is True.

        Returns:
        list: Updated sizes of the key-value caches.
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

    def update_cache_context(self, latent_state, is_init_infer=True, simulation_index=0,
                             latent_state_index_in_search_path=[], valid_context_lengths=None):
        """
        Update the cache context with the given latent state.

        Parameters:
        latent_state (torch.Tensor): The latent state tensor.
        is_init_infer (bool): Flag to indicate if this is the initial inference.
        simulation_index (int): Index of the simulation.
        latent_state_index_in_search_path (list): List of indices in the search path.
        valid_context_lengths (list): List of valid context lengths.
        """
        if self.context_length <= 2:
            # No context to update if the context length is less than or equal to 2.
            return
        for i in range(latent_state.size(0)):
            # ============ Iterate over each environment ============
            state_single_env = latent_state[i]
            quantized_state = state_single_env.detach().cpu().numpy()
            cache_key = quantize_state(quantized_state)
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
                        # Keep only the last self.context_length-3 timesteps of context
                        # For memory environments, training is for H steps, recurrent_inference might exceed H steps
                        # Assuming cache dimension is [batch_size, num_heads, sequence_length, features]
                        k_cache_current = self.keys_values_wm_single_env._keys_values[layer]._k_cache._cache
                        v_cache_current = self.keys_values_wm_single_env._keys_values[layer]._v_cache._cache

                        # Remove the first 2 steps, keep the last self.context_length-3 steps
                        k_cache_trimmed = k_cache_current[:, :, 2:context_length - 1, :].squeeze(0)
                        v_cache_trimmed = v_cache_current[:, :, 2:context_length - 1, :].squeeze(0)

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

            if is_init_infer:
                # Store the latest key-value cache for initial inference
                self.past_kv_cache_init_infer_envs[i][cache_key] = copy.deepcopy(
                    to_device_for_kvcache(self.keys_values_wm_single_env, 'cpu'))
            else:
                # Store the latest key-value cache for recurrent inference
                self.past_kv_cache_recurrent_infer[cache_key] = copy.deepcopy(
                    to_device_for_kvcache(self.keys_values_wm_single_env, 'cpu'))

    def retrieve_or_generate_kvcache(self, latent_state, ready_env_num, simulation_index=0):
        """
        Retrieves or generates key-value caches for each environment based on the latent state.

        For each environment, this method either retrieves a matching cache from the predefined
        caches if available, or generates a new cache if no match is found. The method updates
        the internal lists with these caches and their sizes.

        Parameters:
        - latent_state: list of latent states for each environment.
        - ready_env_num: int, number of environments ready for processing.
        - simulation_index: int, optional index for simulation tracking (default is 0).

        Returns:
        - list of sizes of the key-value caches for each environment.
        """
        for i in range(ready_env_num):
            self.total_query_count += 1
            state_single_env = latent_state[i]  # Get the latent state for a single environment
            cache_key = quantize_state(state_single_env)  # Compute the hash value using the quantized state

            # Try to retrieve the cached value from past_kv_cache_init_infer_envs
            matched_value = self.past_kv_cache_init_infer_envs[i].get(cache_key)

            # If not found, try to retrieve from past_kv_cache_recurrent_infer
            if matched_value is None:
                matched_value = self.past_kv_cache_recurrent_infer.get(cache_key)

            if matched_value is not None:
                # If a matching cache is found, add it to the lists
                self.hit_count += 1
                # Perform a deep copy because the transformer's forward pass might modify matched_value in-place
                self.keys_values_wm_list.append(copy.deepcopy(to_device_for_kvcache(matched_value, self.device)))
                self.keys_values_wm_size_list.append(matched_value.size)
            else:
                # If no matching cache is found, generate a new one using zero reset
                self.keys_values_wm_single_env = self.transformer.generate_empty_keys_values(
                    n=1, max_tokens=self.context_length
                )
                self.forward(
                    {'obs_embeddings': torch.from_numpy(state_single_env).unsqueeze(0).to(self.device)},
                    past_keys_values=self.keys_values_wm_single_env, is_init_infer=True
                )
                self.keys_values_wm_list.append(self.keys_values_wm_single_env)
                self.keys_values_wm_size_list.append(1)

        return self.keys_values_wm_size_list


    def compute_loss(self, batch, target_tokenizer: Tokenizer = None, inverse_scalar_transform_handle=None, **kwargs: Any) -> LossWithIntermediateLosses:
        # 将观察编码为潜在状态表示
        obs_embeddings = self.tokenizer.encode_to_obs_embeddings(batch['observations'])

        # ========= for visual analysis =========
        # for pong
        # self.plot_latent_tsne_each_and_all_for_pong(obs_embeddings, suffix='pong_H10_H4_tsne')
        # self.save_as_image_with_timestep(batch['observations'], suffix='pong_H10_H4_tsne')
        # for visual_match
        # self.plot_latent_tsne_each_and_all(obs_embeddings, suffix='visual_match_memlen1-60-15_tsne')
        # self.save_as_image_with_timestep(batch['observations'], suffix='visual_match_memlen1-60-15_tsne')
        # ========= logging for analysis =========

        if self.analysis_dormant_ratio:
            # calculate dormant ratio of encoder
            shape = batch['observations'].shape  # (..., C, H, W)
            inputs = batch['observations'].contiguous().view(-1, *shape[-3:])  # (32,5,3,64,64) -> (160,3,64,64)
            dormant_ratio_encoder = cal_dormant_ratio(self.tokenizer.representation_network, inputs.detach(),
                                                      percentage=self.dormant_threshold)
            self.past_kv_cache_init_infer.clear()
            self.past_kv_cache_recurrent_infer.clear()
            self.keys_values_wm_list.clear()
            torch.cuda.empty_cache()
        else:
            dormant_ratio_encoder = torch.tensor(0.)

        # 假设latent_state_roots是一个tensor
        latent_state_l2_norms = torch.norm(obs_embeddings, p=2, dim=2).mean()  # 计算L2范数

        if self.obs_type == 'image':
            # 从潜在状态表示重建观察
            reconstructed_images = self.tokenizer.decode_to_obs(obs_embeddings)

            #  ========== for visualize ==========
            # batch['target_policy'].shape torch.Size([2, 17, 4])
            # batch['target_value'].shape torch.Size([2, 17, 101])
            # batch['rewards'].shape torch.Size([2, 17, 101])
            original_images, reconstructed_images = batch['observations'], reconstructed_images
            target_policy = batch['target_policy']
            target_predict_value = inverse_scalar_transform_handle(batch['target_value'].reshape(-1, 101)).reshape(
                batch['observations'].shape[0], batch['observations'].shape[1], 1)  # torch.Size([2, 17, 1])
            true_rewards = inverse_scalar_transform_handle(batch['rewards'].reshape(-1, 101)).reshape(
                batch['observations'].shape[0], batch['observations'].shape[1], 1)  # torch.Size([2, 17, 1])
            #  ========== for visualize ==========

            # 计算重建损失和感知损失 TODO
            # latent_recon_loss = self.tokenizer.reconstruction_loss(batch['observations'].reshape(-1, 3, 64, 64), reconstructed_images) # NOTE: for stack=1
            # perceptual_loss = self.tokenizer.perceptual_loss(batch['observations'].reshape(-1, 3, 64, 64), reconstructed_images) # NOTE: for stack=1

            # latent_recon_loss = self.tokenizer.reconstruction_loss(batch['observations'].reshape(-1, 4, 64, 64), reconstructed_images) # NOTE: for stack=4
            latent_recon_loss = torch.tensor(0., device=batch['observations'].device,
                                             dtype=batch['observations'].dtype)  # NOTE: for stack=4
            perceptual_loss = torch.tensor(0., device=batch['observations'].device,
                                           dtype=batch['observations'].dtype)  # NOTE: for stack=4


        elif self.obs_type == 'vector':
            perceptual_loss = torch.tensor(0., device=batch['observations'].device,
                                           dtype=batch['observations'].dtype)  # NOTE: for stack=4

            # TODO: no decoder
            # latent_recon_loss = torch.tensor(0., device=batch['observations'].device, dtype=batch['observations'].dtype)  # NOTE: for stack=4

            # 从潜在状态表示重建观察
            reconstructed_images = self.tokenizer.decode_to_obs(obs_embeddings.reshape(-1, self.embed_dim))
            # 计算重建损失
            latent_recon_loss = self.tokenizer.reconstruction_loss(batch['observations'].reshape(-1, 25),
                                                                   reconstructed_images)  # NOTE: for stack=1
        elif self.obs_type == 'image_memory':
            # 从潜在状态表示重建观察
            reconstructed_images = self.tokenizer.decode_to_obs(obs_embeddings)

            #  ========== for debugging ==========
            # batch['observations'].shape torch.Size([2, 17, 3, 5, 5])
            # reconstructed_images.shape torch.Size([34, 3, 5, 5])
            # self.visualize_reconstruction_v1(original_images, reconstructed_images)

            #  ========== for visualize ==========
            # batch['target_policy'].shape torch.Size([2, 17, 4])
            # batch['target_value'].shape torch.Size([2, 17, 101])
            # batch['rewards'].shape torch.Size([2, 17, 101])
            original_images, reconstructed_images = batch['observations'], reconstructed_images

            target_policy = batch['target_policy']
            target_predict_value = inverse_scalar_transform_handle(batch['target_value'].reshape(-1, 101)).reshape(
                batch['observations'].shape[0], batch['observations'].shape[1], 1)  # torch.Size([2, 17, 1])
            true_rewards = inverse_scalar_transform_handle(batch['rewards'].reshape(-1, 101)).reshape(
                batch['observations'].shape[0], batch['observations'].shape[1], 1)  # torch.Size([2, 17, 1])
            #  ========== for visualize ==========

            # 计算重建损失和感知损失
            latent_recon_loss = self.tokenizer.reconstruction_loss(batch['observations'].reshape(-1, 3, 5, 5),
                                                                   reconstructed_images)  # NOTE: for stack=1 TODO
            # latent_recon_loss = self.tokenizer.reconstruction_loss(batch['observations'].reshape(-1, 4, 5, 5), reconstructed_images)  # NOTE: for stack=1
            # latent_recon_loss = torch.tensor(0., device=batch['observations'].device, dtype=batch['observations'].dtype)  # NOTE: for stack=4
            perceptual_loss = torch.tensor(0., device=batch['observations'].device,
                                           dtype=batch['observations'].dtype)  # NOTE: for stack=4

        # 动作tokens
        act_tokens = rearrange(batch['actions'], 'b l -> b l 1')

        # 前向传播,得到预测的观察、奖励和策略等
        outputs = self.forward({'obs_embeddings_and_act_tokens': (obs_embeddings, act_tokens)})

        # ========= logging for analysis =========
        if self.analysis_dormant_ratio:
            # calculate dormant ratio of world_model
            dormant_ratio_world_model = cal_dormant_ratio(self, {
                'obs_embeddings_and_act_tokens': (obs_embeddings.detach(), act_tokens.detach())},
                                                          percentage=self.dormant_threshold)
            self.past_kv_cache_init_infer.clear()
            self.past_kv_cache_recurrent_infer.clear()
            self.keys_values_wm_list.clear()
            torch.cuda.empty_cache()
        else:
            dormant_ratio_world_model = torch.tensor(0.)

        #  ========== for visualize ==========
        # outputs.logits_policy.shape torch.Size([2, 17, 4])
        # outputs.logits_value.shape torch.Size([2, 17, 101])
        # outputs.logits_rewards.shape torch.Size([2, 17, 101])
        predict_policy = outputs.logits_policy
        # 使用 softmax 对最后一个维度（dim=-1）进行处理
        predict_policy = F.softmax(outputs.logits_policy, dim=-1)
        predict_value = inverse_scalar_transform_handle(outputs.logits_value.reshape(-1, 101)).reshape(
            batch['observations'].shape[0], batch['observations'].shape[1], 1)  # predict_value: torch.Size([2, 17, 1])
        predict_rewards = inverse_scalar_transform_handle(outputs.logits_rewards.reshape(-1, 101)).reshape(
            batch['observations'].shape[0], batch['observations'].shape[1],
            1)  # predict_rewards: torch.Size([2, 17, 1])
        # self.visualize_reward_value_img_policy(original_images, reconstructed_images, target_predict_value, true_rewards, target_policy, predict_value, predict_rewards, predict_policy, not_plot_timesteps=list(np.arange(4,60)), suffix='visual_match_memlen1-60-15/one_success_episode')
        # self.visualize_reward_value_img_policy(original_images, reconstructed_images, target_predict_value, true_rewards, target_policy, predict_value, predict_rewards, predict_policy, not_plot_timesteps=list(np.arange(4,60)), suffix='visual_match_memlen1-60-15/one_fail_episode')
        # self.visualize_reward_value_img_policy(original_images, reconstructed_images, target_predict_value, true_rewards, target_policy, predict_value, predict_rewards, predict_policy, not_plot_timesteps=[], suffix='pong_H10_H4_0531')
        # import sys
        # sys.exit(0)
        #  ========== for visualize ==========

        # 为了训练稳定性,使用target_tokenizer计算真实的下一个潜在状态表示
        with torch.no_grad():
            traget_obs_embeddings = target_tokenizer.encode_to_obs_embeddings(batch['observations'])

        # 计算观察、奖励和结束标签
        labels_observations, labels_rewards, labels_ends = self.compute_labels_world_model(traget_obs_embeddings,
                                                                                           batch['rewards'],
                                                                                           batch['ends'],
                                                                                           batch['mask_padding'])

        # 重塑观察的logits和labels
        logits_observations = rearrange(outputs.logits_observations[:, :-1], 'b t o -> (b t) o')
        labels_observations = labels_observations.reshape(-1, self.projection_input_dim)

        # 计算观察的预测损失。这里提供了两种选择:MSE和Group KL
        if self.predict_latent_loss_type == 'mse':
            # MSE损失,直接比较logits和labels
            loss_obs = torch.nn.functional.mse_loss(logits_observations, labels_observations, reduction='none').mean(
                -1)  # labels_observations.detach()是冗余的，因为前面是在with torch.no_grad()中计算的
        elif self.predict_latent_loss_type == 'group_kl':
            # Group KL损失,将特征分组,然后计算组内的KL散度
            batch_size, num_features = logits_observations.shape

            logits_reshaped = logits_observations.reshape(batch_size, self.num_groups, self.group_size)
            labels_reshaped = labels_observations.reshape(batch_size, self.num_groups, self.group_size)

            loss_obs = F.kl_div(logits_reshaped.log(), labels_reshaped, reduction='none').sum(dim=-1).mean(dim=-1)

        # 应用mask到loss_obs
        mask_padding_expanded = batch['mask_padding'][:, 1:].contiguous().view(-1)
        loss_obs = (loss_obs * mask_padding_expanded)

        # 计算策略和价值的标签
        labels_policy, labels_value = self.compute_labels_world_model_value_policy(batch['target_value'],
                                                                                   batch['target_policy'],
                                                                                   batch['mask_padding'])

        # 计算奖励、策略和价值的损失
        loss_rewards = self.compute_cross_entropy_loss(outputs, labels_rewards, batch, element='rewards')
        loss_policy, orig_policy_loss, policy_entropy = self.compute_cross_entropy_loss(outputs, labels_policy, batch,
                                                                                        element='policy')
        loss_value = self.compute_cross_entropy_loss(outputs, labels_value, batch, element='value')

        # 计算时间步
        timesteps = torch.arange(batch['actions'].shape[1], device=batch['actions'].device)
        # 计算每个时间步的折扣系数
        discounts = self.gamma ** timesteps

        # 将损失分为第一步、中间步和最后一步
        first_step_losses = {}
        middle_step_losses = {}
        last_step_losses = {}
        #  batch['mask_padding'] 为后面H步的mask情况，如果mask为False则把对应的loss从统计中去掉，以维持平均统计值的准确性
        # 对每个损失项进行分组计算
        for loss_name, loss_tmp in zip(
                ['loss_obs', 'loss_rewards', 'loss_value', 'loss_policy', 'orig_policy_loss', 'policy_entropy'],
                [loss_obs, loss_rewards, loss_value, loss_policy, orig_policy_loss, policy_entropy]
        ):
            if loss_name == 'loss_obs':
                seq_len = batch['actions'].shape[1] - 1
                # 获取对应的 mask_padding
                mask_padding = batch['mask_padding'][:, 1:seq_len]
            else:
                seq_len = batch['actions'].shape[1]
                # 获取对应的 mask_padding
                mask_padding = batch['mask_padding'][:, :seq_len]

            # 将损失调整为 (batch_size, seq_len) 的形状
            loss_tmp = loss_tmp.view(-1, seq_len)

            # 第一步的损失
            first_step_mask = mask_padding[:, 0]
            first_step_losses[loss_name] = loss_tmp[:, 0][first_step_mask].mean()

            # 中间步的损失
            middle_step_index = seq_len // 2
            middle_step_mask = mask_padding[:, middle_step_index]
            middle_step_losses[loss_name] = loss_tmp[:, middle_step_index][middle_step_mask].mean()

            # 最后一步的损失
            last_step_mask = mask_padding[:, -1]
            last_step_losses[loss_name] = loss_tmp[:, -1][last_step_mask].mean()

        # 对重构损失和感知损失进行折扣
        discounted_latent_recon_loss = latent_recon_loss
        discounted_perceptual_loss = perceptual_loss

        # 计算整体的折扣损失
        discounted_loss_obs = (loss_obs.view(-1, batch['actions'].shape[1] - 1) * discounts[1:]).mean()
        discounted_loss_rewards = (loss_rewards.view(-1, batch['actions'].shape[1]) * discounts).mean()
        discounted_loss_value = (loss_value.view(-1, batch['actions'].shape[1]) * discounts).mean()
        discounted_loss_policy = (loss_policy.view(-1, batch['actions'].shape[1]) * discounts).mean()
        discounted_orig_policy_loss = (orig_policy_loss.view(-1, batch['actions'].shape[1]) * discounts).mean()
        discounted_policy_entropy = (policy_entropy.view(-1, batch['actions'].shape[1]) * discounts).mean()

        return LossWithIntermediateLosses(
            latent_recon_loss_weight=self.latent_recon_loss_weight,
            perceptual_loss_weight=self.perceptual_loss_weight,
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
            dormant_ratio_world_model=dormant_ratio_world_model,
            latent_state_l2_norms=latent_state_l2_norms,
        )

    def compute_cross_entropy_loss(self, outputs, labels, batch, element='rewards'):
        # 假设outputs是一个具有'rewards'、'policy'和'value'的logits属性的对象
        # labels是一个与之比较的目标张量。batch是一个带有指示有效时间步的mask的字典。

        logits = getattr(outputs, f'logits_{element}')

        # 重塑你的张量
        logits = rearrange(logits, 'b t e -> (b t) e')
        labels = labels.reshape(-1, labels.shape[-1])  # 假设labels最初的shape是 [batch, time, dim]

        # 重塑你的mask。True表示有效数据。
        mask_padding = rearrange(batch['mask_padding'], 'b t -> (b t)')

        # 计算交叉熵损失
        loss = -(torch.log_softmax(logits, dim=1) * labels).sum(1)
        loss = (loss * mask_padding)

        if element == 'policy':
            # 计算策略熵损失
            policy_entropy = self.compute_policy_entropy_loss(logits, mask_padding)
            # 用指定的权重组合损失
            combined_loss = loss - self.policy_entropy_weight * policy_entropy
            return combined_loss, loss, policy_entropy

        return loss

    def compute_policy_entropy_loss(self, logits, mask):
        # 计算策略的熵
        probs = torch.softmax(logits, dim=1)
        log_probs = torch.log_softmax(logits, dim=1)
        entropy = -(probs * log_probs).sum(1)
        # 应用mask并返回平均熵损失
        entropy_loss = (entropy * mask)
        return entropy_loss

    def compute_labels_world_model(self, obs_embeddings: torch.Tensor, rewards: torch.Tensor, ends: torch.Tensor,
                                   mask_padding: torch.BoolTensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert torch.all(ends.sum(dim=1) <= 1)  # 每个序列样本最多只有1个done
        mask_fill = torch.logical_not(mask_padding)
        labels_observations = obs_embeddings.contiguous().view(rewards.shape[0], -1, self.projection_input_dim)[:, 1:]
        mask_fill_rewards = mask_fill.unsqueeze(-1).expand_as(rewards)
        labels_rewards = rewards.masked_fill(mask_fill_rewards, -100)

        labels_ends = ends.masked_fill(mask_fill, -100)
        return labels_observations, labels_rewards.reshape(-1, self.support_size), labels_ends.reshape(-1)

    def compute_labels_world_model_value_policy(self, target_value: torch.Tensor, target_policy: torch.Tensor,
                                                mask_padding: torch.BoolTensor) -> Tuple[torch.Tensor, torch.Tensor]:

        mask_fill = torch.logical_not(mask_padding)
        mask_fill_policy = mask_fill.unsqueeze(-1).expand_as(target_policy)
        labels_policy = target_policy.masked_fill(mask_fill_policy, -100)

        mask_fill_value = mask_fill.unsqueeze(-1).expand_as(target_value)
        labels_value = target_value.masked_fill(mask_fill_value, -100)
        return labels_policy.reshape(-1, self.action_shape), labels_value.reshape(-1, self.support_size)  # TODO(pu)

    def __repr__(self) -> str:
        return "world_model"

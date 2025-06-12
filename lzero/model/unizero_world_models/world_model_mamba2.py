import logging
from typing import Dict, Union, Optional, List, Tuple, Any
import copy 

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.distributions import Categorical, Independent, Normal, TransformedDistribution, TanhTransform

from lzero.model.common import SimNorm
from lzero.model.utils import cal_dormant_ratio
from .kv_caching import KeysValues
from .slicer import Head, PolicyHeadCont
from .tokenizer import Tokenizer
from .utils import LossWithIntermediateLosses, init_weights, WorldModelOutput, hash_state


logging.getLogger().setLevel(logging.DEBUG)


class WorldModel(nn.Module):
    """
    Overview:
        The WorldModel class is responsible for the scalable latent world model of UniZero (https://arxiv.org/abs/2406.10667),
        which is used to predict the next latent state, rewards, policy, and value based on the current latent state and action.
        The world model consists of three main components:
            - a tokenizer, which encodes observations into embeddings,
            - a transformer, which processes the input sequences,
            - and heads, which generate the logits for observations, rewards, policy, and value.
    """

    def __init__(self, config, tokenizer) -> None:
        """
        Overview:
            Initialize the WorldModel class.
        Arguments:
            - config (:obj:`TransformerConfig`): The configuration for the transformer.
            - tokenizer (:obj:`Tokenizer`): The tokenizer.
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.config = config
        self.mamba_model = Mamba(self.config)

        if self.config.device == 'cpu':
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Move all modules to the specified device
        logging.info(f"self.device: {self.device}")
        self.to(self.device)

        # Initialize configuration parameters
        self._initialize_config_parameters()

        # Initialize patterns for block masks
        self._initialize_patterns()

        self.hidden_size = config.embed_dim

        self.continuous_action_space = self.config.continuous_action_space

        # Initialize action embedding table
        if self.continuous_action_space:
            # TODO: check the effect of SimNorm
            self.act_embedding_table = nn.Sequential(
                nn.Linear(config.action_space_size, config.embed_dim, device=self.device, bias=False),
                SimNorm(simnorm_dim=self.group_size))
        else:
            # for discrete action space
            self.act_embedding_table = nn.Embedding(config.action_space_size, config.embed_dim, device=self.device)
            logging.info(f"self.act_embedding_table.weight.device: {self.act_embedding_table.weight.device}")

        # self.final_norm_option_in_obs_head = getattr(config, 'final_norm_option_in_obs_head', 'SimNorm')
        self.final_norm_option_in_obs_head = getattr(config, 'final_norm_option_in_obs_head', 'LayerNorm') # TODO

        # Head modules
        self.head_rewards = self._create_head(self.act_tokens_pattern, self.support_size)
        self.head_observations = self._create_head(self.all_but_last_latent_state_pattern, self.obs_per_embdding_dim, \
                                                    self._get_final_norm(self.final_norm_option_in_obs_head)  
                                                   ) 
        if self.continuous_action_space:
            self.sigma_type = self.config.sigma_type
            self.bound_type = self.config.bound_type
            self.head_policy = self._create_head_cont(self.value_policy_tokens_pattern, self.action_space_size)
        else:
            self.head_policy = self._create_head(self.value_policy_tokens_pattern, self.action_space_size)
        self.head_value = self._create_head(self.value_policy_tokens_pattern, self.support_size)

        skip_modules = set(self.tokenizer.encoder.pretrained_model.modules())
        def custom_init(module):
            if module in skip_modules:
                return
            init_weights(module, norm_type=self.config.norm_type)
        self.apply(custom_init)

        self._initialize_last_layer()

        # Cache structures
        self._initialize_cache_structures()

        # Projection input dimension
        self._initialize_projection_input_dim()

        # Hit count and query count statistics
        self._initialize_statistics()

        # Initialize keys and values for transformer
        self._initialize_mamba_states()

        self.latent_recon_loss = torch.tensor(0., device=self.device)
        self.perceptual_loss = torch.tensor(0., device=self.device)

        self.reanalyze_phase = False

    def _get_final_norm(self, norm_option: str) -> nn.Module:
        """
        根据指定的归一化选项返回相应的归一化模块。
        """
        if norm_option == 'LayerNorm':
            return nn.LayerNorm(self.config.embed_dim, eps=1e-5)
        elif norm_option == 'SimNorm':
            return SimNorm(simnorm_dim=self.config.group_size)
        else:
            raise ValueError(f"Unsupported final_norm_option_in_obs_head: {norm_option}")

    def _initialize_config_parameters(self) -> None:
        """Initialize configuration parameters."""
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
        self.support_size = self.config.support_size
        self.action_space_size = self.config.action_space_size
        self.max_cache_size = self.config.max_cache_size
        self.env_num = self.config.env_num
        self.num_layers = self.config.num_layers
        self.obs_per_embdding_dim = self.config.embed_dim
        self.sim_norm = SimNorm(simnorm_dim=self.group_size)

    def _initialize_patterns(self) -> None:
        """Initialize patterns for block masks."""
        self.all_but_last_latent_state_pattern = torch.ones(self.config.tokens_per_block)
        self.all_but_last_latent_state_pattern[-2] = 0
        self.act_tokens_pattern = torch.zeros(self.config.tokens_per_block)
        self.act_tokens_pattern[-1] = 1
        self.value_policy_tokens_pattern = torch.zeros(self.config.tokens_per_block)
        self.value_policy_tokens_pattern[-2] = 1

    def _create_head(self, block_mask: torch.Tensor, output_dim: int, norm_layer=None) -> Head:
        """Create head modules for the transformer."""
        modules = [
            nn.Linear(self.config.embed_dim, self.config.embed_dim),
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

    def _create_head_cont(self, block_mask: torch.Tensor, output_dim: int, norm_layer=None) -> Head:
        """Create head modules for the transformer."""
        from ding.model.common import ReparameterizationHead
        self.fc_policy_head = ReparameterizationHead(
            input_size=self.config.embed_dim,
            output_size=output_dim,
            layer_num=2,  # TODO: check the effect of layer_num
            sigma_type=self.sigma_type,
            activation=nn.GELU(approximate='tanh'),
            fixed_sigma_value=self.config.fixed_sigma_value if self.sigma_type == 'fixed' else 0.5,
            norm_type=None,
            bound_type=self.bound_type
        )
        return PolicyHeadCont(
            max_blocks=self.config.max_blocks,
            block_mask=block_mask,
            head_module=self.fc_policy_head
        )

    def _initialize_last_layer(self) -> None:
        """Initialize the last linear layer."""
        last_linear_layer_init_zero = True  # TODO
        if last_linear_layer_init_zero:
            if self.continuous_action_space:
                module_to_initialize = [self.head_value, self.head_rewards, self.head_observations]
            else:
                module_to_initialize = [self.head_policy, self.head_value, self.head_rewards, self.head_observations]
            for head in module_to_initialize:
                for layer in reversed(head.head_module):
                    if isinstance(layer, nn.Linear):
                        nn.init.zeros_(layer.weight)
                        if layer.bias is not None:
                            nn.init.zeros_(layer.bias)
                        break

    def _initialize_cache_structures(self) -> None:
        """Initialize cache structures for past keys and values."""
        from collections import defaultdict
        self.past_mamba_states_cache_recurrent_infer = defaultdict(dict)
        self.past_mamba_states_cache_init_infer_envs = [defaultdict(dict) for _ in range(self.env_num)]
        self.mamba_states_wm_list = []
        self.mamba_states_size_list = []

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

    def _initialize_mamba_states(self) -> None:
        """Initialize Mamba states for the world model."""
        # CORRECTED: Pass max_seq_len to generate_empty_state
        max_len = self.context_length
        self.mamba_states_wm_single_env = self.mamba_model.generate_empty_state(batch_size=1, max_seq_len=max_len)
        self.mamba_states_wm_single_env_tmp = self.mamba_model.generate_empty_state(batch_size=1, max_seq_len=max_len)
        self.mamba_states_wm = self.mamba_model.generate_empty_state(batch_size=self.env_num, max_seq_len=max_len)

    def forward(
        self,
        obs_embeddings_or_act_tokens: Dict[str, Union[torch.Tensor, Tuple]],
        past_keys_values: Optional[torch.Tensor] = None,
        seqlen_offset: Optional[int] = 0
    ) -> Tuple[WorldModelOutput, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Overview:
            Forward pass for the world model. This method processes observation embeddings and/or action tokens,
            optionally adds position encodings (with or without rotary position embeddings), passes the resulting
            sequences through the transformer, and finally generates logits for observations, rewards, policy, and value.
        
        Arguments:
            - obs_embeddings_or_act_tokens (dict): Dictionary containing one or more of the following keys:
                - 'obs_embeddings': torch.Tensor representing observation embeddings.
                - 'act_tokens': torch.Tensor representing action tokens.
                - 'obs_embeddings_and_act_tokens': Combined data for both observations and actions.
            - past_keys_values (Optional[torch.Tensor]): Cached key-value pairs for the transformer. Defaults to None.
            - is_init_infer (bool): Flag to indicate if this is the initial inference step. Defaults to True.
            - valid_context_lengths (Optional[torch.Tensor]): Valid lengths for the context. Defaults to None.
            - search_depth (Optional[List[int]]): List representing the search depth for each batch element, used for
                position encoding adjustment. Defaults to None.
        
        Returns:
            WorldModelOutput: An output instance containing:
                - x: Output features from the transformer.
                - logits for observations.
                - logits for rewards.
                - logits_ends (None).
                - logits for policy.
                - logits for value.
        """

        prev_steps = seqlen_offset

        # Process observation embeddings if available.
        if "obs_embeddings" in obs_embeddings_or_act_tokens:
            obs_embeddings = obs_embeddings_or_act_tokens["obs_embeddings"]
            # If the observation embeddings have 2 dimensions, expand them to include a time dimension.
            if len(obs_embeddings.shape) == 2:
                obs_embeddings = obs_embeddings.unsqueeze(1)
            num_steps = obs_embeddings.size(1)
            sequences = obs_embeddings
            
        # Process action tokens if available.
        elif "act_tokens" in obs_embeddings_or_act_tokens:
            act_tokens = obs_embeddings_or_act_tokens["act_tokens"]
            if self.continuous_action_space:
                num_steps = 1
                act_tokens = act_tokens.float()
                if len(act_tokens.shape) == 2:
                    act_tokens = act_tokens.unsqueeze(1)
            else:
                if len(act_tokens.shape) == 3:
                    act_tokens = act_tokens.squeeze(1)
                num_steps = act_tokens.size(1)
            # Convert action tokens to embeddings using the action embedding table.
            sequences = self.act_embedding_table(act_tokens)

        # Process combined observation embeddings and action tokens.
        elif "obs_embeddings_and_act_tokens" in obs_embeddings_or_act_tokens:
            # Process combined inputs to calculate either the target value (for training)
            # or target policy (for reanalyze phase).
            if self.continuous_action_space:
                sequences, num_steps = self._process_obs_act_combined_cont(obs_embeddings_or_act_tokens)
            else:
                sequences, num_steps = self._process_obs_act_combined(obs_embeddings_or_act_tokens)
        else:
            raise ValueError("Input dictionary must contain one of 'obs_embeddings', 'act_tokens', or 'obs_embeddings_and_act_tokens'.")

        # Pass the sequence through the transformer.
        x, updated_mamba_states = self.mamba_model(
            sequences,
            past_mamba_states=past_mamba_states,
            seqlen_offset=seqlen_offset
        )

        # Generate logits for various components.
        logits_observations = self.head_observations(x, num_steps=num_steps, prev_steps=prev_steps)
        logits_rewards = self.head_rewards(x, num_steps=num_steps, prev_steps=prev_steps)
        logits_policy = self.head_policy(x, num_steps=num_steps, prev_steps=prev_steps)
        logits_value = self.head_value(x, num_steps=num_steps, prev_steps=prev_steps)

        # The 'logits_ends' is intentionally set to None.
        return WorldModelOutput(x, logits_observations, logits_rewards, None, logits_policy, logits_value), updated_mamba_states

    def _process_obs_act_combined_cont(self, obs_embeddings_or_act_tokens):
        """
        Process combined observation embeddings and action tokens.

        Arguments:
            - obs_embeddings_or_act_tokens (:obj:`dict`): Dictionary containing combined observation embeddings and action tokens.
            - prev_steps (:obj:`torch.Tensor`): Previous steps.
        Returns:
            - torch.Tensor: Combined observation and action embeddings with position information added.
        """
        obs_embeddings, act_tokens = obs_embeddings_or_act_tokens['obs_embeddings_and_act_tokens']
        if len(obs_embeddings.shape) == 3:
            obs_embeddings = obs_embeddings.view(act_tokens.shape[0], act_tokens.shape[1], self.num_observations_tokens,
                                                 -1)

        num_steps = int(obs_embeddings.size(1) * (obs_embeddings.size(2) + 1))
        if self.continuous_action_space:
            act_tokens = act_tokens.float()
            if len(act_tokens.shape) == 2:  # TODO
                act_tokens = act_tokens.unsqueeze(-1)

        # B, L, E
        act_embeddings = self.act_embedding_table(act_tokens)

        B, L, K, E = obs_embeddings.size()
        # B, L*2, E
        obs_act_embeddings = torch.empty(B, L * (K + 1), E, device=self.device)

        for i in range(L):
            obs = obs_embeddings[:, i, :, :]
            act = act_embeddings[:, i, :].unsqueeze(1)
            obs_act = torch.cat([obs, act], dim=1)
            obs_act_embeddings[:, i * (K + 1):(i + 1) * (K + 1), :] = obs_act

        return_result = obs_act_embeddings

        return return_result, num_steps

    def _process_obs_act_combined(self, obs_embeddings_or_act_tokens, ):
        """
        Process combined observation embeddings and action tokens.

        Arguments:
            - obs_embeddings_or_act_tokens (:obj:`dict`): Dictionary containing combined observation embeddings and action tokens.
            - prev_steps (:obj:`torch.Tensor`): Previous steps.
        Returns:
            - torch.Tensor: Combined observation and action embeddings with position information added.
        """
        obs_embeddings, act_tokens = obs_embeddings_or_act_tokens['obs_embeddings_and_act_tokens']
        if len(obs_embeddings.shape) == 3:
            obs_embeddings = obs_embeddings.view(act_tokens.shape[0], act_tokens.shape[1], self.num_observations_tokens,
                                                 -1)

        num_steps = int(obs_embeddings.size(1) * (obs_embeddings.size(2) + 1))
        act_embeddings = self.act_embedding_table(act_tokens)

        B, L, K, E = obs_embeddings.size()
        obs_act_embeddings = torch.empty(B, L * (K + 1), E, device=self.device)

        for i in range(L):
            obs = obs_embeddings[:, i, :, :]
            act = act_embeddings[:, i, 0, :].unsqueeze(1)
            obs_act = torch.cat([obs, act], dim=1)
            obs_act_embeddings[:, i * (K + 1):(i + 1) * (K + 1), :] = obs_act
            
        return_result = obs_act_embeddings
        return return_result, num_steps

    @torch.no_grad()
    def reset_for_initial_inference(self, obs_act_dict: torch.FloatTensor) -> torch.FloatTensor:
        """
        Reset the model state based on initial observations and actions.

        Arguments:
            - obs_act_dict (:obj:`torch.FloatTensor`): A dictionary containing 'obs', 'action', and 'current_obs'.
        Returns:
            - torch.FloatTensor: The outputs from the world model and the latent state.
        """
        # Extract observations, actions, and current observations from the dictionary.
        if isinstance(obs_act_dict, dict):
            batch_obs = obs_act_dict['obs']  # obs_act_dict['obs'] is at timestep t
            batch_action = obs_act_dict['action'] # obs_act_dict['action'] is at timestep t
            batch_current_obs = obs_act_dict['current_obs'] # obs_act_dict['current_obs'] is at timestep t+1

        # Encode observations to latent embeddings.
        obs_embeddings = self.tokenizer.encode_to_obs_embeddings(batch_obs)

        if batch_current_obs is not None:
            # ================ Collect and Evaluation Phase ================
            # Encode current observations to latent embeddings
            current_obs_embeddings = self.tokenizer.encode_to_obs_embeddings(batch_current_obs)
            # print(f"current_obs_embeddings.device: {current_obs_embeddings.device}")
            self.latent_state = current_obs_embeddings
            outputs_wm = self.wm_forward_for_initial_infererence(obs_embeddings, batch_action,
                                                                                   current_obs_embeddings)
        else:
            # ================ calculate the target value in Train phase or calculate the target policy in reanalyze phase ================
            self.latent_state = obs_embeddings
            outputs_wm = self.wm_forward_for_initial_infererence(obs_embeddings, batch_action, None)

        return outputs_wm, self.latent_state

    @torch.no_grad()
    def wm_forward_for_initial_infererence(self, last_obs_embeddings: torch.LongTensor,
                                                             batch_action=None,
                                                             current_obs_embeddings=None) -> torch.FloatTensor:
        """
        Refresh key-value pairs with the initial latent state for inference.

        Arguments:
            - last_obs_embeddings (:obj:`torch.LongTensor`): The latent state embeddings.
            - batch_action (optional): Actions taken.
            - current_obs_embeddings (optional): Current observation embeddings.
        Returns:
            - torch.FloatTensor: The outputs from the world model.
        """
        n, num_observations_tokens, _ = last_obs_embeddings.shape
        if n <= self.env_num and current_obs_embeddings is not None:
            # ================ Collect and Evaluation Phase ================
            if current_obs_embeddings is not None:
                 # Determine whether it is the first step in an episode.
                if self.continuous_action_space:
                    first_step_flag = not isinstance(batch_action[0], np.ndarray)
                else:
                    first_step_flag = max(batch_action) == -1
                if first_step_flag:
                    # ------------------------- First Step of an Episode -------------------------
                    self.mamba_states_wm = self.mamba_model.generate_empty_state(
                        batch_size=current_obs_embeddings.shape[0], max_seq_len=self.context_length
                    )
                    # print(f"current_obs_embeddings.device: {current_obs_embeddings.device}")
                    outputs_wm = self.forward({'obs_embeddings': current_obs_embeddings},
                                              past_mamba_states=self.mamba_states_wm, seqlen_offset=0)

                    # Copy and store keys_values_wm for a single environment
                    self.mamba_states_size_list_current = [num_observations_tokens] * current_obs_embeddings.shape[0]
                    self.update_cache_context(current_obs_embeddings, is_init_infer=True)

                else:
                    # --------------------- Continuing an Episode (Multi-environment) ---------------------
                    # current_obs_embeddings is the new latent_state, containing information from ready_env_num environments
                    ready_env_num = current_obs_embeddings.shape[0]
                    self.mamba_states_wm_list, self.mamba_states_size_list = [], []
                    self.mamba_states_size_list = self.retrieve_or_generate_mamba_cache(
                        last_obs_embeddings.cpu().numpy(), ready_env_num
                    )
                    self._batch_mamba_states_from_list()
                    batch_action = batch_action[:ready_env_num]

                    if self.continuous_action_space:
                        act_tokens = torch.from_numpy(np.array(batch_action)).to(last_obs_embeddings.device).unsqueeze(1)
                    else:
                        act_tokens = torch.from_numpy(np.array(batch_action)).to(last_obs_embeddings.device).unsqueeze(-1)
                    
                    seqlen_offset = max(self.mamba_states_size_list) if self.mamba_states_size_list else 0
                    outputs_wm, self.mamba_states_wm = self.forward(
                        {'act_tokens': act_tokens}, past_mamba_states=self.mamba_states_wm, seqlen_offset=seqlen_offset
                    )

                    self.mamba_states_size_list = [s + 1 for s in self.mamba_states_size_list]
                    seqlen_offset = max(self.mamba_states_size_list)

                    outputs_wm, self.mamba_states_wm = self.forward(
                        {'obs_embeddings': current_obs_embeddings}, past_mamba_states=self.mamba_states_wm, seqlen_offset=seqlen_offset
                    )
                    self.mamba_states_size_list_current = [s + num_observations_tokens for s in self.mamba_states_size_list]

                    self.update_cache_context(current_obs_embeddings, is_init_infer=True)

        elif batch_action is not None and current_obs_embeddings is None:
            # ================ calculate the target value in Train phase or calculate the target policy in reanalyze phase ================
            # [192, 16, 64] -> [32, 6, 16, 64]
            last_obs_embeddings = last_obs_embeddings.contiguous().view(batch_action.shape[0], -1, num_observations_tokens,
                                                          self.obs_per_embdding_dim)  # (BL, K) for unroll_step=1

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

            outputs_wm, _ = self.forward({'obs_embeddings_and_act_tokens': (last_obs_embeddings, act_tokens)})

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
    def forward_initial_inference(self, obs_act_dict, start_pos):
        """
        Perform initial inference based on the given observation-action dictionary.

        Arguments:
            - obs_act_dict (:obj:`dict`): Dictionary containing observations and actions.
        Returns:
            - tuple: A tuple containing output sequence, latent state, logits rewards, logits policy, and logits value.
        """
        # UniZero has context in the root node
        outputs_wm, latent_state = self.reset_for_initial_inference(obs_act_dict)
        self.past_mamba_states_cache_recurrent_infer.clear()

        return (outputs_wm.output_sequence, latent_state, outputs_wm.logits_rewards,
                outputs_wm.logits_policy, outputs_wm.logits_value)

    @torch.no_grad()
    def forward_recurrent_inference(self, state_action_history, simulation_index=0,
                                    search_depth=[], start_pos=0):
        """
        Perform recurrent inference based on the state-action history.

        Arguments:
            - state_action_history (:obj:`list`): List containing tuples of state and action history.
            - simulation_index (:obj:`int`, optional): Index of the current simulation. Defaults to 0.
            - search_depth (:obj:`list`, optional): List containing depth of latent states in the search tree. 
        Returns:
            - tuple: A tuple containing output sequence, updated latent state, reward, logits policy, and logits value.
        """
        latest_state, action = state_action_history[-1]
        ready_env_num = latest_state.shape[0]

        self.mamba_states_wm_list, self.mamba_states_size_list = [], []
        self.mamba_states_size_list = self.retrieve_or_generate_mamba_cache(
            latest_state, ready_env_num, simulation_index
        )
       
        latent_state_list = []
        if not self.continuous_action_space:
            token = action.reshape(-1, 1)
        else:
            token = action.reshape(-1, self.action_space_size)

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

        # Trim and pad kv_cache: modify self.keys_values_wm in-place
        self._batch_mamba_states_from_list()
        self.mamba_states_size_list_current = self.mamba_states_size_list

        for k in range(2):
            # action_token obs_token
            if k == 0:
                obs_embeddings_or_act_tokens = {'act_tokens': token}
            else:
                obs_embeddings_or_act_tokens = {'obs_embeddings': token}

            if k == 0:
                step_len = 1
            else:
                step_len = self.num_observations_tokens
            seqlen_offset = max(self.mamba_states_size_list_current) if self.mamba_states_size_list_current else 0

            # Perform forward pass
            outputs_wm, self.mamba_states_wm  = self.forward(
                obs_embeddings_or_act_tokens,
                past_mamba_states=self.mamba_states_wm,
                seqlen_offset=seqlen_offset
            )
            
            self.mamba_states_size_list_current = [s + step_len for s in self.mamba_states_size_list_current]

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
        )

        return (outputs_wm.output_sequence, self.latent_state, reward, outputs_wm.logits_policy, outputs_wm.logits_value)

    def _batch_mamba_states_from_list(self) -> None:
        if not self.mamba_states_wm_list:
            self.mamba_states_wm = None
            return

        batched_states = []
        for layer_idx in range(self.num_layers):
            conv_states_list = [env_states[layer_idx][0] for env_states in self.mamba_states_wm_list]
            ssm_states_list = [env_states[layer_idx][1] for env_states in self.mamba_states_wm_list]
            batched_conv_state = torch.cat(conv_states_list, dim=0)
            batched_ssm_state = torch.cat(ssm_states_list, dim=0)
            batched_states.append((batched_conv_state, batched_ssm_state))
        
        self.mamba_states_wm = batched_states

    def update_cache_context(self, latent_state, is_init_infer=True, simulation_index=0,
                             search_depth=[], valid_context_lengths=None):
        """
        Update the cache context with the given latent state.

        Arguments:
            - latent_state (:obj:`torch.Tensor`): The latent state tensor.
            - is_init_infer (:obj:`bool`): Flag to indicate if this is the initial inference.
            - simulation_index (:obj:`int`): Index of the simulation.
            - search_depth (:obj:`list`): List of depth indices in the search tree.
            - valid_context_lengths (:obj:`list`): List of valid context lengths.
        """
        if not self.mamba_states_wm: 
            return
        for i in range(latent_state.size(0)):
            # ============ Iterate over each environment ============
            cache_key = hash_state(latent_state[i].view(-1).cpu().numpy())  # latent_state[i] is torch.Tensor
            context_length = self.context_length
            single_env_state = []
            for layer_idx in range(self.num_layers):
                conv_s = self.mamba_states_wm[layer_idx][0][i:i+1].detach()
                ssm_s = self.mamba_states_wm[layer_idx][1][i:i+1].detach()
                single_env_state.append((conv_s, ssm_s))
            
            current_seq_len = self.mamba_states_size_list_current[i]
            state_to_cache = (copy.deepcopy(single_env_state), current_seq_len)

            if is_init_infer:
                self.past_mamba_states_cache_init_infer_envs[i][cache_key] = state_to_cache
            else:
                self.past_mamba_states_cache_recurrent_infer[cache_key] = state_to_cache


    def retrieve_or_generate_mamba_cache(self, latent_state: np.ndarray, ready_env_num: int,
                                        simulation_index: int = 0) -> list:
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
        new_mamba_states_size_list = []
        for index in range(ready_env_num):
            self.total_query_count += 1
            state_single_env_np = latent_state[index]
            cache_key = hash_state(state_single_env_np)

            matched_value = None
            if not self.reanalyze_phase:
                matched_value = self.past_mamba_states_cache_init_infer_envs[index].get(cache_key)
                if matched_value is None:
                    matched_value = self.past_mamba_states_cache_recurrent_infer.get(cache_key)

            if matched_value is not None:
                self.hit_count += 1
                self.mamba_states_wm_list.append(copy.deepcopy(matched_value[0]))
                new_mamba_states_size_list.append(matched_value[1])
            else:
                state_single_env_tensor = torch.from_numpy(state_single_env_np).unsqueeze(0).to(self.device)
                # CORRECTED: Pass max_seq_len
                new_single_env_state = self.mamba_model.generate_empty_state(
                    batch_size=1, max_seq_len=self.context_length
                )
                _, new_single_env_state = self.forward(
                    {'obs_embeddings': state_single_env_tensor}, past_mamba_states=new_single_env_state, seqlen_offset=0
                )
                self.mamba_states_wm_list.append(copy.deepcopy(new_single_env_state))
                new_mamba_states_size_list.append(self.num_observations_tokens)

        return new_mamba_states_size_list


    def compute_loss(self, batch, target_tokenizer: Tokenizer = None, inverse_scalar_transform_handle=None,
                     **kwargs: Any) -> LossWithIntermediateLosses:
        # Encode observations into latent state representations
        obs_embeddings = self.tokenizer.encode_to_obs_embeddings(batch['observations'])

        # ========= for visual analysis =========
        # Uncomment the lines below for visual analysis in Pong
        # self.plot_latent_tsne_each_and_all_for_pong(obs_embeddings, suffix='pong_H10_H4_tsne')
        # self.save_as_image_with_timestep(batch['observations'], suffix='pong_H10_H4_tsne')
        # Uncomment the lines below for visual analysis in visual match
        # self.plot_latent_tsne_each_and_all(obs_embeddings, suffix='visual_match_memlen1-60-15_tsne')
        # self.save_as_image_with_timestep(batch['observations'], suffix='visual_match_memlen1-60-15_tsne')

        # ========= logging for analysis =========
        if self.analysis_dormant_ratio:
            # Calculate dormant ratio of the encoder
            shape = batch['observations'].shape  # (..., C, H, W)
            inputs = batch['observations'].contiguous().view(-1, *shape[-3:])  # (32,5,3,64,64) -> (160,3,64,64)
            dormant_ratio_encoder = cal_dormant_ratio(self.tokenizer.representation_network, inputs.detach(),
                                                      percentage=self.dormant_threshold)
            self.past_kv_cache_recurrent_infer.clear()
            self.keys_values_wm_list.clear()
            torch.cuda.empty_cache()
        else:
            dormant_ratio_encoder = torch.tensor(0.)

        # Calculate the L2 norm of the latent state roots
        latent_state_l2_norms = torch.norm(obs_embeddings, p=2, dim=2).mean()

        if self.obs_type == 'image':
            # Reconstruct observations from latent state representations
            # reconstructed_images = self.tokenizer.decode_to_obs(obs_embeddings)

            #  ========== for visualization ==========
            # Uncomment the lines below for visual analysis
            # original_images, reconstructed_images = batch['observations'], reconstructed_images
            # target_policy = batch['target_policy']
            # target_predict_value = inverse_scalar_transform_handle(batch['target_value'].reshape(-1, 101)).reshape(
            #     batch['observations'].shape[0], batch['observations'].shape[1], 1)
            # true_rewards = inverse_scalar_transform_handle(batch['rewards'].reshape(-1, 101)).reshape(
            #     batch['observations'].shape[0], batch['observations'].shape[1], 1)
            #  ========== for visualization ==========

            # ========== Calculate reconstruction loss and perceptual loss ============
            # latent_recon_loss = self.tokenizer.reconstruction_loss(batch['observations'].reshape(-1, 3, 64, 64), reconstructed_images) # NOTE: for stack=1
            # perceptual_loss = self.tokenizer.perceptual_loss(batch['observations'].reshape(-1, 3, 64, 64), reconstructed_images) # NOTE: for stack=1
            
            latent_recon_loss = self.latent_recon_loss
            perceptual_loss = self.perceptual_loss

        elif self.obs_type == 'vector':
            perceptual_loss = torch.tensor(0., device=batch['observations'].device,
                                           dtype=batch['observations'].dtype)

            # Reconstruct observations from latent state representations
            # reconstructed_images = self.tokenizer.decode_to_obs(obs_embeddings.reshape(-1, self.embed_dim))

            # # Calculate reconstruction loss
            # latent_recon_loss = self.tokenizer.reconstruction_loss(batch['observations'].reshape(-1, 25),
            #                                                        reconstructed_images)
            latent_recon_loss = self.latent_recon_loss

        elif self.obs_type == 'text':
            perceptual_loss = torch.tensor(0., device=batch['observations'].device,
                                           dtype=torch.float32)

            # Reconstruct observations from latent state representations
            # reconstructed_images = self.tokenizer.decode_to_obs(obs_embeddings.reshape(-1, self.embed_dim))

            # # Calculate reconstruction loss
            # latent_recon_loss = self.tokenizer.reconstruction_loss(batch['observations'].reshape(-1, 25),
            #                                                        reconstructed_images)
            latent_recon_loss = self.latent_recon_loss

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
            latent_recon_loss = self.latent_recon_loss
            perceptual_loss = self.perceptual_loss

        # Action tokens
        if self.continuous_action_space:
            act_tokens = batch['actions']
        else:
            act_tokens = rearrange(batch['actions'], 'b l -> b l 1')

        # Forward pass to obtain predictions for observations, rewards, and policies
        outputs = self.forward({'obs_embeddings_and_act_tokens': (obs_embeddings, act_tokens)})

        # ========= logging for analysis =========
        if self.analysis_dormant_ratio:
            # Calculate dormant ratio of the world model
            dormant_ratio_world_model = cal_dormant_ratio(self, {
                'obs_embeddings_and_act_tokens': (obs_embeddings.detach(), act_tokens.detach())},
                                                          percentage=self.dormant_threshold)
            self.past_kv_cache_recurrent_infer.clear()
            self.keys_values_wm_list.clear()
            torch.cuda.empty_cache()
        else:
            dormant_ratio_world_model = torch.tensor(0.)

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
            target_obs_embeddings = target_tokenizer.encode_to_obs_embeddings(batch['observations'])

        # Compute labels for observations, rewards, and ends
        labels_observations, labels_rewards, _ = self.compute_labels_world_model(target_obs_embeddings,
                                                                                           batch['rewards'],
                                                                                           batch['ends'],
                                                                                           batch['mask_padding'])

        # Reshape the logits and labels for observations
        logits_observations = rearrange(outputs.logits_observations[:, :-1], 'b t o -> (b t) o')
        labels_observations = labels_observations.reshape(-1, self.projection_input_dim)

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
            # print('loss_obs:', loss_obs.mean())
            # assert not torch.isnan(loss_obs).any(), "loss_obs contains NaN values"
            # assert not torch.isinf(loss_obs).any(), "loss_obs contains Inf values"
            # for name, param in self.tokenizer.encoder.named_parameters():
            #     print('name, param.mean(), param.std():', name, param.mean(), param.std())

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
                orig_policy_loss, policy_entropy_loss, target_policy_entropy, target_sampled_actions, mu, sigma = self._calculate_policy_loss_cont_simple(outputs, batch)
            else:
                orig_policy_loss, policy_entropy_loss, target_policy_entropy, target_sampled_actions, mu, sigma = self._calculate_policy_loss_cont(outputs, batch)
            
            loss_policy = orig_policy_loss + self.policy_entropy_weight * policy_entropy_loss
            policy_entropy = - policy_entropy_loss

        loss_value = self.compute_cross_entropy_loss(outputs, labels_value, batch, element='value')

        # ==== TODO: calculate the new priorities for each transition. ====
        # value_priority = L1Loss(reduction='none')(labels_value.squeeze(-1), outputs['logits_value'][:, 0])
        # value_priority = value_priority.data.cpu().numpy() + 1e-6

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
            middle_timestep = seq_len // 2
            middle_step_mask = mask_padding[:, middle_timestep]
            middle_step_losses[loss_name] = loss_tmp[:, middle_timestep][middle_step_mask].mean()

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
                dormant_ratio_world_model=dormant_ratio_world_model,
                latent_state_l2_norms=latent_state_l2_norms,
                policy_mu=mu,
                policy_sigma=sigma,
                target_sampled_actions=target_sampled_actions,
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
                dormant_ratio_world_model=dormant_ratio_world_model,
                latent_state_l2_norms=latent_state_l2_norms,
            )

    # TODO: test correctness
    def _calculate_policy_loss_cont_simple(self, outputs, batch: dict):
        """
        Simplified policy loss calculation for continuous actions.

        Args:
            - outputs: Model outputs containing policy logits.
            - batch (:obj:`dict`): Batch data containing target policy, mask and sampled actions.

        Returns:
            - policy_loss (:obj:`torch.Tensor`): The simplified policy loss.
        """
        batch_size, num_unroll_steps, action_space_size = outputs.logits_policy.shape[
            0], self.config.num_unroll_steps, self.config.action_space_size

        # Get the policy logits and batch data
        policy_logits_all = outputs.logits_policy
        mask_batch = batch['mask_padding'].contiguous().view(-1)
        target_policy = batch['target_policy'].contiguous().view(batch_size * num_unroll_steps, -1)
        target_sampled_actions = batch['child_sampled_actions'].contiguous().view(batch_size * num_unroll_steps, -1, action_space_size)

        # Flatten for vectorized computation
        policy_logits_all = policy_logits_all.view(batch_size * num_unroll_steps, -1)
        
        # Extract mean and standard deviation from logits
        mu, sigma = policy_logits_all[:, :action_space_size], policy_logits_all[:, action_space_size:]
        dist = Independent(Normal(mu, sigma), 1)  # Create the normal distribution

        # Find the indices of the maximum values in the target policy
        target_best_action_idx = torch.argmax(target_policy, dim=1)

        # Select the best actions based on the indices
        target_best_action = target_sampled_actions[torch.arange(target_best_action_idx.size(0)), target_best_action_idx]

        # Clip the target actions to prevent numerical issues during arctanh
        # target_best_action_clamped = torch.clamp(target_best_action, -1 + 1e-6, 1 - 1e-6)
        target_best_action_clamped = torch.clamp(target_best_action, -0.999, 0.999)
        target_best_action_before_tanh = torch.arctanh(target_best_action_clamped)

        # Calculate the log probability of the best action
        log_prob_best_action = dist.log_prob(target_best_action_before_tanh)

        # Mask the log probability with the padding mask
        log_prob_best_action = log_prob_best_action * mask_batch

        # Return the negative log probability as the policy loss (we want to maximize log_prob)
        # policy_loss = -log_prob_best_action.mean()
        policy_loss = -log_prob_best_action

        policy_entropy = dist.entropy().mean()
        policy_entropy_loss = -policy_entropy * mask_batch
        # Calculate the entropy of the target policy distribution
        non_masked_indices = torch.nonzero(mask_batch).squeeze(-1)
        if len(non_masked_indices) > 0:
            target_normalized_visit_count = target_policy.contiguous().view(batch_size * num_unroll_steps, -1)
            target_dist = Categorical(target_normalized_visit_count[non_masked_indices])
            target_policy_entropy = target_dist.entropy().mean().item()
        else:
            target_policy_entropy = 0.0

        return policy_loss, policy_entropy_loss, target_policy_entropy, target_sampled_actions, mu, sigma

    def _calculate_policy_loss_cont(self, outputs, batch: dict) -> Tuple[torch.Tensor, torch.Tensor, float, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate the policy loss for continuous actions.

        Args:
            - outputs: Model outputs containing policy logits.
            - batch (:obj:`dict`): Batch data containing target policy, mask and sampled actions.
        Returns:
            - policy_loss (:obj:`torch.Tensor`): The calculated policy loss.
            - policy_entropy_loss (:obj:`torch.Tensor`): The entropy loss of the policy.
            - target_policy_entropy (:obj:`float`): The entropy of the target policy distribution.
            - target_sampled_actions (:obj:`torch.Tensor`): The actions sampled from the target policy.
            - mu (:obj:`torch.Tensor`): The mean of the normal distribution.
            - sigma (:obj:`torch.Tensor`): The standard deviation of the normal distribution.
        """
        batch_size, num_unroll_steps, action_space_size = outputs.logits_policy.shape[
            0], self.config.num_unroll_steps, self.config.action_space_size

        policy_logits_all = outputs.logits_policy
        mask_batch = batch['mask_padding']
        child_sampled_actions_batch = batch['child_sampled_actions']
        target_policy = batch['target_policy']

        # Flatten the unroll step dimension for easier vectorized operations
        policy_logits_all = policy_logits_all.view(batch_size * num_unroll_steps, -1)
        mask_batch = mask_batch.contiguous().view(-1)
        child_sampled_actions_batch = child_sampled_actions_batch.contiguous().view(batch_size * num_unroll_steps, -1,
                                                                                    action_space_size)

        mu, sigma = policy_logits_all[:, :action_space_size], policy_logits_all[:, action_space_size:]
        mu = mu.unsqueeze(1).expand(-1, child_sampled_actions_batch.shape[1], -1)
        sigma = sigma.unsqueeze(1).expand(-1, child_sampled_actions_batch.shape[1], -1)
        dist = Independent(Normal(mu, sigma), 1)

        target_normalized_visit_count = target_policy.contiguous().view(batch_size * num_unroll_steps, -1)
        target_sampled_actions = child_sampled_actions_batch

        policy_entropy = dist.entropy().mean(dim=1)
        policy_entropy_loss = -policy_entropy * mask_batch

        # NOTE： Alternative way to calculate the log probability of the target actions
        # y = 1 - target_sampled_actions.pow(2)
        # target_sampled_actions_clamped = torch.clamp(target_sampled_actions, -1 + 1e-6, 1 - 1e-6)
        # target_sampled_actions_before_tanh = torch.arctanh(target_sampled_actions_clamped)
        # log_prob = dist.log_prob(target_sampled_actions_before_tanh)
        # log_prob = log_prob - torch.log(y + 1e-6).sum(-1)
        # log_prob_sampled_actions = log_prob

        base_dist = Normal(mu, sigma)
        tanh_transform = TanhTransform()
        dist = TransformedDistribution(base_dist, [tanh_transform])
        dist = Independent(dist, 1)
        target_sampled_actions_clamped = torch.clamp(target_sampled_actions, -0.999, 0.999)
        # assert torch.all(target_sampled_actions_clamped < 1) and torch.all(target_sampled_actions_clamped > -1), "Actions are not properly clamped."
        log_prob = dist.log_prob(target_sampled_actions_clamped)
        log_prob_sampled_actions = log_prob

        # KL as projector
        target_log_prob_sampled_actions = torch.log(target_normalized_visit_count + 1e-6)
        policy_loss = -torch.sum(
            torch.exp(target_log_prob_sampled_actions.detach()) * log_prob_sampled_actions, 1
        ) * mask_batch

        # Calculate the entropy of the target policy distribution
        non_masked_indices = torch.nonzero(mask_batch).squeeze(-1)
        if len(non_masked_indices) > 0:
            target_dist = Categorical(target_normalized_visit_count[non_masked_indices])
            target_policy_entropy = target_dist.entropy().mean().item()
        else:
            target_policy_entropy = 0.0

        return policy_loss, policy_entropy_loss, target_policy_entropy, target_sampled_actions, mu, sigma

    def compute_cross_entropy_loss(self, outputs, labels, batch, element='rewards'):
        # Assume outputs is an object with logits attributes like 'rewards', 'policy', and 'value'.
        # labels is a target tensor for comparison. batch is a dictionary with a mask indicating valid timesteps.

        logits = getattr(outputs, f'logits_{element}')

        if torch.isnan(logits).any():
            raise ValueError(f"NaN detected in outputs for batch {batch} and element '{element}'")
        
        if torch.isnan(labels).any():
            raise ValueError(f"NaN detected in labels_value for batch {batch} and element '{element}'")

        # Reshape your tensors
        logits = rearrange(logits, 'b t e -> (b t) e')
        labels = labels.reshape(-1, labels.shape[-1])  # Assume labels initially have shape [batch, time, dim]

        # Reshape your mask. True indicates valid data.
        mask_padding = rearrange(batch['mask_padding'], 'b t -> (b t)')

        # Compute cross-entropy loss
        loss = -(torch.log_softmax(logits, dim=1) * labels).sum(1)
        loss = (loss * mask_padding)

        if torch.isnan(loss).any():
            raise ValueError(f"NaN detected in outputs for batch {batch} and element '{element}'")

        if element == 'policy':
            # Compute policy entropy loss
            policy_entropy = self.compute_policy_entropy_loss(logits, mask_padding)
            # Combine losses with specified weight
            combined_loss = loss - self.policy_entropy_weight * policy_entropy
            return combined_loss, loss, policy_entropy

        return loss

    def compute_policy_entropy_loss(self, logits, mask):
        # Compute entropy of the policy
        probs = torch.softmax(logits, dim=1)
        log_probs = torch.log_softmax(logits, dim=1)
        entropy = -(probs * log_probs).sum(1)
        # Apply mask and return average entropy loss
        entropy_loss = (entropy * mask)
        return entropy_loss

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
        # labels_endgs = ends.masked_fill(mask_fill, -100)

        # return labels_observations, labels_rewards.reshape(-1, self.support_size), labels_ends.reshape(-1)
        return labels_observations, labels_rewards.view(-1, self.support_size), None


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

    def clear_caches(self):
        """
        Clears the caches of the world model.
        """
        for mamba_state_cache_dict_env in self.past_mamba_states_cache_init_infer_envs:
            mamba_state_cache_dict_env.clear()
        self.past_mamba_states_cache_recurrent_infer.clear()
        self.mamba_states_wm_list.clear()
        print(f'Cleared {self.__class__.__name__} past_mamba_states_cache.')

    def __repr__(self) -> str:
        return "transformer-based latent world_model of UniZero"

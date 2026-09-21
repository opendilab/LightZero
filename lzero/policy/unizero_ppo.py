"""UniZero+PPO policy kept separate from the legacy MCTS UniZero policy."""

from copy import deepcopy

from .unizero import *  # Reuse UniZero's shared optimizer, cache, and logging machinery.
from .ppo_utils import masked_categorical


@POLICY_REGISTRY.register('unizero_ppo')
class UniZeroPPOPolicy(UniZeroPolicy):
    """Fresh on-policy PPO improvement over a PPO-specific UniZero model."""

    config = deepcopy(UniZeroPolicy.config)
    config.update(
        type='unizero_ppo',
        policy_improvement='ppo',
        collect_with_pure_policy=True,
        ppo=dict(
            gamma=0.997,
            gae_lambda=0.95,
            clip_ratio=0.2,
            entropy_weight=0.01,
            epochs=4,
            minibatch_size=256,
            normalize_advantage=True,
            target_kl=0.03,
            fresh_ratio_tolerance=1e-5,
            world_model_update_per_collect=None,
        ),
    )

    def default_model(self):
        return 'UniZeroPPOModel', ['lzero.model.unizero_ppo_model']

    def _init_learn(self) -> None:
        """
        Overview:
            Learn mode init method. Called by ``self.__init__``. Initialize the learn model, optimizer and MCTS utils.
        """
        self.policy_improvement = getattr(self._cfg, 'policy_improvement', 'mcts')
        if self.policy_improvement not in {'mcts', 'ppo'}:
            raise ValueError(
                f"policy_improvement must be 'mcts' or 'ppo', got {self.policy_improvement!r}"
            )
        if self.policy_improvement == 'ppo':
            if self._cfg.model.continuous_action_space:
                raise NotImplementedError('UniZero+PPO currently supports discrete action spaces only')
            if self._cfg.use_adaptive_entropy_weight:
                raise ValueError('UniZero+PPO uses ppo.entropy_weight; adaptive MCTS entropy is unsupported')
            if self._cfg.accumulation_steps != 1:
                raise ValueError(
                    'UniZero+PPO requires accumulation_steps=1 so actor/critic and '
                    'world-model gradients cannot cross phase boundaries'
                )

        if self._cfg.optim_type == 'SGD':
            # Configure SGD optimizer
            self._optimizer_world_model = torch.optim.SGD(
                self._model.world_model.parameters(),
                lr=self._cfg.learning_rate,
                momentum=self._cfg.momentum,
                weight_decay=self._cfg.weight_decay
            )
        elif self._cfg.optim_type == 'AdamW':
            # NOTE: nanoGPT optimizer
            self._optimizer_world_model = configure_optimizers_nanogpt(
                model=self._model.world_model,
                learning_rate=self._cfg.learning_rate,
                weight_decay=self._cfg.weight_decay,
                device_type=self._cfg.device,
                betas=(0.9, 0.95),
            )
        elif self._cfg.optim_type == 'AdamW_mix_lr_wdecay':
            self._optimizer_world_model = configure_optimizer_unizero(
                model=self._model.world_model,
                learning_rate=self._cfg.learning_rate,
                weight_decay=self._cfg.weight_decay,
                device_type=self._cfg.device,
                betas=(0.9, 0.95),
            )

        if self._cfg.cos_lr_scheduler:
            from torch.optim.lr_scheduler import CosineAnnealingLR
            total_iters = self._cfg.total_iterations
            final_lr = self._cfg.final_learning_rate

            self.lr_scheduler = CosineAnnealingLR(
                self._optimizer_world_model,
                T_max=total_iters,
                eta_min=final_lr
            )
            logging.info(f"CosineAnnealingLR enabled: T_max={total_iters}, eta_min={final_lr}")


        if self._cfg.piecewise_decay_lr_scheduler:
            from torch.optim.lr_scheduler import LambdaLR
            max_step = self._cfg.threshold_training_steps_for_final_lr
            # NOTE: the 1, 0.1, 0.01 is the decay rate, not the lr.
            lr_lambda = lambda step: 1 if step < max_step * 0.5 else (0.1 if step < max_step else 0.01)  # noqa
            self.lr_scheduler = LambdaLR(self._optimizer_world_model, lr_lambda=lr_lambda)

        # use model_wrapper for specialized demands of different modes
        self._target_model = copy.deepcopy(self._model)
        if self._cfg.torch_compile:
            # Ensure that the installed torch version is greater than or equal to 2.0
            assert int(''.join(filter(str.isdigit, torch.__version__))) >= 200, "We need torch version >= 2.0"
            self._model = torch.compile(self._model)
            self._target_model = torch.compile(self._target_model)
        # NOTE: soft target
        self._target_model = model_wrap(
            self._target_model,
            wrapper_name='target',
            update_type='momentum',
            update_kwargs={'theta': self._cfg.target_update_theta}
        )
        self._learn_model = self._model

        if self._cfg.use_augmentation:
            self.image_transforms = ImageTransforms(
                self._cfg.augmentation,
                image_shape=(self._cfg.model.observation_shape[1], self._cfg.model.observation_shape[2])
            )
        self.value_support = DiscreteSupport(*self._cfg.model.value_support_range, self._cfg.device)
        self.reward_support = DiscreteSupport(*self._cfg.model.reward_support_range, self._cfg.device)
        self.value_inverse_scalar_transform_handle = InverseScalarTransform(self.value_support, self._cfg.model.categorical_distribution)
        self.reward_inverse_scalar_transform_handle = InverseScalarTransform(self.reward_support, self._cfg.model.categorical_distribution)

        self.intermediate_losses = defaultdict(float)
        self.l2_norm_before = 0.
        self.l2_norm_after = 0.
        self.grad_norm_before = 0.
        self.grad_norm_after = 0.
        # Sparse stability checks must not appear as zero in every intervening learner log.
        # Cache the last real observation, and force a check on the first batch after resume.
        self._latest_norm_log_dict = {}
        self._last_norm_monitor_iter = -1
        self._latest_replay_diagnostic_metrics = {}
        self._last_gradient_diagnostic_iter = -1

        if self._cfg.model.model_type == 'conv':
            # for image-input env
            self.pad_token_id = -1
        else:
            # for text-input env and vector-input env
            # Retrieve the tokenizer from the encoder module if it exists
            encoder_tokenizer = getattr(self._model.tokenizer.encoder, 'tokenizer', None)

            # Extract the padding token ID from the tokenizer if available, otherwise use 0 as default. Used in _reset_collect()
            # The pad_token_id is used to identify padding tokens in sequences, which is essential for:
            # 1. Masking padded positions during attention computation to prevent them from affecting the output
            # 2. Properly handling variable-length sequences in batch processing
            # 3. Distinguishing between actual tokens and padding in loss calculation
            # Default value 0 is a common convention when no specific padding token is defined
            self.pad_token_id = encoder_tokenizer.pad_token_id if encoder_tokenizer is not None else 0

        if self._cfg.use_wandb:
            # TODO: add the model to wandb
            wandb.watch(self._learn_model.representation_network, log="all")

        self.accumulation_steps = self._cfg.accumulation_steps

        # ==================== START: Target Entropy Regularization Initialization ====================
        # Read whether to enable adaptive alpha from config, and provide a default value
        self.use_adaptive_entropy_weight = self._cfg.use_adaptive_entropy_weight

        # Add configuration in _init_learn
        self.target_entropy_start_ratio = self._cfg.target_entropy_start_ratio
        self.target_entropy_end_ratio = self._cfg.target_entropy_end_ratio
        self.target_entropy_decay_steps = self._cfg.target_entropy_decay_steps  # e.g., complete annealing within 200k steps (2M envsteps)
        self.adaptive_entropy_alpha_min = float(getattr(self._cfg, 'adaptive_entropy_alpha_min', 1e-4))
        self.adaptive_entropy_alpha_max = float(getattr(self._cfg, 'adaptive_entropy_alpha_max', 10.0))
        if self.adaptive_entropy_alpha_min <= 0 or self.adaptive_entropy_alpha_max <= 0:
            raise ValueError("adaptive entropy alpha bounds must be positive")
        if self.adaptive_entropy_alpha_min > self.adaptive_entropy_alpha_max:
            raise ValueError("adaptive_entropy_alpha_min must be <= adaptive_entropy_alpha_max")

        if self.use_adaptive_entropy_weight:
            # 1. Set target entropy. For discrete action spaces, a common heuristic is the negative logarithm
            #    of action space dimension multiplied by a coefficient.
            #    This coefficient (e.g., 0.98) can be used as a hyperparameter.
            action_space_size = self._cfg.model.action_space_size
            self.target_entropy = -np.log(1.0 / action_space_size) * 0.98

            # 2. Initialize a learnable log_alpha parameter.
            #    Initialized to 0, meaning initial alpha = exp(0) = 1.0.
            self.log_alpha = torch.nn.Parameter(torch.zeros(1, device=self._cfg.device), requires_grad=True)

            # 3. Create a dedicated optimizer for log_alpha.
            #    Using a smaller learning rate (e.g., 1e-4) different from the main optimizer is usually more stable.
            alpha_lr = self._cfg.adaptive_entropy_alpha_lr
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

            logging.info("="*20)
            logging.info(">>> Target Entropy Regularization (Adaptive Alpha) Enabled <<<")
            logging.info(f"    Target Entropy: {self.target_entropy:.4f}")
            logging.info(f"    Alpha Optimizer Learning Rate: {alpha_lr:.2e}")
            logging.info(
                f"    Alpha Bounds: [{self.adaptive_entropy_alpha_min:.2e}, "
                f"{self.adaptive_entropy_alpha_max:.2e}]"
            )
            logging.info("="*20)
        # ===================== END: Target Entropy Regularization Initialization =====================

        # ==================== START: Initialize Encoder-Clip Annealing Parameters ====================
        self.use_encoder_clip_annealing = self._cfg.use_encoder_clip_annealing
        self._encoder_clip_apply_count = 0
        self.latent_norm_clip_threshold = self._cfg.latent_norm_clip_threshold  # TODO
        if self.use_encoder_clip_annealing:
            self.encoder_clip_anneal_type = self._cfg.encoder_clip_anneal_type
            self.encoder_clip_start = self._cfg.encoder_clip_start_value
            self.encoder_clip_end = self._cfg.encoder_clip_end_value
            self.encoder_clip_anneal_steps = self._cfg.encoder_clip_anneal_steps

            logging.info("="*20)
            logging.info(">>> Encoder-Clip Annealing Enabled <<<")
            logging.info(f"    Type: {self.encoder_clip_anneal_type}")
            logging.info(f"    Range: {self.encoder_clip_start} -> {self.encoder_clip_end}")
            logging.info(f"    Steps: {self.encoder_clip_anneal_steps}")
            logging.info("="*20)
        else:
            # If annealing is not enabled, use a fixed clip threshold
            self.latent_norm_clip_threshold = self._cfg.latent_norm_clip_threshold
        # ===================== END: Initialize Encoder-Clip Annealing Parameters =====================

        # ==================== START: Initialize Head-Clip Manager ====================
        self.use_head_clip = self._cfg.use_head_clip

        if self.use_head_clip:
            head_clip_config_dict = self._cfg.head_clip_config
            # Ensure enabled is consistent with top-level configuration
            head_clip_config_dict['enabled'] = self.use_head_clip

            # Create HeadClipManager
            self.head_clip_manager = create_head_clip_manager_from_dict(head_clip_config_dict)

            logging.info("=" * 60)
            logging.info(">>> Head-Clip Manager Initialized <<<")
            logging.info(f"    Enabled heads: {self.head_clip_manager.enabled_heads}")
            for head_name in self.head_clip_manager.enabled_heads:
                config = self.head_clip_manager.get_head_config(head_name)
                if config.use_annealing:
                    logging.info(
                        f"    {head_name}: annealing {config.start_value:.1f} → {config.end_value:.1f} "
                        f"over {config.anneal_steps} steps ({config.anneal_type})"
                    )
                else:
                    logging.info(f"    {head_name}: fixed threshold = {config.clip_threshold:.1f}")
            logging.info("=" * 60)
        else:
            self.head_clip_manager = None
        # ===================== END: Initialize Head-Clip Manager =====================

        # Policy Label Smoothing Parameters
        self.policy_ls_eps_start = self._cfg.policy_ls_eps_start
        self.policy_ls_eps_end = self._cfg.policy_ls_eps_end
        self.policy_ls_eps_decay_steps = self._cfg.policy_ls_eps_decay_steps
        logging.info(f"self.policy_ls_eps_start: {self.policy_ls_eps_start}")
    def _forward_learn(self, data: Tuple[torch.Tensor]) -> Dict[str, Union[float, int]]:
        """
        Overview:
            The forward function for learning policy in learn mode, which is the core of the learning process.
            The data is sampled from replay buffer.
            The loss is calculated by the loss function and the loss is backpropagated to update the model.
        Arguments:
            - data (:obj:`Tuple[torch.Tensor]`): The data sampled from replay buffer, which is a tuple of tensors.
                The first tensor is the current_batch, the second tensor is the target_batch.
        Returns:
            - info_dict (:obj:`Dict[str, Union[float, int]]`): The information dict to be logged, which contains \
                current learning loss and learning statistics.
        """
        self._learn_model.train()
        self._target_model.train()

        if len(data) == 4:
            current_batch, target_batch, train_iter, learn_context = data
        else:
            current_batch, target_batch, train_iter = data
            learn_context = {'type': 'mcts'}
        if isinstance(learn_context, str):
            learn_context = {'type': learn_context}
        learn_type = learn_context.get('type', 'mcts')

        ppo_batch = learn_type == 'ppo'
        if ppo_batch:
            if len(current_batch) != 14:
                raise RuntimeError(f'PPO current_batch must contain 14 fields, got {len(current_batch)}')
            (
                obs_batch_ori, action_batch, target_action_batch, mask_batch, indices,
                weights, make_time, timestep_batch, advantage_batch, old_log_prob_batch,
                return_batch, behavior_action_mask_batch, behavior_policy_feature_batch,
                policy_version_batch,
            ) = current_batch
            expected_version = int(learn_context['collection_train_iter'])
            if not np.all(np.asarray(policy_version_batch) == expected_version):
                raise RuntimeError(
                    f'PPO minibatch contains stale policy data; expected collection {expected_version}, '
                    f'got {np.unique(policy_version_batch).tolist()}'
                )
        else:
            obs_batch_ori, action_batch, target_action_batch, mask_batch, indices, weights, make_time, timestep_batch = current_batch[:8]
        target_reward, target_value, target_policy = target_batch
        if ppo_batch:
            # The explicit rollout return is authoritative for PPO. Do not let
            # replay reanalysis or a target-network path silently replace it.
            target_value = np.asarray(return_batch, dtype=np.float32)
        actual_batch_size = int(obs_batch_ori.shape[0])

        # Calculate current epsilon for policy label smoothing
        # ==================== Continuous Label Smoothing ====================
        use_continuous_label_smoothing = self._cfg.use_continuous_label_smoothing
        if use_continuous_label_smoothing:
            # Use fixed high epsilon throughout training
            current_policy_label_eps = self._cfg.continuous_ls_eps
        else:
            # Use original decay schedule
            if self.policy_ls_eps_start > 0:
                progress = min(1.0, train_iter / self.policy_ls_eps_decay_steps)
                current_policy_label_eps = self.policy_ls_eps_start * (1 - progress) + self.policy_ls_eps_end * progress
            else:
                current_policy_label_eps = 0.0
        # ================================================================================

        # PPO updates the actor/critic heads from exact contextual features
        # cached at collection time. Avoid preparing and transferring image
        # observations that the actor-only loss deliberately bypasses.
        if not ppo_batch:
            if self._cfg.model.frame_stack_num > 1:
                obs_batch, obs_target_batch = prepare_obs_stack_for_unizero(obs_batch_ori, self._cfg)
            else:
                obs_batch, obs_target_batch = prepare_obs(obs_batch_ori, self._cfg)

            # Apply augmentations if needed
            if self._cfg.use_augmentation:
                obs_batch = self.image_transforms.transform(obs_batch)
                if self._cfg.model.self_supervised_learning_loss:
                    obs_target_batch = self.image_transforms.transform(obs_target_batch)

        # Prepare action batch and convert to torch tensor
        action_batch = torch.from_numpy(action_batch).to(self._cfg.device).unsqueeze(
            -1).long()  # For discrete action space
        timestep_batch = torch.from_numpy(timestep_batch).to(self._cfg.device).unsqueeze(
            -1).long()
        data_list = [mask_batch, target_reward, target_value, target_policy, weights]
        mask_batch, target_reward, target_value, target_policy, weights = to_torch_float_tensor(data_list,
                                                                                                self._cfg.device)
        target_reward = target_reward.view(actual_batch_size, -1)
        target_value = target_value.view(actual_batch_size, -1)

        # Transform rewards and values to their scaled forms
        transformed_target_reward = scalar_transform(target_reward)
        transformed_target_value = scalar_transform(target_value)

        # Convert to categorical distributions
        target_reward_categorical = phi_transform(self.reward_support, transformed_target_reward, label_smoothing_eps= self._cfg.label_smoothing_eps)
        target_value_categorical = phi_transform(
            self.value_support,
            transformed_target_value,
            # PPO critic targets are Monte-Carlo/GAE returns rather than MCTS
            # distributions; smoothing them biases the value baseline.
            label_smoothing_eps=0.0 if ppo_batch else self._cfg.label_smoothing_eps,
        )

        # Prepare batch for GPT model
        batch_for_gpt = {}
        if not ppo_batch:
            if isinstance(self._cfg.model.observation_shape, int) or len(self._cfg.model.observation_shape) == 1:
                batch_for_gpt['observations'] = torch.cat((obs_batch, obs_target_batch), dim=1).reshape(
                    actual_batch_size, -1, self._cfg.model.observation_shape)
            elif len(self._cfg.model.observation_shape) == 3:
                batch_for_gpt['observations'] = torch.cat((obs_batch, obs_target_batch), dim=1).reshape(
                    actual_batch_size, -1, *self._cfg.model.observation_shape)

        batch_for_gpt['actions'] = action_batch.squeeze(-1)
        batch_for_gpt['timestep'] = timestep_batch.squeeze(-1)

        batch_for_gpt['mask_padding'] = mask_batch == 1.0  # 0 means invalid padding data
        batch_for_gpt['mask_padding'] = batch_for_gpt['mask_padding'][:, :-1]
        batch_for_gpt['target_value'] = target_value_categorical[:, :-1]
        if not ppo_batch:
            batch_for_gpt['rewards'] = target_reward_categorical[:, :-1]
            batch_for_gpt['observations'] = batch_for_gpt['observations'][:, :-1]
            batch_for_gpt['ends'] = torch.zeros(batch_for_gpt['mask_padding'].shape, dtype=torch.long,
                                                device=self._cfg.device)

            # ==================== Apply Policy Label Smoothing ====================
            # This was previously computed but never applied. Now we actually smooth the target_policy.
            smoothed_target_policy = target_policy[:, :-1]
            if current_policy_label_eps > 0:
                num_actions = smoothed_target_policy.shape[-1]
                uniform_dist = torch.ones_like(smoothed_target_policy) / num_actions
                smoothed_target_policy = (1.0 - current_policy_label_eps) * smoothed_target_policy + \
                                        current_policy_label_eps * uniform_dist
            batch_for_gpt['target_policy'] = smoothed_target_policy
            # ===================================================================================

        batch_for_gpt['scalar_target_value'] = target_value
        if ppo_batch:
            ppo_cfg = self._cfg.ppo
            # PPO and replay optimize disjoint objectives in a fixed order.  In
            # particular, large decoder/reward gradients must not dominate the
            # global gradient clip applied to the actor and critic update.
            batch_for_gpt['actor_critic_only'] = True
            batch_for_gpt['ppo_advantages'] = torch.as_tensor(
                advantage_batch, device=self._cfg.device, dtype=torch.float32
            )[:, :batch_for_gpt['actions'].shape[1]]
            batch_for_gpt['ppo_old_log_prob'] = torch.as_tensor(
                old_log_prob_batch, device=self._cfg.device, dtype=torch.float32
            )[:, :batch_for_gpt['actions'].shape[1]]
            batch_for_gpt['ppo_action_mask'] = torch.as_tensor(
                behavior_action_mask_batch, device=self._cfg.device, dtype=torch.bool
            )[:, :batch_for_gpt['actions'].shape[1]]
            batch_for_gpt['ppo_policy_features'] = torch.as_tensor(
                behavior_policy_feature_batch, device=self._cfg.device, dtype=torch.float32
            )[:, :batch_for_gpt['actions'].shape[1]]
            batch_for_gpt['ppo_clip_ratio'] = float(ppo_cfg.clip_ratio)
            batch_for_gpt['ppo_entropy_weight'] = float(ppo_cfg.entropy_weight)
        elif learn_type == 'world_model':
            batch_for_gpt['disable_policy_loss'] = True
            batch_for_gpt['disable_value_loss'] = True

        # Extract valid target policy data and compute entropy
        if ppo_batch:
            average_target_policy_entropy = batch_for_gpt['target_value'].new_tensor(0.)
        else:
            valid_target_policy = batch_for_gpt['target_policy'][batch_for_gpt['mask_padding']]
            if valid_target_policy.numel() == 0:
                average_target_policy_entropy = batch_for_gpt['target_policy'].new_tensor(0.)
            else:
                target_policy_entropy = -torch.sum(
                    valid_target_policy * torch.log(valid_target_policy + 1e-9), dim=-1
                )
                average_target_policy_entropy = target_policy_entropy.mean()

        # Update world model
        losses = self._learn_model.world_model.compute_loss(
            batch_for_gpt, self._target_model.world_model.tokenizer, self.value_inverse_scalar_transform_handle, global_step=train_iter, current_policy_label_eps=current_policy_label_eps,
        )

        # ==================== Integrate norm monitoring logic ====================
        norm_log_dict = {}
        should_monitor_norms = should_run_periodic_monitor(
            train_iter, self._cfg.monitor_norm_freq, self._last_norm_monitor_iter
        )
        # Check if monitoring frequency is reached
        if should_monitor_norms:
            with torch.no_grad():
                # 1. Monitor model parameter norms
                param_norm_metrics = self._monitor_model_norms()
                norm_log_dict.update(param_norm_metrics)

                # 2. Monitor intermediate tensor x (Transformer output)
                intermediate_x = losses.intermediate_losses.get('intermediate_tensor_x')
                if intermediate_x is not None:
                    # x shape is (B, T, E)
                    # Calculate L2 norm for each token
                    token_norms = intermediate_x.norm(p=2, dim=-1)

                    # Record statistics of these norms
                    norm_log_dict['norm/x_token/mean'] = token_norms.mean().item()
                    norm_log_dict['norm/x_token/std'] = token_norms.std().item()
                    norm_log_dict['norm/x_token/max'] = token_norms.max().item()
                    norm_log_dict['norm/x_token/min'] = token_norms.min().item()
                    norm_log_dict.update(representation_health_metrics(intermediate_x))

                # 3. Monitor detailed statistics of logits (Value, Policy, Reward)
                logits_value = losses.intermediate_losses.get('logits_value')
                if logits_value is not None:
                    norm_log_dict['logits/value/mean'] = logits_value.mean().item()
                    norm_log_dict['logits/value/std'] = logits_value.std().item()
                    norm_log_dict['logits/value/max'] = logits_value.max().item()
                    norm_log_dict['logits/value/min'] = logits_value.min().item()
                    norm_log_dict['logits/value/abs_max'] = logits_value.abs().max().item()

                logits_policy = losses.intermediate_losses.get('logits_policy')
                if logits_policy is not None:
                    norm_log_dict['logits/policy/mean'] = logits_policy.mean().item()
                    norm_log_dict['logits/policy/std'] = logits_policy.std().item()
                    norm_log_dict['logits/policy/max'] = logits_policy.max().item()
                    norm_log_dict['logits/policy/min'] = logits_policy.min().item()
                    norm_log_dict['logits/policy/abs_max'] = logits_policy.abs().max().item()

                logits_reward = losses.intermediate_losses.get('logits_reward')
                if logits_reward is not None:
                    norm_log_dict['logits/reward/mean'] = logits_reward.mean().item()
                    norm_log_dict['logits/reward/std'] = logits_reward.std().item()
                    norm_log_dict['logits/reward/max'] = logits_reward.max().item()
                    norm_log_dict['logits/reward/min'] = logits_reward.min().item()
                    norm_log_dict['logits/reward/abs_max'] = logits_reward.abs().max().item()

                # 4. Monitor obs_embeddings (Encoder output) statistics
                obs_embeddings = losses.intermediate_losses.get('obs_embeddings')
                if obs_embeddings is not None:
                    # Calculate L2 norm for each embedding
                    emb_norms = obs_embeddings.norm(p=2, dim=-1)
                    norm_log_dict['embeddings/obs/norm_mean'] = emb_norms.mean().item()
                    norm_log_dict['embeddings/obs/norm_std'] = emb_norms.std().item()
                    norm_log_dict['embeddings/obs/norm_max'] = emb_norms.max().item()
                    norm_log_dict['embeddings/obs/norm_min'] = emb_norms.min().item()

                # ==================== Early Warning System ====================
                # Detect potential training instability and issue warnings
                warnings_issued = []

                # Check 1: Policy logits explosion (should be caught by clip, but warn anyway)
                if 'logits/policy/abs_max' in norm_log_dict:
                    policy_abs_max = norm_log_dict['logits/policy/abs_max']
                    if policy_abs_max > 8.0:
                        warnings_issued.append(f"⚠️ CRITICAL: Policy logits explosion detected! abs_max={policy_abs_max:.2f} (threshold: 8.0)")
                    elif policy_abs_max > 5.0:
                        warnings_issued.append(f"⚠️ WARNING: Policy logits getting large! abs_max={policy_abs_max:.2f} (threshold: 5.0)")

                # Check 2: Embedding norm explosion
                if 'embeddings/obs/norm_std' in norm_log_dict:
                    emb_norm_std = norm_log_dict['embeddings/obs/norm_std']
                    if emb_norm_std > 10.0:
                        warnings_issued.append(f"⚠️ CRITICAL: Embedding norm std explosion! std={emb_norm_std:.2f} (threshold: 10.0)")
                    elif emb_norm_std > 5.0:
                        warnings_issued.append(f"⚠️ WARNING: Embedding norm std increasing! std={emb_norm_std:.2f} (threshold: 5.0)")

                # Check 3: representation collapse. Token-norm std is not used:
                # LayerNorm intentionally makes all token L2 norms nearly equal.
                if 'activation/x_token/feature_std_mean' in norm_log_dict:
                    feature_std_mean = norm_log_dict['activation/x_token/feature_std_mean']
                    near_constant_fraction = norm_log_dict['activation/x_token/near_constant_fraction']
                    if feature_std_mean < 1e-3 or near_constant_fraction > 0.99:
                        warnings_issued.append(
                            "⚠️ CRITICAL: X token representation collapse! "
                            f"feature_std_mean={feature_std_mean:.4g}, "
                            f"near_constant_fraction={near_constant_fraction:.2%}"
                        )
                    elif feature_std_mean < 1e-2 or near_constant_fraction > 0.9:
                        warnings_issued.append(
                            "⚠️ WARNING: X token feature diversity is low! "
                            f"feature_std_mean={feature_std_mean:.4g}, "
                            f"near_constant_fraction={near_constant_fraction:.2%}"
                        )

                # Log warnings if any
                if warnings_issued:
                    logging.warning(f"\n{'='*80}\n[TRAINING STABILITY] Iteration {train_iter}:\n" + "\n".join(warnings_issued) + f"\n{'='*80}")
                    norm_log_dict['stability/warning_count'] = float(len(warnings_issued))
                else:
                    norm_log_dict['stability/warning_count'] = 0.0
                # ====================================================================
        # =================================================================

        # Extract the calculated value_priority from the returned losses.
        value_priority_tensor = losses.intermediate_losses['value_priority']
        # Convert to numpy array for the replay buffer, adding a small epsilon.
        value_priority_np = value_priority_tensor.detach().cpu().numpy() + 1e-6
        replay_log_dict = replay_distribution_metrics(weights, value_priority_tensor)
        logits_value = losses.intermediate_losses.get('logits_value')
        if logits_value is None:
            value_calibration_log_dict = {}
        else:
            value_steps = logits_value.shape[1]
            predicted_values = self.value_inverse_scalar_transform_handle(
                logits_value.reshape(-1, logits_value.shape[-1])
            ).reshape(logits_value.shape[0], value_steps)
            value_calibration_log_dict = value_calibration_metrics(
                predicted_values,
                target_value[:, :value_steps],
                batch_for_gpt['mask_padding'][:, :value_steps],
            )

        # ==================== START: PER importance-sampling weighting ====================
        # NOTE: losses.loss_total is a batch-level scalar, so ``(weights * losses.loss_total).mean()``
        # collapses to ``losses.loss_total * weights.mean()`` and the per-sample IS weights from the
        # replay buffer would have no effect. When the world model returns per-sample loss components
        # ([B] tensors, discrete action space only), rebuild the total loss per sample so that the
        # weights are actually applied.
        weighted_total_loss = apply_per_sample_is_weights(
            weights, losses, losses.intermediate_losses.get('per_sample_loss_policy', None), losses.loss_total
        )
        # ==================== END: PER importance-sampling weighting ====================

        for loss_name, loss_value in losses.intermediate_losses.items():
            self.intermediate_losses[f"{loss_name}"] = loss_value

        # Extract losses from intermediate_losses dictionary
        obs_loss = self.intermediate_losses['loss_obs']
        reward_loss = self.intermediate_losses['loss_rewards']
        policy_loss = self.intermediate_losses['loss_policy']
        value_loss = self.intermediate_losses['loss_value']
        latent_recon_loss = self.intermediate_losses['latent_recon_loss']
        perceptual_loss = self.intermediate_losses['perceptual_loss']
        open_loop_consistency_loss = self.intermediate_losses['open_loop_consistency_loss']
        open_loop_recurrent_loss = self.intermediate_losses['open_loop_recurrent_loss']
        open_loop_recurrent_latent_loss = self.intermediate_losses['open_loop_recurrent_latent_loss']
        open_loop_recurrent_reward_loss = self.intermediate_losses['open_loop_recurrent_reward_loss']
        open_loop_recurrent_value_loss = self.intermediate_losses['open_loop_recurrent_value_loss']
        open_loop_recurrent_policy_loss = self.intermediate_losses['open_loop_recurrent_policy_loss']
        open_loop_recurrent_policy_ce = self.intermediate_losses['open_loop_recurrent_policy_ce']
        open_loop_recurrent_policy_entropy = self.intermediate_losses['open_loop_recurrent_policy_entropy']
        orig_policy_loss = self.intermediate_losses['orig_policy_loss']
        policy_entropy = self.intermediate_losses['policy_entropy']
        first_step_losses = self.intermediate_losses['first_step_losses']
        middle_step_losses = self.intermediate_losses['middle_step_losses']
        last_step_losses = self.intermediate_losses['last_step_losses']
        dormant_ratio_encoder = self.intermediate_losses['dormant_ratio_encoder']
        dormant_ratio_transformer = self.intermediate_losses['dormant_ratio_transformer']
        dormant_ratio_head = self.intermediate_losses['dormant_ratio_head']
        avg_weight_mag_encoder = self.intermediate_losses['avg_weight_mag_encoder']
        avg_weight_mag_transformer = self.intermediate_losses['avg_weight_mag_transformer']
        avg_weight_mag_head = self.intermediate_losses['avg_weight_mag_head']
        e_rank_last_linear = self.intermediate_losses['e_rank_last_linear']
        e_rank_sim_norm = self.intermediate_losses['e_rank_sim_norm']
        latent_state_l2_norms = self.intermediate_losses['latent_state_l2_norms']
        latent_action_l2_norms = self.intermediate_losses['latent_action_l2_norms']

        temperature_value=self.intermediate_losses['temperature_value']
        temperature_reward=self.intermediate_losses['temperature_reward']
        temperature_policy=self.intermediate_losses['temperature_policy']
        ppo_approx_kl = self.intermediate_losses.get('ppo_approx_kl', torch.tensor(0.))
        ppo_clip_fraction = self.intermediate_losses.get('ppo_clip_fraction', torch.tensor(0.))
        ppo_ratio_mean = self.intermediate_losses.get('ppo_ratio_mean', torch.tensor(1.))
        ppo_ratio_min = self.intermediate_losses.get('ppo_ratio_min', torch.tensor(1.))
        ppo_ratio_max = self.intermediate_losses.get('ppo_ratio_max', torch.tensor(1.))

        assert not torch.isnan(losses.loss_total).any(), "Loss contains NaN values"
        assert not torch.isinf(losses.loss_total).any(), "Loss contains Inf values"

        # Core learning model update step
        # Reset gradients at the start of each accumulation cycle
        if (train_iter % self.accumulation_steps) == 0:
            self._optimizer_world_model.zero_grad()


        # ==================== START: Target Entropy Regularization Update Logic ====================
        alpha_loss = None
        per_sample_weighted_policy_loss = None
        current_alpha = self._cfg.model.world_model_cfg.policy_entropy_weight  # Default to fixed value
        if self.use_adaptive_entropy_weight:
            # Dynamically calculate target entropy (this logic is correct and preserved)
            progress = min(1.0, train_iter / self.target_entropy_decay_steps)
            current_ratio = self.target_entropy_start_ratio * (1 - progress) + self.target_entropy_end_ratio * progress
            action_space_size = self._cfg.model.action_space_size
            # Note: We define target_entropy as a positive number, which is more intuitive
            current_target_entropy = -np.log(1.0 / action_space_size) * current_ratio

            # Calculate alpha_loss (corrected sign)
            # This is the core correction: removed the negative sign at the front
            # detach() is still critical to ensure alpha_loss gradient only flows to log_alpha
            alpha_loss = (self.log_alpha * (policy_entropy.detach() - current_target_entropy)).mean()

            # Update log_alpha
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            # Keep alpha numerically bounded without preventing late-stage entropy annealing.
            with torch.no_grad():
                self.log_alpha.clamp_(
                    np.log(self.adaptive_entropy_alpha_min),
                    np.log(self.adaptive_entropy_alpha_max)
                )

            # Use current updated alpha (with gradient flow truncated)
            current_alpha = self.log_alpha.exp().detach()

            # Keep the optional recurrent policy objective on the same entropy coefficient as
            # the main policy objective. The world model initially computes it with the fixed
            # coefficient so non-adaptive training remains unchanged.
            open_loop_recurrent_loss = apply_open_loop_recurrent_entropy_weight(
                open_loop_recurrent_loss,
                open_loop_recurrent_policy_loss,
                open_loop_recurrent_policy_ce,
                open_loop_recurrent_policy_entropy,
                current_alpha,
            )
            open_loop_recurrent_policy_loss = (
                open_loop_recurrent_policy_ce
                - current_alpha * open_loop_recurrent_policy_entropy
            )
            # ``apply_per_sample_is_weights`` reads auxiliary losses from this container.
            losses.intermediate_losses['open_loop_recurrent_loss'] = open_loop_recurrent_loss
            losses.intermediate_losses['open_loop_recurrent_policy_loss'] = (
                open_loop_recurrent_policy_loss
            )
            self.intermediate_losses['open_loop_recurrent_loss'] = open_loop_recurrent_loss
            self.intermediate_losses['open_loop_recurrent_policy_loss'] = (
                open_loop_recurrent_policy_loss
            )

            # Recalculate weighted policy loss and total loss
            # Note: policy_entropy here is already an average value of a batch
            weighted_policy_loss = orig_policy_loss - current_alpha * policy_entropy
            # Rebuild total loss with the same weights used by LossWithIntermediateLosses.
            total_loss = (
                losses.reward_loss_weight * reward_loss +
                losses.value_loss_weight * value_loss +
                losses.policy_loss_weight * weighted_policy_loss +
                losses.obs_loss_weight * obs_loss +
                losses.latent_recon_loss_weight * latent_recon_loss +
                losses.perceptual_loss_weight * perceptual_loss +
                losses.open_loop_consistency_loss_weight * open_loop_consistency_loss +
                losses.open_loop_recurrent_loss_weight * open_loop_recurrent_loss
            )
            # Per-sample counterpart of ``weighted_policy_loss`` for correct IS weighting.
            per_sample_orig_policy_loss = losses.intermediate_losses.get('per_sample_loss_orig_policy', None)
            if per_sample_orig_policy_loss is not None:
                per_sample_weighted_policy_loss = per_sample_orig_policy_loss \
                    - current_alpha * losses.intermediate_losses['per_sample_loss_policy_entropy']
            else:
                per_sample_weighted_policy_loss = None
            weighted_total_loss = apply_per_sample_is_weights(weights, losses, per_sample_weighted_policy_loss, total_loss)
        # ===================== END: Target Entropy Regularization Update Logic =====================

        gradient_component_log_dict = {}
        should_diagnose_gradients = should_run_periodic_monitor(
            train_iter,
            int(self._cfg.gradient_diagnostic_freq),
            self._last_gradient_diagnostic_iter,
        )
        if should_diagnose_gradients:
            per_sample_policy = (
                per_sample_weighted_policy_loss
                if per_sample_weighted_policy_loss is not None
                else losses.intermediate_losses.get('per_sample_loss_policy')
            )
            per_sample_components = {
                'obs': losses.intermediate_losses.get('per_sample_loss_obs'),
                'reward': losses.intermediate_losses.get('per_sample_loss_rewards'),
                'value': losses.intermediate_losses.get('per_sample_loss_value'),
                'policy': per_sample_policy,
            }
            component_weights = {
                'obs': losses.obs_loss_weight,
                'reward': losses.reward_loss_weight,
                'value': losses.value_loss_weight,
                'policy': losses.policy_loss_weight,
            }
            loss_components = {
                name: component_weights[name] * (weights.reshape(-1) * component).mean()
                for name, component in per_sample_components.items()
                if component is not None
            }
            gradient_component_log_dict = component_gradient_norms(
                loss_components, self._gradient_diagnostic_module_groups()
            )
            gradient_component_log_dict['grad_component/last_check_iter'] = float(train_iter)
            self._last_gradient_diagnostic_iter = train_iter

        # Scale the loss by the number of accumulation steps
        weighted_total_loss = weighted_total_loss / self.accumulation_steps
        weighted_total_loss.backward()

        # Still executed within torch.no_grad() context
        # =================================================================
        with torch.no_grad():
            # 1. Encoder-Clip
            # ==================== START: Dynamically calculate current Clip threshold ====================
            current_clip_value = self.latent_norm_clip_threshold  # Default to fixed value
            if self.use_encoder_clip_annealing:
                progress = min(1.0, train_iter / self.encoder_clip_anneal_steps)

                if self.encoder_clip_anneal_type == 'cosine':
                    # Cosine schedule: smoothly transition from 1 to 0
                    cosine_progress = 0.5 * (1.0 + np.cos(np.pi * progress))
                    current_clip_value = self.encoder_clip_end + \
                                         (self.encoder_clip_start - self.encoder_clip_end) * cosine_progress
                else:  # Default to linear schedule
                    current_clip_value = self.encoder_clip_start * (1 - progress) + \
                                         self.encoder_clip_end * progress
            # ===================== END: Dynamically calculate current Clip threshold =====================

            # 1. Encoder-Clip (using dynamically calculated current_clip_value)
            # Bug-fix: previously this block was guarded by `self.use_encoder_clip_annealing`,
            # which made `latent_norm_clip_threshold` a dead config when annealing was disabled.
            # Now the clip fires whenever current_clip_value > 0, regardless of annealing mode.
            encoder_clip_applied = False
            encoder_clip_scale_factor = 1.0
            encoder_clip_max_latent_norm = 0.0
            if 'obs_embeddings' in losses.intermediate_losses:
                obs_embeddings = losses.intermediate_losses['obs_embeddings']
                if obs_embeddings is not None:
                    max_latent_norm = obs_embeddings.norm(p=2, dim=-1).max()
                    encoder_clip_max_latent_norm = max_latent_norm.item()
                    if current_clip_value > 0 and max_latent_norm > current_clip_value:
                        scale_factor = current_clip_value / max_latent_norm.item()
                        encoder_clip_applied = True
                        encoder_clip_scale_factor = scale_factor
                        self._encoder_clip_apply_count += 1
                        if train_iter % 1000 == 0:
                            clip_mode = "Annealing" if self.use_encoder_clip_annealing else "Fixed"
                            logging.info(f"[Encoder-Clip {clip_mode}] Iter {train_iter}: Max latent norm {max_latent_norm.item():.2f} > {current_clip_value:.2f}. Scaling by {scale_factor:.4f}.")
                        # The encoder ends with a LayerNorm, so scaling all of its weights only
                        # changes the output through the final norm's affine parameters; scale those
                        # directly and fall back to full-module scaling when the encoder has no
                        # trainable final-norm parameters (e.g. LayerNormNoAffine/SimNorm).
                        if not scale_encoder_final_norm(self._model.world_model.tokenizer.encoder, scale_factor):
                            scale_module_weights_vectorized(self._model.world_model.tokenizer.encoder, scale_factor)

            if self.use_head_clip and self.head_clip_manager is not None:
                head_clip_results = self.head_clip_manager.apply_head_clip(
                    self._learn_model.world_model,
                    losses,
                    train_iter
                )


        # Check if the current iteration completes an accumulation cycle
        if (train_iter + 1) % self.accumulation_steps == 0:
            # ==================== [NEW] Monitor gradient norms ====================
            # Monitor gradient norms before gradient clipping to diagnose gradient explosion/vanishing issues
            if should_monitor_norms:
                grad_norm_metrics = self._monitor_gradient_norms()
                norm_log_dict.update(grad_norm_metrics)
            # =================================================================

            # Analyze gradient norms if simulation normalization analysis is enabled
            if self._cfg.analysis_sim_norm:
                # Clear previous analysis results to prevent memory overflow
                del self.l2_norm_before, self.l2_norm_after, self.grad_norm_before, self.grad_norm_after
                self.l2_norm_before, self.l2_norm_after, self.grad_norm_before, self.grad_norm_after = self._learn_model.encoder_hook.analyze()
                self._target_model.encoder_hook.clear_data()

            # Clip gradients to prevent exploding gradients
            total_grad_norm_before_clip_wm = torch.nn.utils.clip_grad_norm_(
                self._learn_model.world_model.parameters(), self._cfg.grad_clip_value
            )

            # Synchronize gradients across multiple GPUs if enabled
            if self._cfg.multi_gpu:
                self.sync_gradients(self._learn_model)

            # Update model parameters
            self._optimizer_world_model.step()

            # Clear CUDA cache if using gradient accumulation
            if self.accumulation_steps > 1:
                torch.cuda.empty_cache()
        else:
            total_grad_norm_before_clip_wm = torch.tensor(0.)

        grad_clip_log_dict = gradient_clip_metrics(
            total_grad_norm_before_clip_wm.item(), self._cfg.grad_clip_value
        )
        if should_monitor_norms and total_grad_norm_before_clip_wm.item() > 0:
            total_norm = total_grad_norm_before_clip_wm.item()
            for group_name in self._gradient_diagnostic_module_groups():
                group_norm = norm_log_dict.get(f'grad/{group_name}/_total_norm')
                if group_norm is not None:
                    norm_log_dict[f'grad/{group_name}/global_norm_fraction'] = float(
                        group_norm / total_norm
                    )

        # Update learning rate scheduler if applicable
        if self._cfg.cos_lr_scheduler or self._cfg.piecewise_decay_lr_scheduler:
            self.lr_scheduler.step()

        # Update the target model with the current model's parameters
        self._target_model.update(self._learn_model.state_dict())

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            current_memory_allocated = torch.cuda.memory_allocated()
            max_memory_allocated = torch.cuda.max_memory_allocated()
            current_memory_allocated_gb = current_memory_allocated / (1024 ** 3)
            max_memory_allocated_gb = max_memory_allocated / (1024 ** 3)
        else:
            current_memory_allocated_gb = 0.
            max_memory_allocated_gb = 0.

        return_log_dict = {
            'analysis/first_step_loss_value': first_step_losses['loss_value'].item(),
            'analysis/first_step_loss_policy': first_step_losses['loss_policy'].item(),
            'analysis/first_step_loss_rewards': first_step_losses['loss_rewards'].item(),
            'analysis/first_step_loss_obs': first_step_losses['loss_obs'].item(),

            'analysis/middle_step_loss_value': middle_step_losses['loss_value'].item(),
            'analysis/middle_step_loss_policy': middle_step_losses['loss_policy'].item(),
            'analysis/middle_step_loss_rewards': middle_step_losses['loss_rewards'].item(),
            'analysis/middle_step_loss_obs': middle_step_losses['loss_obs'].item(),

            'analysis/last_step_loss_value': last_step_losses['loss_value'].item(),
            'analysis/last_step_loss_policy': last_step_losses['loss_policy'].item(),
            'analysis/last_step_loss_rewards': last_step_losses['loss_rewards'].item(),
            'analysis/last_step_loss_obs': last_step_losses['loss_obs'].item(),

            'Current_GPU': current_memory_allocated_gb,
            'Max_GPU': max_memory_allocated_gb,
            'collect_mcts_temperature': self._collect_mcts_temperature,
            'collect_epsilon': self._collect_epsilon,
            'cur_lr_world_model': self._optimizer_world_model.param_groups[0]['lr'],
            'weighted_total_loss': weighted_total_loss.item(),
            'obs_loss': obs_loss.item(),
            'latent_recon_loss': latent_recon_loss.item(),
            'perceptual_loss': perceptual_loss.item(),
            'open_loop_consistency_loss': open_loop_consistency_loss.item(),
            'open_loop_recurrent_loss': open_loop_recurrent_loss.item(),
            'open_loop_recurrent_latent_loss': open_loop_recurrent_latent_loss.item(),
            'open_loop_recurrent_reward_loss': open_loop_recurrent_reward_loss.item(),
            'open_loop_recurrent_value_loss': open_loop_recurrent_value_loss.item(),
            'open_loop_recurrent_policy_loss': open_loop_recurrent_policy_loss.item(),
            'policy_loss': policy_loss.item(),
            'orig_policy_loss': orig_policy_loss.item(),
            'policy_entropy': policy_entropy.item(),
            'target_policy_entropy': average_target_policy_entropy.item(),
            'reward_loss': reward_loss.item(),
            'value_loss': value_loss.item(),
            # Add value_priority to the log dictionary.
            'value_priority': value_priority_np.mean().item(),
            'value_priority_orig': value_priority_np,
            'target_reward': target_reward.mean().item(),
            'target_value': target_value.mean().item(),
            'transformed_target_reward': transformed_target_reward.mean().item(),
            'transformed_target_value': transformed_target_value.mean().item(),
            'total_grad_norm_before_clip_wm': total_grad_norm_before_clip_wm.item(),
            'analysis/dormant_ratio_encoder': dormant_ratio_encoder,
            'analysis/dormant_ratio_transformer': dormant_ratio_transformer,
            'analysis/dormant_ratio_head': dormant_ratio_head,

            'analysis/avg_weight_mag_encoder': avg_weight_mag_encoder,
            'analysis/avg_weight_mag_transformer': avg_weight_mag_transformer,
            'analysis/avg_weight_mag_head': avg_weight_mag_head,
            'analysis/e_rank_last_linear': e_rank_last_linear,
            'analysis/e_rank_sim_norm':  e_rank_sim_norm,

            'analysis/latent_state_l2_norms': latent_state_l2_norms.item(),
            'analysis/latent_action_l2_norms': latent_action_l2_norms,
            'analysis/l2_norm_before': self.l2_norm_before,
            'analysis/l2_norm_after': self.l2_norm_after,
            'analysis/grad_norm_before': self.grad_norm_before,
            'analysis/grad_norm_after': self.grad_norm_after,

            "temperature_value":temperature_value,
            "temperature_reward":temperature_reward,
            "temperature_policy":temperature_policy,

            "current_policy_label_eps":current_policy_label_eps,
            'ppo/approx_kl': float(ppo_approx_kl.item()),
            'ppo/clip_fraction': float(ppo_clip_fraction.item()),
            'ppo/ratio_mean': float(ppo_ratio_mean.item()),
            'ppo/ratio_min': float(ppo_ratio_min.item()),
            'ppo/ratio_max': float(ppo_ratio_max.item()),
        }
        for metric_name in (
            'open_loop_latent_mse_mean',
            'open_loop_latent_mse_first',
            'open_loop_latent_mse_middle',
            'open_loop_latent_mse_last',
            'rolling_teacher_latent_mse_mean',
            'rolling_teacher_latent_mse_first',
            'rolling_teacher_latent_mse_middle',
            'rolling_teacher_latent_mse_last',
            'teacher_forced_latent_mse_mean',
            'teacher_forced_latent_mse_first',
            'rolling_context_ratio',
            'open_loop_exposure_ratio',
            'open_loop_total_ratio',
        ):
            metric_value = losses.intermediate_losses.get(metric_name)
            if metric_value is not None:
                return_log_dict[f'analysis/{metric_name}'] = (
                    metric_value.item() if isinstance(metric_value, torch.Tensor)
                    else float(metric_value)
                )
        return_log_dict.update({
            'loss/weighted_total': weighted_total_loss.item(),
            'loss/obs': obs_loss.item(),
            'loss/reward': reward_loss.item(),
            'loss/value': value_loss.item(),
            'loss/policy': policy_loss.item(),
            'loss/policy_orig': orig_policy_loss.item(),
            'loss/policy_entropy': policy_entropy.item(),
            'loss/latent_recon': latent_recon_loss.item(),
            'loss/perceptual': perceptual_loss.item(),
            'target/reward': target_reward.mean().item(),
            'target/value': target_value.mean().item(),
            'target/policy_entropy': average_target_policy_entropy.item(),
            'target/transformed_reward': transformed_target_reward.mean().item(),
            'target/transformed_value': transformed_target_value.mean().item(),
            'priority/value': value_priority_np.mean().item(),
            'lr/world_model': self._optimizer_world_model.param_groups[0]['lr'],
            'grad/world_model_total_norm': total_grad_norm_before_clip_wm.item(),
            'memory/current_gpu_gb': current_memory_allocated_gb,
            'memory/max_gpu_gb': max_memory_allocated_gb,
            'collect/epsilon': self._collect_epsilon,
            'collect/mcts_temperature': self._collect_mcts_temperature,
            'schedule/policy_label_eps': current_policy_label_eps,
        })
        return_log_dict.update(replay_log_dict)
        return_log_dict.update(self._latest_replay_diagnostic_metrics)
        return_log_dict.update(value_calibration_log_dict)
        return_log_dict.update(grad_clip_log_dict)
        return_log_dict.update(gradient_component_log_dict)

        if should_monitor_norms:
            norm_log_dict['stability/last_check_iter'] = float(train_iter)
            self._latest_norm_log_dict = norm_log_dict.copy()
            self._last_norm_monitor_iter = train_iter
        if self._latest_norm_log_dict:
            return_log_dict.update(self._latest_norm_log_dict)

        use_enhanced_policy_monitoring = self._cfg.use_enhanced_policy_monitoring
        if use_enhanced_policy_monitoring:
            # Monitor policy logits statistics
            with torch.no_grad():
                logits_policy = losses.intermediate_losses.get('logits_policy')
                if logits_policy is not None:
                    return_log_dict['policy_logits/norm'] = logits_policy.norm(dim=-1).mean().item()
                    return_log_dict['policy_logits/max'] = logits_policy.max().item()
                    return_log_dict['policy_logits/min'] = logits_policy.min().item()
                    return_log_dict['policy_logits/std'] = logits_policy.std().item()

                # [NEW] Also monitor Value and Reward logits
                logits_value = losses.intermediate_losses.get('logits_value')
                if logits_value is not None:
                    return_log_dict['value_logits/abs_max'] = logits_value.abs().max().item()
                    return_log_dict['value_logits/norm'] = logits_value.norm(dim=-1).mean().item()

                logits_reward = losses.intermediate_losses.get('logits_reward')
                if logits_reward is not None:
                    return_log_dict['reward_logits/abs_max'] = logits_reward.abs().max().item()
                    return_log_dict['reward_logits/norm'] = logits_reward.norm(dim=-1).mean().item()

                # Monitor target_policy entropy statistics (minimum entropy indicates extreme distributions)
                valid_target_policy = batch_for_gpt['target_policy'][batch_for_gpt['mask_padding']]
                if valid_target_policy.numel() == 0:
                    return_log_dict['target_policy_entropy/mean'] = 0.0
                    return_log_dict['target_policy_entropy/min'] = 0.0
                    return_log_dict['target_policy_entropy/max'] = 0.0
                    return_log_dict['target_policy_entropy/std'] = 0.0
                else:
                    target_policy_entropies = -torch.sum(
                        valid_target_policy * torch.log(valid_target_policy + 1e-9), dim=-1
                    )
                    return_log_dict['target_policy_entropy/mean'] = target_policy_entropies.mean().item()
                    return_log_dict['target_policy_entropy/min'] = target_policy_entropies.min().item()
                    return_log_dict['target_policy_entropy/max'] = target_policy_entropies.max().item()
                    return_log_dict['target_policy_entropy/std'] = target_policy_entropies.std(unbiased=False).item()
        # ================================================================================

        if self.use_adaptive_entropy_weight:
            return_log_dict['adaptive_alpha'] = current_alpha.item()
            return_log_dict['adaptive_target_entropy_ratio'] = current_ratio
            return_log_dict['alpha_loss'] = alpha_loss.item()
            return_log_dict['entropy/adaptive_alpha'] = current_alpha.item()
            return_log_dict['entropy/target_ratio'] = current_ratio
            return_log_dict['entropy/alpha_loss'] = alpha_loss.item()

        if self.use_encoder_clip_annealing:
            return_log_dict['current_encoder_clip_value'] = current_clip_value
            return_log_dict['stability/current_encoder_clip_value'] = current_clip_value
        return_log_dict.update(encoder_clip_metrics(
            threshold=current_clip_value,
            applied=encoder_clip_applied,
            apply_count=self._encoder_clip_apply_count,
            scale_factor=encoder_clip_scale_factor,
            max_latent_norm=encoder_clip_max_latent_norm,
        ))

        if self.use_head_clip and self.head_clip_manager is not None:
            # Add head clip results to log (if any)
            if head_clip_results:
                for head_name, info in head_clip_results.items():
                    return_log_dict[f'head_clip/{head_name}/max_logits'] = info['max_logits']
                    return_log_dict[f'head_clip/{head_name}/threshold'] = info['threshold']
                    if info['scaled']:
                        return_log_dict[f'head_clip/{head_name}/scale_factor'] = info['scale_factor']

        if self._cfg.use_wandb:
            wandb.log({'learner_step/' + k: v for k, v in return_log_dict.items()}, step=self.env_step)
            wandb.log({"learner_iter_vs_env_step": self.train_iter}, step=self.env_step)

        return return_log_dict
    def _init_collect(self) -> None:
        """
        Overview:
            Collect mode init method. Called by ``self.__init__``. Initialize the collect model and MCTS utils.
        """
        self._collect_model = self._model
        self.policy_improvement = getattr(self._cfg, 'policy_improvement', 'mcts')
        if self.policy_improvement not in {'mcts', 'ppo'}:
            raise ValueError(
                f"policy_improvement must be 'mcts' or 'ppo', got {self.policy_improvement!r}"
            )
        # Create a configuration copy for collect MCTS and set specific simulation count
        mcts_collect_cfg = copy.deepcopy(self._cfg)
        mcts_collect_cfg.num_simulations = self._cfg.collect_num_simulations
        if self.policy_improvement == 'ppo':
            self._mcts_collect = None
        elif self._cfg.mcts_ctree:
            self._mcts_collect = MCTSCtree(mcts_collect_cfg)
        else:
            # NOTE: a python-tree MCTS variant for UniZero is not implemented in this fork.
            raise NotImplementedError('UniZero policy only supports mcts_ctree=True (C++ tree MCTS).')
        self._collect_mcts_temperature = 1.
        self._collect_epsilon = 0.0
        self.collector_env_num = self._cfg.collector_env_num
        if self._cfg.model.model_type == 'conv':
            self.last_batch_obs_collect = torch.zeros([self.collector_env_num, self._cfg.model.observation_shape[0], 64, 64]).to(self._cfg.device)
            self.last_batch_action_collect = [-1 for i in range(self.collector_env_num)]
        elif self._cfg.model.model_type == 'mlp':
            self.last_batch_obs_collect = torch.zeros(
                [self.collector_env_num, self._cfg.model.observation_shape],
                dtype=torch.float32,
                device=self._cfg.device,
            )
            self.last_batch_action_collect = [-1 for i in range(self.collector_env_num)]
    def _forward_collect(
            self,
            data: torch.Tensor,
            action_mask: List = None,
            temperature: float = 1,
            to_play: List = [-1],
            epsilon: float = 0.25,
            ready_env_id: np.array = None,
            timestep: List = [0],
            task_id: int = None,
    ) -> Dict:
        """
        Overview:
            The forward function for collecting data in collect mode. Use model to execute MCTS search.
            Choosing the action through sampling during the collect mode.
        Arguments:
            - data (:obj:`torch.Tensor`): The input data, i.e. the observation.
            - action_mask (:obj:`list`): The action mask, i.e. the action that cannot be selected.
            - temperature (:obj:`float`): The temperature of the policy.
            - to_play (:obj:`int`): The player to play.
            - ready_env_id (:obj:`list`): The id of the env that is ready to collect.
            - timestep (:obj:`list`): The step index of the env in one episode.
            - task_id (:obj:`int`): The task id. Default is None, which means UniZero is in the single-task mode.
        Shape:
            - data (:obj:`torch.Tensor`):
                - For Atari, :math:`(N, C*S, H, W)`, where N is the number of collect_env, C is the number of channels, \
                    S is the number of stacked frames, H is the height of the image, W is the width of the image.
                - For lunarlander, :math:`(N, O)`, where N is the number of collect_env, O is the observation space size.
            - action_mask: :math:`(N, action_space_size)`, where N is the number of collect_env.
            - temperature: :math:`(1, )`.
            - to_play: :math:`(N, 1)`, where N is the number of collect_env.
            - ready_env_id: None
            - timestep: :math:`(N, 1)`, where N is the number of collect_env.
        Returns:
            - output (:obj:`Dict[int, Any]`): Dict type data, the keys including ``action``, ``distributions``, \
                ``visit_count_distribution_entropy``, ``value``, ``pred_value``, ``policy_logits``.
        """
        self._collect_model.eval()

        self._collect_mcts_temperature = temperature
        self._collect_epsilon = epsilon
        active_collect_env_num = data.shape[0]
        ready_env_id = self._normalize_ready_env_id(ready_env_id, active_collect_env_num)
        output = {i: None for i in ready_env_id}

        with torch.no_grad():
            last_obs_batch, last_action_batch = self._select_last_infer_inputs(
                self.last_batch_obs_collect, self.last_batch_action_collect, ready_env_id, self.collector_env_num
            )
            network_output = self._collect_model.initial_inference(
                last_obs_batch, last_action_batch, data, timestep, ready_env_id=ready_env_id
            )
            latent_state_roots, reward_roots, pred_values, policy_logits = mz_network_output_unpack(network_output)

            pred_values = self.value_inverse_scalar_transform_handle(pred_values).detach().cpu().numpy()
            if self.policy_improvement == 'ppo' or self._cfg.collect_with_pure_policy:
                action_mask_tensor = torch.as_tensor(
                    np.asarray(action_mask), device=policy_logits.device, dtype=torch.bool
                )
                behavior_dist = masked_categorical(policy_logits, action_mask_tensor)
                sampled_actions = behavior_dist.sample()
                behavior_log_probs = behavior_dist.log_prob(sampled_actions)
                behavior_entropies = behavior_dist.entropy()
                batch_action = sampled_actions.detach().cpu().numpy().tolist()
                policy_logits_list = policy_logits.detach().cpu().numpy().tolist()
                policy_features = network_output.policy_features.detach().cpu().numpy()

                for batch_index, env_id in enumerate(ready_env_id):
                    output[env_id] = {
                        'action': int(batch_action[batch_index]),
                        'searched_value': pred_values[batch_index],
                        'predicted_value': pred_values[batch_index],
                        'predicted_policy_logits': policy_logits_list[batch_index],
                        'behavior_log_prob': float(behavior_log_probs[batch_index].item()),
                        'behavior_policy_features': policy_features[batch_index],
                        'policy_entropy': float(behavior_entropies[batch_index].item()),
                        'timestep': timestep[batch_index],
                        'predicted_next_text': None,
                    }

                self._update_last_infer_inputs(
                    'last_batch_obs_collect', 'last_batch_action_collect',
                    data, batch_action, ready_env_id, self.collector_env_num
                )
                return output

            latent_state_roots = latent_state_roots.detach().cpu().numpy()
            policy_logits = policy_logits.detach().cpu().numpy().tolist()

            legal_actions = [np.nonzero(action_mask[j])[0].tolist() for j in range(active_collect_env_num)]
            # the only difference between collect and eval is the dirichlet noise
            noises = [
                np.random.dirichlet([self._cfg.root_dirichlet_alpha] * int(sum(action_mask[j]))
                                    ).astype(np.float32).tolist() for j in range(active_collect_env_num)
            ]
            if self._cfg.mcts_ctree:
                # cpp mcts_tree
                roots = MCTSCtree.roots(active_collect_env_num, legal_actions)
            else:
                # python mcts_tree
                roots = MCTSPtree.roots(active_collect_env_num, legal_actions)

            roots.prepare(self._cfg.root_noise_weight, noises, reward_roots, policy_logits, to_play)

            next_latent_state_with_env = self._mcts_collect.search(roots, self._collect_model, latent_state_roots, to_play, timestep)

            # list of list, shape: ``{list: batch_size} -> {list: action_space_size}``
            roots_visit_count_distributions = roots.get_distributions()
            roots_values = roots.get_values()  # shape: {list: batch_size}


            batch_action = []
            for i, env_id in enumerate(ready_env_id):
                distributions, value = roots_visit_count_distributions[i], roots_values[i]

                if self._cfg.eps.eps_greedy_exploration_in_collect:
                    # eps greedy collect
                    action_index_in_legal_action_set, visit_count_distribution_entropy = select_action(
                        distributions, temperature=self._collect_mcts_temperature, deterministic=True
                    )
                    action = np.where(action_mask[i] == 1.0)[0][action_index_in_legal_action_set]

                    if np.random.rand() < self._collect_epsilon:
                        action = np.random.choice(legal_actions[i])
                else:
                    # normal collect
                    # NOTE: Only legal actions possess visit counts, so the ``action_index_in_legal_action_set`` represents
                    # the index within the legal action set, rather than the index in the entire action set.
                    action_index_in_legal_action_set, visit_count_distribution_entropy = select_action(
                        distributions, temperature=self._collect_mcts_temperature, deterministic=False
                    )
                    # NOTE: Convert the ``action_index_in_legal_action_set`` to the corresponding ``action`` in the entire action set.
                    action = np.where(action_mask[i] == 1.0)[0][action_index_in_legal_action_set]

                exploration_metrics = search_exploration_metrics(
                    np.asarray(policy_logits[i])[legal_actions[i]],
                    np.asarray(distributions),
                    self._collect_mcts_temperature,
                )

                next_latent_state = next_latent_state_with_env[i][action]

                if self._cfg.model.world_model_cfg.obs_type == 'text' and self._cfg.model.world_model_cfg.decode_loss_mode is not None and self._cfg.model.world_model_cfg.decode_loss_mode.lower() != 'none':
                    # Output the plain text content decoded by the decoder from the next latent state
                    predicted_next = self._collect_model.tokenizer.decode_to_plain_text(embeddings=next_latent_state, max_length=256)
                else:
                    predicted_next = None

                output[env_id] = {
                    'action': action,
                    'visit_count_distributions': distributions,
                    'visit_count_distribution_entropy': visit_count_distribution_entropy,
                    'searched_value': value,
                    'predicted_value': pred_values[i],
                    'predicted_policy_logits': policy_logits[i],
                    'timestep': timestep[i],
                    'predicted_next_text': predicted_next,
                    **exploration_metrics,
                }
                batch_action.append(action)

            self._update_last_infer_inputs(
                'last_batch_obs_collect', 'last_batch_action_collect',
                data, batch_action, ready_env_id, self.collector_env_num
            )

            # This logic is a temporary workaround specific to the muzero_segment_collector.
            if active_collect_env_num < self.collector_env_num:
                logging.info(
                    f'Partial collect batch: active envs {active_collect_env_num} < total envs '
                    f'{self.collector_env_num}; preserving per-env KV cache slots.'
                )

                # If the sampling type is 'episode', it's unexpected for the number of active environments to drop,
                # as this suggests an inconsistent state or a potential issue in the collection logic.
                if getattr(self._cfg, 'sample_type', '') == 'episode':
                    logging.warning('Inconsistent state detected. `sample_type` is "episode", but the number of active environments has changed.')

        return output
    def _init_eval(self) -> None:
        """
        Overview:
            Evaluate mode init method. Called by ``self.__init__``. Initialize the eval model and MCTS utils.
        """
        self._eval_model = self._model
        self.policy_improvement = getattr(self._cfg, 'policy_improvement', 'mcts')
        if self.policy_improvement not in {'mcts', 'ppo'}:
            raise ValueError(
                f"policy_improvement must be 'mcts' or 'ppo', got {self.policy_improvement!r}"
            )

        # Create a configuration copy for eval MCTS and set specific simulation count
        mcts_eval_cfg = copy.deepcopy(self._cfg)
        mcts_eval_cfg.num_simulations = self._cfg.eval_num_simulations
        mcts_eval_cfg.deterministic = True

        if self.policy_improvement == 'ppo':
            self._mcts_eval = None
        elif self._cfg.mcts_ctree:
            self._mcts_eval = MCTSCtree(mcts_eval_cfg)
        else:
            # NOTE: a python-tree MCTS variant for UniZero is not implemented in this fork.
            raise NotImplementedError('UniZero policy only supports mcts_ctree=True (C++ tree MCTS).')

        self.evaluator_env_num = self._cfg.evaluator_env_num

        if self._cfg.model.model_type == 'conv':
            self.last_batch_obs_eval = torch.zeros([self.evaluator_env_num, self._cfg.model.observation_shape[0], 64, 64]).to(self._cfg.device)
            self.last_batch_action_eval = [-1 for i in range(self.evaluator_env_num)]
        elif self._cfg.model.model_type == 'mlp':
            self.last_batch_obs_eval = torch.zeros(
                [self.evaluator_env_num, self._cfg.model.observation_shape],
                dtype=torch.float32,
                device=self._cfg.device,
            )
            self.last_batch_action_eval = [-1 for i in range(self.evaluator_env_num)]
    def _forward_eval(self, data: torch.Tensor, action_mask: list, to_play: int = -1,
                      ready_env_id: np.array = None, timestep: List = [0], task_id: int = None,) -> Dict:
        """
        Overview:
            The forward function for evaluating the current policy in eval mode. Use model to execute MCTS search.
            Choosing the action with the highest value (argmax) rather than sampling during the eval mode.
        Arguments:
            - data (:obj:`torch.Tensor`): The input data, i.e. the observation.
            - action_mask (:obj:`list`): The action mask, i.e. the action that cannot be selected.
            - to_play (:obj:`int`): The player to play.
            - ready_env_id (:obj:`list`): The id of the env that is ready to eval.
            - timestep (:obj:`list`): The step index of the env in one episode.
            - task_id (:obj:`int`): The task id. Default is None, which means UniZero is in the single-task mode.
        Shape:
            - data (:obj:`torch.Tensor`):
                - For Atari, :math:`(N, C*S, H, W)`, where N is the number of eval_env, C is the number of channels, \
                    S is the number of stacked frames, H is the height of the image, W is the width of the image.
                - For lunarlander, :math:`(N, O)`, where N is the number of eval_env, O is the observation space size.
            - action_mask: :math:`(N, action_space_size)`, where N is the number of eval_env.
            - to_play: :math:`(N, 1)`, where N is the number of eval_env.
            - ready_env_id: None
            - timestep: :math:`(N, 1)`, where N is the number of eval_env.

        Returns:
            - output (:obj:`Dict[int, Any]`): Dict type data, the keys including ``action``, ``distributions``, \
                ``visit_count_distribution_entropy``, ``value``, ``pred_value``, ``policy_logits``.
        """
        self._eval_model.eval()
        active_eval_env_num = data.shape[0]
        ready_env_id = self._normalize_ready_env_id(ready_env_id, active_eval_env_num)
        output = {i: None for i in ready_env_id}
        with torch.no_grad():
            last_obs_batch, last_action_batch = self._select_last_infer_inputs(
                self.last_batch_obs_eval, self.last_batch_action_eval, ready_env_id, self.evaluator_env_num
            )
            network_output = self._eval_model.initial_inference(
                last_obs_batch, last_action_batch, data, timestep, ready_env_id=ready_env_id
            )
            latent_state_roots, reward_roots, pred_values, policy_logits = mz_network_output_unpack(network_output)

            # if not in training, obtain the scalars of the value/reward
            pred_values = self.value_inverse_scalar_transform_handle(pred_values).detach().cpu().numpy()  # shape（B, 1）
            if self.policy_improvement == 'ppo' or self._cfg.collect_with_pure_policy:
                action_mask_tensor = torch.as_tensor(
                    np.asarray(action_mask), device=policy_logits.device, dtype=torch.bool
                )
                eval_dist = masked_categorical(policy_logits, action_mask_tensor)
                batch_action = eval_dist.probs.argmax(dim=-1).detach().cpu().numpy().tolist()
                policy_logits_list = policy_logits.detach().cpu().numpy().tolist()
                entropies = eval_dist.entropy()
                for batch_index, env_id in enumerate(ready_env_id):
                    output[env_id] = {
                        'action': int(batch_action[batch_index]),
                        'visit_count_distributions': eval_dist.probs[batch_index].detach().cpu().numpy().tolist(),
                        'visit_count_distribution_entropy': float(entropies[batch_index].item()),
                        'searched_value': pred_values[batch_index],
                        'predicted_value': pred_values[batch_index],
                        'predicted_policy_logits': policy_logits_list[batch_index],
                        'timestep': timestep[batch_index],
                        'predicted_next_text': None,
                    }
                self._update_last_infer_inputs(
                    'last_batch_obs_eval', 'last_batch_action_eval',
                    data, batch_action, ready_env_id, self.evaluator_env_num
                )
                return output

            latent_state_roots = latent_state_roots.detach().cpu().numpy()
            policy_logits = policy_logits.detach().cpu().numpy().tolist()  # list shape（B, A）

            legal_actions = [np.nonzero(action_mask[j])[0].tolist() for j in range(active_eval_env_num)]
            if self._cfg.mcts_ctree:
                # cpp mcts_tree
                roots = MCTSCtree.roots(active_eval_env_num, legal_actions)
            else:
                # python mcts_tree
                roots = MCTSPtree.roots(active_eval_env_num, legal_actions)
            roots.prepare_no_noise(reward_roots, policy_logits, to_play)
            next_latent_state_with_env = self._mcts_eval.search(roots, self._eval_model, latent_state_roots, to_play, timestep)

            # list of list, shape: ``{list: batch_size} -> {list: action_space_size}``
            roots_visit_count_distributions = roots.get_distributions()
            roots_values = roots.get_values()  # shape: {list: batch_size}

            batch_action = []

            for i, env_id in enumerate(ready_env_id):
                distributions, value = roots_visit_count_distributions[i], roots_values[i]

                # NOTE: Only legal actions possess visit counts, so the ``action_index_in_legal_action_set`` represents
                # the index within the legal action set, rather than the index in the entire action set.
                #  Setting deterministic=True implies choosing the action with the highest value (argmax) rather than
                # sampling during the evaluation phase.
                action_index_in_legal_action_set, visit_count_distribution_entropy = select_action(
                    distributions, temperature=1, deterministic=True
                )
                # NOTE: Convert the ``action_index_in_legal_action_set`` to the corresponding ``action`` in the
                # entire action set.
                action = np.where(action_mask[i] == 1.0)[0][action_index_in_legal_action_set]

                # Predict the next latent state based on the selected action and policy
                next_latent_state = next_latent_state_with_env[i][action]

                if self._cfg.model.world_model_cfg.obs_type == 'text' and self._cfg.model.world_model_cfg.decode_loss_mode is not None and self._cfg.model.world_model_cfg.decode_loss_mode.lower() != 'none':
                    # Output the plain text content decoded by the decoder from the next latent state
                    predicted_next = self._eval_model.tokenizer.decode_to_plain_text(embeddings=next_latent_state, max_length=256)
                else:
                    predicted_next = None

                output[env_id] = {
                    'action': action,
                    'visit_count_distributions': distributions,
                    'visit_count_distribution_entropy': visit_count_distribution_entropy,
                    'searched_value': value,
                    'predicted_value': pred_values[i],
                    'predicted_policy_logits': policy_logits[i],
                    'timestep': timestep[i],
                    'predicted_next_text': predicted_next,
                }
                batch_action.append(action)

            self._update_last_infer_inputs(
                'last_batch_obs_eval', 'last_batch_action_eval',
                data, batch_action, ready_env_id, self.evaluator_env_num
            )

        return output

"""Opt-in open-loop diagnostics and auxiliary objectives for UniZero."""

from typing import Dict, NamedTuple, Optional, Tuple

import torch
import torch.nn.functional as F


class _OpenLoopBatch(NamedTuple):
    """Prepared tensors and boundaries shared by open-loop objectives."""

    batch_size: int
    observations: torch.Tensor
    targets: torch.Tensor
    raw_context: torch.Tensor
    rollout_start: int
    rollout_end: int

    @property
    def transition_count(self) -> int:
        return self.rollout_end - self.rollout_start


class OpenLoopWorldModelMixin:
    """Optional open-loop diagnostics and auxiliary learner objectives.

    The mixin deliberately owns no state.  It only consumes the world model's
    tokenizer embeddings, transformer and prediction heads, which keeps the
    core inference/cache implementation independent from opt-in experiments.
    """

    _EMPTY_OPEN_LOOP_DIAGNOSTICS = {
        'open_loop_latent_mse_mean': 0.0,
        'open_loop_latent_mse_first': 0.0,
        'open_loop_latent_mse_middle': 0.0,
        'open_loop_latent_mse_last': 0.0,
        'rolling_teacher_latent_mse_mean': 0.0,
        'rolling_teacher_latent_mse_first': 0.0,
        'rolling_teacher_latent_mse_middle': 0.0,
        'rolling_teacher_latent_mse_last': 0.0,
        'teacher_forced_latent_mse_mean': 0.0,
        'teacher_forced_latent_mse_first': 0.0,
        'rolling_context_ratio': 0.0,
        'open_loop_exposure_ratio': 0.0,
        'open_loop_total_ratio': 0.0,
    }

    def _validate_open_loop_support(self, operation: str) -> None:
        """Validate the model assumptions shared by every open-loop path."""
        if not self.rebuild_kv_window_from_tokens:
            raise RuntimeError(
                f'{operation} currently requires rebuild_kv_window_from_tokens=True.'
            )
        if self.context_length < 4:
            raise RuntimeError(f'{operation} requires context_length >= 4.')
        if self.config.rotary_emb:
            raise RuntimeError(f'{operation} does not currently support rotary embeddings.')
        if self.continuous_action_space or self.num_observations_tokens != 1:
            raise NotImplementedError(
                f'{operation} currently supports discrete, one-observation-token models.'
            )

    def _prepare_open_loop_batch(
            self,
            obs_embeddings: torch.Tensor,
            target_obs_embeddings: torch.Tensor,
            actions: torch.Tensor,
            mask_padding: torch.Tensor,
            batch_size_limit: int,
            horizon: Optional[int] = None,
    ) -> _OpenLoopBatch:
        """Reshape a learner batch and choose a valid open-loop interval."""
        batch_size = min(int(actions.size(0)), batch_size_limit)
        observations = obs_embeddings.contiguous().view(
            actions.size(0), -1, self.num_observations_tokens, self.embed_dim
        )[:batch_size]
        targets = target_obs_embeddings.contiguous().view(
            actions.size(0), -1, self.num_observations_tokens, self.embed_dim
        )[:batch_size]
        raw_context, rollout_start = self._build_open_loop_prefix_context(
            observations, actions, batch_size
        )
        transition_count = min(
            int(actions.size(1) - rollout_start),
            int(observations.size(1) - 1 - rollout_start),
            int(targets.size(1) - 1 - rollout_start),
            int(mask_padding.size(1) - 1 - rollout_start),
        )
        if horizon is not None:
            transition_count = min(int(horizon), transition_count)
        return _OpenLoopBatch(
            batch_size=batch_size,
            observations=observations,
            targets=targets,
            raw_context=raw_context,
            rollout_start=rollout_start,
            rollout_end=rollout_start + max(transition_count, 0),
        )

    def _build_open_loop_prefix_context(
            self, obs_sequence: torch.Tensor, actions: torch.Tensor, batch_size: int
    ) -> Tuple[torch.Tensor, int]:
        """Build an observation-aligned teacher prefix for an open-loop rollout.

        A prefix of ``p`` transitions constructs
        ``[o0, a0, ..., a{p-1}, op]``.  It then retains the same raw-token
        window used by online inference; supervised targets begin at action
        ``p``.
        """
        prefix_transitions = min(
            self.open_loop_prefix_transitions,
            int(actions.size(1)),
            max(0, int(obs_sequence.size(1) - 1)),
        )
        context_parts = [obs_sequence[:batch_size, 0]]
        for prefix_step in range(prefix_transitions):
            action_tokens = actions[:batch_size, prefix_step].reshape(batch_size, 1)
            context_parts.extend((
                self.act_embedding_table(action_tokens),
                obs_sequence[:batch_size, prefix_step + 1],
            ))
        raw_context = torch.cat(context_parts, dim=1)
        return self._trim_open_loop_context(raw_context), prefix_transitions

    def _trim_open_loop_context(self, raw_context: torch.Tensor) -> torch.Tensor:
        """Retain an observation-aligned window with room for the next edge."""
        keep_tokens = self.context_length - 3
        if raw_context.size(1) > keep_tokens:
            return raw_context[:, -keep_tokens:]
        return raw_context

    def _open_loop_hidden(self, raw_context: torch.Tensor) -> torch.Tensor:
        """Forward one raw absolute-position context through the transformer."""
        positions = torch.arange(raw_context.size(1), device=self.device)
        return self.transformer(
            raw_context + self._lookup_position_embeddings(positions)
        )

    def _rebuild_open_loop_cache_if_full(
            self, cache, raw_context: torch.Tensor, batch_size: int
    ):
        """Rebuild a full diagnostic cache from its exact retained raw tokens."""
        if cache.size < self.context_length - 1:
            return cache, raw_context
        raw_context = raw_context[:, -(self.context_length - 3):]
        cache = self.transformer.generate_empty_keys_values(
            n=batch_size, max_tokens=self.context_length
        )
        positions = torch.arange(raw_context.size(1), device=self.device)
        self.transformer(
            raw_context + self._lookup_position_embeddings(positions),
            past_keys_values=cache,
        )
        return cache, raw_context

    @staticmethod
    def _masked_horizon_mean(
            errors: torch.Tensor, valid_mask: torch.Tensor, horizon: int
    ) -> torch.Tensor:
        horizon_mask = valid_mask[:, horizon]
        if not horizon_mask.any():
            return errors.new_tensor(0.)
        return errors[horizon_mask, horizon].mean()

    @staticmethod
    def _masked_discounted_mean(
            values: torch.Tensor, mask: torch.Tensor, discounts: torch.Tensor
    ) -> torch.Tensor:
        return (values * mask * discounts).sum() / mask.sum().clamp_min(1)

    @torch.no_grad()
    def compute_open_loop_latent_diagnostics(
            self, obs_embeddings: torch.Tensor, target_obs_embeddings: torch.Tensor,
            actions: torch.Tensor, mask_padding: torch.Tensor
    ) -> Dict[str, float]:
        """Measure teacher-forcing exposure bias with an MCTS-style latent rollout."""
        self._validate_open_loop_support('Open-loop latent diagnostics')
        rollout = self._prepare_open_loop_batch(
            obs_embeddings,
            target_obs_embeddings,
            actions,
            mask_padding,
            self.open_loop_diagnostic_batch_size,
        )
        if rollout.transition_count <= 0:
            return dict(self._EMPTY_OPEN_LOOP_DIAGNOSTICS)

        action_sequence = actions[
            :rollout.batch_size, rollout.rollout_start:rollout.rollout_end
        ]
        valid_mask = mask_padding[
            :rollout.batch_size,
            rollout.rollout_start + 1:rollout.rollout_end + 1,
        ].bool()
        targets = rollout.targets[
            :, rollout.rollout_start + 1:rollout.rollout_end + 1
        ]

        was_training = self.training
        self.eval()
        try:
            # Use one dropout-free regime for all three paths so the ratios only
            # measure context rolling and predicted-latent exposure.
            teacher_outputs = self.forward(
                {
                    'obs_embeddings_and_act_tokens': (
                        rollout.observations[:, :rollout.rollout_end + 1],
                        actions[:rollout.batch_size, :rollout.rollout_end].unsqueeze(-1),
                    )
                },
                start_pos=[0] * rollout.batch_size,
            )
            teacher_predictions = teacher_outputs.logits_observations[
                :, rollout.rollout_start:rollout.rollout_end
            ].view(
                rollout.batch_size,
                rollout.transition_count,
                self.num_observations_tokens,
                self.embed_dim,
            )

            open_loop_cache = self.transformer.generate_empty_keys_values(
                n=rollout.batch_size, max_tokens=self.context_length
            )
            rolling_teacher_cache = self.transformer.generate_empty_keys_values(
                n=rollout.batch_size, max_tokens=self.context_length
            )
            open_loop_context = rollout.raw_context.detach()
            rolling_teacher_context = rollout.raw_context.detach()
            self.forward(
                {'obs_embeddings': open_loop_context},
                past_keys_values=open_loop_cache,
                is_init_infer=True,
                start_pos=0,
            )
            self.forward(
                {'obs_embeddings': rolling_teacher_context},
                past_keys_values=rolling_teacher_cache,
                is_init_infer=True,
                start_pos=0,
            )

            open_loop_predictions = []
            rolling_teacher_predictions = []
            for step in range(rollout.transition_count):
                action_tokens = action_sequence[:, step].reshape(rollout.batch_size, 1)
                open_loop_action_output = self.forward(
                    {'act_tokens': action_tokens},
                    past_keys_values=open_loop_cache,
                    is_init_infer=True,
                    start_pos=0,
                )
                rolling_teacher_action_output = self.forward(
                    {'act_tokens': action_tokens},
                    past_keys_values=rolling_teacher_cache,
                    is_init_infer=True,
                    start_pos=0,
                )
                predicted_observation = open_loop_action_output.logits_observations
                true_next_observation = rollout.observations[
                    :, rollout.rollout_start + step + 1
                ]
                open_loop_predictions.append(predicted_observation)
                rolling_teacher_predictions.append(
                    rolling_teacher_action_output.logits_observations
                )

                embedded_action = self.act_embedding_table(action_tokens)
                open_loop_context = torch.cat((
                    open_loop_context,
                    embedded_action.detach(),
                    predicted_observation.detach(),
                ), dim=1)
                rolling_teacher_context = torch.cat((
                    rolling_teacher_context,
                    embedded_action.detach(),
                    true_next_observation.detach(),
                ), dim=1)
                self.forward(
                    {'obs_embeddings': predicted_observation},
                    past_keys_values=open_loop_cache,
                    is_init_infer=True,
                    start_pos=0,
                )
                self.forward(
                    {'obs_embeddings': true_next_observation},
                    past_keys_values=rolling_teacher_cache,
                    is_init_infer=True,
                    start_pos=0,
                )
                open_loop_cache, open_loop_context = self._rebuild_open_loop_cache_if_full(
                    open_loop_cache, open_loop_context, rollout.batch_size
                )
                rolling_teacher_cache, rolling_teacher_context = (
                    self._rebuild_open_loop_cache_if_full(
                        rolling_teacher_cache,
                        rolling_teacher_context,
                        rollout.batch_size,
                    )
                )
        finally:
            self.train(was_training)

        open_loop_predictions = torch.stack(open_loop_predictions, dim=1)
        rolling_teacher_predictions = torch.stack(rolling_teacher_predictions, dim=1)
        open_loop_errors = F.mse_loss(
            open_loop_predictions, targets, reduction='none'
        ).mean(dim=(-1, -2))
        rolling_teacher_errors = F.mse_loss(
            rolling_teacher_predictions, targets, reduction='none'
        ).mean(dim=(-1, -2))
        teacher_errors = F.mse_loss(
            teacher_predictions, targets, reduction='none'
        ).mean(dim=(-1, -2))

        valid_count = valid_mask.sum().clamp_min(1)
        open_loop_mean = (open_loop_errors * valid_mask).sum() / valid_count
        rolling_teacher_mean = (rolling_teacher_errors * valid_mask).sum() / valid_count
        teacher_mean = (teacher_errors * valid_mask).sum() / valid_count
        valid_horizons = valid_mask.any(dim=0).nonzero(as_tuple=False).flatten()
        last_horizon = int(valid_horizons[-1].item()) if valid_horizons.numel() else 0
        middle_horizon = last_horizon // 2

        horizon_mean = self._masked_horizon_mean
        return {
            'open_loop_latent_mse_mean': float(open_loop_mean.item()),
            'open_loop_latent_mse_first': float(horizon_mean(open_loop_errors, valid_mask, 0).item()),
            'open_loop_latent_mse_middle': float(
                horizon_mean(open_loop_errors, valid_mask, middle_horizon).item()
            ),
            'open_loop_latent_mse_last': float(
                horizon_mean(open_loop_errors, valid_mask, last_horizon).item()
            ),
            'rolling_teacher_latent_mse_mean': float(rolling_teacher_mean.item()),
            'rolling_teacher_latent_mse_first': float(
                horizon_mean(rolling_teacher_errors, valid_mask, 0).item()
            ),
            'rolling_teacher_latent_mse_middle': float(
                horizon_mean(rolling_teacher_errors, valid_mask, middle_horizon).item()
            ),
            'rolling_teacher_latent_mse_last': float(
                horizon_mean(rolling_teacher_errors, valid_mask, last_horizon).item()
            ),
            'teacher_forced_latent_mse_mean': float(teacher_mean.item()),
            'teacher_forced_latent_mse_first': float(
                horizon_mean(teacher_errors, valid_mask, 0).item()
            ),
            'rolling_context_ratio': float(
                (rolling_teacher_mean / teacher_mean.clamp_min(1e-12)).item()
            ),
            'open_loop_exposure_ratio': float(
                (open_loop_mean / rolling_teacher_mean.clamp_min(1e-12)).item()
            ),
            'open_loop_total_ratio': float(
                (open_loop_mean / teacher_mean.clamp_min(1e-12)).item()
            ),
        }

    def compute_open_loop_consistency_loss(
            self, obs_embeddings: torch.Tensor, target_obs_embeddings: torch.Tensor,
            actions: torch.Tensor, mask_padding: torch.Tensor
    ) -> torch.Tensor:
        """Train on a short, differentiable MCTS-style latent rollout."""
        self._validate_open_loop_support('Open-loop consistency')
        rollout = self._prepare_open_loop_batch(
            obs_embeddings,
            target_obs_embeddings,
            actions,
            mask_padding,
            self.open_loop_consistency_batch_size,
            self.open_loop_consistency_horizon,
        )
        if rollout.transition_count <= 0:
            return obs_embeddings.sum() * 0.

        action_sequence = actions[
            :rollout.batch_size, rollout.rollout_start:rollout.rollout_end
        ]
        targets = rollout.targets[
            :, rollout.rollout_start + 1:rollout.rollout_end + 1
        ].detach()
        valid_mask = mask_padding[
            :rollout.batch_size,
            rollout.rollout_start + 1:rollout.rollout_end + 1,
        ].bool()
        raw_context = rollout.raw_context
        predictions = []

        was_training = self.training
        self.eval()
        try:
            for step in range(rollout.transition_count):
                action_tokens = action_sequence[:, step].reshape(rollout.batch_size, 1)
                raw_context = torch.cat(
                    (raw_context, self.act_embedding_table(action_tokens)), dim=1
                )
                hidden = self._open_loop_hidden(raw_context)
                predicted_observation = self.head_observations(
                    hidden, num_steps=raw_context.size(1), prev_steps=0
                )[:, -1:].contiguous()
                predictions.append(predicted_observation)
                raw_context = self._trim_open_loop_context(
                    torch.cat((raw_context, predicted_observation), dim=1)
                )
        finally:
            self.train(was_training)

        predictions = torch.stack(predictions, dim=1)
        per_transition_loss = F.mse_loss(
            predictions, targets, reduction='none'
        ).mean(dim=(-1, -2))
        valid_count = valid_mask.sum().clamp_min(1)
        return (per_transition_loss * valid_mask).sum() / valid_count

    def compute_open_loop_recurrent_loss(
            self, obs_embeddings: torch.Tensor, target_obs_embeddings: torch.Tensor,
            actions: torch.Tensor, mask_padding: torch.Tensor,
            labels_rewards: torch.Tensor, labels_policy: torch.Tensor,
            labels_value: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Mirror MuZero's learner exposure on a short differentiable rollout."""
        self._validate_open_loop_support('Open-loop recurrent training')
        rollout = self._prepare_open_loop_batch(
            obs_embeddings,
            target_obs_embeddings,
            actions,
            mask_padding,
            self.open_loop_consistency_batch_size,
            self.open_loop_consistency_horizon,
        )
        zero = obs_embeddings.sum() * 0.
        if rollout.transition_count <= 0:
            components = {
                name: zero
                for name in ('latent', 'reward', 'value', 'policy', 'policy_ce', 'policy_entropy')
            }
            return zero, components

        batch_slice = slice(0, rollout.batch_size)
        transition_slice = slice(rollout.rollout_start, rollout.rollout_end)
        state_slice = slice(rollout.rollout_start + 1, rollout.rollout_end + 1)
        action_sequence = actions[batch_slice, transition_slice]
        target_observations = rollout.targets[:, state_slice].detach()
        reward_targets = labels_rewards.view(
            actions.size(0), actions.size(1), -1
        )[batch_slice, transition_slice].detach()
        policy_targets = labels_policy.view(
            actions.size(0), actions.size(1), -1
        )[batch_slice, state_slice].detach()
        value_targets = labels_value.view(
            actions.size(0), actions.size(1), -1
        )[batch_slice, state_slice].detach()
        transition_valid_mask = mask_padding[batch_slice, transition_slice].bool()
        state_valid_mask = mask_padding[batch_slice, state_slice].bool()

        raw_context = rollout.raw_context
        predicted_observations = []
        reward_logits = []
        policy_logits = []
        value_logits = []

        was_training = self.training
        self.eval()
        try:
            for step in range(rollout.transition_count):
                action_tokens = action_sequence[:, step].reshape(rollout.batch_size, 1)
                raw_context = torch.cat(
                    (raw_context, self.act_embedding_table(action_tokens)), dim=1
                )
                action_hidden = self._open_loop_hidden(raw_context)
                predicted_observation = self.head_observations(
                    action_hidden, num_steps=raw_context.size(1), prev_steps=0
                )[:, -1:].contiguous()
                predicted_observations.append(predicted_observation)
                reward_logits.append(self.head_rewards(
                    action_hidden, num_steps=raw_context.size(1), prev_steps=0
                )[:, -1])

                raw_context = torch.cat((raw_context, predicted_observation), dim=1)
                state_hidden = self._open_loop_hidden(raw_context)
                next_policy_logits = self.head_policy(
                    state_hidden, num_steps=raw_context.size(1), prev_steps=0
                )[:, -1]
                if self.use_policy_logits_clip:
                    next_policy_logits = self._apply_policy_logits_control(next_policy_logits)
                policy_logits.append(next_policy_logits)
                value_logits.append(self.head_value(
                    state_hidden, num_steps=raw_context.size(1), prev_steps=0
                )[:, -1])
                raw_context = self._trim_open_loop_context(raw_context)
        finally:
            self.train(was_training)

        predicted_observations = torch.stack(predicted_observations, dim=1)
        reward_logits = torch.stack(reward_logits, dim=1)
        policy_logits = torch.stack(policy_logits, dim=1)
        value_logits = torch.stack(value_logits, dim=1)

        latent_loss = F.mse_loss(
            predicted_observations, target_observations, reduction='none'
        ).mean(dim=(-1, -2))
        reward_loss = -(F.log_softmax(reward_logits, dim=-1) * reward_targets).sum(dim=-1)
        value_loss = -(F.log_softmax(value_logits, dim=-1) * value_targets).sum(dim=-1)
        policy_logits_for_loss = policy_logits
        if self.use_policy_loss_temperature and self.policy_loss_temperature != 1.0:
            policy_logits_for_loss = policy_logits_for_loss / self.policy_loss_temperature
        policy_log_probs = F.log_softmax(policy_logits_for_loss, dim=-1)
        policy_ce = -(policy_log_probs * policy_targets).sum(dim=-1)
        policy_probs = F.softmax(policy_logits_for_loss, dim=-1)
        policy_entropy = -(policy_probs * policy_log_probs).sum(dim=-1)

        transition_mask = transition_valid_mask.to(latent_loss.dtype)
        state_mask = state_valid_mask.to(latent_loss.dtype)
        reward_discounts = self.gamma ** torch.arange(
            rollout.transition_count,
            device=transition_mask.device,
            dtype=latent_loss.dtype,
        )
        state_discounts = self.gamma ** torch.arange(
            1,
            rollout.transition_count + 1,
            device=state_mask.device,
            dtype=latent_loss.dtype,
        )
        masked_mean = self._masked_discounted_mean
        policy_ce_mean = masked_mean(policy_ce, state_mask, state_discounts)
        policy_entropy_mean = masked_mean(policy_entropy, state_mask, state_discounts)
        components = {
            'latent': masked_mean(latent_loss, state_mask, state_discounts),
            'reward': masked_mean(reward_loss, transition_mask, reward_discounts),
            'value': masked_mean(value_loss, state_mask, state_discounts),
            'policy': policy_ce_mean - self.policy_entropy_weight * policy_entropy_mean,
            'policy_ce': policy_ce_mean,
            'policy_entropy': policy_entropy_mean,
        }
        total = (
            10. * components['latent']
            + components['reward']
            + 0.5 * components['value']
            + components['policy']
        )
        return total, components

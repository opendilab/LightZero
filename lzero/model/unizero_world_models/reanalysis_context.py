"""Online-aligned root contexts used by UniZero replay reanalysis."""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .utils import hash_state


class ReanalysisContextMixin:
    """Exact online-style replay contexts for UniZero reanalysis targets."""

    def _normalize_reanalysis_action(self, action, latent_dtype: torch.dtype):
        action_tensor = torch.as_tensor(action, device=self.device)
        if self.continuous_action_space:
            return action_tensor.to(dtype=latent_dtype).reshape(-1)
        return int(action_tensor.reshape(-1)[0].item())

    def _embed_reanalysis_action(
            self, action, latent_dtype: torch.dtype, task_id: Optional[int]
    ) -> torch.Tensor:
        """Embed one replay action while preserving the historical helper API."""
        return self._embed_reanalysis_actions(
            action, action_count=1, latent_dtype=latent_dtype, task_id=task_id
        )[0]

    def _embed_reanalysis_actions(
            self, actions, action_count: int, latent_dtype: torch.dtype,
            task_id: Optional[int]
    ) -> torch.Tensor:
        """Embed a flat replay-action batch with a single module invocation.

        Context construction used to call the embedding table once for every action
        in every root window.  With Atari B=256/H=10 that creates roughly 28k tiny
        CUDA kernels per learner update.  Keep the first-action semantics of the old
        scalar helper, but normalize and embed the complete batch at once.
        """
        action_count = int(action_count)
        if action_count <= 0:
            raise ValueError(f'action_count must be positive, got {action_count}')
        try:
            action_tensor = torch.as_tensor(actions, device=self.device)
        except (TypeError, ValueError):
            # Variable sources such as replay lists may contain tensor/ndarray
            # action vectors. Stack those vectors without converting each one to
            # a Python scalar (which would synchronize CUDA tensors).
            action_tensor = torch.stack([
                torch.as_tensor(action, device=self.device).reshape(-1)
                for action in actions
            ])
        if self.continuous_action_space:
            action_tensor = action_tensor.to(dtype=latent_dtype).reshape(action_count, -1)
            action_embedding = self.act_embedding_table
            if isinstance(action_embedding, nn.ModuleList):
                if task_id is None:
                    raise ValueError(
                        'Continuous multi-task action embeddings require an explicit task_id.'
                    )
                action_embedding = action_embedding[task_id]
        else:
            # Legacy normalization selected the first scalar from each action.
            action_tensor = action_tensor.reshape(action_count, -1)[:, 0].long()
            action_embedding = self.act_embedding_table
        return action_embedding(action_tensor).reshape(action_count, -1, self.embed_dim)

    def build_reanalysis_root_token_contexts(
            self,
            latent_state_roots,
            batch_actions,
            roots_per_sequence: int,
            history_latent_segment=None,
            history_action_segment=None,
            task_id=None,
    ):
        """Build the exact online-style raw-token window for every replay root."""
        if task_id is not None and getattr(self, 'task_embed_option', None) not in (None, 'none'):
            raise NotImplementedError(
                'Replay-root raw-token contexts do not yet support task_embed_option='
                f'{self.task_embed_option!r}; use no task-token conditioning or disable '
                'contextual reanalysis/bootstrap targets.'
            )
        roots = torch.as_tensor(latent_state_roots, device=self.device)
        if roots_per_sequence <= 0 or roots.size(0) % roots_per_sequence != 0:
            raise ValueError(
                'Reanalysis roots must form complete fixed-length sequences: '
                f'root_count={roots.size(0)}, roots_per_sequence={roots_per_sequence}.'
            )
        sequence_count = roots.size(0) // roots_per_sequence
        actions = torch.as_tensor(batch_actions, device=self.device)
        if actions.ndim < 2 or actions.size(0) != sequence_count:
            raise ValueError(
                'Reanalysis actions must align with root sequences: '
                f'action_shape={tuple(actions.shape)}, sequence_count={sequence_count}.'
            )
        if actions.size(1) < roots_per_sequence - 1:
            raise ValueError(
                'Each replay sequence needs one action between adjacent roots: '
                f'{actions.size(1)} < {roots_per_sequence - 1}.'
            )

        if history_latent_segment is None:
            history_latent_segment = [[] for _ in range(sequence_count)]
        if history_action_segment is None:
            history_action_segment = [[] for _ in range(sequence_count)]
        if len(history_latent_segment) != sequence_count or (
                len(history_action_segment) != sequence_count
        ):
            raise ValueError('Replay history observations/actions must align with root sequences.')

        keep_tokens = self.context_length - 3
        if keep_tokens <= 0:
            raise ValueError(
                'UniZero recurrent context_length must reserve at least one raw token, '
                f'got {self.context_length}.'
            )

        rollout_action_count = sequence_count * (roots_per_sequence - 1)
        rollout_action_embeddings = None
        if rollout_action_count:
            rollout_action_embeddings = self._embed_reanalysis_actions(
                actions[:, :roots_per_sequence - 1],
                action_count=rollout_action_count,
                latent_dtype=roots.dtype,
                task_id=task_id,
            ).reshape(sequence_count, roots_per_sequence - 1, -1, self.embed_dim)

        prefix_action_lengths = [len(sequence) for sequence in history_action_segment]
        flat_prefix_actions = [
            action for sequence in history_action_segment for action in sequence
        ]
        prefix_action_embeddings = None
        if flat_prefix_actions:
            prefix_action_embeddings = self._embed_reanalysis_actions(
                flat_prefix_actions,
                action_count=len(flat_prefix_actions),
                latent_dtype=roots.dtype,
                task_id=task_id,
            )

        # Build one raw trajectory timeline per replay sequence. Every root context
        # is then a view into that timeline. This is exactly equivalent to repeatedly
        # rebuilding/catting the full prefix for every root, but changes B*(H+1)
        # concatenations and tens of thousands of action-embedding calls into B
        # concatenations and at most two batched embedding calls.
        root_contexts = []
        prefix_action_offset = 0
        for sequence_index in range(sequence_count):
            prefix_latents = history_latent_segment[sequence_index]
            prefix_actions = history_action_segment[sequence_index]
            if len(prefix_latents) != len(prefix_actions):
                raise ValueError(
                    'Every historical observation must have its outgoing replay action: '
                    f'{len(prefix_latents)} != {len(prefix_actions)}.'
                )

            timeline_parts = []
            timeline_token_count = 0
            root_end_offsets = []
            prefix_action_count = prefix_action_lengths[sequence_index]
            for prefix_index, latent in enumerate(prefix_latents):
                observation = torch.as_tensor(
                    latent, device=self.device
                ).reshape(-1, self.embed_dim)
                action_embedding = prefix_action_embeddings[
                    prefix_action_offset + prefix_index
                ]
                timeline_parts.extend((observation, action_embedding))
                timeline_token_count += observation.size(0) + action_embedding.size(0)
            prefix_action_offset += prefix_action_count

            root_start = sequence_index * roots_per_sequence
            for root_offset in range(roots_per_sequence):
                current_root = roots[root_start + root_offset].reshape(-1, self.embed_dim)
                timeline_parts.append(current_root)
                timeline_token_count += current_root.size(0)
                root_end_offsets.append(timeline_token_count)

                if root_offset < roots_per_sequence - 1:
                    action_embedding = rollout_action_embeddings[
                        sequence_index, root_offset
                    ]
                    timeline_parts.append(action_embedding)
                    timeline_token_count += action_embedding.size(0)

            timeline = torch.cat(timeline_parts, dim=0).detach()
            for root_end in root_end_offsets:
                context_start = (
                    root_end - keep_tokens
                    if root_end >= self.context_length - 1
                    else 0
                )
                root_contexts.append(timeline[context_start:root_end])

        return root_contexts

    def _group_root_token_contexts(
            self, root_token_contexts, purpose: str
    ) -> Tuple[
        Dict[int, List[Tuple[int, torch.Tensor]]],
        List[torch.Tensor],
    ]:
        """Normalize contexts and group equal lengths for batched transformer calls."""
        groups = {}
        normalized = []
        for index, context in enumerate(root_token_contexts):
            context = torch.as_tensor(
                context, device=self.device
            ).reshape(-1, self.embed_dim)
            context_length = context.size(0)
            if context_length <= 0 or context_length > self.context_length - 2:
                raise ValueError(
                    f'A {purpose} root context must leave room for action and next observation: '
                    f'length={context_length}, context_length={self.context_length}.'
                )
            normalized.append(context)
            groups.setdefault(context_length, []).append((index, context))
        return groups, normalized

    def _position_root_context_batch(
            self, raw_context_batch: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[int]]:
        """Apply the position convention used by online root inference."""
        if self.config.rotary_emb:
            return raw_context_batch, 0
        positions = torch.arange(raw_context_batch.size(1), device=self.device)
        return raw_context_batch + self._lookup_position_embeddings(positions), None

    def _context_prediction_head(self, name, task_id=None):
        """Select the prediction head matching a contextual replay-root task."""
        multi_task_head = getattr(self, f'{name}_multi_task', None)
        if task_id is not None and multi_task_head is not None and not (
                getattr(self, 'use_moe_head', False)
                or getattr(self, 'use_softmoe_head', False)
        ):
            head_index = 0 if getattr(self, 'share_head', False) else int(task_id)
            return multi_task_head[head_index]
        return getattr(self, name)

    @torch.no_grad()
    def seed_reanalysis_root_caches(
            self, latent_state_roots, root_token_contexts, task_id=None
    ):
        """Materialize replay-root K/V entries and return matching root priors."""
        if len(latent_state_roots) != len(root_token_contexts):
            raise ValueError(
                'Replay root states and token contexts must have equal lengths: '
                f'{len(latent_state_roots)} != {len(root_token_contexts)}.'
            )
        if not root_token_contexts:
            return []
        if not self.reanalyze_phase:
            raise RuntimeError('Replay-root caches may only be seeded during reanalysis.')

        root_states = torch.as_tensor(latent_state_roots, device=self.device)
        length_groups, normalized_contexts = self._group_root_token_contexts(
            root_token_contexts, 'seeded replay'
        )
        root_policy_logits = [None] * len(root_token_contexts)
        transformer_task_id = 0 if task_id is None else int(task_id)

        for context_length, entries in length_groups.items():
            indices = [index for index, _ in entries]
            raw_context_batch = torch.stack([context for _, context in entries])
            positioned_context, transformer_start_pos = self._position_root_context_batch(
                raw_context_batch
            )
            seeded_cache = self.transformer.generate_empty_keys_values(
                n=len(indices), max_tokens=self.context_length
            )
            hidden = self.transformer(
                positioned_context,
                past_keys_values=seeded_cache,
                start_pos=transformer_start_pos,
                task_id=transformer_task_id,
            )
            policy_logits = self._context_prediction_head('head_policy', task_id)(
                hidden, num_steps=context_length, prev_steps=0
            )[:, -1]
            if self.use_policy_logits_clip:
                policy_logits = self._apply_policy_logits_control(policy_logits)
            policy_logits = policy_logits.detach().cpu().numpy()
            for local_index, root_index in enumerate(indices):
                root_policy_logits[root_index] = policy_logits[local_index].tolist()

            # Every replay root is an independent environment in this chunk.
            # Its prior and recurrent cache must come from this identical prefix.
            self.keys_values_wm = seeded_cache
            self.keys_values_wm_size_list_current = [context_length] * len(indices)
            self.keys_values_wm_token_context_list = [
                normalized_contexts[index].detach().clone() for index in indices
            ]
            self.update_cache_context(
                root_states[indices],
                is_init_infer=True,
                valid_context_lengths=[context_length] * len(indices),
                env_ids=indices,
            )
            for index in indices:
                cache_key = hash_state(
                    root_states[index].reshape(-1).detach().cpu().numpy()
                )
                self._reanalysis_seeded_root_keys.add((index, cache_key))
                self.reanalysis_root_seed_count += 1

        return root_policy_logits

    @torch.no_grad()
    def evaluate_root_token_context_values(self, root_token_contexts, task_id=None):
        """Evaluate bootstrap values from exact online-style root contexts."""
        if not root_token_contexts:
            return torch.empty((0, self.support_size), device=self.device)

        length_groups, _ = self._group_root_token_contexts(
            root_token_contexts, 'bootstrap'
        )
        root_value_logits = None
        transformer_task_id = 0 if task_id is None else int(task_id)
        for context_length, entries in length_groups.items():
            indices = [index for index, _ in entries]
            raw_context_batch = torch.stack([context for _, context in entries])
            positioned_context, transformer_start_pos = self._position_root_context_batch(
                raw_context_batch
            )
            hidden = self.transformer(
                positioned_context,
                start_pos=transformer_start_pos,
                task_id=transformer_task_id,
            )
            contextual_values = self._context_prediction_head('head_value', task_id)(
                hidden, num_steps=context_length, prev_steps=0
            )[:, -1]
            if root_value_logits is None:
                root_value_logits = contextual_values.new_empty(
                    (len(root_token_contexts), *contextual_values.shape[1:])
                )
            root_value_logits[torch.as_tensor(indices, device=self.device)] = contextual_values

        return root_value_logits

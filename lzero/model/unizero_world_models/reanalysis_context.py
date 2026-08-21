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
        if self.continuous_action_space:
            action_tensor = torch.as_tensor(
                action, device=self.device, dtype=latent_dtype
            ).reshape(1, -1)
            action_embedding = self.act_embedding_table
            if isinstance(action_embedding, nn.ModuleList):
                if task_id is None:
                    raise ValueError(
                        'Continuous multi-task action embeddings require an explicit task_id.'
                    )
                action_embedding = action_embedding[task_id]
        else:
            action_tensor = torch.as_tensor([action], device=self.device).long()
            action_embedding = self.act_embedding_table
        return action_embedding(action_tensor).reshape(-1, self.embed_dim)

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

        root_contexts = []
        normalize_action = self._normalize_reanalysis_action
        for sequence_index in range(sequence_count):
            prefix_latents = history_latent_segment[sequence_index]
            prefix_actions = history_action_segment[sequence_index]
            if len(prefix_latents) != len(prefix_actions):
                raise ValueError(
                    'Every historical observation must have its outgoing replay action: '
                    f'{len(prefix_latents)} != {len(prefix_actions)}.'
                )

            observation_history = [
                torch.as_tensor(latent, device=self.device).reshape(-1, self.embed_dim)
                for latent in prefix_latents
            ]
            action_history = [
                normalize_action(action, roots.dtype) for action in prefix_actions
            ]

            root_start = sequence_index * roots_per_sequence
            for root_offset in range(roots_per_sequence):
                current_root = roots[root_start + root_offset].reshape(-1, self.embed_dim)
                token_parts = []
                for observation, action in zip(observation_history, action_history):
                    token_parts.extend((
                        observation,
                        self._embed_reanalysis_action(action, roots.dtype, task_id),
                    ))
                token_parts.append(current_root)
                raw_context = torch.cat(token_parts, dim=0).detach()
                if raw_context.size(0) >= self.context_length - 1:
                    raw_context = raw_context[-keep_tokens:]
                root_contexts.append(raw_context)

                if root_offset < roots_per_sequence - 1:
                    observation_history.append(current_root)
                    action_history.append(normalize_action(
                        actions[sequence_index, root_offset], roots.dtype
                    ))

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
        root_value_logits = [None] * len(root_token_contexts)
        transformer_task_id = 0 if task_id is None else int(task_id)
        for context_length, entries in length_groups.items():
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
            for local_index, (root_index, _) in enumerate(entries):
                root_value_logits[root_index] = contextual_values[local_index]

        return torch.stack(root_value_logits)

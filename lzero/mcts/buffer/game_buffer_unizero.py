from contextlib import contextmanager
import logging
from typing import Any, List, Tuple, Union, TYPE_CHECKING, Optional

import numpy as np
import torch
from ding.utils import BUFFER_REGISTRY

from lzero.mcts.tree_search.mcts_ctree import UniZeroMCTSCtree as MCTSCtree
from lzero.mcts.utils import prepare_observation
from lzero.policy import DiscreteSupport, to_detach_cpu_numpy, concat_output, inverse_scalar_transform
from .game_buffer_muzero import MuZeroGameBuffer

if TYPE_CHECKING:
    from lzero.policy import MuZeroPolicy, EfficientZeroPolicy, SampledEfficientZeroPolicy
from line_profiler import line_profiler


@contextmanager
def _world_model_reanalysis_phase(world_model):
    """Temporarily enable replay reanalysis without leaking mutable model state."""
    previous_phase = world_model.reanalyze_phase
    world_model.reanalyze_phase = True
    try:
        yield
    finally:
        world_model.reanalyze_phase = previous_phase


@BUFFER_REGISTRY.register('game_buffer_unizero')
class UniZeroGameBuffer(MuZeroGameBuffer):
    """
    Overview:
        The specific game buffer for MuZero policy.
    """

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        """
        Overview:
            Use the default configuration mechanism. If a user passes in a cfg with a key that matches an existing key
            in the default configuration, the user-provided value will override the default configuration. Otherwise,
            the default configuration will be used.
        """
        default_config = self.default_config()
        default_config.update(cfg)
        self._cfg = default_config
        assert self._cfg.env_type in ['not_board_games', 'board_games']
        self.replay_buffer_size = self._cfg.replay_buffer_size
        self.batch_size = self._cfg.batch_size
        self._alpha = self._cfg.priority_prob_alpha
        self._beta = self._cfg.priority_prob_beta

        self.keep_ratio = 1
        self.model_update_interval = 10
        self.num_of_collected_episodes = 0
        self.base_idx = 0
        self.clear_time = 0

        self.game_segment_buffer = []
        self.game_pos_priorities = []
        self.game_segment_game_pos_look_up = []
        self.sample_type = self._cfg.sample_type  # 'transition' or 'episode'

        if hasattr(self._cfg, 'task_id'):
            self.task_id = self._cfg.task_id
            print(f"Task ID is set to {self.task_id}.")
            try:
                self.action_space_size = self._cfg.model.action_space_size_list[self.task_id]
            except Exception:
                self.action_space_size = self._cfg.model.action_space_size
        else:
            self.task_id = None
            print("No task_id found in configuration. Task ID is set to None.")
            self.action_space_size = self._cfg.model.action_space_size

        self.value_support = DiscreteSupport(*self._cfg.model.value_support_range)
        self.reward_support = DiscreteSupport(*self._cfg.model.reward_support_range)
        self._validate_contextual_reanalysis_config()

    def _validate_contextual_reanalysis_config(self) -> None:
        world_model_cfg = getattr(self._cfg.model, 'world_model_cfg', None)
        task_embed_option = getattr(world_model_cfg, 'task_embed_option', 'none')
        if (
            self._use_contextual_reanalysis()
            and self.task_id is not None
            and task_embed_option not in (None, 'none')
        ):
            raise NotImplementedError(
                'contextual_reanalysis does not yet support multi-task task-token conditioning '
                f'(task_embed_option={task_embed_option!r}). Disable it or use task_embed_option="none".'
            )

    def _use_contextual_reanalysis(self) -> bool:
        """Whether replay MCTS roots should be rebuilt from their short online history.

        H+1 target alignment and bounded MCTS chunking are unconditional correctness fixes.
        Replacing the legacy root prior/cache with a replay-history-conditioned one is an
        algorithmic choice, however, so it remains opt-in for old-experiment reproducibility.
        """
        return bool(getattr(self._cfg, 'contextual_reanalysis', False))

    def _bootstrap_value_context_diagnostic_due(self):
        """Advance the contextual-target counter and select sparse legacy comparisons."""
        self._bootstrap_value_context_batch_count = (
            getattr(self, '_bootstrap_value_context_batch_count', 0) + 1
        )
        return (
            self._bootstrap_value_context_batch_count == 1
            or self._bootstrap_value_context_batch_count % 1000 == 0
        )

    def _encode_bootstrap_root_latents(self, model, observation_batch):
        """Encode replay roots without an unused full-sequence Transformer forward."""
        if self.task_id is None:
            return model.world_model.tokenizer.encode_to_obs_embeddings(observation_batch)
        return model.world_model.tokenizer.encode_to_obs_embeddings(
            observation_batch, task_id=self.task_id
        )

    def _has_contextual_reanalysis_api(self, model) -> bool:
        """Whether the selected model can build and seed exact replay-root contexts."""
        world_model = model.world_model
        return (
            self._use_contextual_reanalysis()
            and bool(getattr(self._cfg, 'mcts_ctree', False))
            and callable(getattr(world_model, 'build_reanalysis_root_token_contexts', None))
            and callable(getattr(world_model, 'seed_reanalysis_root_caches', None))
        )

    def _prepare_reanalysis_root_inputs(
            self, model, observation_batch, batch_action, sequence_start_timesteps
    ):
        """Return root rewards, priors and latents for replay MCTS.

        Contextual reanalysis recomputes every root prior from the exact replay-history
        token window while seeding its KV cache.  Running ``initial_inference`` first
        would therefore discard its Transformer value/policy predictions; only the
        tokenizer output is needed.  The legacy path remains unchanged.
        """
        if self._has_contextual_reanalysis_api(model):
            latent_state_roots = self._encode_bootstrap_root_latents(
                model, observation_batch
            ).detach().cpu().numpy()
            root_count = len(latent_state_roots)
            # MuZero initial inference defines every root reward as exactly zero.
            # Priors are materialized from the matching context immediately before
            # each chunked search, so these placeholders are never consumed.
            return [0.0] * root_count, [None] * root_count, latent_state_roots

        if self.task_id is not None:
            output = model.initial_inference(
                observation_batch,
                batch_action[:self.reanalyze_num],
                task_id=self.task_id,
            )
        else:
            output = model.initial_inference(
                observation_batch,
                batch_action[:self.reanalyze_num],
                start_pos=sequence_start_timesteps,
            )
        output.latent_state, output.value, output.policy_logits = to_detach_cpu_numpy(
            [
                output.latent_state,
                inverse_scalar_transform(output.value, self.value_support),
                output.policy_logits,
            ]
        )
        _, rewards, policy_logits, latent_states = concat_output(
            [output], data_type='muzero'
        )
        return (
            np.asarray(rewards).reshape(-1).tolist(),
            policy_logits.tolist(),
            latent_states,
        )

    def _log_bootstrap_value_context_diagnostics(
            self, legacy_values, contextual_values, value_mask, root_token_contexts
    ):
        """Log a selected valid-root comparison for contextual TD bootstrap values."""
        legacy = np.asarray(legacy_values).reshape(-1)
        contextual = np.asarray(contextual_values).reshape(-1)
        valid = np.asarray(value_mask, dtype=bool).reshape(-1).copy()
        if legacy.size != contextual.size or legacy.size != valid.size:
            raise RuntimeError(
                'Bootstrap-value diagnostics require aligned legacy/contextual/mask arrays: '
                f'{legacy.size}/{contextual.size}/{valid.size}.'
            )
        # The legacy learner target path returns its H+1 slot by repeating the H-th prediction;
        # that placeholder is discarded by the learner and is not a meaningful comparison with
        # the contextual value of the real final observation. Exclude it from diagnostics.
        roots_per_sequence = int(self._cfg.num_unroll_steps) + 1
        if roots_per_sequence > 0 and valid.size % roots_per_sequence == 0:
            valid[roots_per_sequence - 1::roots_per_sequence] = False
        if not valid.any():
            logging.info(
                'UniZero bootstrap-value context diagnostics: batch=%d, valid_roots=0',
                self._bootstrap_value_context_batch_count,
            )
            return
        delta = contextual[valid] - legacy[valid]
        lengths = np.asarray([int(context.size(0)) for context in root_token_contexts])
        valid_lengths = lengths[valid]
        logging.info(
            'UniZero bootstrap-value context diagnostics: batch=%d, valid_roots=%d, '
            'legacy[mean/std]=%.5f/%.5f, contextual[mean/std]=%.5f/%.5f, '
            'delta[abs_mean/rms/max]=%.5f/%.5f/%.5f, context_tokens[min/mean/max]=%d/%.2f/%d',
            self._bootstrap_value_context_batch_count,
            int(valid.sum()),
            float(legacy[valid].mean()),
            float(legacy[valid].std()),
            float(contextual[valid].mean()),
            float(contextual[valid].std()),
            float(np.abs(delta).mean()),
            float(np.sqrt(np.mean(np.square(delta)))),
            float(np.abs(delta).max()),
            int(valid_lengths.min()),
            float(valid_lengths.mean()),
            int(valid_lengths.max()),
        )

    def _prepare_reward_value_context(
            self, batch_index_list, game_segment_list, pos_in_game_segment_list,
            total_transitions
    ):
        """Add the real replay prefix preceding each TD-bootstrap sequence.

        MuZero's stacked root already carries recent frames.  A single-frame UniZero root instead
        relies on its Transformer history, so evaluating bootstrap values from the bootstrap state
        alone changes the state representation used by the target model.  Keep enough replay
        observations/actions to reconstruct the same short raw-token window as online inference.
        """
        context = super()._prepare_reward_value_context(
            batch_index_list, game_segment_list, pos_in_game_segment_list, total_transitions
        )
        # Contextual TD bootstrap is opt-in.  Avoid slicing and retaining replay
        # histories on the default legacy target path, where they are never read.
        if not getattr(self._cfg, 'bootstrap_value_context', False):
            return context

        td_steps_list = context[6]
        roots_per_sequence = self._cfg.num_unroll_steps + 1
        world_model_cfg = getattr(self._cfg.model, 'world_model_cfg', None)
        context_length = int(getattr(world_model_cfg, 'context_length', 2))
        max_history_transitions = max(context_length // 2, 0)
        history_observation_segment, history_action_segment = [], []

        for sequence_index, (game_segment, state_index) in enumerate(zip(
                game_segment_list, pos_in_game_segment_list
        )):
            td_steps = int(td_steps_list[sequence_index * roots_per_sequence])
            bootstrap_start = int(state_index) + td_steps
            history_start = max(0, bootstrap_start - max_history_transitions)
            history_len = bootstrap_start - history_start
            history_game_obs = game_segment.get_unroll_obs(history_start, history_len)
            history_observation_segment.append([
                history_game_obs[offset:offset + self._cfg.model.frame_stack_num]
                for offset in range(history_len)
            ])
            history_action_segment.append(
                game_segment.action_segment[history_start:bootstrap_start].tolist()
            )

        return [*context, history_observation_segment, history_action_segment]

    #@profile
    def sample(
            self, batch_size: int, policy: Union["MuZeroPolicy", "EfficientZeroPolicy", "SampledEfficientZeroPolicy"]
    ) -> List[Any]:
        """
        Overview:
            sample data from ``GameBuffer`` and prepare the current and target batch for training.
        Arguments:
            - batch_size (:obj:`int`): batch size.
            - policy (:obj:`Union["MuZeroPolicy", "EfficientZeroPolicy", "SampledEfficientZeroPolicy"]`): policy.
        Returns:
            - train_data (:obj:`List`): List of train data, including current_batch and target_batch.
        """
        policy._target_model.to(self._cfg.device)
        policy._target_model.eval()

        # obtain the current_batch and prepare target context
        reward_value_context, policy_re_context, policy_non_re_context, current_batch = self._make_batch(
            batch_size, self._cfg.reanalyze_ratio
        )

        # current_batch = [obs_list, action_list, bootstrap_action_list, mask_list, batch_index_list, weights_list, make_time_list, timestep_list]

        # target reward, target value
        batch_rewards, batch_target_values = self._compute_target_reward_value(
            reward_value_context, policy._target_model, current_batch[2], current_batch[7]
        )

        # target policy
        batch_target_policies_re = self._compute_target_policy_reanalyzed(
            policy_re_context, policy._target_model, current_batch[1]
        )  # current_batch[1] is batch_action
        batch_target_policies_non_re = self._compute_target_policy_non_reanalyzed(
            policy_non_re_context, self.action_space_size
        )

        # fusion of batch_target_policies_re and batch_target_policies_non_re to batch_target_policies
        if 0 < self._cfg.reanalyze_ratio < 1:
            batch_target_policies = np.concatenate([batch_target_policies_re, batch_target_policies_non_re])
        elif self._cfg.reanalyze_ratio == 1:
            batch_target_policies = batch_target_policies_re
        elif self._cfg.reanalyze_ratio == 0:
            batch_target_policies = batch_target_policies_non_re

        target_batch = [batch_rewards, batch_target_values, batch_target_policies]

        # a batch contains the current_batch and the target_batch
        train_data = [current_batch, target_batch]
        return train_data

    def get_on_policy_indices(self, collection_train_iter: int) -> np.ndarray:
        """Return flat replay indices produced by exactly one collection policy version."""
        indices = []
        horizon = int(self._cfg.num_unroll_steps)
        for flat_index, (global_segment_index, position) in enumerate(self.game_segment_game_pos_look_up):
            segment_index = global_segment_index - self.base_idx
            segment = self.game_segment_buffer[segment_index]
            if (
                position % horizon == 0
                and getattr(segment, 'collection_train_iter', None) == int(collection_train_iter)
            ):
                indices.append(flat_index)
        return np.asarray(indices, dtype=np.int64)

    def _orig_data_from_indices(self, indices: np.ndarray, collection_train_iter: int) -> Tuple[Any]:
        game_segments, positions = [], []
        for flat_index in np.asarray(indices, dtype=np.int64).reshape(-1):
            global_segment_index, position = self.game_segment_game_pos_look_up[int(flat_index)]
            segment = self.game_segment_buffer[global_segment_index - self.base_idx]
            actual_version = getattr(segment, 'collection_train_iter', None)
            if actual_version != int(collection_train_iter):
                raise RuntimeError(
                    'Stale PPO sample detected: '
                    f'expected policy version {collection_train_iter}, got {actual_version}'
                )
            game_segments.append(segment)
            positions.append(position)
        batch_size = len(game_segments)
        return (
            game_segments,
            positions,
            np.asarray(indices, dtype=np.int64),
            np.ones(batch_size, dtype=np.float32),
            np.zeros(batch_size, dtype=np.float64),
        )

    def sample_on_policy(
            self, indices: np.ndarray, policy: Any, collection_train_iter: int
    ) -> List[Any]:
        """Build a PPO minibatch exclusively from the latest collected rollout."""
        if len(indices) == 0:
            raise ValueError('Cannot build an empty PPO minibatch')
        orig_data = self._orig_data_from_indices(indices, collection_train_iter)
        _, _, _, current_batch = self._make_batch(
            len(indices), reanalyze_ratio=0.0, orig_data=orig_data, include_ppo=True
        )

        horizon = int(self._cfg.num_unroll_steps)
        rewards, returns, policies = [], [], []
        for segment, position in zip(orig_data[0], orig_data[1]):
            reward = np.asarray(segment.reward_segment[position:position + horizon + 1], dtype=np.float32).tolist()
            reward += [0.0] * (horizon + 1 - len(reward))
            value_return = np.asarray(segment.return_segment[position:position + horizon + 1], dtype=np.float32).tolist()
            value_return += [0.0] * (horizon + 1 - len(value_return))
            rewards.append(reward)
            returns.append(value_return)
            policies.append(np.zeros((horizon + 1, self.action_space_size), dtype=np.float32))

        target_batch = [
            np.asarray(rewards, dtype=np.float32),
            np.asarray(returns, dtype=np.float32),
            np.asarray(policies, dtype=np.float32),
        ]
        return [current_batch, target_batch]

    def sample_world_model(self, batch_size: int) -> List[Any]:
        """Sample replay for latent/reward learning without MCTS target work.

        In PPO mode the actor and critic heads are frozen during this phase, so
        target-network value inference and policy reanalysis would be pure cost.
        """
        reward_value_context, _, _, current_batch = self._make_batch(
            batch_size, reanalyze_ratio=0.0
        )
        positions = reward_value_context[2]
        reward_segments = reward_value_context[3]
        horizon = int(self._cfg.num_unroll_steps)
        rewards = []
        for position, reward_segment in zip(positions, reward_segments):
            values = np.asarray(
                reward_segment[position:position + horizon + 1], dtype=np.float32
            ).reshape(-1).tolist()
            values += [0.0] * (horizon + 1 - len(values))
            rewards.append(values)
        target_batch = [
            np.asarray(rewards, dtype=np.float32),
            np.zeros((len(rewards), horizon + 1), dtype=np.float32),
            np.zeros(
                (len(rewards), horizon + 1, self.action_space_size), dtype=np.float32
            ),
        ]
        return [current_batch, target_batch]

    def release_on_policy_data(self, collection_train_iter: int) -> None:
        """Drop rollout-only tensors after PPO epochs to keep replay memory bounded."""
        for segment in self.game_segment_buffer:
            if getattr(segment, 'collection_train_iter', None) == int(collection_train_iter):
                segment.behavior_log_prob_segment = np.asarray([], dtype=np.float32)
                segment.behavior_action_mask_segment = np.asarray([], dtype=np.bool_)
                segment.behavior_policy_feature_segment = np.asarray([], dtype=np.float32)
                segment.advantage_segment = np.asarray([], dtype=np.float32)
                segment.return_segment = np.asarray([], dtype=np.float32)

    #@profile
    def _make_batch(
            self,
            batch_size: int,
            reanalyze_ratio: float,
            orig_data: Optional[Tuple[Any]] = None,
            include_ppo: bool = False,
    ) -> Tuple[Any]:
        """
        Overview:
            first sample orig_data through ``_sample_orig_data()``,
            then prepare the context of a batch:
                reward_value_context:        the context of reanalyzed value targets
                policy_re_context:           the context of reanalyzed policy targets
                policy_non_re_context:       the context of non-reanalyzed policy targets
                current_batch:                the inputs of batch
        Arguments:
            - batch_size (:obj:`int`): the batch size of orig_data from replay buffer.
            - reanalyze_ratio (:obj:`float`): ratio of reanalyzed policy (value is 100% reanalyzed)
        Returns:
            - context (:obj:`Tuple`): reward_value_context, policy_re_context, policy_non_re_context, current_batch
        """
        # obtain the batch context from replay buffer
        if orig_data is None:
            if self.sample_type == 'transition':
                orig_data = self._sample_orig_data(batch_size)
            elif self.sample_type == 'episode':
                orig_data = self._sample_orig_data_episode(batch_size)
        game_segment_list, pos_in_game_segment_list, batch_index_list, weights_list, make_time_list = orig_data
        batch_size = len(batch_index_list)
        obs_list, action_list, mask_list = [], [], []
        timestep_list = []
        bootstrap_action_list = []

        # prepare the inputs of a batch
        for i in range(batch_size):
            game = game_segment_list[i]
            pos_in_game_segment = pos_in_game_segment_list[i]

            actions_tmp = game.action_segment[pos_in_game_segment:pos_in_game_segment +
                                                                  self._cfg.num_unroll_steps].tolist()
            timestep_tmp = game.timestep_segment[pos_in_game_segment:pos_in_game_segment +
                                                                  self._cfg.num_unroll_steps].tolist()

            # TODO: the child_visits after position <self._cfg.game_segment_length> in the segment (with padded part) may not be updated
            # So the corresponding position should not be used in the training
            mask_tmp = [1. for i in range(min(len(actions_tmp), self._cfg.game_segment_length - pos_in_game_segment))]
            mask_tmp += [0. for _ in range(self._cfg.num_unroll_steps + 1 - len(mask_tmp))]

            # pad random action
            actions_tmp += [
                np.random.randint(0, game.action_space_size)
                for _ in range(self._cfg.num_unroll_steps - len(actions_tmp))
            ]
            # TODO: check the effect
            timestep_tmp += [
                0
                for _ in range(self._cfg.num_unroll_steps - len(timestep_tmp))
            ]

            # obtain the current observations sequence
            obs_list.append(
                game_segment_list[i].get_unroll_obs(
                    pos_in_game_segment_list[i], num_unroll_steps=self._cfg.num_unroll_steps, padding=True
                )
            )
            action_list.append(actions_tmp)

            mask_list.append(mask_tmp)
            timestep_list.append(timestep_tmp)

            # NOTE: for unizero
            bootstrap_action_tmp = game.action_segment[pos_in_game_segment+self._cfg.td_steps:pos_in_game_segment +
                                                                  self._cfg.num_unroll_steps+self._cfg.td_steps].tolist()
            # pad random action
            bootstrap_action_tmp += [
                np.random.randint(0, game.action_space_size)
                for _ in range(self._cfg.num_unroll_steps - len(bootstrap_action_tmp))
            ]
            bootstrap_action_list.append(bootstrap_action_tmp)


        # formalize the input observations
        obs_list = prepare_observation(obs_list, self._cfg.model.model_type)

        # formalize the inputs of a batch
        current_batch = [obs_list, action_list, bootstrap_action_list, mask_list, batch_index_list, weights_list, make_time_list, timestep_list]
        if include_ppo:
            advantage_list, behavior_log_prob_list = [], []
            return_list, behavior_action_mask_list = [], []
            behavior_policy_feature_list, policy_version_list = [], []
            horizon = int(self._cfg.num_unroll_steps)
            for game, position in zip(game_segment_list, pos_in_game_segment_list):
                advantages = np.asarray(
                    game.advantage_segment[position:position + horizon], dtype=np.float32
                ).tolist()
                advantages += [0.0] * (horizon - len(advantages))
                behavior_log_probs = np.asarray(
                    game.behavior_log_prob_segment[position:position + horizon], dtype=np.float32
                ).tolist()
                behavior_log_probs += [0.0] * (horizon - len(behavior_log_probs))
                value_returns = np.asarray(
                    game.return_segment[position:position + horizon + 1], dtype=np.float32
                ).tolist()
                value_returns += [0.0] * (horizon + 1 - len(value_returns))
                action_masks = np.asarray(
                    game.behavior_action_mask_segment[position:position + horizon], dtype=np.bool_
                ).tolist()
                action_masks += [[True] * self.action_space_size for _ in range(horizon - len(action_masks))]
                policy_features = np.asarray(
                    game.behavior_policy_feature_segment[position:position + horizon], dtype=np.float32
                ).tolist()
                feature_dim = int(game.behavior_policy_feature_segment.shape[-1])
                policy_features += [[0.0] * feature_dim for _ in range(horizon - len(policy_features))]
                advantage_list.append(advantages)
                behavior_log_prob_list.append(behavior_log_probs)
                return_list.append(value_returns)
                behavior_action_mask_list.append(action_masks)
                behavior_policy_feature_list.append(policy_features)
                policy_version_list.append(int(game.collection_train_iter))
            current_batch.extend([
                advantage_list,
                behavior_log_prob_list,
                return_list,
                behavior_action_mask_list,
                behavior_policy_feature_list,
                policy_version_list,
            ])
        for i in range(len(current_batch)):
            current_batch[i] = np.asarray(current_batch[i])

        total_transitions = self.get_num_of_transitions()

        # obtain the context of value targets
        reward_value_context = self._prepare_reward_value_context(
            batch_index_list, game_segment_list, pos_in_game_segment_list, total_transitions
        )
        """
        only reanalyze recent reanalyze_ratio (e.g. 50%) data
        if self._cfg.reanalyze_outdated is True, batch_index_list is sorted according to its generated env_steps
        0: reanalyze_num -> reanalyzed policy, reanalyze_num:end -> non reanalyzed policy
        """
        reanalyze_num = max(int(batch_size * reanalyze_ratio), 1) if reanalyze_ratio > 0 else 0
        # print(f'reanalyze_ratio: {reanalyze_ratio}, reanalyze_num: {reanalyze_num}')
        self.reanalyze_num = reanalyze_num
        # reanalyzed policy
        if reanalyze_num > 0:
            # obtain the context of reanalyzed policy targets
            policy_re_context = self._prepare_policy_reanalyzed_context(
                batch_index_list[:reanalyze_num], game_segment_list[:reanalyze_num],
                pos_in_game_segment_list[:reanalyze_num]
            )
        else:
            policy_re_context = None

        # non reanalyzed policy
        if reanalyze_num < batch_size:
            # obtain the context of non-reanalyzed policy targets
            policy_non_re_context = self._prepare_policy_non_reanalyzed_context(
                batch_index_list[reanalyze_num:], game_segment_list[reanalyze_num:],
                pos_in_game_segment_list[reanalyze_num:]
            )
        else:
            policy_non_re_context = None

        context = reward_value_context, policy_re_context, policy_non_re_context, current_batch
        return context

    def reanalyze_buffer(
            self, batch_size: int, policy: Union["MuZeroPolicy", "EfficientZeroPolicy", "SampledEfficientZeroPolicy"]
    ) -> List[Any]:
        """
        Overview:
            sample data from ``GameBuffer`` and prepare the current and target batch for training.
        Arguments:
            - batch_size (:obj:`int`): batch size.
            - policy (:obj:`Union["MuZeroPolicy", "EfficientZeroPolicy", "SampledEfficientZeroPolicy"]`): policy.
        Returns:
            - train_data (:obj:`List`): List of train data, including current_batch and target_batch.
        """
        policy._target_model.to(self._cfg.device)
        policy._target_model.eval()

        # obtain the current_batch and prepare target context
        policy_re_context, current_batch = self._make_batch_for_reanalyze(batch_size)
        # target policy
        self._compute_target_policy_reanalyzed(policy_re_context, policy._target_model, current_batch[1])

    def _make_batch_for_reanalyze(self, batch_size: int) -> Tuple[Any]:
        """
        Overview:
            first sample orig_data through ``_sample_orig_data()``,
            then prepare the context of a batch:
                reward_value_context:        the context of reanalyzed value targets
                policy_re_context:           the context of reanalyzed policy targets
                policy_non_re_context:       the context of non-reanalyzed policy targets
                current_batch:                the inputs of batch
        Arguments:
            - batch_size (:obj:`int`): the batch size of orig_data from replay buffer.
        Returns:
            - context (:obj:`Tuple`): reward_value_context, policy_re_context, policy_non_re_context, current_batch
        """
        # obtain the batch context from replay buffer
        if self.sample_type == 'transition':
            orig_data = self._sample_orig_reanalyze_batch(batch_size)
        # elif self.sample_type == 'episode': # TODO
        #     orig_data = self._sample_orig_data_episode(batch_size)
        game_segment_list, pos_in_game_segment_list, batch_index_list, weights_list, make_time_list = orig_data
        batch_size = len(batch_index_list)
        obs_list, action_list, mask_list = [], [], []
        bootstrap_action_list = []
        timestep_list = []

        # prepare the inputs of a batch
        for i in range(batch_size):
            game = game_segment_list[i]
            pos_in_game_segment = pos_in_game_segment_list[i]

            actions_tmp = game.action_segment[pos_in_game_segment:pos_in_game_segment +
                                                                  self._cfg.num_unroll_steps].tolist()

            # add mask for invalid actions (out of trajectory), 1 for valid, 0 for invalid
            mask_tmp = [1. for i in range(len(actions_tmp))]
            mask_tmp += [0. for _ in range(self._cfg.num_unroll_steps + 1 - len(mask_tmp))]
            timestep_tmp = game.timestep_segment[pos_in_game_segment:pos_in_game_segment +
                                                                  self._cfg.num_unroll_steps].tolist()

            # pad random action
            actions_tmp += [
                np.random.randint(0, game.action_space_size)
                for _ in range(self._cfg.num_unroll_steps - len(actions_tmp))
            ]

            # TODO: check the effect
            timestep_tmp += [
                0
                for _ in range(self._cfg.num_unroll_steps - len(timestep_tmp))
            ]

            # obtain the current observations sequence
            obs_list.append(
                game_segment_list[i].get_unroll_obs(
                    pos_in_game_segment_list[i], num_unroll_steps=self._cfg.num_unroll_steps, padding=True
                )
            )
            action_list.append(actions_tmp)
            mask_list.append(mask_tmp)

            timestep_list.append(timestep_tmp)

            # NOTE: for unizero
            bootstrap_action_tmp = game.action_segment[pos_in_game_segment+self._cfg.td_steps:pos_in_game_segment +
                                                                  self._cfg.num_unroll_steps+self._cfg.td_steps].tolist()
            # pad random action
            bootstrap_action_tmp += [
                np.random.randint(0, game.action_space_size)
                for _ in range(self._cfg.num_unroll_steps - len(bootstrap_action_tmp))
            ]
            bootstrap_action_list.append(bootstrap_action_tmp)

        # formalize the input observations
        obs_list = prepare_observation(obs_list, self._cfg.model.model_type)

        # formalize the inputs of a batch
        current_batch = [obs_list, action_list, bootstrap_action_list, mask_list, batch_index_list, weights_list, make_time_list, timestep_list]
        for i in range(len(current_batch)):
            current_batch[i] = np.asarray(current_batch[i])

        # reanalyzed policy
        # obtain the context of reanalyzed policy targets
        policy_re_context = self._prepare_policy_reanalyzed_context(
            batch_index_list, game_segment_list,
            pos_in_game_segment_list
        )

        context = policy_re_context, current_batch
        self.reanalyze_num = batch_size
        return context

    def _prepare_policy_reanalyzed_context(
            self, batch_index_list: List[str], game_segment_list: List[Any], pos_in_game_segment_list: List[str]
    ) -> List[Any]:
        """
        Overview:
            prepare the context of policies for calculating policy target in reanalyzing part.
        Arguments:
            - batch_index_list (:obj:'list'): start transition index in the replay buffer
            - game_segment_list (:obj:'list'): list of game segments
            - pos_in_game_segment_list (:obj:'list'): position of transition index in one game history
        Returns:
            - policy_re_context (:obj:`list`): policy_obs_list, policy_mask, pos_in_game_segment_list, indices,
              child_visits, game_segment_lens, action_mask_segment, to_play_segment
        """
        zero_obs = game_segment_list[0].zero_obs()
        with torch.no_grad():
            # for policy
            policy_obs_list = []
            policy_mask = []
            # 0 -> Invalid target policy for padding outside of game segments,
            # 1 -> Previous target policy for game segments.
            rewards, child_visits, game_segment_lens = [], [], []
            # for board games
            action_mask_segment, to_play_segment = [], []
            timestep_segment = []
            # A replay root must enter recurrent MCTS with the same short history
            # that online inference would have accumulated.  Keep a small prefix
            # before the sampled unroll so root 0 is not forced to be Markovian.
            history_observation_segment, history_action_segment = [], []
            world_model_cfg = getattr(self._cfg.model, 'world_model_cfg', None)
            context_length = int(getattr(world_model_cfg, 'context_length', 2))
            # This is deliberately a conservative upper bound.  The world model
            # performs the exact token-level truncation after observations have
            # been encoded (an observation can contain more than one token).
            max_history_transitions = (
                max(context_length // 2, 0) if self._use_contextual_reanalysis() else 0
            )
            for game_segment, state_index in zip(game_segment_list, pos_in_game_segment_list):
                game_segment_len = len(game_segment)
                game_segment_lens.append(game_segment_len)
                rewards.append(game_segment.reward_segment)
                # for board games
                action_mask_segment.append(game_segment.action_mask_segment)
                to_play_segment.append(game_segment.to_play_segment)
                timestep_segment.append(game_segment.timestep_segment)
                child_visits.append(game_segment.child_visit_segment)
                if max_history_transitions > 0:
                    history_start = max(0, state_index - max_history_transitions)
                    history_len = state_index - history_start
                    history_game_obs = game_segment.get_unroll_obs(
                        history_start, history_len
                    )
                    history_observation_segment.append([
                        history_game_obs[offset:offset + self._cfg.model.frame_stack_num]
                        for offset in range(history_len)
                    ])
                    history_action_segment.append(
                        game_segment.action_segment[history_start:state_index].tolist()
                    )
                else:
                    history_observation_segment.append([])
                    history_action_segment.append([])
                # prepare the corresponding observations
                game_obs = game_segment.get_unroll_obs(state_index, self._cfg.num_unroll_steps)
                for current_index in range(state_index, state_index + self._cfg.num_unroll_steps + 1):

                    if current_index < game_segment_len:
                        policy_mask.append(1)
                        beg_index = current_index - state_index
                        end_index = beg_index + self._cfg.model.frame_stack_num
                        obs = game_obs[beg_index:end_index]
                    else:
                        policy_mask.append(0)
                        obs = zero_obs
                    policy_obs_list.append(obs)

        policy_re_context = [
            policy_obs_list, policy_mask, pos_in_game_segment_list, batch_index_list, child_visits, game_segment_lens,
            action_mask_segment, to_play_segment, timestep_segment,
            history_observation_segment, history_action_segment,
        ]
        return policy_re_context

    def _compute_target_policy_reanalyzed(self, policy_re_context: List[Any], model: Any, batch_action) -> np.ndarray:
        """
        Overview:
            prepare policy targets from the reanalyzed context of policies
        Arguments:
            - policy_re_context (:obj:`List`): List of policy context to reanalyzed
        Returns:
            - batch_target_policies_re
        """
        if policy_re_context is None:
            return []
        batch_target_policies_re = []

        # for board games
        policy_obs_list, policy_mask, pos_in_game_segment_list, batch_index_list, child_visits, game_segment_lens, action_mask_segment, \
            to_play_segment, timestep_segment, *history_context = policy_re_context  # noqa
        if history_context:
            history_observation_segment, history_action_segment = history_context
        else:
            # Backward-compatible fallback for custom callers and old tests.
            history_observation_segment = [[] for _ in pos_in_game_segment_list]
            history_action_segment = [[] for _ in pos_in_game_segment_list]
        transition_batch_size = len(policy_obs_list)
        game_segment_batch_size = len(pos_in_game_segment_list)

        root_timesteps, sequence_start_timesteps = self._preprocess_reanalyze_timesteps(
            timestep_segment, pos_in_game_segment_list
        )
        if len(root_timesteps) != transition_batch_size:
            raise RuntimeError(
                'UniZero reanalysis timestep expansion must match the number of MCTS roots: '
                f'{len(root_timesteps)} != {transition_batch_size}.'
            )

        to_play, action_mask = self._preprocess_to_play_and_action_mask(
            game_segment_batch_size, to_play_segment, action_mask_segment, pos_in_game_segment_list
        )

        if self._cfg.model.continuous_action_space is True:
            # when the action space of the environment is continuous, action_mask[:] is None.
            action_mask = [
                list(np.ones(self.action_space_size, dtype=np.int8)) for _ in range(transition_batch_size)
            ]
            # NOTE: in continuous action space env: we set all legal_actions as -1
            legal_actions = [
                [-1 for _ in range(self.action_space_size)] for _ in range(transition_batch_size)
            ]
        else:
            legal_actions = [np.nonzero(action_mask[j])[0].tolist() for j in range(transition_batch_size)]

        with _world_model_reanalysis_phase(model.world_model), torch.no_grad():
            policy_obs_list = prepare_observation(policy_obs_list, self._cfg.model.model_type)
            batch_obs = torch.from_numpy(policy_obs_list).to(self._cfg.device)
            reward_pool, policy_logits_pool, latent_state_roots = (
                self._prepare_reanalysis_root_inputs(
                    model, batch_obs, batch_action, sequence_start_timesteps
                )
            )
            root_token_contexts = None
            if self._has_contextual_reanalysis_api(model):
                history_latent_segment = self._encode_reanalysis_history_observations(
                    model, history_observation_segment
                )
                root_token_contexts = model.world_model.build_reanalysis_root_token_contexts(
                    latent_state_roots=latent_state_roots,
                    batch_actions=batch_action[:self.reanalyze_num],
                    roots_per_sequence=self._cfg.num_unroll_steps + 1,
                    history_latent_segment=history_latent_segment,
                    history_action_segment=history_action_segment,
                    task_id=self.task_id,
                )
            noises = [
                np.random.dirichlet([self._cfg.root_dirichlet_alpha] * self.action_space_size
                                    ).astype(np.float32).tolist() for _ in range(transition_batch_size)
            ]
            roots_distributions = self._search_reanalyzed_roots_in_chunks(
                model=model,
                latent_state_roots=latent_state_roots,
                legal_actions=legal_actions,
                noises=noises,
                reward_pool=reward_pool,
                policy_logits_pool=policy_logits_pool,
                to_play=to_play,
                root_timesteps=root_timesteps,
                root_token_contexts=root_token_contexts,
            )

            roots_legal_actions_list = legal_actions
            policy_index = 0
            for state_index, child_visit, game_index in zip(pos_in_game_segment_list, child_visits, batch_index_list):
                target_policies = []
                for current_index in range(state_index, state_index + self._cfg.num_unroll_steps + 1):
                    distributions = roots_distributions[policy_index]
                    if policy_mask[policy_index] == 0:
                        # NOTE: the invalid padding target policy, O is to make sure the corresponding cross_entropy_loss=0
                        target_policies.append([0 for _ in range(self.action_space_size)])
                    else:
                        # NOTE: It is very important to use the latest MCTS visit count distribution.
                        if distributions is None:
                            # if at some obs, the legal_action is None, add the fake target_policy
                            target_policies.append(
                                list(np.ones(self.action_space_size) / self.action_space_size)
                            )
                        else:
                            sum_visits = sum(distributions)
                            child_visit[current_index] = [
                                visit_count / sum_visits for visit_count in distributions
                            ]
                            if self._cfg.env_type == 'not_board_games':
                                # for atari/classic_control/box2d environments that only have one player.
                                sum_visits = sum(distributions)
                                policy = [visit_count / sum_visits for visit_count in distributions]
                                target_policies.append(policy)
                            else:
                                # for board games that have two players and legal_actions is dy
                                policy_tmp = [0 for _ in range(self.action_space_size)]
                                # to make sure target_policies have the same dimension
                                sum_visits = sum(distributions)
                                policy = [visit_count / sum_visits for visit_count in distributions]
                                for index, legal_action in enumerate(roots_legal_actions_list[policy_index]):
                                    policy_tmp[legal_action] = policy[index]
                                target_policies.append(policy_tmp)

                    policy_index += 1

                batch_target_policies_re.append(target_policies)

        batch_target_policies_re = np.array(batch_target_policies_re)

        return batch_target_policies_re

    def _search_reanalyzed_roots_in_chunks(
            self, model, latent_state_roots, legal_actions, noises, reward_pool,
            policy_logits_pool, to_play, root_timesteps, root_token_contexts=None
    ):
        """Run replay reanalysis within the online recurrent-KV pool capacity.

        A periodic refresh commonly contains ``reanalyze_batch_size * (H+1)`` roots (1760 for
        Atari), while the recurrent cache pool is intentionally sized for one online search
        (``env_num * num_simulations``, normally 400 entries). Searching all replay roots together
        wraps that ring several times within a single simulation and overwrites live tree caches.
        Chunking roots to the online env batch preserves the pool invariant and output order.
        """
        if not self._cfg.mcts_ctree:
            raise NotImplementedError('UniZero replay reanalysis only supports mcts_ctree=True.')

        root_count = len(legal_actions)
        if not (
            len(noises) == len(reward_pool) == len(policy_logits_pool)
            == len(to_play) == len(root_timesteps) == root_count
        ):
            raise ValueError('All replay-reanalysis root inputs must have the same length.')
        if root_token_contexts is not None and len(root_token_contexts) != root_count:
            raise ValueError(
                'Replay-reanalysis root token contexts must align with MCTS roots: '
                f'{len(root_token_contexts)} != {root_count}.'
            )

        all_distributions = []
        seed_count_before = getattr(model.world_model, 'reanalysis_root_seed_count', 0)
        seed_hit_before = getattr(model.world_model, 'reanalysis_root_seed_hit_count', 0)
        for start, end in self._iter_reanalyzed_root_slices(model, root_count):
            chunk_policy_logits = policy_logits_pool[start:end]
            if root_token_contexts is not None:
                contextual_policy_logits = model.world_model.seed_reanalysis_root_caches(
                    latent_state_roots[start:end], root_token_contexts[start:end],
                    **({} if self.task_id is None else {'task_id': self.task_id}),
                )
                if contextual_policy_logits is not None:
                    chunk_policy_logits = contextual_policy_logits
            roots = MCTSCtree.roots(end - start, legal_actions[start:end])
            roots.prepare(
                self._cfg.root_noise_weight,
                noises[start:end],
                reward_pool[start:end],
                chunk_policy_logits,
                to_play[start:end],
            )
            if self.task_id is not None:
                MCTSCtree(self._cfg).search(
                    roots,
                    model,
                    latent_state_roots[start:end],
                    to_play[start:end],
                    task_id=self.task_id,
                )
            else:
                MCTSCtree(self._cfg).search(
                    roots,
                    model,
                    latent_state_roots[start:end],
                    to_play[start:end],
                    root_timesteps[start:end],
                )
            all_distributions.extend(roots.get_distributions())

        if root_token_contexts is not None and hasattr(model.world_model, 'reanalysis_root_seed_count'):
            context_lengths = [int(context.size(0)) for context in root_token_contexts]
            seeded = model.world_model.reanalysis_root_seed_count - seed_count_before
            seed_hits = model.world_model.reanalysis_root_seed_hit_count - seed_hit_before
            logging.info(
                'UniZero reanalysis root-context diagnostics: roots=%d, seeded=%d, first_lookup_hits=%d, '
                'context_tokens[min/mean/max]=%d/%.2f/%d',
                root_count,
                seeded,
                seed_hits,
                min(context_lengths),
                float(np.mean(context_lengths)),
                max(context_lengths),
            )

        return all_distributions

    def _encode_reanalysis_history_observations(self, model, history_observation_segment):
        """Encode only the replay observations preceding each sampled unroll."""
        lengths = [len(sequence) for sequence in history_observation_segment]
        flat_observations = [obs for sequence in history_observation_segment for obs in sequence]
        if not flat_observations:
            return [[] for _ in history_observation_segment]

        prepared = prepare_observation(flat_observations, self._cfg.model.model_type)
        observation_batch = torch.from_numpy(prepared).to(self._cfg.device)
        if self.task_id is None:
            encoded = model.world_model.tokenizer.encode_to_obs_embeddings(observation_batch)
        else:
            encoded = model.world_model.tokenizer.encode_to_obs_embeddings(
                observation_batch, task_id=self.task_id
            )

        split_history, offset = [], 0
        for length in lengths:
            split_history.append([
                encoded[index].detach() for index in range(offset, offset + length)
            ])
            offset += length
        return split_history

    def _iter_reanalyzed_root_slices(self, model, root_count):
        """Yield isolated root slices that fit the online recurrent-cache capacity."""
        world_model_cfg = getattr(self._cfg.model, 'world_model_cfg', None)
        online_capacity = getattr(model.world_model, 'env_num', None)
        if online_capacity is None:
            online_capacity = getattr(world_model_cfg, 'env_num', None)
        # Older/custom configs may not expose the world-model online batch size. A single-root
        # fallback is conservative (slower, but it cannot overrun the recurrent cache pool).
        default_chunk_size = int(online_capacity or 1)
        chunk_size = int(getattr(self._cfg, 'reanalyze_search_chunk_size', default_chunk_size))
        if chunk_size <= 0:
            raise ValueError(f'reanalyze_search_chunk_size must be positive, got {chunk_size}')
        if online_capacity is not None and chunk_size > int(online_capacity):
            raise ValueError(
                'reanalyze_search_chunk_size cannot exceed the online world-model env capacity: '
                f'{chunk_size} > {int(online_capacity)}'
            )

        for start in range(0, root_count, chunk_size):
            end = min(start + chunk_size, root_count)
            # Recurrent entries belong only to this independent group of trees. Root priors and
            # latent states are already materialized above, so clearing scratch state is safe.
            model.world_model.clear_caches()
            yield start, end

    def _preprocess_reanalyze_timesteps(self, timestep_segment, pos_in_game_segment_list):
        """Align episode positions with flattened H+1 reanalysis roots."""
        root_timesteps = []
        sequence_start_timesteps = []
        roots_per_sequence = self._cfg.num_unroll_steps + 1

        for timesteps, state_index in zip(timestep_segment, pos_in_game_segment_list):
            valid = list(timesteps[state_index:state_index + roots_per_sequence])
            valid = [int(value.item() if hasattr(value, 'item') else value) for value in valid]
            sequence_start_timesteps.append(valid[0] if valid else 0)
            root_timesteps.extend(valid)
            root_timesteps.extend([0] * (roots_per_sequence - len(valid)))

        return (
            np.asarray(root_timesteps, dtype=np.int64),
            np.asarray(sequence_start_timesteps, dtype=np.int64),
        )

    def _compute_target_reward_value(self, reward_value_context: List[Any], model: Any, batch_action, batch_timestep) -> Tuple[
        Any, Any]:
        """
        Overview:
            prepare reward and value targets from the context of rewards and values.
        Arguments:
            - reward_value_context (:obj:'list'): the reward value context
            - model (:obj:'torch.tensor'):model of the target model
        Returns:
            - batch_value_prefixs (:obj:'np.ndarray): batch of value prefix
            - batch_target_values (:obj:'np.ndarray): batch of value estimation
        """
        value_obs_list, value_mask, pos_in_game_segment_list, rewards_list, root_values, game_segment_lens, td_steps_list, action_mask_segment, \
            to_play_segment, *history_context = reward_value_context  # noqa
        if history_context:
            history_observation_segment, history_action_segment = history_context
        else:
            history_observation_segment = [[] for _ in pos_in_game_segment_list]
            history_action_segment = [[] for _ in pos_in_game_segment_list]
        # transition_batch_size = game_segment_batch_size * (num_unroll_steps+1)
        transition_batch_size = len(value_obs_list)

        batch_target_values, batch_rewards = [], []
        with torch.no_grad():
            value_obs_list = prepare_observation(value_obs_list, self._cfg.model.model_type)
            batch_obs = torch.from_numpy(value_obs_list).to(self._cfg.device)

            if self._cfg.use_root_value:
                value_numpy = np.array(root_values)
            else:
                use_contextual_bootstrap = getattr(
                    self._cfg, 'bootstrap_value_context', False
                )
                diagnostic_due = (
                    self._bootstrap_value_context_diagnostic_due()
                    if use_contextual_bootstrap else False
                )
                if use_contextual_bootstrap and not diagnostic_due:
                    latent_state_roots = self._encode_bootstrap_root_latents(
                        model, batch_obs
                    )
                    legacy_value_numpy = None
                else:
                    # The legacy comparison needs the full training-sequence Transformer.
                    # Contextual targets skip it on 999/1000 batches and encode roots directly.
                    if self.task_id is not None:
                        m_output = model.initial_inference(
                            batch_obs, batch_action, task_id=self.task_id
                        )
                    else:
                        m_output = model.initial_inference(
                            batch_obs, batch_action, start_pos=batch_timestep
                        )
                    latent_state_roots = m_output.latent_state
                    legacy_value_numpy = to_detach_cpu_numpy([
                        inverse_scalar_transform(m_output.value, self.value_support)
                    ])[0]

                if use_contextual_bootstrap:
                    history_latent_segment = self._encode_reanalysis_history_observations(
                        model, history_observation_segment
                    )
                    root_token_contexts = model.world_model.build_reanalysis_root_token_contexts(
                        latent_state_roots=latent_state_roots,
                        batch_actions=batch_action,
                        roots_per_sequence=self._cfg.num_unroll_steps + 1,
                        history_latent_segment=history_latent_segment,
                        history_action_segment=history_action_segment,
                        task_id=self.task_id,
                    )
                    contextual_value_logits = model.world_model.evaluate_root_token_context_values(
                        root_token_contexts, task_id=self.task_id
                    )
                    value_numpy = to_detach_cpu_numpy([
                        inverse_scalar_transform(contextual_value_logits, self.value_support)
                    ])[0]
                    if diagnostic_due:
                        self._log_bootstrap_value_context_diagnostics(
                            legacy_value_numpy, value_numpy, value_mask, root_token_contexts
                        )
                else:
                    value_numpy = legacy_value_numpy

            # get last state value
            if self._cfg.env_type == 'board_games' and to_play_segment[0][0] in [1, 2]:
                # TODO(pu): for board_games, very important, to check
                value_numpy = value_numpy.reshape(-1) * np.array(
                    [
                        self._cfg.discount_factor ** td_steps_list[i] if int(td_steps_list[i]) %
                                                                         2 == 0 else -self._cfg.discount_factor **
                                                                                      td_steps_list[i]
                        for i in range(transition_batch_size)
                    ]
                )
            else:
                value_numpy = value_numpy.reshape(-1) * (
                        np.array([self._cfg.discount_factor for _ in range(transition_batch_size)]) ** td_steps_list
                )

            value_numpy= value_numpy * np.array(value_mask)
            value_list = value_numpy.tolist()
            horizon_id, value_index = 0, 0

            for game_segment_len_non_re, reward_list, state_index, to_play_list in zip(game_segment_lens, rewards_list,
                                                                                       pos_in_game_segment_list,
                                                                                       to_play_segment):
                target_values = []
                target_rewards = []
                base_index = state_index

                # =========== NOTE ===============
                # if game_segment_len_non_re < self._cfg.game_segment_length:
                #     # The last segment of one episode, the target value of excess part should be 0
                #     truncation_length = game_segment_len_non_re
                # else:
                #     # game_segment_len is game_segment.action_segment.shape[0]
                #     # action_segment.shape[0] = reward_segment.shape[0] or action_segment.shape[0] = reward_segment.shape[0] + 1
                #     truncation_length = game_segment_len_non_re
                #     assert reward_list.shape[0] + 1 == game_segment_len_non_re or reward_list.shape[0] == game_segment_len_non_re

                truncation_length = game_segment_len_non_re

                for current_index in range(state_index, state_index + self._cfg.num_unroll_steps + 1):
                    bootstrap_index = current_index + td_steps_list[value_index]
                    for i, reward in enumerate(reward_list[current_index:bootstrap_index]):
                        if self._cfg.env_type == 'board_games' and to_play_segment[0][0] in [1, 2]:
                            # TODO(pu): for board_games, very important, to check
                            if to_play_list[base_index] == to_play_list[i]:
                                value_list[value_index] += reward * self._cfg.discount_factor ** i
                            else:
                                value_list[value_index] += -reward * self._cfg.discount_factor ** i
                        else:
                            value_list[value_index] += reward * self._cfg.discount_factor ** i
                    horizon_id += 1

                    # TODO: check the boundary condition
                    target_values.append(value_list[value_index])
                    if current_index < len(reward_list):
                        target_rewards.append(reward_list[current_index])
                    else:
                        target_rewards.append(np.array(0.))

                    value_index += 1

                batch_rewards.append(target_rewards)
                batch_target_values.append(target_values)

        batch_rewards = np.asarray(batch_rewards)
        batch_target_values = np.asarray(batch_target_values)

        return batch_rewards, batch_target_values
    
    def update_priority(self, train_data: List[np.ndarray], batch_priorities: np.ndarray) -> None:
        """
        Overview:
            Update the priority of training data.
        Arguments:
            - train_data (:obj:`List[np.ndarray]`): training data to be updated priority.
            - batch_priorities (:obj:`np.ndarray`): priorities to update to.
        NOTE:
            train_data = [current_batch, target_batch]
            current_batch = [obs_list, action_list, bootstrap_action_list, mask_list, batch_index_list, weights_list, make_time_list, timestep_list]
        """
        current_batch = train_data[0]
        indices = current_batch[4]
        make_times = current_batch[6]
        metas = {'make_time': make_times, 'batch_priorities': batch_priorities}
        # only update the priorities for data still in replay buffer
        for i in range(len(indices)):

            make_time = np.asarray(metas['make_time'][i]).reshape(-1)
            first_transition_time = float(make_time[0]) if make_time.size > 0 else 0.0

            if first_transition_time > self.clear_time:
                # Handle IndexError by converting the float index to an integer before use.
                idx = int(indices[i])
                prio = metas['batch_priorities'][i]

                # Now, idx is a valid integer index.
                self.game_pos_priorities[idx] = prio

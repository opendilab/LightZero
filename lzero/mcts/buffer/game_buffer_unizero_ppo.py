"""PPO-only replay extensions for UniZero."""

from .game_buffer_unizero import *  # Reuse UniZero replay/reanalysis behavior unchanged.


class UniZeroPPOGameBuffer(UniZeroGameBuffer):
    """Fresh-rollout buffer with PPO targets and fast actor/critic batches."""

    def get_on_policy_indices(self, collection_train_iter: int) -> np.ndarray:
        """Return flat replay indices produced by exactly one collection policy version."""
        indices = []
        horizon = int(self._cfg.num_unroll_steps)
        matching_segment_indices = {
            self.base_idx + segment_index
            for segment_index, segment in enumerate(self.game_segment_buffer)
            if getattr(segment, 'collection_train_iter', None) == int(collection_train_iter)
        }
        for flat_index, (global_segment_index, position) in enumerate(self.game_segment_game_pos_look_up):
            if position % horizon == 0 and global_segment_index in matching_segment_indices:
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
            len(indices), reanalyze_ratio=0.0, orig_data=orig_data, include_ppo=True,
            prepare_observations=False, prepare_target_context=False,
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
        if self.sample_type == 'transition':
            orig_data = self._sample_orig_data(batch_size)
        elif self.sample_type == 'episode':
            orig_data = self._sample_orig_data_episode(batch_size)
        else:
            raise ValueError(f'Unsupported sample_type: {self.sample_type!r}')
        _, _, _, current_batch = self._make_batch(
            batch_size, reanalyze_ratio=0.0, orig_data=orig_data,
            prepare_target_context=False,
        )
        horizon = int(self._cfg.num_unroll_steps)
        rewards = []
        for game_segment, position in zip(orig_data[0], orig_data[1]):
            values = np.asarray(
                game_segment.reward_segment[position:position + horizon + 1], dtype=np.float32
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
            prepare_observations: bool = True,
            prepare_target_context: bool = True,
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
            if prepare_observations:
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
        if prepare_observations:
            obs_list = prepare_observation(obs_list, self._cfg.model.model_type)
        else:
            # Actor/critic PPO updates consume the exact contextual features
            # captured during collection. Materializing Atari image sequences
            # here would only copy data that compute_ppo_loss never reads.
            obs_list = np.empty((batch_size, 0), dtype=np.float32)

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

        if not prepare_target_context:
            return None, None, None, current_batch

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

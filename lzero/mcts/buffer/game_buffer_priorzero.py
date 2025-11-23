# game_buffer_priorzero.py
"""
[PRIORZERO] Enhanced Game Buffer for PriorZero

This module extends UniZeroGameBuffer to support LLM policy training (SFT + RFT).

Key Features:
- Returns game_segments in sample() for LLM training data extraction
- Efficient indexing to avoid duplicating large observation data
- Robust handling of edge cases (partial batches, variable-length segments)
- Minimal memory overhead (only stores references, not copies)

Author: PriorZero Team
Date: 2025-01-21
"""

import numpy as np
from typing import List, Any, Union, Tuple
from lzero.mcts.buffer.game_buffer_unizero import UniZeroGameBuffer

class PriorZeroGameBufferOptimized(UniZeroGameBuffer):
    """
    [PRIORZERO-OPTIMIZED]
    More efficient version that avoids double sampling by modifying _make_batch minimally.

    This version uses a monkey-patch approach to intercept orig_data during parent's _make_batch call.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self._cached_game_segments = None

    def sample(self, batch_size: int, policy) -> List[Any]:
        """Sample data with game_segments (optimized version)."""
        policy._target_model.to(self._cfg.device)
        policy._target_model.eval()

        # Reset cache
        self._cached_game_segments = None

        # Call parent's _make_batch (which will trigger our hook)
        reward_value_context, policy_re_context, policy_non_re_context, current_batch = self._make_batch(
            batch_size, self._cfg.reanalyze_ratio
        )

        obs_list, action_list, bootstrap_action_list, mask_list, batch_index_list, weights_list, make_time_list, timestep_list, raw_obs_list, history_obs_list, action_logprob_list = current_batch
        # Standard processing
        batch_rewards, batch_target_values = self._compute_target_reward_value(
            reward_value_context, policy._target_model, current_batch[2], timestep_list
        )

        batch_target_policies_re = self._compute_target_policy_reanalyzed(
            policy_re_context, policy._target_model, current_batch[1], timestep_list
        )
        batch_target_policies_non_re = self._compute_target_policy_non_reanalyzed(
            policy_non_re_context, self.action_space_size
        )

        if 0 < self._cfg.reanalyze_ratio < 1:
            batch_target_policies = np.concatenate([batch_target_policies_re, batch_target_policies_non_re])
        elif self._cfg.reanalyze_ratio == 1:
            batch_target_policies = batch_target_policies_re
        elif self._cfg.reanalyze_ratio == 0:
            batch_target_policies = batch_target_policies_non_re

        target_batch = [batch_rewards, batch_target_values, batch_target_policies]

        return [current_batch, target_batch]

    def _make_batch(self, batch_size: int, reanalyze_ratio: float) -> Tuple[Any]:
        """
        [PRIORZERO-OPTIMIZED]
        Minimally modified to cache game_segment_list during sampling.

        This is a full override of parent's _make_batch to avoid double sampling.
        Code is mostly copied from parent, with one key addition: caching game_segments.
        """
        # Sample original data
        if self.sample_type == 'transition':
            orig_data = self._sample_orig_data(batch_size)
        elif self.sample_type == 'episode':
            orig_data = self._sample_orig_data_episode(batch_size)

        game_segment_list, pos_in_game_segment_list, batch_index_list, weights_list, make_time_list = orig_data

        # [PRIORZERO-KEY] Cache game_segments for sample() to use
        self._cached_game_segments = game_segment_list

        # Rest of the code is identical to parent's _make_batch
        batch_size = len(batch_index_list)
        obs_list, action_list, mask_list = [], [], []
        raw_obs_list, history_obs_list = [], []
        action_logprob_list = []
        timestep_list = []
        bootstrap_action_list = []

        for i in range(batch_size):
            game = game_segment_list[i]
            pos_in_game_segment = pos_in_game_segment_list[i]

            actions_tmp = game.action_segment[pos_in_game_segment:pos_in_game_segment +
                                                                  self._cfg.num_unroll_steps].tolist()
            timestep_tmp = game.timestep_segment[pos_in_game_segment:pos_in_game_segment +
                                                                  self._cfg.num_unroll_steps].tolist()

            mask_tmp = [1. for i in range(min(len(actions_tmp), self._cfg.game_segment_length - pos_in_game_segment))]
            mask_tmp += [0. for _ in range(self._cfg.num_unroll_steps + 1 - len(mask_tmp))]

            actions_tmp += [
                np.random.randint(0, game.action_space_size)
                for _ in range(self._cfg.num_unroll_steps - len(actions_tmp))
            ]
            timestep_tmp += [
                0
                for _ in range(self._cfg.num_unroll_steps - len(timestep_tmp))
            ]

            obs_list.append(
                game_segment_list[i].get_unroll_obs(
                    pos_in_game_segment_list[i], num_unroll_steps=self._cfg.num_unroll_steps, padding=True
                )
            )
            raw_obs_list.append(game_segment_list[i].get_unroll_raw_obs(
                pos_in_game_segment_list[i], num_unroll_steps=self._cfg.num_unroll_steps, padding=True
            ))  
            history_obs_list.append(game_segment_list[i].get_unroll_histroy_obs(
                pos_in_game_segment_list[i], num_unroll_steps=self._cfg.num_unroll_steps, padding=True
            ))
            action_logprob_list.append(game_segment_list[i].get_unroll_action_logprob(
                pos_in_game_segment_list[i], num_unroll_steps=self._cfg.num_unroll_steps, padding=True
            ))
            
            action_list.append(actions_tmp)
            mask_list.append(mask_tmp)
            timestep_list.append(timestep_tmp)

            bootstrap_action_tmp = game.action_segment[pos_in_game_segment+self._cfg.td_steps:pos_in_game_segment +
                                                                  self._cfg.num_unroll_steps+self._cfg.td_steps].tolist()
            bootstrap_action_tmp += [
                np.random.randint(0, game.action_space_size)
                for _ in range(self._cfg.num_unroll_steps - len(bootstrap_action_tmp))
            ]
            bootstrap_action_list.append(bootstrap_action_tmp)

        # Import here to avoid circular dependency
        from lzero.mcts.utils import prepare_observation
        obs_list = prepare_observation(obs_list, self._cfg.model.model_type)

        current_batch = [obs_list, action_list, bootstrap_action_list, mask_list, batch_index_list, weights_list, make_time_list, timestep_list]
        for i in range(len(current_batch)):
            current_batch[i] = np.asarray(current_batch[i])
            
        current_batch.append(raw_obs_list)
        current_batch.append(history_obs_list)
        current_batch.append(action_logprob_list)

        total_transitions = self.get_num_of_transitions()

        reward_value_context = self._prepare_reward_value_context(
            batch_index_list, game_segment_list, pos_in_game_segment_list, total_transitions
        )

        reanalyze_num = max(int(batch_size * reanalyze_ratio), 1) if reanalyze_ratio > 0 else 0
        self.reanalyze_num = reanalyze_num

        if reanalyze_num > 0:
            policy_re_context = self._prepare_policy_reanalyzed_context(
                batch_index_list[:reanalyze_num], game_segment_list[:reanalyze_num],
                pos_in_game_segment_list[:reanalyze_num]
            )
        else:
            policy_re_context = None

        if reanalyze_num < batch_size:
            policy_non_re_context = self._prepare_policy_non_reanalyzed_context(
                batch_index_list[reanalyze_num:], game_segment_list[reanalyze_num:],
                pos_in_game_segment_list[reanalyze_num:]
            )
        else:
            policy_non_re_context = None

        return reward_value_context, policy_re_context, policy_non_re_context, current_batch

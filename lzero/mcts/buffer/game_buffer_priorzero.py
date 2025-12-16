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
        self.last_pos_in_transition = 0
    
    def fetch_latest_batch(self, batch_size: int, policy) -> List[Any]:
        policy._target_model.to(self._cfg.device)
        policy._target_model.eval()
        
        reward_value_context, policy_re_context, policy_non_re_context, current_batch = self._make_batch(
            batch_size, self._cfg.reanalyze_ratio, fetch_latest=True
        )

        obs_list, action_list, bootstrap_action_list, mask_list, batch_index_list, weights_list, make_time_list, timestep_list, raw_obs_list, history_obs_list, action_logprob_list = current_batch
        # Standard processing
        batch_rewards, batch_target_values = self._compute_target_reward_value(
            reward_value_context, policy._target_model, current_batch[2], timestep_list
        )

        batch_target_policies = self._compute_target_policy_non_reanalyzed(
            policy_non_re_context, self.action_space_size
        )

        target_batch = [batch_rewards, batch_target_values, batch_target_policies]

        return [current_batch, target_batch]
    
    def sample(self, batch_size: int, policy) -> List[Any]:
        """Sample data with game_segments (optimized version)."""
        policy._target_model.to(self._cfg.device)
        policy._target_model.eval()

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

    def _make_batch(self, batch_size: int, reanalyze_ratio: float, fetch_latest: bool = False) -> Tuple[Any]:
        """
        [PRIORZERO-OPTIMIZED]
        Minimally modified to cache game_segment_list during sampling.

        This is a full override of parent's _make_batch to avoid double sampling.
        Code is mostly copied from parent, with one key addition: caching game_segments.
        """
        # Sample original data
        if not fetch_latest:
            if self.sample_type == 'transition':
                orig_data = self._sample_orig_data(batch_size)
            elif self.sample_type == 'episode':
                orig_data = self._sample_orig_data_episode(batch_size)
        else:
            if self.sample_type == 'transition':
                orig_data = self._fetch_latest_orig_data(batch_size)
            elif self.sample_type == 'episode':
                raise ValueError("fetch_latest with episode sampling not supported.")

        game_segment_list, pos_in_game_segment_list, batch_index_list, weights_list, make_time_list = orig_data

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

    def _clear(self):
        self.game_pos_priorities = []
        self.game_segment_buffer = []
        self.game_segment_game_pos_look_up = []
    
    
    def _fetch_latest_orig_data(self, batch_size: int) -> Tuple:
        """
        Overview:
            Sample original data which includes:
                - game_segment_list: A list of game segments.
                - pos_in_game_segment_list: Transition index in the game (relative index).
                - batch_index_list: The index of the start transition of the sampled mini-batch in the replay buffer.
                - weights_list: The weight concerning the priority.
                - make_time: The time the batch is made (for correctly updating the replay buffer when data is deleted).
        Arguments:
            - batch_size (:obj:`int`): The size of the batch.
            - print_priority_logs (:obj:`bool`): Whether to print logs related to priority statistics, defaults to False.
        """
        assert self._beta > 0, "Beta should be greater than 0"
        num_of_transitions = self.get_num_of_transitions()

        probs = self.game_pos_priorities ** self._alpha + 1e-6
        probs /= probs.sum()

        # 主要改动： 由sample改成了确定的取最后batch_size个样本
        if batch_size == -1:
            batch_index_list = list(range(num_of_transitions))[self.last_pos_in_transition:]
            self.last_pos_in_transition = num_of_transitions
        else:
            batch_index_list = list(range(num_of_transitions))[-batch_size:]
            
        if self._cfg.reanalyze_outdated:
            batch_index_list.sort()
        
        weights_list = (num_of_transitions * probs[batch_index_list]) ** (-self._beta)
        weights_list /= weights_list.max()  # Normalize weights

        game_segment_list = []
        pos_in_game_segment_list = []

        for idx in batch_index_list:
            game_segment_idx, pos_in_game_segment = self.game_segment_game_pos_look_up[idx]
            game_segment_idx -= self.base_idx  # Adjust index based on base index
            game_segment = self.game_segment_buffer[game_segment_idx]

            game_segment_list.append(game_segment)

            # print(f'len(game_segment)=:len(game_segment.action_segment): {len(game_segment)}')
            # print(f'len(game_segment.obs_segment): {game_segment.obs_segment.shape[0]}')

            # In the reanalysis phase, `pos_in_game_segment` should be a multiple of `num_unroll_steps`.
            # Indices exceeding `game_segment_length` are padded with the next segment and are not updated
            # in the current implementation. Therefore, we need to sample `pos_in_game_segment` within
            # [0, game_segment_length - num_unroll_steps] to avoid padded data.
            
            if self._cfg.action_type == 'varied_action_space':
                # For some environments (e.g., Jericho), the action space size may be different.
                # To ensure we can always unroll `num_unroll_steps` steps starting from the sampled position (without exceeding segment length),
                # we avoid sampling from the last `num_unroll_steps` steps of the game segment. 
                if pos_in_game_segment >= self._cfg.game_segment_length - self._cfg.num_unroll_steps - self._cfg.td_steps:
                    pos_in_game_segment = np.random.choice(self._cfg.game_segment_length - self._cfg.num_unroll_steps - self._cfg.td_steps, 1).item()
                
                segment_len = len(game_segment.action_segment)
                if pos_in_game_segment >= segment_len - 1:
                    # If the segment is very short (length 0 or 1), we can't randomly sample a position
                    # before the last one. The only safe position is 0.
                    if segment_len > 1:
                        # If the segment has at least 2 actions, we can safely sample from [0, len-2].
                        # The upper bound for np.random.choice is exclusive, so (segment_len - 1) is correct.
                        pos_in_game_segment = np.random.choice(segment_len - 1, 1).item()
                    else:
                        # If segment length is 0 or 1, the only valid/safe position is 0.
                        pos_in_game_segment = 0

            else:
                # For environments with a fixed action space (e.g., Atari),
                # we can safely sample from the entire game segment range.
                if pos_in_game_segment >= self._cfg.game_segment_length:
                    pos_in_game_segment = np.random.choice(self._cfg.game_segment_length, 1).item()
                
                segment_len = len(game_segment.action_segment)
                if pos_in_game_segment >= segment_len - 1:
                    # If the segment is very short (length 0 or 1), we can't randomly sample a position
                    # before the last one. The only safe position is 0.
                    if segment_len > 1:
                        # If the segment has at least 2 actions, we can safely sample from [0, len-2].
                        # The upper bound for np.random.choice is exclusive, so (segment_len - 1) is correct.
                        pos_in_game_segment = np.random.choice(segment_len - 1, 1).item()
                    else:
                        # If segment length is 0 or 1, the only valid/safe position is 0.
                        pos_in_game_segment = 0

            pos_in_game_segment_list.append(pos_in_game_segment)
            

        # make_time = [time.time() for _ in range(len(batch_index_list))]

        # Set the make_time for each sample (set to 0 for now, but can be the actual time if needed).
        make_time = [0. for _ in range(len(batch_index_list))]

        orig_data = (game_segment_list, pos_in_game_segment_list, batch_index_list, weights_list, make_time)
            
        return orig_data
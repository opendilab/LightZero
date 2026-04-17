from typing import Any, List, Tuple, Union, TYPE_CHECKING, Optional

import numpy as np
import torch
from ding.utils import BUFFER_REGISTRY

from lzero.mcts.tree_search.mcts_ctree import UniZeroMCTSCtree as MCTSCtree
from lzero.mcts.utils import prepare_observation
from lzero.policy import DiscreteSupport, to_detach_cpu_numpy, concat_output, concat_output_value, inverse_scalar_transform
from .game_buffer_muzero import MuZeroGameBuffer

if TYPE_CHECKING:
    from lzero.policy import MuZeroPolicy, EfficientZeroPolicy, SampledEfficientZeroPolicy


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

        self.value_support = DiscreteSupport(*self._cfg.model.value_support_range)
        self.reward_support = DiscreteSupport(*self._cfg.model.reward_support_range)

    def _sample_original(
            self, batch_size: int, policy: Union["MuZeroPolicy", "EfficientZeroPolicy", "SampledEfficientZeroPolicy"]
    ) -> List[Any]:
        """
        Overview:
            Original sampling logic (before mixed sampling).
        """
        # obtain the current_batch and prepare target context
        reward_value_context, policy_re_context, policy_non_re_context, current_batch = self._make_batch(
            batch_size, self._cfg.reanalyze_ratio
        )

        # Unpack for readability:
        # 0: obs_list, 1: action_list, 2: bootstrap_action_list, 3: mask_list,
        # 4: batch_index_list, 5: weights_list, 6: make_time_list,
        # 7: timestep_list, 8: advantage_list, 9: old_log_prob_list, 10: return_list
        timestep_list = current_batch[7]

        # target reward, target value
        batch_rewards, batch_target_values = self._compute_target_reward_value(
            reward_value_context, policy._target_model, current_batch[2], timestep_list
        )

        # target policy
        batch_target_policies_re = self._compute_target_policy_reanalyzed(
            policy_re_context, policy._target_model, current_batch[1], timestep_list
        )
        batch_target_policies_non_re = self._compute_target_policy_non_reanalyzed(
            policy_non_re_context, self._cfg.model.action_space_size
        )

        # fusion of batch_target_policies_re and batch_target_policies_non_re to batch_target_policies
        if 0 < self._cfg.reanalyze_ratio < 1:
            batch_target_policies = np.concatenate([batch_target_policies_re, batch_target_policies_non_re])
        elif self._cfg.reanalyze_ratio == 1:
            batch_target_policies = batch_target_policies_re
        elif self._cfg.reanalyze_ratio == 0:
            batch_target_policies = batch_target_policies_non_re

        target_batch = [batch_rewards, batch_target_values, batch_target_policies]
        train_data = [current_batch, target_batch]
        return train_data

    def _sample_from_segment_range(self, batch_size: int, policy, start_idx: int, end_idx: int) -> List[Any]:
        """
        Overview:
            Sample from a specific range of segments.
        Arguments:
            - batch_size: Number of samples to draw
            - policy: Policy model
            - start_idx: Start index of segment range (inclusive)
            - end_idx: End index of segment range (exclusive)
        Returns:
            - train_data: [current_batch, target_batch]
        """
        # 临时修改 buffer 以仅从指定范围采样，用完后必须恢复
        original_buffer = self.game_segment_buffer
        original_game_pos_priorities = self.game_pos_priorities
        try:
            self.game_segment_buffer = original_buffer[start_idx:end_idx]
            self.game_pos_priorities = original_game_pos_priorities[start_idx:end_idx]
            train_data = self._sample_original(batch_size, policy)
            return train_data
        finally:
            self.game_segment_buffer = original_buffer
            self.game_pos_priorities = original_game_pos_priorities
     

    def sample(
            self, batch_size: int, policy: Union["MuZeroPolicy", "EfficientZeroPolicy", "SampledEfficientZeroPolicy"]
    ) -> Tuple[List[Any], List[Any]]:
        """
        Overview:
            Sample data from buffer with mixed strategy: part from new data, part from old data.
        Arguments:
            - batch_size (:obj:`int`): batch size.
            - policy (:obj:`Union["MuZeroPolicy", "EfficientZeroPolicy", "SampledEfficientZeroPolicy"]`): policy.
        Returns:
            - (train_data_new, train_data_old): Tuple of train data
              - train_data_new: Data sampled from latest pushed segments
              - train_data_old: Data sampled from older segments
              - Each train_data is [current_batch, target_batch] format
        """
        policy._target_model.to(self._cfg.device)
        policy._target_model.eval()
        
        total_segments = len(self.game_segment_buffer)
        
        # ✅ 检查是否有足够的新旧数据分离
        # if self.latest_push_count > 0 and total_segments > self.latest_push_count:
        #     print("有足够数据进行新旧数据分离")
        #     # 计算新旧数据的 batch_size 分配
        #     new_batch_size = int(batch_size * self.new_data_ratio)
        #     old_batch_size = batch_size - new_batch_size
            
        #     # 新数据：最后 latest_push_count 个 segments
        #     new_start_idx = total_segments - self.latest_push_count
            
        #     # 从新数据采样
        #     train_data_new = self._sample_from_segment_range(
        #         new_batch_size, policy, new_start_idx, total_segments
        #     )
            
        #     # 从老数据采样
        #     train_data_old = self._sample_from_segment_range(
        #         old_batch_size, policy, 0, new_start_idx
        #     )
            
        #     return (train_data_new, train_data_old)
        # else:
        print("没有足够数据进行新旧数据分离")
        # Fallback: 没有足够的新旧数据区分，使用原始方法
        train_data = self._sample_original(batch_size, policy)
        return (train_data, None)


    
    
    def _make_batch(self, batch_size: int, reanalyze_ratio: float) -> Tuple[Any]:
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
        # import pudb;pudb.set_trace()
        if self.sample_type == 'transition':
            orig_data = self._sample_orig_data(batch_size)
        elif self.sample_type == 'episode':
            orig_data = self._sample_orig_data_episode(batch_size)
        game_segment_list, pos_in_game_segment_list, batch_index_list, weights_list, make_time_list = orig_data
        batch_size = len(batch_index_list)
        obs_list, action_list, mask_list = [], [], []
        timestep_list = []
        bootstrap_action_list = []
        advantage_list = []  # PPO: for storing GAE advantages
        old_log_prob_list = []  # PPO: for storing old log probabilities
        return_list = []  # PPO: for storing returns

        # prepare the inputs of a batch
        for i in range(batch_size):
            game = game_segment_list[i]
            pos_in_game_segment = pos_in_game_segment_list[i]

            actions_tmp = game.action_segment[pos_in_game_segment:pos_in_game_segment +
                                                                  self._cfg.num_unroll_steps].tolist()
            timestep_tmp = game.timestep_segment[pos_in_game_segment:pos_in_game_segment +
                                                                  self._cfg.num_unroll_steps].tolist()
            # add mask for invalid actions (out of trajectory), 1 for valid, 0 for invalid
            # mask_tmp = [1. for i in range(len(actions_tmp))]
            # mask_tmp += [0. for _ in range(self._cfg.num_unroll_steps + 1 - len(mask_tmp))]

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

            # import pudb;pudb.set_trace()
            
            # Extract advantages for the sampled positions
            advantage_tmp = game.advantage_segment[pos_in_game_segment:pos_in_game_segment +
                                                                    self._cfg.num_unroll_steps].tolist()
            # Pad with zeros if not enough advantages (shouldn't happen if GAE is computed correctly)
            advantage_tmp += [0.0 for _ in range(self._cfg.num_unroll_steps - len(advantage_tmp))]
         
            advantage_list.append(advantage_tmp)
            
            logits_tmp = game.old_log_prob_segment[pos_in_game_segment:pos_in_game_segment + self._cfg.num_unroll_steps]  # [T, A]
            logits_tmp = logits_tmp.tolist()  # Convert to list
            # Pad if necessary: add zero logits [0, 0, ..., 0] for missing timesteps
            action_space_size = len(logits_tmp[0]) if len(logits_tmp) > 0 else 6
            num_pad = self._cfg.num_unroll_steps - len(logits_tmp)
            if num_pad > 0:
                logits_tmp += [[0.0] * action_space_size for _ in range(num_pad)]
    
            old_log_prob_list.append(logits_tmp)
            
            # PPO: extract return if available
            if hasattr(game, 'return_segment') and len(game.return_segment) > 0:
                return_tmp = game.return_segment[pos_in_game_segment:pos_in_game_segment +
                                                                      self._cfg.num_unroll_steps].tolist()
                return_tmp += [0.0 for _ in range(self._cfg.num_unroll_steps - len(return_tmp))]
            else:
                return_tmp = [0.0 for _ in range(self._cfg.num_unroll_steps)]
            return_list.append(return_tmp)


        # formalize the input observations
        obs_list = prepare_observation(obs_list, self._cfg.model.model_type)

        # formalize the inputs of a batch
        # PPO: added advantage_list (9th), old_log_prob_list (10th), return_list (11th)
        current_batch = [obs_list, action_list, bootstrap_action_list, mask_list, batch_index_list, weights_list, make_time_list, timestep_list, advantage_list, old_log_prob_list, return_list]
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
        # current_batch structure is aligned with _make_batch: index 7 is timestep_list
        self._compute_target_policy_reanalyzed(policy_re_context, policy._target_model, current_batch[1], current_batch[7])

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
        advantage_list = []  # PPO: for storing GAE advantages
        old_log_prob_list = []  # PPO: for storing old log probabilities
        return_list = []  # PPO: for storing returns

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
            # TODO: original buffer mask
            # mask_tmp = [1. for i in range(min(len(actions_tmp), self._cfg.game_segment_length - pos_in_game_segment))]
            # mask_tmp += [0. for _ in range(self._cfg.num_unroll_steps + 1 - len(mask_tmp))]

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

            # PPO: extract GAE advantages if available
            if hasattr(game, 'advantage_segment') and len(game.advantage_segment) > 0:
                advantage_tmp = game.advantage_segment[pos_in_game_segment:pos_in_game_segment +
                                                                      self._cfg.num_unroll_steps].tolist()
                advantage_tmp += [0.0 for _ in range(self._cfg.num_unroll_steps - len(advantage_tmp))]
            else:
                advantage_tmp = [0.0 for _ in range(self._cfg.num_unroll_steps)]
            advantage_list.append(advantage_tmp)
            
            # PPO: extract policy_logits if available (now [T, A] instead of [T])
            if hasattr(game, 'old_log_prob_segment') and len(game.old_log_prob_segment) > 0:
                logits_tmp = game.old_log_prob_segment[pos_in_game_segment:pos_in_game_segment +
                                                                      self._cfg.num_unroll_steps]  # [T, A]
                logits_tmp = logits_tmp.tolist()  # Convert to list
                # Pad if necessary: add zero logits [0, 0, ..., 0] for missing timesteps
                action_space_size = len(logits_tmp[0]) if len(logits_tmp) > 0 else 6
                num_pad = self._cfg.num_unroll_steps - len(logits_tmp)
                if num_pad > 0:
                    logits_tmp += [[0.0] * action_space_size for _ in range(num_pad)]
            else:
                # Default: zero logits for all timesteps
                action_space_size = 6  # Default action space size
                logits_tmp = [[0.0] * action_space_size for _ in range(self._cfg.num_unroll_steps)]
            old_log_prob_list.append(logits_tmp)
            
            # PPO: extract return if available
            if hasattr(game, 'return_segment') and len(game.return_segment) > 0:
                return_tmp = game.return_segment[pos_in_game_segment:pos_in_game_segment +
                                                                      self._cfg.num_unroll_steps].tolist()
                return_tmp += [0.0 for _ in range(self._cfg.num_unroll_steps - len(return_tmp))]
            else:
                return_tmp = [0.0 for _ in range(self._cfg.num_unroll_steps)]
            return_list.append(return_tmp)

        # formalize the input observations
        obs_list = prepare_observation(obs_list, self._cfg.model.model_type)

        # formalize the inputs of a batch
        # PPO: added advantage_list (9th), old_log_prob_list (10th), return_list (11th)
        current_batch = [obs_list, action_list, bootstrap_action_list, mask_list, batch_index_list, weights_list, make_time_list, timestep_list, advantage_list, old_log_prob_list, return_list]
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
            for game_segment, state_index in zip(game_segment_list, pos_in_game_segment_list):
                game_segment_len = len(game_segment)
                game_segment_lens.append(game_segment_len)
                rewards.append(game_segment.reward_segment)
                # for board games
                action_mask_segment.append(game_segment.action_mask_segment)
                to_play_segment.append(game_segment.to_play_segment)
                timestep_segment.append(game_segment.timestep_segment)
                child_visits.append(game_segment.child_visit_segment)
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
            action_mask_segment, to_play_segment, timestep_segment
        ]
        return policy_re_context

    def _compute_target_policy_reanalyzed(self, policy_re_context: List[Any], model: Any, batch_action, batch_timestep = None) -> np.ndarray:
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
            to_play_segment, timestep_segment = policy_re_context  # noqa
        transition_batch_size = len(policy_obs_list)
        game_segment_batch_size = len(pos_in_game_segment_list)

        # TODO: timestep_segment
        to_play, action_mask = self._preprocess_to_play_and_action_mask(
            game_segment_batch_size, to_play_segment, action_mask_segment, pos_in_game_segment_list
        )

        if self._cfg.model.continuous_action_space is True:
            # when the action space of the environment is continuous, action_mask[:] is None.
            action_mask = [
                list(np.ones(self._cfg.model.action_space_size, dtype=np.int8)) for _ in range(transition_batch_size)
            ]
            # NOTE: in continuous action space env: we set all legal_actions as -1
            legal_actions = [
                [-1 for _ in range(self._cfg.model.action_space_size)] for _ in range(transition_batch_size)
            ]
        else:
            legal_actions = [[i for i, x in enumerate(action_mask[j]) if x == 1] for j in range(transition_batch_size)]

        # NOTE: check the effect of reanalyze_phase
        model.world_model.reanalyze_phase = True

        with torch.no_grad():
            policy_obs_list = prepare_observation(policy_obs_list, self._cfg.model.model_type)
            network_output = []
            batch_obs = torch.from_numpy(policy_obs_list).to(self._cfg.device)

            # =============== NOTE: The key difference with MuZero =================
            # To obtain the target policy from MCTS guided by the recent target model
            # TODO: batch_obs (policy_obs_list) is at timestep t, batch_action is at timestep t
            m_output = model.initial_inference(batch_obs, batch_action[:self.reanalyze_num], start_pos=batch_timestep[:self.reanalyze_num])  # NOTE: :self.reanalyze_num
            # =======================================================================

            if not model.training:
                # if not in training, obtain the scalars of the value/reward
                [m_output.latent_state, m_output.value, m_output.policy_logits] = to_detach_cpu_numpy(
                    [
                        m_output.latent_state,
                        inverse_scalar_transform(m_output.value, self.value_support),
                        m_output.policy_logits
                    ]
                )

            network_output.append(m_output)

            _, reward_pool, policy_logits_pool, latent_state_roots = concat_output(network_output, data_type='muzero')
            reward_pool = reward_pool.squeeze().tolist()
            policy_logits_pool = policy_logits_pool.tolist()
            noises = [
                np.random.dirichlet([self._cfg.root_dirichlet_alpha] * self._cfg.model.action_space_size
                                    ).astype(np.float32).tolist() for _ in range(transition_batch_size)
            ]
            if self._cfg.mcts_ctree:
                # cpp mcts_tree
                roots = MCTSCtree.roots(transition_batch_size, legal_actions)
                roots.prepare(self._cfg.root_noise_weight, noises, reward_pool, policy_logits_pool, to_play)
                # do MCTS for a new policy with the recent target model
                MCTSCtree(self._cfg).search(roots, model, latent_state_roots, to_play, batch_timestep[:self.reanalyze_num])
            else:
                # python mcts_tree
                roots = MCTSPtree.roots(transition_batch_size, legal_actions)
                roots.prepare(self._cfg.root_noise_weight, noises, reward_pool, policy_logits_pool, to_play)
                # do MCTS for a new policy with the recent target model
                MCTSPtree(self._cfg).search(roots, model, latent_state_roots, to_play, batch_timestep[:self.reanalyze_num])

            roots_legal_actions_list = legal_actions
            roots_distributions = roots.get_distributions()
            policy_index = 0
            for state_index, child_visit, game_index in zip(pos_in_game_segment_list, child_visits, batch_index_list):
                target_policies = []
                for current_index in range(state_index, state_index + self._cfg.num_unroll_steps + 1):
                    distributions = roots_distributions[policy_index]
                    if policy_mask[policy_index] == 0:
                        # NOTE: the invalid padding target policy, O is to make sure the corresponding cross_entropy_loss=0
                        target_policies.append([0 for _ in range(self._cfg.model.action_space_size)])
                    else:
                        # NOTE: It is very important to use the latest MCTS visit count distribution.
                        sum_visits = sum(distributions)
                        child_visit[current_index] = [visit_count / sum_visits for visit_count in distributions]

                        if distributions is None:
                            # if at some obs, the legal_action is None, add the fake target_policy
                            target_policies.append(
                                list(np.ones(self._cfg.model.action_space_size) / self._cfg.model.action_space_size)
                            )
                        else:
                            if self._cfg.env_type == 'not_board_games':
                                # for atari/classic_control/box2d environments that only have one player.
                                sum_visits = sum(distributions)
                                policy = [visit_count / sum_visits for visit_count in distributions]
                                target_policies.append(policy)
                            else:
                                # for board games that have two players and legal_actions is dy
                                policy_tmp = [0 for _ in range(self._cfg.model.action_space_size)]
                                # to make sure target_policies have the same dimension
                                sum_visits = sum(distributions)
                                policy = [visit_count / sum_visits for visit_count in distributions]
                                for index, legal_action in enumerate(roots_legal_actions_list[policy_index]):
                                    policy_tmp[legal_action] = policy[index]
                                target_policies.append(policy_tmp)

                    policy_index += 1

                batch_target_policies_re.append(target_policies)

        batch_target_policies_re = np.array(batch_target_policies_re)

       # NOTE: TODO
        model.world_model.reanalyze_phase = False

        return batch_target_policies_re

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
            to_play_segment = reward_value_context  # noqa
        # transition_batch_size = game_segment_batch_size * (num_unroll_steps+1)
        transition_batch_size = len(value_obs_list)

        batch_target_values, batch_rewards = [], []
        with torch.no_grad():
            value_obs_list = prepare_observation(value_obs_list, self._cfg.model.model_type)
            network_output = []
            batch_obs = torch.from_numpy(value_obs_list).to(self._cfg.device)

            # =============== NOTE: The key difference with MuZero =================
            # calculate the bootstrapped value and target value
            # NOTE: batch_obs(value_obs_list) is at t+td_steps, batch_action is at timestep t+td_steps
            m_output = model.initial_inference(batch_obs, batch_action, start_pos=batch_timestep)
            # ======================================================================

            # if not in training, obtain the scalars of the value/reward
            [m_output.latent_state, m_output.value, m_output.policy_logits] = to_detach_cpu_numpy(
                [
                    m_output.latent_state,
                    inverse_scalar_transform(m_output.value, self.value_support),
                    m_output.policy_logits
                ]
            )
            network_output.append(m_output)

            if self._cfg.use_root_value:
                value_numpy = np.array(root_values)
            else:
                # use the predicted values
                value_numpy = concat_output_value(network_output)

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

    def clear(self) -> None:
        """
        Overview:
            Clear all data in the replay buffer for online learning.
            This method resets the buffer to its initial empty state.
        """
        self.game_segment_buffer.clear()
        # game_pos_priorities might be a list or numpy array, reset to empty list
        self.game_pos_priorities = []
        self.game_segment_game_pos_look_up.clear()
        self.num_of_collected_episodes = 0
        self.base_idx = 0
        self.clear_time += 1

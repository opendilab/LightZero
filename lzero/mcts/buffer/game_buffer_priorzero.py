import numpy as np
from typing import List, Any, Union, Tuple
from lzero.mcts.buffer.game_buffer_unizero import UniZeroGameBuffer
from lzero.policy import to_detach_cpu_numpy, concat_output_value, inverse_scalar_transform
from lzero.mcts.utils import prepare_observation
import torch


class PriorZeroGameBufferOptimized(UniZeroGameBuffer):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.last_pos_in_transition = 0
    
    def mark_latest_transitions_consumed(self) -> None:
        self.last_pos_in_transition = self.get_num_of_transitions()
    
    def fetch_latest_batch(self, batch_size: int, policy) -> List[Any]:
        """
        Fetch latest batch for LLM training.

        Returns:
            [raw_obs_list, history_obs_list, llm_prior_per_tok_list, batch_target_values, batch_pred_values, cot_prefix_list, llm_action]
            CoT prefix list is added for CoT reuse optimization.
        """
        policy._target_model.to(self._cfg.device)
        policy._target_model.eval()

        reward_value_context, policy_re_context, policy_non_re_context, current_batch = self._make_batch(
            batch_size, self._cfg.reanalyze_ratio, fetch_latest=True
        )
        if not current_batch:
            return [[], [], [], [], [], [], []]

        obs_list, action_list, bootstrap_action_list, mask_list, batch_index_list, weights_list, make_time_list, timestep_list, raw_obs_list, history_obs_list, llm_prior_per_tok_list, cot_prefix_list, llm_action_list = current_batch

        # Standard processing
        batch_rewards, batch_target_values, batch_pred_values = self._compute_target_reward_value_and_pred_value(
            reward_value_context, policy._target_model, action_list, bootstrap_action_list, timestep_list
        )

        batch_target_policies = self._compute_target_policy_non_reanalyzed(
            policy_non_re_context, self.action_space_size
        )

        # CoT reuse optimization: return cot_prefix_list
        # IMPORTANT: Validate return value before returning to ensure broadcast compatibility
        result = [raw_obs_list, history_obs_list, llm_prior_per_tok_list, batch_target_values, batch_pred_values, cot_prefix_list, llm_action_list]

        return result
    
    def sample(self, batch_size: int, policy) -> List[Any]:
        """Sample data with game_segments (optimized version)."""
        policy._target_model.to(self._cfg.device)
        policy._target_model.eval()

        reward_value_context, policy_re_context, policy_non_re_context, current_batch = self._make_batch(
            batch_size, self._cfg.reanalyze_ratio
        )

        obs_list, action_list, bootstrap_action_list, mask_list, batch_index_list, weights_list, make_time_list, timestep_list, raw_obs_list, history_obs_list, llm_prior_per_tok_list, cot_prefix_list, llm_action_list = current_batch
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
        if not pos_in_game_segment_list:
            return [], [], [], []
        
        # Rest of the code is identical to parent's _make_batch
        batch_size = len(batch_index_list)
        obs_list, action_list, mask_list = [], [], []
        raw_obs_list, history_obs_list = [], []
        llm_prior_per_tok_list = []
        cot_prefix_list = []  # CoT reuse optimization
        llm_action_list = []
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
            llm_prior_per_tok_list.append(game_segment_list[i].get_unroll_llm_prior_per_tok(
                pos_in_game_segment_list[i], num_unroll_steps=self._cfg.num_unroll_steps, padding=True
            ))
            cot_prefix_list.append(game_segment_list[i].get_unroll_cot_prefix(
                pos_in_game_segment_list[i], num_unroll_steps=self._cfg.num_unroll_steps, padding=True
            ))
            llm_action_list.append(game_segment_list[i].get_unroll_llm_action(
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
        # 检查 vllm和policy_model的输入上下文是否一致 (only for non-padded positions)
        assert len(raw_obs_list) == len(history_obs_list) == len(llm_prior_per_tok_list) == len(cot_prefix_list) == len(llm_action_list)
        B, T = len(raw_obs_list), len(raw_obs_list[0])
        for b in range(B):
            for t in range(T - 1):
                # Skip padded positions: mask[t] == 0 means the action at step t is padding,
                # so llm_prior_per_tok at t+1 is also padding and the alignment invariant doesn't hold.
                if mask_list[b][t] == 0.:
                    continue
                current_obs = raw_obs_list[b][t]
                current_hist = history_obs_list[b][t]

                old_prefix_cot = llm_prior_per_tok_list[b][t+1]['prefix_cot']
                old_current_obs = llm_prior_per_tok_list[b][t+1]['current_obs']
                old_history = llm_prior_per_tok_list[b][t+1]['history']
                old_logprob = llm_prior_per_tok_list[b][t+1]['rollout_action_logprob']
                cot_prefix = cot_prefix_list[b][t+1]
                llm_action = llm_action_list[b][t+1]

                assert llm_action in old_logprob
                assert old_current_obs == current_obs and old_history == current_hist and old_prefix_cot == cot_prefix           

        current_batch.append(raw_obs_list)
        current_batch.append(history_obs_list)
        current_batch.append(llm_prior_per_tok_list)
        current_batch.append(cot_prefix_list)  # CoT reuse optimization
        current_batch.append(llm_action_list)

        total_transitions = self.get_num_of_transitions()

        if not fetch_latest:
            reward_value_context = self._prepare_reward_value_context(
                batch_index_list, game_segment_list, pos_in_game_segment_list, total_transitions
            )
        else:
            reward_value_context = self._prepare_reward_value_context_and_pred_values(
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
        latest_new_indices = list(range(self.last_pos_in_transition, num_of_transitions))
        if batch_size == -1:
            candidate_batch_index_list = latest_new_indices 
        else:
            candidate_batch_index_list = latest_new_indices[-batch_size:]

        game_segment_list = []
        pos_in_game_segment_list = []
        batch_index_list = []

        for idx in candidate_batch_index_list:
            game_segment_idx, pos_in_game_segment = self.game_segment_game_pos_look_up[idx]
            game_segment_idx -= self.base_idx  # Adjust index based on base index
            game_segment = self.game_segment_buffer[game_segment_idx]

            assert len(game_segment.obs_segment) == len(game_segment.raw_obs_segment) == len(game_segment.cot_prefix_segment)
            segment_len = len(game_segment.action_segment)
            if self._cfg.action_type == 'varied_action_space':
                within_obs_window = pos_in_game_segment + self._cfg.num_unroll_steps + self._cfg.model.frame_stack_num <= len(game_segment.obs_segment)
                within_td_window = pos_in_game_segment < self._cfg.game_segment_length - self._cfg.num_unroll_steps
                valid_next_action = pos_in_game_segment < segment_len - 1
                is_valid_latest_transition = within_obs_window and within_td_window and valid_next_action
            else:
                within_obs_window = pos_in_game_segment + self._cfg.num_unroll_steps + self._cfg.model.frame_stack_num <= len(game_segment.obs_segment)
                within_segment_window = pos_in_game_segment < self._cfg.game_segment_length
                valid_next_action = pos_in_game_segment < segment_len - 1
                is_valid_latest_transition = within_obs_window and within_segment_window and valid_next_action

            if not is_valid_latest_transition:
                continue
            
            game_segment_list.append(game_segment)
            pos_in_game_segment_list.append(pos_in_game_segment)
            batch_index_list.append(idx)
                
        import random
        n = min(256, len(game_segment_list))
        print(f"new transition={len(latest_new_indices)} | valid_pos_in_gamesemt={len(game_segment_list)} | final_pos_in_gamesemt={n}")
        indices = random.sample(range(len(game_segment_list)), n)
        game_segment_list = [game_segment_list[i] for i in indices]
        pos_in_game_segment_list = [pos_in_game_segment_list[i] for i in indices]
        batch_index_list = [batch_index_list[i] for i in indices]
        # make_time = [time.time() for _ in range(len(batch_index_list))]

        # Set the make_time for each sample (set to 0 for now, but can be the actual time if needed).
        make_time = [0. for _ in range(len(batch_index_list))]

        orig_data = (game_segment_list, pos_in_game_segment_list, batch_index_list, None, make_time)
            
        return orig_data
    
    # 从原来的_prepare_reward_value_context函数修改得到
    def _prepare_reward_value_context_and_pred_values(
            self, batch_index_list: List[str], game_segment_list: List[Any], pos_in_game_segment_list: List[Any],
            total_transitions: int
    ) -> List[Any]:
        """
        Overview:
            prepare the context of rewards and values for calculating TD value target in reanalyzing part.
        Arguments:
            - batch_index_list (:obj:`list`): the index of start transition of sampled minibatch in replay buffer
            - game_segment_list (:obj:`list`): list of game segments
            - pos_in_game_segment_list (:obj:`list`): list of transition index in game_segment
            - total_transitions (:obj:`int`): number of collected transitions
        Returns:
            - reward_value_context (:obj:`list`): value_obs_list, value_mask, pos_in_game_segment_list, rewards_list, game_segment_lens,
              td_steps_list, action_mask_segment, to_play_segment
        """
        zero_obs = game_segment_list[0].zero_obs()
        
        pred_obs_list = []
        pred_mask = []
        
        value_obs_list = []
        # the value is valid or not (out of game_segment)
        value_mask = []
        rewards_list = []
        game_segment_lens = []
        # for board games
        action_mask_segment, to_play_segment = [], []

        root_values = []

        td_steps_list = []
        for game_segment, state_index in zip(game_segment_list, pos_in_game_segment_list):
            game_segment_len = len(game_segment)
            game_segment_lens.append(game_segment_len)
            # original buffer td-steps
            td_steps = np.clip(self._cfg.td_steps, 1, max(1, game_segment_len - state_index)).astype(np.int32)

            # prepare the corresponding observations for bootstrapped values o_{t+k}
            # o[t+ td_steps, t + td_steps + stack frames + num_unroll_steps]
            # t=2+3 -> o[2+3, 2+3+4+5] -> o[5, 14]
            game_obs_pred = game_segment.get_unroll_obs(state_index, self._cfg.num_unroll_steps)
            game_obs = game_segment.get_unroll_obs(state_index + td_steps, self._cfg.num_unroll_steps)

            rewards_list.append(game_segment.reward_segment)
            
            # for board games
            action_mask_segment.append(game_segment.action_mask_segment)
            to_play_segment.append(game_segment.to_play_segment)

            truncation_length = game_segment_len

            for current_index in range(state_index, state_index + self._cfg.num_unroll_steps + 1):
                # get the <num_unroll_steps+1>  bootstrapped target obs
                td_steps_list.append(td_steps)
                # index of bootstrapped obs o_{t+td_steps}
                bootstrap_index = current_index + td_steps
                
                beg_index = current_index - state_index
                end_index = beg_index + self._cfg.model.frame_stack_num
                
                if bootstrap_index < truncation_length:
                    value_mask.append(1)
                    # the stacked obs in time t
                    obs = game_obs[beg_index:end_index]
                else:
                    value_mask.append(0)
                    obs = zero_obs
                
                if current_index < truncation_length:
                    pred_mask.append(1)
                    obs_pred = game_obs_pred[beg_index:end_index]
                else:
                    pred_mask.append(0)
                    obs_pred = zero_obs

                value_obs_list.append(obs)
                pred_obs_list.append(obs_pred)

        reward_value_context = [
            value_obs_list, value_mask, pos_in_game_segment_list, rewards_list, root_values, game_segment_lens, td_steps_list,
            action_mask_segment, to_play_segment, pred_obs_list, pred_mask
        ]
        return reward_value_context
    
    # 从原来的_compute_target_reward_value函数修改得到
    def _compute_target_reward_value_and_pred_value(self, reward_value_context: List[Any], model: Any, batch_action_pred, batch_action, batch_timestep) -> Tuple[Any, Any]:
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
            to_play_segment, pred_obs_list, pred_mask = reward_value_context  # noqa
        # transition_batch_size = game_segment_batch_size * (num_unroll_steps+1)
        transition_batch_size = len(value_obs_list)

        batch_target_values, batch_rewards, batch_pred_values = [], [], []
        with torch.no_grad():
            value_obs_list = prepare_observation(value_obs_list, self._cfg.model.model_type)
            pred_obs_list = prepare_observation(pred_obs_list, self._cfg.model.model_type)
            
            network_output = []
            network_output_pred = []
            
            batch_obs = torch.from_numpy(value_obs_list).to(self._cfg.device)
            batch_obs_pred = torch.from_numpy(pred_obs_list).to(self._cfg.device)

            # =============== NOTE: The key difference with MuZero =================
            # calculate the bootstrapped value and target value
            # NOTE: batch_obs(value_obs_list) is at t+td_steps, batch_action is at timestep t+td_steps
            if self.task_id is not None:
                # m_output = model.initial_inference(batch_obs, batch_action, start_pos=batch_timestep, task_id=self.task_id)
                m_output = model.initial_inference(batch_obs, batch_action, task_id=self.task_id)
                m_output_pred = model.initial_inference(batch_obs_pred, batch_action_pred, task_id=self.task_id)

            else:
                m_output = model.initial_inference(batch_obs, batch_action, start_pos=batch_timestep)
                m_output_pred = model.initial_inference(batch_obs_pred, batch_action_pred, start_pos=batch_timestep)

            # ======================================================================

            # if not in training, obtain the scalars of the value/reward
            [m_output.latent_state, m_output.value, m_output.policy_logits] = to_detach_cpu_numpy(
                [
                    m_output.latent_state,
                    inverse_scalar_transform(m_output.value, self.value_support),
                    m_output.policy_logits
                ]
            )
            [m_output_pred.latent_state, m_output_pred.value, m_output_pred.policy_logits] = to_detach_cpu_numpy(
                [
                    m_output_pred.latent_state,
                    inverse_scalar_transform(m_output_pred.value, self.value_support),
                    m_output_pred.policy_logits
                ]
            )
            
            network_output.append(m_output)
            network_output_pred.append(m_output_pred)

            if self._cfg.use_root_value:
                value_numpy = np.array(root_values)
                raise ValueError("error!!!")
            else:
                # use the predicted values
                value_numpy = concat_output_value(network_output)
                pred_numpy = concat_output_value(network_output_pred)

            # 不考虑 board_games的情况
            value_numpy = value_numpy.reshape(-1) * (
                    np.array([self._cfg.discount_factor for _ in range(transition_batch_size)]) ** td_steps_list
            )
            pred_numpy = pred_numpy.reshape(-1)
            
            value_numpy= value_numpy * np.array(value_mask)
            value_list = value_numpy.tolist()
            
            pred_numpy = pred_numpy * np.array(pred_mask)                 
            pred_list = pred_numpy.tolist()

            
            horizon_id, value_index = 0, 0

            for game_segment_len_non_re, reward_list, state_index, to_play_list in zip(game_segment_lens, rewards_list,
                                                                                       pos_in_game_segment_list,
                                                                                       to_play_segment):
                target_values = []
                target_rewards = []
                pred_values = []
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
                        # 不考虑 board_games的情况
                        value_list[value_index] += reward * self._cfg.discount_factor ** i
                    horizon_id += 1

                    # TODO: check the boundary condition
                    target_values.append(value_list[value_index])
                    pred_values.append(pred_list[value_index])
                    
                    if current_index < len(reward_list):
                        target_rewards.append(reward_list[current_index])
                    else:
                        target_rewards.append(np.array(0.))

                    value_index += 1

                batch_rewards.append(target_rewards)
                batch_target_values.append(target_values)
                batch_pred_values.append(pred_values)

        batch_rewards = np.asarray(batch_rewards)
        batch_target_values = np.asarray(batch_target_values)
        batch_pred_values = np.asarray(batch_pred_values)

        return batch_rewards, batch_target_values, batch_pred_values
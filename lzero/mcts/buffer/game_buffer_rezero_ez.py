from typing import Any, List, Union, TYPE_CHECKING

import numpy as np
import torch
from ding.utils import BUFFER_REGISTRY, EasyTimer

from lzero.mcts.tree_search.mcts_ctree import EfficientZeroMCTSCtree as MCTSCtree
from lzero.mcts.utils import prepare_observation
from lzero.policy import DiscreteSupport, to_detach_cpu_numpy, concat_output, inverse_scalar_transform
from .game_buffer_efficientzero import EfficientZeroGameBuffer
from .game_buffer_rezero_mz import ReZeroMZGameBuffer, compute_all_filters

# from line_profiler import line_profiler

if TYPE_CHECKING:
    from lzero.policy import MuZeroPolicy, EfficientZeroPolicy, SampledEfficientZeroPolicy


@BUFFER_REGISTRY.register('game_buffer_rezero_ez')
class ReZeroEZGameBuffer(EfficientZeroGameBuffer, ReZeroMZGameBuffer):
    """
    Overview:
        The specific game buffer for ReZero-EfficientZero policy.
    """

    def __init__(self, cfg: dict):
        """
        Overview:
            Initialize the ReZeroEZGameBuffer with the given configuration. If a user passes in a cfg with a key that matches an existing key
            in the default configuration, the user-provided value will override the default configuration. Otherwise,
            the default configuration will be used.
        """
        super().__init__(cfg)

        # Update the default configuration with the provided configuration.
        default_config = self.default_config()
        default_config.update(cfg)
        self._cfg = default_config

        # Ensure the configuration values are valid.
        assert self._cfg.env_type in ['not_board_games', 'board_games']
        assert self._cfg.action_type in ['fixed_action_space', 'varied_action_space']

        # Initialize various parameters from the configuration.
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

        # Timers for performance monitoring
        self._compute_target_timer = EasyTimer()
        self._reuse_search_timer = EasyTimer()
        self._origin_search_timer = EasyTimer()
        self.buffer_reanalyze = True

        # Performance metrics
        self.compute_target_re_time = 0
        self.reuse_search_time = 0
        self.origin_search_time = 0
        self.sample_times = 0
        self.active_root_num = 0
        self.average_infer = 0

        self.value_support = DiscreteSupport(*self._cfg.model.value_support_range)
        self.reward_support = DiscreteSupport(*self._cfg.model.reward_support_range)

    def sample(
            self, batch_size: int, policy: Union["MuZeroPolicy", "EfficientZeroPolicy", "SampledEfficientZeroPolicy"]
    ) -> List[Any]:
        """
        Overview:
            Sample data from the GameBuffer and prepare the current and target batch for training.
        Arguments:
            - batch_size (int): Batch size.
            - policy (Union["MuZeroPolicy", "EfficientZeroPolicy", "SampledEfficientZeroPolicy"]): Policy.
        Returns:
            - train_data (List): List of train data, including current_batch and target_batch.
        """
        policy._target_model.to(self._cfg.device)
        policy._target_model.eval()

        # Obtain the current_batch and prepare target context
        reward_value_context, policy_re_context, policy_non_re_context, current_batch = self._make_batch(
            batch_size, self._cfg.reanalyze_ratio
        )

        # Compute target values and policies
        batch_value_prefixs, batch_target_values = self._compute_target_reward_value(
            reward_value_context, policy._target_model
        )

        with self._compute_target_timer:
            batch_target_policies_re = self._compute_target_policy_reanalyzed(policy_re_context, policy._target_model)
        self.compute_target_re_time += self._compute_target_timer.value

        batch_target_policies_non_re = self._compute_target_policy_non_reanalyzed(
            policy_non_re_context, self._cfg.model.action_space_size
        )

        # Fuse reanalyzed and non-reanalyzed target policies
        if 0 < self._cfg.reanalyze_ratio < 1:
            batch_target_policies = np.concatenate([batch_target_policies_re, batch_target_policies_non_re])
        elif self._cfg.reanalyze_ratio == 1:
            batch_target_policies = batch_target_policies_re
        elif self._cfg.reanalyze_ratio == 0:
            batch_target_policies = batch_target_policies_non_re

        target_batch = [batch_value_prefixs, batch_target_values, batch_target_policies]

        # A batch contains the current_batch and the target_batch
        train_data = [current_batch, target_batch]
        if not self.buffer_reanalyze:
            self.sample_times += 1
        return train_data

    def _compute_target_policy_reanalyzed(self, policy_re_context: List[Any], model: Any, length=None) -> np.ndarray:
        """
        Overview:
            Prepare policy targets from the reanalyzed context of policies.
        Arguments:
            - policy_re_context (List): List of policy context to be reanalyzed.
            - model (Any): The model used for inference.
            - length (int, optional): The length of unroll steps.
        Returns:
            - batch_target_policies_re (np.ndarray): The reanalyzed policy targets.
        """
        if policy_re_context is None:
            return []

        batch_target_policies_re = []

        unroll_steps = length - 1 if length is not None else self._cfg.num_unroll_steps

        policy_obs_list, true_action, policy_mask, pos_in_game_segment_list, batch_index_list, child_visits, root_values, game_segment_lens, action_mask_segment, to_play_segment = policy_re_context

        transition_batch_size = len(policy_obs_list)
        game_segment_batch_size = len(pos_in_game_segment_list)

        to_play, action_mask = self._preprocess_to_play_and_action_mask(
            game_segment_batch_size, to_play_segment, action_mask_segment, pos_in_game_segment_list, length
        )

        if self._cfg.model.continuous_action_space:
            action_mask = [
                list(np.ones(self._cfg.model.action_space_size, dtype=np.int8)) for _ in range(transition_batch_size)
            ]
            legal_actions = [
                [-1 for _ in range(self._cfg.model.action_space_size)] for _ in range(transition_batch_size)
            ]
        else:
            legal_actions = [[i for i, x in enumerate(action_mask[j]) if x == 1] for j in range(transition_batch_size)]

        with torch.no_grad():
            policy_obs_list = prepare_observation(policy_obs_list, self._cfg.model.model_type)
            slices = int(np.ceil(transition_batch_size / self._cfg.mini_infer_size))
            network_output = []

            for i in range(slices):
                beg_index = self._cfg.mini_infer_size * i
                end_index = self._cfg.mini_infer_size * (i + 1)
                m_obs = torch.from_numpy(policy_obs_list[beg_index:end_index]).to(self._cfg.device)
                m_output = model.initial_inference(m_obs)

                if not model.training:
                    [m_output.latent_state, m_output.value, m_output.policy_logits] = to_detach_cpu_numpy(
                        [
                            m_output.latent_state,
                            inverse_scalar_transform(m_output.value, self.value_support),
                            m_output.policy_logits
                        ]
                    )
                    m_output.reward_hidden_state = (
                        m_output.reward_hidden_state[0].detach().cpu().numpy(),
                        m_output.reward_hidden_state[1].detach().cpu().numpy()
                    )

                network_output.append(m_output)

            _, value_prefix_pool, policy_logits_pool, latent_state_roots, reward_hidden_state_roots = concat_output(
                network_output, data_type='efficientzero'
            )
            value_prefix_pool = value_prefix_pool.squeeze().tolist()
            policy_logits_pool = policy_logits_pool.tolist()
            noises = [
                np.random.dirichlet([self._cfg.root_dirichlet_alpha] * self._cfg.model.action_space_size).astype(
                    np.float32).tolist()
                for _ in range(transition_batch_size)
            ]

            if self._cfg.mcts_ctree:
                legal_actions_by_iter = compute_all_filters(legal_actions, unroll_steps)
                noises_by_iter = compute_all_filters(noises, unroll_steps)
                value_prefix_pool_by_iter = compute_all_filters(value_prefix_pool, unroll_steps)
                policy_logits_pool_by_iter = compute_all_filters(policy_logits_pool, unroll_steps)
                to_play_by_iter = compute_all_filters(to_play, unroll_steps)
                latent_state_roots_by_iter = compute_all_filters(latent_state_roots, unroll_steps)

                batch1_core_by_iter = compute_all_filters(reward_hidden_state_roots[0][0], unroll_steps)
                batch2_core_by_iter = compute_all_filters(reward_hidden_state_roots[1][0], unroll_steps)
                true_action_by_iter = compute_all_filters(true_action, unroll_steps)

                temp_values = []
                temp_distributions = []
                mcts_ctree = MCTSCtree(self._cfg)
                temp_search_time = 0
                temp_length = 0
                temp_infer = 0

                if self._cfg.reuse_search:
                    for iter in range(unroll_steps + 1):
                        iter_batch_size = transition_batch_size / (unroll_steps + 1)
                        roots = MCTSCtree.roots(iter_batch_size, legal_actions_by_iter[iter])
                        if self._cfg.reanalyze_noise:
                            roots.prepare(self._cfg.root_noise_weight, noises_by_iter[iter],
                                          value_prefix_pool_by_iter[iter], policy_logits_pool_by_iter[iter],
                                          to_play_by_iter[iter])
                        else:
                            roots.prepare_no_noise(value_prefix_pool_by_iter[iter], policy_logits_pool_by_iter[iter],
                                                   to_play_by_iter[iter])

                        if iter == 0:
                            with self._origin_search_timer:
                                mcts_ctree.search(roots, model, latent_state_roots_by_iter[iter],
                                                  [[batch1_core_by_iter[iter]], [batch2_core_by_iter[iter]]],
                                                  to_play_by_iter[iter])
                            self.origin_search_time += self._origin_search_timer.value
                        else:
                            with self._reuse_search_timer:
                                # ===================== Core implementation of ReZero: search_with_reuse =====================
                                length, average_infer = mcts_ctree.search_with_reuse(roots, model,
                                                                                     latent_state_roots_by_iter[iter],
                                                                                     [[batch1_core_by_iter[iter]],
                                                                                      [batch2_core_by_iter[iter]]],
                                                                                     to_play_by_iter[iter],
                                                                                     true_action_list=
                                                                                     true_action_by_iter[iter],
                                                                                     reuse_value_list=iter_values)
                            temp_search_time += self._reuse_search_timer.value
                            temp_length += length
                            temp_infer += average_infer

                        iter_values = roots.get_values()
                        iter_distributions = roots.get_distributions()
                        temp_values.append(iter_values)
                        temp_distributions.append(iter_distributions)

                else:
                    for iter in range(unroll_steps + 1):
                        iter_batch_size = transition_batch_size / (unroll_steps + 1)
                        roots = MCTSCtree.roots(iter_batch_size, legal_actions_by_iter[iter])
                        if self._cfg.reanalyze_noise:
                            roots.prepare(self._cfg.root_noise_weight, noises_by_iter[iter],
                                          value_prefix_pool_by_iter[iter], policy_logits_pool_by_iter[iter],
                                          to_play_by_iter[iter])
                        else:
                            roots.prepare_no_noise(value_prefix_pool_by_iter[iter], policy_logits_pool_by_iter[iter],
                                                   to_play_by_iter[iter])

                        with self._origin_search_timer:
                            mcts_ctree.search(roots, model, latent_state_roots_by_iter[iter],
                                              [[batch1_core_by_iter[iter]], [batch2_core_by_iter[iter]]],
                                              to_play_by_iter[iter])
                        self.origin_search_time += self._origin_search_timer.value

                        iter_values = roots.get_values()
                        iter_distributions = roots.get_distributions()
                        temp_values.append(iter_values)
                        temp_distributions.append(iter_distributions)

                    self.origin_search_time = self.origin_search_time / (unroll_steps + 1)

                if unroll_steps == 0:
                    self.reuse_search_time = 0
                    self.active_root_num = 0
                else:
                    self.reuse_search_time += (temp_search_time / unroll_steps)
                    self.active_root_num += (temp_length / unroll_steps)
                    self.average_infer += (temp_infer / unroll_steps)

                roots_legal_actions_list = legal_actions
                temp_values.reverse()
                temp_distributions.reverse()
                roots_values = []
                roots_distributions = []
                [roots_values.extend(column) for column in zip(*temp_values)]
                [roots_distributions.extend(column) for column in zip(*temp_distributions)]

                policy_index = 0
                for state_index, child_visit, root_value in zip(pos_in_game_segment_list, child_visits, root_values):
                    target_policies = []

                    for current_index in range(state_index, state_index + unroll_steps + 1):
                        distributions = roots_distributions[policy_index]
                        searched_value = roots_values[policy_index]

                        if policy_mask[policy_index] == 0:
                            target_policies.append([0 for _ in range(self._cfg.model.action_space_size)])
                        else:
                            if distributions is None:
                                target_policies.append(list(
                                    np.ones(self._cfg.model.action_space_size) / self._cfg.model.action_space_size))
                            else:
                                sim_num = sum(distributions)
                                child_visit[current_index] = [visit_count / sim_num for visit_count in distributions]
                                root_value[current_index] = searched_value
                                if self._cfg.action_type == 'fixed_action_space':
                                    sum_visits = sum(distributions)
                                    policy = [visit_count / sum_visits for visit_count in distributions]
                                    target_policies.append(policy)
                                else:
                                    policy_tmp = [0 for _ in range(self._cfg.model.action_space_size)]
                                    sum_visits = sum(distributions)
                                    policy = [visit_count / sum_visits for visit_count in distributions]
                                    for index, legal_action in enumerate(roots_legal_actions_list[policy_index]):
                                        policy_tmp[legal_action] = policy[index]
                                    target_policies.append(policy_tmp)

                        policy_index += 1

                    batch_target_policies_re.append(target_policies)

        return np.array(batch_target_policies_re)

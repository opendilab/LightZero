from typing import Any, List, Tuple, Union, TYPE_CHECKING

import numpy as np
import torch
from ding.torch_utils.data_helper import to_list
from ding.utils import BUFFER_REGISTRY, EasyTimer

from lzero.mcts.tree_search.mcts_ctree import MuZeroMCTSCtree as MCTSCtree
from lzero.mcts.tree_search.mcts_ptree import MuZeroMCTSPtree as MCTSPtree
from lzero.mcts.utils import prepare_observation
from lzero.policy import to_detach_cpu_numpy, concat_output, inverse_scalar_transform
from .game_buffer_muzero import MuZeroGameBuffer

# from line_profiler import line_profiler
if TYPE_CHECKING:
    from lzero.policy import MuZeroPolicy, EfficientZeroPolicy, SampledEfficientZeroPolicy


def compute_all_filters(data, num_unroll_steps):
    data_by_iter = []
    for iter in range(num_unroll_steps + 1):
        iter_data = [x for i, x in enumerate(data)
                     if (i + 1) % (num_unroll_steps + 1) ==
                     ((num_unroll_steps + 1 - iter) % (num_unroll_steps + 1))]
        data_by_iter.append(iter_data)
    return data_by_iter


@BUFFER_REGISTRY.register('game_buffer_rezero_mz')
class ReZeroMZGameBuffer(MuZeroGameBuffer):
    """
    Overview:
        The specific game buffer for ReZero-MuZero policy.
    """

    def __init__(self, cfg: dict):
        """
        Overview:
            Use the default configuration mechanism. If a user passes in a cfg with a key that matches an existing key
            in the default configuration, the user-provided value will override the default configuration. Otherwise,
            the default configuration will be used.
        """
        super().__init__(cfg)
        default_config = self.default_config()
        default_config.update(cfg)
        self._cfg = default_config

        # Ensure valid configuration values
        assert self._cfg.env_type in ['not_board_games', 'board_games']
        assert self._cfg.action_type in ['fixed_action_space', 'varied_action_space']

        # Initialize buffer parameters
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

        self._compute_target_timer = EasyTimer()
        self._reuse_search_timer = EasyTimer()
        self._origin_search_timer = EasyTimer()
        self.buffer_reanalyze = True
        self.compute_target_re_time = 0
        self.reuse_search_time = 0
        self.origin_search_time = 0
        self.sample_times = 0
        self.active_root_num = 0
        self.average_infer = 0

    def reanalyze_buffer(
            self, batch_size: int, policy: Union["MuZeroPolicy", "EfficientZeroPolicy", "SampledEfficientZeroPolicy"]
    ) -> List[Any]:
        """
        Overview:
            Sample data from ``GameBuffer`` and prepare the current and target batch for training.
        Arguments:
            - batch_size (:obj:`int`): Batch size.
            - policy (:obj:`Union["MuZeroPolicy", "EfficientZeroPolicy", "SampledEfficientZeroPolicy"]`): Policy.
        Returns:
            - train_data (:obj:`List`): List of train data, including current_batch and target_batch.
        """
        assert self._cfg.mcts_ctree is True, "ReZero-MuZero only supports cpp mcts_ctree now!"
        policy._target_model.to(self._cfg.device)
        policy._target_model.eval()

        # Obtain the current batch and prepare target context
        policy_re_context = self._make_reanalyze_batch(batch_size)

        with self._compute_target_timer:
            segment_length = self.get_num_of_transitions() // 2000
            batch_target_policies_re = self._compute_target_policy_reanalyzed(
                policy_re_context, policy._target_model, segment_length
            )

        self.compute_target_re_time += self._compute_target_timer.value

        if self.buffer_reanalyze:
            self.sample_times += 1

    def _make_reanalyze_batch(self, batch_size: int) -> Tuple[Any]:
        """
        Overview:
            First sample orig_data through ``_sample_orig_data()``, then prepare the context of a batch:
                reward_value_context:        The context of reanalyzed value targets.
                policy_re_context:           The context of reanalyzed policy targets.
                policy_non_re_context:       The context of non-reanalyzed policy targets.
                current_batch:               The inputs of batch.
        Arguments:
            - batch_size (:obj:`int`): The batch size of orig_data from replay buffer.
        Returns:
            - context (:obj:`Tuple`): reward_value_context, policy_re_context, policy_non_re_context, current_batch
        """
        # Obtain the batch context from replay buffer
        orig_data = self._sample_orig_reanalyze_data(batch_size)
        game_segment_list, pos_in_game_segment_list, batch_index_list, _, make_time_list = orig_data
        segment_length = self.get_num_of_transitions() // 2000
        policy_re_context = self._prepare_policy_reanalyzed_context(
            [], game_segment_list, pos_in_game_segment_list, segment_length
        )
        return policy_re_context

    def _prepare_policy_reanalyzed_context(
            self, batch_index_list: List[str], game_segment_list: List[Any], pos_in_game_segment_list: List[str],
            length=None
    ) -> List[Any]:
        """
        Overview:
            Prepare the context of policies for calculating policy target in reanalyzing part.
        Arguments:
            - batch_index_list (:obj:`list`): Start transition index in the replay buffer.
            - game_segment_list (:obj:`list`): List of game segments.
            - pos_in_game_segment_list (:obj:`list`): Position of transition index in one game history.
            - length (:obj:`int`, optional): Length of segments.
        Returns:
            - policy_re_context (:obj:`list`): policy_obs_list, policy_mask, pos_in_game_segment_list, indices,
              child_visits, game_segment_lens, action_mask_segment, to_play_segment
        """
        zero_obs = game_segment_list[0].zero_obs()
        with torch.no_grad():
            policy_obs_list = []
            true_action = []
            policy_mask = []

            unroll_steps = length - 1 if length is not None else self._cfg.num_unroll_steps
            rewards, child_visits, game_segment_lens, root_values = [], [], [], []
            action_mask_segment, to_play_segment = [], []

            for game_segment, state_index in zip(game_segment_list, pos_in_game_segment_list):
                game_segment_len = len(game_segment)
                game_segment_lens.append(game_segment_len)
                rewards.append(game_segment.reward_segment)
                action_mask_segment.append(game_segment.action_mask_segment)
                to_play_segment.append(game_segment.to_play_segment)
                child_visits.append(game_segment.child_visit_segment)
                root_values.append(game_segment.root_value_segment)

                # Prepare the corresponding observations
                game_obs = game_segment.get_unroll_obs(state_index, unroll_steps)
                for current_index in range(state_index, state_index + unroll_steps + 1):
                    if current_index < game_segment_len:
                        policy_mask.append(1)
                        beg_index = current_index - state_index
                        end_index = beg_index + self._cfg.model.frame_stack_num
                        obs = game_obs[beg_index:end_index]
                        action = game_segment.action_segment[current_index]
                        if current_index == game_segment_len - 1:
                            action = -64
                    else:
                        policy_mask.append(0)
                        obs = zero_obs
                        action = -64
                    policy_obs_list.append(obs)
                    true_action.append(action)

        policy_re_context = [
            policy_obs_list, true_action, policy_mask, pos_in_game_segment_list, batch_index_list, child_visits,
            root_values, game_segment_lens, action_mask_segment, to_play_segment
        ]
        return policy_re_context

    # @profile
    def _compute_target_policy_reanalyzed(self, policy_re_context: List[Any], model: Any, length=None) -> np.ndarray:
        """
        Overview:
            Prepare policy targets from the reanalyzed context of policies.
        Arguments:
            - policy_re_context (:obj:`List`): List of policy context to be reanalyzed.
            - model (:obj:`Any`): The model used for inference.
            - length (:obj:`int`, optional): The length of the unroll steps.
        Returns:
            - batch_target_policies_re (:obj:`np.ndarray`): The reanalyzed batch target policies.
        """
        if policy_re_context is None:
            return []

        batch_target_policies_re = []

        unroll_steps = length - 1 if length is not None else self._cfg.num_unroll_steps

        # Unpack the policy reanalyze context
        (
            policy_obs_list, true_action, policy_mask, pos_in_game_segment_list, batch_index_list,
            child_visits, root_values, game_segment_lens, action_mask_segment, to_play_segment
        ) = policy_re_context

        transition_batch_size = len(policy_obs_list)
        game_segment_batch_size = len(pos_in_game_segment_list)

        to_play, action_mask = self._preprocess_to_play_and_action_mask(
            game_segment_batch_size, to_play_segment, action_mask_segment, pos_in_game_segment_list, unroll_steps
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
                    m_output.latent_state, m_output.value, m_output.policy_logits = to_detach_cpu_numpy(
                        [
                            m_output.latent_state,
                            inverse_scalar_transform(m_output.value, self._cfg.model.support_scale),
                            m_output.policy_logits
                        ]
                    )

                network_output.append(m_output)

            _, reward_pool, policy_logits_pool, latent_state_roots = concat_output(network_output, data_type='muzero')
            reward_pool = reward_pool.squeeze().tolist()
            policy_logits_pool = policy_logits_pool.tolist()
            noises = [
                np.random.dirichlet([self._cfg.root_dirichlet_alpha] * self._cfg.model.action_space_size).astype(
                    np.float32).tolist()
                for _ in range(transition_batch_size)
            ]

            if self._cfg.mcts_ctree:
                legal_actions_by_iter = compute_all_filters(legal_actions, unroll_steps)
                noises_by_iter = compute_all_filters(noises, unroll_steps)
                reward_pool_by_iter = compute_all_filters(reward_pool, unroll_steps)
                policy_logits_pool_by_iter = compute_all_filters(policy_logits_pool, unroll_steps)
                to_play_by_iter = compute_all_filters(to_play, unroll_steps)
                latent_state_roots_by_iter = compute_all_filters(latent_state_roots, unroll_steps)
                true_action_by_iter = compute_all_filters(true_action, unroll_steps)

                temp_values, temp_distributions = [], []
                mcts_ctree = MCTSCtree(self._cfg)
                temp_search_time, temp_length, temp_infer = 0, 0, 0

                if self._cfg.reuse_search:
                    for iter in range(unroll_steps + 1):
                        iter_batch_size = transition_batch_size / (unroll_steps + 1)
                        roots = MCTSCtree.roots(iter_batch_size, legal_actions_by_iter[iter])

                        if self._cfg.reanalyze_noise:
                            roots.prepare(
                                self._cfg.root_noise_weight,
                                noises_by_iter[iter],
                                reward_pool_by_iter[iter],
                                policy_logits_pool_by_iter[iter],
                                to_play_by_iter[iter]
                            )
                        else:
                            roots.prepare_no_noise(
                                reward_pool_by_iter[iter],
                                policy_logits_pool_by_iter[iter],
                                to_play_by_iter[iter]
                            )

                        if iter == 0:
                            with self._origin_search_timer:
                                mcts_ctree.search(roots, model, latent_state_roots_by_iter[iter], to_play_by_iter[iter])
                            self.origin_search_time += self._origin_search_timer.value
                        else:
                            with self._reuse_search_timer:
                                # ===================== Core implementation of ReZero: search_with_reuse =====================
                                length, average_infer = mcts_ctree.search_with_reuse(
                                    roots, model, latent_state_roots_by_iter[iter], to_play_by_iter[iter],
                                    true_action_list=true_action_by_iter[iter], reuse_value_list=iter_values
                                )
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
                            roots.prepare(
                                self._cfg.root_noise_weight,
                                noises_by_iter[iter],
                                reward_pool_by_iter[iter],
                                policy_logits_pool_by_iter[iter],
                                to_play_by_iter[iter]
                            )
                        else:
                            roots.prepare_no_noise(
                                reward_pool_by_iter[iter],
                                policy_logits_pool_by_iter[iter],
                                to_play_by_iter[iter]
                            )

                        with self._origin_search_timer:
                            mcts_ctree.search(roots, model, latent_state_roots_by_iter[iter], to_play_by_iter[iter])
                        self.origin_search_time += self._origin_search_timer.value

                        iter_values = roots.get_values()
                        iter_distributions = roots.get_distributions()
                        temp_values.append(iter_values)
                        temp_distributions.append(iter_distributions)

                    self.origin_search_time /= (unroll_steps + 1)

                if unroll_steps == 0:
                    self.reuse_search_time, self.active_root_num = 0, 0
                else:
                    self.reuse_search_time += (temp_search_time / unroll_steps)
                    self.active_root_num += (temp_length / unroll_steps)
                    self.average_infer += (temp_infer / unroll_steps)

                roots_legal_actions_list = legal_actions
                temp_values.reverse()
                temp_distributions.reverse()
                roots_values, roots_distributions = [], []
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
                                target_policies.append(
                                    list(np.ones(self._cfg.model.action_space_size) / self._cfg.model.action_space_size)
                                )
                            else:
                                # ===================== Update the data in buffer =====================
                                # After the reanalysis search, new target policies and root values are obtained.
                                # These target policies and root values are stored in the game segment, specifically in the ``child_visit_segment`` and ``root_value_segment``.
                                # We replace the data at the corresponding locations with the latest search results to maintain the most up-to-date targets.
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


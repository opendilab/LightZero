import copy
from typing import Any, List, Tuple

import numpy as np
import torch
from easydict import EasyDict

from ding.utils import BUFFER_REGISTRY
from lzero.mcts.tree_search.mcts_ctree_sampled import SampledEfficientZeroMCTSCtree as MCTSCtree
from lzero.mcts.tree_search.mcts_ptree_sampled import SampledEfficientZeroMCTSPtree as MCTSPtree
from lzero.mcts.utils import prepare_observation_list
from lzero.policy import to_detach_cpu_numpy, concat_output, concat_output_value, inverse_scalar_transform
from .game_buffer_efficientzero import EfficientZeroGameBuffer


@BUFFER_REGISTRY.register('game_buffer_sampled_efficientzero')
class SampledEfficientZeroGameBuffer(EfficientZeroGameBuffer):
    """
    Overview:
        The specific game buffer for Sampled EfficientZero policy.
    """

    @classmethod
    def default_config(cls: type) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    # the default_config for SampledEfficientZeroGameBuffer.
    config = dict(
        model=dict(
            action_space_size=6,
            continuous_action_space=False,
            num_of_sampled_actions=6,
            representation_network_type='conv_res_blocks',  # options={'conv_res_blocks', 'identity'}
        ),
        # learn_mode config
        learn=dict(
            # (int) How many samples in a training batch
            batch_size=256,
        ),
        # ==============================================================
        # begin of additional game_config
        # ==============================================================
        ## common
        mcts_ctree=True,
        device='cuda',
        env_type='not_board_games',
        # the size/capacity of replay_buffer, in the terms of transitions.
        replay_buffer_size=int(1e6),

        ## learn
        num_simulations=50,
        td_steps=5,
        num_unroll_steps=5,
        lstm_horizon_len=5,

        ## reanalyze
        reanalyze_ratio=0.3,
        reanalyze_outdated=True,
        # whether to use root value in reanalyzing part
        use_root_value=False,
        mini_infer_size=256,

        ## priority
        use_priority=True,
        use_max_priority_for_new_data=True,
        # how much prioritization is used: 0 means no prioritization while 1 means full prioritization
        priority_prob_alpha=0.6,
        # how much correction is used: 0 means no correction while 1 means full correction
        priority_prob_beta=0.4,
        prioritized_replay_eps=1e-6,

        ## UCB
        root_dirichlet_alpha=0.3,
        root_exploration_fraction=0.25,
        pb_c_base=19652,
        pb_c_init=1.25,
        discount_factor=0.997,
        value_delta_max=0.01,
        # ==============================================================
        # end of additional game_config
        # ==============================================================
    )

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

        self.game_segment_buffer = []
        self.game_pos_priorities = []
        self.game_segment_game_pos_look_up = []

        self.keep_ratio = 1
        self._eps_collected = 0
        self.base_idx = 0
        self.clear_time = 0

    def sample(self, batch_size: int, policy: Any) -> List[Any]:
        """
        Overview:
            sample data from ``GameBuffer`` and prepare the current and target batch for training
        Arguments:
            - batch_size (:obj:`int`): batch size
            - policy (:obj:`torch.tensor`): model of policy
        Returns:
            - train_data (:obj:`List`): List of train data
        """

        policy._target_model.to(self._cfg.device)
        policy._target_model.eval()

        reward_value_context, policy_re_context, policy_non_re_context, current_batch = self._make_batch(batch_size,
                                                                                                         self._cfg.reanalyze_ratio)

        # target reward, target value
        batch_value_prefixs, batch_target_values = self._compute_target_reward_value(reward_value_context,
                                                                                     policy._target_model)

        batch_target_policies_non_re = self._compute_target_policy_non_reanalyzed(policy_non_re_context,
                                                                                  self._cfg.model.num_of_sampled_actions)

        if self._cfg.reanalyze_ratio > 0:
            # target policy
            batch_target_policies_re, root_sampled_actions = self._compute_target_policy_reanalyzed(policy_re_context,
                                                                                                    policy._target_model)
            # ==============================================================
            # fix reanalyze in sez:
            # use the latest root_sampled_actions after the reanalyze process,
            # because the batch_target_policies_re is corresponding to the latest root_sampled_actions
            # ==============================================================

            assert (self._cfg.reanalyze_ratio > 0 and self._cfg.reanalyze_outdated is True), \
                "in sampled effiicientzero, if self._cfg.reanalyze_ratio>0, you must set self._cfg.reanalyze_outdated=True"
            # current_batch = [obs_list, action_list, root_sampled_actions_list, mask_list, batch_index_list, weights_list, make_time_list]
            if self._cfg.model.continuous_action_space:
                current_batch[2][:int(batch_size * self._cfg.reanalyze_ratio)] = root_sampled_actions.reshape(
                    int(batch_size * self._cfg.reanalyze_ratio), self._cfg.num_unroll_steps + 1,
                    self._cfg.model.num_of_sampled_actions, self._cfg.model.action_space_size)
            else:
                current_batch[2][:int(batch_size * self._cfg.reanalyze_ratio)] = root_sampled_actions.reshape(
                    int(batch_size * self._cfg.reanalyze_ratio), self._cfg.num_unroll_steps + 1,
                    self._cfg.model.num_of_sampled_actions, 1)

        if 0 < self._cfg.reanalyze_ratio < 1:
            try:
                batch_target_policies = np.concatenate([batch_target_policies_re, batch_target_policies_non_re])
            except Exception as error:
                print(error)
        elif self._cfg.reanalyze_ratio == 1:
            batch_target_policies = batch_target_policies_re
        elif self._cfg.reanalyze_ratio == 0:
            batch_target_policies = batch_target_policies_non_re

        targets_batch = [batch_value_prefixs, batch_target_values, batch_target_policies]
        # a batch contains the current_batch and the targets_batch
        train_data = [current_batch, targets_batch]
        return train_data

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
        orig_data = self._sample_orig_data(batch_size)
        game_lst, pos_in_game_segment_list, batch_index_list, weights_list, make_time_list = orig_data
        batch_size = len(batch_index_list)
        obs_list, action_list, mask_list = [], [], []
        root_sampled_actions_list = []
        # prepare the inputs of a batch
        for i in range(batch_size):
            game = game_lst[i]
            pos_in_game_segment = pos_in_game_segment_list[i]
            # ==============================================================
            # sampled related core code
            # ==============================================================
            actions_tmp = game.action_segment[
                          pos_in_game_segment:pos_in_game_segment + self._cfg.num_unroll_steps].tolist()

            # NOTE: self._cfg.num_unroll_steps + 1
            root_sampled_actions_tmp = game.root_sampled_actions[
                                       pos_in_game_segment:pos_in_game_segment + self._cfg.num_unroll_steps + 1]

            mask_tmp = [1. for i in range(len(root_sampled_actions_tmp))]
            mask_tmp += [0. for _ in range(self._cfg.num_unroll_steps + 1 - len(mask_tmp))]

            # pad random action
            if self._cfg.model.continuous_action_space:
                actions_tmp += [
                    np.random.randn(self._cfg.model.action_space_size)
                    for _ in range(self._cfg.num_unroll_steps - len(actions_tmp))
                ]
                root_sampled_actions_tmp += [
                    np.random.rand(self._cfg.model.num_of_sampled_actions, self._cfg.model.action_space_size)
                    for _ in range(self._cfg.num_unroll_steps + 1 - len(root_sampled_actions_tmp))
                ]
            else:
                actions_tmp += [
                    np.random.randint(0, self._cfg.model.action_space_size, 1).item()
                    for _ in range(self._cfg.num_unroll_steps - len(actions_tmp))
                ]
                if len(root_sampled_actions_tmp[0].shape) == 1:
                    root_sampled_actions_tmp += [
                        np.arange(self._cfg.model.action_space_size)
                        # NOTE: self._cfg.num_unroll_steps + 1
                        for _ in range(self._cfg.num_unroll_steps + 1 - len(root_sampled_actions_tmp))
                    ]
                else:
                    root_sampled_actions_tmp += [
                        np.random.randint(0, self._cfg.model.action_space_size,
                                          self._cfg.model.num_of_sampled_actions).reshape(
                            self._cfg.model.num_of_sampled_actions, 1
                        )  # NOTE: self._cfg.num_unroll_steps + 1
                        for _ in range(self._cfg.num_unroll_steps + 1 - len(root_sampled_actions_tmp))
                    ]

            # obtain the input observations
            # stack+num_unroll_steps  4+5
            # pad if length of obs in game_segment is less than stack+num_unroll_steps
            obs_list.append(
                game_lst[i].get_obs(pos_in_game_segment_list[i], num_unroll_steps=self._cfg.num_unroll_steps,
                                    padding=True))
            action_list.append(actions_tmp)
            root_sampled_actions_list.append(root_sampled_actions_tmp)

            mask_list.append(mask_tmp)

        # formalize the input observations
        obs_list = prepare_observation_list(obs_list)
        # ==============================================================
        # sampled related core code
        # ==============================================================
        # formalize the inputs of a batch
        current_batch = [obs_list, action_list, root_sampled_actions_list, mask_list, batch_index_list, weights_list,
                         make_time_list]

        for i in range(len(current_batch)):
            current_batch[i] = np.asarray(current_batch[i])

        total_transitions = self.get_num_of_transitions()

        # obtain the context of value targets
        reward_value_context = self._prepare_reward_value_context(
            batch_index_list, game_lst, pos_in_game_segment_list, total_transitions
        )

        """
        only reanalyze recent reanalyze_ratio (e.g. 50%) data
        if self._cfg.reanalyze_outdated is True, batch_index_list is sorted according to its generated env_steps
        0: reanalyze_num -> reanalyzed policy, reanalyze_num:end -> non reanalyzed policy
        """
        reanalyze_num = int(batch_size * reanalyze_ratio)
        # reanalyzed policy
        if reanalyze_num > 0:
            # obtain the context of reanalyzed policy targets
            policy_re_context = self._prepare_policy_reanalyzed_context(
                batch_index_list[:reanalyze_num], game_lst[:reanalyze_num], pos_in_game_segment_list[:reanalyze_num]
            )
        else:
            policy_re_context = None

        # non reanalyzed policy
        if reanalyze_num < batch_size:
            # obtain the context of non-reanalyzed policy targets
            policy_non_re_context = self._prepare_policy_non_reanalyzed_context(
                batch_index_list[reanalyze_num:], game_lst[reanalyze_num:], pos_in_game_segment_list[reanalyze_num:]
            )
        else:
            policy_non_re_context = None

        context = reward_value_context, policy_re_context, policy_non_re_context, current_batch
        return context

    def _compute_target_reward_value(self, reward_value_context: List[Any], model: Any) -> List[np.ndarray]:
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
        value_obs_list, value_mask, pos_in_game_segment_list, rewards_list, game_segment_lens, td_steps_list, action_mask_segment, \
        to_play_segment = reward_value_context

        # transition_batch_size = game_segment_batch_size * (num_unroll_steps+1)
        transition_batch_size = len(value_obs_list)
        game_segment_batch_size = len(pos_in_game_segment_list)

        to_play, action_mask = self._preprocess_to_play_and_action_mask(game_segment_batch_size, to_play_segment,
                                                                        action_mask_segment, pos_in_game_segment_list)

        batch_target_values, batch_value_prefixs = [], []
        with torch.no_grad():
            value_obs_list = prepare_observation_list(value_obs_list)
            # split a full batch into slices of mini_infer_size: to save the GPU memory for more GPU actors
            slices = int(np.ceil(transition_batch_size / self._cfg.mini_infer_size))
            network_output = []
            for i in range(slices):
                beg_index = self._cfg.mini_infer_size * i
                end_index = self._cfg.mini_infer_size * (i + 1)
                m_obs = torch.from_numpy(value_obs_list[beg_index:end_index]).to(self._cfg.device).float()

                # calculate the target value
                m_output = model.initial_inference(m_obs)

                # TODO(pu)
                if not model.training:
                    # if not in training, obtain the scalars of the value/reward
                    [m_output.hidden_state, m_output.value, m_output.policy_logits] = to_detach_cpu_numpy(
                        [m_output.hidden_state, inverse_scalar_transform(m_output.value, self._cfg.model.support_scale),
                         m_output.policy_logits])
                    m_output.reward_hidden_state = (
                        m_output.reward_hidden_state[0].detach().cpu().numpy(),
                        m_output.reward_hidden_state[1].detach().cpu().numpy()
                    )

                network_output.append(m_output)

            # concat the output slices after model inference
            if self._cfg.use_root_value:
                # use the root values from MCTS
                # the root values have limited improvement but require much more GPU actors;
                _, value_prefix_pool, policy_logits_pool, hidden_state_roots, reward_hidden_state_roots = concat_output(
                    network_output
                )
                value_prefix_pool = value_prefix_pool.squeeze().tolist()
                policy_logits_pool = policy_logits_pool.tolist()

                if self._cfg.mcts_ctree:
                    # cpp mcts_tree
                    if to_play_segment[0][0] in [None, -1]:
                        # we use to_play=-1 means play_with_bot_mode game
                        to_play = [-1 for _ in range(transition_batch_size)]
                        # NOTE: in continuous action space env: all legal_actions is -1
                        legal_actions = [
                            [-1 for _ in range(self._cfg.model.action_space_size)] for _ in range(transition_batch_size)
                        ]
                    else:
                        legal_actions = [[i for i, x in enumerate(action_mask[j]) if x == 1] for j in
                                         range(transition_batch_size)]

                    roots = MCTSCtree.Roots(
                        transition_batch_size, legal_actions, self._cfg.model.action_space_size,
                        self._cfg.model.num_of_sampled_actions, self._cfg.model.continuous_action_space
                    )
                    noises = [
                        np.random.dirichlet([self._cfg.root_dirichlet_alpha] * self._cfg.model.num_of_sampled_actions
                                            ).astype(np.float32).tolist() for _ in range(transition_batch_size)
                    ]
                    roots.prepare(
                        self._cfg.root_exploration_fraction, noises, value_prefix_pool, policy_logits_pool, to_play
                    )
                    # do MCTS for a new policy with the recent target model
                    MCTSCtree(self._cfg).search(roots, model, hidden_state_roots, reward_hidden_state_roots, to_play)
                else:
                    # python mcts_tree
                    if to_play_segment[0][0] in [None, -1]:
                        # for one_player atari games
                        action_mask = [
                            list(np.ones(self._cfg.model.action_space_size, dtype=np.int8)) for _ in
                            range(transition_batch_size)
                        ]

                    legal_actions = [[i for i, x in enumerate(action_mask[j]) if x == 1] for j in
                                     range(transition_batch_size)]
                    roots = MCTSPtree.Roots(
                        transition_batch_size, legal_actions,
                        num_of_sampled_actions=self._cfg.model.num_of_sampled_actions
                    )
                    noises = [
                        np.random.dirichlet([self._cfg.root_dirichlet_alpha] * int(sum(action_mask[j]))
                                            ).astype(np.float32).tolist() for j in range(transition_batch_size)
                    ]

                    if to_play_segment[0][0] in [None, -1]:
                        roots.prepare(
                            self._cfg.root_exploration_fraction,
                            noises,
                            value_prefix_pool,
                            policy_logits_pool,
                            to_play=-1
                        )
                        # do MCTS for a new policy with the recent target model
                        MCTSPtree.Roots(self._cfg).search(
                            roots, model, hidden_state_roots, reward_hidden_state_roots, to_play=-1
                        )
                    else:
                        roots.prepare(
                            self._cfg.root_exploration_fraction,
                            noises,
                            value_prefix_pool,
                            policy_logits_pool,
                            to_play=to_play
                        )
                        # do MCTS for a new policy with the recent target model
                        MCTSPtree.Roots(self._cfg).search(
                            roots, model, hidden_state_roots, reward_hidden_state_roots, to_play=to_play
                        )

                roots_values = roots.get_values()
                value_list = np.array(roots_values)
            else:
                # use the predicted values
                value_list = concat_output_value(network_output)

            # get last state value
            if to_play_segment[0][0] in [1, 2]:
                # TODO(pu): for board_games, very important, to check
                value_list = value_list.reshape(-1) * np.array([self._cfg.discount_factor ** td_steps_list[i]
                                                                if int(td_steps_list[i]) % 2 == 0
                                                                else -self._cfg.discount_factor ** td_steps_list[i]
                                                                for i in range(transition_batch_size)])
            else:
                value_list = value_list.reshape(-1) * (
                        np.array([self._cfg.discount_factor for _ in range(transition_batch_size)]) ** td_steps_list
                )

            value_list = value_list * np.array(value_mask)
            value_list = value_list.tolist()

            horizon_id, value_index = 0, 0
            for game_segment_len_non_re, reward_list, state_index, to_play_list in zip(game_segment_lens, rewards_list,
                                                                                       pos_in_game_segment_list,
                                                                                       to_play_segment):
                target_values = []
                target_value_prefixs = []

                value_prefix = 0.0
                base_index = state_index
                for current_index in range(state_index, state_index + self._cfg.num_unroll_steps + 1):
                    bootstrap_index = current_index + td_steps_list[value_index]
                    # for i, reward in enumerate(game.rewards[current_index:bootstrap_index]):
                    for i, reward in enumerate(reward_list[current_index:bootstrap_index]):
                        if to_play_segment[0][0] in [1, 2]:
                            # TODO(pu): for board_games, very important, to check
                            if to_play_list[current_index] == to_play_list[i]:
                                value_list[value_index] += reward * self._cfg.discount_factor ** i
                            else:
                                value_list[value_index] += -reward * self._cfg.discount_factor ** i
                        else:
                            value_list[value_index] += reward * self._cfg.discount_factor ** i
                            # TODO(pu): why value don't use discount_factor factor

                    # reset every lstm_horizon_len
                    if horizon_id % self._cfg.lstm_horizon_len == 0:
                        value_prefix = 0.0
                        base_index = current_index
                    horizon_id += 1

                    if current_index < game_segment_len_non_re:
                        target_values.append(value_list[value_index])
                        # Since the horizon is small and the discount_factor is close to 1.
                        # Compute the reward sum to approximate the value prefix for simplification
                        value_prefix += reward_list[
                            current_index]  # * config.discount_factor ** (current_index - base_index)
                        target_value_prefixs.append(value_prefix)
                    else:
                        target_values.append(0)
                        target_value_prefixs.append(value_prefix)

                    value_index += 1

                batch_value_prefixs.append(target_value_prefixs)
                batch_target_values.append(target_values)

        batch_value_prefixs = np.asarray(batch_value_prefixs, dtype=object)
        batch_target_values = np.asarray(batch_target_values, dtype=object)

        return batch_value_prefixs, batch_target_values

    def _compute_target_policy_reanalyzed(self, policy_re_context: List[Any], model: Any) -> np.ndarray:
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

        policy_obs_list, policy_mask, pos_in_game_segment_list, batch_index_list, child_visits, game_segment_lens, action_mask_segment, \
        to_play_segment = policy_re_context
        # transition_batch_size = game_segment_batch_size * (self._cfg.num_unroll_steps + 1)
        transition_batch_size = len(policy_obs_list)
        game_segment_batch_size = len(pos_in_game_segment_list)

        to_play, action_mask = self._preprocess_to_play_and_action_mask(game_segment_batch_size, to_play_segment,
                                                                        action_mask_segment, pos_in_game_segment_list)

        with torch.no_grad():
            policy_obs_list = prepare_observation_list(policy_obs_list)
            # split a full batch into slices of mini_infer_size: to save the GPU memory for more GPU actors
            self._cfg.mini_infer_size = self._cfg.mini_infer_size
            slices = np.ceil(transition_batch_size / self._cfg.mini_infer_size).astype(np.int_)
            network_output = []
            for i in range(slices):
                beg_index = self._cfg.mini_infer_size * i
                end_index = self._cfg.mini_infer_size * (i + 1)
                m_obs = torch.from_numpy(policy_obs_list[beg_index:end_index]).to(self._cfg.device).float()

                m_output = model.initial_inference(m_obs)

                if not model.training:
                    # if not in training, obtain the scalars of the value/reward
                    [m_output.hidden_state, m_output.value, m_output.policy_logits] = to_detach_cpu_numpy(
                        [m_output.hidden_state, inverse_scalar_transform(m_output.value, self._cfg.model.support_scale),
                         m_output.policy_logits])
                    m_output.reward_hidden_state = (
                        m_output.reward_hidden_state[0].detach().cpu().numpy(),
                        m_output.reward_hidden_state[1].detach().cpu().numpy()
                    )

                network_output.append(m_output)

            _, value_prefix_pool, policy_logits_pool, hidden_state_roots, reward_hidden_state_roots = concat_output(
                network_output
            )

            value_prefix_pool = value_prefix_pool.squeeze().tolist()
            policy_logits_pool = policy_logits_pool.tolist()
            if self._cfg.mcts_ctree:
                # ==============================================================
                # sampled related core code
                # ==============================================================
                # cpp mcts_tree
                if to_play_segment[0][0] in [None, -1]:
                    # we use to_play=-1 means play_with_bot_mode game
                    to_play = [-1 for i in range(transition_batch_size)]
                    # NOTE: in continuous action space env: all legal_actions is -1
                    legal_actions = [[-1 for _ in range(self._cfg.model.action_space_size)] for _ in
                                     range(transition_batch_size)]
                else:
                    legal_actions = [[i for i, x in enumerate(action_mask[j]) if x == 1] for j in
                                     range(transition_batch_size)]

                roots = MCTSCtree.Roots(
                    transition_batch_size, legal_actions, self._cfg.model.action_space_size,
                    self._cfg.model.num_of_sampled_actions, self._cfg.model.continuous_action_space
                )
                noises = [
                    np.random.dirichlet([self._cfg.root_dirichlet_alpha] * self._cfg.model.num_of_sampled_actions
                                        ).astype(np.float32).tolist() for _ in range(transition_batch_size)
                ]
                roots.prepare(
                    self._cfg.root_exploration_fraction, noises, value_prefix_pool, policy_logits_pool, to_play
                )
                # do MCTS for a new policy with the recent target model
                MCTSCtree(self._cfg).search(roots, model, hidden_state_roots, reward_hidden_state_roots, to_play)
            else:
                # python mcts_tree
                if to_play_segment[0][0] in [None, -1]:
                    # we use to_play=-1 means play_with_bot_mode game in mcts_ptree
                    to_play = [-1 for i in range(transition_batch_size)]
                    # NOTE: in continuous action space env: all legal_actions is -1
                    legal_actions = [
                        [-1 for _ in range(self._cfg.model.num_of_sampled_actions)] for _ in
                        range(transition_batch_size)
                    ]
                else:
                    # if board_games, we have action_mask
                    legal_actions = [[i for i, x in enumerate(action_mask[j]) if x == 1] for j in
                                     range(transition_batch_size)]

                roots = MCTSPtree.Roots(
                    transition_batch_size,
                    legal_actions,
                    action_space_size=self._cfg.model.action_space_size,
                    num_of_sampled_actions=self._cfg.model.num_of_sampled_actions,
                    continuous_action_space=self._cfg.model.continuous_action_space
                )
                noises = [
                    np.random.dirichlet([self._cfg.root_dirichlet_alpha] * self._cfg.model.num_of_sampled_actions
                                        ).astype(np.float32).tolist() for _ in range(transition_batch_size)
                ]
                roots.prepare(
                    self._cfg.root_exploration_fraction, noises, value_prefix_pool, policy_logits_pool, to_play
                )
                # do MCTS for a new policy with the recent target model
                MCTSPtree.Roots(self._cfg).search(roots, model, hidden_state_roots, reward_hidden_state_roots, to_play)

                roots_legal_actions_list = roots.legal_actions_list

            roots_distributions = roots.get_distributions()

            # ==============================================================
            # fix reanalyze in sez
            # ==============================================================
            roots_sampled_actions = roots.get_sampled_actions()
            try:
                root_sampled_actions = np.array([action.value for action in roots_sampled_actions])
            except Exception as error:
                root_sampled_actions = np.array([action for action in roots_sampled_actions])

            policy_index = 0
            for state_index, game_idx in zip(pos_in_game_segment_list, batch_index_list):
                target_policies = []
                for current_index in range(state_index, state_index + self._cfg.num_unroll_steps + 1):
                    distributions = roots_distributions[policy_index]
                    # ==============================================================
                    # sampled related core code
                    # ==============================================================
                    if policy_mask[policy_index] == 0:
                        # the null padding target policy
                        target_policies.append([0 for _ in range(self._cfg.model.num_of_sampled_actions)])
                    else:
                        if distributions is None:
                            # if at some obs, the legal_action is None, then add the fake target_policy
                            target_policies.append(
                                list(
                                    np.ones(self._cfg.model.num_of_sampled_actions) /
                                    self._cfg.model.num_of_sampled_actions
                                )
                            )
                        else:
                            if self._cfg.mcts_ctree:
                                ## cpp mcts_tree
                                if to_play_segment[0][0] in [None, -1]:
                                    sum_visits = sum(distributions)
                                    policy = [visit_count / sum_visits for visit_count in distributions]
                                    target_policies.append(policy)
                                else:
                                    # for two_player board games
                                    policy_tmp = [0 for _ in range(self._cfg.model.num_of_sampled_actions)]
                                    # to make sure target_policies have the same dimension
                                    sum_visits = sum(distributions)
                                    policy = [visit_count / sum_visits for visit_count in distributions]
                                    for index, legal_action in enumerate(roots_legal_actions_list[policy_index]):
                                        policy_tmp[legal_action] = policy[index]
                                    target_policies.append(policy_tmp)
                            else:
                                # python mcts_tree
                                if to_play_segment[0][0] in [None, -1]:
                                    # TODO(pu): very important
                                    sum_visits = sum(distributions)
                                    policy = [visit_count / sum_visits for visit_count in distributions]
                                    target_policies.append(policy)
                                else:
                                    # for two_player board games
                                    policy_tmp = [0 for _ in range(self._cfg.model.num_of_sampled_actions)]
                                    # to make sure target_policies have the same dimension
                                    sum_visits = sum(distributions)
                                    policy = [visit_count / sum_visits for visit_count in distributions]
                                    for index, legal_action in enumerate(roots_legal_actions_list[policy_index]):
                                        policy_tmp[legal_action] = policy[index]
                                    target_policies.append(policy_tmp)

                    policy_index += 1

                batch_target_policies_re.append(target_policies)

        batch_target_policies_re = np.array(batch_target_policies_re)

        return batch_target_policies_re, root_sampled_actions

    def update_priority(self, train_data: List[np.ndarray],
                        batch_priorities: Any) -> None:
        """
        Overview:
            Update the priority of training data.
        Arguments:
            - train_data (:obj:`Optional[List[Optional[np.ndarray]]]`): training data to be updated priority.
            - batch_priorities (:obj:`batch_priorities`): priorities to update to.
        NOTE:
            train_data = [current_batch, targets_batch]
            current_batch = [obs_list, action_list, root_sampled_actions_list, mask_list, batch_index_list, weights_list, make_time_list]
        """

        batch_index_list = train_data[0][4]
        metas = {'make_time': train_data[0][6], 'batch_priorities': batch_priorities}
        # only update the priorities for data still in replay buffer
        for i in range(len(batch_index_list)):
            if metas['make_time'][i] > self.clear_time:
                idx, prio = batch_index_list[i], metas['batch_priorities'][i]
                self.game_pos_priorities[idx] = prio

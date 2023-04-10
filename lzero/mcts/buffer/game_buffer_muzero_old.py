import copy
import itertools
import logging
import random
import time
from typing import Any, List, Optional, Union

import numpy as np
import torch
from ding.data.buffer import Buffer
from ding.torch_utils.data_helper import to_ndarray, to_list
from ding.utils import BUFFER_REGISTRY
from easydict import EasyDict

# python mcts_tree
import lzero.mcts.ptree.ptree_mz as ptree
# cpp mcts_tree
from lzero.mcts.ctree.ctree_muzero import mz_tree as ctree
from lzero.mcts.tree_search.mcts_ctree import MuZeroMCTSCtree as MCTS_ctree
from lzero.mcts.tree_search.mcts_ptree import MuZeroMCTSPtree as MCTS_ptree
from lzero.mcts.utils import BufferedData
from lzero.mcts.utils import prepare_observation_list
from lzero.policy import to_detach_cpu_numpy, concat_output, concat_output_value, inverse_scalar_transform


@BUFFER_REGISTRY.register('game_buffer_muzero')
class MuZeroGameBuffer(Buffer):
    """
    Overview:
        The specific game buffer for MuZero policy.
    """

    @classmethod
    def default_config(cls: type) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    # the default_config for MuZeroGameBuffer.
    config = dict(
        model=dict(
            action_space_size=6,
            representation_network_type='conv_res_blocks',  # options={'conv_res_blocks', 'identity'}
            frame_stack_num=4,
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
        super().__init__(cfg.replay_buffer_size)
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
        self.keep_ratio = 1

        self.model_index = 0
        self.model_update_interval = 10

        self.game_block_buffer = []
        self.game_pos_priorities = []
        self.game_block_game_pos_look_up = []

        self._eps_collected = 0
        self.base_idx = 0
        self._alpha = self._cfg.priority_prob_alpha
        self.clear_time = 0

    def sample(self, batch_size, policy):
        """
        Overview:
            sample data from ``GameBuffer`` and prepare the current and target batch for training.
        Arguments:
            - batch_size (:obj:`int`): batch size
            - policy (:obj:`torch.tensor`): model of policy
        Returns:
            - train_data (:obj:`List`): List of train data
        """
        policy._target_model.to(self._cfg.device)
        policy._target_model.eval()

        batch_context = self.prepare_batch_context(batch_size, self._cfg.priority_prob_beta)
        input_context = self.make_batch(batch_context, self._cfg.reanalyze_ratio)
        reward_value_context, policy_re_context, policy_non_re_context, current_batch = input_context

        # target reward, value
        batch_rewards, batch_target_values = self.compute_target_reward_value(reward_value_context,
                                                                              policy._target_model)
        # target policy
        batch_target_policies_re = self.compute_target_policy_reanalyzed(policy_re_context, policy._target_model)
        batch_target_policies_non_re = self.compute_target_policy_non_reanalyzed(policy_non_re_context)

        if 0 < self._cfg.reanalyze_ratio < 1:
            batch_policies = np.concatenate([batch_target_policies_re, batch_target_policies_non_re])
        elif self._cfg.reanalyze_ratio == 1:
            batch_policies = batch_target_policies_re
        elif self._cfg.reanalyze_ratio == 0:
            batch_policies = batch_target_policies_non_re

        targets_batch = [batch_rewards, batch_target_values, batch_policies]
        # a batch contains the inputs and the targets
        train_data = [current_batch, targets_batch]
        return train_data

    def make_batch(self, batch_context, ratio):
        """
        Overview:
            prepare the context of a batch
            reward_value_context:        the context of reanalyzed value targets
            policy_re_context:           the context of reanalyzed policy targets
            policy_non_re_context:       the context of non-reanalyzed policy targets
            current_batch:                the inputs of batch
        Arguments:
            batch_context: Any batch context from replay buffer
            ratio: float ratio of reanalyzed policy (value is 100% reanalyzed)
        Returns:
            - context (:obj:`Tuple`): reward_value_context, policy_re_context, policy_non_re_context, current_batch
        """
        # obtain the batch context from replay buffer
        game_block_list, pos_in_game_block_list, batch_index_list, weights_list, make_time_list = batch_context
        batch_size = len(batch_index_list)
        obs_list, action_list, mask_list = [], [], []
        # prepare the inputs of a batch
        for i in range(batch_size):
            game = game_block_list[i]
            pos_in_game_block = pos_in_game_block_list[i]

            actions_tmp = game.action_segment[pos_in_game_block:pos_in_game_block +
                                                                self._cfg.num_unroll_steps].tolist()
            # add mask for invalid actions (out of trajectory)
            mask_tmp = [1. for i in range(len(actions_tmp))]
            mask_tmp += [0. for _ in range(self._cfg.num_unroll_steps - len(mask_tmp))]

            # pad random action
            actions_tmp += [
                np.random.randint(0, game.action_space_size) for _ in
                range(self._cfg.num_unroll_steps - len(actions_tmp))
            ]

            # obtain the input observations
            # stack+num_unroll_steps  4+5
            # pad if length of obs in game_block is less than stack+num_unroll_steps
            obs_list.append(
                game_block_list[i].get_unroll_obs(
                    pos_in_game_block_list[i], num_unroll_steps=self._cfg.num_unroll_steps, padding=True
                )
            )
            action_list.append(actions_tmp)
            mask_list.append(mask_tmp)

        # formalize the input observations
        obs_list = prepare_observation_list(obs_list)

        # formalize the inputs of a batch
        current_batch = [obs_list, action_list, mask_list, batch_index_list, weights_list, make_time_list]
        for i in range(len(current_batch)):
            current_batch[i] = np.asarray(current_batch[i])

        total_transitions = self.get_num_of_transitions()

        # obtain the context of value targets
        reward_value_context = self.prepare_reward_value_context(
            batch_index_list, game_block_list, pos_in_game_block_list, total_transitions
        )
        """
        only reanalyze recent ratio (e.g. 50%) data
        """
        reanalyze_num = int(batch_size * ratio)
        # if self._cfg.reanalyze_outdated is True:
        # batch_index_list is sorted according to its generated enn_steps

        # 0:reanalyze_num -> reanalyzed policy, reanalyze_num: end -> non reanalyzed policy
        # reanalyzed policy
        if reanalyze_num > 0:
            # obtain the context of reanalyzed policy targets
            policy_re_context = self.prepare_policy_reanalyzed_context(
                batch_index_list[:reanalyze_num], game_block_list[:reanalyze_num],
                pos_in_game_block_list[:reanalyze_num]
            )
        else:
            policy_re_context = None

        # non reanalyzed policy
        if reanalyze_num < batch_size:
            # obtain the context of non-reanalyzed policy targets
            policy_non_re_context = self.prepare_policy_non_reanalyzed_context(
                batch_index_list[reanalyze_num:], game_block_list[reanalyze_num:],
                pos_in_game_block_list[reanalyze_num:]
            )
        else:
            policy_non_re_context = None

        context = reward_value_context, policy_re_context, policy_non_re_context, current_batch
        return context

    def prepare_batch_context(self, batch_size, beta):
        """
        Overview:
            Prepare a batch context that contains:
            game_block_list: a list of game histories
            pos_in_game_block_list: transition index in game (relative index)
            batch_index_list: the index of start transition of sampled minibatch in replay buffer
            weights_list: the weight concerning the priority
            make_time: the time the batch is made (for correctly updating replay buffer
                when data is deleted)
        Arguments:
            - batch_size: int batch size
            - beta: float the parameter in PER for calculating the priority
        """
        assert beta > 0

        # total number of transitions
        total = self.get_num_of_transitions()

        if self._cfg.use_priority is False:
            self.game_pos_priorities = np.ones_like(self.game_pos_priorities)

        # +1e-6 for numerical stability
        probs = self.game_pos_priorities ** self._alpha + 1e-6

        probs /= probs.sum()
        # TODO(pu): sample data in PER way
        # sample according to transition index
        # TODO(pu): replace=True
        # batch_index_list = np.random.choice(total, batch_size, p=probs, replace=True)
        batch_index_list = np.random.choice(total, batch_size, p=probs, replace=False)

        # TODO
        if self._cfg.reanalyze_outdated is True:
            batch_index_list.sort()

        weights_list = (total * probs[batch_index_list]) ** (-beta)
        weights_list /= weights_list.max()

        game_block_list = []
        pos_in_game_block_list = []

        for idx in batch_index_list:
            game_block_idx, pos_in_game_block = self.game_block_game_pos_look_up[idx]
            game_block_idx -= self.base_idx
            game_block = self.game_block_buffer[game_block_idx]

            game_block_list.append(game_block)
            pos_in_game_block_list.append(pos_in_game_block)

        make_time = [time.time() for _ in range(len(batch_index_list))]

        context = (game_block_list, pos_in_game_block_list, batch_index_list, weights_list, make_time)
        return context

    def prepare_reward_value_context(
            self, batch_index_list, game_block_list, pos_in_game_block_list, total_transitions
    ):
        """
        Overview:
            prepare the context of rewards and values for calculating TD value target in reanalyzing part.
        Arguments:
            - batch_index_list (:obj:`list`): the index of start transition of sampled minibatch in replay buffer
            - game_block_list (:obj:`list`): list of game histories
            - pos_in_game_block_list (:obj:`list`): list of transition index in game_block
            - total_transitions (:obj:`int`): number of collected transitions
        Returns:
            - reward_value_context (:obj:`list`): value_obs_lst, value_mask, state_index_lst, rewards_lst, game_block_lens, 
              td_steps_lst, action_mask_segment, to_play_segment
        """
        zero_obs = game_block_list[0].zero_obs()
        value_obs_list = []
        # the value is valid or not (out of trajectory)
        value_mask = []
        rewards_list = []
        game_block_lens = []
        # for two_player board games
        action_mask_segment, to_play_segment = [], []

        td_steps_list = []
        for game_block, state_index, idx in zip(game_block_list, pos_in_game_block_list, batch_index_list):
            game_block_len = len(game_block)
            game_block_lens.append(game_block_len)

            # TODO(pu):
            # for atari
            # # off-policy correction: shorter horizon of td steps
            # delta_td = (total_transitions - idx) // config.auto_td_steps
            # td_steps = config.td_steps - delta_td
            # td_steps = np.clip(td_steps, 1, 5).astype(np.int)
            td_steps = np.clip(self._cfg.td_steps, 1, max(1, game_block_len - state_index)).astype(np.int32)

            # prepare the corresponding observations for bootstrapped values o_{t+k}
            # o[t+ td_steps, t + td_steps + stack frames + num_unroll_steps]
            # t=2+3 -> o[2+3, 2+3+4+5] -> o[5, 14]
            game_obs = game_block.get_unroll_obs(state_index + td_steps, self._cfg.num_unroll_steps)

            rewards_list.append(game_block.reward_segment)

            # for two_player board games
            action_mask_segment.append(game_block.action_mask_segment)
            to_play_segment.append(game_block.to_play_segment)

            for current_index in range(state_index, state_index + self._cfg.num_unroll_steps + 1):
                # get the <num_unroll_steps+1>  bootstrapped target obs
                td_steps_list.append(td_steps)
                # index of bootstrapped obs o_{t+td_steps}
                bootstrap_index = current_index + td_steps

                if bootstrap_index < game_block_len:
                    value_mask.append(1)
                    # beg_index = bootstrap_index - (state_index + td_steps)
                    # max of beg_index is num_unroll_steps
                    beg_index = current_index - state_index
                    end_index = beg_index + self._cfg.model.frame_stack_num
                    # the stacked obs in time t
                    obs = game_obs[beg_index:end_index]
                else:
                    value_mask.append(0)
                    obs = zero_obs

                value_obs_list.append(obs)

        reward_value_context = [
            value_obs_list, value_mask, pos_in_game_block_list, rewards_list, game_block_lens, td_steps_list,
            action_mask_segment, to_play_segment
        ]
        return reward_value_context

    def prepare_policy_non_reanalyzed_context(self, batch_index_list, game_block_list, pos_in_game_block_list):
        """
        Overview:
            prepare the context of policies for calculating policy target in non-reanalyzing part, just return the policy in self-play
        Arguments:
            - batch_index_list (:obj:`list`): the index of start transition of sampled minibatch in replay buffer
            - game_block_list (:obj:`list`): list of game histories
            - pos_in_game_block_list (:obj:`list`): list transition index in game
        Returns:
            - policy_non_re_context (:obj:`list`): state_index_lst, child_visits, game_block_lens, action_mask_segment, to_play_segment
        """
        child_visits = []
        game_block_lens = []
        # for two_player board games
        action_mask_segment, to_play_segment = [], []

        for game_block, state_index, idx in zip(game_block_list, pos_in_game_block_list, batch_index_list):
            game_block_len = len(game_block)
            game_block_lens.append(game_block_len)
            # for two_player board games
            action_mask_segment.append(game_block.action_mask_segment)
            to_play_segment.append(game_block.to_play_segment)

            child_visits.append(game_block.child_visit_segment)

        policy_non_re_context = [
            pos_in_game_block_list, child_visits, game_block_lens, action_mask_segment, to_play_segment
        ]
        return policy_non_re_context

    def prepare_policy_reanalyzed_context(self, batch_index_list, game_block_list, pos_in_game_block_list):
        """
        Overview:
            prepare the context of policies for calculating policy target in reanalyzing part.
        Arguments:
            - batch_index_list (:obj:'list'): start transition index in the replay buffer
            - game_block_list (:obj:'list'): list of game histories
            - pos_in_game_block_list (:obj:'list'): position of transition index in one game segment
        Returns:
            - policy_re_context (:obj:`list`): policy_obs_lst, policy_mask, state_index_lst, indices,
              child_visits, game_block_lens, action_mask_segment, to_play_segment
        """
        zero_obs = game_block_list[0].zero_obs()

        with torch.no_grad():
            # for policy
            policy_obs_list = []
            policy_mask = []  # 0 -> out of traj, 1 -> new policy
            rewards, child_visits, game_block_lens = [], [], []
            # for two_player board games
            action_mask_segment, to_play_segment = [], []
            for game_block, state_index in zip(game_block_list, pos_in_game_block_list):
                game_block_len = len(game_block)
                game_block_lens.append(game_block_len)
                rewards.append(game_block.reward_segment)
                # for two_player board games
                action_mask_segment.append(game_block.action_mask_segment)
                to_play_segment.append(game_block.to_play_segment)

                child_visits.append(game_block.child_visit_segment)
                # prepare the corresponding observations
                game_obs = game_block.get_unroll_obs(state_index, self._cfg.num_unroll_steps)
                for current_index in range(state_index, state_index + self._cfg.num_unroll_steps + 1):

                    if current_index < game_block_len:
                        policy_mask.append(1)
                        beg_index = current_index - state_index
                        end_index = beg_index + self._cfg.model.frame_stack_num
                        obs = game_obs[beg_index:end_index]
                    else:
                        policy_mask.append(0)
                        obs = zero_obs
                    policy_obs_list.append(obs)

        policy_re_context = [
            policy_obs_list, policy_mask, pos_in_game_block_list, batch_index_list, child_visits, game_block_lens,
            action_mask_segment, to_play_segment
        ]
        return policy_re_context

    def compute_target_reward_value(self, reward_value_context, model):
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
        value_obs_list, value_mask, pos_in_game_block_list, rewards_list, game_block_lens, td_steps_list, action_mask_segment, \
        to_play_segment = reward_value_context
        # game_block_batch_size * (num_unroll_steps+1)
        transition_batch_size = len(value_obs_list)
        game_block_batch_size = len(pos_in_game_block_list)

        if to_play_segment[0][0] in [1, 2]:
            # for two_player board games

            """
            prepare the to_play and action_mask for the target obs in ``value_obs_list``
                - to_play: {list: game_block_batch_size * (num_unroll_steps+1)}
                - action_mask: {list: game_block_batch_size * (num_unroll_steps+1)}
            """
            to_play = []
            for bs in range(game_block_batch_size):
                to_play_tmp = list(
                    to_play_segment[bs][pos_in_game_block_list[bs]:pos_in_game_block_list[bs] +
                                                                   self._cfg.num_unroll_steps + 1]
                )
                if len(to_play_tmp) < self._cfg.num_unroll_steps + 1:
                    # effective play index is {1,2}, for padding data, we set to_play=0
                    to_play_tmp += [1 for _ in range(self._cfg.num_unroll_steps + 1 - len(to_play_tmp))]
                to_play.append(to_play_tmp)
            to_play = sum(to_play, [])

            action_mask = []
            for bs in range(game_block_batch_size):
                action_mask_tmp = list(
                    action_mask_segment[bs][pos_in_game_block_list[bs]:pos_in_game_block_list[bs] +
                                                                       self._cfg.num_unroll_steps + 1]
                )
                if len(action_mask_tmp) < self._cfg.num_unroll_steps + 1:
                    action_mask_tmp += [
                        list(np.ones(self._cfg.model.action_space_size, dtype=np.int8))
                        for _ in range(self._cfg.num_unroll_steps + 1 - len(action_mask_tmp))
                    ]
                action_mask.append(action_mask_tmp)
            action_mask = to_list(action_mask)
            action_mask = sum(action_mask, [])

        batch_target_values, batch_rewards = [], []
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

                if not model.training:
                    # if not in training, obtain the scalars of the value/reward
                    [m_output.hidden_state, m_output.value, m_output.policy_logits] = to_detach_cpu_numpy(
                        [m_output.hidden_state, inverse_scalar_transform(m_output.value, self._cfg.model.support_scale),
                         m_output.policy_logits])

                network_output.append(m_output)

            # concat the output slices after model inference
            if self._cfg.use_root_value:
                # use the root values from MCTS
                # the root values have limited improvement but require much more GPU actors;
                _, reward_pool, policy_logits_pool, hidden_state_roots = concat_output(network_output)
                reward_pool = reward_pool.squeeze().tolist()
                policy_logits_pool = policy_logits_pool.tolist()

                if self._cfg.mcts_ctree:
                    # cpp mcts_tree
                    if to_play_segment[0][0] in [None, -1]:
                        # for one_player atari games
                        action_mask = [
                            list(np.ones(self._cfg.model.action_space_size, dtype=np.int8)) for _ in range(transition_batch_size)
                        ]
                        to_play = [-1 for i in range(transition_batch_size)]

                    legal_actions = [[i for i, x in enumerate(action_mask[j]) if x == 1] for j in range(transition_batch_size)]
                    roots = ctree.Roots(transition_batch_size, legal_actions)

                    noises = [
                        np.random.dirichlet([self._cfg.root_dirichlet_alpha] * self._cfg.model.action_space_size
                                            ).astype(np.float32).tolist() for _ in range(transition_batch_size)
                    ]
                    roots.prepare(self._cfg.root_exploration_fraction, noises, reward_pool, policy_logits_pool, to_play)
                    # do MCTS for a new policy with the recent target model
                    MCTS_ctree(self._cfg).search(roots, model, hidden_state_roots, to_play)
                else:
                    # python mcts_tree
                    if to_play_segment[0][0] in [None, -1]:
                        # for one_player atari games
                        action_mask = [
                            list(np.ones(self._cfg.model.action_space_size, dtype=np.int8)) for _ in range(transition_batch_size)
                        ]
                    legal_actions = [[i for i, x in enumerate(action_mask[j]) if x == 1] for j in range(transition_batch_size)]
                    roots = ptree.Roots(transition_batch_size, legal_actions, self._cfg.num_simulations)
                    noises = [
                        np.random.dirichlet([self._cfg.root_dirichlet_alpha] * int(sum(action_mask[j]))
                                            ).astype(np.float32).tolist() for j in range(transition_batch_size)
                    ]

                    if to_play_segment[0][0] in [None, -1]:
                        roots.prepare(
                            self._cfg.root_exploration_fraction, noises, reward_pool, policy_logits_pool, to_play=-1
                        )
                        # do MCTS for a new policy with the recent target model
                        MCTS_ptree(self._cfg).search(roots, model, hidden_state_roots, to_play=-1)
                    else:
                        roots.prepare(
                            self._cfg.root_exploration_fraction,
                            noises,
                            reward_pool,
                            policy_logits_pool,
                            to_play=to_play
                        )
                        # do MCTS for a new policy with the recent target model
                        MCTS_ptree(self._cfg).search(roots, model, hidden_state_roots, to_play=to_play)

                roots_values = roots.get_values()
                value_list = np.array(roots_values)
            else:
                # use the predicted values
                value_list = concat_output_value(network_output)

            # get last state value
            value_list = value_list.reshape(-1) * (
                    np.array([self._cfg.discount_factor for _ in range(transition_batch_size)]) ** td_steps_list
            )
            value_list = value_list * np.array(value_mask)
            value_list = value_list.tolist()

            horizon_id, value_index = 0, 0
            for game_block_len_non_re, reward_list, state_index in zip(game_block_lens, rewards_list,
                                                                       pos_in_game_block_list):
                # game_block_len = len(game)
                target_values = []
                target_rewards = []

                reward = 0.0
                base_index = state_index
                for current_index in range(state_index, state_index + self._cfg.num_unroll_steps + 1):
                    bootstrap_index = current_index + td_steps_list[value_index]
                    # for i, reward in enumerate(game.rewards[current_index:bootstrap_index]):
                    for i, reward in enumerate(reward_list[current_index:bootstrap_index]):
                        value_list[value_index] += reward * self._cfg.discount_factor ** i

                    horizon_id += 1

                    if current_index < game_block_len_non_re:
                        target_values.append(value_list[value_index])
                        # Since the horizon is small and the discount_factor is close to 1.
                        # Compute the reward sum to approximate the value prefix for simplification
                        # reward += reward_list[current_index]  # * self._cfg.discount_factor ** (current_index - base_index)
                        # reward += reward_list[current_index]  # * self._cfg.discount_factor ** (current_index - base_index)
                        # target_rewards.append(reward)
                        target_rewards.append(reward_list[current_index])
                    else:
                        target_values.append(0)
                        # target_rewards.append(reward)
                        target_rewards.append(0.0)
                    value_index += 1

                batch_rewards.append(target_rewards)
                batch_target_values.append(target_values)

        batch_rewards = np.asarray(batch_rewards)
        batch_target_values = np.asarray(batch_target_values)
        return batch_rewards, batch_target_values

    def compute_target_policy_reanalyzed(self, policy_re_context, model):
        """
        Overview:
            prepare policy targets from the reanalyzed context of policies
        Arguments:
            - policy_re_context (:obj:`List`): List of policy context to reanalyzed
        Returns:
            - batch_target_policies_re
        """
        batch_target_policies_re = []
        if policy_re_context is None:
            return batch_target_policies_re

        # for two_player board games
        policy_obs_list, policy_mask, pos_in_game_block_list, batch_index_list, child_visits, game_block_lens, action_mask_segment, \
        to_play_segment = policy_re_context
        transition_batch_size = len(policy_obs_list)
        game_block_batch_size = len(pos_in_game_block_list)


        if self._cfg.env_type == 'board_games':
            # for two_player board games: prepare the to_play and action_mask mini-batch

            # to_play
            to_play = []
            for bs in range(game_block_batch_size):
                to_play_tmp = list(
                    to_play_segment[bs][pos_in_game_block_list[bs]:pos_in_game_block_list[bs] +
                                                                   self._cfg.num_unroll_steps + 1]
                )
                if len(to_play_tmp) < self._cfg.num_unroll_steps + 1:
                    # effective play index is {1,2}
                    to_play_tmp += [1 for _ in range(self._cfg.num_unroll_steps + 1 - len(to_play_tmp))]
                to_play.append(to_play_tmp)

            tmp = []
            for i in to_play:
                tmp += list(i)
            to_play = tmp

            # action_mask
            action_mask = []
            for bs in range(game_block_batch_size):
                action_mask_tmp = list(
                    action_mask_segment[bs][pos_in_game_block_list[bs]:pos_in_game_block_list[bs] +
                                                                       self._cfg.num_unroll_steps + 1]
                )
                if len(action_mask_tmp) < self._cfg.num_unroll_steps + 1:
                    action_mask_tmp += [
                        list(np.ones(self._cfg.model.action_space_size, dtype=np.int8))
                        for _ in range(self._cfg.num_unroll_steps + 1 - len(action_mask_tmp))
                    ]
                action_mask.append(action_mask_tmp)

            action_mask = to_ndarray(action_mask)
            tmp = []
            for i in action_mask:
                tmp += i
            action_mask = tmp

            # the minimal size is <self._cfg. num_unroll_steps+1>
            legal_actions = [
                [i for i, x in enumerate(action_mask[j]) if x == 1]
                for j in range(max(self._cfg.num_unroll_steps + 1, transition_batch_size))
            ]

        with torch.no_grad():
            policy_obs_list = prepare_observation_list(policy_obs_list)
            # split a full batch into slices of mini_infer_size: to save the GPU memory for more GPU actors
            slices = int(np.ceil(transition_batch_size / self._cfg.mini_infer_size))
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

                network_output.append(m_output)

            _, reward_pool, policy_logits_pool, hidden_state_roots = concat_output(network_output)
            reward_pool = reward_pool.squeeze().tolist()
            policy_logits_pool = policy_logits_pool.tolist()
            if self._cfg.mcts_ctree:
                # cpp mcts_tree
                if to_play_segment[0][0] in [None, -1]:
                    # for one_player atari games
                    action_mask = [
                        list(np.ones(self._cfg.model.action_space_size, dtype=np.int8)) for _ in range(transition_batch_size)
                    ]
                    to_play = [-1 for i in range(transition_batch_size)]

                legal_actions = [[i for i, x in enumerate(action_mask[j]) if x == 1] for j in range(transition_batch_size)]
                roots = ctree.Roots(transition_batch_size, legal_actions)

                noises = [
                    np.random.dirichlet([self._cfg.root_dirichlet_alpha] * self._cfg.model.action_space_size
                                        ).astype(np.float32).tolist() for _ in range(transition_batch_size)
                ]
                roots.prepare(self._cfg.root_exploration_fraction, noises, reward_pool, policy_logits_pool, to_play)
                # do MCTS for a new policy with the recent target model
                MCTS_ctree(self._cfg).search(roots, model, hidden_state_roots, to_play)
                # roots_legal_actions_list = roots.legal_actions_list
                roots_legal_actions_list = legal_actions
            else:
                # python mcts_tree
                if to_play_segment[0][0] in [None, -1]:
                    # for one_player atari games
                    action_mask = [
                        list(np.ones(self._cfg.model.action_space_size, dtype=np.int8)) for _ in range(transition_batch_size)
                    ]
                    legal_actions = [[i for i, x in enumerate(action_mask[j]) if x == 1] for j in range(transition_batch_size)]

                roots = ptree.Roots(transition_batch_size, legal_actions, self._cfg.num_simulations)
                noises = [
                    np.random.dirichlet([self._cfg.root_dirichlet_alpha] * int(sum(action_mask[j]))
                                        ).astype(np.float32).tolist() for j in range(transition_batch_size)
                ]
                if to_play_segment[0][0] in [None, -1]:
                    roots.prepare(
                        self._cfg.root_exploration_fraction, noises, reward_pool, policy_logits_pool, to_play=-1
                    )
                    # do MCTS for a new policy with the recent target model
                    MCTS_ptree(self._cfg).search(roots, model, hidden_state_roots, to_play=-1)
                else:
                    roots.prepare(
                        self._cfg.root_exploration_fraction, noises, reward_pool, policy_logits_pool, to_play=to_play
                    )
                    # do MCTS for a new policy with the recent target model
                    MCTS_ptree(self._cfg).search(roots, model, hidden_state_roots, to_play=to_play)
                roots_legal_actions_list = roots.legal_actions_list

            roots_distributions = roots.get_distributions()

            policy_index = 0
            for state_index, game_index in zip(pos_in_game_block_list, batch_index_list):
                target_policies = []

                for current_index in range(state_index, state_index + self._cfg.num_unroll_steps + 1):
                    distributions = roots_distributions[policy_index]

                    if policy_mask[policy_index] == 0:
                        target_policies.append([0 for _ in range(self._cfg.model.action_space_size)])
                    else:
                        if distributions is None:
                            # if at some obs, the legal_action is None, add the fake target_policy
                            target_policies.append(
                                list(np.ones(self._cfg.model.action_space_size) / self._cfg.model.action_space_size)
                            )
                        else:
                            if self._cfg.mcts_ctree:
                                """
                                cpp mcts_tree
                                """
                                if to_play_segment[0][0] in [None, -1]:
                                    # for one_player atari games
                                    # TODO(pu): very important
                                    sum_visits = sum(distributions)
                                    policy = [visit_count / sum_visits for visit_count in distributions]
                                    target_policies.append(policy)
                                    # target_policies.append(distributions)
                                else:
                                    # for two_player board games
                                    policy_tmp = [0 for _ in range(self._cfg.model.action_space_size)]
                                    # to make sure target_policies have the same dimension
                                    # target_policy = torch.from_numpy(target_policy) be correct
                                    sum_visits = sum(distributions)
                                    policy = [visit_count / sum_visits for visit_count in distributions]
                                    for index, legal_action in enumerate(roots_legal_actions_list[policy_index]):
                                        policy_tmp[legal_action] = policy[index]
                                    target_policies.append(policy_tmp)
                            else:
                                """
                                python mcts_tree
                                """
                                if to_play_segment[0][0] in [None, -1]:
                                    # TODO(pu): very important
                                    sum_visits = sum(distributions)
                                    policy = [visit_count / sum_visits for visit_count in distributions]
                                    target_policies.append(policy)
                                    # target_policies.append(distributions)
                                else:
                                    # for two_player board games
                                    policy_tmp = [0 for _ in range(self._cfg.model.action_space_size)]
                                    # to make sure target_policies have the same dimension
                                    # target_policy = torch.from_numpy(target_policy) be correct
                                    sum_visits = sum(distributions)
                                    policy = [visit_count / sum_visits for visit_count in distributions]
                                    for index, legal_action in enumerate(roots_legal_actions_list[policy_index]):
                                        policy_tmp[legal_action] = policy[index]
                                    target_policies.append(policy_tmp)

                    policy_index += 1

                batch_target_policies_re.append(target_policies)

        batch_target_policies_re = np.array(batch_target_policies_re)

        return batch_target_policies_re

    def compute_target_policy_non_reanalyzed(self, policy_non_re_context):
        """
        Overview:
            prepare policy targets from the non-reanalyzed context of policies
        Arguments:
            - policy_non_re_context (:obj:`List`): List containing:
                - pos_in_game_block_list
                - child_visits
                - game_block_lens
                - action_mask_segment
                - to_play_segment
        Returns:
            - batch_target_policies_non_re
        """
        batch_target_policies_non_re = []
        if policy_non_re_context is None:
            return batch_target_policies_non_re

        pos_in_game_block_list, child_visits, game_block_lens, action_mask_segment, to_play_segment = policy_non_re_context

        game_block_batch_size = len(pos_in_game_block_list)

        if self._cfg.env_type == 'board_games':
            # for two_player board games
            # action_mask
            action_mask = []
            for bs in range(game_block_batch_size):
                action_mask_tmp = list(
                    action_mask_segment[bs][pos_in_game_block_list[bs]:pos_in_game_block_list[bs] +
                                                                       self._cfg.num_unroll_steps + 1]
                )
                if len(action_mask_tmp) < self._cfg.num_unroll_steps + 1:
                    action_mask_tmp += [
                        list(np.ones(self._cfg.model.action_space_size, dtype=np.int8))
                        for _ in range(self._cfg.num_unroll_steps + 1 - len(action_mask_tmp))
                    ]
                action_mask.append(action_mask_tmp)
            action_mask = to_ndarray(action_mask)
            tmp = []
            for i in action_mask:
                tmp += i
            action_mask = tmp

            # the minimal size is <self._cfg. num_unroll_steps+1>
            legal_actions = [
                [i for i, x in enumerate(action_mask[j]) if x == 1]
                for j in range(game_block_batch_size * (self._cfg.num_unroll_steps + 1))
            ]

        with torch.no_grad():
            policy_index = 0
            # for policy
            policy_mask = []  # 0 -> out of traj, 1 -> old policy
            # for game, state_index in zip(games, pos_in_game_block_list):
            for game_block_len, child_visit, state_index in zip(game_block_lens, child_visits, pos_in_game_block_list):
                # game_block_len = len(game)
                target_policies = []

                for current_index in range(state_index, state_index + self._cfg.num_unroll_steps + 1):
                    if current_index < game_block_len:
                        # target_policies.append(child_visit[current_index])
                        policy_mask.append(1)
                        # child_visit is already a distribution
                        distributions = child_visit[current_index]
                        if self._cfg.mcts_ctree:
                            """
                            cpp mcts_tree
                            """
                            if self._cfg.env_type == 'not_board_games':
                                # for one_player atari games
                                target_policies.append(distributions)
                            else:
                                # for two_player board games
                                policy_tmp = [0 for _ in range(self._cfg.model.action_space_size)]
                                # to make sure target_policies have the same dimension <self._cfg.model.action_space_size>
                                # sum_visits = sum(distributions)
                                # distributions = [visit_count / sum_visits for visit_count in distributions]
                                for index, legal_action in enumerate(legal_actions[policy_index]):
                                    # try:
                                    # only the action in ``legal_action`` the policy logits is nonzero
                                    policy_tmp[legal_action] = distributions[index]
                                    # except Exception as error:
                                    #     print(error)
                                target_policies.append(policy_tmp)
                        else:
                            """
                            python mcts_tree
                            """
                            if self._cfg.env_type == 'not_board_games':
                                # for one_player atari games
                                target_policies.append(distributions)
                            else:
                                # for two_player board games
                                policy_tmp = [0 for _ in range(self._cfg.model.action_space_size)]
                                # to make sure target_policies have the same dimension <self._cfg.model.action_space_size>
                                # sum_visits = sum(distributions)
                                # distributions = [visit_count / sum_visits for visit_count in distributions]
                                for index, legal_action in enumerate(legal_actions[policy_index]):
                                    # try:
                                    # only the action in ``legal_action`` the policy logits is nonzero
                                    policy_tmp[legal_action] = distributions[index]
                                    # except Exception as error:
                                    #     print(error)
                                target_policies.append(policy_tmp)

                    else:
                        # the invalid target policy
                        policy_mask.append(0)
                        target_policies.append([0 for _ in range(self._cfg.model.action_space_size)])

                    policy_index += 1

                batch_target_policies_non_re.append(target_policies)
        batch_target_policies_non_re = np.asarray(batch_target_policies_non_re)
        return batch_target_policies_non_re

    def push_game_segments(self, data_and_meta: Any):
        """
        Overview:
            Push game data and it's meta information in buffer.
            Save a game block
        Keys:
            - data (:obj:`Any`): The data which will be pushed into buffer.
                                 i.e. a game block
            - meta (:obj:`dict`): Meta information
        """
        data, meta = data_and_meta
        for (data_game, meta_game) in zip(data, meta):
            self.push_game_block(data_game, meta_game)

    def push_game_block(self, data: Any, meta: Optional[dict] = None):
        """
        Overview:
            Push data and it's meta information in buffer.
            Save a game block.
        Arguments:
            - data (:obj:`Any`): The data (a game block) which will be pushed into buffer.
            - meta (:obj:`dict`): Meta information, e.g. priority, count, staleness.
                - done (:obj:`bool`): whether the game is finished.
                - unroll_plus_td_steps (:obj:`int`): if the game is not finished, we only save the transitions that can be computed
                - priorities (:obj:`list`): the priorities corresponding to the transitions in the game segment
        Returns:
            - buffered_data (:obj:`BufferedData`): The pushed data.
        """
        if meta['done']:
            self._eps_collected += 1
            valid_len = len(data)
        else:
            valid_len = len(data) - meta['unroll_plus_td_steps']

        if meta['priorities'] is None:
            max_prio = self.game_pos_priorities.max() if self.game_block_buffer else 1
            # if no 'priorities' provided, set the valid part of the new-added game segment the max_prio
            self.game_pos_priorities = np.concatenate(
                (self.game_pos_priorities,
                 [max_prio for _ in range(valid_len)] + [0. for _ in range(valid_len, len(data))])
            )
        else:
            assert len(data) == len(meta['priorities']), " priorities should be of same length as the game steps"
            priorities = meta['priorities'].copy().reshape(-1)
            priorities[valid_len:len(data)] = 0.
            self.game_pos_priorities = np.concatenate((self.game_pos_priorities, priorities))

        self.game_block_buffer.append(data)
        self.game_block_game_pos_look_up += [(self.base_idx + len(self.game_block_buffer) - 1, step_pos) for step_pos in
                                             range(len(data))]

    def get_transition(self, idx):
        """
        Overview:
            Sample one transition according to the idx
        Arguments:
            - idx: transition index
        Returns:
            - transition (:obj:`tuple`): One transition of pushed data.
        """
        game_block_idx, pos_in_game_block = self.game_block_game_pos_look_up[idx]
        game_block_idx -= self.base_idx
        transition = self.game_block_buffer[game_block_idx][pos_in_game_block]
        return transition

    def get_game_block_from_idx(self, idx: int) -> BufferedData:
        """
        Overview:
            Get one game according to the idx
        Arguments:
            - idx: game index
        Returns:
            - game: (:obj:`Gamesegment`): One game segment of pushed data.
        """
        return self.get_game_block(idx)

    def get_game_block(self, idx):
        """
        Overview:
            sample one game segment according to the idx
        Arguments:
            - idx: transition index
        Returns:
            - game: (:obj:`Gamesegment`): One game segment of pushed data.
        """
        game_block_idx, pos_in_game_block = self.game_block_game_pos_look_up[idx]
        game_block_idx -= self.base_idx
        game_block = self.game_block_buffer[game_block_idx]
        return game_block

    def batch_update(self, indices: List[str], metas: Optional[List[Optional[dict]]] = None) -> None:
        """
        Overview:
            Batch update meta by indices, maybe useful in some data architectures.
        Arguments:
            - indices (:obj:`List[str]`): Index of data.
            - metas (:obj:`Optional[List[Optional[dict]]]`): Meta information.
        """
        # only update the priorities for data still in replay buffer
        for i in range(len(indices)):
            if metas['make_time'][i] > self.clear_time:
                idx, prio = indices[i], metas['batch_priorities'][i]
                self.game_pos_priorities[idx] = prio

    def update(self, index, data: Optional[Any] = None, meta: Optional[dict] = None) -> bool:
        """
        Overview:
            Update data and meta by index
        Arguments:
            - index (:obj:`str`): Index of one transition to be updated.
            - data (:obj:`any`): Pure data.  one transition.
            - meta (:obj:`dict`): Meta information.
        Returns:
            - success (:obj:`bool`): Success or not, if data with the index not exist in buffer, return false.
        """

        success = False
        if index < self.get_num_of_transitions():
            prio = meta['priorities']
            self.game_pos_priorities[index] = prio
            game_block_idx, pos_in_game_block = self.game_block_game_pos_look_up[index]
            game_block_idx -= self.base_idx
            # update one transition
            self.game_block_buffer[game_block_idx][pos_in_game_block] = data
            success = True

        return success

    def update_priority(self, train_data, batch_priorities) -> None:
        """
        Overview:
            Update the priority of training data.
        Arguments:
            - train_data (:obj:`Optional[List[Optional[np.ndarray]]]`): training data to be updated priority.
            - batch_priorities (:obj:`batch_priorities`): priorities to update to.
        """
        self.batch_update(indices=train_data[0][3],
                          metas={'make_time': train_data[0][5], 'batch_priorities': batch_priorities})

    def remove_oldest_data_to_fit(self):
        """
        Overview:
            remove some oldest data if the replay buffer is full.
        """
        nums_of_game_histoty = self.get_num_of_game_blocks()
        total_transition = self.get_num_of_transitions()
        if total_transition > self.replay_buffer_size:
            index = 0
            for i in range(nums_of_game_histoty):
                total_transition -= len(self.game_block_buffer[i])
                if total_transition <= self.replay_buffer_size * self.keep_ratio:
                    index = i
                    break

            if total_transition >= self._cfg.batch_size:
                self._remove(index + 1)

    def _remove(self, num_excess_games):
        """
        Overview:
            delete game histories in index [0: excess_game_block_index]
        Arguments:
            - excess_game_block_index (:obj:`List[str]`): Index of data.
        """
        excess_games_steps = sum([len(game) for game in self.game_block_buffer[:num_excess_games]])
        del self.game_block_buffer[:num_excess_games]
        self.game_pos_priorities = self.game_pos_priorities[excess_games_steps:]
        del self.game_block_game_pos_look_up[:excess_games_steps]
        self.base_idx += num_excess_games

        self.clear_time = time.time()

    def delete(self, index: str):
        """
        Overview:
            Delete one data sample by index
        Arguments:
            - index (:obj:`str`): Index
        """
        pass

    def clear(self) -> None:
        del self.game_block_buffer[:]

    def get_batch_size(self):
        return self.batch_size

    def get_priorities(self):
        return self.game_pos_priorities

    def get_num_of_episodes(self):
        # number of collected episodes
        return self._eps_collected

    def get_num_of_game_blocks(self) -> int:
        # number of games, i.e. num of game blocks
        return len(self.game_block_buffer)

    def count(self):
        # number of games, i.e. num of game blocks
        return len(self.game_block_buffer)

    def get_num_of_transitions(self):
        # total number of transitions
        return len(self.game_pos_priorities)

    def __copy__(self) -> "GameBuffer":
        buffer = type(self)(cfg=self._cfg)
        buffer.storage = self.game_block_buffer
        return buffer

    # the following is to be compatible with Buffer class.
    def sample_(
            self,
            size: Optional[int] = None,
            indices: Optional[List[str]] = None,
            replace: bool = False,
            sample_range: Optional[slice] = None,
            ignore_insufficient: bool = False,
            groupby: str = None,
            rolling_window: int = None
    ) -> Union[List[BufferedData], List[List[BufferedData]]]:
        """
        Overview:
            Sample data with length ``size``.
        Arguments:
            - size (:obj:`Optional[int]`): The number of the data that will be sampled.
            - indices (:obj:`Optional[List[str]]`): Sample with multiple indices.
            - replace (:obj:`bool`): If use replace is true, you may receive duplicated data from the buffer.
            - sample_range (:obj:`slice`): Sample range slice.
            - ignore_insufficient (:obj:`bool`): If ignore_insufficient is true, sampling more than buffer size
                with no repetition will not cause an exception.
            - groupby (:obj:`str`): Groupby key in meta.
            - rolling_window (:obj:`int`): Return batches of window size.
        Returns:
            - sample_data (:obj:`Union[List[BufferedData], List[List[BufferedData]]]`):
                A list of data with length ``size``, may be nested if groupby or rolling_window is set.
        """
        storage = self.game_block_buffer
        if sample_range:
            storage = list(itertools.islice(self.storage, sample_range.start, sample_range.stop, sample_range.step))

        # Size and indices
        assert size or indices, "One of size and indices must not be empty."
        if (size and indices) and (size != len(indices)):
            raise AssertionError("Size and indices length must be equal.")
        if not size:
            size = len(indices)
        # Indices and groupby
        assert not (indices and groupby), "Cannot use groupby and indicex at the same time."
        # Groupby and rolling_window
        assert not (groupby and rolling_window), "Cannot use groupby and rolling_window at the same time."
        assert not (indices and rolling_window), "Cannot use indices and rolling_window at the same time."

        value_error = None
        sampled_data = []
        if indices:
            sampled_data = [self.game_block_buffer[game_block_idx] for game_block_idx in indices]

        elif groupby:
            sampled_data = self._sample_by_group(size=size, groupby=groupby, replace=replace, storage=storage)
        elif rolling_window:
            sampled_data = self._sample_by_rolling_window(
                size=size, replace=replace, rolling_window=rolling_window, storage=storage
            )
        else:
            if replace:
                sampled_data = random.choices(storage, k=size)
            else:
                try:
                    sampled_data = random.sample(storage, k=size)
                except ValueError as e:
                    value_error = e

        if value_error or len(sampled_data) != size:
            if ignore_insufficient:
                logging.warning(
                    "Sample operation is ignored due to data insufficient, current buffer is {} while sample is {}".
                        format(self.count(), size)
                )
            else:
                raise ValueError("There are less than {} records/groups in buffer({})".format(size, self.count()))

        return sampled_data

    def save_data(self):
        pass

    def load_data(self):
        pass

    def get(self):
        pass

    def push(self):
        pass

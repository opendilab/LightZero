"""
The following code is adapted from https://github.com/YeWR/EfficientZero/core/mcts.py
"""
import copy

import numpy as np
import torch
from easydict import EasyDict

from lzero.rl_utils.mcts.ctree_efficientzero import ez_tree as tree_efficientzero
from ..scaling_transform import inverse_scalar_transform

###########################################################
# EfficientZero
###########################################################


class EfficientZeroMCTSCtree(object):
    config = dict(
        device='gpu',
        pb_c_base=19652,
        pb_c_init=1.25,
        support_size=300,
        discount=0.997,
        num_simulations=50,
        lstm_horizon_len=5,
        categorical_distribution=True,
    )

    @classmethod
    def default_config(cls: type) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    def __init__(self, config=None):
        if config is None:
            config = config
        self.config = config

    def search(self, roots, model, hidden_state_roots, reward_hidden_state_roots, to_play_batch):
        """
        Overview:
            Do MCTS for the roots (a batch of root nodes in parallel). Parallel in model inference.
             Use the cpp tree.
        Arguments:
            - roots (:obj:`Any`): a batch of expanded root nodes
            - hidden_state_roots (:obj:`list`): the hidden states of the roots
            - reward_hidden_state_roots (:obj:`list`): the value prefix hidden states in LSTM of the roots
            - to_play_batch (:obj:`list`): the to_play_batch list used in two_player mode board games
        """
        with torch.no_grad():
            model.eval()

            # preparation
            num = roots.num
            device = self.config.device
            pb_c_base, pb_c_init, discount = self.config.pb_c_base, self.config.pb_c_init, self.config.discount
            # the data storage of hidden states: storing the states of all the tree nodes
            hidden_state_pool = [hidden_state_roots]
            # 1 x batch x 64
            # the data storage of value prefix hidden states in LSTM
            reward_hidden_state_c_pool = [reward_hidden_state_roots[0]]
            reward_hidden_state_h_pool = [reward_hidden_state_roots[1]]

            # the index of each layer in the tree
            hidden_state_index_x = 0
            # minimax value storage
            min_max_stats_lst = tree_efficientzero.MinMaxStatsList(num)
            min_max_stats_lst.set_delta(self.config.value_delta_max)

            for index_simulation in range(self.config.num_simulations):
                hidden_states = []
                hidden_states_c_reward = []
                hidden_states_h_reward = []

                # prepare a result wrapper to transport results between python and c++ parts
                results = tree_efficientzero.ResultsWrapper(num=num)

                # traverse to select actions for each root
                # hidden_state_index_x_lst: the first index of leaf node states in hidden_state_pool
                # hidden_state_index_y_lst: the second index of leaf node states in hidden_state_pool
                # the hidden state of the leaf node is hidden_state_pool[x, y]; value prefix states are the same
                hidden_state_index_x_lst, hidden_state_index_y_lst, last_actions, virtual_to_play_batch = tree_efficientzero.batch_traverse(
                    roots, pb_c_base, pb_c_init, discount, min_max_stats_lst, results, copy.deepcopy(to_play_batch)
                )
                # obtain the search horizon for leaf nodes
                search_lens = results.get_search_len()

                # obtain the states for leaf nodes
                for ix, iy in zip(hidden_state_index_x_lst, hidden_state_index_y_lst):
                    hidden_states.append(hidden_state_pool[ix][iy])
                    hidden_states_c_reward.append(reward_hidden_state_c_pool[ix][0][iy])
                    hidden_states_h_reward.append(reward_hidden_state_h_pool[ix][0][iy])

                hidden_states = torch.from_numpy(np.asarray(hidden_states)).to(device).float()
                hidden_states_c_reward = torch.from_numpy(np.asarray(hidden_states_c_reward)).to(device).unsqueeze(0)
                hidden_states_h_reward = torch.from_numpy(np.asarray(hidden_states_h_reward)).to(device).unsqueeze(0)
                # only for discrete action
                last_actions = torch.from_numpy(np.asarray(last_actions)).to(device).long()

                # evaluation for leaf nodes
                network_output = model.recurrent_inference(
                    hidden_states, (hidden_states_c_reward, hidden_states_h_reward), last_actions
                )
                # TODO(pu)
                if not model.training:
                    # if not in training, obtain the scalars of the value/reward
                    network_output.value = inverse_scalar_transform(
                        network_output.value,
                        self.config.support_size,
                    ).detach().cpu().numpy()
                    network_output.value_prefix = inverse_scalar_transform(
                        network_output.value_prefix,
                        self.config.support_size,
                    ).detach().cpu().numpy()
                    network_output.hidden_state = network_output.hidden_state.detach().cpu().numpy()
                    network_output.reward_hidden_state = (
                        network_output.reward_hidden_state[0].detach().cpu().numpy(),
                        network_output.reward_hidden_state[1].detach().cpu().numpy()
                    )
                    network_output.policy_logits = network_output.policy_logits.detach().cpu().numpy()

                hidden_state_nodes = network_output.hidden_state
                value_prefix_pool = network_output.value_prefix.reshape(-1).tolist()
                value_pool = network_output.value.reshape(-1).tolist()
                policy_logits_pool = network_output.policy_logits.tolist()
                reward_hidden_state_nodes = network_output.reward_hidden_state

                hidden_state_pool.append(hidden_state_nodes)
                # reset 0
                # reset the hidden states in LSTM every horizon steps in search
                # only need to predict the value prefix in a range (eg: s0 -> s5)
                assert self.config.lstm_horizon_len > 0
                reset_idx = (np.array(search_lens) % self.config.lstm_horizon_len == 0)
                assert len(reset_idx) == num
                reward_hidden_state_nodes[0][:, reset_idx, :] = 0
                reward_hidden_state_nodes[1][:, reset_idx, :] = 0
                is_reset_lst = reset_idx.astype(np.int32).tolist()

                reward_hidden_state_c_pool.append(reward_hidden_state_nodes[0])
                reward_hidden_state_h_pool.append(reward_hidden_state_nodes[1])
                hidden_state_index_x += 1

                # backpropagation along the search path to update the attributes
                tree_efficientzero.batch_back_propagate(
                    hidden_state_index_x, discount, value_prefix_pool, value_pool, policy_logits_pool,
                    min_max_stats_lst, results, is_reset_lst, virtual_to_play_batch
                )


###########################################################
# MuZero
###########################################################

from lzero.rl_utils.mcts.ctree_muzero import mz_tree as tree_muzero


class MuZeroMCTSCtree(object):
    config = dict(
        device='gpu',
        pb_c_base=19652,
        pb_c_init=1.25,
        support_size=300,
        discount=0.997,
        num_simulations=50,
        lstm_horizon_len=5,
    )

    @classmethod
    def default_config(cls: type) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    def __init__(self, config=None):
        if config is None:
            config = config
        self.config = config

    def search(self, roots, model, hidden_state_roots, to_play_batch):
        """
        Overview:
            Do MCTS for the roots (a batch of root nodes in parallel). Parallel in model inference.
             Use the cpp tree.
        Arguments:
            - roots (:obj:`Any`): a batch of expanded root nodes
            - hidden_state_roots (:obj:`list`): the hidden states of the roots
            - to_play_batch (:obj:`list`): the to_play_batch list used in two_player mode board games
        """
        with torch.no_grad():
            model.eval()

            # preparation
            num = roots.num
            device = self.config.device
            pb_c_base, pb_c_init, discount = self.config.pb_c_base, self.config.pb_c_init, self.config.discount
            # the data storage of hidden states: storing the states of all the tree nodes
            hidden_state_pool = [hidden_state_roots]

            # the index of each layer in the tree
            hidden_state_index_x = 0
            # minimax value storage
            min_max_stats_lst = tree_muzero.MinMaxStatsList(num)
            min_max_stats_lst.set_delta(self.config.value_delta_max)

            for index_simulation in range(self.config.num_simulations):
                hidden_states = []
                hidden_states_c_reward = []
                hidden_states_h_reward = []

                # prepare a result wrapper to transport results between python and c++ parts
                results = tree_muzero.ResultsWrapper(num=num)

                # traverse to select actions for each root
                # hidden_state_index_x_lst: the first index of leaf node states in hidden_state_pool
                # hidden_state_index_y_lst: the second index of leaf node states in hidden_state_pool
                # the hidden state of the leaf node is hidden_state_pool[x, y]; value prefix states are the same
                hidden_state_index_x_lst, hidden_state_index_y_lst, last_actions, virtual_to_play_batch = tree_muzero.batch_traverse(
                    roots, pb_c_base, pb_c_init, discount, min_max_stats_lst, results, copy.deepcopy(to_play_batch)
                )
                # obtain the search horizon for leaf nodes
                search_lens = results.get_search_len()

                # obtain the states for leaf nodes
                for ix, iy in zip(hidden_state_index_x_lst, hidden_state_index_y_lst):
                    hidden_states.append(hidden_state_pool[ix][iy])

                hidden_states = torch.from_numpy(np.asarray(hidden_states)).to(device).float()
                # only for discrete action
                last_actions = torch.from_numpy(np.asarray(last_actions)).to(device).long()

                # evaluation for leaf nodes
                network_output = model.recurrent_inference(hidden_states, last_actions)
                # TODO(pu)
                if not model.training:
                    # if not in training, obtain the scalars of the value/reward
                    network_output.value = inverse_scalar_transform(
                        network_output.value,
                        self.config.support_size,
                        categorical_distribution=self.config.categorical_distribution
                    ).detach().cpu().numpy()
                    network_output.reward = inverse_scalar_transform(
                        network_output.reward,
                        self.config.support_size,
                        categorical_distribution=self.config.categorical_distribution
                    ).detach().cpu().numpy()
                    network_output.hidden_state = network_output.hidden_state.detach().cpu().numpy()
                    network_output.policy_logits = network_output.policy_logits.detach().cpu().numpy()

                hidden_state_nodes = network_output.hidden_state
                reward_pool = network_output.reward.reshape(-1).tolist()
                value_pool = network_output.value.reshape(-1).tolist()
                policy_logits_pool = network_output.policy_logits.tolist()

                hidden_state_pool.append(hidden_state_nodes)
                hidden_state_index_x += 1

                # backpropagation along the search path to update the attributes
                tree_muzero.batch_back_propagate(
                    hidden_state_index_x, discount, reward_pool, value_pool, policy_logits_pool, min_max_stats_lst,
                    results, virtual_to_play_batch
                )
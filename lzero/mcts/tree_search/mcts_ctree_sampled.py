import copy

import numpy as np
import torch
from easydict import EasyDict

from ..scaling_transform import inverse_scalar_transform

from lzero.mcts.ctree.ctree_sampled_efficientzero import ezs_tree as tree_efficientzero

# ==============================================================
# Sampled EfficientZero
# ==============================================================


class SampledEfficientZeroMCTSCtree(object):
    """
    Overview:
        MCTSCtree for Sampled EfficientZero. The core ``batch_traverse`` and ``batch_backpropagate`` function is implemented in C++.
    Interfaces:
        __init__, search
    """
    
    # the default_config for SampledEfficientZeroMCTSCtree.
    config = dict(
        device='cpu',
        support_scale=300,
        discount_factor=0.997,
        num_simulations=50,
        lstm_horizon_len=5,
        # UCB related config
        root_dirichlet_alpha=0.3,
        root_exploration_fraction=0.25,
        pb_c_base=19652,
        pb_c_init=1.25,
        value_delta_max=0.01,
    )

    @classmethod
    def default_config(cls: type) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    def __init__(self, cfg=None):
        """
        Overview:
            Use the default configuration mechanism. If a user passes in a cfg with a key that matches an existing key 
            in the default configuration, the user-provided value will override the default configuration. Otherwise, 
            the default configuration will be used.
        """
        default_config = self.default_config()
        default_config.update(cfg)
        self._cfg = default_config

    def search(self, roots, model, hidden_state_roots, reward_hidden_state_roots, to_play_batch):
        """
        Overview:
            Do MCTS for the roots (a batch of root nodes in parallel). Parallel in model inference.
             Use the cpp ctree.
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
            device = self._cfg.device
            pb_c_base, pb_c_init, discount_factor = self._cfg.pb_c_base, self._cfg.pb_c_init, self._cfg.discount_factor
            # the data storage of hidden states: storing the states of all the ctree nodes
            hidden_state_pool = [hidden_state_roots]
            # 1 x batch x 64
            # the data storage of value prefix hidden states in LSTM
            reward_hidden_state_c_pool = [reward_hidden_state_roots[0]]
            reward_hidden_state_h_pool = [reward_hidden_state_roots[1]]

            # the index of each layer in the ctree
            hidden_state_index_x = 0
            # minimax value storage
            min_max_stats_lst = tree_efficientzero.MinMaxStatsList(num)
            min_max_stats_lst.set_delta(self._cfg.value_delta_max)

            for index_simulation in range(self._cfg.num_simulations):
                hidden_states = []
                hidden_states_c_reward = []
                hidden_states_h_reward = []

                # prepare a result wrapper to transport results between python and c++ parts
                results = tree_efficientzero.ResultsWrapper(num=num)

                # traverse to select actions for each root.
                # hidden_state_index_x_lst: the first index of leaf node states in hidden_state_pool, i.e. the search depth.
                # index hidden_state_index_y_lst: the second index of leaf node states in hidden_state_pool, i.e. the batch root node index, maximum is ``env_num``.
                # the hidden state of the leaf node is hidden_state_pool[x, y]; the index of value prefix hidden state of the leaf node are in the same manner.

                # MCTS stage 1: Each simulation starts from the internal root state s0, and finishes when the
                # simulation reaches a leaf node s_l.
                hidden_state_index_x_lst, hidden_state_index_y_lst, last_actions, virtual_to_play_batch = tree_efficientzero.batch_traverse(
                    roots, pb_c_base, pb_c_init, discount_factor, min_max_stats_lst, results, copy.deepcopy(to_play_batch),
                    self._cfg.model.continuous_action_space
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

                if self._cfg.model.continuous_action_space is True:
                    # continuous action
                    last_actions = torch.from_numpy(np.asarray(last_actions)).to(device).float()

                else:
                    # discrete action
                    last_actions = torch.from_numpy(np.asarray(last_actions)).to(device).long()

                # MCTS stage 2: Expansion: At the final time-step l of the simulation, the reward and state are
                # computed by the dynamics function

                # evaluation for leaf nodes

                # Inside the search ctree we use the dynamics function to obtain the next hidden
                # state given an action and the previous hidden state
                network_output = model.recurrent_inference(
                    hidden_states, (hidden_states_c_reward, hidden_states_h_reward), last_actions
                )
                # TODO(pu)
                if not model.training:
                    # if not in training, obtain the scalars of the value/reward
                    network_output.value = inverse_scalar_transform(
                        network_output.value, self._cfg.model.support_scale
                    ).detach().cpu().numpy()
                    network_output.value_prefix = inverse_scalar_transform(
                        network_output.value_prefix, self._cfg.model.support_scale
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
                assert self._cfg.lstm_horizon_len > 0
                reset_idx = (np.array(search_lens) % self._cfg.lstm_horizon_len == 0)
                assert len(reset_idx) == num
                reward_hidden_state_nodes[0][:, reset_idx, :] = 0
                reward_hidden_state_nodes[1][:, reset_idx, :] = 0
                is_reset_lst = reset_idx.astype(np.int32).tolist()

                reward_hidden_state_c_pool.append(reward_hidden_state_nodes[0])
                reward_hidden_state_h_pool.append(reward_hidden_state_nodes[1])
                hidden_state_index_x += 1

                # MCTS stage 3:
                # Backup: At the end of the simulation, the statistics along the trajectory are updated.

                # backpropagation along the search path to update the attributes
                tree_efficientzero.batch_backpropagate(
                    hidden_state_index_x, discount_factor, value_prefix_pool, value_pool, policy_logits_pool,
                    min_max_stats_lst, results, is_reset_lst, virtual_to_play_batch
                )


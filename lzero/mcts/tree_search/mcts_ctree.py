import copy
from typing import TYPE_CHECKING, List, Any, Union

import numpy as np
import torch
from easydict import EasyDict

from lzero.mcts.ctree.ctree_efficientzero import ez_tree as tree_efficientzero
from lzero.mcts.ctree.ctree_muzero import mz_tree as tree_muzero
from lzero.mcts.ctree.ctree_gumbel_muzero import gmz_tree as tree_gumbel_muzero
from lzero.policy import InverseScalarTransform

if TYPE_CHECKING:
    from lzero.mcts.ctree.ctree_efficientzero import ez_tree as ez_ctree
    from lzero.mcts.ctree.ctree_muzero import mz_tree as mz_ctree
    from lzero.mcts.ctree.ctree_gumbel_muzero import gmz_tree as gmz_ctree

# ==============================================================
# EfficientZero
# ==============================================================


class EfficientZeroMCTSCtree(object):
    """
    Overview:
        MCTSCtree for EfficientZero. The core ``batch_traverse`` and ``batch_backpropagate`` function is implemented in C++.
    Interfaces:
        __init__, search
    """

    config = dict(
        # (float) The alpha value used in the Dirichlet distribution for exploration at the root node of the search tree.
        root_dirichlet_alpha=0.3,
        # (float) The noise weight at the root node of the search tree.
        root_noise_weight=0.25,
        # (int) The base constant used in the PUCT formula for balancing exploration and exploitation during tree search.
        pb_c_base=19652,
        # (float) The initialization constant used in the PUCT formula for balancing exploration and exploitation during tree search.
        pb_c_init=1.25,
        # (float) The maximum change in value allowed during the backup step of the search tree update.
        value_delta_max=0.01,
    )

    @classmethod
    def default_config(cls: type) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    def __init__(self, cfg: EasyDict = None) -> None:
        """
        Overview:
            Use the default configuration mechanism. If a user passes in a cfg with a key that matches an existing key
            in the default configuration, the user-provided value will override the default configuration. Otherwise,
            the default configuration will be used.
        """
        default_config = self.default_config()
        default_config.update(cfg)
        self._cfg = default_config
        self.inverse_scalar_transform_handle = InverseScalarTransform(
            self._cfg.model.support_scale, self._cfg.device, self._cfg.model.categorical_distribution
        )

    @classmethod
    def roots(cls: int, active_collect_env_num: int, legal_actions: List[Any]) -> "ez_ctree.Roots":
        """
        Overview:
            The initialization of CRoots with root num and legal action lists.
        Arguments:
            - root_num (:obj:'int'): the number of the current root.
            - legal_action_list (:obj:'List'): the vector of the legal action of this root.
        """
        from lzero.mcts.ctree.ctree_efficientzero import ez_tree as ctree
        return ctree.Roots(active_collect_env_num, legal_actions)

    def search(
            self, roots: Any, model: torch.nn.Module, latent_state_roots: List[Any],
            reward_hidden_state_roots: List[Any], to_play_batch: Union[int, List[Any]]
    ) -> None:
        """
        Overview:
            Do MCTS for the roots (a batch of root nodes in parallel). Parallel in model inference.
             Use the cpp ctree.
        Arguments:
            - roots (:obj:`Any`): a batch of expanded root nodes
            - latent_state_roots (:obj:`list`): the hidden states of the roots
            - reward_hidden_state_roots (:obj:`list`): the value prefix hidden states in LSTM of the roots
            - to_play_batch (:obj:`list`): the to_play_batch list used in two_player mode board games
        """
        with torch.no_grad():
            model.eval()

            # preparation
            num = roots.num
            pb_c_base, pb_c_init, discount_factor = self._cfg.pb_c_base, self._cfg.pb_c_init, self._cfg.discount_factor
            # the data storage of hidden states: storing the states of all the ctree nodes
            latent_state_pool = [latent_state_roots]
            # 1 x batch x 64
            # the data storage of value prefix hidden states in LSTM
            reward_hidden_state_c_pool = [reward_hidden_state_roots[0]]
            reward_hidden_state_h_pool = [reward_hidden_state_roots[1]]

            # the index of each layer in the ctree
            latent_state_index_x = 0
            # minimax value storage
            min_max_stats_lst = tree_efficientzero.MinMaxStatsList(num)
            min_max_stats_lst.set_delta(self._cfg.value_delta_max)

            for index_simulation in range(self._cfg.num_simulations):
                latent_states = []
                hidden_states_c_reward = []
                hidden_states_h_reward = []

                # prepare a result wrapper to transport results between python and c++ parts
                results = tree_efficientzero.ResultsWrapper(num=num)

                # latent_state_index_x_lst: the first index of leaf node states in latent_state_pool, i.e. the search depth.
                # latent_state_index_y_lst: the second index of leaf node states in latent_state_pool, i.e. the batch root node index, maximum is ``env_num``.
                # the latent state of the leaf node is latent_state_pool[x, y].
                # the index of value prefix hidden state of the leaf node are in the same manner.
                """
                MCTS stage 1: Selection
                    Each simulation starts from the internal root state s0, and finishes when the simulation reaches a leaf node s_l.
                """
                latent_state_index_x_lst, latent_state_index_y_lst, last_actions, virtual_to_play_batch = tree_efficientzero.batch_traverse(
                    roots, pb_c_base, pb_c_init, discount_factor, min_max_stats_lst, results,
                    copy.deepcopy(to_play_batch)
                )
                # obtain the search horizon for leaf nodes
                search_lens = results.get_search_len()

                # obtain the states for leaf nodes
                for ix, iy in zip(latent_state_index_x_lst, latent_state_index_y_lst):
                    latent_states.append(latent_state_pool[ix][iy])
                    hidden_states_c_reward.append(reward_hidden_state_c_pool[ix][0][iy])
                    hidden_states_h_reward.append(reward_hidden_state_h_pool[ix][0][iy])

                latent_states = torch.from_numpy(np.asarray(latent_states)).to(self._cfg.device).float()
                hidden_states_c_reward = torch.from_numpy(np.asarray(hidden_states_c_reward)).to(self._cfg.device
                                                                                                 ).unsqueeze(0)
                hidden_states_h_reward = torch.from_numpy(np.asarray(hidden_states_h_reward)).to(self._cfg.device
                                                                                                 ).unsqueeze(0)
                # .long() only for discrete action
                last_actions = torch.from_numpy(np.asarray(last_actions)).to(self._cfg.device).long()
                """
                MCTS stage 2: Expansion
                    At the final time-step l of the simulation, the next_latent_state and reward/value_prefix are computed by the dynamics function.
                    Then we calculate the policy_logits and value for the leaf node (next_latent_state) by the prediction function. (aka. evaluation)
                """
                network_output = model.recurrent_inference(
                    latent_states, (hidden_states_c_reward, hidden_states_h_reward), last_actions
                )
                if not model.training:
                    # if not in training, obtain the scalars of the value/reward
                    network_output.value = self.inverse_scalar_transform_handle(network_output.value
                                                                                ).detach().cpu().numpy()
                    network_output.value_prefix = self.inverse_scalar_transform_handle(network_output.value_prefix
                                                                                       ).detach().cpu().numpy()
                    network_output.latent_state = network_output.latent_state.detach().cpu().numpy()
                    network_output.reward_hidden_state = (
                        network_output.reward_hidden_state[0].detach().cpu().numpy(),
                        network_output.reward_hidden_state[1].detach().cpu().numpy()
                    )
                    network_output.policy_logits = network_output.policy_logits.detach().cpu().numpy()

                latent_state_nodes = network_output.latent_state
                value_prefix_pool = network_output.value_prefix.reshape(-1).tolist()
                value_pool = network_output.value.reshape(-1).tolist()
                policy_logits_pool = network_output.policy_logits.tolist()
                reward_latent_state_nodes = network_output.reward_hidden_state

                latent_state_pool.append(latent_state_nodes)
                # reset the hidden states in LSTM every horizon steps in search
                # only need to predict the value prefix in a range (eg: s0 -> s5)
                assert self._cfg.lstm_horizon_len > 0
                reset_idx = (np.array(search_lens) % self._cfg.lstm_horizon_len == 0)
                assert len(reset_idx) == num
                reward_latent_state_nodes[0][:, reset_idx, :] = 0
                reward_latent_state_nodes[1][:, reset_idx, :] = 0
                is_reset_lst = reset_idx.astype(np.int32).tolist()

                reward_hidden_state_c_pool.append(reward_latent_state_nodes[0])
                reward_hidden_state_h_pool.append(reward_latent_state_nodes[1])
                latent_state_index_x += 1
                """
                MCTS stage 3: Backup
                    At the end of the simulation, the statistics along the trajectory are updated.
                """
                # backpropagation along the search path to update the attributes
                tree_efficientzero.batch_backpropagate(
                    latent_state_index_x, discount_factor, value_prefix_pool, value_pool, policy_logits_pool,
                    min_max_stats_lst, results, is_reset_lst, virtual_to_play_batch
                )


# ==============================================================
# MuZero
# ==============================================================


class MuZeroMCTSCtree(object):
    """
    Overview:
        MCTSCtree for MuZero. The core ``batch_traverse`` and ``batch_backpropagate`` function is implemented in C++.

    Interfaces:
        __init__, search
    """

    config = dict(
        # (float) The alpha value used in the Dirichlet distribution for exploration at the root node of the search tree.
        root_dirichlet_alpha=0.3,
        # (float) The noise weight at the root node of the search tree.
        root_noise_weight=0.25,
        # (int) The base constant used in the PUCT formula for balancing exploration and exploitation during tree search.
        pb_c_base=19652,
        # (float) The initialization constant used in the PUCT formula for balancing exploration and exploitation during tree search.
        pb_c_init=1.25,
        # (float) The maximum change in value allowed during the backup step of the search tree update.
        value_delta_max=0.01,
    )

    @classmethod
    def default_config(cls: type) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    def __init__(self, cfg: EasyDict = None) -> None:
        """
        Overview:
            Use the default configuration mechanism. If a user passes in a cfg with a key that matches an existing key
            in the default configuration, the user-provided value will override the default configuration. Otherwise,
            the default configuration will be used.
        """
        default_config = self.default_config()
        default_config.update(cfg)
        self._cfg = default_config
        self.inverse_scalar_transform_handle = InverseScalarTransform(
            self._cfg.model.support_scale, self._cfg.device, self._cfg.model.categorical_distribution
        )

    @classmethod
    def roots(cls: int, active_collect_env_num: int, legal_actions: List[Any]) -> "mz_ctree":
        """
        Overview:
            The initialization of CRoots with root num and legal action lists.
        Arguments:
            - root_num (:obj:`int`): the number of the current root.
            - legal_action_list (:obj:`list`): the vector of the legal action of this root.
        """
        from lzero.mcts.ctree.ctree_muzero import mz_tree as ctree
        return ctree.Roots(active_collect_env_num, legal_actions)

    def search(
            self, roots: Any, model: torch.nn.Module, latent_state_roots: List[Any], to_play_batch: Union[int,
                                                                                                          List[Any]]
    ) -> None:
        """
        Overview:
            Do MCTS for the roots (a batch of root nodes in parallel). Parallel in model inference.
             Use the cpp ctree.
        Arguments:
            - roots (:obj:`Any`): a batch of expanded root nodes
            - latent_state_roots (:obj:`list`): the hidden states of the roots
            - to_play_batch (:obj:`list`): the to_play_batch list used in two_player mode board games
        """
        with torch.no_grad():
            model.eval()

            # preparation
            num = roots.num
            pb_c_base, pb_c_init, discount_factor = self._cfg.pb_c_base, self._cfg.pb_c_init, self._cfg.discount_factor
            # the data storage of hidden states: storing the states of all the ctree nodes
            latent_state_pool = [latent_state_roots]

            # the index of each layer in the ctree
            latent_state_index_x = 0
            # minimax value storage
            min_max_stats_lst = tree_muzero.MinMaxStatsList(num)
            min_max_stats_lst.set_delta(self._cfg.value_delta_max)

            for index_simulation in range(self._cfg.num_simulations):
                latent_states = []

                # prepare a result wrapper to transport results between python and c++ parts
                results = tree_muzero.ResultsWrapper(num=num)

                # latent_state_index_x_lst: the first index of leaf node states in latent_state_pool, i.e. the search depth.
                # latent_state_index_y_lst: the second index of leaf node states in latent_state_pool, i.e. the batch root node index, maximum is ``env_num``.
                # the latent state of the leaf node is latent_state_pool[x, y].
                # the index of value prefix hidden state of the leaf node are in the same manner.
                """
                MCTS stage 1: Selection
                    Each simulation starts from the internal root state s0, and finishes when the simulation reaches a leaf node s_l.
                """
                latent_state_index_x_lst, latent_state_index_y_lst, last_actions, virtual_to_play_batch = tree_muzero.batch_traverse(
                    roots, pb_c_base, pb_c_init, discount_factor, min_max_stats_lst, results,
                    copy.deepcopy(to_play_batch)
                )

                # obtain the states for leaf nodes
                for ix, iy in zip(latent_state_index_x_lst, latent_state_index_y_lst):
                    latent_states.append(latent_state_pool[ix][iy])

                latent_states = torch.from_numpy(np.asarray(latent_states)).to(self._cfg.device).float()
                # only for discrete action
                last_actions = torch.from_numpy(np.asarray(last_actions)).to(self._cfg.device).long()
                """
                MCTS stage 2: Expansion
                    At the final time-step l of the simulation, the next_latent_state and reward/value_prefix are computed by the dynamics function.
                    Then we calculate the policy_logits and value for the leaf node (next_latent_state) by the prediction function. (aka. evaluation)
                """
                network_output = model.recurrent_inference(latent_states, last_actions)

                if not model.training:
                    # if not in training, obtain the scalars of the value/reward
                    network_output.value = self.inverse_scalar_transform_handle(network_output.value
                                                                                ).detach().cpu().numpy()
                    network_output.reward = self.inverse_scalar_transform_handle(network_output.reward
                                                                                 ).detach().cpu().numpy()
                    network_output.latent_state = network_output.latent_state.detach().cpu().numpy()
                    network_output.policy_logits = network_output.policy_logits.detach().cpu().numpy()

                latent_state_nodes = network_output.latent_state
                reward_pool = network_output.reward.reshape(-1).tolist()
                value_pool = network_output.value.reshape(-1).tolist()
                policy_logits_pool = network_output.policy_logits.tolist()

                latent_state_pool.append(latent_state_nodes)
                latent_state_index_x += 1
                """
                MCTS stage 3: Backup
                    At the end of the simulation, the statistics along the trajectory are updated.
                """
                # backpropagation along the search path to update the attributes
                tree_muzero.batch_backpropagate(
                    latent_state_index_x, discount_factor, reward_pool, value_pool, policy_logits_pool,
                    min_max_stats_lst, results, virtual_to_play_batch
                )

class GumbelMuZeroMCTSCtree(object):
    config = dict(
        device='cpu',
        support_size=300,
        discount=0.997,
        num_simulations=50,
        lstm_horizon_len=5,
        # (float) The alpha value used in the Dirichlet distribution for exploration at the root node of the search tree.
        root_dirichlet_alpha=0.3,
        # (float) The noise weight at the root node of the search tree.
        root_noise_weight=0.25,
        # (float) The maximum change in value allowed during the backup step of the search tree update.
        value_delta_max=0.01,
    )

    @classmethod
    def default_config(cls: type) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    def __init__(self, cfg: EasyDict = None):
        default_config = self.default_config()
        default_config.update(cfg)
        self._cfg = default_config
        self.inverse_scalar_transform_handle = InverseScalarTransform(
            self._cfg.model.support_scale, self._cfg.device, self._cfg.model.categorical_distribution
        )
    
    @classmethod
    def roots(cls: int, active_collect_env_num: int, legal_actions: List[Any]) -> "mz_ctree":
        """
        Overview:
            The initialization of CRoots with root num and legal action lists.
        Arguments:
            - root_num (:obj:`int`): the number of the current root.
            - legal_action_list (:obj:`list`): the vector of the legal action of this root.
        """
        from lzero.mcts.ctree.ctree_gumbel_muzero import gmz_tree as ctree
        return ctree.Roots(active_collect_env_num, legal_actions)

    def search(self, roots, model, latent_state_roots, to_play_batch):
        """
        Overview:
            Do MCTS for the roots (a batch of root nodes in parallel). Parallel in model inference.
             Use the cpp tree.
        Arguments:
            - roots (:obj:`Any`): a batch of expanded root nodes
            - latent_state_roots (:obj:`list`): the hidden states of the roots
            - to_play_batch (:obj:`list`): the to_play_batch list used in two_player mode board games
        """
        with torch.no_grad():
            model.eval()

            # preparation
            num = roots.num
            device = self._cfg.device
            discount = self._cfg.discount
            # the data storage of hidden states: storing the states of all the tree nodes
            latent_state_pool = [latent_state_roots]

            # the index of each layer in the tree
            latent_state_index_x = 0
            # minimax value storage
            min_max_stats_lst = tree_gumbel_muzero.MinMaxStatsList(num)
            min_max_stats_lst.set_delta(self._cfg.value_delta_max)

            for index_simulation in range(self._cfg.num_simulations):
                latent_states = []

                # prepare a result wrapper to transport results between python and c++ parts
                results = tree_gumbel_muzero.ResultsWrapper(num=num)

                # traverse to select actions for each root
                # hidden_state_index_x_lst: the first index of leaf node states in hidden_state_pool
                # hidden_state_index_y_lst: the second index of leaf node states in hidden_state_pool
                # the hidden state of the leaf node is hidden_state_pool[x, y]; value prefix states are the same
                latent_state_index_x_lst, latent_state_index_y_lst, last_actions, virtual_to_play_batch = tree_gumbel_muzero.batch_traverse(
                    roots, self._cfg.num_simulations, self._cfg.max_num_considered_actions, discount, results, copy.deepcopy(to_play_batch)
                )
                # obtain the search horizon for leaf nodes
                search_lens = results.get_search_len()

                # obtain the states for leaf nodes
                for ix, iy in zip(latent_state_index_x_lst, latent_state_index_y_lst):
                    latent_states.append(latent_state_pool[ix][iy])

                latent_states = torch.from_numpy(np.asarray(latent_states)).to(device).float()
                last_actions = torch.from_numpy(np.asarray(last_actions)).to(device).unsqueeze(1).long()

                # evaluation for leaf nodes
                network_output = model.recurrent_inference(latent_states, last_actions)
                # TODO(pu)
                if not model.training:
                    # if not in training, obtain the scalars of the value/reward
                    network_output.value = self.inverse_scalar_transform_handle(network_output.value
                                                                                ).detach().cpu().numpy()
                    network_output.reward = self.inverse_scalar_transform_handle(network_output.reward
                                                                                ).detach().cpu().numpy()
                    network_output.latent_state = network_output.latent_state.detach().cpu().numpy()
                    network_output.policy_logits = network_output.policy_logits.detach().cpu().numpy()

                latent_state_nodes = network_output.latent_state
                reward_pool = network_output.reward.reshape(-1).tolist()
                value_pool = network_output.value.reshape(-1).tolist()
                policy_logits_pool = network_output.policy_logits.tolist()

                latent_state_pool.append(latent_state_nodes)
                latent_state_index_x += 1

                # backpropagation along the search path to update the attributes
                tree_gumbel_muzero.batch_back_propagate(
                    latent_state_index_x, discount, reward_pool, value_pool, policy_logits_pool,
                    min_max_stats_lst, results, virtual_to_play_batch
                )

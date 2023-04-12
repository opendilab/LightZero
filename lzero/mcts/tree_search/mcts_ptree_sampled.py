import copy
from typing import TYPE_CHECKING, List, Dict, Any, Tuple, Union

import numpy as np
import torch
from easydict import EasyDict

from lzero.policy.scaling_transform import inverse_scalar_transform
from lzero.mcts.ptree import MinMaxStatsList

if TYPE_CHECKING:
    import lzero.mcts.ptree.ptree_sez as ptree

# ==============================================================
# Sampled EfficientZero
# ==============================================================
import lzero.mcts.ptree.ptree_sez as tree_sez


class SampledEfficientZeroMCTSPtree(object):
    """
    Overview:
        MCTSPtree for Sampled EfficientZero. The core ``batch_traverse`` and ``batch_backpropagate`` function is implemented in python.
    Interfaces:
        __init__, search
    """

    # the default_config for SampledEfficientZeroMCTSPtree.
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

    @classmethod
    def roots(
            cls: int, root_num: int, legal_action_lis: List[Any], action_space_size: int, num_of_sampled_actions: int,
            continuous_action_space: bool
    ) -> "ptree.Roots":
        """
        Overview:
            Initialization of CNode with root_num, legal_actions_list, action_space_size, num_of_sampled_actions, continuous_action_space.
        Arguments:
            - root_num (:obj:'int'): the number of the current root.
            - legal_action_lis (:obj:'List'): the vector of the legal action of this root.
            - action_space_size (:obj:'int'): the size of action space of the current env.
            - num_of_sampled_actions (:obj:'int'): the number of sampled actions, i.e. K in the Sampled MuZero papers.
            - continuous_action_space (:obj:'bool'): whether the action space is continous in current env.
        """
        import lzero.mcts.ptree.ptree_sez as ptree
        return ptree.Roots(
            root_num, legal_action_lis, action_space_size, num_of_sampled_actions, continuous_action_space
        )

    def search(
            self,
            roots: Any,
            model: torch.nn.Module,
            latent_state_roots: List[Any],
            reward_hidden_state_roots: List[Any],
            to_play: Union[int, List[Any]] = -1
    ) -> None:
        """
        Overview:
            Do MCTS for the roots (a batch of root nodes in parallel). Parallel in model inference.
            Use the python ctree.
        Arguments:
            - roots (:obj:`Any`): a batch of expanded root nodes
            - latent_state_roots (:obj:`list`): the hidden states of the roots
            - reward_hidden_state_roots (:obj:`list`): the value prefix hidden states in LSTM of the roots
            - to_play (:obj:`list`): the to_play list used in two_player mode board games
        """
        with torch.no_grad():
            model.eval()

            # preparation
            num = roots.num
            pb_c_base, pb_c_init, discount_factor = self._cfg.pb_c_base, self._cfg.pb_c_init, self._cfg.discount_factor
            # the data storage of hidden states: storing the hidden states of all the ctree root nodes
            # latent_state_roots.shape  (2, 12, 3, 3)
            hidden_state_pool = [latent_state_roots]
            # 1 x batch x 64
            # EfficientZero related
            # the data storage of value prefix hidden states in LSTM
            # reward_hidden_state_roots[0].shape  (1, 2, 64)
            reward_hidden_state_c_pool = [reward_hidden_state_roots[0]]
            reward_hidden_state_h_pool = [reward_hidden_state_roots[1]]

            # the index of each layer in the ctree
            latent_state_index_x = 0
            # minimax value storage
            min_max_stats_lst = MinMaxStatsList(num)

            # virtual_to_play = copy.deepcopy(to_play)
            for index_simulation in range(self._cfg.num_simulations):
                """
                each simulation, we expanded a new node, so we have `num_simulations` num of node at most
                """
                latent_states = []
                hidden_states_c_reward = []
                hidden_states_h_reward = []

                # prepare a result wrapper to transport results between python and c++ parts
                results = tree_sez.SearchResults(num=num)

                # latent_state_index_x_lst: the first index of leaf node states in latent_state_pool, i.e. the search depth.
                # latent_state_index_y_lst: the second index of leaf node states in latent_state_pool, i.e. the batch root node index, maximum is ``env_num``.
                # the latent state of the leaf node is latent_state_pool[x, y].
                # the index of value prefix hidden state of the leaf node are in the same manner.

                """
                MCTS stage 1: Selection
                    Each simulation starts from the internal root state s0, and finishes when the simulation reaches a leaf node s_l.
                """
                latent_state_index_x_lst, latent_state_index_y_lst, last_actions, virtual_to_play = tree_sez.batch_traverse(
                    roots, pb_c_base, pb_c_init, discount_factor, min_max_stats_lst, results, copy.deepcopy(to_play),
                    self._cfg.model.continuous_action_space
                )
                # obtain the search horizon for leaf nodes (not expanded)
                search_lens = results.search_lens

                # obtain the states for leaf nodes
                for ix, iy in zip(latent_state_index_x_lst, latent_state_index_y_lst):
                    latent_states.append(hidden_state_pool[ix][iy])  # hidden_state_pool[ix][iy] shape (12,3,3)
                    hidden_states_c_reward.append(
                        reward_hidden_state_c_pool[ix][0][iy]
                    )  # reward_hidden_state_c_pool[ix][0][iy] shape (64,)
                    hidden_states_h_reward.append(
                        reward_hidden_state_h_pool[ix][0][iy]
                    )  # reward_hidden_state_h_pool[ix][0][iy] shape (64,)

                latent_states = torch.from_numpy(np.asarray(latent_states)).to(self._cfg.device).float()
                hidden_states_c_reward = torch.from_numpy(np.asarray(hidden_states_c_reward)
                                                          ).to(self._cfg.device).unsqueeze(0)  # shape (1,1, 64)
                hidden_states_h_reward = torch.from_numpy(np.asarray(hidden_states_h_reward)
                                                          ).to(self._cfg.device).unsqueeze(0)  # shape (1,1, 64)

                if self._cfg.model.continuous_action_space is True:
                    # continuous action
                    last_actions = torch.from_numpy(np.asarray(last_actions)).to(self._cfg.device).float()

                else:
                    # discrete action
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
                    network_output.latent_state = network_output.latent_state.detach().cpu().numpy()
                    network_output.reward_hidden_state = (
                        network_output.reward_hidden_state[0].detach().cpu().numpy(),
                        network_output.reward_hidden_state[1].detach().cpu().numpy()
                    )
                    # if not in training, obtain the scalars of the value/reward
                    network_output.value = inverse_scalar_transform(network_output.value,
                                                                    self._cfg.support_scale).detach().cpu().numpy()
                    network_output.value_prefix = inverse_scalar_transform(
                        network_output.value_prefix, self._cfg.support_scale
                    ).detach().cpu().numpy()

                    network_output.policy_logits = network_output.policy_logits.detach().cpu().numpy()

                latent_state_nodes = network_output.latent_state
                reward_latent_state_nodes = network_output.reward_hidden_state

                value_pool = network_output.value.reshape(-1).tolist()
                value_prefix_pool = network_output.value_prefix.reshape(-1).tolist()
                policy_logits_pool = network_output.policy_logits.tolist()

                hidden_state_pool.append(latent_state_nodes)
                # reset 0
                # reset the hidden states in LSTM every horizon steps in search
                # only need to predict the value prefix in a range (eg: s0 -> s5)
                assert self._cfg.lstm_horizon_len > 0
                reset_idx = (np.array(search_lens) % self._cfg.lstm_horizon_len == 0)

                reward_latent_state_nodes[0][:, reset_idx, :] = 0
                reward_latent_state_nodes[1][:, reset_idx, :] = 0
                is_reset_lst = reset_idx.astype(np.int32).tolist()

                reward_hidden_state_c_pool.append(reward_latent_state_nodes[0])
                reward_hidden_state_h_pool.append(reward_latent_state_nodes[1])
                # increase the index of leaf node
                latent_state_index_x += 1

                """
                MCTS stage 3: Backup
                    At the end of the simulation, the statistics along the trajectory are updated.
                """
                # backpropagation along the search path to update the attributes
                tree_sez.batch_backpropagate(
                    latent_state_index_x, discount_factor, value_prefix_pool, value_pool, policy_logits_pool,
                    min_max_stats_lst, results, is_reset_lst, virtual_to_play
                )

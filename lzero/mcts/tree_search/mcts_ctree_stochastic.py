import copy
from typing import TYPE_CHECKING, List, Any, Union

import numpy as np
import torch
from easydict import EasyDict

from lzero.policy import InverseScalarTransform
from lzero.mcts.ctree.ctree_stochastic_muzero import stochastic_mz_tree


# ==============================================================
# Stochastic MuZero
# ==============================================================


class StochasticMuZeroMCTSCtree(object):
    """
    Overview:
        MCTSCtree for Stochastic MuZero. The core ``batch_traverse`` and ``batch_backpropagate`` function is implemented in C++.

    Interfaces:
        __init__, roots, search
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
    def roots(cls: int, active_collect_env_num: int, legal_actions: List[Any],
              chance_space_size: int = 2) -> "stochastic_mz_tree.Roots":
        """
        Overview:
            The initialization of CRoots with root num and legal action lists.
        Arguments:
            - root_num (:obj:`int`): the number of the current root.
            - legal_action_list (:obj:`list`): the vector of the legal action of this root.
        """
        from lzero.mcts.ctree.ctree_stochastic_muzero import stochastic_mz_tree as ctree
        return ctree.Roots(active_collect_env_num, legal_actions, chance_space_size)

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
            - to_play_batch (:obj:`list`): the to_play_batch list used in in self-play-mode board games
        """
        with torch.no_grad():
            model.eval()

            # preparation some constant
            batch_size = roots.num
            pb_c_base, pb_c_init, discount_factor = self._cfg.pb_c_base, self._cfg.pb_c_init, self._cfg.discount_factor
            # the data storage of latent states: storing the latent state of all the nodes in the search.
            latent_state_batch_in_search_path = [latent_state_roots]

            # minimax value storage
            min_max_stats_lst = stochastic_mz_tree.MinMaxStatsList(batch_size)
            min_max_stats_lst.set_delta(self._cfg.value_delta_max)

            for simulation_index in range(self._cfg.num_simulations):
                # In each simulation, we expanded a new node, so in one search, we have ``num_simulations`` num of nodes at most.

                latent_states = []

                # prepare a result wrapper to transport results between python and c++ parts
                results = stochastic_mz_tree.ResultsWrapper(num=batch_size)

                # latent_state_index_in_search_path: the first index of leaf node states in latent_state_batch_in_search_path, i.e. is current_latent_state_index in one the search.
                # latent_state_index_in_batch: the second index of leaf node states in latent_state_batch_in_search_path, i.e. the index in the batch, whose maximum is ``batch_size``.
                # e.g. the latent state of the leaf node in (x, y) is latent_state_batch_in_search_path[x, y], where x is current_latent_state_index, y is batch_index.
                # The index of value prefix hidden state of the leaf node are in the same manner.
                """
                MCTS stage 1: Selection
                    Each simulation starts from the internal root state s0, and finishes when the simulation reaches a leaf node s_l.
                """
                leaf_node_is_chance, latent_state_index_in_search_path, latent_state_index_in_batch, last_actions, virtual_to_play_batch = stochastic_mz_tree.batch_traverse(
                    roots, pb_c_base, pb_c_init, discount_factor, min_max_stats_lst, results,
                    copy.deepcopy(to_play_batch)
                )

                # obtain the latent state for leaf node
                for ix, iy in zip(latent_state_index_in_search_path, latent_state_index_in_batch):
                    latent_states.append(latent_state_batch_in_search_path[ix][iy])

                latent_states = torch.from_numpy(np.asarray(latent_states)).to(self._cfg.device).float()
                # .long() is only for discrete action
                last_actions = torch.from_numpy(np.asarray(last_actions)).to(self._cfg.device).long()
                """
                MCTS stage 2: Expansion
                    At the final time-step l of the simulation, the next_latent_state and reward/value_prefix are computed by the dynamics function.
                    Then we calculate the policy_logits and value for the leaf node (next_latent_state) by the prediction function. (aka. evaluation)
                MCTS stage 3: Backup
                    At the end of the simulation, the statistics along the trajectory are updated.
                """
                # network_output = model.recurrent_inference(latent_states, last_actions)

                num = len(leaf_node_is_chance)
                leaf_idx_list = list(range(num))
                latent_state_batch = [None] * num
                value_batch = [None] * num
                reward_batch = [None] * num
                policy_logits_batch = [None] * num
                child_is_chance_batch = [None] * num
                chance_nodes_index = []
                decision_nodes_index = []

                for i, leaf_node_is_chance_ in enumerate(leaf_node_is_chance):
                    if leaf_node_is_chance_:
                        chance_nodes_index.append(i)
                    else:
                        decision_nodes_index.append(i)

                def process_nodes(nodes_index, is_chance):
                    # Return early if nodes_index is empty
                    if not nodes_index:
                        return

                    # Slice and stack latent_states and last_actions based on nodes_index
                    latent_states_stack = torch.stack([latent_states[i] for i in nodes_index], dim=0)
                    last_actions_stack = torch.stack([last_actions[i] for i in nodes_index], dim=0)

                    # Pass the stacked batch through the recurrent_inference function
                    network_output_batch = model.recurrent_inference(latent_states_stack,
                                                                     last_actions_stack,
                                                                     afterstate=not is_chance)

                    # Split the batch output into separate nodes
                    latent_state_splits = torch.split(network_output_batch.latent_state, 1, dim=0)
                    value_splits = torch.split(network_output_batch.value, 1, dim=0)
                    reward_splits = torch.split(network_output_batch.reward, 1, dim=0)
                    policy_logits_splits = torch.split(network_output_batch.policy_logits, 1, dim=0)

                    for i, (latent_state, value, reward, policy_logits) in zip(nodes_index,
                                                                               zip(latent_state_splits, value_splits,
                                                                                   reward_splits,
                                                                                   policy_logits_splits)):
                        if not model.training:
                            value = self.inverse_scalar_transform_handle(value).detach().cpu().numpy()
                            reward = self.inverse_scalar_transform_handle(reward).detach().cpu().numpy()
                            latent_state = latent_state.detach().cpu().numpy()
                            policy_logits = policy_logits.detach().cpu().numpy()

                        latent_state_batch[i] = latent_state
                        value_batch[i] = value.reshape(-1).tolist()
                        reward_batch[i] = reward.reshape(-1).tolist()
                        policy_logits_batch[i] = policy_logits.tolist()
                        child_is_chance_batch[i] = is_chance

                process_nodes(chance_nodes_index, True)
                process_nodes(decision_nodes_index, False)
                chance_nodes = chance_nodes_index
                decision_nodes = decision_nodes_index

                value_batch_chance = [value_batch[leaf_idx] for leaf_idx in chance_nodes]
                value_batch_decision = [value_batch[leaf_idx] for leaf_idx in decision_nodes]
                reward_batch_chance = [reward_batch[leaf_idx] for leaf_idx in chance_nodes]
                reward_batch_decision = [reward_batch[leaf_idx] for leaf_idx in decision_nodes]
                policy_logits_batch_chance = [policy_logits_batch[leaf_idx] for leaf_idx in chance_nodes]
                policy_logits_batch_decision = [policy_logits_batch[leaf_idx] for leaf_idx in decision_nodes]

                latent_state_batch = np.concatenate(latent_state_batch, axis=0)
                latent_state_batch_in_search_path.append(latent_state_batch)

                # In ``batch_backpropagate()``, we first expand the leaf node using ``the policy_logits`` and
                # ``reward`` predicted by the model, then perform backpropagation along the search path to update the
                # statistics.

                # NOTE: simulation_index + 1 is very important, which is the depth of the current leaf node.
                current_latent_state_index = simulation_index + 1

                if (len(chance_nodes) > 0):
                    value_batch_chance = np.concatenate(value_batch_chance, axis=0).reshape(-1).tolist()
                    reward_batch_chance = np.concatenate(reward_batch_chance, axis=0).reshape(-1).tolist()
                    policy_logits_batch_chance = np.concatenate(policy_logits_batch_chance, axis=0).tolist()
                    stochastic_mz_tree.batch_backpropagate(
                        current_latent_state_index, discount_factor, reward_batch_chance, value_batch_chance,
                        policy_logits_batch_chance,
                        min_max_stats_lst, results, virtual_to_play_batch, child_is_chance_batch, chance_nodes
                    )
                if (len(decision_nodes) > 0):
                    value_batch_decision = np.concatenate(value_batch_decision, axis=0).reshape(-1).tolist()
                    reward_batch_decision = np.concatenate(reward_batch_decision, axis=0).reshape(-1).tolist()
                    policy_logits_batch_decision = np.concatenate(policy_logits_batch_decision, axis=0).tolist()
                    stochastic_mz_tree.batch_backpropagate(
                        current_latent_state_index, discount_factor, reward_batch_decision, value_batch_decision,
                        policy_logits_batch_decision,
                        min_max_stats_lst, results, virtual_to_play_batch, child_is_chance_batch, decision_nodes
                    )

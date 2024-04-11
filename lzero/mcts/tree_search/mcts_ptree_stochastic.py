import copy
from typing import TYPE_CHECKING, List, Any, Union

import numpy as np
import torch
from easydict import EasyDict

import lzero.mcts.ptree.ptree_stochastic_mz as tree_stochastic_muzero
from lzero.mcts.ptree import MinMaxStatsList
from lzero.policy import InverseScalarTransform

if TYPE_CHECKING:
    import lzero.mcts.ptree.ptree_stochastic_mz as stochastic_mz_ptree


class StochasticMuZeroMCTSPtree(object):
    """
    Overview:
        The Python implementation of MCTS (batch format) for Stochastic MuZero.  \
        It completes the ``roots``and ``search`` methods by calling functions in module ``ptree_stochastic_mz``, \
        which are implemented in Python.
    Interfaces:
        ``__init__``, ``roots``, ``search``

    ..note::
        The benefit of searching for a batch of nodes at the same time is that \
        it can be parallelized during model inference, thus saving time.
    """

    # the default_config for MuZeroMCTSPtree.
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
        """
        Overview:
            A class method that returns a default configuration in the form of an EasyDict object.
        Returns:
            - cfg (:obj:`EasyDict`): The dict of the default configuration.
        """
        # Create a deep copy of the `config` attribute of the class.
        cfg = EasyDict(copy.deepcopy(cls.config))
        # Add a new attribute `cfg_type` to the `cfg` object.
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    def __init__(self, cfg: EasyDict = None) -> None:
        """
        Overview:
            Use the default configuration mechanism. If a user passes in a cfg with a key that matches an existing key
            in the default configuration, the user-provided value will override the default configuration. Otherwise,
            the default configuration will be used.
        Arguments:
            - cfg (:obj:`EasyDict`): The configuration passed in by the user.
        """
        # Get the default configuration.
        default_config = self.default_config()
        # Update the default configuration with the values provided by the user in ``cfg``.
        default_config.update(cfg)
        self._cfg = default_config
        self.inverse_scalar_transform_handle = InverseScalarTransform(
            self._cfg.model.support_scale, self._cfg.device, self._cfg.model.categorical_distribution
        )

    @classmethod
    def roots(cls: int, root_num: int, legal_actions: List[Any]) -> "stochastic_mz_ptree.Roots":
        """
        Overview:
            Initializes a batch of roots to search parallelly later.
        Arguments:
            - root_num (:obj:`int`): the number of the roots in a batch.
            - legal_action_list (:obj:`List[Any]`): the vector of the legal actions for the roots.
        
        ..note::
            The initialization is achieved by the ``Roots`` class from the ``ptree_stochastic_mz`` module.
        """
        return tree_stochastic_muzero.Roots(root_num, legal_actions)

    def search(
            self,
            roots: Any,
            model: torch.nn.Module,
            latent_state_roots: List[Any],
            to_play_batch: Union[int, List[Any]] = -1
    ) -> None:
        """
        Overview:
            Do MCTS for a batch of roots. Parallel in model inference. \
            Use Python to implement the tree search.
        Arguments:
            - roots (:obj:`Any`): a batch of expanded root nodes.
            - latent_state_roots (:obj:`list`): the hidden states of the roots.
            - model (:obj:`torch.nn.Module`): The model used for inference.
            - to_play_batch (:obj:`list`): the to_play_batch list used in in self-play-mode board games.
        
        .. note::
            The core functions ``batch_traverse`` and ``batch_backpropagate`` are implemented in Python.
        """
        with torch.no_grad():
            model.eval()

            # preparation
            num = roots.num
            device = self._cfg.device
            pb_c_base, pb_c_init, discount_factor = self._cfg.pb_c_base, self._cfg.pb_c_init, self._cfg.discount_factor
            # the data storage of hidden states: storing the hidden states of all the ctree root nodes
            # latent_state_roots.shape  (2, 12, 3, 3)
            latent_state_batch_in_search_path = [latent_state_roots]

            # the index of each layer in the ctree
            current_latent_state_index = 0
            # minimax value storage
            min_max_stats_lst = MinMaxStatsList(num)

            for simulation_index in range(self._cfg.num_simulations):
                # In each simulation, we expanded a new node, so in one search, we have ``num_simulations`` num of nodes at most.

                latent_states = []

                # prepare a result wrapper to transport results between python and c++ parts
                results = tree_stochastic_muzero.SearchResults(num=num)

                # latent_state_index_in_search_path: The first index of the latent state corresponding to the leaf node in latent_state_batch_in_search_path, that is, the search depth.
                # latent_state_index_in_batch: The second index of the latent state corresponding to the leaf node in latent_state_batch_in_search_path, i.e. the index in the batch, whose maximum is ``batch_size``.
                # e.g. the latent state of the leaf node in (x, y) is latent_state_batch_in_search_path[x, y], where x is current_latent_state_index, y is batch_index.
                """
                MCTS stage 1: Selection
                    Each simulation starts from the internal root state s0, and finishes when the simulation reaches a leaf node s_l.
                """
                # leaf_nodes, latent_state_index_in_search_path, latent_state_index_in_batch, last_actions, virtual_to_play_batch = tree_stochastic_muzero.batch_traverse(
                #     roots, pb_c_base, pb_c_init, discount_factor, min_max_stats_lst, results, copy.deepcopy(to_play_batch)
                # )
                if self._cfg.env_type == 'not_board_games':
                    results, virtual_to_play_batch = tree_stochastic_muzero.batch_traverse(
                        roots, pb_c_base, pb_c_init, discount_factor, min_max_stats_lst, results, to_play_batch
                    )
                else:
                    # the ``to_play_batch`` is only used in board games, here we need to deepcopy it to avoid changing the original data.
                    results, virtual_to_play_batch = tree_stochastic_muzero.batch_traverse(
                        roots, pb_c_base, pb_c_init, discount_factor, min_max_stats_lst, results, copy.deepcopy(to_play_batch)
                    )

                leaf_nodes, latent_state_index_in_search_path, latent_state_index_in_batch, last_actions = results.nodes, results.latent_state_index_in_search_path, results.latent_state_index_in_batch, results.last_actions

                # obtain the states for leaf nodes
                for ix, iy in zip(latent_state_index_in_search_path, latent_state_index_in_batch):
                    latent_states.append(
                        latent_state_batch_in_search_path[ix][
                            iy])  # latent_state_batch_in_search_path[ix][iy] shape e.g. (64,4,4)

                latent_states = torch.from_numpy(np.asarray(latent_states)).to(device)
                # only for discrete action
                last_actions = torch.from_numpy(np.asarray(last_actions)).to(device).long()
                """
                MCTS stage 2: Expansion
                   At the final time-step l of the simulation, the next_latent_state and reward/value_prefix are computed by the dynamics function.
                   Then we calculate the policy_logits and value for the leaf node (next_latent_state) by the prediction function. (aka. evaluation)
                MCTS stage 3: Backup
                   At the end of the simulation, the statistics along the trajectory are updated.
                """
                
                # network_output = model.recurrent_inference(latent_states, last_actions)
                num = len(leaf_nodes)
                latent_state_batch = [None] * num
                value_batch = [None] * num
                reward_batch = [None] * num
                policy_logits_batch = [None] * num
                child_is_chance_batch = [None] * num
                chance_nodes = []
                decision_nodes = []
                for i, node in enumerate(leaf_nodes):
                    if node.is_chance:
                        chance_nodes.append(i)
                    else:
                        decision_nodes.append(i)

                def process_nodes(node_indices, is_chance):
                    # Return early if node_indices is empty
                    if not node_indices:
                        return
                    # Slice and stack latent_states and last_actions based on node_indices
                    latent_states_stack = torch.stack([latent_states[i] for i in node_indices], dim=0)
                    last_actions_stack = torch.stack([last_actions[i] for i in node_indices], dim=0)

                    # Pass the stacked batch through the recurrent_inference function
                    network_output_batch = model.recurrent_inference(latent_states_stack,
                                                                     last_actions_stack,
                                                                     afterstate=not is_chance)

                    # Split the batch output into separate nodes
                    latent_state_splits = torch.split(network_output_batch.latent_state, 1, dim=0)
                    value_splits = torch.split(network_output_batch.value, 1, dim=0)
                    reward_splits = torch.split(network_output_batch.reward, 1, dim=0)
                    policy_logits_splits = torch.split(network_output_batch.policy_logits, 1, dim=0)

                    for i, (latent_state, value, reward, policy_logits) in zip(node_indices,
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

                process_nodes(chance_nodes, True)
                process_nodes(decision_nodes, False)

                value_batch_chance = [value_batch[leaf_idx] for leaf_idx in chance_nodes]
                value_batch_decision = [value_batch[leaf_idx] for leaf_idx in decision_nodes]
                reward_batch_chance = [reward_batch[leaf_idx] for leaf_idx in chance_nodes]
                reward_batch_decision = [reward_batch[leaf_idx] for leaf_idx in decision_nodes]
                policy_logits_batch_chance = [policy_logits_batch[leaf_idx] for leaf_idx in chance_nodes]
                policy_logits_batch_decision = [policy_logits_batch[leaf_idx] for leaf_idx in decision_nodes]

                latent_state_batch = np.concatenate(latent_state_batch, axis=0)
                latent_state_batch_in_search_path.append(latent_state_batch)
                current_latent_state_index = simulation_index + 1

                if len(chance_nodes) > 0:
                    value_batch_chance = np.concatenate(value_batch_chance, axis=0)
                    reward_batch_chance = np.concatenate(reward_batch_chance, axis=0)
                    policy_logits_batch_chance = np.concatenate(policy_logits_batch_chance, axis=0)
                    tree_stochastic_muzero.batch_backpropagate(
                        current_latent_state_index, discount_factor, reward_batch_chance, value_batch_chance,
                        policy_logits_batch_chance,
                        min_max_stats_lst, results, virtual_to_play_batch, child_is_chance_batch, chance_nodes
                    )
                if len(decision_nodes) > 0:
                    value_batch_decision = np.concatenate(value_batch_decision, axis=0)
                    reward_batch_decision = np.concatenate(reward_batch_decision, axis=0)
                    policy_logits_batch_decision = np.concatenate(policy_logits_batch_decision, axis=0)
                    tree_stochastic_muzero.batch_backpropagate(
                        current_latent_state_index, discount_factor, reward_batch_decision, value_batch_decision,
                        policy_logits_batch_decision,
                        min_max_stats_lst, results, virtual_to_play_batch, child_is_chance_batch, decision_nodes
                    )

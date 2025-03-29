import copy
from typing import TYPE_CHECKING, List, Any, Union

import numpy as np
import torch
from easydict import EasyDict

from lzero.mcts.ctree.ctree_efficientzero import ez_tree as tree_efficientzero
from lzero.mcts.ctree.ctree_gumbel_muzero import gmz_tree as tree_gumbel_muzero
from lzero.mcts.ctree.ctree_muzero import mz_tree as tree_muzero
from lzero.policy import InverseScalarTransform, to_detach_cpu_numpy

if TYPE_CHECKING:
    from lzero.mcts.ctree.ctree_efficientzero import ez_tree as ez_ctree
    from lzero.mcts.ctree.ctree_muzero import mz_tree as mz_ctree
    from lzero.mcts.ctree.ctree_gumbel_muzero import gmz_tree as gmz_ctree


class UniZeroMCTSCtree(object):
    """
    Overview:
        MCTSCtree for UniZero. The core ``batch_traverse`` and ``batch_backpropagate`` function is implemented in C++.

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
        env_type='not_board_games',
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

    # @profile
    def search(
            self, roots: Any, model: torch.nn.Module, latent_state_roots: List[Any], to_play_batch: Union[int,
            List[Any]], timestep: Union[int, List[Any]]
    ) -> None:
        """
        Overview:
            Perform Monte Carlo Tree Search (MCTS) for a batch of root nodes in parallel. 
            This method utilizes the C++ implementation of the tree search for efficiency.

        Arguments:
            - roots (:obj:`Any`): A batch of expanded root nodes.
            - model (:obj:`torch.nn.Module`): The neural network model used for inference.
            - latent_state_roots (:obj:`List[Any]`): The hidden states of the root nodes.
            - to_play_batch (:obj:`Union[int, List[Any]]`): The list of players in self-play mode.
            - timestep (:obj:`Union[int, List[Any]]`): The step index of the environment in one episode.
        """
        with torch.no_grad():
            model.eval()

            # preparation some constant
            batch_size = roots.num
            pb_c_base, pb_c_init, discount_factor = self._cfg.pb_c_base, self._cfg.pb_c_init, self._cfg.discount_factor
            # the data storage of latent states: storing the latent state of all the nodes in the search.
            latent_state_batch_in_search_path = [latent_state_roots]

            # minimax value storage
            min_max_stats_lst = tree_muzero.MinMaxStatsList(batch_size)
            min_max_stats_lst.set_delta(self._cfg.value_delta_max)

            state_action_history = []
            for simulation_index in range(self._cfg.num_simulations):
                # In each simulation, we expanded a new node, so in one search, we have ``num_simulations`` num of nodes at most.
                latent_states = []

                # prepare a result wrapper to transport results between python and c++ parts
                results = tree_muzero.ResultsWrapper(num=batch_size)

                # latent_state_index_in_search_path: the first index of leaf node states in latent_state_batch_in_search_path, i.e. is current_latent_state_index in one the search.
                # latent_state_index_in_batch: the second index of leaf node states in latent_state_batch_in_search_path, i.e. the index in the batch, whose maximum is ``batch_size``.
                # e.g. the latent state of the leaf node in (x, y) is latent_state_batch_in_search_path[x, y], where x is current_latent_state_index, y is batch_index.
                # The index of value prefix hidden state of the leaf node are in the same manner.
                """
                MCTS stage 1: Selection
                    Each simulation starts from the internal root state s0, and finishes when the simulation reaches a leaf node s_l.
                """
                if self._cfg.env_type == 'not_board_games':
                    latent_state_index_in_search_path, latent_state_index_in_batch, last_actions, virtual_to_play_batch = tree_muzero.batch_traverse(
                        roots, pb_c_base, pb_c_init, discount_factor, min_max_stats_lst, results,
                        to_play_batch
                    )
                else:
                    latent_state_index_in_search_path, latent_state_index_in_batch, last_actions, virtual_to_play_batch = tree_muzero.batch_traverse(
                        roots, pb_c_base, pb_c_init, discount_factor, min_max_stats_lst, results,
                        copy.deepcopy(to_play_batch)
                    )

                # obtain the latent state for leaf node
                for ix, iy in zip(latent_state_index_in_search_path, latent_state_index_in_batch):
                    latent_states.append(latent_state_batch_in_search_path[ix][iy])

                latent_states = torch.from_numpy(np.asarray(latent_states)).to(self._cfg.device)
                # TODO: .long() is only for discrete action
                last_actions = torch.from_numpy(np.asarray(last_actions)).to(self._cfg.device).long()

                # Update state_action_history after each simulation
                state_action_history.append((latent_states.detach().cpu().numpy(), last_actions))

                """
                MCTS stage 2: Expansion
                    At the final time-step l of the simulation, the next_latent_state and reward/value_prefix are computed by the dynamics function.
                    Then we calculate the policy_logits and value for the leaf node (next_latent_state) by the prediction function. (aka. evaluation)
                MCTS stage 3: Backup
                    At the end of the simulation, the statistics along the trajectory are updated.
                """
                # search_depth is used for rope in UniZero
                search_depth = results.get_search_len()
                # print(f'simulation_index:{simulation_index}, search_depth:{search_depth}, latent_state_index_in_search_path:{latent_state_index_in_search_path}')
                network_output = model.recurrent_inference(state_action_history, simulation_index, search_depth, timestep)

                network_output.latent_state = to_detach_cpu_numpy(network_output.latent_state)
                network_output.policy_logits = to_detach_cpu_numpy(network_output.policy_logits)
                network_output.value = to_detach_cpu_numpy(self.inverse_scalar_transform_handle(network_output.value))
                network_output.reward = to_detach_cpu_numpy(self.inverse_scalar_transform_handle(network_output.reward))

                latent_state_batch_in_search_path.append(network_output.latent_state)

                # tolist() is to be compatible with cpp datatype.
                reward_batch = network_output.reward.reshape(-1).tolist()
                value_batch = network_output.value.reshape(-1).tolist()
                policy_logits_batch = network_output.policy_logits.tolist()

                # In ``batch_backpropagate()``, we first expand the leaf node using ``the policy_logits`` and
                # ``reward`` predicted by the model, then perform backpropagation along the search path to update the
                # statistics.

                # NOTE: simulation_index + 1 is very important, which is the depth of the current leaf node.
                current_latent_state_index = simulation_index + 1
                tree_muzero.batch_backpropagate(
                    current_latent_state_index, discount_factor, reward_batch, value_batch, policy_logits_batch,
                    min_max_stats_lst, results, virtual_to_play_batch
                )


class MuZeroMCTSCtree(object):
    """
    Overview:
        MCTSCtree for MuZero. The core ``batch_traverse`` and ``batch_backpropagate`` function is implemented in C++.

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
        env_type='not_board_games',
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

    # @profile
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
            min_max_stats_lst = tree_muzero.MinMaxStatsList(batch_size)
            min_max_stats_lst.set_delta(self._cfg.value_delta_max)

            last_latent_state = latent_state_roots
            for simulation_index in range(self._cfg.num_simulations):
                # In each simulation, we expanded a new node, so in one search, we have ``num_simulations`` num of nodes at most.

                latent_states = []

                # prepare a result wrapper to transport results between python and c++ parts
                results = tree_muzero.ResultsWrapper(num=batch_size)

                # latent_state_index_in_search_path: the first index of leaf node states in latent_state_batch_in_search_path, i.e. is current_latent_state_index in one the search.
                # latent_state_index_in_batch: the second index of leaf node states in latent_state_batch_in_search_path, i.e. the index in the batch, whose maximum is ``batch_size``.
                # e.g. the latent state of the leaf node in (x, y) is latent_state_batch_in_search_path[x, y], where x is current_latent_state_index, y is batch_index.
                # The index of value prefix hidden state of the leaf node are in the same manner.
                """
                MCTS stage 1: Selection
                    Each simulation starts from the internal root state s0, and finishes when the simulation reaches a leaf node s_l.
                """
                if self._cfg.env_type == 'not_board_games':
                    latent_state_index_in_search_path, latent_state_index_in_batch, last_actions, virtual_to_play_batch = tree_muzero.batch_traverse(
                        roots, pb_c_base, pb_c_init, discount_factor, min_max_stats_lst, results,
                        to_play_batch
                    )
                else:
                    latent_state_index_in_search_path, latent_state_index_in_batch, last_actions, virtual_to_play_batch = tree_muzero.batch_traverse(
                        roots, pb_c_base, pb_c_init, discount_factor, min_max_stats_lst, results,
                        copy.deepcopy(to_play_batch)
                    )

                # obtain the latent state for leaf node
                for ix, iy in zip(latent_state_index_in_search_path, latent_state_index_in_batch):
                    latent_states.append(latent_state_batch_in_search_path[ix][iy])

                latent_states = torch.from_numpy(np.asarray(latent_states)).to(self._cfg.device)
                # latent_states = torch.from_numpy(np.asarray(latent_states)).to(self._cfg.device).float()
                # TODO: .long() is only for discrete action
                last_actions = torch.from_numpy(np.asarray(last_actions)).to(self._cfg.device).long()

                """
                MCTS stage 2: Expansion
                    At the final time-step l of the simulation, the next_latent_state and reward/value_prefix are computed by the dynamics function.
                    Then we calculate the policy_logits and value for the leaf node (next_latent_state) by the prediction function. (aka. evaluation)
                MCTS stage 3: Backup
                    At the end of the simulation, the statistics along the trajectory are updated.
                """
                network_output = model.recurrent_inference(latent_states, last_actions)  # for classic muzero

                network_output.latent_state = to_detach_cpu_numpy(network_output.latent_state)
                network_output.policy_logits = to_detach_cpu_numpy(network_output.policy_logits)
                network_output.value = to_detach_cpu_numpy(self.inverse_scalar_transform_handle(network_output.value))
                network_output.reward = to_detach_cpu_numpy(self.inverse_scalar_transform_handle(network_output.reward))

                latent_state_batch_in_search_path.append(network_output.latent_state)

                # tolist() is to be compatible with cpp datatype.
                reward_batch = network_output.reward.reshape(-1).tolist()
                value_batch = network_output.value.reshape(-1).tolist()
                policy_logits_batch = network_output.policy_logits.tolist()

                # In ``batch_backpropagate()``, we first expand the leaf node using ``the policy_logits`` and
                # ``reward`` predicted by the model, then perform backpropagation along the search path to update the
                # statistics.

                # NOTE: simulation_index + 1 is very important, which is the depth of the current leaf node.
                current_latent_state_index = simulation_index + 1
                tree_muzero.batch_backpropagate(
                    current_latent_state_index, discount_factor, reward_batch, value_batch, policy_logits_batch,
                    min_max_stats_lst, results, virtual_to_play_batch
                )

    def search_with_reuse(
            self,
            roots: Any,
            model: torch.nn.Module,
            latent_state_roots: List[Any],
            to_play_batch: Union[int, List[Any]],
            true_action_list=None,
            reuse_value_list=None
    ) -> None:
        """
        Overview:
            Perform Monte Carlo Tree Search (MCTS) for the root nodes in parallel. Utilizes the cpp ctree for efficiency.
            Please refer to https://arxiv.org/abs/2404.16364 for more details.
        Arguments:
            - roots (:obj:`Any`): A batch of expanded root nodes.
            - model (:obj:`torch.nn.Module`): The neural network model.
            - latent_state_roots (:obj:`list`): The hidden states of the root nodes.
            - to_play_batch (:obj:`Union[int, list]`): The list or batch indicator for players in self-play mode.
            - true_action_list (:obj:`list`, optional): A list of true actions for reuse.
            - reuse_value_list (:obj:`list`, optional): A list of values for reuse.
        """

        with torch.no_grad():
            model.eval()

            # Initialize constants and variables
            batch_size = roots.num
            pb_c_base, pb_c_init, discount_factor = self._cfg.pb_c_base, self._cfg.pb_c_init, self._cfg.discount_factor
            latent_state_batch_in_search_path = [latent_state_roots]
            min_max_stats_lst = tree_muzero.MinMaxStatsList(batch_size)
            min_max_stats_lst.set_delta(self._cfg.value_delta_max)
            infer_sum = 0

            for simulation_index in range(self._cfg.num_simulations):
                latent_states = []
                temp_actions = []
                no_inference_lst = []
                reuse_lst = []
                results = tree_muzero.ResultsWrapper(num=batch_size)

                # Selection phase: traverse the tree to select a leaf node
                if self._cfg.env_type == 'not_board_games':
                    latent_state_index_in_search_path, latent_state_index_in_batch, last_actions, virtual_to_play_batch = tree_muzero.batch_traverse_with_reuse(
                        roots, pb_c_base, pb_c_init, discount_factor, min_max_stats_lst, results,
                        to_play_batch, true_action_list, reuse_value_list
                    )
                else:
                    latent_state_index_in_search_path, latent_state_index_in_batch, last_actions, virtual_to_play_batch = tree_muzero.batch_traverse_with_reuse(
                        roots, pb_c_base, pb_c_init, discount_factor, min_max_stats_lst, results,
                        copy.deepcopy(to_play_batch), true_action_list, reuse_value_list
                    )

                # Collect latent states and actions for expansion
                for count, (ix, iy) in enumerate(zip(latent_state_index_in_search_path, latent_state_index_in_batch)):
                    if ix != -1:
                        latent_states.append(latent_state_batch_in_search_path[ix][iy])
                        temp_actions.append(last_actions[count])
                    else:
                        no_inference_lst.append(iy)
                    if ix == 0 and last_actions[count] == true_action_list[count]:
                        reuse_lst.append(count)

                length = len(temp_actions)
                latent_states = torch.from_numpy(np.asarray(latent_states)).to(self._cfg.device)
                temp_actions = torch.from_numpy(np.asarray(temp_actions)).to(self._cfg.device).long()

                # Expansion phase: expand the leaf node and evaluate the new node
                if length != 0:
                    network_output = model.recurrent_inference(latent_states, temp_actions)
                    network_output.latent_state = to_detach_cpu_numpy(network_output.latent_state)
                    network_output.policy_logits = to_detach_cpu_numpy(network_output.policy_logits)
                    network_output.value = to_detach_cpu_numpy(
                        self.inverse_scalar_transform_handle(network_output.value))
                    network_output.reward = to_detach_cpu_numpy(
                        self.inverse_scalar_transform_handle(network_output.reward))

                    latent_state_batch_in_search_path.append(network_output.latent_state)
                    reward_batch = network_output.reward.reshape(-1).tolist()
                    value_batch = network_output.value.reshape(-1).tolist()
                    policy_logits_batch = network_output.policy_logits.tolist()
                else:
                    latent_state_batch_in_search_path.append([])
                    reward_batch = []
                    value_batch = []
                    policy_logits_batch = []

                # Backup phase: propagate the evaluation results back through the tree
                current_latent_state_index = simulation_index + 1
                no_inference_lst.append(-1)
                reuse_lst.append(-1)
                tree_muzero.batch_backpropagate_with_reuse(
                    current_latent_state_index, discount_factor, reward_batch, value_batch, policy_logits_batch,
                    min_max_stats_lst, results, virtual_to_play_batch, no_inference_lst, reuse_lst, reuse_value_list
                )
                infer_sum += length

            average_infer = infer_sum / self._cfg.num_simulations
        return length, average_infer


class MuZeroRNNFullObsMCTSCtree(object):
    """
    Overview:
        The C++ implementation of MCTS (batch format) for EfficientZero.  \
        It completes the ``roots``and ``search`` methods by calling functions in module ``ctree_muzero``, \
        which are implemented in C++.
    Interfaces:
        ``__init__``, ``roots``, ``search``
    
    ..note::
        The benefit of searching for a batch of nodes at the same time is that \
        it can be parallelized during model inference, thus saving time.
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
        env_type='not_board_games',
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
    def roots(cls: int, active_collect_env_num: int, legal_actions: List[Any]) -> "ez_ctree.Roots":
        """
        Overview:
            Initializes a batch of roots to search parallel later.
        Arguments:
            - root_num (:obj:`int`): the number of the roots in a batch.
            - legal_action_list (:obj:`List[Any]`): the vector of the legal actions for the roots.
        
        ..note::
            The initialization is achieved by the ``Roots`` class from the ``ctree_muzero`` module.
        """
        return tree_muzero.Roots(active_collect_env_num, legal_actions)

    # @profile
    def search(
            self, roots: Any, model: torch.nn.Module, latent_state_roots: List[Any],
            world_model_latent_history_roots: List[Any], to_play_batch: Union[int, List[Any]], ready_env_id=None,
    ) -> None:
        """
        Overview:
            Do MCTS for a batch of roots. Parallel in model inference. \
            Use C++ to implement the tree search.
        Arguments:
            - roots (:obj:`Any`): a batch of expanded root nodes.
            - latent_state_roots (:obj:`list`): the hidden states of the roots.
            - reward_hidden_state_roots (:obj:`list`): the value prefix hidden states in LSTM of the roots.
            - model (:obj:`torch.nn.Module`): The model used for inference.
            - to_play (:obj:`list`): the to_play list used in self-play-mode board games.
        
        .. note::
            The core functions ``batch_traverse`` and ``batch_backpropagate`` are implemented in C++.
        """
        with torch.no_grad():
            model.eval()

            # preparation some constant
            batch_size = roots.num
            pb_c_base, pb_c_init, discount_factor = self._cfg.pb_c_base, self._cfg.pb_c_init, self._cfg.discount_factor

            # the data storage of latent states: storing the latent state of all the nodes in one search.
            latent_state_batch_in_search_path = [latent_state_roots]
            # the data storage of value prefix hidden states in RNN
            world_model_latent_history_batch = [world_model_latent_history_roots]

            # minimax value storage
            min_max_stats_lst = tree_muzero.MinMaxStatsList(batch_size)
            min_max_stats_lst.set_delta(self._cfg.value_delta_max)

            state_action_history = []
            for simulation_index in range(self._cfg.num_simulations):
                # In each simulation, we expanded a new node, so in one search, we have ``num_simulations`` num of nodes at most.

                latent_states = []
                world_model_latent_history = []

                # prepare a result wrapper to transport results between python and c++ parts
                results = tree_muzero.ResultsWrapper(num=batch_size)

                # latent_state_index_in_search_path: the first index of leaf node states in latent_state_batch_in_search_path, i.e. is current_latent_state_index in one the search.
                # latent_state_index_in_batch: the second index of leaf node states in latent_state_batch_in_search_path, i.e. the index in the batch, whose maximum is ``batch_size``.
                # e.g. the latent state of the leaf node in (x, y) is latent_state_batch_in_search_path[x, y], where x is current_latent_state_index, y is batch_index.
                # The index of value prefix hidden state of the leaf node is in the same manner.
                """
                MCTS stage 1: Selection
                    Each simulation starts from the internal root state s0, and finishes when the simulation reaches a leaf node s_l.
                """
                if self._cfg.env_type == 'not_board_games':
                    latent_state_index_in_search_path, latent_state_index_in_batch, last_actions, virtual_to_play_batch = tree_muzero.batch_traverse(
                        roots, pb_c_base, pb_c_init, discount_factor, min_max_stats_lst, results,
                        to_play_batch
                    )
                else:
                    latent_state_index_in_search_path, latent_state_index_in_batch, last_actions, virtual_to_play_batch = tree_muzero.batch_traverse(
                        roots, pb_c_base, pb_c_init, discount_factor, min_max_stats_lst, results,
                        copy.deepcopy(to_play_batch)
                    )

                # obtain the search horizon for leaf nodes
                search_lens = results.get_search_len()

                # obtain the latent state for leaf node
                for ix, iy in zip(latent_state_index_in_search_path, latent_state_index_in_batch):
                    latent_states.append(latent_state_batch_in_search_path[ix][iy])
                    world_model_latent_history.append(world_model_latent_history_batch[ix][0][iy])

                latent_states = torch.from_numpy(np.asarray(latent_states)).to(self._cfg.device)
                world_model_latent_history = torch.from_numpy(np.asarray(world_model_latent_history)).to(
                    self._cfg.device).unsqueeze(0)

                # TODO: .long() is only for discrete action
                last_actions = torch.from_numpy(np.asarray(last_actions)).to(self._cfg.device).long()

                # NOTE
                state_action_history.append((latent_states.detach().cpu().numpy(), last_actions))

                """
                MCTS stage 2: Expansion
                    At the final time-step l of the simulation, the next_latent_state and reward/value_prefix are computed by the dynamics function.
                    Then we calculate the policy_logits and value for the leaf node (next_latent_state) by the prediction function. (aka. evaluation)
                MCTS stage 3: Backup
                    At the end of the simulation, the statistics along the trajectory are updated.
                """
                # ===================== MuZeroRNN full_obs =====================
                if ready_env_id is None:
                    # for train
                    network_output = model.recurrent_inference(
                        latent_states, world_model_latent_history, last_actions
                    )
                else:
                    # for inference in collector and evaluator
                    network_output = model.recurrent_inference(
                        latent_states, world_model_latent_history, last_actions, ready_env_id=ready_env_id
                    )
                network_output.predict_next_latent_state = to_detach_cpu_numpy(network_output.predict_next_latent_state)
                network_output.policy_logits = to_detach_cpu_numpy(network_output.policy_logits)
                network_output.value = to_detach_cpu_numpy(self.inverse_scalar_transform_handle(network_output.value))
                network_output.value_prefix = to_detach_cpu_numpy(self.inverse_scalar_transform_handle(network_output.value_prefix))
                network_output.reward_hidden_state = network_output.reward_hidden_state.detach().cpu().numpy()
                latent_state_batch_in_search_path.append(network_output.predict_next_latent_state)

                # tolist() is to be compatible with cpp datatype.
                reward_batch = network_output.value_prefix.reshape(-1).tolist()
                value_batch = network_output.value.reshape(-1).tolist()
                policy_logits_batch = network_output.policy_logits.tolist()

                world_model_latent_history = network_output.reward_hidden_state
                world_model_latent_history_batch.append(world_model_latent_history)

                # In ``batch_backpropagate()``, we first expand the leaf node using ``the policy_logits`` and
                # ``reward`` predicted by the model, then perform backpropagation along the search path to update the
                # statistics.

                # NOTE: simulation_index + 1 is very important, which is the depth of the current leaf node.
                current_latent_state_index = simulation_index + 1
                tree_muzero.batch_backpropagate(
                    current_latent_state_index, discount_factor, reward_batch, value_batch, policy_logits_batch,
                    min_max_stats_lst, results, virtual_to_play_batch
                )


class EfficientZeroMCTSCtree(object):
    """
    Overview:
        The C++ implementation of MCTS (batch format) for EfficientZero.  \
        It completes the ``roots``and ``search`` methods by calling functions in module ``ctree_efficientzero``, \
        which are implemented in C++.
    Interfaces:
        ``__init__``, ``roots``, ``search``
    
    ..note::
        The benefit of searching for a batch of nodes at the same time is that \
        it can be parallelized during model inference, thus saving time.
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
    def roots(cls: int, active_collect_env_num: int, legal_actions: List[Any]) -> "ez_ctree.Roots":
        """
        Overview:
            Initializes a batch of roots to search parallelly later.
        Arguments:
            - root_num (:obj:`int`): the number of the roots in a batch.
            - legal_action_list (:obj:`List[Any]`): the vector of the legal actions for the roots.
        
        ..note::
            The initialization is achieved by the ``Roots`` class from the ``ctree_efficientzero`` module.
        """
        return tree_efficientzero.Roots(active_collect_env_num, legal_actions)

    def search(
            self, roots: Any, model: torch.nn.Module, latent_state_roots: List[Any],
            reward_hidden_state_roots: List[Any], to_play_batch: Union[int, List[Any]]
    ) -> None:
        """
        Overview:
            Do MCTS for a batch of roots. Parallel in model inference. \
            Use C++ to implement the tree search.
        Arguments:
            - roots (:obj:`Any`): a batch of expanded root nodes.
            - latent_state_roots (:obj:`list`): the hidden states of the roots.
            - reward_hidden_state_roots (:obj:`list`): the value prefix hidden states in LSTM of the roots.
            - model (:obj:`torch.nn.Module`): The model used for inference.
            - to_play (:obj:`list`): the to_play list used in in self-play-mode board games.
        
        .. note::
            The core functions ``batch_traverse`` and ``batch_backpropagate`` are implemented in C++.
        """
        with torch.no_grad():
            model.eval()

            # preparation some constant
            batch_size = roots.num
            pb_c_base, pb_c_init, discount_factor = self._cfg.pb_c_base, self._cfg.pb_c_init, self._cfg.discount_factor

            # the data storage of latent states: storing the latent state of all the nodes in one search.
            latent_state_batch_in_search_path = [latent_state_roots]
            # the data storage of value prefix hidden states in LSTM
            # print(f"reward_hidden_state_roots[0]={reward_hidden_state_roots[0]}")
            # print(f"reward_hidden_state_roots[1]={reward_hidden_state_roots[1]}")
            reward_hidden_state_c_batch = [reward_hidden_state_roots[0]]
            reward_hidden_state_h_batch = [reward_hidden_state_roots[1]]

            # minimax value storage
            min_max_stats_lst = tree_efficientzero.MinMaxStatsList(batch_size)
            min_max_stats_lst.set_delta(self._cfg.value_delta_max)

            for simulation_index in range(self._cfg.num_simulations):
                # In each simulation, we expanded a new node, so in one search, we have ``num_simulations`` num of nodes at most.

                latent_states = []
                hidden_states_c_reward = []
                hidden_states_h_reward = []

                # prepare a result wrapper to transport results between python and c++ parts
                results = tree_efficientzero.ResultsWrapper(num=batch_size)

                # latent_state_index_in_search_path: the first index of leaf node states in latent_state_batch_in_search_path, i.e. is current_latent_state_index in one the search.
                # latent_state_index_in_batch: the second index of leaf node states in latent_state_batch_in_search_path, i.e. the index in the batch, whose maximum is ``batch_size``.
                # e.g. the latent state of the leaf node in (x, y) is latent_state_batch_in_search_path[x, y], where x is current_latent_state_index, y is batch_index.
                # The index of value prefix hidden state of the leaf node is in the same manner.
                """
                MCTS stage 1: Selection
                    Each simulation starts from the internal root state s0, and finishes when the simulation reaches a leaf node s_l.
                """
                if self._cfg.env_type == 'not_board_games':
                    latent_state_index_in_search_path, latent_state_index_in_batch, last_actions, virtual_to_play_batch = tree_efficientzero.batch_traverse(
                        roots, pb_c_base, pb_c_init, discount_factor, min_max_stats_lst, results,
                        to_play_batch
                    )
                else:
                    # the ``to_play_batch`` is only used in board games, here we need to deepcopy it to avoid changing the original data.
                    latent_state_index_in_search_path, latent_state_index_in_batch, last_actions, virtual_to_play_batch = tree_efficientzero.batch_traverse(
                        roots, pb_c_base, pb_c_init, discount_factor, min_max_stats_lst, results,
                        copy.deepcopy(to_play_batch)
                    )
                # obtain the search horizon for leaf nodes
                search_lens = results.get_search_len()

                # obtain the latent state for leaf node
                for ix, iy in zip(latent_state_index_in_search_path, latent_state_index_in_batch):
                    latent_states.append(latent_state_batch_in_search_path[ix][iy])
                    hidden_states_c_reward.append(reward_hidden_state_c_batch[ix][0][iy])
                    hidden_states_h_reward.append(reward_hidden_state_h_batch[ix][0][iy])

                latent_states = torch.from_numpy(np.asarray(latent_states)).to(self._cfg.device)
                hidden_states_c_reward = torch.from_numpy(np.asarray(hidden_states_c_reward)).to(self._cfg.device
                                                                                                 ).unsqueeze(0)
                hidden_states_h_reward = torch.from_numpy(np.asarray(hidden_states_h_reward)).to(self._cfg.device
                                                                                                 ).unsqueeze(0)
                # .long() is only for discrete action
                last_actions = torch.from_numpy(np.asarray(last_actions)).to(self._cfg.device).long()
                """
                MCTS stage 2: Expansion
                    At the final time-step l of the simulation, the next_latent_state and reward/value_prefix are computed by the dynamics function.
                    Then we calculate the policy_logits and value for the leaf node (next_latent_state) by the prediction function. (aka. evaluation)
                MCTS stage 3: Backup
                    At the end of the simulation, the statistics along the trajectory are updated.
                """
                network_output = model.recurrent_inference(
                    latent_states, (hidden_states_c_reward, hidden_states_h_reward), last_actions
                )

                network_output.latent_state = to_detach_cpu_numpy(network_output.latent_state)
                network_output.policy_logits = to_detach_cpu_numpy(network_output.policy_logits)
                network_output.value = to_detach_cpu_numpy(self.inverse_scalar_transform_handle(network_output.value))
                network_output.value_prefix = to_detach_cpu_numpy(
                    self.inverse_scalar_transform_handle(network_output.value_prefix))

                network_output.reward_hidden_state = (
                    network_output.reward_hidden_state[0].detach().cpu().numpy(),
                    network_output.reward_hidden_state[1].detach().cpu().numpy()
                )

                latent_state_batch_in_search_path.append(network_output.latent_state)
                # tolist() is to be compatible with cpp datatype.
                value_prefix_batch = network_output.value_prefix.reshape(-1).tolist()
                value_batch = network_output.value.reshape(-1).tolist()
                policy_logits_batch = network_output.policy_logits.tolist()

                reward_latent_state_batch = network_output.reward_hidden_state
                # reset the hidden states in LSTM every ``lstm_horizon_len`` steps in one search.
                # which enable the model only need to predict the value prefix in a range (e.g.: [s0,...,s5])
                assert self._cfg.lstm_horizon_len > 0
                reset_idx = (np.array(search_lens) % self._cfg.lstm_horizon_len == 0)
                assert len(reset_idx) == batch_size
                reward_latent_state_batch[0][:, reset_idx, :] = 0
                reward_latent_state_batch[1][:, reset_idx, :] = 0
                is_reset_list = reset_idx.astype(np.int32).tolist()
                reward_hidden_state_c_batch.append(reward_latent_state_batch[0])
                reward_hidden_state_h_batch.append(reward_latent_state_batch[1])

                # In ``batch_backpropagate()``, we first expand the leaf node using ``the policy_logits`` and
                # ``reward`` predicted by the model, then perform backpropagation along the search path to update the
                # statistics.

                # NOTE: simulation_index + 1 is very important, which is the depth of the current leaf node.
                current_latent_state_index = simulation_index + 1
                tree_efficientzero.batch_backpropagate(
                    current_latent_state_index, discount_factor, value_prefix_batch, value_batch, policy_logits_batch,
                    min_max_stats_lst, results, is_reset_list, virtual_to_play_batch
                )

    def search_with_reuse(
            self, roots: Any, model: torch.nn.Module, latent_state_roots: List[Any],
            reward_hidden_state_roots: List[Any], to_play_batch: Union[int, List[Any]],
            true_action_list=None, reuse_value_list=None
    ) -> None:
        """
        Perform Monte Carlo Tree Search (MCTS) for the root nodes in parallel, utilizing model inference in parallel.
        This method uses the cpp ctree for efficiency.
        Please refer to https://arxiv.org/abs/2404.16364 for more details.

        Arguments:
            - roots (:obj:`Any`): A batch of expanded root nodes.
            - model (:obj:`torch.nn.Module`): The model to use for inference.
            - latent_state_roots (:obj:`List[Any]`): The hidden states of the root nodes.
            - reward_hidden_state_roots (:obj:`List[Any]`): The value prefix hidden states in the LSTM of the roots.
            - to_play_batch (:obj:`Union[int, List[Any]]`): The to_play_batch list used in self-play-mode board games.
            - true_action_list (:obj:`Optional[List[Any]]`): List of true actions for reuse.
            - reuse_value_list (:obj:`Optional[List[Any]]`): List of values for reuse.
        Returns:
            None
        """
        with torch.no_grad():
            model.eval()

            batch_size = roots.num
            pb_c_base, pb_c_init, discount_factor = self._cfg.pb_c_base, self._cfg.pb_c_init, self._cfg.discount_factor

            latent_state_batch_in_search_path = [latent_state_roots]
            reward_hidden_state_c_batch = [reward_hidden_state_roots[0]]
            reward_hidden_state_h_batch = [reward_hidden_state_roots[1]]

            min_max_stats_lst = tree_efficientzero.MinMaxStatsList(batch_size)
            min_max_stats_lst.set_delta(self._cfg.value_delta_max)

            infer_sum = 0

            for simulation_index in range(self._cfg.num_simulations):
                latent_states, hidden_states_c_reward, hidden_states_h_reward = [], [], []
                temp_actions, temp_search_lens, no_inference_lst, reuse_lst = [], [], [], []

                results = tree_efficientzero.ResultsWrapper(num=batch_size)

                if self._cfg.env_type == 'not_board_games':
                    latent_state_index_in_search_path, latent_state_index_in_batch, last_actions, virtual_to_play_batch = tree_efficientzero.batch_traverse_with_reuse(
                        roots, pb_c_base, pb_c_init, discount_factor, min_max_stats_lst, results,
                        to_play_batch, true_action_list, reuse_value_list
                    )
                else:
                    latent_state_index_in_search_path, latent_state_index_in_batch, last_actions, virtual_to_play_batch = tree_efficientzero.batch_traverse_with_reuse(
                        roots, pb_c_base, pb_c_init, discount_factor, min_max_stats_lst, results,
                        copy.deepcopy(to_play_batch), true_action_list, reuse_value_list
                    )

                search_lens = results.get_search_len()

                for count, (ix, iy) in enumerate(zip(latent_state_index_in_search_path, latent_state_index_in_batch)):
                    if ix != -1:
                        latent_states.append(latent_state_batch_in_search_path[ix][iy])
                        hidden_states_c_reward.append(reward_hidden_state_c_batch[ix][0][iy])
                        hidden_states_h_reward.append(reward_hidden_state_h_batch[ix][0][iy])
                        temp_actions.append(last_actions[count])
                        temp_search_lens.append(search_lens[count])
                    else:
                        no_inference_lst.append(iy)
                    if ix == 0 and last_actions[count] == true_action_list[count]:
                        reuse_lst.append(count)

                length = len(temp_actions)
                latent_states = torch.tensor(latent_states, device=self._cfg.device)
                hidden_states_c_reward = torch.tensor(hidden_states_c_reward, device=self._cfg.device).unsqueeze(0)
                hidden_states_h_reward = torch.tensor(hidden_states_h_reward, device=self._cfg.device).unsqueeze(0)
                temp_actions = torch.tensor(temp_actions, device=self._cfg.device).long()

                if length != 0:
                    network_output = model.recurrent_inference(
                        latent_states, (hidden_states_c_reward, hidden_states_h_reward), temp_actions
                    )

                    network_output.latent_state = to_detach_cpu_numpy(network_output.latent_state)
                    network_output.policy_logits = to_detach_cpu_numpy(network_output.policy_logits)
                    network_output.value = to_detach_cpu_numpy(
                        self.inverse_scalar_transform_handle(network_output.value))
                    network_output.value_prefix = to_detach_cpu_numpy(
                        self.inverse_scalar_transform_handle(network_output.value_prefix))

                    network_output.reward_hidden_state = (
                        network_output.reward_hidden_state[0].detach().cpu().numpy(),
                        network_output.reward_hidden_state[1].detach().cpu().numpy()
                    )

                    latent_state_batch_in_search_path.append(network_output.latent_state)
                    value_prefix_batch = network_output.value_prefix.reshape(-1).tolist()
                    value_batch = network_output.value.reshape(-1).tolist()
                    policy_logits_batch = network_output.policy_logits.tolist()

                    reward_latent_state_batch = network_output.reward_hidden_state
                    assert self._cfg.lstm_horizon_len > 0
                    reset_idx = (np.array(temp_search_lens) % self._cfg.lstm_horizon_len == 0)
                    reward_latent_state_batch[0][:, reset_idx, :] = 0
                    reward_latent_state_batch[1][:, reset_idx, :] = 0
                    is_reset_list = reset_idx.astype(np.int32).tolist()
                    reward_hidden_state_c_batch.append(reward_latent_state_batch[0])
                    reward_hidden_state_h_batch.append(reward_latent_state_batch[1])
                else:
                    latent_state_batch_in_search_path.append([])
                    value_batch, policy_logits_batch, value_prefix_batch = [], [], []
                    reward_hidden_state_c_batch.append([])
                    reward_hidden_state_h_batch.append([])
                    assert self._cfg.lstm_horizon_len > 0
                    reset_idx = (np.array(search_lens) % self._cfg.lstm_horizon_len == 0)
                    assert len(reset_idx) == batch_size
                    is_reset_list = reset_idx.astype(np.int32).tolist()

                current_latent_state_index = simulation_index + 1
                no_inference_lst.append(-1)
                reuse_lst.append(-1)
                tree_efficientzero.batch_backpropagate_with_reuse(
                    current_latent_state_index, discount_factor, value_prefix_batch, value_batch, policy_logits_batch,
                    min_max_stats_lst, results, is_reset_list, virtual_to_play_batch, no_inference_lst, reuse_lst,
                    reuse_value_list
                )
                infer_sum += length

            average_infer = infer_sum / self._cfg.num_simulations
        return length, average_infer


class GumbelMuZeroMCTSCtree(object):
    """
    Overview:
        The C++ implementation of MCTS (batch format) for  Gumbel MuZero.  \
        It completes the ``roots`` and ``search`` methods by calling functions in module ``ctree_gumbel_muzero``, \
        which are implemented in C++.
    Interfaces:
        ``__init__``, ``roots``, ``search``
    
    ..note::
        The benefit of searching for a batch of nodes at the same time is that \
        it can be parallelized during model inference, thus saving time.
    """
    config = dict(
        # (int) The max limitation of simluation times during the simulation.
        num_simulations=50,
        # (float) The alpha value used in the Dirichlet distribution for exploration at the root node of the search tree.
        root_dirichlet_alpha=0.3,
        # (float) The noise weight at the root node of the search tree.
        root_noise_weight=0.25,
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
            Use the default configuration mechanism. If a user passes in a cfg with a key that matches an existing key \
            in the default configuration, the user-provided value will override the default configuration. Otherwise, \
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
    def roots(cls: int, active_collect_env_num: int, legal_actions: List[Any]) -> "gmz_ctree":
        """
        Overview:
            Initializes a batch of roots to search parallelly later.
        Arguments:
            - root_num (:obj:`int`): the number of the roots in a batch.
            - legal_action_list (:obj:`List[Any]`): the vector of the legal actions for the roots.
        
        ..note::
            The initialization is achieved by the ``Roots`` class from the ``ctree_gumbel_muzero`` module.
        """
        return tree_gumbel_muzero.Roots(active_collect_env_num, legal_actions)

    def search(self, roots: Any, model: torch.nn.Module, latent_state_roots: List[Any], to_play_batch: Union[int, List[Any]]) -> None:
        """
        Overview:
            Do MCTS for a batch of roots. Parallel in model inference. \
            Use C++ to implement the tree search.
        Arguments:
            - roots (:obj:`Any`): a batch of expanded root nodes.
            - latent_state_roots (:obj:`list`): the hidden states of the roots.
            - model (:obj:`torch.nn.Module`): The model used for inference.
            - to_play (:obj:`list`): the to_play list used in in self-play-mode board games.
        
        .. note::
            The core functions ``batch_traverse`` and ``batch_backpropagate`` are implemented in C++.
        """
        with torch.no_grad():
            model.eval()

            # preparation some constant
            batch_size = roots.num
            device = self._cfg.device
            discount_factor = self._cfg.discount_factor
            # the data storage of hidden states: storing the states of all the tree nodes
            latent_state_batch_in_search_path = [latent_state_roots]

            # minimax value storage
            min_max_stats_lst = tree_gumbel_muzero.MinMaxStatsList(batch_size)
            min_max_stats_lst.set_delta(self._cfg.value_delta_max)

            for simulation_index in range(self._cfg.num_simulations):
                # In each simulation, we expanded a new node, so in one search, we have ``num_simulations`` num of nodes at most.

                latent_states = []

                # prepare a result wrapper to transport results between python and c++ parts
                results = tree_gumbel_muzero.ResultsWrapper(num=batch_size)

                # traverse to select actions for each root
                # hidden_state_index_x_lst: the first index of leaf node states in hidden_state_pool
                # hidden_state_index_y_lst: the second index of leaf node states in hidden_state_pool
                # the hidden state of the leaf node is hidden_state_pool[x, y]; value prefix states are the same
                """
                MCTS stage 1: Selection
                    Each simulation starts from the internal root state s0, and finishes when the simulation reaches a leaf node s_l.
                    In gumbel muzero, the action at the root node is selected using the Sequential Halving algorithm, while the action 
                    at the interier node is selected based on the completion of the action values.
                """
                if self._cfg.env_type == 'not_board_games':
                    latent_state_index_in_search_path, latent_state_index_in_batch, last_actions, virtual_to_play_batch = tree_gumbel_muzero.batch_traverse(
                        roots, self._cfg.num_simulations, self._cfg.max_num_considered_actions, discount_factor,
                        results, to_play_batch
                    )
                else:
                    # the ``to_play_batch`` is only used in board games, here we need to deepcopy it to avoid changing the original data.
                    latent_state_index_in_search_path, latent_state_index_in_batch, last_actions, virtual_to_play_batch = tree_gumbel_muzero.batch_traverse(
                        roots, self._cfg.num_simulations, self._cfg.max_num_considered_actions, discount_factor,
                        results, copy.deepcopy(to_play_batch)
                    )

                # obtain the states for leaf nodes
                for ix, iy in zip(latent_state_index_in_search_path, latent_state_index_in_batch):
                    latent_states.append(latent_state_batch_in_search_path[ix][iy])

                latent_states = torch.from_numpy(np.asarray(latent_states)).to(device)
                # .long() is only for discrete action
                last_actions = torch.from_numpy(np.asarray(last_actions)).to(device).unsqueeze(1).long()
                """
                MCTS stage 2: Expansion
                    At the final time-step l of the simulation, the next_latent_state and reward/value_prefix are computed by the dynamics function.
                    Then we calculate the policy_logits and value for the leaf node (next_latent_state) by the prediction function. (aka. evaluation)
                MCTS stage 3: Backup
                    At the end of the simulation, the statistics along the trajectory are updated.
                """
                network_output = model.recurrent_inference(latent_states, last_actions)

                network_output.latent_state = to_detach_cpu_numpy(network_output.latent_state)
                network_output.policy_logits = to_detach_cpu_numpy(network_output.policy_logits)
                network_output.value = to_detach_cpu_numpy(self.inverse_scalar_transform_handle(network_output.value))
                network_output.reward = to_detach_cpu_numpy(self.inverse_scalar_transform_handle(network_output.reward))

                latent_state_batch_in_search_path.append(network_output.latent_state)
                # tolist() is to be compatible with cpp datatype.
                reward_batch = network_output.reward.reshape(-1).tolist()
                value_batch = network_output.value.reshape(-1).tolist()
                policy_logits_batch = network_output.policy_logits.tolist()

                # In ``batch_backpropagate()``, we first expand the leaf node using ``the policy_logits`` and
                # ``reward`` predicted by the model, then perform backpropagation along the search path to update the
                # statistics.

                # NOTE: simulation_index + 1 is very important, which is the depth of the current leaf node.
                current_latent_state_index = simulation_index + 1

                # backpropagation along the search path to update the attributes
                tree_gumbel_muzero.batch_back_propagate(
                    current_latent_state_index, discount_factor, reward_batch, value_batch, policy_logits_batch,
                    min_max_stats_lst, results, virtual_to_play_batch
                )

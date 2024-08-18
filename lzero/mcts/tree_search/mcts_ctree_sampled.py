import copy
from typing import TYPE_CHECKING, List, Any, Union

import numpy as np
import torch
from easydict import EasyDict

from lzero.mcts.ctree.ctree_sampled_efficientzero import ezs_tree as tree_sampled_efficientzero
from lzero.mcts.ctree.ctree_sampled_muzero import smz_tree as tree_sampled_muzero
from lzero.policy import InverseScalarTransform, to_detach_cpu_numpy

if TYPE_CHECKING:
    from lzero.mcts.ctree.ctree_sampled_efficientzero import ezs_tree as ezs_ctree
    from lzero.mcts.ctree.ctree_sampled_muzero import smz_tree as smz_ctree


class SampledUniZeroMCTSCtree(object):
    """
    Overview:
        MCTSCtree for Sampled UniZero. The core ``batch_traverse`` and ``batch_backpropagate`` function is implemented in C++.

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
    def roots(
            cls: int, root_num: int, legal_action_lis: List[Any], action_space_size: int, num_of_sampled_actions: int,
            continuous_action_space: bool
    ) -> "smz_ctree.Roots":
        """
        Overview:
            Initializes a batch of roots to search parallelly later.
        Arguments:
            - root_num (:obj:`int`): the number of the roots in a batch.
            - legal_action_list (:obj:`List[Any]`): the vector of the legal actions for the roots.
            - action_space_size (:obj:'int'): the size of action space of the current env.
            - num_of_sampled_actions (:obj:'int'): the number of sampled actions, i.e. K in the Sampled MuZero paper.
            - continuous_action_space (:obj:'bool'): whether the action space is continous in current env.

        ..note::
            The initialization is achieved by the ``Roots`` class from the ``ctree_sampled_efficientzero`` module.
        """
        return tree_sampled_muzero.Roots(
            root_num, legal_action_lis, action_space_size, num_of_sampled_actions, continuous_action_space
        )

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
            min_max_stats_lst = tree_sampled_muzero.MinMaxStatsList(batch_size)
            min_max_stats_lst.set_delta(self._cfg.value_delta_max)

            state_action_history = []
            for simulation_index in range(self._cfg.num_simulations):
                # In each simulation, we expanded a new node, so in one search, we have ``num_simulations`` num of nodes at most.
                latent_states = []

                # prepare a result wrapper to transport results between python and c++ parts
                results = tree_sampled_muzero.ResultsWrapper(num=batch_size)

                # latent_state_index_in_search_path: the first index of leaf node states in latent_state_batch_in_search_path, i.e. is current_latent_state_index in one the search.
                # latent_state_index_in_batch: the second index of leaf node states in latent_state_batch_in_search_path, i.e. the index in the batch, whose maximum is ``batch_size``.
                # e.g. the latent state of the leaf node in (x, y) is latent_state_batch_in_search_path[x, y], where x is current_latent_state_index, y is batch_index.
                # The index of value prefix hidden state of the leaf node are in the same manner.
                """
                MCTS stage 1: Selection
                    Each simulation starts from the internal root state s0, and finishes when the simulation reaches a leaf node s_l.
                """
                if self._cfg.env_type == 'not_board_games':
                    latent_state_index_in_search_path, latent_state_index_in_batch, last_actions, virtual_to_play_batch = tree_sampled_muzero.batch_traverse(
                        roots, pb_c_base, pb_c_init, discount_factor, min_max_stats_lst, results,
                        to_play_batch, self._cfg.model.continuous_action_space
                    )
                else:
                    latent_state_index_in_search_path, latent_state_index_in_batch, last_actions, virtual_to_play_batch = tree_sampled_muzero.batch_traverse(
                        roots, pb_c_base, pb_c_init, discount_factor, min_max_stats_lst, results,
                        copy.deepcopy(to_play_batch), self._cfg.model.continuous_action_space
                    )

                # obtain the latent state for leaf node
                for ix, iy in zip(latent_state_index_in_search_path, latent_state_index_in_batch):
                    latent_states.append(latent_state_batch_in_search_path[ix][iy])

                latent_states = torch.from_numpy(np.asarray(latent_states)).to(self._cfg.device)
                if self._cfg.model.continuous_action_space is True:
                    # continuous action
                    last_actions = torch.from_numpy(np.asarray(last_actions)).to(self._cfg.device)
                else:
                    # discrete action
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
                # for Sampled UniZero
                network_output = model.recurrent_inference(state_action_history, simulation_index,
                                                           latent_state_index_in_search_path)

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
                tree_sampled_muzero.batch_backpropagate(
                    current_latent_state_index, discount_factor, reward_batch, value_batch, policy_logits_batch,
                    min_max_stats_lst, results, virtual_to_play_batch
                )


class SampledMuZeroMCTSCtree(object):
    """
    Overview:
        The C++ implementation of MCTS (batch format) for Sampled MuZero.  \
        It completes the ``roots``and ``search`` methods by calling functions in module ``ctree_sampled_efficientzero``, \
        which are implemented in C++.
    Interfaces:
        ``__init__``, ``roots``, ``search``

    ..note::
        The benefit of searching for a batch of nodes at the same time is that \
        it can be parallelized during model inference, thus saving time.
    """

    # the default_config for SampledMuZeroMCTSCtree.
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
    def roots(
            cls: int, root_num: int, legal_action_lis: List[Any], action_space_size: int, num_of_sampled_actions: int,
            continuous_action_space: bool
    ) -> "smz_ctree.Roots":
        """
        Overview:
            Initializes a batch of roots to search parallelly later.
        Arguments:
            - root_num (:obj:`int`): the number of the roots in a batch.
            - legal_action_list (:obj:`List[Any]`): the vector of the legal actions for the roots.
            - action_space_size (:obj:'int'): the size of action space of the current env.
            - num_of_sampled_actions (:obj:'int'): the number of sampled actions, i.e. K in the Sampled MuZero paper.
            - continuous_action_space (:obj:'bool'): whether the action space is continous in current env.

        ..note::
            The initialization is achieved by the ``Roots`` class from the ``ctree_sampled_efficientzero`` module.
        """
        return tree_sampled_muzero.Roots(
            root_num, legal_action_lis, action_space_size, num_of_sampled_actions, continuous_action_space
        )

    def search(
            self, roots: Any, model: torch.nn.Module, latent_state_roots: List[Any],
            to_play_batch: Union[int, List[Any]]
    ) -> None:
        """
        Overview:
            Do MCTS for a batch of roots. Parallel in model inference. \
            Use C++ to implement the tree search.
        Arguments:
            - roots (:obj:`Any`): a batch of expanded root nodes.
            - model (:obj:`torch.nn.Module`): The model used for inference.
            - latent_state_roots (:obj:`list`): the hidden states of the roots.
            - to_play (:obj:`list`): the to_play list used in in self-play-mode board games.

        .. note::
            The core functions ``batch_traverse`` and ``batch_backpropagate`` are implemented in C++.
        """
        with torch.no_grad():
            model.eval()

            # preparation some constant
            batch_size = roots.num
            device = self._cfg.device
            pb_c_base, pb_c_init, discount_factor = self._cfg.pb_c_base, self._cfg.pb_c_init, self._cfg.discount_factor

            # the data storage of latent states: storing the latent state of all the nodes in one search.
            latent_state_batch_in_search_path = [latent_state_roots]

            # minimax value storage
            min_max_stats_lst = tree_sampled_muzero.MinMaxStatsList(batch_size)
            min_max_stats_lst.set_delta(self._cfg.value_delta_max)

            for simulation_index in range(self._cfg.num_simulations):
                # In each simulation, we expanded a new node, so in one search, we have ``num_simulations`` num of nodes at most.
                latent_states = []

                # prepare a result wrapper to transport results between python and c++ parts
                results = tree_sampled_muzero.ResultsWrapper(num=batch_size)

                # latent_state_index_in_search_path: the first index of leaf node states in latent_state_batch_in_search_path, i.e. is current_latent_state_index in one the search.
                # latent_state_index_in_batch: the second index of leaf node states in latent_state_batch_in_search_path, i.e. the index in the batch, whose maximum is ``batch_size``.
                # e.g. the latent state of the leaf node in (x, y) is latent_state_batch_in_search_path[x, y], where x is current_latent_state_index, y is batch_index.
                # The index of value prefix hidden state of the leaf node are in the same manner.
                """
                MCTS stage 1: Selection
                    Each simulation starts from the internal root state s0, and finishes when the simulation reaches a leaf node s_l.
                """
                if self._cfg.env_type == 'not_board_games':
                    latent_state_index_in_search_path, latent_state_index_in_batch, last_actions, virtual_to_play_batch = tree_sampled_muzero.batch_traverse(
                        roots, pb_c_base, pb_c_init, discount_factor, min_max_stats_lst, results,
                        to_play_batch, self._cfg.model.continuous_action_space
                    )
                else:
                    # the ``to_play_batch`` is only used in board games, here we need to deepcopy it to avoid changing the original data.
                    latent_state_index_in_search_path, latent_state_index_in_batch, last_actions, virtual_to_play_batch = tree_sampled_muzero.batch_traverse(
                        roots, pb_c_base, pb_c_init, discount_factor, min_max_stats_lst, results,
                        copy.deepcopy(to_play_batch), self._cfg.model.continuous_action_space
                    )

                # obtain the search horizon for leaf nodes
                search_lens = results.get_search_len()
                # obtain the latent state for leaf node
                for ix, iy in zip(latent_state_index_in_search_path, latent_state_index_in_batch):
                    latent_states.append(latent_state_batch_in_search_path[ix][iy])
                latent_states = torch.from_numpy(np.asarray(latent_states)).to(device)

                if self._cfg.model.continuous_action_space is True:
                    # continuous action
                    last_actions = torch.from_numpy(np.asarray(last_actions)).to(device)
                else:
                    # discrete action
                    last_actions = torch.from_numpy(np.asarray(last_actions)).to(device).long()
                """
                MCTS stage 2: Expansion
                    At the final time-step l of the simulation, the next_latent_state and reward/value_prefix are computed by the dynamics function.
                    Then we calculate the policy_logits and value for the leaf node (next_latent_state) by the prediction function. (aka. evaluation)
                MCTS stage 3: Backup
                    At the end of the simulation, the statistics along the trajectory are updated.
                """
                network_output = model.recurrent_inference(
                    latent_states, last_actions
                )

                [
                    network_output.latent_state, network_output.policy_logits, network_output.value,
                    network_output.reward
                ] = to_detach_cpu_numpy(
                    [
                        network_output.latent_state,
                        network_output.policy_logits,
                        self.inverse_scalar_transform_handle(network_output.value),
                        self.inverse_scalar_transform_handle(network_output.reward),
                    ]
                )
                latent_state_batch_in_search_path.append(network_output.latent_state)
                # tolist() is to be compatible with cpp datatype.
                reward_pool = network_output.reward.reshape(-1).tolist()
                value_pool = network_output.value.reshape(-1).tolist()
                policy_logits_pool = network_output.policy_logits.tolist()

                # In ``batch_backpropagate()``, we first expand the leaf node using ``the policy_logits`` and
                # ``reward`` predicted by the model, then perform backpropagation along the search path to update the
                # statistics.

                # NOTE: simulation_index + 1 is very important, which is the depth of the current leaf node.
                current_latent_state_index = simulation_index + 1
                tree_sampled_muzero.batch_backpropagate(
                    current_latent_state_index, discount_factor, reward_pool, value_pool, policy_logits_pool,
                    min_max_stats_lst, results, virtual_to_play_batch
                )


class SampledEfficientZeroMCTSCtree(object):
    """
    Overview:
        The C++ implementation of MCTS (batch format) for Sampled EfficientZero.  \
        It completes the ``roots``and ``search`` methods by calling functions in module ``ctree_sampled_efficientzero``, \
        which are implemented in C++.
    Interfaces:
        ``__init__``, ``roots``, ``search``
    
    ..note::
        The benefit of searching for a batch of nodes at the same time is that \
        it can be parallelized during model inference, thus saving time.
    """

    # the default_config for SampledEfficientZeroMCTSCtree.
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
    def roots(
            cls: int, root_num: int, legal_action_lis: List[Any], action_space_size: int, num_of_sampled_actions: int,
            continuous_action_space: bool
    ) -> "ezs_ctree.Roots":
        """
        Overview:
            Initializes a batch of roots to search parallelly later.
        Arguments:
            - root_num (:obj:`int`): the number of the roots in a batch.
            - legal_action_list (:obj:`List[Any]`): the vector of the legal actions for the roots.
            - action_space_size (:obj:'int'): the size of action space of the current env.
            - num_of_sampled_actions (:obj:'int'): the number of sampled actions, i.e. K in the Sampled MuZero paper.
            - continuous_action_space (:obj:'bool'): whether the action space is continous in current env.
        
        ..note::
            The initialization is achieved by the ``Roots`` class from the ``ctree_sampled_efficientzero`` module.
        """
        return tree_sampled_efficientzero.Roots(
            root_num, legal_action_lis, action_space_size, num_of_sampled_actions, continuous_action_space
        )

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
            - model (:obj:`torch.nn.Module`): The model used for inference.
            - latent_state_roots (:obj:`list`): the hidden states of the roots.
            - reward_hidden_state_roots (:obj:`list`): the value prefix hidden states in LSTM of the roots.
            - to_play (:obj:`list`): the to_play list used in in self-play-mode board games.
        
        .. note::
            The core functions ``batch_traverse`` and ``batch_backpropagate`` are implemented in C++.
        """
        with torch.no_grad():
            model.eval()

            # preparation some constant
            batch_size = roots.num
            device = self._cfg.device
            pb_c_base, pb_c_init, discount_factor = self._cfg.pb_c_base, self._cfg.pb_c_init, self._cfg.discount_factor

            # the data storage of latent states: storing the latent state of all the nodes in one search.
            latent_state_batch_in_search_path = [latent_state_roots]
            # the data storage of value prefix hidden states in LSTM
            reward_hidden_state_c_pool = [reward_hidden_state_roots[0]]
            reward_hidden_state_h_pool = [reward_hidden_state_roots[1]]

            # minimax value storage
            min_max_stats_lst = tree_sampled_efficientzero.MinMaxStatsList(batch_size)
            min_max_stats_lst.set_delta(self._cfg.value_delta_max)

            for simulation_index in range(self._cfg.num_simulations):
                # In each simulation, we expanded a new node, so in one search, we have ``num_simulations`` num of nodes at most.
                latent_states = []
                hidden_states_c_reward = []
                hidden_states_h_reward = []

                # prepare a result wrapper to transport results between python and c++ parts
                results = tree_sampled_efficientzero.ResultsWrapper(num=batch_size)

                # latent_state_index_in_search_path: the first index of leaf node states in latent_state_batch_in_search_path, i.e. is current_latent_state_index in one the search.
                # latent_state_index_in_batch: the second index of leaf node states in latent_state_batch_in_search_path, i.e. the index in the batch, whose maximum is ``batch_size``.
                # e.g. the latent state of the leaf node in (x, y) is latent_state_batch_in_search_path[x, y], where x is current_latent_state_index, y is batch_index.
                # The index of value prefix hidden state of the leaf node are in the same manner.
                """
                MCTS stage 1: Selection
                    Each simulation starts from the internal root state s0, and finishes when the simulation reaches a leaf node s_l.
                """
                if self._cfg.env_type == 'not_board_games':
                    latent_state_index_in_search_path, latent_state_index_in_batch, last_actions, virtual_to_play_batch = tree_sampled_efficientzero.batch_traverse(
                        roots, pb_c_base, pb_c_init, discount_factor, min_max_stats_lst, results,
                        to_play_batch, self._cfg.model.continuous_action_space
                    )
                else:
                    # the ``to_play_batch`` is only used in board games, here we need to deepcopy it to avoid changing the original data.
                    latent_state_index_in_search_path, latent_state_index_in_batch, last_actions, virtual_to_play_batch = tree_sampled_efficientzero.batch_traverse(
                        roots, pb_c_base, pb_c_init, discount_factor, min_max_stats_lst, results,
                        copy.deepcopy(to_play_batch), self._cfg.model.continuous_action_space
                    )

                # obtain the search horizon for leaf nodes
                search_lens = results.get_search_len()
                # obtain the latent state for leaf node
                for ix, iy in zip(latent_state_index_in_search_path, latent_state_index_in_batch):
                    latent_states.append(latent_state_batch_in_search_path[ix][iy])
                    hidden_states_c_reward.append(reward_hidden_state_c_pool[ix][0][iy])
                    hidden_states_h_reward.append(reward_hidden_state_h_pool[ix][0][iy])
                latent_states = torch.from_numpy(np.asarray(latent_states)).to(device)
                hidden_states_c_reward = torch.from_numpy(np.asarray(hidden_states_c_reward)).to(device).unsqueeze(0)
                hidden_states_h_reward = torch.from_numpy(np.asarray(hidden_states_h_reward)).to(device).unsqueeze(0)

                if self._cfg.model.continuous_action_space is True:
                    # continuous action
                    last_actions = torch.from_numpy(np.asarray(last_actions)).to(device)
                else:
                    # discrete action
                    last_actions = torch.from_numpy(np.asarray(last_actions)).to(device).long()
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

                [
                    network_output.latent_state, network_output.policy_logits, network_output.value,
                    network_output.value_prefix
                ] = to_detach_cpu_numpy(
                    [
                        network_output.latent_state,
                        network_output.policy_logits,
                        self.inverse_scalar_transform_handle(network_output.value),
                        self.inverse_scalar_transform_handle(network_output.value_prefix),
                    ]
                )
                network_output.reward_hidden_state = (
                    network_output.reward_hidden_state[0].detach().cpu().numpy(),
                    network_output.reward_hidden_state[1].detach().cpu().numpy()
                )
                latent_state_batch_in_search_path.append(network_output.latent_state)
                # tolist() is to be compatible with cpp datatype.
                value_prefix_pool = network_output.value_prefix.reshape(-1).tolist()
                value_pool = network_output.value.reshape(-1).tolist()
                policy_logits_pool = network_output.policy_logits.tolist()
                reward_latent_state_batch = network_output.reward_hidden_state

                # reset the hidden states in LSTM every ``lstm_horizon_len`` steps in one search.
                # which enable the model only need to predict the value prefix in a range (e.g.: [s0,...,s5]).
                assert self._cfg.lstm_horizon_len > 0
                reset_idx = (np.array(search_lens) % self._cfg.lstm_horizon_len == 0)
                assert len(reset_idx) == batch_size
                reward_latent_state_batch[0][:, reset_idx, :] = 0
                reward_latent_state_batch[1][:, reset_idx, :] = 0
                is_reset_list = reset_idx.astype(np.int32).tolist()
                reward_hidden_state_c_pool.append(reward_latent_state_batch[0])
                reward_hidden_state_h_pool.append(reward_latent_state_batch[1])

                # In ``batch_backpropagate()``, we first expand the leaf node using ``the policy_logits`` and
                # ``reward`` predicted by the model, then perform backpropagation along the search path to update the
                # statistics.

                # NOTE: simulation_index + 1 is very important, which is the depth of the current leaf node.
                current_latent_state_index = simulation_index + 1
                tree_sampled_efficientzero.batch_backpropagate(
                    current_latent_state_index, discount_factor, value_prefix_pool, value_pool, policy_logits_pool,
                    min_max_stats_lst, results, is_reset_list, virtual_to_play_batch
                )

"""
The Node, Roots class and related core functions for Stochastic MuZero.
"""
import math
import random
from typing import List, Dict, Any, Tuple, Union

import numpy as np
import torch

from .minimax import MinMaxStats


class Node:
    """
    Overview:
        The Node class for  Stochastic MuZero. The basic functions of the node are implemented.
    Interfaces:
        ``__init__``, ``expand``, ``add_exploration_noise``, ``compute_mean_q``, ``get_trajectory``, \
        ``get_children_distribution``, ``get_child``, ``expanded``, ``value``.
    """

    def __init__(self, prior: float, legal_actions: List = None, action_space_size: int = 9, is_chance: bool = False, chance_space_size: int = 2) -> None:
        """
        Overview:
            Initializes a Node instance.
        Arguments:
            - prior (:obj:`float`): The prior probability of the node.
            - legal_actions (:obj:`List`, optional): The list of legal actions of the node. Defaults to None.
            - action_space_size (:obj:`int`, optional): The size of the action space. Defaults to 9.
            - is_chance (:obj:`bool`) Whether the node is a chance node.
        """
        self.prior = prior
        self.legal_actions = legal_actions
        self.action_space_size = action_space_size

        self.visit_count = 0
        self.value_sum = 0
        self.best_action = -1
        self.to_play = 0  # default 0 means play_with_bot_mode
        self.reward = 0
        self.value_prefix = 0.0
        self.children = {}
        self.children_index = []
        self.latent_state_index_in_search_path = 0
        self.latent_state_index_in_batch = 0
        self.parent_value_prefix = 0  # only used in update_tree_q method

        self.is_chance = is_chance
        self.chance_space_size = chance_space_size

    def expand(
            self, to_play: int, latent_state_index_in_search_path: int, latent_state_index_in_batch: int, reward: float,
            policy_logits: List[float], child_is_chance: bool = True
    ) -> None:
        """
        Overview:
            Expand the child nodes of the current node.
        Arguments:
            - to_play (:obj:`int`): which player to play the game in the current node.
            - latent_state_index_in_search_path (:obj:`int`): the x/first index of latent state vector of the current node, i.e. the search depth.
            - latent_state_index_in_batch (:obj:`int`): the y/second index of latent state vector of the current node, i.e. the index of batch root node, its maximum is ``batch_size``/``env_num``.
            - reward: (:obj:`float`): the value prefix of the current node.
            - policy_logits: (:obj:`List`): the policy logit of the child nodes.
        """
        self.to_play = to_play
        self.reward = reward

        if self.is_chance is True:
            child_is_chance = False
            self.reward = 0.0

            if self.legal_actions is None:
                self.legal_actions = np.arange(self.chance_space_size)
            self.latent_state_index_in_search_path = latent_state_index_in_search_path
            self.latent_state_index_in_batch = latent_state_index_in_batch
            policy_values = torch.softmax(torch.tensor([policy_logits[a] for a in self.legal_actions]), dim=0).tolist()
            policy = {legal_action: policy_values[index] for index, legal_action in enumerate(self.legal_actions)}
            for action, prior in policy.items():
                self.children[action] = Node(prior, is_chance=child_is_chance)
        else:
            child_is_chance = True
            self.legal_actions = np.arange(len(policy_logits))
            self.latent_state_index_in_search_path = latent_state_index_in_search_path
            self.latent_state_index_in_batch = latent_state_index_in_batch
            policy_values = torch.softmax(torch.tensor([policy_logits[a] for a in self.legal_actions]), dim=0).tolist()
            policy = {legal_action: policy_values[index] for index, legal_action in enumerate(self.legal_actions)}
            for action, prior in policy.items():
                self.children[action] = Node(prior, is_chance=child_is_chance)

    def add_exploration_noise(self, exploration_fraction: float, noises: List[float]) -> None:
        """
        Overview:
            Add exploration noise to the priors.
        Arguments:
            - exploration_fraction (:obj:`float`): The fraction of exploration noise to be added.
            - noises (:obj:`List[float]`): The list of noises to be added to the priors.
        """
        for i, a in enumerate(self.legal_actions):
            """
            i in index, a is action, e.g. self.legal_actions = [0,1,2,4,6,8], i=[0,1,2,3,4,5], a=[0,1,2,4,6,8]
            """
            try:
                noise = noises[i]
            except Exception as error:
                print(error)
            child = self.get_child(a)
            prior = child.prior
            child.prior = prior * (1 - exploration_fraction) + noise * exploration_fraction

    def compute_mean_q(self, is_root: bool, parent_q: float, discount_factor: float) -> float:
        """
        Overview:
            Compute the mean of the action values of all legal actions.
        Arguments:
            - is_root (:obj:`bool`): Whether the current node is a root node.
            - parent_q (:obj:`float`): The q value of the parent node.
            - discount_factor (:obj:`float`): The discount factor of the reward.
        Returns:
            - mean_q (:obj:`float`): The mean of the action values.
        """
        total_unsigned_q = 0.0
        total_visits = 0
        for a in self.legal_actions:
            child = self.get_child(a)
            if child.visit_count > 0:
                true_reward = child.reward
                # TODO(pu): only one step bootstrap?
                q_of_s_a = true_reward + discount_factor * child.value
                total_unsigned_q += q_of_s_a
                total_visits += 1
        if is_root and total_visits > 0:
            mean_q = total_unsigned_q / total_visits
        else:
            # if is not root node,
            # TODO(pu): why parent_q?
            mean_q = (parent_q + total_unsigned_q) / (total_visits + 1)
        return mean_q

    def get_trajectory(self) -> List[Union[int, float]]:
        """
        Overview:
            Find the current best trajectory starting from the current node.
        Returns:
            - traj (:obj:`List[Union[int, float]]`): A vector of node indices representing the current best trajectory.
        """
        # TODO(pu): best action
        traj = []
        node = self
        best_action = node.best_action
        while best_action >= 0:
            traj.append(best_action)

            node = node.get_child(best_action)
            best_action = node.best_action
        return traj

    def get_children_distribution(self) -> List[Union[int, float]]:
        """
        Overview:
            Get the distribution of visit counts among the child nodes.
        Returns:
            - distribution (:obj:`List[Union[int, float]]` or :obj:`None`): The distribution of visit counts among the children nodes. \
              If the legal_actions list is empty, returns None.
        """
        if self.legal_actions == []:
            return None
        distribution = {a: 0 for a in self.legal_actions}
        if self.expanded:
            for a in self.legal_actions:
                child = self.get_child(a)
                distribution[a] = child.visit_count
            # only take the visit counts
            distribution = [v for k, v in distribution.items()]
        return distribution

    def get_child(self, action: Union[int, float]) -> "Node":
        """
        Overview:
            Get the child node according to the input action.
        Arguments:
            - action (:obj:`Union[int, float]`): The action for which the child node is to be retrieved.
        Returns:
            - child (:obj:`Node`): The child node corresponding to the input action.
        """
        if not isinstance(action, np.int64):
            action = int(action)
        return self.children[action]

    @property
    def expanded(self) -> bool:
        """
        Overview:
            Check if the node has been expanded.
        Returns:
            - expanded (:obj:`bool`): True if the node has been expanded, False otherwise.
        """
        return len(self.children) > 0

    @property
    def value(self) -> float:
        """
        Overview:
            Return the estimated value of the current node.
        Returns:
            - value (:obj:`float`): The estimated value of the current node.
        """
        if self.visit_count == 0:
            return 0
        else:
            return self.value_sum / self.visit_count


class Roots:
    """
    Overview:
        The class to process a batch of roots(Node instances) at the same time.
    Interfaces:
        ``__init__``, ``prepare``,  ``prepare_no_noise``, ``clear``, ``get_trajectories``, \
        ``get_distributions``, ``get_values``
    """

    def __init__(self, root_num: int, legal_actions_list: List) -> None:
        """
        Overview:
            Initializes an instance of the Roots class with the specified number of roots and legal actions.
        Arguments:
            - root_num (:obj:`int`): The number of roots.
            - legal_actions_list(:obj:`List`): A list of the legal actions for each root.
        """
        self.num = root_num
        self.root_num = root_num
        self.legal_actions_list = legal_actions_list  # list of list

        self.roots = []
        for i in range(self.root_num):
            if isinstance(legal_actions_list, list):
                self.roots.append(Node(0, legal_actions_list[i]))
            else:
                # if legal_actions_list is int
                self.roots.append(Node(0, np.arange(legal_actions_list)))

    def prepare(
            self,
            root_noise_weight: float,
            noises: List[float],
            rewards: List[float],
            policies: List[List[float]],
            to_play: int = -1
    ) -> None:
        """
        Overview:
            Expand the roots and add noises for exploration.
        Arguments:
            - root_noise_weight (:obj:`float`): the exploration fraction of roots
            - noises (:obj:`List[float]`): the vector of noise add to the roots.
            - rewards (:obj:`List[float]`): the vector of rewards of each root.
            - policies (:obj:`List[List[float]]`): the vector of policy logits of each root.
            - to_play(:obj:`List`): The vector of the player side of each root.
        """
        for i in range(self.root_num):
            #  to_play: int, latent_state_index_in_search_path: int, latent_state_index_in_batch: int,
            if to_play is None:
                # TODO(pu): why latent_state_index_in_search_path=0, latent_state_index_in_batch=i?
                self.roots[i].expand(-1, 0, i, rewards[i], policies[i])
            else:
                self.roots[i].expand(to_play[i], 0, i, rewards[i], policies[i])

            self.roots[i].add_exploration_noise(root_noise_weight, noises[i])
            self.roots[i].visit_count += 1

    def prepare_no_noise(self, rewards: List[float], policies: List[List[float]], to_play: int = -1) -> None:
        """
        Overview:
            Expand the roots without noise.
        Arguments:
            - rewards (:obj:`List[float]`): the vector of rewards of each root.
            - policies (:obj:`List[List[float]]`): the vector of policy logits of each root.
            - to_play(:obj:`List`): The vector of the player side of each root.
        """
        for i in range(self.root_num):
            if to_play is None:
                self.roots[i].expand(-1, 0, i, rewards[i], policies[i])
            else:
                self.roots[i].expand(to_play[i], 0, i, rewards[i], policies[i])

            self.roots[i].visit_count += 1

    def clear(self) -> None:
        """
        Overview:
            Clear all the roots in the list.
        """
        self.roots.clear()

    def get_trajectories(self) -> List[List[Union[int, float]]]:
        """
        Overview:
            Find the current best trajectory starts from each root.
        Returns:
            - traj (:obj:`List[List[Union[int, float]]]`): a vector of node index, which is the current best trajectory from each root.
        """
        trajs = []
        for i in range(self.root_num):
            trajs.append(self.roots[i].get_trajectory())
        return trajs

    def get_distributions(self) -> List[List[Union[int, float]]]:
        """
        Overview:
            Get the visit count distribution of child nodes for each root.
        Returns:
            - distribution (:obj:`List[List[Union[int, float]]]`): a vector of distribution of child nodes in the format of visit count (i.e. [1,3,0,2,5]).
        """
        distributions = []
        for i in range(self.root_num):
            distributions.append(self.roots[i].get_children_distribution())

        return distributions

    def get_values(self) -> List[float]:
        """
        Overview:
            Get the estimated value of each root.
        Returns:
            - values (:obj:`List[float]`): The estimated value of each root.
        """
        values = []
        for i in range(self.root_num):
            values.append(self.roots[i].value)
        return values


class SearchResults:
    """
    Overview:
        The class to record the results of the simulations for the batch of roots.
    Interfaces:
        ``__init__``.
    """

    def __init__(self, num: int) -> None:
        """
        Overview:
            Initiaizes the attributes to be recorded.
        Arguments:
            -num (:obj:`int`): The number of search results(equal to ``batch_size``).
        """
        self.num = num
        self.nodes = []
        self.search_paths = []
        self.latent_state_index_in_search_path = []
        self.latent_state_index_in_batch = []
        self.last_actions = []
        self.search_lens = []

def select_child(
        node: Node, min_max_stats: MinMaxStats, pb_c_base: float, pb_c_int: float, discount_factor: float,
        mean_q: float, players: int
) -> Union[int, float]:
    """
    Overview:
        Select the child node of the roots according to ucb scores.
    Arguments:
        - node(:obj:`Node`): The root to select the child node.
        - min_max_stats (:obj:`MinMaxStats`):  A tool used to min-max normalize the score.
        - pb_c_base (:obj:`float`): Constant c1 used in pUCT rule, typically 1.25.
        - pb_c_int (:obj:`float`): Constant c2 used in pUCT rule, typically 19652.
        - discount_factor (:obj:`float`): The discount factor used in calculating bootstrapped value, if env is board_games, we set discount_factor=1.
        - mean_q (:obj:`float`): The mean q value of the parent node.
        - players (:obj:`int`): The number of players. In two-player games such as board games, the value need to be negatived when backpropating.
    Returns:
        - action (:obj:`Union[int, float]`): Choose the action with the highest ucb score.
    """

    if node.is_chance:
        # print("root->is_chance: True ")

        # If the node is chance node, we sample from the prior outcome distribution.
        outcomes, probs = zip(*[(o, n.prior) for o, n in node.children.items()])
        outcome = np.random.choice(outcomes, p=probs)
        # print(outcome, probs)
        return outcome

    # print("root->is_chance: False ")
    # If the node is decision node, we select the action with the highest ucb score.
    max_score = -np.inf
    epsilon = 0.000001
    max_index_lst = []
    for a in node.legal_actions:
        child = node.get_child(a)
        temp_score = compute_ucb_score(
            child, min_max_stats, mean_q, node.visit_count, pb_c_base, pb_c_int, discount_factor, players
        )
        if max_score < temp_score:
            max_score = temp_score
            max_index_lst.clear()
            max_index_lst.append(a)
        elif temp_score >= max_score - epsilon:
            # TODO(pu): if the difference is less than epsilon = 0.000001, we random choice action from  max_index_lst
            max_index_lst.append(a)

    action = 0
    if len(max_index_lst) > 0:
        action = random.choice(max_index_lst)
    return action


def compute_ucb_score(
        child: Node,
        min_max_stats: MinMaxStats,
        parent_mean_q: float,
        total_children_visit_counts: float,
        pb_c_base: float,
        pb_c_init: float,
        discount_factor: float,
        players: int = 1,
) -> float:
    """
    Overview:
        Compute the ucb score of the child.
    Arguments:
        - child (:obj:`Node`): the child node to compute ucb score.
        - min_max_stats (:obj:`MinMaxStats`): a tool used to min-max normalize the score.
        - parent_mean_q (:obj:`float`): the mean q value of the parent node.
        - total_children_visit_counts (:obj:`float`): the total visit counts of the child nodes of the parent node.
        - pb_c_base (:obj:`float`): constants c2 in muzero.
        - pb_c_init (:obj:`float`): constants c1 in muzero.
        - disount_factor (:obj:`float`): the discount factor of reward.
        - players (:obj:`int`): the number of players.
    Returns:
        - ucb_value (:obj:`float`): the ucb score of the child.
    """
    # Compute the prior score.
    pb_c = math.log((total_children_visit_counts + pb_c_base + 1) / pb_c_base) + pb_c_init
    pb_c *= (math.sqrt(total_children_visit_counts) / (child.visit_count + 1))
    prior_score = pb_c * child.prior

    # Compute the value score.
    if child.visit_count == 0:
        value_score = parent_mean_q
    else:
        true_reward = child.reward
        if players == 1:
            value_score = true_reward + discount_factor * child.value
        elif players == 2:
            value_score = true_reward + discount_factor * (-child.value)

    # Normalize the value score.
    value_score = min_max_stats.normalize(value_score)
    if value_score < 0:
        value_score = 0
    if value_score > 1:
        value_score = 1
    
    ucb_score = prior_score + value_score

    return ucb_score


def batch_traverse(
        roots: Any,
        pb_c_base: float,
        pb_c_init: float,
        discount_factor: float,
        min_max_stats_lst: List[MinMaxStats],
        results: SearchResults,
        virtual_to_play: List,
) -> Tuple[Any, Any]:

    """
    Overview:
        traverse, also called expansion. Process a batch roots at once.
    Arguments:
        - roots (:obj:`Any`): A batch of root nodes to be expanded.
        - pb_c_base (:obj:`float`): Constant c1 used in pUCT rule, typically 1.25.
        - pb_c_init (:obj:`float`): Constant c2 used in pUCT rule, typically 19652.
        - discount_factor (:obj:`float`): The discount factor used in calculating bootstrapped value, if env is board_games, we set discount_factor=1.
        - results (:obj:`SearchResults`): An instance to record the simulation results for all the roots in the batch.
        - virtual_to_play (:obj:`list`): The to_play list used in self_play collecting and training in board games,
            `virtual` is to emphasize that actions are performed on an imaginary hidden state.
    Returns:
        - latent_state_index_in_search_path (:obj:`list`): The list of x/first index of hidden state vector of the searched node, i.e. the search depth.
        - latent_state_index_in_batch (:obj:`list`): The list of y/second index of hidden state vector of the searched node, i.e. the index of batch root node, its maximum is ``batch_size``/``env_num``.
        - last_actions (:obj:`list`): The action performed by the previous node.
        - virtual_to_play (:obj:`list`): The to_play list used in self_play collecting and trainin gin board games,
            `virtual` is to emphasize that actions are performed on an imaginary hidden state.
    """
    parent_q = 0.0
    results.search_lens = [None for i in range(results.num)]
    results.last_actions = [None for i in range(results.num)]

    results.nodes = [None for i in range(results.num)]
    results.latent_state_index_in_search_path = [None for i in range(results.num)]
    results.latent_state_index_in_batch = [None for i in range(results.num)]
    if virtual_to_play in [1, 2] or virtual_to_play[0] in [1, 2]:
        players = 2
    elif virtual_to_play in [-1, None] or virtual_to_play[0] in [-1, None]:
        players = 1

    results.search_paths = {i: [] for i in range(results.num)}
    for i in range(results.num):
        node = roots.roots[i]
        is_root = 1
        search_len = 0
        results.search_paths[i].append(node)

        """
        MCTS stage 1: Selection
            Each simulation starts from the internal root state s0, and finishes when the simulation reaches a leaf node s_l.
        """
        # the leaf node is not expanded
        while node.expanded:
            mean_q = node.compute_mean_q(is_root, parent_q, discount_factor)
            is_root = 0
            parent_q = mean_q

            # select action according to the pUCT rule.
            action = select_child(
                node, min_max_stats_lst.stats_lst[i], pb_c_base, pb_c_init, discount_factor, mean_q, players
            )
            if players == 2:
                # Players play turn by turn
                if virtual_to_play[i] == 1:
                    virtual_to_play[i] = 2
                else:
                    virtual_to_play[i] = 1

            node.best_action = action
            # move to child node according to selected action.
            node = node.get_child(action)

            last_action = action

            results.search_paths[i].append(node)
            search_len += 1

            # note this return the parent node of the current searched node
            parent = results.search_paths[i][len(results.search_paths[i]) - 1 - 1]
            results.latent_state_index_in_search_path[i] = parent.latent_state_index_in_search_path
            results.latent_state_index_in_batch[i] = parent.latent_state_index_in_batch
            results.last_actions[i] = last_action
            results.search_lens[i] = search_len
            # while we break out the while loop, results.nodes[i] save the leaf node.
            results.nodes[i] = node

    # print(f'env {i} one simulation done!')
    return results, virtual_to_play


def backpropagate(
        search_path: List[Node], min_max_stats: MinMaxStats, to_play: int, value: float, discount_factor: float
) -> None:
    """
    Overview:
        Update the value sum and visit count of nodes along the search path.
    Arguments:
        - search_path (:obj:`List[Node]`): a vector of nodes on the search path.
        - min_max_stats (:obj:`MinMaxStats`): a tool used to min-max normalize the q value.
        - to_play (:obj:`int`): which player to play the game in the current node.
        - value (:obj:`float`): the value to propagate along the search path.
        - discount_factor (:obj:`float`): the discount factor of reward.
    """
    assert to_play is None or to_play in [-1, 1, 2], to_play
    if to_play is None or to_play == -1:
        # for play-with-bot mode
        bootstrap_value = value
        path_len = len(search_path)
        for i in range(path_len - 1, -1, -1):
            node = search_path[i]
            node.value_sum += bootstrap_value
            node.visit_count += 1
            true_reward = node.reward
            min_max_stats.update(true_reward + discount_factor * node.value)
            bootstrap_value = true_reward + discount_factor * bootstrap_value
    else:
        # for self-play-mode
        bootstrap_value = value
        path_len = len(search_path)
        for i in range(path_len - 1, -1, -1):
            node = search_path[i]
            # to_play related
            node.value_sum += bootstrap_value if node.to_play == to_play else -bootstrap_value

            node.visit_count += 1

            # NOTE: in self-play-mode,
            # we should calculate the true_reward according to the perspective of current player of node
            # true_reward = node.value_prefix - (- parent_value_prefix)
            true_reward = node.reward

            # min_max_stats.update(true_reward + discount_factor * node.value)
            min_max_stats.update(true_reward + discount_factor * -node.value)

            # TODO(pu): to_play related
            # true_reward is in the perspective of current player of node
            bootstrap_value = (-true_reward if node.to_play == to_play else true_reward) + discount_factor * bootstrap_value


def batch_backpropagate(
        latent_state_index_in_search_path: int,
        discount_factor: float,
        value_prefixs: List[float],
        values: List[float],
        policies: List[float],
        min_max_stats_lst: List[MinMaxStats],
        results: SearchResults,
        to_play: list = None,
        is_chance_list: list = None,
        leaf_idx_list: list = None,
) -> None:
    """
    Overview:
        Update the value sum and visit count of nodes along the search paths for each root in the batch.
    Arguments:
        - latent_state_index_in_search_path (:obj:`int`): the index of latent state vector.
        - discount_factor (:obj:`float`): discount_factor factor used i calculating bootstrapped value,
            if env is board_games, we set discount_factor=1.
        - value_prefixs (:obj:`List`): the value prefixs of nodes along the search path.
        - values (:obj:`List`):  the values to propagate along the search path.
        - policies (:obj:`List`): the policy logits of nodes along the search path.
        - min_max_stats_lst (:obj:`List[MinMaxStats]`):  a tool used to min-max normalize the q value.
        - results (:obj:`List`): the search results.
        - to_play (:obj:`List`):  the batch of which player is playing on this node.
    """
    if leaf_idx_list is None:
        leaf_idx_list = list(range(results.num))
    for leaf_order, i in enumerate(leaf_idx_list):
        # ****** expand the leaf node ******
        if to_play is None:
            # set to_play=-1, because two_player mode to_play = {1,2}
            results.nodes[i].expand(-1, latent_state_index_in_search_path, i, value_prefixs[leaf_order], policies[leaf_order], is_chance_list[i])
        else:
            results.nodes[i].expand(to_play[i], latent_state_index_in_search_path, i, value_prefixs[leaf_order], policies[leaf_order], is_chance_list[i])

        # ****** backpropagate ******
        if to_play is None:
            backpropagate(results.search_paths[i], min_max_stats_lst.stats_lst[i], 0, values[leaf_order], discount_factor)
        else:
            backpropagate(
                results.search_paths[i], min_max_stats_lst.stats_lst[i], to_play[i], values[leaf_order], discount_factor
            )

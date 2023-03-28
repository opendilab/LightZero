"""
The Node, Roots class and related core functions for MuZero.
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
         the node base class for MuZero.
     Arguments:
     """

    def __init__(self, prior: float, legal_actions: List = None, action_space_size: int = 9) -> None:
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
        self.hidden_state_index_x = 0
        self.hidden_state_index_y = 0
        self.parent_value_prefix = 0  # only used in update_tree_q method

    def expand(self, to_play: int, hidden_state_index_x: int, hidden_state_index_y: int, reward: float,
               policy_logits: List[float]) -> None:
        """
        Overview:
            Expand the child nodes of the current node.
        Arguments:
            - to_play (:obj:`Class int`): which player to play the game in the current node.
            - hidden_state_index_x (:obj:`Class int`): the x/first index of hidden state vector of the current node, i.e. the search depth.
            - hidden_state_index_y (:obj:`Class int`): the y/second index of hidden state vector of the current node, i.e. the index of batch root node, its maximum is ``batch_size``/``env_num``.
            - value_prefix: (:obj:`Class float`): the value prefix of the current node.
            - policy_logits: (:obj:`Class List`): the policy logit of the child nodes.
        """
        self.to_play = to_play
        if self.legal_actions is None:
            # TODO
            self.legal_actions = np.arange(len(policy_logits))

        self.hidden_state_index_x = hidden_state_index_x
        self.hidden_state_index_y = hidden_state_index_y
        self.reward = reward

        policy_values = torch.softmax(torch.tensor([policy_logits[a] for a in self.legal_actions]), dim=0).tolist()
        policy = {a: policy_values[i] for i, a in enumerate(self.legal_actions)}
        for action, p in policy.items():
            self.children[action] = Node(p)

    def add_exploration_noise(self, exploration_fraction: float, noises: List[float]) -> None:
        """
        Overview:
            add exploration noise to priors
        Arguments:
            - noises (:obj: list): length is len(self.legal_actions)
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

    def compute_mean_q(self, is_root: int, parent_q: float, discount_factor: float) -> float:
        """
        Overview:
            Compute the mean q value of the current node.
        Arguments:
            - is_root (:obj:`int`): whether the current node is a root node.
            - parent_q (:obj:`float`): the q value of the parent node.
            - discount_factor (:obj:`float`): the discount_factor of reward.
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
            Find the current best trajectory starts from the current node.
        Outputs:
            - traj: a vector of node index, which is the current best trajectory from this node.
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
            get children node according to the input action.
        """
        if not isinstance(action, np.int64):
            action = int(action)
        return self.children[action]

    @property
    def expanded(self) -> bool:
        return len(self.children) > 0

    @property
    def value(self) -> float:
        """
        Overview:
            Return the estimated value of the current root node.
        """
        if self.visit_count == 0:
            return 0
        else:
            return self.value_sum / self.visit_count


class Roots:

    def __init__(self, root_num: int, legal_actions_list: List) -> None:
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

    def prepare(self, root_exploration_fraction: float, noises: List[float], rewards: List[float], policies: List[List[float]], to_play: int = -1) -> None:
        """
        Overview:
            Expand the roots and add noises.
        Arguments:
            - root_exploration_fraction: the exploration fraction of roots
            - noises: the vector of noise add to the roots.
            - rewards: the vector of rewards of each root.
            - policies: the vector of policy logits of each root.
            - to_play_batch: the vector of the player side of each root.
        """
        for i in range(self.root_num):
            #  to_play: int, hidden_state_index_x: int, hidden_state_index_y: int,
            # TODO(pu): why hidden_state_index_x=0, hidden_state_index_y=i?
            if to_play is None:
                self.roots[i].expand(0, 0, i, rewards[i], policies[i])
            else:
                self.roots[i].expand(to_play[i], 0, i, rewards[i], policies[i])

            self.roots[i].add_exploration_noise(root_exploration_fraction, noises[i])
            self.roots[i].visit_count += 1

    def prepare_no_noise(self, rewards: List[float], policies: List[List[float]], to_play: int = -1) -> None:
        """
        Overview:
            Expand the roots without noise.
        Arguments:
            - rewards: the vector of rewards of each root.
            - policies: the vector of policy logits of each root.
            - to_play_batch: the vector of the player side of each root.
        """
        for i in range(self.root_num):
            if to_play is None:
                self.roots[i].expand(0, 0, i, rewards[i], policies[i])
            else:
                self.roots[i].expand(to_play[i], 0, i, rewards[i], policies[i])

            self.roots[i].visit_count += 1

    def clear(self) -> None:
        self.roots.clear()

    def get_trajectories(self) -> List[List[Union[int, float]]]:
        """
        Overview:
            Find the current best trajectory starts from each root.
        Outputs:
            - traj: a vector of node index, which is the current best trajectory from each root.
        """
        trajs = []
        for i in range(self.root_num):
            trajs.append(self.roots[i].get_trajectory())
        return trajs

    def get_distributions(self) -> List[List[Union[int, float]]]:
        """
        Overview:
            Get the children distribution of each root.
        Outputs:
            - distribution: a vector of distribution of child nodes in the format of visit count (i.e. [1,3,0,2,5]).
        """
        distributions = []
        for i in range(self.root_num):
            distributions.append(self.roots[i].get_children_distribution())

        return distributions

    def get_values(self) -> float:
        """
        Overview:
            Return the estimated value of each root.
        """
        values = []
        for i in range(self.root_num):
            values.append(self.roots[i].value)
        return values


class SearchResults:

    def __init__(self, num: int) -> None:
        self.num = num
        self.nodes = []
        self.search_paths = []
        self.hidden_state_index_x_lst = []
        self.hidden_state_index_y_lst = []
        self.last_actions = []
        self.search_lens = []


def update_tree_q(root: Node, min_max_stats: MinMaxStats, discount_factor: float, players: int = 1) -> None:
    """
    Overview:
        Update the value sum and visit count of nodes along the search path.
    Arguments:
        - search_path: a vector of nodes on the search path.
        - min_max_stats: a tool used to min-max normalize the q value.
        - to_play: which player to play the game in the current node.
        - value: the value to propagate along the search path.
        - discount_factor: the discount factor of reward.
    """
    node_stack = []
    node_stack.append(root)
    while len(node_stack) > 0:
        node = node_stack[-1]
        node_stack.pop()

        if node != root:
            true_reward = node.reward
            if players == 1:
                q_of_s_a = true_reward + discount_factor * node.value
            elif players == 2:
                q_of_s_a = true_reward + discount_factor * (-node.value)

            min_max_stats.update(q_of_s_a)

        for a in node.legal_actions:
            child = node.get_child(a)
            if child.expanded:
                node_stack.append(child)

def select_child(
        root: Node, min_max_stats: MinMaxStats, pb_c_base: float, pb_c_int: float, discount_factor: float, mean_q: float, players: int
) -> Union[int, float]:
    """
    Overview:
        Select the child node of the roots according to ucb scores.
    Arguments:
        - root: the roots to select the child node.
        - min_max_stats (:obj:`Class MinMaxStats`):  a tool used to min-max normalize the score.
        - pb_c_base (:obj:`Class Float`): constant c1 used in pUCT rule, typically 1.25.
        - pb_c_int (:obj:`Class Float`): constant c2 used in pUCT rule, typically 19652.
        - discount_factor (:obj:`Class Float`): discount_factor factor used i calculating bootstrapped value, if env is board_games, we set discount_factor=1.
        - mean_q (:obj:`Class Float`): the mean q value of the parent node.
        - players (:obj:`Class Int`): the number of players. one/two_player mode board games.
    Returns:
        - action (:obj:`Union[int, float]`): Choose the action with the highest ucb score.
    """
    max_score = -np.inf
    epsilon = 0.000001
    max_index_lst = []
    for a in root.legal_actions:
        child = root.get_child(a)
        temp_score = compute_ucb_score(
            child, min_max_stats, mean_q, root.visit_count, pb_c_base, pb_c_int, discount_factor, players
        )
        if max_score < temp_score:
            max_score = temp_score
            max_index_lst.clear()
            max_index_lst.append(a)
        elif temp_score >= max_score - epsilon:
            # TODO(pu): if the difference is less than  epsilon = 0.000001, we random choice action from  max_index_lst
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
            - child: the child node to compute ucb score.
            - min_max_stats: a tool used to min-max normalize the score.
            - parent_mean_q: the mean q value of the parent node.
            - is_reset: whether the value prefix needs to be reset.
            - total_children_visit_counts: the total visit counts of the child nodes of the parent node.
            - parent_value_prefix: the value prefix of parent node.
            - pb_c_base: constants c2 in muzero.
            - pb_c_init: constants c1 in muzero.
            - disount_factor: the discount factor of reward.
            - players: the number of players.
            - continuous_action_space: whether the action space is continous in current env.
        Outputs:
            - ucb_value: the ucb score of the child.
    """
    pb_c = math.log((total_children_visit_counts + pb_c_base + 1) / pb_c_base) + pb_c_init
    pb_c *= (math.sqrt(total_children_visit_counts) / (child.visit_count + 1))

    prior_score = pb_c * child.prior
    if child.visit_count == 0:
        value_score = parent_mean_q
    else:
        true_reward = child.reward
        if players == 1:
            value_score = true_reward + discount_factor * child.value
        elif players == 2:
            value_score = true_reward + discount_factor * (-child.value)

    value_score = min_max_stats.normalize(value_score)
    if value_score < 0:
        value_score = 0
    if value_score > 1:
        value_score = 1
    ucb_score = prior_score + value_score

    return ucb_score


def batch_traverse(
        roots: Any, pb_c_base: float, pb_c_init: float, discount_factor: float, min_max_stats_lst: List[MinMaxStats], results: SearchResults,
        virtual_to_play: List,
) -> Tuple[List[int], List[int], List[Union[int, float]], List]:
    """
    Overview:
        traverse, also called expansion. process a batch roots parallely.
    Arguments:
        - roots (:obj:`Any`): a batch of root nodes to be expanded.
        - pb_c_base (:obj:`float`): constant c1 used in pUCT rule, typically 1.25.
        - pb_c_init (:obj:`float`): constant c2 used in pUCT rule, typically 19652.
        - discount_factor (:obj:`float`): discount_factor factor used i calculating bootstrapped value, if env is board_games, we set discount_factor=1.
        - virtual_to_play (:obj:`list`): the to_play list used in self_play collecting and training in board games,
            `virtual` is to emphasize that actions are performed on an imaginary hidden state.
        - continuous_action_space: whether the action space is continous in current env.
    Returns:
        - hidden_state_index_x_lst (:obj:`list`): the list of x/first index of hidden state vector of the searched node, i.e. the search depth.
        - hidden_state_index_y_lst (:obj:`list`): the list of y/second index of hidden state vector of the searched node, i.e. the index of batch root node, its maximum is ``batch_size``/``env_num``.
        - last_actions (:obj:`list`): the action performed by the previous node.
        - virtual_to_play (:obj:`list`): the to_play list used in self_play collecting and trainin gin board games,
            `virtual` is to emphasize that actions are performed on an imaginary hidden state.
    """
    parent_q = 0.0
    results.search_lens = [None for i in range(results.num)]
    results.last_actions = [None for i in range(results.num)]

    results.nodes = [None for i in range(results.num)]
    results.hidden_state_index_x_lst = [None for i in range(results.num)]
    results.hidden_state_index_y_lst = [None for i in range(results.num)]
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

        # MCTS stage 1:
        # Each simulation starts from the internal root state s0, and finishes when the simulation reaches a leaf node s_l.
        # the leaf node is not expanded
        while node.expanded:

            mean_q = node.compute_mean_q(is_root, parent_q, discount_factor)
            is_root = 0
            parent_q = mean_q

            # select action according to the pUCT rule
            action = select_child(node, min_max_stats_lst.stats_lst[i], pb_c_base, pb_c_init, discount_factor, mean_q,
                                  players)
            if players == 2:
                # Players play turn by turn
                if virtual_to_play[i] == 1:
                    virtual_to_play[i] = 2
                else:
                    virtual_to_play[i] = 1
            node.best_action = action

            # move to child node according to action
            node = node.get_child(action)
            last_action = action
            results.search_paths[i].append(node)
            search_len += 1

            # note this return the parent node of the current searched node
            parent = results.search_paths[i][len(results.search_paths[i]) - 1 - 1]

            results.hidden_state_index_x_lst[i] = parent.hidden_state_index_x
            results.hidden_state_index_y_lst[i] = parent.hidden_state_index_y
            results.last_actions[i] = last_action
            results.search_lens[i] = search_len
            # the leaf node
            results.nodes[i] = node

    # print(f'env {i} one simulation done!')
    return results.hidden_state_index_x_lst, results.hidden_state_index_y_lst, results.last_actions, virtual_to_play


def backpropagate(search_path: List[Node], min_max_stats: MinMaxStats, to_play: int, value: float, discount_factor: float) -> None:
    """
    Overview:
        Update the value sum and visit count of nodes along the search path.
    Arguments:
        - search_path: a vector of nodes on the search path.
        - min_max_stats: a tool used to min-max normalize the q value.
        - to_play: which player to play the game in the current node.
        - value: the value to propagate along the search path.
        - discount_factor: the discount factor of reward.
    """
    assert to_play is None or to_play in [-1, 1, 2]
    if to_play is None or to_play == -1:
        # for 1 player mode
        bootstrap_value = value
        path_len = len(search_path)
        for i in range(path_len - 1, -1, -1):
            node = search_path[i]
            node.value_sum += bootstrap_value
            node.visit_count += 1

            true_reward = node.reward

            # TODO(pu): the effect of different ways to update min_max_stats
            min_max_stats.update(true_reward + discount_factor * node.value)
            bootstrap_value = true_reward + discount_factor * bootstrap_value

        # TODO(pu): the effect of different ways to update min_max_stats
        # min_max_stats.clear()
        # root = search_path[0]
        # update_tree_q(root, min_max_stats, discount_factor, 1)
    else:
        # for 2 player mode
        bootstrap_value = value
        path_len = len(search_path)
        for i in range(path_len - 1, -1, -1):
            node = search_path[i]
            # to_play related
            node.value_sum += bootstrap_value if node.to_play == to_play else -bootstrap_value

            node.visit_count += 1

            # NOTE: in two player mode,
            # we should calculate the true_reward according to the perspective of current player of node
            # true_reward = node.value_prefix - (- parent_value_prefix)
            true_reward = node.reward

            # min_max_stats.update(true_reward + discount_factor * node.value)
            min_max_stats.update(true_reward + discount_factor * -node.value)

            # to_play related
            # true_reward is in the perspective of current player of node
            # bootstrap_value = (true_reward if node.to_play == to_play else - true_reward) + discount_factor * bootstrap_value
            bootstrap_value = (
                                  -true_reward if node.to_play == to_play else true_reward) + discount_factor * bootstrap_value


def batch_backpropagate(
        hidden_state_index_x: int,
        discount_factor: float,
        value_prefixs: List[float],
        values: List[float],
        policies: List[float],
        min_max_stats_lst: List[MinMaxStats],
        results: SearchResults,
        to_play: list = None
) -> None:
    """
    Overview:
        Backpropagation along the search path to update the attributes.
    Arguments:
        - hidden_state_index_x (:obj:`Class Int`): the index of hidden state vector.
        - discount_factor (:obj:`Class Float`): discount_factor factor used i calculating bootstrapped value, if env is board_games, we set discount_factor=1.
        - value_prefixs (:obj:`Class List`): the value prefixs of nodes along the search path.
        - values (:obj:`Class List`):  the values to propagate along the search path.
        - policies (:obj:`Class List`): the policy logits of nodes along the search path.
        - min_max_stats_lst (:obj:`Class List[MinMaxStats]`):  a tool used to min-max normalize the q value.
        - results (:obj:`Class List`): the search results.
        - to_play (:obj:`Class List`):  the batch of which player is playing on this node.
    """
    for i in range(results.num):
        # expand the leaf node
        #  to_play: int, hidden_state_index_x: int, hidden_state_index_y: int,
        if to_play is None:
            # set to_play=0, because two_player mode to_play = {1,2}
            results.nodes[i].expand(0, hidden_state_index_x, i, value_prefixs[i], policies[i])
        else:
            results.nodes[i].expand(to_play[i], hidden_state_index_x, i, value_prefixs[i], policies[i])

        if to_play is None:
            backpropagate(results.search_paths[i], min_max_stats_lst.stats_lst[i], 0, values[i], discount_factor)
        else:
            backpropagate(results.search_paths[i], min_max_stats_lst.stats_lst[i], to_play[i], values[i],
                          discount_factor)

"""
The Node, Roots class and related core functions for Sampled EfficientZero.
"""
import math
import random
from typing import List, Any, Tuple, Union

import numpy as np
import torch
from torch.distributions import Normal, Independent

from .minimax import MinMaxStats


class Node:
    """
     Overview:
         the node base class for Sampled EfficientZero.
     """

    def __init__(
            self,
            prior: Union[list, float],
            legal_actions: List = None,
            action_space_size: int = 9,
            num_of_sampled_actions: int = 20,
            continuous_action_space: bool = False,
    ) -> None:
        self.prior = prior
        self.mu = None
        self.sigma = None
        self.legal_actions = legal_actions
        self.action_space_size = action_space_size
        self.num_of_sampled_actions = num_of_sampled_actions
        self.continuous_action_space = continuous_action_space

        self.is_reset = 0
        self.visit_count = 0
        self.value_sum = 0
        self.best_action = -1
        self.to_play = -1  # default -1 means play_with_bot_mode
        self.value_prefix = 0.0
        self.children = {}
        self.children_index = []
        self.simulation_index = 0
        self.batch_index = 0

    def expand(
            self, to_play: int, simulation_index: int, batch_index: int, value_prefix: float, policy_logits: List[float]
    ) -> None:
        """
        Overview:
            Expand the child nodes of the current node.
        Arguments:
            - to_play (:obj:`Class int`): which player to play the game in the current node.
            - simulation_index (:obj:`Class int`): the x/first index of hidden state vector of the current node, i.e. the search depth.
            - batch_index (:obj:`Class int`): the y/second index of hidden state vector of the current node, i.e. the index of batch root node, its maximum is ``batch_size``/``env_num``.
            - value_prefix: (:obj:`Class float`): the value prefix of the current node.
            - policy_logits: (:obj:`Class List`): the policy logit of the child nodes.
        """
        """
        to varify ctree_efficientzero:
            import numpy as np
            import torch
            from torch.distributions import Normal, Independent
            mu= torch.tensor([0.1,0.1])
            sigma= torch.tensor([0.1,0.1])
            dist = Independent(Normal(mu, sigma), 1)
            sampled_actions=torch.tensor([0.282769,0.376611])
            dist.log_prob(sampled_actions)
        """
        self.to_play = to_play
        self.simulation_index = simulation_index
        self.batch_index = batch_index
        self.value_prefix = value_prefix

        # ==============================================================
        # TODO(pu): legal actions
        # ==============================================================
        # policy_values = torch.softmax(torch.tensor([policy_logits[a] for a in self.legal_actions]), dim=0).tolist()
        # policy = {a: policy_values[i] for i, a in enumerate(self.legal_actions)}
        # for action, p in policy.items():
        #     self.children[action] = Node(p)

        # ==============================================================
        # sampled related core code
        # ==============================================================
        if self.continuous_action_space:
            (mu, sigma) = torch.tensor(policy_logits[:self.action_space_size]
                                       ), torch.tensor(policy_logits[-self.action_space_size:])
            self.mu = mu
            self.sigma = sigma
            dist = Independent(Normal(mu, sigma), 1)
            # print(dist.batch_shape, dist.event_shape)
            sampled_actions_before_tanh = dist.sample(torch.tensor([self.num_of_sampled_actions]))

            sampled_actions = torch.tanh(sampled_actions_before_tanh)
            y = 1 - sampled_actions.pow(2) + 1e-6
            # keep dimension for loss computation (usually for action space is 1 env. e.g. pendulum)
            log_prob = dist.log_prob(sampled_actions_before_tanh).unsqueeze(-1)
            log_prob = log_prob - torch.log(y).sum(-1, keepdim=True)
            self.legal_actions = []

            for action_index in range(self.num_of_sampled_actions):
                self.children[Action(sampled_actions[action_index].detach().cpu().numpy())] = Node(
                    log_prob[action_index],
                    action_space_size=self.action_space_size,
                    num_of_sampled_actions=self.num_of_sampled_actions,
                    continuous_action_space=self.continuous_action_space
                )
                self.legal_actions.append(Action(sampled_actions[action_index].detach().cpu().numpy()))
        else:
            if self.legal_actions is not None:
                # first use the self.legal_actions to exclude the illegal actions
                policy_tmp = [0. for _ in range(self.action_space_size)]
                for index, legal_action in enumerate(self.legal_actions):
                    policy_tmp[legal_action] = policy_logits[index]
                policy_logits = policy_tmp
            # then empty the self.legal_actions
            self.legal_actions = []
            prob = torch.softmax(torch.tensor(policy_logits), dim=-1)
            sampled_actions = torch.multinomial(prob, self.num_of_sampled_actions, replacement=False)

            for action_index in range(self.num_of_sampled_actions):
                self.children[Action(sampled_actions[action_index].detach().cpu().numpy())] = Node(
                    # prob[action_index], # NOTE: this is a bug
                    prob[sampled_actions[action_index]],  #
                    action_space_size=self.action_space_size,
                    num_of_sampled_actions=self.num_of_sampled_actions,
                    continuous_action_space=self.continuous_action_space
                )
                self.legal_actions.append(Action(sampled_actions[action_index].detach().cpu().numpy()))

    def add_exploration_noise_to_sample_distribution(
            self, exploration_fraction: float, noises: List[float], policy_logits: List[float]
    ) -> None:
        """
        Overview:
            add exploration noise to priors.
        Arguments:
            - noises (:obj: list): length is len(self.legal_actions)
        """
        # ==============================================================
        # sampled related core code
        # ==============================================================
        # TODO(pu): add noise to sample distribution \beta logits
        for i in range(len(policy_logits)):
            if self.continuous_action_space:
                # probs is log_prob
                pass
            else:
                # probs is prob
                policy_logits[i] = policy_logits[i] * (1 - exploration_fraction) + noises[i] * exploration_fraction

    def add_exploration_noise(self, exploration_fraction: float, noises: List[float]) -> None:
        """
        Overview:
            Add a noise to the prior of the child nodes.
        Arguments:
            - exploration_fraction: the fraction to add noise.
            - noises (:obj: list): the vector of noises added to each child node. length is len(self.legal_actions)
        """
        # ==============================================================
        # sampled related core code
        # ==============================================================
        actions = list(self.children.keys())
        for a, n in zip(actions, noises):
            if self.continuous_action_space:
                # prior is log_prob
                self.children[a].prior = np.log(
                    np.exp(self.children[a].prior) * (1 - exploration_fraction) + n * exploration_fraction
                )
            else:
                # prior is prob
                self.children[a].prior = self.children[a].prior * (1 - exploration_fraction) + n * exploration_fraction

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
        parent_value_prefix = self.value_prefix
        for a in self.legal_actions:
            child = self.get_child(a)
            if child.visit_count > 0:
                true_reward = child.value_prefix - parent_value_prefix
                if self.is_reset == 1:
                    true_reward = child.value_prefix
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

    def print_out(self) -> None:
        pass

    def get_trajectory(self) -> List[Union[int, float]]:
        """
        Overview:
            Find the current best trajectory starts from the current node.
        Outputs:
            - traj: a vector of node index, which is the current best trajectory from this node.
        """
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
        # distribution = {a: 0 for a in self.legal_actions}
        distribution = {}
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
        if isinstance(action, Action):
            return self.children[action]
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

    def __init__(
            self,
            root_num: int,
            legal_actions_list: List,
            action_space_size: int = 9,
            num_of_sampled_actions: int = 20,
            continuous_action_space: bool = False,
    ) -> None:
        self.num = root_num
        self.root_num = root_num
        self.legal_actions_list = legal_actions_list  # list of list
        self.num_of_sampled_actions = num_of_sampled_actions
        self.continuous_action_space = continuous_action_space

        self.roots = []

        # ==============================================================
        # sampled related core code
        # ==============================================================
        for i in range(self.root_num):
            if isinstance(legal_actions_list, list):
                # TODO(pu): sampled in board_games
                self.roots.append(
                    Node(
                        0,
                        legal_actions_list[i],
                        action_space_size=action_space_size,
                        num_of_sampled_actions=self.num_of_sampled_actions,
                        continuous_action_space=self.continuous_action_space
                    )
                )
            elif isinstance(legal_actions_list, int):
                # if legal_actions_list is int
                self.roots.append(
                    Node(
                        0,
                        None,
                        action_space_size=action_space_size,
                        num_of_sampled_actions=self.num_of_sampled_actions,
                        continuous_action_space=self.continuous_action_space
                    )
                )
            elif legal_actions_list is None:
                # continuous action space
                self.roots.append(
                    Node(
                        0,
                        None,
                        action_space_size=action_space_size,
                        num_of_sampled_actions=self.num_of_sampled_actions,
                        continuous_action_space=self.continuous_action_space
                    )
                )

    def prepare(
            self,
            root_noise_weight: float,
            noises: List[float],
            value_prefixs: List[float],
            policies: List[List[float]],
            to_play: int = -1
    ) -> None:
        """
        Overview:
            Expand the roots and add noises.
        Arguments:
            - root_noise_weight: the exploration fraction of roots
            - noises: the vector of noise add to the roots.
            - value_prefixs: the vector of value prefixs of each root.
            - policies: the vector of policy logits of each root.
            - to_play_batch: the vector of the player side of each root.
        """
        for i in range(self.root_num):

            if to_play is None:
                self.roots[i].expand(-1, 0, i, value_prefixs[i], policies[i])
            else:
                self.roots[i].expand(to_play[i], 0, i, value_prefixs[i], policies[i])
            self.roots[i].add_exploration_noise(root_noise_weight, noises[i])

            self.roots[i].visit_count += 1

    def prepare_no_noise(self, value_prefixs: List[float], policies: List[List[float]], to_play: int = -1) -> None:
        """
        Overview:
            Expand the roots without noise.
        Arguments:
            - value_prefixs: the vector of value prefixs of each root.
            - policies: the vector of policy logits of each root.
            - to_play_batch: the vector of the player side of each root.
        """
        for i in range(self.root_num):
            if to_play is None:
                self.roots[i].expand(-1, 0, i, value_prefixs[i], policies[i])
            else:
                self.roots[i].expand(to_play[i], 0, i, value_prefixs[i], policies[i])

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

    # ==============================================================
    # sampled related core code
    # ==============================================================
    def get_sampled_actions(self) -> List[List[Union[int, float]]]:
        """
        Overview:
            Get the sampled_actions of each root.
        Outputs:
            - python_sampled_actions: a vector of sampled_actions for each root, e.g. the size of original action space is 6, the K=3,
            python_sampled_actions = [[1,3,0], [2,4,0], [5,4,1]].
        """
        sampled_actions = []
        for i in range(self.root_num):
            sampled_actions.append(self.roots[i].legal_actions)

        return sampled_actions

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

    def __init__(self, num: int):
        self.num = num
        self.nodes = []
        self.search_paths = []
        self.latent_state_index_in_search_path = []
        self.latent_state_index_in_batch = []
        self.last_actions = []
        self.search_lens = []


def select_child(
        root: Node,
        min_max_stats: MinMaxStats,
        pb_c_base: float,
        pb_c_int: float,
        discount_factor: float,
        mean_q: float,
        players: int,
        continuous_action_space: bool = False,
) -> Union[int, float]:
    """
    Overview:
        Select the child node of the roots according to ucb scores.
    Arguments:
        - root: the roots to select the child node.
        - min_max_stats (:obj:`Class MinMaxStats`):  a tool used to min-max normalize the score.
        - pb_c_base (:obj:`Class Float`): constant c1 used in pUCT rule, typically 1.25.
        - pb_c_int (:obj:`Class Float`): constant c2 used in pUCT rule, typically 19652.
        - discount_factor (:obj:`Class Float`): The discount factor used in calculating bootstrapped value, if env is board_games, we set discount_factor=1.
        - mean_q (:obj:`Class Float`): the mean q value of the parent node.
        - players (:obj:`Class Float`): the number of players. one/in self-play-mode board games.
        - continuous_action_space: whether the action space is continous in current env.
    Returns:
        - action (:obj:`Union[int, float]`): Choose the action with the highest ucb score.
    """
    # ==============================================================
    # sampled related core code
    # ==============================================================
    # TODO(pu): Progressive widening (See https://hal.archives-ouvertes.fr/hal-00542673v2/document)
    max_score = -np.inf
    epsilon = 0.000001
    max_index_lst = []
    for action, child in root.children.items():
        # ==============================================================
        # sampled related core code
        # ==============================================================
        # use root as input argument
        temp_score = compute_ucb_score(
            root, child, min_max_stats, mean_q, root.is_reset, root.visit_count, root.value_prefix, pb_c_base, pb_c_int,
            discount_factor, players, continuous_action_space
        )
        if max_score < temp_score:
            max_score = temp_score
            max_index_lst.clear()
            max_index_lst.append(action)
        elif temp_score >= max_score - epsilon:
            # TODO(pu): if the difference is less than epsilon = 0.000001, we random choice action from max_index_lst
            max_index_lst.append(action)

    if len(max_index_lst) > 0:
        action = random.choice(max_index_lst)

    return action


def compute_ucb_score(
        parent: Node,
        child: Node,
        min_max_stats: MinMaxStats,
        parent_mean_q: float,
        is_reset: int,
        total_children_visit_counts: float,
        parent_value_prefix: float,
        pb_c_base: float,
        pb_c_init: float,
        discount_factor: float,
        players: int = 1,
        continuous_action_space: bool = False,
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
    assert total_children_visit_counts == parent.visit_count
    pb_c = math.log((total_children_visit_counts + pb_c_base + 1) / pb_c_base) + pb_c_init
    pb_c *= (math.sqrt(total_children_visit_counts) / (child.visit_count + 1))

    # ==============================================================
    # sampled related core code
    # ==============================================================
    # TODO(pu)
    node_prior = "density"
    # node_prior = "uniform"
    if node_prior == "uniform":
        # Uniform prior for continuous action space
        prior_score = pb_c * (1 / len(parent.children))
    elif node_prior == "density":
        # TODO(pu): empirical distribution
        if continuous_action_space:
            # prior is log_prob
            prior_score = pb_c * (
                torch.exp(child.prior) / (sum([torch.exp(node.prior) for node in parent.children.values()]) + 1e-6)
            )
        else:
            # prior is prob
            prior_score = pb_c * (child.prior / (sum([node.prior for node in parent.children.values()]) + 1e-6))
            # print('prior_score: ', prior_score)
    else:
        raise ValueError("{} is unknown prior option, choose uniform or density")

    if child.visit_count == 0:
        value_score = parent_mean_q
    else:
        true_reward = child.value_prefix - parent_value_prefix
        if is_reset == 1:
            true_reward = child.value_prefix
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
        roots: Any,
        pb_c_base: float,
        pb_c_init: float,
        discount_factor: float,
        min_max_stats_lst,
        results: SearchResults,
        virtual_to_play: List,
        continuous_action_space: bool = False,
) -> Tuple[List[int], List[int], List[Union[int, float]], List]:
    """
    Overview:
        traverse, also called expansion. process a batch roots parallely.
    Arguments:
        - roots (:obj:`Any`): a batch of root nodes to be expanded.
        - pb_c_base (:obj:`float`): constant c1 used in pUCT rule, typically 1.25.
        - pb_c_init (:obj:`float`): constant c2 used in pUCT rule, typically 19652.
        - discount_factor (:obj:`float`): The discount factor used in calculating bootstrapped value, if env is board_games, we set discount_factor=1.
        - virtual_to_play (:obj:`list`): the to_play list used in self_play collecting and training in board games,
            `virtual` is to emphasize that actions are performed on an imaginary hidden state.
        - continuous_action_space: whether the action space is continous in current env.
    Returns:
        - latent_state_index_in_search_path (:obj:`list`): the list of x/first index of hidden state vector of the searched node, i.e. the search depth.
        - latent_state_index_in_batch (:obj:`list`): the list of y/second index of hidden state vector of the searched node, i.e. the index of batch root node, its maximum is ``batch_size``/``env_num``.
        - last_actions (:obj:`list`): the action performed by the previous node.
        - virtual_to_play (:obj:`list`): the to_play list used in self_play collecting and trainin gin board games,
            `virtual` is to emphasize that actions are performed on an imaginary hidden state.
    """
    parent_q = 0.0
    results.search_lens = [None for _ in range(results.num)]
    results.last_actions = [None for _ in range(results.num)]

    results.nodes = [None for _ in range(results.num)]
    results.latent_state_index_in_search_path = [None for _ in range(results.num)]
    results.latent_state_index_in_batch = [None for _ in range(results.num)]
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
            The leaf node is the node that is currently not expanded.
        """
        while node.expanded:

            mean_q = node.compute_mean_q(is_root, parent_q, discount_factor)
            is_root = 0
            parent_q = mean_q

            # select action according to the pUCT rule
            action = select_child(
                node, min_max_stats_lst.stats_lst[i], pb_c_base, pb_c_init, discount_factor, mean_q, players,
                continuous_action_space
            )

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

            results.latent_state_index_in_search_path[i] = parent.simulation_index
            results.latent_state_index_in_batch[i] = parent.batch_index
            # results.last_actions[i] = last_action
            results.last_actions[i] = last_action.value
            results.search_lens[i] = search_len
            # the leaf node
            results.nodes[i] = node

    # print(f'env {i} one simulation done!')
    return results.latent_state_index_in_search_path, results.latent_state_index_in_batch, results.last_actions, virtual_to_play


def backpropagate(
        search_path: List[Node], min_max_stats: MinMaxStats, to_play: int, value: float, discount_factor: float
) -> None:
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
    assert to_play is None or to_play in [-1, 1, 2], to_play
    if to_play is None or to_play == -1:
        # for play-with-bot-mode
        bootstrap_value = value
        path_len = len(search_path)
        for i in range(path_len - 1, -1, -1):
            node = search_path[i]
            node.value_sum += bootstrap_value
            node.visit_count += 1

            parent_value_prefix = 0.0
            is_reset = 0
            if i >= 1:
                parent = search_path[i - 1]
                parent_value_prefix = parent.value_prefix
                is_reset = parent.is_reset

            true_reward = node.value_prefix - parent_value_prefix
            min_max_stats.update(true_reward + discount_factor * node.value)

            if is_reset == 1:
                true_reward = node.value_prefix

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

            parent_value_prefix = 0.0
            is_reset = 0
            if i >= 1:
                parent = search_path[i - 1]
                parent_value_prefix = parent.value_prefix
                is_reset = parent.is_reset

            # NOTE: in self-play-mode, value_prefix is not calculated according to the perspective of current player of node.
            # TODO: true_reward = node.value_prefix - (- parent_value_prefix)
            true_reward = node.value_prefix - parent_value_prefix
            if is_reset == 1:
                true_reward = node.value_prefix

            min_max_stats.update(true_reward + discount_factor * -node.value)

            # true_reward is in the perspective of current player of node
            bootstrap_value = (
                -true_reward if node.to_play == to_play else true_reward
            ) + discount_factor * bootstrap_value


def batch_backpropagate(
        simulation_index: int,
        discount_factor: float,
        value_prefixs: List,
        values: List[float],
        policies: List[float],
        min_max_stats_lst: List[MinMaxStats],
        results: SearchResults,
        is_reset_list: List,
        to_play: list = None
) -> None:
    """
    Overview:
        Backpropagation along the search path to update the attributes.
    Arguments:
        - simulation_index (:obj:`Class Int`): The index of latent state of the leaf node in the search path.
        - discount_factor (:obj:`Class Float`): The discount factor used in calculating bootstrapped value, if env is board_games, we set discount_factor=1.
        - value_prefixs (:obj:`Class List`): the value prefixs of nodes along the search path.
        - values (:obj:`Class List`):  the values to propagate along the search path.
        - policies (:obj:`Class List`): the policy logits of nodes along the search path.
        - min_max_stats_lst (:obj:`Class List[MinMaxStats]`):  a tool used to min-max normalize the q value.
        - results (:obj:`Class List`): the search results.
        - is_reset_list (:obj:`Class List`): the vector of is_reset nodes along the search path, where is_reset represents for whether the parent value prefix needs to be reset.
        - to_play (:obj:`Class List`):  the batch of which player is playing on this node.
    """
    for i in range(results.num):
        # ****** expand the leaf node ******
        if to_play is None:
            # we set to_play=-1, because in self-play-mode of board_games to_play = {1, 2}.
            results.nodes[i].expand(-1, simulation_index, i, value_prefixs[i], policies[i])
        else:
            results.nodes[i].expand(to_play[i], simulation_index, i, value_prefixs[i], policies[i])

        # reset
        results.nodes[i].is_reset = is_reset_list[i]

        # ****** backpropagate ******
        if to_play is None:
            backpropagate(results.search_paths[i], min_max_stats_lst.stats_lst[i], 0, values[i], discount_factor)
        else:
            backpropagate(
                results.search_paths[i], min_max_stats_lst.stats_lst[i], to_play[i], values[i], discount_factor
            )


class Action:
    """Class that represent an action of a game."""

    def __init__(self, value: float) -> None:
        self.value = value

    def __hash__(self) -> hash:
        return hash(self.value.tostring())

    def __eq__(self, other: "Action") -> bool:
        return (self.value == other.value).all()

    def __gt__(self, other: "Action") -> bool:
        return self.value[0] > other.value[0]

    def __repr__(self) -> str:
        return str(self.value)

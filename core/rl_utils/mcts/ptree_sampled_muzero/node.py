"""
The Node and Roots class for MCTS in board games in which we must consider legal_actions and to_play.
"""
import math
import random
from typing import List, Any, Optional

import numpy as np
import torch
from torch.distributions import Normal, Independent


class Node:
    """
     Overview:
         the node base class for mcts.
     Arguments:
     """

    def __init__(self, prior: float, legal_actions: Any = None, action_space_size=9, num_of_sampled_actions=20,
                 continuous_action_space=False):
        self.prior = prior
        self.legal_actions = legal_actions
        self.action_space_size = action_space_size
        self.num_of_sampled_actions = num_of_sampled_actions
        self.continuous_action_space = continuous_action_space

        self.visit_count = 0
        self.value_sum = 0
        self.best_action = -1
        self.to_play = 0  # default 0 means one_player_mode
        self.reward = 0
        self.value_prefix = 0.0
        self.children = {}
        self.children_index = []
        # self.hidden_state_index_x = -1
        # self.hidden_state_index_y = -1
        self.hidden_state_index_x = 0
        self.hidden_state_index_y = 0
        self.parent_value_prefix = 0  # only used in update_tree_q method

    def expand(
            self, to_play: int, hidden_state_index_x: int, hidden_state_index_y: int, reward: float,
            policy_logits: List[float]
    ):
        self.to_play = to_play
        # if self.legal_actions is None:
        #     self.legal_actions = np.arange(len(policy_logits))

        self.legal_actions = []

        self.hidden_state_index_x = hidden_state_index_x
        self.hidden_state_index_y = hidden_state_index_y
        self.reward = reward

        ######################
        # sampled related code
        ######################
        if self.continuous_action_space:
            # policy_logits = {'mu': torch.randn([1, 2]), 'sigma': torch.zeros([1, 2]) + 1e-7}
            # (mu, sigma) = policy_logits['mu'], policy_logits['sigma']
            # (mu, sigma) = policy_logits[:,: self.action_space_size ], policy_logits[:,- self.action_space_size:]
            (mu, sigma) = torch.tensor(policy_logits[: self.action_space_size]), torch.tensor(
                policy_logits[- self.action_space_size:])
            self.mu = mu
            self.sigma = sigma
            dist = Independent(Normal(mu, sigma), 1)
            # print(dist.batch_shape, dist.event_shape)
            sampled_actions_before_tanh = dist.sample(torch.tensor([self.num_of_sampled_actions]))

            # way 1:
            # log_prob = dist.log_prob(sampled_actions_before_tanh)

            # way 2:
            sampled_actions = torch.tanh(sampled_actions_before_tanh)
            y = 1 - sampled_actions.pow(2) + 1e-6
            # keep dimension for loss computation (usually for action space is 1 env. e.g. pendulum)
            log_prob = dist.log_prob(sampled_actions_before_tanh).unsqueeze(-1)
            log_prob = log_prob - torch.log(y).sum(-1, keepdim=True)
            # if self.legal_actions is None:
            self.legal_actions = []

            # TODO: factored policy representation
            # empirical_distribution = [1/self.num_of_sampled_actions]
            for action_index in range(self.num_of_sampled_actions):
                self.children[Action(sampled_actions[action_index].detach().cpu().numpy())] = Node(
                    log_prob[action_index],
                    action_space_size=self.action_space_size,
                    num_of_sampled_actions=self.num_of_sampled_actions,
                    continuous_action_space=self.continuous_action_space)
                self.legal_actions.append(Action(sampled_actions[action_index].detach().cpu().numpy()))
        else:
            if self.legal_actions is not None:
                # fisrt use theself.legal_actions to exclude the illegal actions
                policy_tmp = [0. for _ in range(self.action_space_size)]
                for index, legal_action in enumerate(self.legal_actions):
                    policy_tmp[legal_action] = policy_logits[index]
                policy_logits = policy_tmp
            # then empty the self.legal_actions
            self.legal_actions = []

            # prob = torch.softmax(torch.tensor(policy_logits), dim=-1)
            # dist = Categorical(prob)
            # sampled_actions = dist.sample(torch.tensor([self.num_of_sampled_actions]))
            # log_prob = dist.log_prob(sampled_actions)

            prob = torch.softmax(torch.tensor(policy_logits), dim=-1)
            sampled_actions = torch.multinomial(prob, self.num_of_sampled_actions, replacement=False)

            # TODO: factored policy representation
            # empirical_distribution = [1/self.num_of_sampled_actions]
            for action_index in range(self.num_of_sampled_actions):
                self.children[Action(sampled_actions[action_index].detach().cpu().numpy())] = Node(
                    prob[action_index],
                    action_space_size=self.action_space_size,
                    num_of_sampled_actions=self.num_of_sampled_actions,
                    continuous_action_space=self.continuous_action_space)
                self.legal_actions.append(Action(sampled_actions[action_index].detach().cpu().numpy()))

    def add_exploration_noise(self, exploration_fraction: float, noises: List[float]):
        """
        Overview:
            add exploration noise to priors
        Arguments:
            - noises (:obj: list): length is len(self.legal_actions)
        """
        ######################
        # sampled related code
        ######################
        actions = list(self.children.keys())
        for a, n in zip(actions, noises):
            if self.continuous_action_space:
                # prior is log_prob
                self.children[a].prior = np.log(
                    np.exp(self.children[a].prior) * (1 - exploration_fraction) + n * exploration_fraction)
            else:
                # prior is prob
                self.children[a].prior = self.children[a].prior * (1 - exploration_fraction) + n * exploration_fraction

        # for i, a in enumerate(self.legal_actions):
        #     """
        #     i in index, a is action, e.g. self.legal_actions = [0,1,2,4,6,8], i=[0,1,2,3,4,5], a=[0,1,2,4,6,8]
        #     """
        #     try:
        #         noise = noises[i]
        #     except Exception as error:
        #         print(error)
        #     child = self.get_child(a)
        #     prior = child.prior
        #     child.prior = prior * (1 - exploration_fraction) + noise * exploration_fraction

    def get_mean_q(self, is_root: int, parent_q: float, discount: float):
        """
        Overview:
            get mean q
        Arguments:
            - is_root (:obj:`int`):
        """
        total_unsigned_q = 0.0
        total_visits = 0
        # parent_value_prefix = self.value_prefix
        for a in self.legal_actions:
            child = self.get_child(a)
            if child.visit_count > 0:
                true_reward = child.reward
                # TODO(pu): only one step bootstrap?
                q_of_s_a = true_reward + discount * child.value
                total_unsigned_q += q_of_s_a
                total_visits += 1
        if is_root and total_visits > 0:
            mean_q = total_unsigned_q / total_visits
        else:
            # if is not root node,
            # TODO(pu): why parent_q?
            mean_q = (parent_q + total_unsigned_q) / (total_visits + 1)
        return mean_q

    def print_out(self):
        pass

    def get_trajectory(self):
        """
        Overview:
            get best trajectory
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

    def get_children_distribution(self):
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

    def get_child(self, action):
        """
        Overview:
            get children node according to action.
        """
        if isinstance(action, Action):
            return self.children[action]
        if not isinstance(action, np.int64):
            action = int(action)
        return self.children[action]

    @property
    def expanded(self):
        return len(self.children) > 0

    @property
    def value(self):
        """
            estimated Q value
        """
        if self.visit_count == 0:
            return 0
        else:
            return self.value_sum / self.visit_count


class Roots:

    def __init__(self, root_num: int, legal_actions_list: Any, action_space_size: Optional = None,
                 num_of_sampled_actions=20, continuous_action_space=False):
        self.num = root_num
        self.root_num = root_num
        self.legal_actions_list = legal_actions_list  # list of list
        self.num_of_sampled_actions = num_of_sampled_actions
        self.continuous_action_space = continuous_action_space

        self.roots = []
        ##################
        # sampled related code
        ##################
        for i in range(self.root_num):
            if isinstance(legal_actions_list, list):
                self.roots.append(Node(0, legal_actions_list[i], action_space_size=action_space_size,
                                       num_of_sampled_actions=self.num_of_sampled_actions,
                                       continuous_action_space=self.continuous_action_space))
            elif isinstance(legal_actions_list, int):
                # if legal_actions_list is int
                self.roots.append(
                    Node(0, None, action_space_size=action_space_size,
                         num_of_sampled_actions=self.num_of_sampled_actions,
                         continuous_action_space=self.continuous_action_space))
                # self.roots.append(
                #     Node(0, np.arange(legal_actions_list), action_space_size=action_space_size, num_of_sampled_actions=self.num_of_sampled_actions,
                #                        continuous_action_space=self.continuous_action_space))
            elif legal_actions_list is None:
                # continuous action space
                self.roots.append(Node(0, None, action_space_size=action_space_size,
                                       num_of_sampled_actions=self.num_of_sampled_actions,
                                       continuous_action_space=self.continuous_action_space))

    def prepare(self, root_exploration_fraction, noises, rewards, policies, to_play=None):
        for i in range(self.root_num):
            #  to_play: int, hidden_state_index_x: int, hidden_state_index_y: int,
            # TODO(pu): why hidden_state_index_x=0, hidden_state_index_y=i?
            if to_play is None:
                self.roots[i].expand(0, 0, i, rewards[i], policies[i])
            elif to_play is [None]:
                print('debug')
            else:
                self.roots[i].expand(to_play[i], 0, i, rewards[i], policies[i])

            self.roots[i].add_exploration_noise(root_exploration_fraction, noises[i])
            self.roots[i].visit_count += 1

    def prepare_no_noise(self, rewards, policies, to_play=None):
        for i in range(self.root_num):
            if to_play is None:
                self.roots[i].expand(0, 0, i, rewards[i], policies[i])
            else:
                self.roots[i].expand(to_play[i], 0, i, rewards[i], policies[i])

            self.roots[i].visit_count += 1

    def clear(self):
        self.roots.clear()

    def get_trajectories(self):
        trajs = []
        for i in range(self.root_num):
            trajs.append(self.roots[i].get_trajectory())
        return trajs

    def get_distributions(self):
        distributions = []
        for i in range(self.root_num):
            distributions.append(self.roots[i].get_children_distribution())

        return distributions

    ##################
    # sampled related code
    ##################
    def get_sampled_actions(self):
        sampled_actions = []
        for i in range(self.root_num):
            sampled_actions.append(self.roots[i].legal_actions)

        return sampled_actions

    def get_values(self):
        values = []
        for i in range(self.root_num):
            values.append(self.roots[i].value)
        return values


class SearchResults:

    def __init__(self, num):
        self.num = num
        self.nodes = []
        self.search_paths = []
        self.hidden_state_index_x_lst = []
        self.hidden_state_index_y_lst = []
        self.last_actions = []
        self.search_lens = []


def update_tree_q(root: Node, min_max_stats, discount: float, players=1, to_play=0):
    # root.parent_value_prefix = 0
    node_stack = []
    node_stack.append(root)
    while len(node_stack) > 0:
        node = node_stack[-1]
        node_stack.pop()

        if node != root:
            true_reward = node.reward
            if players == 1:
                q_of_s_a = true_reward + discount * node.value
            elif players == 2:
                q_of_s_a = true_reward + discount * (-node.value)

            min_max_stats.update(q_of_s_a)

        for a in node.legal_actions:
            child = node.get_child(a)
            if child.expanded:
                # child.parent_value_prefix = node.value_prefix
                node_stack.append(child)


def back_propagate(search_path, min_max_stats, to_play, value: float, discount: float):
    if to_play is None or to_play == 0:
        # for 1 player mode
        bootstrap_value = value
        path_len = len(search_path)
        for i in range(path_len - 1, -1, -1):
            node = search_path[i]
            node.value_sum += bootstrap_value
            node.visit_count += 1

            true_reward = node.reward

            # TODO(pu): the effect of different ways to update min_max_stats
            min_max_stats.update(true_reward + discount * node.value)

            bootstrap_value = true_reward + discount * bootstrap_value

        # TODO(pu): the effect of different ways to update min_max_stats
        # min_max_stats.clear()
        # root = search_path[0]
        # update_tree_q(root, min_max_stats, discount, 1)
    else:
        # for 2 player mode
        bootstrap_value = value
        path_len = len(search_path)
        for i in range(path_len - 1, -1, -1):
            node = search_path[i]
            # to_play related
            node.value_sum += bootstrap_value if node.to_play == to_play else - bootstrap_value

            node.visit_count += 1

            # NOTE: in two player mode,
            # we should calculate the true_reward according to the perspective of current player of node
            # true_reward = node.value_prefix - (- parent_value_prefix)
            true_reward = node.reward

            # min_max_stats.update(true_reward + discount * node.value)
            # TODO(pu): why in muzero-general is - node.value
            min_max_stats.update(true_reward + discount * - node.value)

            # to_play related
            # true_reward is in the perspective of current player of node
            # bootstrap_value = (true_reward if node.to_play == to_play else - true_reward) + discount * bootstrap_value
            # TODO(pu): why in muzero-general is - true_reward
            bootstrap_value = (- true_reward if node.to_play == to_play else true_reward) + discount * bootstrap_value

        # TODO(pu): the effect of different ways to update min_max_stats
        # min_max_stats.clear()
        # root = search_path[0]
        # update_tree_q(root, min_max_stats, discount, 2)


def batch_back_propagate(
        hidden_state_index_x: int,
        discount: float,
        value_prefixs: List,
        values: List[float],
        policies: List[float],
        min_max_stats_lst,
        results,
        to_play: list = None
) -> None:
    for i in range(results.num):

        # expand the leaf node
        #  to_play: int, hidden_state_index_x: int, hidden_state_index_y: int,
        if to_play is None:
            # set to_play=0, because two_player mode to_play = {1,2}
            results.nodes[i].expand(0, hidden_state_index_x, i, value_prefixs[i], policies[i])
        else:
            results.nodes[i].expand(to_play[i], hidden_state_index_x, i, value_prefixs[i], policies[i])

        if to_play is None:
            back_propagate(results.search_paths[i], min_max_stats_lst.stats_lst[i], 0, values[i], discount)
        else:
            back_propagate(results.search_paths[i], min_max_stats_lst.stats_lst[i], to_play[i], values[i], discount)


def select_child(
        root: Node, min_max_stats, pb_c_base: int, pb_c_int: float, discount: float, mean_q: float, players: int,
        continuous_action_space=False,
) -> int:
    ##################
    # sampled related code
    ##################
    # Progressive widening (See https://hal.archives-ouvertes.fr/hal-00542673v2/document)
    # pw_alpha = 0.49
    # TODO(pu)
    # pw_alpha = -1
    # if len(root.children) < (root.visit_count + 1) ** pw_alpha:
    #     distribution = torch.distributions.normal.Normal(root.mu, root.sigma)
    #     sampled_action = distribution.sample().squeeze(0).detach().cpu().numpy()
    #
    #     while Action(sampled_action) in root.children.keys():
    #         # if sampled_action is sampled before, we sample a new action now
    #         sampled_action = distribution.sample().squeeze(0).detach().cpu()
    #     action = Action(sampled_action)
    #
    #     log_prob = distribution.log_prob(torch.tensor(sampled_action))
    #     # TODO: factored policy representation
    #     # empirical_distribution = [1/self.num_of_sampled_actions]
    #
    #     # pi, beta
    #     root.children[action] = Node(prior=log_prob[0], legal_actions=None,
    #                                  action_space_size=sampled_action.shape[0])  # TODO(pu): action_space_size
    #     # TODO
    #     # root.legal_actions.append(action)
    #     # return action, root.children[action]
    #     return action
    #
    # else:
    max_score = -np.inf
    epsilon = 0.000001
    max_index_lst = []
    for action, child in root.children.items():
        ##################
        # sampled related code
        ##################
        # use root as input argument
        temp_score = compute_ucb_score(
            root, child, min_max_stats, mean_q, root.visit_count, pb_c_base,
            pb_c_int,
            discount, players, continuous_action_space
        )
        if max_score < temp_score:
            max_score = temp_score
            max_index_lst.clear()
            max_index_lst.append(action)
        elif temp_score >= max_score - epsilon:
            # TODO(pu): if the difference is less than epsilon = 0.000001, we random choice action from  max_index_lst
            max_index_lst.append(action)

    if len(max_index_lst) > 0:
        action = random.choice(max_index_lst)

    return action


def compute_ucb_score(
        parent: Node,
        child: Node,
        min_max_stats,
        parent_mean_q,
        total_children_visit_counts: float,
        # parent_value_prefix: float,
        pb_c_base: float,
        pb_c_init: float,
        discount: float,
        players=1,
        continuous_action_space=False,
):
    """
    Overview:
        calculate the pUCB score.
    Arguments:
        - child (:obj:`Any`): a child node
        - players (:obj:`int`): one/two_player mode board games
    """
    pb_c = math.log((total_children_visit_counts + pb_c_base + 1) / pb_c_base) + pb_c_init
    pb_c *= (math.sqrt(total_children_visit_counts) / (child.visit_count + 1))

    # prior_score = pb_c * child.prior

    ##################
    # sampled related code
    ##################
    # TODO
    node_prior = "density"
    # Uniform prior for continuous action space
    if node_prior == "uniform":
        prior_score = pb_c * (1 / len(parent.children))
    elif node_prior == "density":
        # TODO(pu): empirical distribution
        if continuous_action_space:
            # prior is log_prob
            prior_score = pb_c * (
                    torch.exp(child.prior) / (sum([torch.exp(node.prior) for node in parent.children.values()]) + 1e-9)
            )
        else:
            # prior is prob
            prior_score = pb_c * (
                    child.prior / (sum([node.prior for node in parent.children.values()]) + 1e-9)
            )
    else:
        raise ValueError("{} is unknown prior option, choose uniform or density")
    if child.visit_count == 0:
        value_score = parent_mean_q
    else:
        true_reward = child.reward
        if players == 1:
            value_score = true_reward + discount * child.value
        elif players == 2:
            value_score = true_reward + discount * (-child.value)

    value_score = min_max_stats.normalize(value_score)
    if value_score < 0:
        value_score = 0
    if value_score > 1:
        value_score = 1
    ucb_score = prior_score + value_score

    return ucb_score


def batch_traverse(
        roots, pb_c_base: int, pb_c_init: float, discount: float, min_max_stats_lst, results: SearchResults,
        virtual_to_play, continuous_action_space=False
):
    """
    Overview:
        traverse, also called expandsion. process a batch roots parallely
    Arguments:
        - roots (:obj:`Any`): a batch of root nodes to be expanded.
        - pb_c_base (:obj:`int`): constant c1 used in pUCT rule, typically 1.25
        - pb_c_init (:obj:`int`): constant c2 used in pUCT rule, typically 19652
        - discount (:obj:`int`): discount factor used i calculating bootstrapped value, if env is board_games, we set discount=1
        - virtual_to_play (:obj:`list`): the to_play list used in self_play collecting and trainin gin board games,
            `virtual` is to emphasize that actions are performed on an imaginary hidden state.
    """

    last_action = 0
    parent_q = 0.0
    results.search_lens = [None for i in range(results.num)]
    results.last_actions = [None for i in range(results.num)]

    results.nodes = [None for i in range(results.num)]
    results.hidden_state_index_x_lst = [None for i in range(results.num)]
    results.hidden_state_index_y_lst = [None for i in range(results.num)]
    if virtual_to_play is not None and virtual_to_play[0] is not None:
        players = 2
    else:
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
            mean_q = node.get_mean_q(is_root, parent_q, discount)
            is_root = 0
            parent_q = mean_q

            # select action according to the pUCT rule
            action = select_child(node, min_max_stats_lst.stats_lst[i], pb_c_base, pb_c_init, discount, mean_q, players,
                                  continuous_action_space)
            if virtual_to_play is not None and virtual_to_play[i] is not None:
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


class Action:
    """Class that represent an action of a game."""

    def __init__(self, value):
        self.value = value

    def __hash__(self):
        return hash(self.value.tostring())
        # return hash(self.value.tobyte())

    def __eq__(self, other):
        return (self.value == other.value).all()

    def __gt__(self, other):
        return self.value[0] > other.value[0]

    def __repr__(self):
        return str(self.value)

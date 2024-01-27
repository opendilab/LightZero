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
        The base class of Node for MuZero.
     """
    def __init__(self, prior: float, legal_actions: List = None, action_space_size: int = 9) -> None:
        self.prior = prior
        self.legal_actions = legal_actions
        self.action_space_size = action_space_size
        self.visit_count = 0
        self.value_sum = 0
        self.best_action = -1
        self.to_play = -1  # default -1 means play_with_bot_mode
        self.reward = 0
        self.value_prefix = 0.0
        self.children = {}
        self.children_index = []
        self.simulation_index = 0
        self.batch_index = 0
        self.parent_value_prefix = 0  # only used in update_tree_q method

    def expand(self, to_play: int, simulation_index: int, batch_index: int, reward: float,
               policy_logits: List[float]) -> None:
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
        self.to_play = to_play
        if self.legal_actions is None:
            self.legal_actions = np.arange(len(policy_logits))

        self.simulation_index = simulation_index
        self.batch_index = batch_index
        self.reward = reward

        policy_values = torch.softmax(torch.tensor([policy_logits[a] for a in self.legal_actions]), dim=0).tolist()
        policy = {a: policy_values[i] for i, a in enumerate(self.legal_actions)}
        for action, prior in policy.items():
            self.children[action] = Node(prior)

    # 另一个版本的expand函数，将复用数据中的Q值在expand时加入
    # def expand_with_reuse(self, to_play: int, simulation_index: int, batch_index: int, reward: float,
    #            policy_logits: List[float], arm: int, arm_value, arm_visit) -> None:
    #     """
    #     Overview:
    #         Expand the child nodes of the current node.
    #     Arguments:
    #         - to_play (:obj:`Class int`): which player to play the game in the current node.
    #         - simulation_index (:obj:`Class int`): the x/first index of hidden state vector of the current node, i.e. the search depth.
    #         - batch_index (:obj:`Class int`): the y/second index of hidden state vector of the current node, i.e. the index of batch root node, its maximum is ``batch_size``/``env_num``.
    #         - value_prefix: (:obj:`Class float`): the value prefix of the current node.
    #         - policy_logits: (:obj:`Class List`): the policy logit of the child nodes.
    #     """
    #     self.to_play = to_play
    #     if self.legal_actions is None:
    #         self.legal_actions = np.arange(len(policy_logits))

    #     self.simulation_index = simulation_index
    #     self.batch_index = batch_index
    #     self.reward = reward

    #     policy_values = torch.softmax(torch.tensor([policy_logits[a] for a in self.legal_actions]), dim=0).tolist()
    #     policy = {a: policy_values[i] for i, a in enumerate(self.legal_actions)}
    #     for action, prior in policy.items():
    #         self.children[action] = Node(prior)
    #         # 判断是不是已知的arm，如果是的话就预先填好visit和value
    #         if action == arm:
    #             self.children[action].visit_count = arm_visit
    #             self.children[action].value_sum = arm_visit * arm_value

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
            noise = noises[i]
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
                # TODO(pu): why only one step bootstrap?
                q_of_s_a = true_reward + discount_factor * child.value
                total_unsigned_q += q_of_s_a
                total_visits += 1
        if is_root and total_visits > 0:
            mean_q = total_unsigned_q / total_visits
        else:
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
            Expand the roots and add noises.
        Arguments:
            - root_noise_weight: the exploration fraction of roots
            - noises: the vector of noise add to the roots.
            - rewards: the vector of rewards of each root.
            - policies: the vector of policy logits of each root.
            - to_play_batch: the vector of the player side of each root.

        """
        for i in range(self.root_num):
        #这里写个特别的expand函数，能把复用信息expand到节点里
            if to_play is None:
                self.roots[i].expand(-1, 0, i, rewards[i], policies[i])
            else:
                self.roots[i].expand(to_play[i], 0, i, rewards[i], policies[i])

            # 这里加噪不变
            self.roots[i].add_exploration_noise(root_noise_weight, noises[i])
            self.roots[i].visit_count += 1

    # 实现一板prepare函数，能单独对已知arm进行特殊展开
    # 复用的形式可以有几种，但是每一种都要把已知arm的访问次数和搜索的value结果得到,搜索得到的概率分布要不要呢？怎么利用呢？（暂时不用）
    def prepare_with_reuse(
                    self,
            root_noise_weight: float,
            noises: List[float],
            rewards: List[float],
            policies: List[List[float]],
            arms,
            arm_values,
            arm_visit,
            to_play: int = -1
    ) -> None:
        """
        Overview:
            Expand the roots and add noises.
        Arguments:
            - root_noise_weight: the exploration fraction of roots
            - noises: the vector of noise add to the roots.
            - rewards: the vector of rewards of each root.
            - policies: the vector of policy logits of each root.
            - to_play_batch: the vector of the player side of each root.

        """
        # 这边在expand的时候就给arm所对应的节点赋予value和visit值
        for i in range(self.root_num):
            if to_play is None:
                self.roots[i].expand_with_reuse(-1, 0, i, rewards[i], policies[i], arms[i], arm_values[i], arm_visit)
            else:
                self.roots[i].expand_with_reuse(to_play[i], 0, i, rewards[i], policies[i], arms[i], arm_values[i], arm_visit)

            self.roots[i].add_exploration_noise(root_noise_weight, noises[i])
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
                self.roots[i].expand(-1, 0, i, rewards[i], policies[i])
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
        self.latent_state_index_in_search_path = []
        self.latent_state_index_in_batch = []
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
        root: Node, min_max_stats: MinMaxStats, pb_c_base: float, pb_c_int: float, discount_factor: float,
        mean_q: float, players: int
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
        - players (:obj:`Class Int`): the number of players. one/in self-play-mode board games.
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
            # NOTE: if the difference is less than epsilon = 0.000001, we random choice action from max_index_lst
            max_index_lst.append(a)

    action = 0
    if len(max_index_lst) > 0:
        action = random.choice(max_index_lst)
    return action

def select_root_child(
        root: Node, min_max_stats: MinMaxStats, pb_c_base: float, pb_c_int: float, discount_factor: float,
        mean_q: float, players: int, true_action=None, reuse_value = 0
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
        - players (:obj:`Class Int`): the number of players. one/in self-play-mode board games.
    Returns:
        - action (:obj:`Union[int, float]`): Choose the action with the highest ucb score.
    """
    max_score = -np.inf
    epsilon = 0.000001
    max_index_lst = []
    # 如果a是已知动作则不进行ucbscore的计算
    for a in root.legal_actions:
        child = root.get_child(a)
        if a == true_action:
            temp_score = compute_arm_score(
                child, min_max_stats, mean_q, reuse_value, root.visit_count, pb_c_base, pb_c_int, discount_factor, players
            )
        else:
            temp_score = compute_ucb_score(
                child, min_max_stats, mean_q, root.visit_count, pb_c_base, pb_c_int, discount_factor, players
            )
        if max_score < temp_score:
            max_score = temp_score
            max_index_lst.clear()
            max_index_lst.append(a)
        elif temp_score >= max_score - epsilon:
            # NOTE: if the difference is less than epsilon = 0.000001, we random choice action from max_index_lst
            max_index_lst.append(a)

    action = 0
    if len(max_index_lst) > 0:
        action = random.choice(max_index_lst)
    return action



# !!!!!!!!!!!!!!!!!!!搞懂ucb score是怎么计算的！！！！！！！！！！！！！！！！！！！
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
            - continuous_action_space: whether the action space is continuous in current env.
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


def compute_arm_score(
        child: Node,
        min_max_stats: MinMaxStats,
        parent_mean_q: float,
        reuse_value,
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
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!想想对于这个已知的arm，prior要怎么处理
    pb_c = math.log((total_children_visit_counts + pb_c_base + 1) / pb_c_base) + pb_c_init
    pb_c *= (math.sqrt(total_children_visit_counts) / (child.visit_count + 1))

    prior_score = pb_c * child.prior
    # if child.visit_count == 0:
    #     value_score = parent_mean_q
    # else:
    #     true_reward = child.reward
    #     if players == 1:
    #         value_score = true_reward + discount_factor * child.value
    #     elif players == 2:
    #         value_score = true_reward + discount_factor * (-child.value)

    # value_score = min_max_stats.normalize(value_score)
    # 这边visit_count为0时不用预设为parent_q了，直接将value值设定为reuse_value值，以及将reuse_value update到minmaxstatus里
    if child.visit_count == 0:
        value_score = parent_mean_q
    else:
        # !!!!!!!!!!!!!!!!!!!这边暂时不考虑玩家切换的问题！！！！！！！！！！！！！！！！！！！！！！！！1
        true_reward = child.reward
        value_score = true_reward + discount_factor * reuse_value
    min_max_stats.update(value_score)
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
        true_action,
        reuse_value,
) -> Tuple[List[None], List[None], List[None], list]:
    """
    Overview:
        traverse, also called expansion. process a batch roots parallelly.
    Arguments:
        - roots (:obj:`Any`): a batch of root nodes to be expanded.
        - pb_c_base (:obj:`float`): constant c1 used in pUCT rule, typically 1.25.
        - pb_c_init (:obj:`float`): constant c2 used in pUCT rule, typically 19652.
        - discount_factor (:obj:`float`): The discount factor used in calculating bootstrapped value, if env is board_games, we set discount_factor=1.
        - virtual_to_play (:obj:`list`): the to_play list used in self_play collecting and training in board games,
            `virtual` is to emphasize that actions are performed on an imaginary hidden state.
        - continuous_action_space: whether the action space is continuous in current env.
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
    # print(f"the traverse range is {results.num}")
    for i in range(results.num):
        # print(f"i is {i}")
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
            parent_q = mean_q

            # select action according to the pUCT rule.
            # 修改select child函数，使其在根节点处选择节点时，考虑已知节点
            if is_root and true_action is not None and reuse_value is not None:
                # print(f"min_max_stats_lst.stats_lst[i] is {min_max_stats_lst.stats_lst[i]}")
                # print(f"true_action[i] is {true_action[i]}")
                # print(f"reuse_value[i] is {reuse_value[i]}")
                action = select_root_child(
                    node, min_max_stats_lst.stats_lst[i], pb_c_base, pb_c_init, discount_factor, mean_q, players, true_action[i], reuse_value[i]   
                )
            else:
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

            results.latent_state_index_in_search_path[i] = parent.simulation_index
            results.latent_state_index_in_batch[i] = parent.batch_index
            results.last_actions[i] = last_action
            results.search_lens[i] = search_len
            # while we break out the while loop, results.nodes[i] save the leaf node.
            results.nodes[i] = node

            if true_action is not None and reuse_value is not None:
                if is_root and action == true_action[i] and node.expanded:
                    # print("break is triggered")
                    print()
                    break
            is_root = 0

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

            true_reward = node.reward

            # TODO(pu): the effect of different ways to update min_max_stats
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

            # true_reward is in the perspective of current player of node
            # bootstrap_value = (true_reward if node.to_play == to_play else - true_reward) + discount_factor * bootstrap_value
            bootstrap_value = (
                                  -true_reward if node.to_play == to_play else true_reward
                              ) + discount_factor * bootstrap_value


def batch_backpropagate(
        simulation_index: int,
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
        - simulation_index (:obj:`Class Int`): The index of latent state of the leaf node in the search path.
        - discount_factor (:obj:`Class Float`): The discount factor used in calculating bootstrapped value, if env is board_games, we set discount_factor=1.
        - value_prefixs (:obj:`Class List`): the value prefixs of nodes along the search path.
        - values (:obj:`Class List`):  the values to propagate along the search path.
        - policies (:obj:`Class List`): the policy logits of nodes along the search path.
        - min_max_stats_lst (:obj:`Class List[MinMaxStats]`):  a tool used to min-max normalize the q value.
        - results (:obj:`Class List`): the search results.
        - to_play (:obj:`Class List`): the batch of which player is playing on this node.
    """
    for i in range(results.num):
        # ****** expand the leaf node ******
        if to_play is None:
            # set to_play=-1, because in self-play-mode to_play = {1,2}
            results.nodes[i].expand(-1, simulation_index, i, value_prefixs[i], policies[i])
        else:
            results.nodes[i].expand(to_play[i], simulation_index, i, value_prefixs[i], policies[i])

        # ****** backpropagate ******
        if to_play is None:
            backpropagate(results.search_paths[i], min_max_stats_lst.stats_lst[i], 0, values[i], discount_factor)
        else:
            backpropagate(
                results.search_paths[i], min_max_stats_lst.stats_lst[i], to_play[i], values[i], discount_factor
            )

"""
The Node and MCTS class for AlphaZero.
"""

import copy
import math

import numpy as np
import torch
import torch.nn as nn
from easydict import EasyDict
from typing import List, Dict, Any, Tuple, Union, TYPE_CHECKING
if TYPE_CHECKING:
    from ding.framework import OnlineRLContext


class Node(object):
    """
    Overview:
        The node base class for tree_search.
    """

    def __init__(self, parent: Node, prior_p: float) -> None:
        self._parent = parent
        self._children = {}
        self._visit_count = 0
        self._value_sum = 0
        self.prior_p = prior_p

    @property
    def value(self) -> float:
        """
        Overview:
            The value of the current node.
        Returns:
            - output (:obj:`Int`): Current value, used to compute ucb score.
        """
        if self._visit_count == 0:
            return 0
        return self._value_sum / self._visit_count

    def update(self, value: float) -> None:
        """
        Overview:
            Updata the current node information, such as visit_count and value_sum.
        Arguments:
            - value (:obj:`Int`): The value of the node.
        """
        self._visit_count += 1
        self._value_sum += value

    def update_recursive(self, leaf_value: float, mcts_mode: str) -> None:
        """
        Overview:
            Update node information recursively.
        Arguments:
            - leaf_value (:obj:`Int`): The value of the node.
        """
        if mcts_mode == 'self_play_mode':
            self.update(leaf_value)
            if self.is_root():
                return
            self._parent.update_recursive(-leaf_value, mcts_mode)
        if mcts_mode == 'play_with_bot_mode':
            self.update(leaf_value)
            if self.is_root():
                return
            self._parent.update_recursive(leaf_value, mcts_mode)

    def is_leaf(self) -> Dict:
        """
        Overview:
            Check if the current node is a leaf node or not.
        Returns:
            - output (:obj:`Dict`): Dict type children node.
        """
        return self._children == {}

    def is_root(self) -> bool:
        """
        Overview:
            Check if the current node is a root node or not.
        Returns:
            - output (:obj:`Bool`): Whether it is the parent node.
        """
        return self._parent is None

    @property
    def parent(self) -> None:
        return self._parent

    @property
    def children(self) -> None:
        return self._children

    @property
    def visit_count(self) -> None:
        return self._visit_count


class MCTS(object):
    """
    Overview:
        MCTS search process.
    """

    def __init__(self, cfg: EasyDict) -> None:
        self._cfg = cfg

        self._max_moves = self._cfg.get('max_moves', 512)  # for chess and shogi, 722 for Go.
        self._num_simulations = self._cfg.get('num_simulations', 800)

        # UCB formula
        self._pb_c_base = self._cfg.get('pb_c_base', 19652)  # 19652
        self._pb_c_init = self._cfg.get('pb_c_init', 1.25)  # 1.25

        # Root prior exploration noise.
        self._root_dirichlet_alpha = self._cfg.get(
            'root_dirichlet_alpha', 0.3
        )  # 0.3  # for chess, 0.03 for Go and 0.15 for shogi.
        self._root_exploration_fraction = self._cfg.get('root_exploration_fraction', 0.25)  # 0.25

    def get_next_action(self, simulate_env: BaseEnv, policy_forward_fn: function, temperature: int = 1.0, sample: bool = True) -> Tuple[int, List[float]]:
        """
        Overview:
            calculate the move probabilities based on visit counts at the root node.
        Arguments:
            - simulate_env (:obj:`Class BaseGameEnv`): The class of simulate env.
            - policy_forward_fn (:obj:`Function`): The function to compute the action probs and state value.
            - temperature (:obj:`Int`): Temperature is a parameter that controls the "softness" of the probability distribution..
            - sample (:obj:`Bool`): The value of the node.
        Returns:
            - action (:obj:`Bool`): Select the action with the most visits as the final action.
            - action_probs (:obj:`List`): The output probability of each action.
        """
        root = Node(None, 1.0)
        self._expand_leaf_node(root, simulate_env, policy_forward_fn)
        if sample:
            self._add_exploration_noise(root)

        # for debugging
        # print(simulate_env.board)
        # print('value= {}'.format([(k, v.value) for k,v in root.children.items()]))
        # print('visit_count= {}'.format([(k, v.visit_count) for k,v in root.children.items()]))
        # print('legal_action= {}',format(simulate_env.legal_actions))

        for n in range(self._num_simulations):
            simulate_env_copy = copy.deepcopy(simulate_env)
            simulate_env_copy.battle_mode = simulate_env_copy.mcts_mode
            self._simulate(root, simulate_env_copy, policy_forward_fn)

        # for debugging
        # print('after simulation')
        # print('value= {}'.format([(k, v.value) for k,v in root.children.items()]))
        # print('visit_count= {}'.format([(k, v.visit_count) for k,v in root.children.items()]))

        action_visits = []
        for action in range(simulate_env.action_space.n):
            if action in root.children:
                action_visits.append((action, root.children[action].visit_count))
            else:
                action_visits.append((action, 0))

        actions, visits = zip(*action_visits)
        action_probs = nn.functional.softmax(1.0 / temperature * np.log(torch.as_tensor(visits) + 1e-10), dim=0).numpy()
        if sample:
            action = np.random.choice(actions, p=action_probs)
        else:
            action = actions[np.argmax(action_probs)]
        # print(action)
        return action, action_probs

    def _simulate(self, node: Node, simulate_env: BaseEnv, policy_forward_fn: function) -> None:
        """
        Overview:
            Run a single playout from the root to the leaf, getting a value at the leaf and propagating it back through its parents.
            State is modified in-place, so a deepcopy must be provided.
        Arguments:
            - node (:obj:`Class Node`): Current node when performing mcts search.
            - simulate_env (:obj:`Class BaseGameEnv`): The class of simulate env.
            - policy_forward_fn (:obj:`Function`): The function to compute the action probs and state value.
        """
        while not node.is_leaf():
            # print(node.children.keys())
            action, node = self._select_child(node, simulate_env)
            if action is None:
                break
            # print('legal_action={}'.format(simulate_env.legal_actions))
            # print('action={}'.format(action))
            simulate_env.step(action)
            # print(node.is_leaf())

        done, winner = simulate_env.get_done_winner()

        """
        in ``self_play_mode``, the leaf_value is calculated from the perspective of player ``simulate_env.current_player``.
        in ``play_with_bot_mode``, the leaf_value is calculated from the perspective of player 1.
        """

        if not done:
            leaf_value = self._expand_leaf_node(node, simulate_env, policy_forward_fn)
        else:
            if simulate_env.mcts_mode == 'self_play_mode':
                if winner == -1:
                    leaf_value = 0
                else:
                    leaf_value = 1 if simulate_env.current_player == winner else -1

            if simulate_env.mcts_mode == 'play_with_bot_mode':
                # in ``play_with_bot_mode``, the leaf_value should be transformed to the perspective of player 1.
                if winner == -1:
                    leaf_value = 0
                elif winner == 1:
                    leaf_value = 1
                elif winner == 2:
                    leaf_value = -1

        # Update value and visit count of nodes in this traversal.
        if simulate_env.mcts_mode == 'play_with_bot_mode':
            node.update_recursive(leaf_value, simulate_env.mcts_mode)
        elif simulate_env.mcts_mode == 'self_play_mode':
            # NOTE: e.g.
            #       to_play: 1  ---------->  2  ---------->  1  ----------> 2
            #         state: s1 ---------->  s2 ---------->  s3 ----------> s4
            #                                     action    node
            #                                            leaf_value
            # leaf_value is calculated from the perspective of player 1, leaf_value = value_func(s3),
            # but node.value should be the value of E[q(s2, action)], i.e. calculated from the perspective of player 2.
            # thus we add the negative when call update_recursive().
            node.update_recursive(-leaf_value, simulate_env.mcts_mode)

    def _select_child(self, node: Node, simulate_env: BaseEnv) -> Tuple[Union[int, float], Node]:
        """
        Overview:
            Select the child with the highest UCB score.
        Arguments:
            - node (:obj:`Class Node`): Current node.
        Returns:
            - action (:obj:`Int`): choose the action with the highest ucb score.
            - child (:obj:`Node`): the child node reached by executing the action with the highest ucb score.
        """
        action = None
        child = None
        best_score = -9999999
        for action_tmp, child_tmp in node.children.items():
            # print(a, simulate_env.legal_actions)
            if action_tmp in simulate_env.legal_actions:
                score = self._ucb_score(node, child_tmp)
                if score > best_score:
                    best_score = score
                    action = action_tmp
                    child = child_tmp
        if child is None:
            child = node  # child==None, node is leaf node in play_with_bot_mode.

        return action, child

    def _expand_leaf_node(self, node: Node, simulate_env: BaseEnv, policy_forward_fn: function) -> float:
        """
        Overview:
            expand the node with the policy_forward_fn.
        Arguments:
            - node (:obj:`Class Node`): current node when performing mcts search.
            - simulate_env (:obj:`Class BaseGameEnv`): the class of simulate env.
            - policy_forward_fn (:obj:`Function`): the function to compute the action probs and state value.
        Returns:
            - leaf_value (:obj:`Bool`): the leaf node's value.
        """
        action_probs_dict, leaf_value = policy_forward_fn(simulate_env)
        for action, prior_p in action_probs_dict.items():
            if action in simulate_env.legal_actions:
                node.children[action] = Node(parent=node, prior_p=prior_p)
        return leaf_value

    def _ucb_score(self, parent: Node, child: Node) -> float:
        """
        Overview:
            Compute UCB score. The score for a node is based on its value, plus an exploration bonus based on the prior.
        Arguments:
            - parent (:obj:`Class Node`): Current node.
            - child (:obj:`Class Node`): Current node's child.
        Returns:
            - score (:obj:`Bool`): The UCB score.
        """
        pb_c = math.log((parent.visit_count + self._pb_c_base + 1) / self._pb_c_base) + self._pb_c_init
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

        prior_score = pb_c * child.prior_p
        value_score = child.value
        return prior_score + value_score

    def _add_exploration_noise(self, node: Node) -> None:
        """
        Overview:
            Add exploration noise.
        Arguments:
            - node (:obj:`Class Node`): Current node.
        """
        actions = node.children.keys()
        noise = np.random.gamma(self._root_dirichlet_alpha, 1, len(actions))
        frac = self._root_exploration_fraction
        for a, n in zip(actions, noise):
            node.children[a].prior_p = node.children[a].prior_p * (1 - frac) + n * frac

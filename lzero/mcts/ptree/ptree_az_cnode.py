"""
The Node and MCTS class for AlphaZero.
"""

import copy
import math

import numpy as np
import torch
import torch.nn as nn
from easydict import EasyDict
from typing import List, Dict, Any, Tuple, Union, Callable, Type
from ding.envs import BaseEnv

import sys
sys.path.append('/Users/puyuan/code/LightZero/lzero/mcts/ctree/ctree_alphazero/build')

import mcts_alphazero


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
        self._root_noise_weight = self._cfg.get('root_noise_weight', 0.25)  # 0.25

    def get_next_action(
            self,
            simulate_env: Type[BaseEnv],
            policy_forward_fn: Callable,
            temperature: float = 1.0,
            sample: bool = True
    ) -> Tuple[int, List[float]]:
        """
        Overview:
            calculate the move probabilities based on visit counts at the root node.
        Arguments:
            - simulate_env (:obj:`Class BaseGameEnv`): The class of simulate env.
            - policy_forward_fn (:obj:`Function`): The Callable to compute the action probs and state value.
            - temperature (:obj:`Int`): Temperature is a parameter that controls the "softness" of the probability distribution.
            - sample (:obj:`Bool`): The value of the node.
        Returns:
            - action (:obj:`Bool`): Select the action with the most visits as the final action.
            - action_probs (:obj:`List`): The output probability of each action.
        """
        # root = Node()
        
        # TODO(pu)
        # import sys
        # sys.path.append('/Users/puyuan/code/LightZero/lzero/mcts/ctree/ctree_alphazero/build')
        # import mcts_alphazero
        # root = mcts_alphazero.Node()
        # root.update_recursive(1, 'self_play_mode')

        root = mcts_alphazero.Node()

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

        # TODO(pu)
        # root.end_game()

        actions, visits = zip(*action_visits)
        action_probs = nn.functional.softmax(1.0 / temperature * np.log(torch.as_tensor(visits) + 1e-10), dim=0).numpy()
        if sample:
            action = np.random.choice(actions, p=action_probs)
        else:
            action = actions[np.argmax(action_probs)]
        # print(action)
        return action, action_probs

    def _simulate(self, node, simulate_env: Type[BaseEnv], policy_forward_fn: Callable) -> None:
        """
        Overview:
            Run a single playout from the root to the leaf, getting a value at the leaf and propagating it back through its parents.
            State is modified in-place, so a deepcopy must be provided.
        Arguments:
            - node (:obj:`Class Node`): Current node when performing mcts search.
            - simulate_env (:obj:`Class BaseGameEnv`): The class of simulate env.
            - policy_forward_fn (:obj:`Function`): The Callable to compute the action probs and state value.
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
        print('position0')

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
            print('position1')
            node.update_recursive(-leaf_value, simulate_env.mcts_mode)
            print('position2')
        
        # TODO(pu)
        # node.end_game(node)


    def _select_child(self, node, simulate_env: Type[BaseEnv]):# -> Tuple[Union[int, float], Node]:
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
        # print(simulate_env._raw_env._go.board, simulate_env.legal_actions)
        for action_tmp, child_tmp in node.children.items():
            if action_tmp in simulate_env.legal_actions:
                score = self._ucb_score(node, child_tmp)
                if score > best_score:
                    best_score = score
                    action = action_tmp
                    child = child_tmp
        if child is None:
            child = node  # child==None, node is leaf node in play_with_bot_mode.

        return action, child

    def _expand_leaf_node(self, node, simulate_env: Type[BaseEnv], policy_forward_fn: Callable) -> float:
        """
        Overview:
            expand the node with the policy_forward_fn.
        Arguments:
            - node (:obj:`Class Node`): current node when performing mcts search.
            - simulate_env (:obj:`Class BaseGameEnv`): the class of simulate env.
            - policy_forward_fn (:obj:`Function`): the Callable to compute the action probs and state value.
        Returns:
            - leaf_value (:obj:`Bool`): the leaf node's value.
        """
        action_probs_dict, leaf_value = policy_forward_fn(simulate_env)
        # simulate_env._raw_env._go.board
        for action, prior_p in action_probs_dict.items():
            if action in simulate_env.legal_actions:
                # node.children[action] = mcts_alphazero.Node(parent=node, prior_p=prior_p)
                node.add_child(action, mcts_alphazero.Node(parent=node, prior_p=prior_p))
        return leaf_value

    def _ucb_score(self, parent, child) -> float:
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

    def _add_exploration_noise(self, node) -> None:
        """
        Overview:
            Add exploration noise.
        Arguments:
            - node (:obj:`Class Node`): Current node.
        """
        actions = node.children.keys()
        # TODO: check if this is correct.
        noise = np.random.gamma(self._root_dirichlet_alpha, 1, len(actions))
        frac = self._root_noise_weight
        for a, n in zip(actions, noise):
            node.children[a].prior_p = node.children[a].prior_p * (1 - frac) + n * frac

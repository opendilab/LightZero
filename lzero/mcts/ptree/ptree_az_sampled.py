"""
Overview:
    This code implements the Monte Carlo Tree Search (MCTS) algorithm with the integration of neural networks. 
    The Node class represents a node in the Monte Carlo tree and implements the basic functionalities expected in a node. 
    The MCTS class implements the specific search functionality and provides the optimal action through the ``get_next_action`` method. 
    Compared to traditional MCTS, the introduction of value networks and policy networks brings several advantages.
    During the expansion of nodes, it is no longer necessary to explore every single child node, but instead, 
    the child nodes are directly selected based on the prior probabilities provided by the neural network. 
    This reduces the breadth of the search. When estimating the value of leaf nodes, there is no need for a rollout; 
    instead, the value output by the neural network is used, which saves the depth of the search.
"""

import math
from typing import List, Tuple, Union, Callable, Type, Dict, Any

import numpy as np
import torch
from ding.envs import BaseEnv
from easydict import EasyDict

from lzero.mcts.ptree.ptree_sez import Action


class Node(object):
    """
    Overview:
        A class for a node in a Monte Carlo Tree. The properties of this class store basic information about the node, 
        such as its parent node, child nodes, and the number of times the node has been visited. 
        The methods of this class implement basic functionalities that a node should have, such as propagating the value back, 
        checking if the node is the root node, and determining if it is a leaf node.
    """

    def __init__(self, parent: "Node" = None, prior_p: float = 1.0) -> None:
        """
        Overview:
            Initialize a Node object.
        Arguments:
            - parent (:obj:`Node`): The parent node of the current node.
            - prior_p (:obj:`Float`): The prior probability of selecting this node.
        """
        # The parent node.
        self._parent = parent
        # A dictionary representing the children of the current node. The keys are the actions, and the values are
        # the child nodes.
        self._children = {}
        # The number of times this node has been visited.
        self._visit_count = 0
        # The sum of the values of all child nodes of this node.
        self._value_sum = 0
        # The prior probability of selecting this node.
        self.prior_p = prior_p

    @property
    def value(self) -> float:
        """
        Overview:
            The value of the current node.
        Returns:
            - output (:obj:`Int`): Current value, used to compute ucb score.
        """
        # Computes the average value of the current node.
        if self._visit_count == 0:
            return 0
        return self._value_sum / self._visit_count

    def update(self, value: float) -> None:
        """
        Overview:
            Update the current node information, such as ``_visit_count`` and ``_value_sum``.
        Arguments:
            - value (:obj:`Float`): The value of the node.
        """
        # Updates the number of times this node has been visited.
        self._visit_count += 1
        # Updates the sum of the values of all child nodes of this node.
        self._value_sum += value

    def update_recursive(self, leaf_value: float, mcts_mode: str) -> None:
        """
        Overview:
            Update node information recursively.
            The same game state has opposite values in the eyes of two players playing against each other. 
            The value of a node is evaluated from the perspective of the player corresponding to its parent node. 
            In ``self_play_mode``, because the player corresponding to a node changes every step during the backpropagation process, the value needs to be negated once. 
            In ``play_with_bot_mode``, since all nodes correspond to the same player, the value does not need to be negated.

        Arguments:
            - leaf_value (:obj:`Float`): The value of the node.
            - mcts_mode (:obj:`str`): The mode of MCTS, can be 'self_play_mode' or 'play_with_bot_mode'.
        """
        # Update the node information recursively based on the MCTS mode.
        if mcts_mode == 'self_play_mode':
            # Update the current node's information.
            self.update(leaf_value)
            # If the current node is the root node, return.
            if self.is_root():
                return
            # Update the parent node's information recursively. When propagating the value back to the parent node,
            # the value needs to be negated once because the perspective of evaluation has changed.
            self._parent.update_recursive(-leaf_value, mcts_mode)
        if mcts_mode == 'play_with_bot_mode':
            # Update the current node's information.
            self.update(leaf_value)
            # If the current node is the root node, return.
            if self.is_root():
                return
            # Update the parent node's information recursively. In ``play_with_bot_mode``, since the nodes' values
            # are always evaluated from the perspective of the agent player, there is no need to negate the value
            # during value propagation.
            self._parent.update_recursive(leaf_value, mcts_mode)

    def is_leaf(self) -> bool:
        """
        Overview:
            Check if the current node is a leaf node or not.
        Returns:
            - output (:obj:`Bool`): If self._children is empty, it means that the node has not 
            been expanded yet, which indicates that the node is a leaf node.
        """
        # Returns True if the node is a leaf node (i.e., has no children), and False otherwise.
        return self._children == {}

    def is_root(self) -> bool:
        """
        Overview:
            Check if the current node is a root node or not.
        Returns:
            - output (:obj:`Bool`): If the node does not have a parent node,
            then it is a root node.
        """
        return self._parent is None

    @property
    def parent(self) -> None:
        """
        Overview:
            Get the parent node of the current node.
        Returns:
            - output (:obj:`Node`): The parent node of the current node.
        """
        return self._parent

    @property
    def children(self) -> None:
        """
        Overview:
            Get the dictionary of children nodes of the current node.
        Returns:
            - output (:obj:`dict`): A dictionary representing the children of the current node. 
        """
        return self._children

    @property
    def visit_count(self) -> None:
        """
        Overview:
            Get the number of times the current node has been visited.
        Returns:
            - output (:obj:`Int`): The number of times the current node has been visited.
        """
        return self._visit_count


class MCTS(object):
    """
    Overview:
        A class for Monte Carlo Tree Search (MCTS). The methods in this class implement the steps involved in MCTS, such as selection and expansion. 
        Based on this, the ``_simulate`` method is used to traverse from the root node to a leaf node. 
        Finally, by repeatedly calling ``_simulate`` through ``get_next_action``, the optimal action is obtained.
    """

    def __init__(self, cfg: EasyDict, simulate_env: Type[BaseEnv]) -> None:
        """
        Overview:
            Initializes the MCTS process.
        Arguments:
            - cfg (:obj:`EasyDict`): A dictionary containing the configuration parameters for the MCTS process.
        """
        # Stores the configuration parameters for the MCTS search process.
        self._cfg = cfg

        # ==============================================================
        # sampled related core code
        # ==============================================================
        self.legal_actions = self._cfg.legal_actions
        self.action_space_size = self._cfg.action_space_size
        self.num_of_sampled_actions = self._cfg.num_of_sampled_actions
        print(f'num_of_sampled_actions: {self.num_of_sampled_actions}')
        self.continuous_action_space = self._cfg.continuous_action_space

        # The maximum number of moves allowed in a game.
        self._max_moves = self._cfg.get('max_moves', 512)  # for chess and shogi, 722 for Go.
        # The number of simulations to run for each move.
        self._num_simulations = self._cfg.get('num_simulations', 800)

        # UCB formula
        self._pb_c_base = self._cfg.get('pb_c_base', 19652)  # 19652
        self._pb_c_init = self._cfg.get('pb_c_init', 1.25)  # 1.25

        # Root prior exploration noise.
        self._root_dirichlet_alpha = self._cfg.get(
            'root_dirichlet_alpha', 0.3
        )  # 0.3  # for chess, 0.03 for Go and 0.15 for shogi.
        self._root_noise_weight = self._cfg.get('root_noise_weight', 0.25)  # 0.25
        self.mcts_search_cnt = 0
        self.simulate_env = simulate_env

    def get_next_action(
            self,
            state_config_for_env_reset: Dict[str, Any],
            policy_value_func: Callable,
            temperature: float = 1.0,
            sample: bool = True
    ) -> Tuple[int, List[float]]:
        """
        Overview:
            Get the next action to take based on the current state of the game.
        Arguments:
            - state_config_for_env_reset (:obj:`Dict`): The config of state when reset the env.
            - policy_value_func (:obj:`Function`): The Callable to compute the action probs and state value.
            - temperature (:obj:`Float`): The exploration temperature.
            - sample (:obj:`Bool`): Whether to sample an action from the probabilities or choose the most probable action.
        Returns:
            - action (:obj:`Int`): The selected action to take.
            - action_probs (:obj:`List`): The output probability of each action.
        """
        # Create a new root node for the MCTS search.
        self.root = Node()
        self.simulate_env.reset(
            start_player_index=state_config_for_env_reset.start_player_index,
            init_state=state_config_for_env_reset.init_state,
            katago_policy_init=True,
            katago_game_state=state_config_for_env_reset.katago_game_state,
        )
        # self.simulate_env_root = copy.deepcopy(self.simulate_env)
        self._expand_leaf_node(self.root, self.simulate_env, policy_value_func)

        if sample:
            self._add_exploration_noise(self.root)

        for n in range(self._num_simulations):
            self.simulate_env.reset(
                start_player_index=state_config_for_env_reset.start_player_index,
                init_state=state_config_for_env_reset.init_state,
                katago_policy_init=True,
                katago_game_state=state_config_for_env_reset.katago_game_state,
            )
            self.simulate_env.battle_mode = self.simulate_env.mcts_mode
            self._simulate(self.root, self.simulate_env, policy_value_func)

        # sampled related code
        # Get the visit count for each possible action at the root node.
        action_visits = []
        for action in range(self.simulate_env.action_space.n):
            # Create an Action object for the current action
            current_action_object = Action(action)

            # Use the Action object to look up the child node in the dictionary
            if current_action_object in self.root.children:
                action_visits.append((action, self.root.children[current_action_object].visit_count))
            else:
                action_visits.append((action, 0))

        # Unpack the tuples in action_visits list into two separate tuples: actions and visits.
        actions, visits = zip(*action_visits)
        # print('action_visits= {}'.format(visits))

        visits_t = torch.as_tensor(visits, dtype=torch.float32)
        visits_t /= temperature
        action_probs = (visits_t / visits_t.sum()).numpy()

        if sample:
            action = np.random.choice(actions, p=action_probs)
        else:
            action = actions[np.argmax(action_probs)]
        self.mcts_search_cnt += 1

        # print(f'self.simulate_env_root: {self.simulate_env_root.legal_actions}')
        # print(f'mcts_search_cnt: {self.mcts_search_cnt}')
        # print('action= {}'.format(action))
        # print('action_probs= {}'.format(action_probs))
        # Return the selected action and the output probability of each action.
        return action, action_probs

    def _simulate(self, node: Node, simulate_env: Type[BaseEnv], policy_value_func: Callable) -> None:
        """
        Overview:
            Run a single playout from the root to the leaf, getting a value at the leaf and propagating it back through its parents.
            State is modified in-place, so a deepcopy must be provided.
        Arguments:
            - node (:obj:`Class Node`): Current node when performing mcts search.
            - simulate_env (:obj:`Class BaseGameEnv`): The class of simulate env.
            - policy_value_func (:obj:`Function`): The Callable to compute the action probs and state value.
        """
        while not node.is_leaf():
            # only for debug
            # print('=='*20)
            # print('node.legal_actions: ', node.legal_actions)
            # print(node.children.keys())
            # print('simulate_env.board: ', simulate_env.board)
            # print('simulate_env.legal_actions:', simulate_env.legal_actions)

            # Traverse the tree until the leaf node.
            action, node = self._select_child(node, simulate_env)
            if action is None:
                break
            # sampled related code
            simulate_env.step(action.value)

        done, winner = simulate_env.get_done_winner()
        """
        in ``self_play_mode``, the leaf_value is calculated from the perspective of player ``simulate_env.current_player``.
        in ``play_with_bot_mode``, the leaf_value is calculated from the perspective of player 1.
        """

        if not done:
            # The leaf_value here is obtained from the neural network. The perspective of this value is from the
            # player corresponding to the game state input to the neural network. For example, if the current_player
            # of the current node is player 1, the value output by the network represents the goodness of the current
            # game state from the perspective of player 1.
            leaf_value = self._expand_leaf_node(node, simulate_env, policy_value_func)

        else:
            if simulate_env.mcts_mode == 'self_play_mode':
                # In a tie game, the value corresponding to a terminal node is 0.
                if winner == -1:
                    leaf_value = 0
                else:
                    # To maintain consistency with the perspective of the neural network, the value of a terminal
                    # node is also calculated from the perspective of the current_player of the terminal node,
                    # which is convenient for subsequent updates.
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

    def _select_child(self, node: Node, simulate_env: Type[BaseEnv]) -> Tuple[Union[int, float], Node]:
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
        child_node = None
        best_score = -9999999
        # print(simulate_env._raw_env._go.board, simulate_env.legal_actions)
        # Iterate over each child of the current node.
        for action_tmp, child_node_tmp in node.children.items():
            # print(a, simulate_env.legal_actions)
            # print('node.legal_actions: ', node.legal_actions)
            if action_tmp.value in simulate_env.legal_actions:
                score = self._ucb_score(node, child_node_tmp)
                # Check if the score of the current child is higher than the best score so far.
                if score > best_score:
                    best_score = score
                    action = action_tmp
                    child_node = child_node_tmp
            else:
                print(f'error: {action_tmp} not in {simulate_env.legal_actions}')
        if child_node is None:
            child_node = node  # child==None, node is leaf node in play_with_bot_mode.
        if action is None:
            print('error: action is None')

        return action, child_node

    def _expand_leaf_node(self, node: Node, simulate_env: Type[BaseEnv], policy_value_func: Callable) -> float:
        """
        Overview:
            expand the node with the policy_value_func.
        Arguments:
            - node (:obj:`Class Node`): current node when performing mcts search.
            - simulate_env (:obj:`Class BaseGameEnv`): the class of simulate env.
            - policy_value_func (:obj:`Function`): the Callable to compute the action probs and state value.
        Returns:
            - leaf_value (:obj:`Bool`): the leaf node's value.
        """
        # ==============================================================
        # sampled related core code
        # ==============================================================
        if self.continuous_action_space:
            pass
        else:
            # discrete action space

            # Call the policy_value_func function to compute the action probabilities and state value, and return a
            # dictionary and the value of the leaf node.
            legal_action_probs_dict, leaf_value = policy_value_func(simulate_env)

            node.legal_actions = []

            # Extract actions and their corresponding probabilities from the dictionary
            actions = list(legal_action_probs_dict.keys())
            probabilities = list(legal_action_probs_dict.values())

            # Normalize the probabilities so they sum to 1
            probabilities = np.array(probabilities)
            probabilities /= probabilities.sum()

            # self.num_of_sampled_actions = len(actions)

            # If there are fewer legal actions than the desired number of samples,
            # adjust the number of samples to the number of legal actions
            num_samples = min(len(actions), self.num_of_sampled_actions)
            # Use numpy to randomly sample actions according to the given probabilities, without replacement
            sampled_actions = np.random.choice(actions, size=num_samples, p=probabilities, replace=False)
            sampled_actions = sampled_actions.tolist()  # Convert numpy array to list

            for action_index in range(num_samples):
                node.children[Action(sampled_actions[action_index])] = \
                    Node(
                        parent=node,
                        prior_p=legal_action_probs_dict[sampled_actions[action_index]],
                    )
                node.legal_actions.append(Action(sampled_actions[action_index]))

            # Return the value of the leaf node.
            return leaf_value

    def _ucb_score(self, parent: Node, child: Node) -> float:
        """
        Overview:
            Compute UCB score. The score for a node is based on its value, plus an exploration bonus based on the prior.
            For more details, please refer to this paper: http://gauss.ececs.uc.edu/Workshops/isaim2010/papers/rosin.pdf
            UCB = Q(s,a) + P(s,a) \cdot \frac{N(\text{parent})}{1+N(\text{child})} \cdot \left(c_1 + \log\left(\frac{N(\text{parent})+c_2+1}{c_2}\right)\right)
            - Q(s,a): value of a child node.
            - P(s,a): The prior of a child node.
            - N(parent): The number of the visiting of the parent node.
            - N(child): The number of the visiting of the child node.
            - c_1: a parameter given by self._pb_c_init to control the influence of the prior P(s,a) relative to the value Q(s,a).
            - c_2: a parameter given by self._pb_c_base to control the influence of the prior P(s,a) relative to the value Q(s,a).
        Arguments:
            - parent (:obj:`Class Node`): Current node.
            - child (:obj:`Class Node`): Current node's child.
        Returns:
            - score (:obj:`Bool`): The UCB score.
        """
        # Compute the value of parameter pb_c using the formula of the UCB algorithm.
        pb_c = math.log((parent.visit_count + self._pb_c_base + 1) / self._pb_c_base) + self._pb_c_init
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

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
            if self.continuous_action_space:
                # prior is log_prob
                prior_score = pb_c * (
                        torch.exp(child.prior_p) / (
                        sum([torch.exp(node.prior_p) for node in parent.children.values()]) + 1e-6)
                )
            else:
                # prior is prob
                prior_score = pb_c * (child.prior_p / (sum([node.prior_p for node in parent.children.values()]) + 1e-6))
        else:
            raise ValueError("{} is unknown prior option, choose uniform or density")

        # Compute the UCB score by combining the prior score and value score.
        value_score = child.value
        prior_score = pb_c * child.prior_p

        return prior_score + value_score

    def _add_exploration_noise(self, node: Node) -> None:
        """
        Overview:
            Add exploration noise.
        Arguments:
            - node (:obj:`Class Node`): Current node.
        """
        # Get a list of actions corresponding to the child nodes.
        actions = list(node.children.keys())
        # Create a list of alpha values for Dirichlet noise.
        alpha = [self._root_dirichlet_alpha] * len(actions)
        # Generate Dirichlet noise using the alpha values.
        noise = np.random.dirichlet(alpha)
        # Compute the weight of the exploration noise.
        frac = self._root_noise_weight
        # Update the prior probability of each child node with the exploration noise.
        for a, n in zip(actions, noise):
            node.children[a].prior_p = node.children[a].prior_p * (1 - frac) + n * frac

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
        sampled_actions = self.root.legal_actions
        sampled_actions = np.array([action.value for action in sampled_actions])

        return sampled_actions

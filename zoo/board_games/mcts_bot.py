"""
Overview:
    This code implements an MCTSbot that uses MCTS to make decisions. 
    The MCTSnode is an abstract base class that specifies the basic methods that a Monte Carlo Tree node should have. 
    The TwoPlayersMCTSnode class inherits from this base class and implements the specific methods. 
    MCTS implements the search function, which takes in a root node and performs a search to obtain the optimal action. 
    MCTSbot integrates the above functions and can create a root node based on the current game environment, 
    and then calls MCTS to perform a search and make a decision. 
    For more details, you can refer to: https://github.com/int8/monte-carlo-tree-search.
"""

import time
from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np
from easydict import EasyDict

class MCTSNode(ABC):
    """
    Overview:
        This is an abstract base class that outlines the fundamental methods for a Monte Carlo Tree node. 
        Each specific method must be implemented in the subclasses for specific use-cases.
    """

    def __init__(self, env, parent=None):
        """
        Arguments:
            - env (:obj:`BaseEnv`): The game environment of the current node. 
                The properties of this object contain information about the current game environment. 
                For instance, in a game of tictactoe: 
                    - env.board: A (3,3) array representing the game board, e.g., 
                        [[0,2,0], 
                        [1,1,0], 
                        [2,0,0]] 
                        Here, 0 denotes an unplayed position, 1 represents a position occupied by player 1, and 2 indicates a position taken by player 2.
                    - env.players: A list [1,2] representing the two players, player 1 and player 2 respectively.
                    - env._current_player: Denotes the player who is to make a move in the current turn, which is alterating in each turn not only in the reset phase.
                The methods of this object implement functionalities such as game state transitions and retrieving game results.
            - parent (:obj:`MCTSNode`):  The parent node of the current node. The parent node is primarily used for backpropagation during the Monte Carlo Tree Search. 
                For the root node, this parent returns None as it does not have a parent node.
        """    
        self.env = env
        self.parent = parent
        self.children = []
        self.expanded_actions = []
        self.best_action = -1

    @property
    @abstractmethod
    def legal_actions(self):
        pass

    @property
    @abstractmethod
    def value(self):
        pass

    @property
    @abstractmethod
    def visit_count(self):
        pass

    @abstractmethod
    def expand(self):
        pass

    @abstractmethod
    def is_terminal_node(self):
        pass

    @abstractmethod
    def rollout(self):
        pass

    @abstractmethod
    def backpropagate(self, reward):
        pass

    def is_fully_expanded(self):
        """
        Overview:
            This method checks if the node is fully expanded. 
            A node is considered fully expanded when all of its child nodes have been selected at least once. 
            Whenever a new child node is selected for the first time, a corresponding action is removed from the list of legal actions. 
            Once the list of legal actions is depleted, it signifies that all child nodes have been selected, 
            thereby indicating that the parent node is fully expanded.
        """
        return len(self.legal_actions) == 0

    def best_child(self, c_param=1.4):
        """
        Overview:
            This function finds the best child node which has the highest UCB (Upper Confidence Bound) score. 
            The UCB formula is: 
            {UCT}(v_i, v) = \frac{Q(v_i)}{N(v_i)} + c \sqrt{\frac{\log(N(v))}{N(v_i)}}
                - Q(v_i) is the estimated value of the child node v_i.
                - N(v_i) is a counter of how many times the child node v_i has been on the backpropagation path.
                - N(v) is a counter of how many times the parent (current) node v has been on the backpropagation path.
                - c is a parameter which balances exploration and exploitation.
        Arguments:
            - c_param (:obj:`float`): a parameter which controls the balance between exploration and exploitation. Default value is 1.4.

        Returns:
            - node (:obj:`MCTSnode`)The child node which has the highest UCB score.

    """
        # Calculate the ucb score for every child node in the list.
        choices_weights = [(child_node.value / child_node.visit_count) + c_param * np.sqrt((2 * np.log(self.visit_count) / child_node.visit_count)) for child_node in self.children]
        # Save the best action based on the highest UCB score.
        self.best_action = self.expanded_actions[np.argmax(choices_weights)]
        # Choose the child node which has the highest ucb score.
        return self.children[np.argmax(choices_weights)]

    def rollout_policy(self, possible_actions):
        """
        Overview:
            This method implements the rollout policy for a node during the Monte Carlo Tree Search. 
            The rollout policy is used to determine the action taken during the simulation phase of the MCTS. 
            In this case, the policy is to randomly choose an action from the list of possible actions.
        Arguments:
            - possible_actions(:obj:`list`): A list of all possible actions that can be taken from the current state.
        Return: 
            - action(:obj:`int`): A randomly chosen action from the list of possible actions.
        """
        return possible_actions[np.random.randint(len(possible_actions))]

class TwoPlayersMCTSNode(MCTSNode):
    """
    Overview:
        This subclass inherits from the abstract base class and implements the specific methods required for a two players' Monte Carlo Tree node.
    """

    def __init__(self, env, parent=None):
        """
        Overview:
            This function initializes a new instance of the class. It sets the parent node, environment, and initializes the number of visits, results, and legal actions.
        Arguments:
            - env (:obj:`BaseEnv`): the environment object which contains information about the current game state.
            - parent (:obj:`MCTSNode`): the parent node of this node. If None, then this node is the root node.
        """
        super().__init__(env, parent)
        self._number_of_visits = 0.
        # A default dictionary which sets the value to 0 for undefined keys.
        self._results = defaultdict(int)
        self._legal_actions = None

    # Get all legal actions in current state from the environment object.
    @property
    def legal_actions(self):
        if self._legal_actions is None:
            self._legal_actions = self.env.legal_actions
        return self._legal_actions


    @property
    def value(self):
        """
        Overview:
            This property represents the estimated value (Q-value) of the current node. 
            self._results[1] represents the number of wins for player 1. 
            self._results[-1] represents the number of wins for player 2. 
            The Q-value is calculated depends on which player is the current player at the parent node, 
            and is computed as the difference between the wins of the current player and the opponent.
            If the parent's current player is player 1, Q-value is the difference of player 1's wins and player 2's wins.
            If the parent's current player is player 2, Q-value is the difference of player 2's wins and player 1's wins.
            For example, if self._results[1] = 10 (player 1's wins) and self._results[-1] = 5 (player 2's wins):
                - If the parent's current player is player 1, then Q-value = 10 - 5 = 5.
                - If the parent's current player is player 2, then Q-value = 5 - 10 = -5.
            This way, a higher Q-value for a node indicates a higher win rate for the parent's current player.
        """

        # Determine the number of wins and losses based on the current player at the parent node.
        wins, loses = (self._results[1], self._results[-1]) if self.parent.env.current_player == 1 else (self._results[-1], self._results[1])

        # Calculate and return the Q-value as the difference between wins and losses.
        return wins - loses

    @property
    def visit_count(self):
        """
        Overview: 
            This property represents the number of times the node has been visited during the search.
        """
        return self._number_of_visits

    def expand(self):
        """
        Overview:
            This method expands the current node by creating a new child node.
            It pops an action from the list of legal actions, simulates the action to get the next game state,
            and creates a new child node with that state. The new child node is then added to the list of children nodes.
        Returns:
            - node(:obj:`TwoPlayersMCTSNode`): The child node object that has been created.
        """
        
        # Choose an untried action from the list of legal actions and pop it out. Only untried actions are left in the list.
        action = self.legal_actions.pop()
        
        # The simulate_action() function returns a new environment which resets the board and the current player flag.
        next_simulator_env = self.env.simulate_action(action)
        
        # Create a new node object for the child node and append it to the children list.
        child_node = TwoPlayersMCTSNode(next_simulator_env, parent=self)
        self.children.append(child_node)
        
        # Add the action that has been tried to the expanded_actions list.
        self.expanded_actions.append(action)
        
        # Return the child node object.
        return child_node

    def is_terminal_node(self):
        """
            Overview:
                This function checks whether the current node is a terminal node.
                It uses the game environment's get_done_reward method to check if the game has ended.
            Returns:
                - A bool flag representing whether the game is over.
        """
        # The get_done_reward() returns a tuple (done, reward).
        # reward = ±1 when player 1 wins/loses the game.
        # reward = 0 when it is a tie.
        # reward = None when current node is not a teminal node.
        # done is the bool flag representing whether the game is over.
        return self.env.get_done_reward()[0]

    def rollout(self):
        """
        Overview:
            This method performs a rollout (simulation) from the current node.
            It repeatedly selects an action based on the rollout policy and simulates the action until the game ends.
            The method then returns the reward of the game's final state.
        Returns:
            -reward (:obj:`int`): reward = ±1 when player 1 wins/loses the game, reward = 0 when it is a tie, reward = None when current node is not a teminal node.
        """
        # print('simulation begin')
        current_rollout_env = self.env
        # print(current_rollout_env.board)
        while not current_rollout_env.get_done_reward()[0]:
            possible_actions = current_rollout_env.legal_actions
            action = self.rollout_policy(possible_actions)
            current_rollout_env = current_rollout_env.simulate_action(action)
            # print('\n')
            # print(current_rollout_env.board)
        # print('simulation end \n')
        return current_rollout_env.get_done_reward()[1]

    def backpropagate(self, result):
        """
        Overview:
            This method performs backpropagation from the current node.
            It increments the number of times the node has been visited and the number of wins for the result.
            If the current node has a parent, the method recursively backpropagates the result to the parent.
        """
        self._number_of_visits += 1.
        # result is the index of the self._results list.
        # result = ±1 when player 1 wins/loses the game.
        self._results[result] += 1.
        if self.parent:
            self.parent.backpropagate(result)


class MCTS(object):
    """
    Overview:
        This class implements Monte Carlo Tree Search from the root node, whose environment is the real environment of the game at the current moment. 
        After the tree search and rollout simulation, every child node of the root node has a UCB value.
        Then the decision for the root node is to choose the child node with the highest UCB value. 
    """

    def __init__(self, node):
        """
        Overview:
            This function initializes a new instance of the MCTS class with the given root node object.

        Parameters:
            - node (:obj:`TwoPlayersMCTSNode`): The root node object for the MCTS.
        """
        self.root = node

    def best_action(self, simulations_number=None, total_simulation_seconds=None, best_action_type="UCB"):
        """
        Overview:
            This function simulates the game by constantly selecting the best child node and backpropagating the result. 
        Arguments:
            - simulations_number (:obj:`int`): The number of simulations performed to get the best action.
            - total_simulation_seconds (:obj:`float`): The amount of time the algorithm has to run. Specified in seconds.
            - best_action_type (:obj:`str`): The type of best action selection to use. Either "UCB" or "most visited".
        Returns:
            - node(:obj:`TwoPlayersMCTSNode`): The best children node object, which contains the best action to take. 
        """

        # The search cost is determined by either the maximum number of simulations or the longest simulation time.
        # If no simulations number is provided, run simulations for the specified time.
        if simulations_number is None:
            assert (total_simulation_seconds is not None)
            end_time = time.time() + total_simulation_seconds
            while True:
                # Get the leaf node.
                leaf_node = self._tree_policy()
                # Rollout from the leaf node.
                reward = leaf_node.rollout()
                # Backpropagate from the leaf node to the root node.
                leaf_node.backpropagate(reward)
                if time.time() > end_time:
                    break
        # If simulations number is provided, run the specified number of simulations.
        else:
            for i in range(0, simulations_number):
                # print('****simlulation-{}****'.format(i))
                # Get the leaf node.
                leaf_node = self._tree_policy()
                # Rollout from the leaf node.
                reward = leaf_node.rollout()
                # print('reward={}\n'.format(reward))
                # Backpropagate from the leaf node to the root node.
                leaf_node.backpropagate(reward)
        # To select best child go for exploitation only.
        if best_action_type == "UCB":
            return self.root.best_child(c_param=0.)
        
        else:
            children_visit_counts = [child_node.visit_count for child_node in self.root.children]
            self.root.best_action = self.root.expanded_actions[np.argmax(children_visit_counts)]
            return self.root.children[np.argmax(children_visit_counts)]


    #
    def _tree_policy(self):
        """
        Overview:
            This function implements the tree search from the root node to the leaf node, which is either visited for the first time or is the terminal node.
            At each step, if the current node is not fully expanded, it expands.
            If it is fully expanded, it moves to the best child according to the tree policy.
        Returns:
            - node(:obj:`TwoPlayersMCTSNode`): The leaf node object that has been reached by the tree search. 
        """
        current_node = self.root
        while not current_node.is_terminal_node():
            if not current_node.is_fully_expanded():
                # choose a child node which has not been visited before
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node


class MCTSBot:
    """
    Overview:
        A robot which can use MCTS to make decision, choose an action to take.
    """
 
    def __init__(self, env, bot_name, num_simulation=10000):
        """
        Overview:
            This function initializes a new instance of the MCTSBot class.
        Arguments:
            - env (:obj:`BaseEnv`): The environment object for the game.
            - bot_name (:obj:`str`): The name of the MCTS Bot.
            - num_simulation (:obj:`int`): The number of simulations to perform during the MCTS. 
        """
        self.name = bot_name
        self.num_simulation = num_simulation
        self.simulator_env = env

    def get_actions(self, state, player_index, best_action_type = "UCB"):
        """
        Overview:
            This function gets the actions that the MCTS Bot will take.
            The environment is reset to the given state.
            Then, MCTS is performed with the specified number of simulations to find the best action.
        Arguments:
            - state (:obj:`list`): The current game state.
            - player_index (:obj:`int`): The index of the current player.
            - best_action_type (:obj:`str`): The type of best action selection to use. Either "UCB" or "most visited".
        Returns:
            - action (:obj:`int`): The best action that the MCTS Bot will take. 
        """
        # Every time before make a decision, reset the environment to current environment of the game.
        self.simulator_env.reset(start_player_index=player_index, init_state=state)
        root = TwoPlayersMCTSNode(self.simulator_env)
        # Do the MCTS to find the best action to take.
        mcts = MCTS(root)
        mcts.best_action(self.num_simulation, best_action_type=best_action_type)
        return root.best_action
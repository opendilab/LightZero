import time
from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np
from easydict import EasyDict


class MCTSNode(ABC):

    def __init__(self, env, parent=None):
        """
        Overview:
            This is an abstract base class that outlines the fundamental methods for a Monte Carlo Tree Node.
            Each specific method must be implemented in the subclasses for specific use-cases.
            For more details, you can refer to: https://github.com/int8/monte-carlo-tree-search.
        Arguments:
            - env (:obj:`object`): This represents the game environment of the current node. The properties of this environment object
                encapsulate the state information of the game and retrieving game results. For instance, in a game of tictactoe:
                - env.board: A (3,3) array representing the game board, e.g.,
                    [[0,2,0],
                    [1,1,0],
                    [2,0,0]]
                Here, 0 denotes an unplayed position, 1 represents a position occupied by player 1, and 2 indicates a position taken by player 2.
                - env.players: A list [1,2] representing the two players, player 1 and player 2 respectively.
                - env.start_player_index: An integer (0 or 1) that is reset every episode to determine the starting player.
                - env._current_player: Denotes the player who is to make a move in the current turn, which is alterating in each turn not only in the reset phase.
                - Class Env, such as
                    zoo.board_games.tictactoe.envs.tictactoe_env.TicTacToeEnv,
                    zoo.board_games.gomoku.envs.gomoku_env.GomokuEnv
                - parent (:obj:`MCTSNode`): This parent node of the current node. The parent node is primarily used for backpropagation
                during the Monte Carlo Tree Search. For the root node, this parent returns None as it does not have a parent node.
        """
        self.env = env
        self.parent = parent
        self.children = []
        self.parent_action = []
        self.best_action = -1

    @property
    @abstractmethod
    def legal_actions(self):
        """
        Returns:
            list of zoo.board_games.xxx.envs.xxx_env.XXXEnv.legal_actions
        """
        pass

    @property
    @abstractmethod
    def q(self):
        pass

    @property
    @abstractmethod
    def n(self):
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
            This method checks if the node is fully expanded. A node is considered fully expanded when all of its child nodes have been selected at least once.
            Whenever a new child node is selected for the first time, a corresponding action is removed from the list of legal actions.
            Once the list of legal actions is depleted, it signifies that all child nodes have been selected, thereby indicating that the parent node is fully expanded.
        """
        return len(self.legal_actions) == 0

    def best_child(self, c_param=1.4):
        """
        Overview:
            Find the best child node which has the highest ucb score.
            {UCT}(v_i, v) = \frac{Q(v_i)}{N(v_i)} + c \sqrt{\frac{\log(N(v))}{N(v_i)}}
                - Q(v_i) is the estimated value of the child node v_i.
                - N(v_i) is a counter of how many times the child node v_i has been on the backpropagation path.
                - N(v) is a counter of how many times the parent(current) node v has been on the backpropagation path.
                - c is a parameter which balance exploration and exploitation.
        """
        choices_weights = [(c.q / c.n) + c_param * np.sqrt((2 * np.log(self.n) / c.n)) for c in self.children]
        self.best_action = self.parent_action[np.argmax(choices_weights)]
        return self.children[np.argmax(choices_weights)]

    def rollout_policy(self, legal_actions, last_legal_actions=None, last_action=None):
        """
        Overview:
            This method implements the rollout policy for a node during the Monte Carlo Tree Search.
            The rollout policy is used to determine the action taken during the simulation phase of the MCTS. In this case, the policy is to randomly choose an action from the list of possible actions.
        Arguments:
            - possible_actions: A list of all possible actions that can be taken from the current state.
        Return:
            A randomly chosen action from the list of possible actions.
        """
        if last_legal_actions is not None and last_legal_actions == [
            self.env.board_size ** 2] and last_action is not None and last_action == self.env.board_size ** 2:
            # for Go env, if last_action is not None, and last_action == self.env.board_size ** 2, then pass
            return legal_actions[-1]
        return legal_actions[np.random.randint(len(legal_actions))]


class TwoPlayersMCTSNode(MCTSNode):

    def __init__(self, env, parent=None):
        super().__init__(env, parent)
        self._number_of_visits = 0.
        self._results = defaultdict(int)
        self._legal_actions = None

    @property
    def legal_actions(self):
        if self._legal_actions is None:
            self._legal_actions = self.env.legal_actions
        return self._legal_actions

    @property
    def q(self):
        """
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
        wins, loses = (self._results[1], self._results[-1]) if self.parent.env.current_player == 1 else (
        self._results[-1], self._results[1])

        # Calculate and return the Q-value as the difference between wins and losses.
        return wins - loses

    @property
    def n(self):
        """
        This property represents the number of times the node has been visited during the search.
        """
        return self._number_of_visits

    def expand(self):
        """
         This method expands the current node by creating a new child node.

         It pops an action from the list of legal actions, simulates the action to get the next game state,
         and creates a new child node with that state. The new child node is then added to the list of children nodes.
         """
        action = self.legal_actions.pop()
        next_simulator_env = self.env.simulate_action(action)
        child_node = TwoPlayersMCTSNode(next_simulator_env, parent=self)
        self.children.append(child_node)
        self.parent_action.append(action)
        return child_node

    def is_terminal_node(self):
        """
        This method checks if the current node is a terminal node.

        It uses the game environment's get_done_reward method to check if the game has ended.
        """
        return self.env.get_done_reward()[0]

    def rollout(self):
        """
        This method performs a rollout (simulation) from the current node.

        It repeatedly selects an action based on the rollout policy and simulates the action until the game ends.
        The method then returns the reward of the game's final state.
        """
        # print('simulation begin')
        current_rollout_env = self.env
        # print(current_rollout_env.board)
        step = 0
        last_action = None
        last_legal_actions = None
        while not current_rollout_env.get_done_reward()[0]:
            step += 1
            legal_actions = current_rollout_env.legal_actions
            action = self.rollout_policy(legal_actions, last_legal_actions, last_action)
            # print('step={}'.format(step))
            # print(current_rollout_env.board)
            # print('legal_actions={}'.format(legal_actions))
            # print('action={}'.format(action))
            current_rollout_env = current_rollout_env.simulate_action(action)
            last_action = action
            last_legal_actions = legal_actions
        # print('simulation end')
        return current_rollout_env.get_done_reward()[1]

    def backpropagate(self, result):
        """
        This method performs backpropagation from the current node.

        It increments the number of times the node has been visited and the number of wins for the result.
        If the current node has a parent, the method recursively backpropagates the result to the parent.
        """
        self._number_of_visits += 1.
        self._results[result] += 1.
        if self.parent:
            self.parent.backpropagate(result)


class MCTSSearch(object):

    def __init__(self, node):
        """
        Initialize the Monte Carlo Tree Search object.

        Arguments:
            - node (TwoPlayersMCTSNode): The root node for the MCTS algorithm.
        """
        self.root = node

    def best_action(self, simulations_number=None, total_simulation_seconds=None):
        """
        Conduct Monte Carlo simulations and backpropagation to determine the best action.

        Arguments:
            - simulations_number (int): Number of simulations performed to get the best action.
            - total_simulation_seconds (float): Amount of time the algorithm has to run, specified in seconds.

        Returns:
            The best child node after simulations. The best action can be retrieved from Node.best_action_index.
        """

        if simulations_number is None:
            # If no simulations number is provided, run simulations for the specified time
            assert (total_simulation_seconds is not None)
            end_time = time.time() + total_simulation_seconds
            while True:
                v = self._tree_policy()
                reward = v.rollout()
                v.backpropagate(reward)
                if time.time() > end_time:
                    break
        else:
            # If simulations number is provided, run the specified number of simulations
            for i in range(0, simulations_number):
                # print('****simlulation-{}****'.format(i))
                v = self._tree_policy()
                reward = v.rollout()
                # print('reward={}\n'.format(reward))
                v.backpropagate(reward)
        # to select best child go for exploitation only
        return self.root.best_child(c_param=0.)

    def _tree_policy(self):
        """
          Implement the tree policy for action selection during rollout.

          At each step, if the current node is not fully expanded, it expands.
          If it is fully expanded, it moves to the best child according to the tree policy.
          This continues until a terminal node is reached, which is then returned.

          Returns:
              The terminal node reached by following the tree policy.
          """
        current_node = self.root
        while not current_node.is_terminal_node():
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node


class MCTSBot:

    def __init__(self, ENV, cfg, bot_name, num_simulation=10000):
        """
         Initialize the Monte Carlo Tree Search Bot.

         Arguments:
             - ENV: The environment in which the MCTS bot operates.
             - cfg: Configuration parameters for the bot.
             - bot_name (str): The name of the bot.
             - num_simulation (int): Number of simulations to perform for each move. Default is 10000.
         """
        self.name = bot_name
        self.num_simulation = num_simulation
        self.ENV = ENV
        self.cfg = cfg

    def get_actions(self, state, player_index):
        """
         Get the best action for a given game state.

         A new environment is created for simulations. The environment is reset to the given state.
         Then, MCTS is performed with the specified number of simulations to find the best action.

         Arguments:
             - state: The current game state.
             - player_index (int): The index of the current player.

         Returns:
             The best action determined by MCTS.
         """
        # Create a new environment for simulations
        simulator_env = self.ENV(EasyDict(self.cfg))

        # Reset the environment to the given state
        simulator_env.reset(start_player_index=player_index, init_state=state, katago_policy_init=False)

        # Create the root node for MCTS
        root = TwoPlayersMCTSNode(simulator_env)

        # Perform MCTS
        mcts = MCTSSearch(root)
        mcts.best_action(self.num_simulation)

        # Return the best action
        return root.best_action

import time
from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np
from easydict import EasyDict

class MCTSNode(ABC):
    """
    Overview:
        An abstract base class.
        Define basic methods for a Monte Carlo Tree Node.
        The specific methods are implemented in the subclasses.
        https://github.com/int8/monte-carlo-tree-search
    """

    def __init__(self, env, parent=None):
        """
        Arguments:
            - env(): The game environment of the current node. 
                - The properties of this object reflect the state information. For example, in tictactoe:
                    - env.board  is an arryay of shape(3,3), such as
                        [[0,2,0],  
                        [1,1,0],   
                        [2,0,0]]   
                        where 0 stands for position that has not been played yet, 
                        1 represents a position that has been played by player 1.
                        2 represents a position taken by player 2.
                    - env.players = [1,2], representing two players, player 1 and player 2.
                    - env.start_player_index = 0 or 1. It will be reset every round.
                    - env._current_player = env.players[env.start_player_index], representing which player 
                        should make a move in this round.
                - The methods in this object implement functionalities such as environment transitions and obtaining game results.
            #!!!!!!!为什么回合开始不直接重置current player!!!!!!!!!
            - parent(): The parent node of current node. It is used for backpropagation. For the root node, this parameter is None.
        """
        self.env = env
        self.parent = parent
        self.children = []
        # 这个变量好像没什么用!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
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
            Check whether the node is fully expanded. Whenever a new child node is selected first time, an element 
            is popped out from the list of legal actions. When the list becomes empty, it indicates that all child 
            nodes have been selected. This means that the parent node has been completed expanded.
        """
        return len(self.legal_actions) == 0

    def best_child(self, c_param=1.4):
        '''
        Overview:
            Find the best child node which has the highest ucb score.
            {UCT}(v_i, v) = \frac{Q(v_i)}{N(v_i)} + c \sqrt{\frac{\log(N(v))}{N(v_i)}}
                - Q(v_i) is the estimated value of the child node v_i.
                - N(v_i) is a counter of how many times the child node v_i has been on the backpropagation path.
                - N(v) is a counter of how many times the parent(current) node v has been on the backpropagation path.
                - c is a parameter which balance exploration and exploitation.
        '''
        # Calculate the ucb score for every child node in the list.
        choices_weights = [(c.q / c.n) + c_param * np.sqrt((2 * np.log(self.n) / c.n)) for c in self.children]
        self.best_action = self.parent_action[np.argmax(choices_weights)]
        # Choose the child node which has the highest ucb score.
        return self.children[np.argmax(choices_weights)]

    def rollout_policy(self, possible_actions):
        # randomly choose an action 
        return possible_actions[np.random.randint(len(possible_actions))]

class TwoPlayersMCTSNode(MCTSNode):
    """
    Overview:
        The subclass, which inherits from the abstract base class, implements the specific methods required for a MCTS node.
    """

    def __init__(self, env, parent=None):
        super().__init__(env, parent)
        self._number_of_visits = 0.
        # A default dictionary which sets the value to 0 for undefined keys.
        self._results = defaultdict(int)
        self._legal_actions = None

    # Get all legal actions in current state from the environment object.
    # 为什么不直接在initialize里定义啊？？？？？？？？？？？？？？？？？？？？？？？？？
    @property
    def legal_actions(self):
        if self._legal_actions is None:
            self._legal_actions = self.env.legal_actions
        return self._legal_actions


    @property
    def q(self):
        '''
        Overview:
            The estimated value of Node. 
            self._results[1]  means current_player 1 number of wins.
            self._results[-1] means current_player 2 number of wins.
            For parent nodes with different current players(self.env._current_player), the calculation of q is reversed.
            In this way, the child node with highest q is the node which has the highest net win count for the player of the parent node.
        Example:
            result[1] = 10, result[-1] = 5,
            where result[1] stores the numeber of wins of player 1, result[-1] stores the number of wins of player 2.
            As player of parent node is player 1, then q of current node is q = 10 - 5 = 5
            As player of parent node is player 2, then q of current node is q = 5 - 10 = -5
        '''
        # print(self._results)
        # print('parent.current_player={}'.format(self.parent.env.current_player))
        if self.parent.env.current_player == 1:
            wins = self._results[1]
            loses = self._results[-1]

        if self.parent.env.current_player == 2:
            wins = self._results[-1]
            loses = self._results[1]
        # print("wins={}, loses={}".format(wins, loses))
        return wins - loses

    @property
    def n(self):
        return self._number_of_visits

    def expand(self):
        # Choose an untried action in the list and pop it out. Only untried actions are left in the lsit.
        action = self.legal_actions.pop()
        # simulate_action() returns a new environment object which reset the board and the current player flag.
        next_simulator_env = self.env.simulate_action(action)
        # Create a new node object for the child node.
        child_node = TwoPlayersMCTSNode(next_simulator_env, parent=self)
        self.children.append(child_node)
        self.parent_action.append(action)
        return child_node

    def is_terminal_node(self):
        # get_done_reward() returns a tuple (done, reward)
        # reward = ±1 when player 1 wins/loses the game
        # reward = 0 when it is a tie
        # reward = None when current node is not a teminal node
        # done is the bool flag representing whether the game is over
        return self.env.get_done_reward()[0]

    def rollout(self):
        """
        Overview:
            Traverse the tree starting from the current node until reaching a terminal node.
            Take action according to the rollout policy, such as random policy.
        Returns:
            reward (int): 
                reward = ±1 when player 1  wins/loses the game
                reward = 0 when it is a tie
                reward = None when current node is not a teminal node
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
        self._number_of_visits += 1.
        # result is the index of the self._results list
        # result = ±1 when player 1 wins/loses the game
        self._results[result] += 1.
        if self.parent:
            self.parent.backpropagate(result)


class MCTSSearchNode(object):
    """
    Overview:
        This class implement Monte Carlo Tree Search from the root node, whose environment is the real environment of the game right moment.
        After the tree search and rollout simulation, every child node of the root node has a ucb value.
        Then the decision for the root node is to choose the child node with highest ucb. 
    """

    def __init__(self, node):
        """
        Overview:
            Monte Carlo Tree Search Node.
        Arguments:
            - node (:obj:`TwoPlayersMCTSNode`):
        """
        self.root = node

    def best_action(self, simulations_number=None, total_simulation_seconds=None):
        """
        Overview:
            By constantly simulating and backpropagating, get the best action and the best children node.
        Arguments:
            - simulations_number (:obj:`int`): number of simulations performed to get the best action
            - total_simulation_seconds (:obj:`float`): Amount of time the algorithm has to run. Specified in seconds
        Returns:
            Returns the best children node, and can get action from Node.best_action_index.
        """

        # The search cost is determined by either the maximum number of simulations or the longest simulation time.
        if simulations_number is None:
            assert (total_simulation_seconds is not None)
            end_time = time.time() + total_simulation_seconds
            while True:
                # get the leaf node.
                v = self._tree_policy()
                # rollout from the leaf node.
                reward = v.rollout()
                # backpropagate from the leaf node to the root node.
                v.backpropagate(reward)
                if time.time() > end_time:
                    break
        else:
            for i in range(0, simulations_number):
                # print('****simlulation-{}****'.format(i))
                # get the leaf node.
                v = self._tree_policy()
                # rollout from the leaf node.
                reward = v.rollout()
                # print('reward={}\n'.format(reward))
                # backpropagate from the leaf node to the root node.
                v.backpropagate(reward)
        # to select best child go for exploitation only
        # 这里和算法理论里选择最高访问次数似乎不太一样？？？？？？？？？？？？？？？？？？
        return self.root.best_child(c_param=0.)

    #
    def _tree_policy(self):
        """
        Overview:
            This function implement the tree search from the root node to the leaf node, which is visited for the first time or is the terminal node.
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
 
    def __init__(self, ENV, cfg, bot_name, num_simulation=10000):
        self.name = bot_name
        self.num_simulation = num_simulation
        self.ENV = ENV
        self.cfg = cfg

    def get_actions(self, state, player_index):
        # 为什么把类作为属性，每次重新生成一个，而不是把环境实例直接作为属性？
        # an instance of the ENV class.
        simulator_env = self.ENV(EasyDict(self.cfg))
        # Every time before make a decision, reset the environment to current environment of the game.
        simulator_env.reset(start_player_index=player_index, init_state=state)
        legal_actions = simulator_env.legal_actions
        root = TwoPlayersMCTSNode(simulator_env)
        # Do the MCTS to find the best action to take.
        mcts = MCTSSearchNode(root)
        mcts.best_action(self.num_simulation)
        return root.best_action
    

# 这样有一个浪费的点，在于每次重新进行搜索，之前的上层节点搜索记录的数据无法复用

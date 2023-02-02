import sys
import time
from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np
from easydict import EasyDict

sys.path.append('/YOUR/PATH/LightZero')


class MCTSNode(ABC):
    def __init__(self, env, parent=None):
        """
        Overview:
            Monte Carlo Tree Search Base Node
            https://github.com/int8/monte-carlo-tree-search
        Arguments:
            env: Class Env, such as 
                 zoo.board_games.tictactoe.envs.tictactoe_env.TicTacToeEnv,
                 zoo.board_games.gomoku.envs.gomoku_env.GomokuEnv
            parent: TwoPlayersMCTSNode / MCTSNode
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
        return len(self.legal_actions) == 0

    def best_child(self, c_param=1.4):
        '''
        Overview:
                 - computer ucb score.
                    ucb = (q / n) + c_param * np.sqrt((2 * np.log(visited_num) / n))
                    - q: The estimated value of Node. 
                    - n: The simulation num of Node.
                    - visited_num: The visited num of Node
                    - c_param: constant num=1.4.
                 - Select the node with the highest ucb score.
        '''
        choices_weights = [
            (c.q / c.n) + c_param * np.sqrt((2 * np.log(self.n) / c.n))
            for c in self.children
        ]
        self.best_action = self.parent_action[np.argmax(choices_weights)]
        return self.children[np.argmax(choices_weights)]

    def rollout_policy(self, possible_actions):
        return possible_actions[np.random.randint(len(possible_actions))]


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
        '''
        Overview:
                  The estimated value of Node. 
                  self._results[1]  means current_player 1 number of wins.
                  self._results[-1] means current_player 2 number of wins.
        Example:
                result[1] = 10, result[-1] = 5,
                As current_player_1, q = 10 - 5 = 5
                As current_player_2, q = 5 - 10 = -5
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
        action = self.legal_actions.pop()
        next_simulator_env = self.env.simulate_action(action)
        child_node = TwoPlayersMCTSNode(
            next_simulator_env, parent=self
        )
        self.children.append(child_node)
        self.parent_action.append(action)
        return child_node

    def is_terminal_node(self):
        return self.env.is_game_over()[0]

    def rollout(self):
        # print('simulation begin')
        current_rollout_env = self.env
        # print(current_rollout_env.board)
        while not current_rollout_env.is_game_over()[0]:
            possible_actions = current_rollout_env.legal_actions
            action = self.rollout_policy(possible_actions)
            current_rollout_env = current_rollout_env.simulate_action(action)
            # print('\n')
            # print(current_rollout_env.board)
        # print('simulation end \n')
        return current_rollout_env.is_game_over()[1]

    def backpropagate(self, result):
        self._number_of_visits += 1.
        self._results[result] += 1.
        if self.parent:
            self.parent.backpropagate(result)

class MCTSSearchNode(object):

    def __init__(self, node):
        """
        Overview:
            Monte Carlo Tree Search Node
        Arguments:
            node : TwoPlayersMCTSNode
        """
        self.root = node

    def best_action(self, simulations_number=None, total_simulation_seconds=None):
        """
        Overview:
            By constantly simulating and backpropagating, get the best action and the best children node.
        Arguments:
            simulations_number : int
                number of simulations performed to get the best action
            total_simulation_seconds : float
                Amount of time the algorithm has to run. Specified in seconds
        Returns:
            Returns the best children node, and can get action from Node.best_action_index.
        -------
        """

        if simulations_number is None:
            assert (total_simulation_seconds is not None)
            end_time = time.time() + total_simulation_seconds
            while True:
                v = self._tree_policy()
                reward = v.rollout()
                v.backpropagate(reward)
                if time.time() > end_time:
                    break
        else:
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
        Overview:
            The policy used to select action to rollout a episode.
        """
        current_node = self.root
        while not current_node.is_terminal_node():
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node

##########################
###### MINI-MAX A-B ######
##########################

class AlphaBeta:
    # print utility value of root node (assuming it is max)
    # print names of all nodes visited during search
    # def __init__(self, game_tree):
    #     self.game_tree = game_tree  # GameTree
    #     self.root = game_tree.root  # GameNode
    #     return
    def __init__(self, game_tree_root):
        self.root = game_tree_root  # GameNode
        return

    def alpha_beta_search(self):
        node = self.root
        infinity = float('inf')
        best_val = -infinity
        beta = infinity

        successors = self.getSuccessors(node)
        best_state = None
        for state in successors:
            value = self.min_value(state, best_val, beta)
            if value > best_val:
                best_val = value
                best_state = state
        print( "AlphaBeta:  Utility Value of Root Node: = " + str(best_val))
        # print( "AlphaBeta:  Best State is: " + best_state.Name)
        return best_state

    def max_value(self, node, alpha, beta):
        # print( "AlphaBeta-->MAX: Visited Node :: " + node.Name)
        if self.isTerminal(node):
            return self.getUtility(node)
        infinity = float('inf')
        value = -infinity

        successors = self.getSuccessors(node)
        for state in successors:
            value = max(value, self.min_value(state, alpha, beta))
            if value >= beta:
                return value
            alpha = max(alpha, value)
        return value

    def min_value(self, node, alpha, beta):
        # print( "AlphaBeta-->MIN: Visited Node :: " + node.Name)
        if self.isTerminal(node):
            return self.getUtility(node)
        infinity = float('inf')
        value = infinity

        successors = self.getSuccessors(node)
        for state in successors:
            value = min(value, self.max_value(state, alpha, beta))
            if value <= alpha:
                return value
            beta = min(beta, value)

        return value
    #                     #
    #   UTILITY METHODS   #
    #                     #

    # successor states in a game tree are the child nodes...
    def getSuccessors(self, node):
        assert node is not None
        return node.children

    # return true if the node has NO children (successor states)
    # return false if the node has children (successor states)
    def isTerminal(self, node):
        # assert node is not None
        # return len(node.children) == 0

        return node.is_terminal_node()

    def getUtility(self, node):
        assert node is not None
        return node.value

class AlphaBetaPrunningBot:
    def __init__(self, ENV, cfg, bot_name, num_simulation=10000):
        self.name = bot_name
        self.num_simulation = num_simulation
        self.ENV = ENV
        self.cfg = cfg

    def get_actions(self, state, player_index):
        simulator_env = self.ENV(EasyDict(self.cfg))
        simulator_env.reset(start_player_index=player_index, init_state=state)
        legal_actions = simulator_env.legal_actions
        root = TwoPlayersMCTSNode(simulator_env)

        mcts = MCTSSearchNode(root)
        mcts.best_action(self.num_simulation)

        bot = AlphaBeta(root)
        best_state = bot.alpha_beta_search()
        print(best_state)

        return root.best_action
# https://tonypoer.io/2016/10/28/implementing-minimax-and-alpha-beta-pruning-using-python
from zoo.board_games.tictactoe.envs.tictactoe_env import TicTacToeEnv

cfg = dict(
    prob_random_agent=0,
    prob_expert_agent=0,
    battle_mode='two_player_mode',
    agent_vs_human=False,

)
env = TicTacToeEnv(EasyDict(cfg))
env.reset()
state = env.board
player_0 = AlphaBetaPrunningBot(TicTacToeEnv, cfg, 'a', 1000)  # player_index = 0, player = 1
player_1 = AlphaBetaPrunningBot(TicTacToeEnv, cfg, 'b', 1)  # player_index = 1, player = 2

player_index = 0  # A fist
print('#' * 15)
print(state)
print('#' * 15)
print('\n')
while not env.is_game_over()[0]:
    if player_index == 0:
        action = player_0.get_actions(state, player_index=player_index)
        player_index = 1
    else:
        print('-' * 40)
        action = player_1.get_actions(state, player_index=player_index)
        player_index = 0
        print('-' * 40)
    env.step(action)
    state = env.board
    print('#' * 15)
    print(state)
    print('#' * 15)
assert env.have_winner()[1] == 1
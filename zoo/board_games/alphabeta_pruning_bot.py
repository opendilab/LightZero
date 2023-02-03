import sys
import time
from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np
from easydict import EasyDict
import copy

sys.path.append('/YOUR/PATH/LightZero')


class BaseNode(ABC):
    def __init__(self, env, parent=None):
        """
        Overview:
            Monte Carlo Tree Search Base Node
            https://github.com/int8/monte-carlo-tree-search
        Arguments:
            env: Class Env, such as 
                 zoo.board_games.tictactoe.envs.tictactoe_env.TicTacToeEnv,
                 zoo.board_games.gomoku.envs.gomoku_env.GomokuEnv
            parent: Node / BaseNode
        """
        self.env = env
        self.parent = parent
        self.children = []
        self.parent_action = []
        self.best_action = -1

    # @property
    # @abstractmethod
    # def legal_actions(self):
    #     """
    #     Returns:
    #         list of zoo.board_games.xxx.envs.xxx_env.XXXEnv.legal_actions
    #     """
    #     pass

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
        return len(self.env.legal_actions) == 0

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


# class Node(BaseNode):
class Node():

    def __init__(self, env, legal_actions,  start_player_index=0, parent=None, recursive=True, prev_action=None):
        # super().__init__(env, parent)
        super().__init__()
        self.env = env
        self._number_of_visits = 0.
        self._results = defaultdict(int)
        self._legal_actions = legal_actions
        self.children = []
        self.parent = parent
        self.prev_action = prev_action
        self.start_player_index = start_player_index

        if recursive is True:
            is_terminal_node = self.is_terminal_node()
            if is_terminal_node is True:
                return
            if not self.is_terminal_node():
                # while not self.is_fully_expanded():
                #     action = self.env.legal_actions.pop()
                # legal_actions = self.env.legal_actions
                while len(self._legal_actions) > 0:
                    # self.expand()
                    action = self._legal_actions.pop()
                    next_simulator_env = self.env.simulate_action(action)
                    # print(next_simulator_env.board)
                    child_node = Node(
                        next_simulator_env, copy.deepcopy(self._legal_actions), start_player_index=next_simulator_env.start_player_index, parent=self, recursive=True, prev_action=action
                    )
                    # print('add one edge')
                    self.children.append(child_node)

    def is_terminal_node(self):
        return self.env.is_game_over()[0]

    @property
    def value(self):
        if self.start_player_index == 0:
            return self.env.is_game_over()[1]
        elif self.start_player_index == 1:
            return -self.env.is_game_over()[1]

##########################
###### MINI-MAX A-B ######
##########################

class AlphaBeta:
    # print utility value of root node (assuming it is max)
    # print names of all nodes visited during search
    # def __init__(self, game_tree):
    #     self.game_tree = game_tree  # GameTree
    #     self.root = game_tree.root  # GameNode
    def __init__(self, game_tree_root):
        self.root = game_tree_root  # GameNode

    def alpha_beta_search(self):
        node = self.root
        infinity = float('inf')
        best_val = -infinity
        beta = infinity

        children_states = self.get_children(node)
        best_state = None
        for state in children_states:
            value = self.min_value(state, best_val, beta)
            if value > best_val:
                best_val = value
                best_state = state
        # print("AlphaBeta:  Utility Value of Root Node: = " + str(best_val))
        # print( "AlphaBeta:  Best State is: " + best_state.Name)
        return best_state

    def max_value(self, node, alpha, beta):
        # print( "AlphaBeta-->MAX: Visited Node :: " + node.Name)
        if self.is_terminal(node):
            return self.get_utility(node)
        infinity = float('inf')
        value = -infinity

        children_states = self.get_children(node)
        for state in children_states:
            value = max(value, self.min_value(state, alpha, beta))
            if value >= beta:
                return value
            alpha = max(alpha, value)
        return value

    def min_value(self, node, alpha, beta):
        # print( "AlphaBeta-->MIN: Visited Node :: " + node.Name)
        if self.is_terminal(node):
            return self.get_utility(node)
        infinity = float('inf')
        value = infinity

        children_states = self.get_children(node)
        for state in children_states:
            value = min(value, self.max_value(state, alpha, beta))
            if value <= alpha:
                return value
            beta = min(beta, value)

        return value

    #   UTILITY METHODS   #
    # successor states in a game tree are the child nodes...
    def get_children(self, node):
        assert node is not None
        return node.children

    # return true if the node has NO children (successor states)
    # return false if the node has children (successor states)
    def is_terminal(self, node):
        # assert node is not None
        # return len(node.children) == 0

        return node.is_terminal_node()

    def get_utility(self, node):
        assert node is not None
        return node.value


class AlphaBetaPruningBot:
    def __init__(self, ENV, cfg, bot_name, num_simulation=10000):
        self.name = bot_name
        self.num_simulation = num_simulation
        self.ENV = ENV
        self.cfg = cfg

    def get_actions(self, state, player_index):
        simulator_env = self.ENV(EasyDict(self.cfg))
        simulator_env.reset(start_player_index=player_index, init_state=state)
        legal_actions = simulator_env.legal_actions
        root = Node(simulator_env, legal_actions, start_player_index=player_index)

        # mcts = MCTSSearchNode(root)
        # mcts.best_action(self.num_simulation)
        # mcts_best_action = root.best_action

        bot = AlphaBeta(root)
        best_state = bot.alpha_beta_search()
        # print(best_state)
        # best_action_index = (best_state.env.board.nonzero()[0][0], best_state.env.board.nonzero()[1][0])
        # best_action = best_state.env.board.nonzero()[0][0] * 3 + best_state.env.board.nonzero()[1][0]
        return best_state.prev_action

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
player_0 = AlphaBetaPruningBot(TicTacToeEnv, cfg, 'a', 1000)  # player_index = 0, player = 1
player_1 = AlphaBetaPruningBot(TicTacToeEnv, cfg, 'b', 1)  # player_index = 1, player = 2

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
        # print('-' * 40)
    env.step(action)
    state = env.board
    # print('#' * 15)
    print(state)
    print('#' * 15)

assert env.have_winner()[1] == 1

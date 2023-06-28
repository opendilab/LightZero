import time
from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np
from easydict import EasyDict


class MCTSNode(ABC):

    def __init__(self, env, parent=None):
        """
        Overview:
            Monte Carlo Tree Search Base Node
            https://github.com/int8/monte-carlo-tree-search
        Arguments:
            - env: Class Env, such as
                 zoo.board_games.tictactoe.envs.tictactoe_env.TicTacToeEnv,
                 zoo.board_games.gomoku.envs.gomoku_env.GomokuEnv
            - parent: TwoPlayersMCTSNode / MCTSNode
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
        """
        Overview:
                 - computer ucb score.
                    ucb = (q / n) + c_param * np.sqrt((2 * np.log(visited_num) / n))
                    - q: The estimated value of Node.
                    - n: The simulation num of Node.
                    - visited_num: The visited num of Node
                    - c_param: constant num=1.4.
                 - Select the node with the highest ucb score.
        """
        choices_weights = [(c.q / c.n) + c_param * np.sqrt((2 * np.log(self.n) / c.n)) for c in self.children]
        self.best_action = self.parent_action[np.argmax(choices_weights)]
        return self.children[np.argmax(choices_weights)]

    def rollout_policy(self, legal_actions, last_legal_actions=None, last_action=None):
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
        Overview:
                  The estimated value of Node.
                  self._results[1]  means current_player 1 number of wins.
                  self._results[-1] means current_player 2 number of wins.
        Example:
                result[1] = 10, result[-1] = 5,
                As current_player_1, q = 10 - 5 = 5
                As current_player_2, q = 5 - 10 = -5
        """
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
        child_node = TwoPlayersMCTSNode(next_simulator_env, parent=self)
        self.children.append(child_node)
        self.parent_action.append(action)
        return child_node

    def is_terminal_node(self):
        return self.env.get_done_reward()[0]

    def rollout(self):
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
        self._number_of_visits += 1.
        self._results[result] += 1.
        if self.parent:
            self.parent.backpropagate(result)


class MCTSSearchNode(object):

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


class MCTSBot:

    def __init__(self, ENV, cfg, bot_name, num_simulation=10000):
        self.name = bot_name
        self.num_simulation = num_simulation
        self.ENV = ENV
        self.cfg = cfg

    def get_actions(self, state, player_index):
        simulator_env = self.ENV(EasyDict(self.cfg))
        simulator_env.reset(start_player_index=player_index, init_state=state)
        # legal_actions = simulator_env.legal_actions
        root = TwoPlayersMCTSNode(simulator_env)
        mcts = MCTSSearchNode(root)
        mcts.best_action(self.num_simulation)
        return root.best_action

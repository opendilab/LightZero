"""
reference: https://mathspp.com/blog/minimax-algorithm-and-alpha-beta-pruning
"""

import numpy as np
from easydict import EasyDict


class Node():
    """
    Overview:
        Alpha-Beta-Pruning Search Node.
        https://mathspp.com/blog/minimax-algorithm-and-alpha-beta-pruning
    Arguments:
        env: Class Env, such as
             zoo.board_games.tictactoe.envs.tictactoe_env.TicTacToeEnv,
             zoo.board_games.gomoku.envs.gomoku_env.GomokuEnv
    """

    def __init__(self, env, start_player_index=0, parent=None, prev_action=None):
        super().__init__()
        self.env = env
        self.legal_actions = self.env.legal_actions
        self.children = []
        self.parent = parent
        self.prev_action = prev_action
        self.start_player_index = start_player_index
        self.tree_expanded = False

    def __str__(self):
        return f"Tree({', '.join(str(sub) for sub in self.children)})"

    def expand(self):
        if self.start_player_index == 0:
            next_start_player_index = 1
        else:
            next_start_player_index = 0
        if self.is_terminal_node is False:
            while len(self.legal_actions) > 0:
                action = self.legal_actions.pop(0)
                next_simulator_env = self.env.simulate_action(action)
                # print(next_simulator_env.board)
                child_node = Node(
                    next_simulator_env, start_player_index=next_start_player_index, parent=self, prev_action=action
                )
                # print('add one edge')
                self.children.append(child_node)
            self.tree_expanded = True

    @property
    def expanded(self):
        # return len(self.children) > 0
        return self.tree_expanded

    def is_fully_expanded(self):
        return len(self.children) == len(self.env.legal_actions)

    @property
    def is_terminal_node(self):
        return self.env.is_game_over()[0]

    @property
    def value(self):
        """
        def is_game_over(self):
            Overview:
                To judge game whether over, and get reward
            Returns:
                [game_over, reward]
                if winner = 1  reward = 1
                if winner = 2  reward = -1
                if winner = -1 reward = 0
        """
        return self.env.is_game_over()[1]

    @property
    def state(self):
        return self.env.board


def pruning(tree, maximising_player, alpha=float("-inf"), beta=float("+inf"), first_level=False):
    if tree.is_terminal_node is True:
        return tree.value

    # print(tree)
    if tree.expanded is False:
        tree.expand()
        print('expand one node!')
    # else:
    #     print('current node has already expanded!')
    #     print(tree.children)

    # if (tree.state == np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])).all():
    #     print('p1')
    # if (tree.state == np.array([[0, 0, 1], [2, 1, 2], [1, 2, 1]])).all():
    #     print('p2')

    val, func = (float("-inf"), max) if maximising_player else (float("+inf"), min)
    for subtree in tree.children:
        val = func(pruning(subtree, not maximising_player, alpha, beta, first_level=False), val)
        if maximising_player:
            if val > alpha:
                best_subtree = subtree
            alpha = max(alpha, val)
        else:
            if val < beta:
                best_subtree = subtree
            beta = min(beta, val)

        if (maximising_player and val >= beta) or (not maximising_player and val <= alpha):
            break

    if first_level is True:
        return val, best_subtree
    else:
        return val


class AlphaBetaPruningBot:

    def __init__(self, ENV, cfg, bot_name):
        self.name = bot_name
        self.ENV = ENV
        self.cfg = cfg

    def get_best_action(self, state, player_index):
        try:
            simulator_env = self.ENV(EasyDict(self.cfg))
        except:
            simulator_env = self.ENV
        simulator_env.reset(start_player_index=player_index, init_state=state)
        root = Node(simulator_env, start_player_index=player_index)
        if player_index == 0:
            val, best_subtree = pruning(root, True, first_level=True)
        else:
            val, best_subtree = pruning(root, False, first_level=True)

        # print('val, best_subtree:', val, best_subtree)
        print(f'alpha-beta searched best_action: {best_subtree.prev_action}, its val: {val}')

        return best_subtree.prev_action


if __name__ == "__main__":
    """TicTacToe"""
    # from zoo.board_games.tictactoe.envs.tictactoe_env import TicTacToeEnv
    # cfg = dict(
    #     prob_random_agent=0,
    #     prob_expert_agent=0,
    #     battle_mode='self_play_mode',
    #     agent_vs_human=False,
    # )
    # env = TicTacToeEnv(EasyDict(cfg))
    # player_0 = AlphaBetaPruningBot(TicTacToeEnv, cfg, 'a')  # player_index = 0, player = 1
    # player_1 = AlphaBetaPruningBot(TicTacToeEnv, cfg, 'b')  # player_index = 1, player = 2
    # player_index = 0  # A fist
    # init_state = [[1, 0, 1],
    #               [0, 0, 2],
    #               [2, 0, 1]]
    # env.reset(player_index, init_state)
    #
    # state = env.board
    # print('-' * 15)
    # print(state)
    #
    # while not env.is_game_over()[0]:
    #     if player_index == 0:
    #         action = player_0.get_best_action(state, player_index=player_index)
    #         player_index = 1
    #     else:
    #         print('-' * 40)
    #         action = player_1.get_best_action(state, player_index=player_index)
    #         player_index = 0
    #     env.step(action)
    #     state = env.board
    #     print('-' * 15)
    #     print(state)
    #     row, col = env.action_to_coord(action)
    #
    # assert env.have_winner()[1] == 1
    # assert (row == 0, col == 1) or (row == 1, col == 1)
    # assert env.have_winner()[0] is True, env.have_winner()[1] == 1

    """Gomoku"""
    from zoo.board_games.gomoku.envs.gomoku_env import GomokuEnv
    cfg = dict(
        board_size=5,
        prob_random_agent=0,
        prob_expert_agent=0,
        battle_mode='self_play_mode',
        channel_last=True,
        agent_vs_human=False,
        expert_action_type='alpha_beta_pruning',  # {'v0', 'alpha_beta_pruning'}
    )
    env = GomokuEnv(EasyDict(cfg))
    player_0 = AlphaBetaPruningBot(GomokuEnv, cfg, 'player 1')  # player_index = 0, player = 1
    player_1 = AlphaBetaPruningBot(GomokuEnv, cfg, 'player 2')  # player_index = 1, player = 2

    ### test from the init empty board ###
    # player_index = 0  # player 1 fist
    # env.reset()

    ### test from the init specified board ###
    player_index = 1  # player 2 fist
    init_state = [[1, 1, 1, 1, 0],
                  [1, 0, 0, 0, 2],
                  [0, 0, 2, 0, 2],
                  [0, 2, 0, 0, 2],
                  [2, 1, 1, 0, 0], ]
    # init_state = [[1, 1, 1, 1, 2],
    #               [1, 1, 2, 1, 2],
    #               [2, 1, 2, 2, 2],
    #               [0, 0, 0, 2, 2],
    #               [2, 1, 1, 1, 0], ]
    env.reset(player_index, init_state)

    state = env.board
    print('-' * 15)
    print(state)

    while not env.is_game_over()[0]:
        if player_index == 0:
            action = player_0.get_best_action(state, player_index=player_index)
            player_index = 1
        else:
            print('-' * 40)
            action = player_1.get_best_action(state, player_index=player_index)
            player_index = 0
        env.step(action)
        state = env.board
        print('-' * 15)
        print(state)

    assert env.have_winner()[0] is False, env.have_winner()[1] == -1
    # assert env.have_winner()[0] is True, env.have_winner()[1] == 2


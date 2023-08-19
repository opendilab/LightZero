from easydict import EasyDict
import copy


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

    def __init__(self, board, legal_actions, start_player_index=0, parent=None, prev_action=None, env=None):
        super().__init__()
        self.env = env
        self.board = board
        self.legal_actions = legal_actions
        self.children = []
        self.parent = parent
        self.prev_action = prev_action
        self.start_player_index = start_player_index
        self.tree_expanded = False

    def __str__(self):
        return f"Tree({', '.join(str(child) for child in self.children)})"

    def expand(self):
        if self.start_player_index == 0:
            next_start_player_index = 1
        else:
            next_start_player_index = 0
        if self.is_terminal_node is False:
            # Ensure self.legal_actions is valid before the loop
            # self.legal_actions = self.env.get_legal_actions(self.board, self.start_player_index)
            while len(self.legal_actions) > 0:
                action = self.legal_actions.pop(0)
                board, legal_actions = self.env.simulate_action_v2(self.board, self.start_player_index, action)
                child_node = Node(
                    board,
                    legal_actions,
                    start_player_index=next_start_player_index,
                    parent=self,
                    prev_action=action,
                    env=self.env
                )
                # print('add one edge')
                self.children.append(child_node)
            self.tree_expanded = True

    @property
    def expanded(self):
        # return len(self.children) > 0
        return self.tree_expanded

    def is_fully_expanded(self):
        return len(self.children) == len(self.legal_actions)

    @property
    def is_terminal_node(self):
        self.env.reset_v2(self.start_player_index, init_state=self.board)  # index
        return self.env.get_done_reward()[0]

    @property
    def value(self):
        """
        def get_done_reward(self):
            Overview:
                To judge game whether over, and get reward
            Returns:
                [game_over, reward]
                if winner = 1  reward = 1
                if winner = 2  reward = -1
                if winner = -1 reward = 0
        """
        self.env.reset_v2(self.start_player_index, init_state=self.board)  # index
        return self.env.get_done_reward()[1]

    @property
    def estimated_value(self):
        return 0

    @property
    def state(self):
        return self.board


def pruning(tree, maximising_player, alpha=float("-inf"), beta=float("+inf"), depth=999, first_level=True):
    if tree.is_terminal_node is True:
        return tree.value
    # TODO(pu): use a limited search depth
    if depth == 0:
        return tree.estimated_value

    # print(ctree)
    if tree.expanded is False:
        tree.expand()
        # print('expand one node!')

    # for debug
    # if (ctree.state == np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])).all():
    #     print('p1')
    # if (ctree.state == np.array([[0, 0, 1], [2, 1, 2], [1, 2, 1]])).all():
    #     print('p2')

    val = float("-inf") if maximising_player else float("+inf")
    for subtree in tree.children:
        sub_val = pruning(subtree, not maximising_player, alpha, beta, depth - 1, first_level=False)
        if maximising_player:
            val = max(sub_val, val)
            if val > alpha:
                best_subtree = subtree
                alpha = val
        else:
            val = min(sub_val, val)
            if val < beta:
                best_subtree = subtree
                beta = val
        if beta <= alpha:
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

    def get_best_action(self, board, player_index, depth=999):
        try:
            simulator_env = copy.deepcopy(self.ENV(EasyDict(self.cfg)))
        except:
            simulator_env = copy.deepcopy(self.ENV)
        simulator_env.reset(start_player_index=player_index, init_state=board)
        root = Node(board, simulator_env.legal_actions, start_player_index=player_index, env=simulator_env)
        if player_index == 0:
            val, best_subtree = pruning(root, True, depth=depth, first_level=True)
        else:
            val, best_subtree = pruning(root, False, depth=depth, first_level=True)

        # print(f'player_index: {player_index}, alpha-beta searched best_action: {best_subtree.prev_action}, its val: {val}')

        return best_subtree.prev_action


if __name__ == "__main__":
    import time
    ##### TicTacToe #####
    from zoo.board_games.tictactoe.envs.tictactoe_env import TicTacToeEnv
    cfg = dict(
        prob_random_agent=0,
        prob_expert_agent=0,
        battle_mode='self_play_mode',
        agent_vs_human=False,
        bot_action_type='alpha_beta_pruning',  # {'v0', 'alpha_beta_pruning'}
        channel_last=True,
        scale=True,
    )
    env = TicTacToeEnv(EasyDict(cfg))
    player_0 = AlphaBetaPruningBot(TicTacToeEnv, cfg, 'player 1')  # player_index = 0, player = 1
    player_1 = AlphaBetaPruningBot(TicTacToeEnv, cfg, 'player 2')  # player_index = 1, player = 2

    ### test from the init empty board ###
    player_index = 0  # player 1 fist
    env.reset()

    ### test from the init specified board ###
    # player_index = 0  # player 1 fist
    # init_state = [[1, 0, 1],
    #               [0, 0, 2],
    #               [2, 0, 1]]
    # env.reset(player_index, init_state)

    state = env.board
    print('-' * 15)
    print(state)

    while not env.get_done_reward()[0]:
        if player_index == 0:
            start = time.time()
            action = player_0.get_best_action(state, player_index=player_index)
            print('player 1 action time: ', time.time() - start)
            player_index = 1
        else:
            start = time.time()
            action = player_1.get_best_action(state, player_index=player_index)
            print('player 2 action time: ', time.time() - start)
            player_index = 0
        env.step(action)
        state = env.board
        print('-' * 15)
        print(state)
        row, col = env.action_to_coord(action)

    ### test from the init empty board ###
    assert env.get_done_winner()[0] is False, env.get_done_winner()[1] == -1

    ### test from the init specified board ###
    # assert (row == 0, col == 1) or (row == 1, col == 1)
    # assert env.get_done_winner()[0] is True, env.get_done_winner()[1] == 1
    """

    ##### Gomoku #####
    from zoo.board_games.gomoku.envs.gomoku_env import GomokuEnv
    cfg = dict(
        board_size=5,
        prob_random_agent=0,
        prob_expert_agent=0,
        battle_mode='self_play_mode',
        scale=True,
        channel_last=True,
        agent_vs_human=False,
        bot_action_type='alpha_beta_pruning',  # {'v0', 'alpha_beta_pruning'}
        prob_random_action_in_bot=0.,
        check_action_to_connect4_in_bot_v0=False,
    )
    env = GomokuEnv(EasyDict(cfg))
    player_0 = AlphaBetaPruningBot(GomokuEnv, cfg, 'player 1')  # player_index = 0, player = 1
    player_1 = AlphaBetaPruningBot(GomokuEnv, cfg, 'player 2')  # player_index = 1, player = 2

    ### test from the init empty board ###
    player_index = 0  # player 1 fist
    env.reset()

    ### test from the init specified board ###
    # player_index = 1  # player 2 fist
    # init_state = [[1, 1, 1, 1, 0],
    #               [1, 0, 0, 0, 2],
    #               [0, 0, 2, 0, 2],
    #               [0, 2, 0, 0, 2],
    #               [2, 1, 1, 0, 0], ]
    # # init_state = [[1, 1, 1, 1, 2],
    # #               [1, 1, 2, 1, 2],
    # #               [2, 1, 2, 2, 2],
    # #               [0, 0, 0, 2, 2],
    # #               [2, 1, 1, 1, 0], ]
    # env.reset(player_index, init_state)

    state = env.board
    print('-' * 15)
    print(state)

    while not env.get_done_reward()[0]:
        if player_index == 0:
            start = time.time()
            action = player_0.get_best_action(state, player_index=player_index)
            print('player 1 action time: ', time.time() - start)
            player_index = 1
        else:
            start = time.time()
            action = player_1.get_best_action(state, player_index=player_index)
            print('player 2 action time: ', time.time() - start)
            player_index = 0
        env.step(action)
        state = env.board
        print('-' * 15)
        print(state)

    assert env.get_done_winner()[0] is False, env.get_done_winner()[1] == -1
    # assert env.get_done_winner()[0] is True, env.get_done_winner()[1] == 2
    """


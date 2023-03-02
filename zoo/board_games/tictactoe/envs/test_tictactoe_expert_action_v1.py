import numpy as np
from easydict import EasyDict
import pytest

from zoo.board_games.tictactoe.envs.tictactoe_env import TicTacToeEnv


@pytest.mark.envtest
class TestExpertAction:

    def test_bot_action(self):
        cfg = EasyDict(
            channel_last=True,
            scale=False,
            prob_random_agent=0,
            prob_expert_agent=0,
            battle_mode='play_with_bot_mode',
            agent_vs_human=False,
            bot_action_type='v1',  # {'v0', 'v1', 'alpha_beta_pruning'}
        )
        env = TicTacToeEnv(cfg)
        env.reset()
        print('init board state: ')
        env.render()
        # TODO(pu): How to fully test all cases
        # case 1
        env.board = np.array([[1, 2, 1], [1, 2, 0], [0, 0, 2]])
        env.current_player = 1
        assert 6 == env.bot_action()
        # case 2
        env.board = np.array([[1, 2, 1], [2, 2, 0], [1, 0, 0]])
        env.current_player = 1
        assert env.bot_action() in [5, 7]
        # case 3
        env.board = np.array([[1, 2, 1], [1, 2, 2], [0, 0, 1]])
        env.current_player = 2
        assert 7 == env.bot_action()
        # case 4
        env.board = np.array([[1, 2, 1], [1, 0, 2], [0, 0, 0]])
        env.current_player = 2
        assert 6 == env.bot_action()

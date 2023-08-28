import copy
from collections import namedtuple
from typing import List, Dict, Any, Tuple, Union

import numpy as np
import torch
from ding.policy.base_policy import Policy
from ding.utils import POLICY_REGISTRY


def legal_actions(board):
    legal_actions = []
    for i in range(3):
        for j in range(3):
            if board[i][j] == 0:
                legal_actions.append(i * 3 + j)
    return legal_actions


@POLICY_REGISTRY.register('tictactoe_bot_policy_v0')
class TictactoeBotPolicyV0(Policy):
    """
    Overview:
        Hard coded expert agent for tictactoe env.
    """
    config = dict(
        # (str) RL policy register name (refer to function "POLICY_REGISTRY").
        type='tictactoe_bot_policy_v0',
        cuda=False,
        on_policy=True,
        learn=dict(
            multi_gpu=False,
        ),
        collect=dict(
            unroll_len=1,
        ),
        eval=dict(),
    )

    def rule_bot_v0(self, obs):

        """
        Overview:
            Hard coded expert agent for tictactoe env.
            First random sample an action from legal_actions, then take the action that will lead a connect3 of current player's pieces.
        Returns:
            - action (:obj:`int`): the expert action to take in the current game state.
        """
        # To easily calculate expert action, we convert the chessboard notation:
        # from player 1:  1, player 2: 2
        # to   player 1: -1, player 2: 1
        # TODO: more elegant implementation
        board = obs['board']
        current_player_to_compute_bot_action = -1 if obs['to_play'] == 1 else 1
        board = copy.deepcopy(board)
        for i in range(board.shape[0]):
            for j in range(board.shape[1]):
                if board[i][j] == 1:
                    board[i][j] = -1
                elif board[i][j] == 2:
                    board[i][j] = 1

        # first random sample an action from legal_actions
        action = np.random.choice(legal_actions(board))

        # Horizontal and vertical checks
        for i in range(3):
            if abs(sum(board[i, :])) == 2:
                # if i-th horizontal line has two same pieces and one empty position,
                # find the index in the i-th horizontal line
                ind = np.where(board[i, :] == 0)[0][0]
                # convert ind to action
                action = np.ravel_multi_index((np.array([i]), np.array([ind])), (3, 3))[0]
                if current_player_to_compute_bot_action * sum(board[i, :]) > 0:
                    # only take the action that will lead a connect3 of current player's pieces
                    return action

            if abs(sum(board[:, i])) == 2:
                # if i-th vertical line has two same pieces and one empty position
                # find the index in the i-th vertical line
                ind = np.where(board[:, i] == 0)[0][0]
                # convert ind to action
                action = np.ravel_multi_index((np.array([ind]), np.array([i])), (3, 3))[0]
                if current_player_to_compute_bot_action * sum(board[:, i]) > 0:
                    # only take the action that will lead a connect3 of current player's pieces
                    return action

        # Diagonal checks
        diag = board.diagonal()
        anti_diag = np.fliplr(board).diagonal()
        if abs(sum(diag)) == 2:
            # if diagonal has two same pieces and one empty position
            # find the index in the diag vector
            ind = np.where(diag == 0)[0][0]
            # convert ind to action
            action = np.ravel_multi_index((np.array([ind]), np.array([ind])), (3, 3))[0]
            if current_player_to_compute_bot_action * sum(diag) > 0:
                # only take the action that will lead a connect3 of current player's pieces
                return action

        if abs(sum(anti_diag)) == 2:
            # if anti-diagonal has two same pieces and one empty position
            # find the index in the anti_diag vector
            ind = np.where(anti_diag == 0)[0][0]
            # convert ind to action
            action = np.ravel_multi_index((np.array([ind]), np.array([2 - ind])), (3, 3))[0]
            if current_player_to_compute_bot_action * sum(anti_diag) > 0:
                # only take the action that will lead a connect3 of current player's pieces
                return action

        return action

    def _init_learn(self) -> None:
        pass

    def _init_collect(self) -> None:
        pass

    def _init_eval(self) -> None:
        pass

    def _forward_learn(self, data: dict) -> dict:
        pass

    def _forward_collect(self, envs: Dict, obs: Dict, temperature: float = 1) -> Dict[str, torch.Tensor]:
        ready_env_id = list(envs.keys())
        output = {}
        for env_id in ready_env_id:
            action = self.rule_bot_v0(obs[env_id])
            output[env_id] = {
                'action': action,
                'probs': None,
            }
        return output

    def _forward_eval(self, data: dict) -> dict:
        pass

    def _process_transition(self, obs: Any, model_output: dict, timestep: namedtuple) -> dict:
        pass

    def _get_train_sample(self, data: list) -> Union[None, List[Any]]:
        pass

    def default_model(self) -> Tuple[str, List[str]]:
        return 'bot_model', ['lzero.model.bot_model']

    def _monitor_vars_learn(self) -> List[str]:
        pass

    def reset(self):
        pass

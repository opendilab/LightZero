import copy
import logging
from collections import namedtuple
from typing import List, Dict, Any, Tuple, Union

import numpy as np
import torch
from ding.policy.base_policy import Policy
from ding.utils import POLICY_REGISTRY

from zoo.board_games.gomoku.envs.utils import check_action_to_special_connect4_case1, \
    check_action_to_special_connect4_case2, \
    check_action_to_connect4


@POLICY_REGISTRY.register('gomoku_bot_v0')
class GomokuBotV0(Policy):
    """
    Overview:
        Hard coded expert agent for tictactoe env.
    """
    config = dict(
        # (str) RL policy register name (refer to function "POLICY_REGISTRY").
        type='gomoku_bot_v0',
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

    def legal_actions(self, board):
        legal_actions = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if board[i][j] == 0:
                    legal_actions.append(i * self.board_size + j)
        return legal_actions
    def rule_bot_v0(self, obs):
        """
        Overview:
            Hard coded agent v0 for gomoku env.
            Considering the situation of to-connect-4 and to-connect-5 in a sliding window of 5X5, and lacks the consideration of the entire chessboard.
            In each sliding window of 5X5, first random sample a action from legal_actions,
            then take the action that will lead a connect4 or connect-5 of current/oppenent player's pieces.
        Returns:
            - action (:obj:`int`): the expert action to take in the current game state.
        """
        board = obs['board']
        current_player_to_compute_bot_action = -1 if obs['to_play'] == 1 else 1

        assert self.board_size >= 5, "current rule_bot_v0 is only support self.board_size>=5!"
        # To easily calculate expert action, we convert the chessboard notation:
        # from player 1:  1, player 2: 2
        # to   player 1: -1, player 2: 1
        # TODO: more elegant implementation
        board_deepcopy = copy.deepcopy(board)
        for i in range(board_deepcopy.shape[0]):
            for j in range(board_deepcopy.shape[1]):
                if board_deepcopy[i][j] == 1:
                    board_deepcopy[i][j] = -1
                elif board_deepcopy[i][j] == 2:
                    board_deepcopy[i][j] = 1

        # first random sample a action from legal_actions
        action = np.random.choice(self.legal_actions(board_deepcopy))

        size_of_board_template = 5
        shift_distance = [
            [i, j] for i in range(self.board_size - size_of_board_template + 1)
            for j in range(self.board_size - size_of_board_template + 1)
        ]
        action_block_opponent_to_connect5 = None
        action_to_connect4 = None
        action_to_special_connect4_case1 = None
        action_to_special_connect4_case2 = None

        min_to_connect = 3

        for board_block_index in range((self.board_size - size_of_board_template + 1) ** 2):
            """
            e.g., self.board_size=6
            board_block_index =[0,1,2,3]
            shift_distance = (0,0), (0,1), (1,0), (1,1)
            """
            shfit_tmp_board = copy.deepcopy(
                board_deepcopy[shift_distance[board_block_index][0]:size_of_board_template +
                                                                    shift_distance[board_block_index][0],
                shift_distance[board_block_index][1]:size_of_board_template +
                                                     shift_distance[board_block_index][1]]
            )

            # Horizontal and vertical checks
            for i in range(size_of_board_template):
                if abs(sum(shfit_tmp_board[i, :])) >= min_to_connect:
                    # if i-th horizontal line has three same pieces and two empty position, or four same pieces and one opponent piece.
                    # e.g., case1: .xxx. , case2: oxxxx

                    # find the index in the i-th horizontal line
                    zero_position_index = np.where(shfit_tmp_board[i, :] == 0)[0]
                    if zero_position_index.shape[0] == 0:
                        logging.debug(
                            'there is no empty position in this searched five positions, continue to search...'
                        )
                    else:
                        if zero_position_index.shape[0] == 2:
                            ind = np.random.choice(zero_position_index)
                        elif zero_position_index.shape[0] == 1:
                            ind = zero_position_index[0]
                        # convert ind to action
                        # the action that will lead a connect5 of current or opponent player's pieces
                        action = np.ravel_multi_index(
                            (
                                np.array([i + shift_distance[board_block_index][0]]
                                         ), np.array([ind + shift_distance[board_block_index][1]])
                            ), (self.board_size, self.board_size)
                        )[0]
                        if self.check_action_to_connect4_in_bot_v0:
                            if check_action_to_special_connect4_case1(shfit_tmp_board[i, :]):
                                action_to_special_connect4_case1 = action
                            if check_action_to_special_connect4_case2(shfit_tmp_board[i, :]):
                                action_to_special_connect4_case2 = action
                            if check_action_to_connect4(shfit_tmp_board[i, :]):
                                action_to_connect4 = action
                        if (current_player_to_compute_bot_action * sum(shfit_tmp_board[i, :]) > 0) and abs(sum(
                                shfit_tmp_board[i, :])) == size_of_board_template - 1:
                            # immediately take the action that will lead a connect5 of current player's pieces
                            return action
                        if (current_player_to_compute_bot_action * sum(shfit_tmp_board[i, :]) < 0) and abs(sum(
                                shfit_tmp_board[i, :])) == size_of_board_template - 1:
                            # memory the action that will lead a connect5 of opponent player's pieces, to avoid the forget
                            action_block_opponent_to_connect5 = action

                if abs(sum(shfit_tmp_board[:, i])) >= min_to_connect:
                    # if i-th vertical has three same pieces and two empty positions, or four same pieces and one opponent piece.
                    # e.g., case1: .xxx. , case2: oxxxx

                    # find the index in the i-th vertical line
                    zero_position_index = np.where(shfit_tmp_board[:, i] == 0)[0]
                    if zero_position_index.shape[0] == 0:
                        logging.debug(
                            'there is no empty position in this searched five positions, continue to search...'
                        )
                    else:
                        if zero_position_index.shape[0] == 2:
                            ind = np.random.choice(zero_position_index)
                        elif zero_position_index.shape[0] == 1:
                            ind = zero_position_index[0]

                        # convert ind to action
                        # the action that will lead a connect5 of current or opponent player's pieces
                        action = np.ravel_multi_index(
                            (
                                np.array([ind + shift_distance[board_block_index][0]]
                                         ), np.array([i + shift_distance[board_block_index][1]])
                            ), (self.board_size, self.board_size)
                        )[0]
                        if self.check_action_to_connect4_in_bot_v0:
                            if check_action_to_special_connect4_case1(shfit_tmp_board[:, i]):
                                action_to_special_connect4_case1 = action
                            if check_action_to_special_connect4_case2(shfit_tmp_board[:, i]):
                                action_to_special_connect4_case2 = action
                            if check_action_to_connect4(shfit_tmp_board[:, i]):
                                action_to_connect4 = action
                        if (current_player_to_compute_bot_action * sum(shfit_tmp_board[:, i]) > 0) and abs(sum(
                                shfit_tmp_board[:, i])) == size_of_board_template - 1:
                            # immediately take the action that will lead a connect5 of current player's pieces
                            return action
                        if (current_player_to_compute_bot_action * sum(shfit_tmp_board[:, i]) < 0) and abs(sum(
                                shfit_tmp_board[:, i])) == size_of_board_template - 1:
                            # memory the action that will lead a connect5 of opponent player's pieces, to avoid the forgetting
                            action_block_opponent_to_connect5 = action

            # Diagonal checks
            diag = shfit_tmp_board.diagonal()
            anti_diag = np.fliplr(shfit_tmp_board).diagonal()
            if abs(sum(diag)) >= min_to_connect:
                # if diagonal has three same pieces and two empty positions, or four same pieces and one opponent piece.
                # e.g., case1: .xxx. , case2: oxxxx
                #  finds the index in the diag vector

                zero_position_index = np.where(diag == 0)[0]
                if zero_position_index.shape[0] == 0:
                    logging.debug(
                        'there is no empty position in this searched five positions, continue to search...')
                else:
                    if zero_position_index.shape[0] == 2:
                        ind = np.random.choice(zero_position_index)
                    elif zero_position_index.shape[0] == 1:
                        ind = zero_position_index[0]

                    # convert ind to action
                    # the action that will lead a connect5 of current or opponent player's pieces
                    action = np.ravel_multi_index(
                        (
                            np.array([ind + shift_distance[board_block_index][0]]
                                     ), np.array([ind + shift_distance[board_block_index][1]])
                        ), (self.board_size, self.board_size)
                    )[0]
                    if self.check_action_to_connect4_in_bot_v0:
                        if check_action_to_special_connect4_case1(diag):
                            action_to_special_connect4_case1 = action
                        if check_action_to_special_connect4_case2(diag):
                            action_to_special_connect4_case2 = action
                        if check_action_to_connect4(diag):
                            action_to_connect4 = action
                    if current_player_to_compute_bot_action * sum(diag) > 0 and abs(
                            sum(diag)) == size_of_board_template - 1:
                        # immediately take the action that will lead a connect5 of current player's pieces
                        return action
                    if current_player_to_compute_bot_action * sum(diag) < 0 and abs(
                            sum(diag)) == size_of_board_template - 1:
                        # memory the action that will lead a connect5 of opponent player's pieces, to avoid the forget
                        action_block_opponent_to_connect5 = action

            if abs(sum(anti_diag)) >= min_to_connect:
                # if anti-diagonal has three same pieces and two empty position, or four same pieces and one opponent piece.
                # e.g., case1: .xxx. , case2: oxxxx

                # find the index in the anti_diag vector
                zero_position_index = np.where(anti_diag == 0)[0]
                if zero_position_index.shape[0] == 0:
                    logging.debug(
                        'there is no empty position in this searched five positions, continue to search...')
                else:
                    if zero_position_index.shape[0] == 2:
                        ind = np.random.choice(zero_position_index)
                    elif zero_position_index.shape[0] == 1:
                        ind = zero_position_index[0]
                    # convert ind to action
                    # the action that will lead a connect5 of current or opponent player's pieces
                    action = np.ravel_multi_index(
                        (
                            np.array([ind + shift_distance[board_block_index][0]]),
                            np.array([size_of_board_template - 1 - ind + shift_distance[board_block_index][1]])
                        ), (self.board_size, self.board_size)
                    )[0]
                    if self.check_action_to_connect4_in_bot_v0:
                        if check_action_to_special_connect4_case1(anti_diag):
                            action_to_special_connect4_case1 = action
                        if check_action_to_special_connect4_case2(anti_diag):
                            action_to_special_connect4_case2 = action
                        if check_action_to_connect4(anti_diag):
                            action_to_connect4 = action
                    if current_player_to_compute_bot_action * sum(anti_diag) > 0 and abs(
                            sum(anti_diag)) == size_of_board_template - 1:
                        # immediately take the action that will lead a connect5 of current player's pieces
                        return action
                    if current_player_to_compute_bot_action * sum(anti_diag) < 0 and abs(
                            sum(anti_diag)) == size_of_board_template - 1:
                        # memory the action that will lead a connect5 of opponent player's pieces, to avoid the forget
                        action_block_opponent_to_connect5 = action

        if action_block_opponent_to_connect5 is not None:
            return action_block_opponent_to_connect5
        elif action_to_special_connect4_case1 is not None:
            return action_to_special_connect4_case1
        elif action_to_special_connect4_case2 is not None:
            return action_to_special_connect4_case2
        elif action_to_connect4 is not None:
            return action_to_connect4
        else:
            return action

    def _init_learn(self) -> None:
        pass

    def _init_collect(self) -> None:
        self.board_size = self._cfg.board_size
        self.check_action_to_connect4_in_bot_v0 = False

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

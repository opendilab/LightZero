import copy
import random
import sys
from typing import List

import gym
import numpy as np
from ding.envs import BaseEnv, BaseEnvTimestep
from ding.utils import ENV_REGISTRY
from ditk import logging
from easydict import EasyDict

from zoo.board_games.alphabeta_pruning_bot import AlphaBetaPruningBot
from zoo.board_games.gomoku.envs.gomoku_rule_bot_v1 import GomokuRuleBotV1
from zoo.board_games.gomoku.envs.utils import check_action_to_special_connect4_case1, \
    check_action_to_special_connect4_case2, \
    check_action_to_connect4


@ENV_REGISTRY.register('gomoku')
class GomokuEnv(BaseEnv):

    config = dict(
        env_name="Gomoku",
        prob_random_agent=0,
        board_size=6,
        battle_mode='self_play_mode',
        mcts_mode='self_play_mode',  # only used in AlphaZero
        channel_last=False,
        scale=True,
        agent_vs_human=False,
        bot_action_type='v0',  # {'v0', 'alpha_beta_pruning'}
        prob_random_action_in_bot=0.,
        check_action_to_connect4_in_bot_v0=False,
        stop_value=1,
    )

    @classmethod
    def default_config(cls: type) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    def __init__(self, cfg: dict = None):
        self.cfg = cfg
        self.battle_mode = cfg.battle_mode
        # ``self.mcts_mode`` is only used in AlphaZero
        self.mcts_mode = cfg.get('mcts_mode', cfg.battle_mode)
        self.board_size = cfg.board_size
        self.prob_random_agent = cfg.prob_random_agent
        self.prob_random_action_in_bot = cfg.prob_random_action_in_bot

        self.channel_last = cfg.channel_last
        self.scale = cfg.scale
        self.check_action_to_connect4_in_bot_v0 = self.cfg.check_action_to_connect4_in_bot_v0

        self.players = [1, 2]
        self.board_markers = [str(i + 1) for i in range(self.board_size)]
        self.total_num_actions = self.board_size * self.board_size
        self.gomoku_rule_bot_v1 = GomokuRuleBotV1()
        self._env = self
        self.agent_vs_human = cfg.agent_vs_human
        self.bot_action_type = cfg.bot_action_type
        if self.bot_action_type == 'alpha_beta_pruning':
            self.alpha_beta_pruning_player = AlphaBetaPruningBot(self, cfg, 'alpha_beta_pruning_player')

    def reset(self, start_player_index=0, init_state=None):
        self._observation_space = gym.spaces.Box(
            low=0, high=2, shape=(self.board_size, self.board_size, 3), dtype=np.int32
        )
        self._action_space = gym.spaces.Discrete(self.board_size ** 2)
        self._reward_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.start_player_index = start_player_index
        self._current_player = self.players[self.start_player_index]
        if init_state is not None:
            self.board = np.array(copy.deepcopy(init_state), dtype="int32")
        else:
            self.board = np.zeros((self.board_size, self.board_size), dtype="int32")
        action_mask = np.zeros(self.total_num_actions, 'int8')
        action_mask[self.legal_actions] = 1
        if self.battle_mode == 'self_play_mode' or self.battle_mode == 'eval_mode':
            # In contrast with ``play_with_bot_mode``, in ``self_play_mode`` and ``eval_mode``,
            # we make ``to_play=self.current_player`` in obs, which is used to differentiate
            # the alternation of 2 players in the game when do Q calculation.
            obs = {
                'observation': self.current_state()[1],
                'action_mask': action_mask,
                'board': copy.deepcopy(self.board),
                'current_player_index': self.start_player_index,
                'to_play': self.current_player
            }
        else:
            obs = {
                'observation': self.current_state()[1],
                'action_mask': action_mask,
                'board': copy.deepcopy(self.board),
                'current_player_index': self.start_player_index,
                'to_play': -1
            }
        return obs

    def reset_v2(self, start_player_index=0, init_state=None):
        """
        Overview:
            only used in alpha-beta pruning bot.
        """
        self.start_player_index = start_player_index
        self._current_player = self.players[self.start_player_index]
        if init_state is not None:
            self.board = np.array(init_state, dtype="int32")
        else:
            self.board = np.zeros((self.board_size, self.board_size), dtype="int32")

    def step(self, action):
        if self.battle_mode == 'self_play_mode':
            if np.random.rand() < self.prob_random_agent:
                action = self.random_action()
            timestep = self._player_step(action)
            # print(self.board)
            return timestep
        elif self.battle_mode == 'play_with_bot_mode':
            # player 1 battle with expert player 2

            # player 1's turn
            timestep_player1 = self._player_step(action)
            # print('player 1 (efficientzero player): ' + self.action_to_string(action))  # Note: visualize
            if timestep_player1.done:

                # in play_with_bot_mode, we set to_play as None/-1, because we don't consider the alternation between players
                timestep_player1.obs['to_play'] = -1

                return timestep_player1

            # player 2's turn
            bot_action = self.bot_action()
            # print('player 2 (expert player): ' + self.action_to_string(bot_action))  # Note: visualize
            timestep_player2 = self._player_step(bot_action)
            # self.render()  # Note: visualize
            # the eval_episode_return is calculated from Player 1's perspective
            timestep_player2.info['eval_episode_return'] = -timestep_player2.reward
            timestep_player2 = timestep_player2._replace(reward=-timestep_player2.reward)

            timestep = timestep_player2

            # in ``play_with_bot_mode``, we set to_play as None/-1, because we don't consider the alternation between players
            timestep.obs['to_play'] = -1

            return timestep

        elif self.battle_mode == 'eval_mode':
            # player 1 battle with expert player 2

            # player 1's turn
            timestep_player1 = self._player_step(action)
            if self.agent_vs_human:
                print('player 1 (agent): ' + self.action_to_string(action))  # Note: visualize
                self.render()

            if timestep_player1.done:
                return timestep_player1

            # player 2's turn
            if self.agent_vs_human:
                bot_action = self.human_to_action()
            else:
                bot_action = self.bot_action()
                # bot_action = self.random_action()

            timestep_player2 = self._player_step(bot_action)
            if self.agent_vs_human:
                print('player 2 (human): ' + self.action_to_string(bot_action))  # Note: visualize
                self.render()

            # the eval_episode_return is calculated from Player 1's perspective
            timestep_player2.info['eval_episode_return'] = -timestep_player2.reward
            timestep_player2 = timestep_player2._replace(reward=-timestep_player2.reward)

            timestep = timestep_player2
            return timestep

    def _player_step(self, action):
        if action in self.legal_actions:
            row, col = self.action_to_coord(action)
            self.board[row, col] = self.current_player
        else:
            logging.warning(
                f"You input illegal action: {action}, the legal_actions are {self.legal_actions}. "
                f"Now we randomly choice a action from self.legal_actions."
            )
            action = np.random.choice(self.legal_actions)
            row, col = self.action_to_coord(action)
            self.board[row, col] = self.current_player

        # Check whether the game is ended or not and give the winner
        done, winner = self.get_done_winner()

        reward = np.array(float(winner == self.current_player)).astype(np.float32)
        info = {'next player to play': self.to_play}
        """
        NOTE: here exchange the player
        """
        self.current_player = self.to_play

        if done:
            info['eval_episode_return'] = reward
            # print('gomoku one episode done: ', info)

        action_mask = np.zeros(self.total_num_actions, 'int8')
        action_mask[self.legal_actions] = 1
        obs = {
            'observation': self.current_state()[1],
            'action_mask': action_mask,
            'board': copy.deepcopy(self.board),
            'current_player_index': self.players.index(self.current_player),
            'to_play': self.current_player
        }
        return BaseEnvTimestep(obs, reward, done, info)

    def current_state(self):
        """
        Overview:
            self.board is nd-array, 0 indicates that no stones is placed here,
            1 indicates that player 1's stone is placed here, 2 indicates player 2's stone is placed here
        Arguments:
            - raw_obs (:obj:`array`):
                the 0 dim means which positions is occupied by self.current_player,
                the 1 dim indicates which positions are occupied by self.to_play,
                the 2 dim indicates which player is the to_play player, 1 means player 1, 2 means player 2
        """
        board_curr_player = np.where(self.board == self.current_player, 1, 0)
        board_opponent_player = np.where(self.board == self.to_play, 1, 0)
        board_to_play = np.full((self.board_size, self.board_size), self.current_player)
        raw_obs = np.array([board_curr_player, board_opponent_player, board_to_play], dtype=np.float32)
        if self.scale:
            scale_obs = copy.deepcopy(raw_obs / 2)
        else:
            scale_obs = copy.deepcopy(raw_obs)

        if self.channel_last:
            # move channel dim to last axis
            # (C, W, H) -> (W, H, C)
            # e.g. (3, 6, 6) -> (6, 6, 3)
            return np.transpose(raw_obs, [1, 2, 0]), np.transpose(scale_obs, [1, 2, 0])
        else:
            # (C, W, H) e.g. (3, 6, 6)
            return raw_obs, scale_obs

    def get_done_winner(self):
        """
        Overview:
             Check if the game is over and who the winner is. Return 'done' and 'winner'.
        Returns:
            - outputs (:obj:`Tuple`): Tuple containing 'done' and 'winner',
                - if player 1 win,     'done' = True, 'winner' = 1
                - if player 2 win,     'done' = True, 'winner' = 2
                - if draw,             'done' = True, 'winner' = -1
                - if game is not over, 'done' = False, 'winner' = -1
        """
        # has_legal_actions i.e. not done
        has_legal_actions = False
        directions = ((1, -1), (1, 0), (1, 1), (0, 1))
        for i in range(self.board_size):
            for j in range(self.board_size):
                # if no stone is on the position, don't need to consider this position
                if self.board[i][j] == 0:
                    has_legal_actions = True
                    continue
                # value-value at a coord, i-row, j-col
                player = self.board[i][j]
                # check if there exist 5 in a line
                for d in directions:
                    x, y = i, j
                    count = 0
                    for _ in range(5):
                        if (x not in range(self.board_size)) or (y not in range(self.board_size)):
                            break
                        if self.board[x][y] != player:
                            break
                        x += d[0]
                        y += d[1]
                        count += 1
                        # if 5 in a line, store positions of all stones, return value
                        if count == 5:
                            return True, player
        # if the players don't have legal actions, return done=True
        return not has_legal_actions, -1

    def get_done_reward(self):
        """
        Overview:
             Check if the game is over and what is the reward in the perspective of player 1.
             Return 'done' and 'reward'.
        Returns:
            - outputs (:obj:`Tuple`): Tuple containing 'done' and 'reward',
                - if player 1 win,     'done' = True, 'reward' = 1
                - if player 2 win,     'done' = True, 'reward' = -1
                - if draw,             'done' = True, 'reward' = 0
                - if game is not over, 'done' = False,'reward' = None
        """
        done, winner = self.get_done_winner()
        if winner == 1:
            reward = 1
        elif winner == 2:
            reward = -1
        elif winner == -1 and done:
            reward = 0
        elif winner == -1 and not done:
            # episode is not done
            reward = None
        return done, reward

    @property
    def legal_actions(self):
        legal_actions = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i][j] == 0:
                    legal_actions.append(self.coord_to_action(i, j))
        return legal_actions

    def random_action(self):
        action_list = self.legal_actions
        return np.random.choice(action_list)

    def bot_action(self):
        if np.random.rand() < self.prob_random_action_in_bot:
            return self.random_action()
        else:
            if self.bot_action_type == 'v0':
                return self.rule_bot_v0()
            elif self.bot_action_type == 'v1':
                return self.rule_bot_v1()
            elif self.bot_action_type == 'alpha_beta_pruning':
                return self.bot_action_alpha_beta_pruning()

    def bot_action_alpha_beta_pruning(self):
        action = self.alpha_beta_pruning_player.get_best_action(self.board, player_index=self.current_player_index)
        return action

    def rule_bot_v1(self):
        action_mask = np.zeros(self.total_num_actions, 'int8')
        action_mask[self.legal_actions] = 1
        # NOTE: we use the original raw_obs for ``gomoku_rule_bot_v1.get_action()``
        obs = {'observation': self.current_state()[0], 'action_mask': action_mask}
        return self.gomoku_rule_bot_v1.get_action(obs)

    def rule_bot_v0(self):
        """
        Overview:
            Hard coded agent v0 for gomoku env.
            Considering the situation of to-connect-4 and to-connect-5 in a sliding window of 5X5, and lacks the consideration of the entire chessboard.
            In each sliding window of 5X5, first random sample a action from legal_actions,
            then take the action that will lead a connect4 or connect-5 of current/oppenent player's pieces.
        Returns:
            - action (:obj:`int`): the expert action to take in the current game state.
        """
        assert self.board_size >= 5, "current rule_bot_v0 is only support board_size>=5!"
        # To easily calculate expert action, we convert the chessboard notation:
        # from player 1:  1, player 2: 2
        # to   player 1: -1, player 2: 1
        # TODO: more elegant implementation
        board_deepcopy = copy.deepcopy(self.board)
        for i in range(board_deepcopy.shape[0]):
            for j in range(board_deepcopy.shape[1]):
                if board_deepcopy[i][j] == 1:
                    board_deepcopy[i][j] = -1
                elif board_deepcopy[i][j] == 2:
                    board_deepcopy[i][j] = 1

        # first random sample a action from legal_actions
        action = np.random.choice(self.legal_actions)

        size_of_board_template = 5
        shift_distance = [[i, j] for i in range(self.board_size - size_of_board_template + 1) for j in
                          range(self.board_size - size_of_board_template + 1)]
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
            shfit_tmp_board = copy.deepcopy(board_deepcopy[
                                            shift_distance[board_block_index][0]:size_of_board_template +
                                                                                 shift_distance[board_block_index][0],
                                            shift_distance[board_block_index][1]:size_of_board_template +
                                                                                 shift_distance[board_block_index][1]])

            # Horizontal and vertical checks
            for i in range(size_of_board_template):
                if abs(sum(shfit_tmp_board[i, :])) >= min_to_connect:
                    # if i-th horizontal line has three same pieces and two empty position, or four same pieces and one opponent piece.
                    # e.g., case1: .xxx. , case2: oxxxx

                    # find the index in the i-th horizontal line
                    zero_position_index = np.where(shfit_tmp_board[i, :] == 0)[0]
                    if zero_position_index.shape[0] == 0:
                        logging.debug(
                            'there is no empty position in this searched five positions, continue to search...')
                    else:
                        if zero_position_index.shape[0] == 2:
                            ind = random.choice(zero_position_index)
                        elif zero_position_index.shape[0] == 1:
                            ind = zero_position_index[0]
                        # convert ind to action
                        # the action that will lead a connect5 of current or opponent player's pieces
                        action = np.ravel_multi_index((np.array([i + shift_distance[board_block_index][0]]),
                                                       np.array([ind + shift_distance[board_block_index][1]])),
                                                      (self.board_size, self.board_size))[0]
                        if self.check_action_to_connect4_in_bot_v0:
                            if check_action_to_special_connect4_case1(shfit_tmp_board[i, :]):
                                action_to_special_connect4_case1 = action
                            if check_action_to_special_connect4_case2(shfit_tmp_board[i, :]):
                                action_to_special_connect4_case2 = action
                            if check_action_to_connect4(shfit_tmp_board[i, :]):
                                action_to_connect4 = action
                        if (self.current_player_to_compute_bot_action * sum(shfit_tmp_board[i, :]) > 0) and abs(
                                sum(shfit_tmp_board[i, :])) == size_of_board_template - 1:
                            # immediately take the action that will lead a connect5 of current player's pieces
                            return action
                        if (self.current_player_to_compute_bot_action * sum(shfit_tmp_board[i, :]) < 0) and abs(
                                sum(shfit_tmp_board[i, :])) == size_of_board_template - 1:
                            # memory the action that will lead a connect5 of opponent player's pieces, to avoid the forget
                            action_block_opponent_to_connect5 = action

                if abs(sum(shfit_tmp_board[:, i])) >= min_to_connect:
                    # if i-th vertical has three same pieces and two empty position, or four same pieces and one opponent piece.
                    # e.g., case1: .xxx. , case2: oxxxx

                    # find the index in the i-th vertical line
                    zero_position_index = np.where(shfit_tmp_board[:, i] == 0)[0]
                    if zero_position_index.shape[0] == 0:
                        logging.debug(
                            'there is no empty position in this searched five positions, continue to search...')
                    else:
                        if zero_position_index.shape[0] == 2:
                            ind = random.choice(zero_position_index)
                        elif zero_position_index.shape[0] == 1:
                            ind = zero_position_index[0]

                        # convert ind to action
                        # the action that will lead a connect5 of current or opponent player's pieces
                        action = np.ravel_multi_index((np.array([ind + shift_distance[board_block_index][0]]),
                                                       np.array([i + shift_distance[board_block_index][1]])),
                                                      (self.board_size, self.board_size))[0]
                        if self.check_action_to_connect4_in_bot_v0:
                            if check_action_to_special_connect4_case1(shfit_tmp_board[:, i]):
                                action_to_special_connect4_case1 = action
                            if check_action_to_special_connect4_case2(shfit_tmp_board[:, i]):
                                action_to_special_connect4_case2 = action
                            if check_action_to_connect4(shfit_tmp_board[:, i]):
                                action_to_connect4 = action
                        if (self.current_player_to_compute_bot_action * sum(shfit_tmp_board[:, i]) > 0) and abs(
                                sum(shfit_tmp_board[:, i])) == size_of_board_template - 1:
                            # immediately take the action that will lead a connect5 of current player's pieces
                            return action
                        if (self.current_player_to_compute_bot_action * sum(shfit_tmp_board[:, i]) < 0) and abs(
                                sum(shfit_tmp_board[:, i])) == size_of_board_template - 1:
                            # memory the action that will lead a connect5 of opponent player's pieces, to avoid the forget
                            action_block_opponent_to_connect5 = action

            # Diagonal checks
            diag = shfit_tmp_board.diagonal()
            anti_diag = np.fliplr(shfit_tmp_board).diagonal()
            if abs(sum(diag)) >= min_to_connect:
                # if diagonal has three same pieces and two empty position, or four same pieces and one opponent piece.
                # e.g., case1: .xxx. , case2: oxxxx
                # find the index in the diag vector

                zero_position_index = np.where(diag == 0)[0]
                if zero_position_index.shape[0] == 0:
                    logging.debug(
                        'there is no empty position in this searched five positions, continue to search...')
                else:
                    if zero_position_index.shape[0] == 2:
                        ind = random.choice(zero_position_index)
                    elif zero_position_index.shape[0] == 1:
                        ind = zero_position_index[0]

                    # convert ind to action
                    # the action that will lead a connect5 of current or opponent player's pieces
                    action = np.ravel_multi_index((np.array([ind + shift_distance[board_block_index][0]]),
                                                   np.array([ind + shift_distance[board_block_index][1]])),
                                                  (self.board_size, self.board_size))[0]
                    if self.check_action_to_connect4_in_bot_v0:
                        if check_action_to_special_connect4_case1(diag):
                            action_to_special_connect4_case1 = action
                        if check_action_to_special_connect4_case2(diag):
                            action_to_special_connect4_case2 = action
                        if check_action_to_connect4(diag):
                            action_to_connect4 = action
                    if self.current_player_to_compute_bot_action * sum(diag) > 0 and abs(
                            sum(diag)) == size_of_board_template - 1:
                        # immediately take the action that will lead a connect5 of current player's pieces
                        return action
                    if self.current_player_to_compute_bot_action * sum(diag) < 0 and abs(
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
                        ind = random.choice(zero_position_index)
                    elif zero_position_index.shape[0] == 1:
                        ind = zero_position_index[0]
                    # convert ind to action
                    # the action that will lead a connect5 of current or opponent player's pieces
                    action = np.ravel_multi_index((np.array([ind + shift_distance[board_block_index][0]]), np.array(
                        [size_of_board_template - 1 - ind + shift_distance[board_block_index][1]])),
                                                  (self.board_size, self.board_size))[0]
                    if self.check_action_to_connect4_in_bot_v0:
                        if check_action_to_special_connect4_case1(anti_diag):
                            action_to_special_connect4_case1 = action
                        if check_action_to_special_connect4_case2(anti_diag):
                            action_to_special_connect4_case2 = action
                        if check_action_to_connect4(anti_diag):
                            action_to_connect4 = action
                    if self.current_player_to_compute_bot_action * sum(anti_diag) > 0 and abs(
                            sum(anti_diag)) == size_of_board_template - 1:
                        # immediately take the action that will lead a connect5 of current player's pieces
                        return action
                    if self.current_player_to_compute_bot_action * sum(anti_diag) < 0 and abs(
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

    def naive_rule_bot_v0_for_board_size_5(self):
        """
        Overview:
            Hard coded expert agent for gomoku env.
            First random sample a action from legal_actions, then take the action that will lead a connect4 of current player's pieces.
        Returns:
            - action (:obj:`int`): the expert action to take in the current game state.
        """
        assert self.board_size == 5, "current naive_rule_bot_v0 is only support board_size=5!"
        # To easily calculate expert action, we convert the chessboard notation:
        # from player 1:  1, player 2: 2
        # to   player 1: -1, player 2: 1
        # TODO: more elegant implementation
        board = copy.deepcopy(self.board)
        for i in range(board.shape[0]):
            for j in range(board.shape[1]):
                if board[i][j] == 1:
                    board[i][j] = -1
                elif board[i][j] == 2:
                    board[i][j] = 1

        # first random sample a action from legal_actions
        action = np.random.choice(self.legal_actions)
        # Horizontal and vertical checks
        for i in range(self.board_size):
            if abs(sum(board[i, :])) == 4:
                # if i-th horizontal line has four same pieces and one empty position
                # find the index in the i-th horizontal line
                ind = np.where(board[i, :] == 0)[0][0]
                # convert ind to action
                action = np.ravel_multi_index((np.array([i]), np.array([ind])), (self.board_size, self.board_size))[0]
                if self.current_player_to_compute_bot_action * sum(board[i, :]) > 0:
                    # immediately take the action that will lead a connect5 of current player's pieces
                    return action

            if abs(sum(board[:, i])) == 4:
                # if i-th vertical line has two same pieces and one empty position
                # find the index in the i-th vertical line
                ind = np.where(board[:, i] == 0)[0][0]
                # convert ind to action
                action = np.ravel_multi_index((np.array([ind]), np.array([i])), (self.board_size, self.board_size))[0]
                if self.current_player_to_compute_bot_action * sum(board[:, i]) > 0:
                    # immediately take the action that will lead a connect5 of current player's pieces
                    return action

        # Diagonal checks
        diag = board.diagonal()
        anti_diag = np.fliplr(board).diagonal()
        if abs(sum(diag)) == 4:
            # if diagonal has two same pieces and one empty position
            # find the index in the diag vector
            ind = np.where(diag == 0)[0][0]
            # convert ind to action
            action = np.ravel_multi_index((np.array([ind]), np.array([ind])), (self.board_size, self.board_size))[0]
            if self.current_player_to_compute_bot_action * sum(diag) > 0:
                # immediately take the action that will lead a connect5 of current player's pieces
                return action

        if abs(sum(anti_diag)) == 4:
            # if anti-diagonal has two same pieces and one empty position
            # find the index in the anti_diag vector
            ind = np.where(anti_diag == 0)[0][0]
            # convert ind to action
            action = np.ravel_multi_index((np.array([ind]), np.array([self.board_size - 1 - ind])),
                                          (self.board_size, self.board_size))[0]
            if self.current_player_to_compute_bot_action * sum(anti_diag) > 0:
                # immediately take the action that will lead a connect5 of current player's pieces
                return action

        return action

    @property
    def current_player(self):
        return self._current_player

    @property
    def current_player_index(self):
        """
        current_player_index = 0, current_player = 1
        current_player_index = 1, current_player = 2
        """
        return 0 if self._current_player == 1 else 1

    @property
    def to_play(self):
        return self.players[0] if self.current_player == self.players[1] else self.players[1]

    @property
    def current_player_to_compute_bot_action(self):
        """
        Overview: to compute expert action easily.
        """
        return -1 if self.current_player == 1 else 1

    def human_to_action(self):
        """
        Overview:
            For multiplayer games, ask the user for a legal action
            and return the corresponding action number.
        Returns:
            An integer from the action space.
        """
        # print(self.board)
        while True:
            try:
                row = int(input(
                    f"Enter the row (1, 2, ...,{self.board_size}, from up to bottom) to play for the player {self.current_player}: "
                )
                )
                col = int(input(
                    f"Enter the column (1, 2, ...,{self.board_size}, from left to right) to play for the player {self.current_player}: "
                )
                )
                choice = self.coord_to_action(row - 1, col - 1)
                if (choice in self.legal_actions and 1 <= row and 1 <= col and row <= self.board_size
                        and col <= self.board_size):
                    break
                else:
                    print("Wrong input, try again")
            except KeyboardInterrupt:
                print("exit")
                sys.exit(0)
            except Exception as e:
                print("Wrong input, try again")
        return choice

    def coord_to_action(self, i, j):
        """
        Overview:
            convert coordinate i, j to action index a in [0, board_size**2)
        """
        return i * self.board_size + j

    def action_to_coord(self, a):
        """
        Overview:
            convert action index a in [0, board_size**2) to coordinate (i, j)
        """
        return a // self.board_size, a % self.board_size

    def action_to_string(self, action_number):
        """
        Overview:
            Convert an action number to a string representing the action.
        Arguments:
            - action_number: an integer from the action space.
        Returns:
            - String representing the action.
        """
        row = action_number // self.board_size + 1
        col = action_number % self.board_size + 1
        return f"Play row {row}, column {col}"

    def simulate_action(self, action):
        """
        Overview:
            execute action and get next_simulator_env. used in AlphaZero.
        Returns:
            Returns Gomoku instance.
        """
        if action not in self.legal_actions:
            raise ValueError("action {0} on board {1} is not legal".format(action, self.board))
        new_board = copy.deepcopy(self.board)
        row, col = self.action_to_coord(action)
        new_board[row, col] = self.current_player
        if self.start_player_index == 0:
            start_player_index = 1  # self.players = [1, 2], start_player = 2, start_player_index = 1
        else:
            start_player_index = 0  # self.players = [1, 2], start_player = 1, start_player_index = 0
        next_simulator_env = copy.deepcopy(self)
        next_simulator_env.reset(start_player_index, init_state=new_board)  # index
        return next_simulator_env

    def simulate_action_v2(self, board, start_player_index, action):
        """
        Overview:
            execute action from board and get new_board, new_legal_actions. used in AlphaZero.
        Returns:
            - new_board (:obj:`np.array`):
            - new_legal_actions (:obj:`np.array`):
        """
        self.reset(start_player_index, init_state=board)
        if action not in self.legal_actions:
            raise ValueError("action {0} on board {1} is not legal".format(action, self.board))
        row, col = self.action_to_coord(action)
        self.board[row, col] = self.current_player
        new_legal_actions = copy.deepcopy(self.legal_actions)
        new_board = copy.deepcopy(self.board)
        return new_board, new_legal_actions

    def clone(self):
        return copy.deepcopy(self)

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def render(self, mode="human"):
        marker = "   "
        for i in range(self.board_size):
            if i <= 8:
                marker = marker + self.board_markers[i] + "  "
            else:
                marker = marker + self.board_markers[i] + " "
        print(marker)
        for row in range(self.board_size):
            if row <= 8:
                print(str(1 + row) + ' ', end=" ")
            else:
                print(str(1 + row), end=" ")
            for col in range(self.board_size):
                ch = self.board[row][col]
                if ch == 0:
                    print(".", end="  ")
                elif ch == 1:
                    print("X", end="  ")
                elif ch == 2:
                    print("O", end="  ")
            print()

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        return self._action_space

    @property
    def reward_space(self) -> gym.spaces.Space:
        return self._reward_space

    @current_player.setter
    def current_player(self, value):
        self._current_player = value

    @staticmethod
    def create_collector_env_cfg(cfg: dict) -> List[dict]:
        collector_env_num = cfg.pop('collector_env_num')
        cfg = copy.deepcopy(cfg)
        return [cfg for _ in range(collector_env_num)]

    @staticmethod
    def create_evaluator_env_cfg(cfg: dict) -> List[dict]:
        evaluator_env_num = cfg.pop('evaluator_env_num')
        cfg = copy.deepcopy(cfg)
        # When we collect and train agent in ``self_play_mode``, in eval phase,
        # we use ``eval_mode`` to make agent paly with the built-in bot to
        # evaluate the performance of the current agent.
        if cfg.battle_mode == 'self_play_mode':
            cfg.battle_mode = 'eval_mode'

        return [cfg for _ in range(evaluator_env_num)]

    def __repr__(self) -> str:
        return "LightZero Gomoku Env"

    def close(self) -> None:
        pass

import copy
import os
import sys
from typing import List

import gym
import numpy as np
import pygame
from ding.envs import BaseEnv, BaseEnvTimestep
from ding.torch_utils import to_list
from ding.utils import ENV_REGISTRY
from ditk import logging
from easydict import EasyDict
from gym import spaces
from pettingzoo.classic.go import coords, go_base
from pettingzoo.classic.go.go import raw_env
from pettingzoo.utils.agent_selector import agent_selector

from zoo.board_games.go.envs.katago_policy import str_coord, GameState, str_coord, KatagoPolicy, parse_coord
import imageio
import time
import signal
import time


# def timeout_handler(signum, frame):
#     raise TimeoutError("Execution time too long")
#
#
# # 设置超时
# signal.signal(signal.SIGALRM, timeout_handler)
# signal.alarm(50)  # 设置50秒的闹钟

def get_image(path):
    from os import path as os_path

    import pygame
    cwd = os_path.dirname(__file__)
    image = pygame.image.load(cwd + '/' + path)
    sfc = pygame.Surface(image.get_size(), flags=pygame.SRCALPHA)
    sfc.blit(image, (0, 0))
    return sfc


from datetime import datetime


def generate_gif_filename(prefix="go", extension=".gif"):
    current_time = datetime.now()
    timestamp = current_time.strftime("%Y%m%d-%H%M%S")
    filename = f"{prefix}-{timestamp}{extension}"
    return filename


def flatten_action_to_gtp_action(flatten_action, board_size):
    if flatten_action == board_size * board_size:
        return "pass"

    row = board_size - 1 - (flatten_action // board_size)
    col = flatten_action % board_size

    # 跳过字母 'I'
    if col >= ord('I') - ord('A'):
        col += 1

    col_str = chr(col + ord('A'))
    row_str = str(row + 1)

    gtp_action = col_str + row_str
    return gtp_action


@ENV_REGISTRY.register('go_lightzero')
class GoEnv(BaseEnv):
    """
    Overview:
        Go environment.
        board: X black, O white, . empty
        Represent a board as a numpy array, with 0 empty, 1 is black, -1 is white.

        self._raw_env._go.to_play: 1 black, -1 white
    Interface:
        reset, step, seed, close, render, close, seed
    Property:
        action_space, observation_space, reward_range, spec
    """

    config = dict(
        env_name="Go",
        stop_value=1,
        board_size=6,
        komi=7.5,
        battle_mode='self_play_mode',
        mcts_mode='self_play_mode',  # only used in AlphaZero
        save_gif_replay=False,
        save_gif_path='./',
        render_in_ui=False,
        bot_action_type='v0',  # {'v0', 'alpha_beta_pruning'}
        agent_vs_human=False,
        prob_random_agent=0,
        prob_expert_agent=0,
        channel_last=True,
        scale=True,
        ignore_pass_if_have_other_legal_actions=True,
        device='cpu',
        katago_policy=None,
    )

    @classmethod
    def default_config(cls: type) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    def __init__(self, cfg=None):

        # board_size: a int, representing the board size (board has a board_size x board_size shape)
        # komi: a float, representing points given to the second player.
        self.cfg = cfg
        self.channel_last = cfg.channel_last
        self.scale = cfg.scale
        self.battle_mode = cfg.battle_mode
        # The mode of interaction between the agent and the environment.
        assert self.battle_mode in ['self_play_mode', 'play_with_bot_mode', 'eval_mode']
        # The mode of MCTS is only used in AlphaZero.
        self.mcts_mode = 'self_play_mode'

        self.board_size = cfg.board_size
        self.prob_random_agent = cfg.prob_random_agent
        self.prob_random_action_in_bot = cfg.prob_random_action_in_bot
        self.channel_last = cfg.channel_last
        self.scale = cfg.scale
        self.agent_vs_human = cfg.agent_vs_human
        self.bot_action_type = cfg.bot_action_type

        self.players = [1, 2]
        self.board_markers = [str(i + 1) for i in range(self.board_size)]
        self.total_num_actions = self.board_size * self.board_size + 1

        self._komi = cfg.komi
        self.board_size = cfg.board_size
        self.agents = ['black_0', 'white_0']
        self.num_agents = len(self.agents)
        self.possible_agents = self.agents[:]
        self._agent_selector = agent_selector(self.agents)
        self.has_reset = False
        self.screen = None

        self._observation_space = spaces.Dict(
            {
                'observation': spaces.Box(low=0, high=1, shape=(self.board_size, self.board_size, 17),
                                          dtype=bool),
                'action_mask': spaces.Box(low=0, high=1, shape=((self.board_size * self.board_size) + 1,),
                                          dtype=np.int8)
            })
        self._action_space = spaces.Discrete(self.board_size * self.board_size + 1)
        self._reward_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

        self.board_history = np.zeros((self.board_size, self.board_size, 16), dtype=bool)

        self.save_gif_replay = cfg.save_gif_replay
        self.render_in_ui = cfg.render_in_ui
        self.katago_checkpoint_path = cfg.katago_checkpoint_path
        self.ignore_pass_if_have_other_legal_actions = cfg.ignore_pass_if_have_other_legal_actions
        # self.device = cfg.device
        self.katago_policy = cfg.katago_policy
        # self.katago_policy = KatagoPolicy(checkpoint_path=self.katago_checkpoint_path, board_size=self.board_size,
        #                               ignore_pass_if_have_other_legal_actions=self.ignore_pass_if_have_other_legal_actions, device=self.device)

    # Represent a board as a numpy array, with 0 empty, 1 is black, -1 is white.
    def reset(self, start_player_index=0, init_state=None, katago_policy_init=True, katago_game_state=None):
        if katago_policy_init:
            # 使用函数生成基于当前时间的 GIF 文件名
            gif_filename = generate_gif_filename(prefix=f"go_bs{self.board_size}", )
            # print(gif_filename)
            self.save_gif_path = self.cfg.save_gif_path + gif_filename
            self.frames = []

        # TODO(pu): katago_game_state init
        self.katago_game_state = GameState(self.board_size)


        # from katago_policy import Board
        # self.katago_board = Board(self.board_size)

        self.len_of_legal_actions_for_current_player = self.board_size * self.board_size + 1
        self.start_player_index = start_player_index
        self._current_player = self.players[self.start_player_index]

        if self.current_player == 1:
            agent_id = 'black_0'
            self._agent_selector = agent_selector(['black_0', 'white_0'])
        elif self.current_player == 2:
            agent_id = 'white_0'
            self._agent_selector = agent_selector(['white_0', 'black_0'])

        self.agent_selection = self._agent_selector.next()
        self._raw_env = raw_env(board_size=self.board_size, komi=self._komi)
        self._raw_env.reset()

        if init_state is not None:
            # Represent a board as a numpy array, with 0 empty, 1 is black, -1 is white.
            # Note, to_play in Position is different from to_play in GoEnv.
            self._raw_env._go = go_base.Position(board=copy.deepcopy(init_state), komi=self._komi,
                                                 to_play=1 if self.start_player_index == 0 else -1)

            if katago_policy_init:
                # TODO(pu)
                # ****** update katago internal game state ******
                # self.reset_katago_game_state_v0(copy.deepcopy(init_state))
                self.reset_katago_game_state_v1(copy.deepcopy(katago_game_state))

        else:
            self._raw_env._go = go_base.Position(board=np.zeros((self.board_size, self.board_size), dtype="int32"),
                                                 komi=self._komi, to_play=1 if self.start_player_index == 0 else -1)

        self._cumulative_rewards = self._convert_to_dict(np.array([0.0, 0.0]))
        self.rewards = self._convert_to_dict(np.array([0.0, 0.0]))

        self.dones = self._convert_to_dict([False for _ in range(self.num_agents)])
        self.infos = self._convert_to_dict([{} for _ in range(self.num_agents)])

        self.next_legal_moves = self._raw_env._encode_legal_actions(self._raw_env._go.all_legal_moves())
        self.board_history = np.zeros((self.board_size, self.board_size, 16), dtype=bool)

        for agent, reward in self.rewards.items():
            self._cumulative_rewards[agent] += reward

        # obs = self._go.observe(agent_id)
        # obs = self._raw_env.observe(agent_id)
        obs = self.observe(agent_id)

        # obs['action_mask'] is the action mask for the last player
        self.action_mask = np.zeros(self.total_num_actions, 'int8')
        self.action_mask[self.legal_actions] = 1

        obs['action_mask'] = self.action_mask
        obs['observation'] = obs['observation'].astype(int)
        obs['board'] = copy.deepcopy(self._raw_env._go.board)
        obs['current_player_index'] = self.current_player_index
        obs['to_play'] = self.current_player
        obs['katago_game_state'] = self.katago_game_state
        # self.step_num = 0
        # obs['step_num'] = self.step_num

        return obs

    def reset_katago_game_state_v0(self, init_state):
        # TODO(pu): have bug now
        # NOTE: Represent a board as a numpy array < init_state>, with 0 empty, 1 is black, -1 is white.
        # ****** update katago internal game state ******
        self.katago_game_state_board = np.zeros(shape=(self.katago_game_state.board.arrsize), dtype=np.int8)
        # 将init_state转化为self.board的格式
        for i in range(self.board_size):
            for j in range(self.board_size):
                # black, white, empty
                # 1, -1, 0 -> 1, 2, 0
                if init_state[i][j] == 1:
                    position_num = 1
                elif init_state[i][j] == -1:
                    position_num = 2
                elif init_state[i][j] == 0:
                    position_num = 0
                self.katago_game_state_board[self.katago_game_state.board.loc(i, j)] = position_num
        # 设置棋盘边界为Board.WALL
        for i in range(-1, self.board_size + 1):
            self.katago_game_state_board[self.katago_game_state.board.loc(i, -1)] = 3  # Board.WALL
            self.katago_game_state_board[self.katago_game_state.board.loc(i, self.board_size)] = 3  # Board.WALL
            self.katago_game_state_board[self.katago_game_state.board.loc(-1, i)] = 3  # Board.WALL
            self.katago_game_state_board[self.katago_game_state.board.loc(self.board_size, i)] = 3  # Board.WALL
        # 设置最后一个元素
        self.katago_game_state_board[-1] = 0  # Board.EMPTY
        self.katago_game_state.board.board = self.katago_game_state_board
        self.katago_game_state.board.pla = self._current_player

        # print(self.katago_game_state.board.board[:-1].reshape(self.board_size+2,self.board_size+1))

        # ****** update katago internal game state ******

    def reset_katago_game_state_v1(self, katago_game_state):
        # ****** update katago internal game state ******
        # TODO(pu)
        self.katago_game_state = katago_game_state
        # TODO(pu): clear the ko point
        self.katago_game_state.board.simple_ko_point = None

        # print(self.katago_game_state.board.board[:-1].reshape(self.board_size+2,self.board_size+1))

    def _player_step(self, action):
        if self.current_player == 1:
            agent_id = 'black_0'
        elif self.current_player == 2:
            agent_id = 'white_0'

        # the count of empty position
        zero_count = np.count_nonzero(self.board == 0)
        if zero_count == 1:
            # must give pass
            action = self.board_size * self.board_size  # pass

        self.len_of_legal_actions_for_current_player = len(self.legal_actions)

        if action in self.legal_actions:
            self._raw_env._go = self._raw_env._go.play_move(coords.from_flat(action))
        else:
            logging.warning(
                f"You input illegal action: {action}, the legal_actions are {self.legal_actions}. "
                f"Now we randomly choice a action from self.legal_actions."
            )
            action = np.random.choice(self.legal_actions)
            assert action in self.legal_actions, f'action: {action}, legal_actions: {self.legal_actions}'
            self._raw_env._go = self._raw_env._go.play_move(coords.from_flat(action))

        try:
            # ****** update katago internal game state ******
            pla = (1 if agent_id == 'black_0' else 2)
            gtp_action = flatten_action_to_gtp_action(action, self.board_size)
            loc = parse_coord(gtp_action, self.katago_game_state.board)
            # print(self.katago_game_state.board.board[:-1].reshape(self.board_size+2,self.board_size+1))
            # 禁用闹钟
            # signal.alarm(0)
            # try:
            #     # 执行你的代码
            #     self.katago_game_state.board.play(pla, loc)
            # except TimeoutError as e:
            #     print("Timeout! The function is taking too long to complete. It might be stuck in an infinite loop.")
            #     # 这里你可以打印出一些有助于调试的信息，例如pla和loc的值
            #     print("pla: ", pla)
            #     print("loc: ", loc)
            self.katago_game_state.board.play(pla, loc)
            self.katago_game_state.moves.append((pla, loc))
            self.katago_game_state.boards.append(self.katago_game_state.board.copy())


            # ****** update katago internal game state ******
        except Exception as e:
            print('update katago internal game state exception', e)

        obs = self.observe(agent_id)

        current_agent_plane, opponent_agent_plane = self._raw_env._encode_board_planes(agent_id)
        self.board_history = np.dstack((current_agent_plane, opponent_agent_plane, self.board_history[:, :, :-2]))
        # self.board_history[:,:,0], self.board_history[:,:,1]

        current_agent = self.agent_selection
        # next_player: 'black_0', 'white_0'
        """
        NOTE: here exchange the player
        """
        self.agent_selection = self._agent_selector.next()
        next_agent = self.agent_selection
        self._current_player = self.to_play

        # obs['action_mask'] is the action mask for the last player
        action_mask = np.zeros(self.total_num_actions, 'int8')
        action_mask[self.legal_actions] = 1
        obs['action_mask'] = action_mask
        obs['observation'] = obs['observation'].astype(int)
        obs['board'] = copy.deepcopy(self._raw_env._go.board)
        # obs['current_player_index'] = self.players.index(self.current_player)
        obs['current_player_index'] = self.current_player_index
        obs['to_play'] = self.current_player
        obs['katago_game_state'] = self.katago_game_state

        if self._raw_env._go.is_game_over():
            self._raw_env.terminations = self._convert_to_dict(
                [True for _ in range(self.num_agents)]
            )
            self.rewards = self._convert_to_dict(
                self._encode_rewards(self._raw_env._go.result())
            )
            # TODO(pu): modify the self._raw_env._go.result()
            # if self.len_of_legal_actions_for_last_player > 1 and self.len_of_legal_actions_for_current_player == 1:
            #     # The last player has other legal actions but chooses to pass, and the current player only has one legal action, which is to pass.
            #     # This indicates that the last player wins.
            #     self.rewards[current_agent] = -1
            #     self.rewards[next_agent] = 1
            #     print(f'current_agent: {current_agent}', f'self.len_of_legal_actions_for_last_player: {self.len_of_legal_actions_for_last_player}, '
            #                                              f'self.len_of_legal_actions_for_current_player: {self.len_of_legal_actions_for_current_player}')

            self.next_legal_moves = [self.board_size * self.board_size]
        else:
            self.next_legal_moves = self._encode_legal_actions(self._raw_env._go.all_legal_moves())

        self.len_of_legal_actions_for_last_player = self.len_of_legal_actions_for_current_player

        for agent, reward in self.rewards.items():
            self._cumulative_rewards[agent] += reward

        self.dones[current_agent] = (
                self._raw_env.terminations[current_agent]
                or self._raw_env.truncations[current_agent]
        )
        if self.dones[current_agent]:
            self.infos[current_agent]['eval_episode_return'] = self._cumulative_rewards[current_agent]

        # The returned reward and done is calculated from current_agent's perspective.
        return BaseEnvTimestep(obs, self.rewards[current_agent], self.dones[current_agent], self.infos[current_agent])

    def step(self, action):
        if self.save_gif_replay and not self.render_in_ui:
            self.render_and_capture_frame(mode='only_save_gif')
        if self.save_gif_replay and self.render_in_ui:
            self.render_and_capture_frame(mode='render_in_ui')

        if self.battle_mode == 'self_play_mode':
            if np.random.rand() < self.prob_random_agent:
                action = self.random_action()
            timestep = self._player_step(action)
            if timestep.done:
                # The eval_episode_return is calculated from Player 1's perspective.
                timestep.info['eval_episode_return'] = -timestep.reward if timestep.obs[
                                                                               'to_play'] == 1 else timestep.reward
                if self.save_gif_replay and len(self.frames) > 0:
                    # Save the frames as a GIF file
                    self.save_gif(self.save_gif_path)

            return timestep
        elif self.battle_mode == 'play_with_bot_mode':
            # player 1 battle with expert player 2

            # player 1's turn
            timestep_player1 = self._player_step(action)
            # print('player 1 (efficientzero player): ' + self.action_to_string(action))  # Note: visualize
            if timestep_player1.done:
                # in play_with_bot_mode, we set to_play as None/-1, because we don't consider the alternation between players
                timestep_player1.obs['to_play'] = -1

                if self.save_gif_replay:
                    # Save the frames as a GIF file
                    self.save_gif(self.save_gif_path)

            # player 2's turn
            bot_action = self.bot_action()
            # print('player 2 (expert player): ' + self.action_to_string(bot_action))  # Note: visualize
            timestep_player2 = self._player_step(bot_action)
            # self.render()  # Note: visualize
            # the eval_episode_return is calculated from Player 1's perspective
            timestep_player2.info['eval_episode_return'] = -timestep_player2.reward
            timestep_player2 = timestep_player2._replace(reward=-timestep_player2.reward)

            timestep = timestep_player2
            # NOTE: in play_with_bot_mode, we must set to_play as -1, because we don't consider the alternation between players.
            # And the to_play is used in MCTS.
            timestep.obs['to_play'] = -1
            return timestep

        elif self.battle_mode == 'eval_mode':
            # player 1 battle with bot player 2

            # ==============================================================
            # player 1's turn
            # ==============================================================
            # ****** update katago internal game state ******
            # TODO(pu): how to avoid this?
            # katago_flatten_action = self.lz_flatten_to_katago_flatten(action, self.board_size)
            # print('player 1:', str_coord(katago_flatten_action, self.katago_game_state.board))
            # self.update_katago_internal_game_state(katago_flatten_action, to_play=1)

            timestep_player1 = self._player_step(action)
            # print(self.board)
            # self.show_katago_board()

            if timestep_player1.done:
                # in eval_mode, we set to_play as None/-1, because we don't consider the alternation between players
                timestep_player1.obs['to_play'] = -1

                if self.save_gif_replay:
                    # Save the frames as a GIF file
                    self.save_gif(self.save_gif_path)

                return timestep_player1

            if self.agent_vs_human:
                print('player 1 (alphazero): ' + self.action_to_string(action))  # Note: visualize
                self.render()

            # ==============================================================
            # player 2's turn
            # ==============================================================
            if self.agent_vs_human:
                # bot_action = self.human_to_action()
                bot_action = self.human_to_gtp_action()
            else:
                # s_time = time.time()
                bot_action = self.get_katago_action(to_play=2)
                # e_time = time.time()
                # print(f'katago_action time: {e_time - s_time}')
                if bot_action not in self.legal_actions:
                    logging.warning(
                        f"You input illegal *bot* action: {bot_action}, the legal_actions are {self.legal_actions}. "
                        f"Now we randomly choice a action from self.legal_actions."
                    )
                    bot_action = np.random.choice(self.legal_actions)
                # ****** update katago internal game state ******
                # TODO(pu): how to avoid this?
                katago_flatten_action = self.lz_flatten_to_katago_flatten(bot_action, self.board_size)
                print('player 2 (katago):', str_coord(katago_flatten_action, self.katago_game_state.board))
                # self.update_katago_internal_game_state(katago_flatten_action, to_play=2)

            timestep_player2 = self._player_step(bot_action)
            # print(self.board)
            # self.show_katago_board()

            if timestep_player2.done:
                if self.save_gif_replay:
                    # Save the frames as a GIF file
                    self.save_gif(self.save_gif_path)

            if self.agent_vs_human:
                print('player 2 (human): ' + self.action_to_string(bot_action))  # Note: visualize
                self.render()

            # the eval_episode_return is calculated from Player 1's perspective
            timestep_player2.info['eval_episode_return'] = -timestep_player2.reward
            timestep_player2 = timestep_player2._replace(reward=-timestep_player2.reward)

            timestep = timestep_player2
            # NOTE: in eval_mode, we must set to_play as -1, because we don't consider the alternation between players.
            # And the to_play is used in MCTS.
            timestep.obs['to_play'] = -1
            return timestep

    def update_katago_internal_game_state(self, katago_flatten_action, to_play):
        # Note: cannot use self.to_play, because self.to_play is updated after the self._player_step(action)
        # ****** update internal game state ******
        gtp_action = str_coord(katago_flatten_action, self.katago_game_state.board)
        if to_play == 1:
            command = ['play', 'b', gtp_action]
        else:
            command = ['play', 'w', gtp_action]
        self.katago_policy.katago_command(self.katago_game_state, command, to_play)

    # ==============================================================
    # katago related
    # ==============================================================
    def get_katago_action(self, to_play):
        command = ['get_katago_action']
        # self.current_player is the player who will play
        flatten_action = self.katago_policy.katago_command(self.katago_game_state, command, to_play=to_play)
        return flatten_action

    def show_katago_board(self):
        command = ["showboard"]
        # self.current_player is the player who will play
        self.katago_policy.katago_command(self.katago_game_state, command)

    def lz_flatten_to_katago_flatten(self, lz_flatten_action, board_size):
        """ Convert lz Flattened Coordinate to katago Flattened Coordinate."""
        # self.arrsize = (board_size + 1) * (board_size + 2) + 1
        # xxxxxxxxxx
        # .........x
        # .........x
        # .........x
        # .........x
        # .........x
        # .........x
        # .........x
        # .........x
        # .........x
        # xxxxxxxxxx

        if lz_flatten_action == board_size * board_size:
            return 0  # Board.PASS_LOC
        # convert action index in [0, board_size**2) to coordinate (i, j)
        y, x = lz_flatten_action // board_size, lz_flatten_action % board_size
        return (board_size + 1) * (y + 1) + x + 1
        # 0 -> (0, 0) -> 11
        # 1 -> (0, 1) -> 12
        # 9 -> (1, 0) -> 21

        # row = action_number // self.board_size + 1
        # col = action_number % self.board_size + 1
        # return f"Play row {row}, column {col}"

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
        if self._raw_env._go.is_game_over():
            result = self._raw_env._go.result()
            if result == 1:
                return True, 1
            elif result == -1:
                return True, 2
            elif result == 0:
                return True, -1
        else:
            return False, -1

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
        if self._raw_env._go.is_game_over():
            result = self._raw_env._go.result()
            if result == 1:
                return True, 1
            elif result == -1:
                return True, -1
            elif result == 0:
                return True, 0
        else:
            return False, None

    def observe(self, agent):
        current_agent_plane, opponent_agent_plane = self._raw_env._encode_board_planes(agent)
        player_plane = self._raw_env._encode_player_plane(agent)

        observation = np.dstack((self.board_history, player_plane))

        legal_moves = self.next_legal_moves if agent == self.agent_selection else []
        action_mask = np.zeros((self.board_size * self.board_size) + 1, "int8")
        for i in legal_moves:
            action_mask[i] = 1

        return {"observation": observation, "action_mask": action_mask}

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
        if self.current_player == 1:
            agent_id = 'black_0'
        elif self.current_player == 2:
            agent_id = 'white_0'
        # obs = self._go.observe(agent_id)
        # obs = self._raw_env.observe(agent_id)
        obs = self.observe(agent_id)

        obs['observation'] = obs['observation'].astype(int)
        raw_obs = obs['observation']

        if self.channel_last:
            # (W, H, C) (6, 6, 17)
            return raw_obs, raw_obs
        else:
            # move channel dim to first axis
            # (W, H, C) -> (C, W, H)
            # e.g. (6, 6, 17) - > (17, 6, 6)
            return np.transpose(raw_obs, [2, 0, 1]), np.transpose(raw_obs, [2, 0, 1])

    def legal_moves(self):
        if self._raw_env._go.is_game_over():
            self.terminations = self._convert_to_dict(
                [True for _ in range(self.num_agents)]
            )
            self.rewards = self._convert_to_dict(
                self._encode_rewards(self._raw_env._go.result())
            )
            self.next_legal_moves = [self.board_size * self.board_size]
        else:
            self.next_legal_moves = self._encode_legal_actions(self._raw_env._go.all_legal_moves())

        return self.next_legal_moves

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
        if self.start_player_index == 0:
            start_player_index = 1  # self.players = [1, 2], start_player = 2, start_player_index = 1
        else:
            start_player_index = 0  # self.players = [1, 2], start_player = 1, start_player_index = 0
        # next_simulator_env = copy.deepcopy(self)
        raw_env = copy.deepcopy(self._raw_env)
        # tmp_position = next_simulator_env._raw_env._go.play_move(coords.from_flat(action))
        tmp_position = raw_env._go.play_move(coords.from_flat(action))
        new_board = copy.deepcopy(tmp_position.board)
        # TODO(pu)
        # katago_game_state_copy = copy.deepcopy(self.katago_game_state)
        next_simulator_env = copy.deepcopy(self)
        next_simulator_env.reset(start_player_index, init_state=new_board, katago_policy_init=False)  # index
        # NOTE: when calling reset method, self.recent is cleared, so we need to restore it.
        next_simulator_env._raw_env._go.recent = tmp_position.recent
        # TODO(pu)
        # next_simulator_env.katago_game_state = katago_game_state_copy

        return next_simulator_env

    def random_action(self):
        return np.random.choice(self.legal_actions)

    def bot_action(self):
        return self.random_action()

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
                row = int(
                    input(
                        f"Enter the row (1, 2, ...,{self.board_size}, from up to bottom) to play for the player {self.current_player}: "
                    )
                )
                col = int(
                    input(
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

    def human_to_gtp_action(self):
        """
        Overview:
            For multiplayer games, ask the user for a legal action
            and return the corresponding action number.
        Returns:
            An integer from the action space.
        """
        while True:
            try:
                gtp_action = str(
                    input(
                        f"Enter the GTP action to play for the player {self.current_player}: "
                    )
                )

                flatten_action = self.gtp_action_to_flatten_action(gtp_action, board_size=self.board_size)
                if flatten_action in self.legal_actions:
                    break
                else:
                    print("Wrong input, try again")
            except KeyboardInterrupt:
                print("exit")
                sys.exit(0)
            except Exception as e:
                print("Wrong input, try again")
        return flatten_action

    def gtp_action_to_flatten_action(self, gtp_action, board_size):
        if gtp_action.lower() == "pass":
            return board_size * board_size

        col_str, row_str = gtp_action[0], gtp_action[1:]
        col = ord(col_str.upper()) - ord('A')
        if col >= ord('I') - ord('A'):
            col -= 1  # 跳过字母 'I'
        row = int(row_str) - 1

        flatten_action = (board_size - 1 - row) * board_size + col
        return flatten_action

    # ==============================================================
    # render related
    # ==============================================================

    def render_and_capture_frame(self, mode='only_save_gif'):
        self.render(mode=mode)
        self.capture_frame()

    def render(self, mode='render_in_ui'):
        if not hasattr(self, "frames"):
            self.frames = []

        if mode == "board":
            # print(self._raw_env._go.board)
            print(self.board)
            return

        screen_width = 1026
        screen_height = 1026

        if self.screen is None:
            if mode in ['only_save_gif', "rgb_array"]:
                self.screen = pygame.Surface((screen_width, screen_height))
            elif mode == "render_in_ui":
                pygame.init()
                pygame.display.init()
                self.screen = pygame.display.set_mode((screen_width, screen_height))
        if mode in ["render_in_ui"]:
            pygame.event.get()

        size = self.board_size
        # Load and scale all of the necessary images
        tile_size = (screen_width) / size

        black_stone = get_image(os.path.join('../img', 'GoBlackPiece.png'))
        black_stone = pygame.transform.scale(black_stone, (int(tile_size * (5 / 6)), int(tile_size * (5 / 6))))

        white_stone = get_image(os.path.join('../img', 'GoWhitePiece.png'))
        white_stone = pygame.transform.scale(white_stone, (int(tile_size * (5 / 6)), int(tile_size * (5 / 6))))

        tile_img = get_image(os.path.join('../img', 'GO_Tile0.png'))
        tile_img = pygame.transform.scale(tile_img, ((int(tile_size * (7 / 6))), int(tile_size * (7 / 6))))

        # blit board tiles
        for i in range(1, size - 1):
            for j in range(1, size - 1):
                self.screen.blit(tile_img, ((i * (tile_size)), int(j) * (tile_size)))

        for i in range(1, 9):
            tile_img = get_image(os.path.join('../img', 'GO_Tile' + str(i) + '.png'))
            tile_img = pygame.transform.scale(tile_img, ((int(tile_size * (7 / 6))), int(tile_size * (7 / 6))))
            for j in range(1, size - 1):
                if i == 1:
                    self.screen.blit(tile_img, (0, int(j) * (tile_size)))
                elif i == 2:
                    self.screen.blit(tile_img, ((int(j) * (tile_size)), 0))
                elif i == 3:
                    self.screen.blit(tile_img, ((size - 1) * (tile_size), int(j) * (tile_size)))
                elif i == 4:
                    self.screen.blit(tile_img, ((int(j) * (tile_size)), (size - 1) * (tile_size)))
            if i == 5:
                self.screen.blit(tile_img, (0, 0))
            elif i == 6:
                self.screen.blit(tile_img, ((size - 1) * (tile_size), 0))
            elif i == 7:
                self.screen.blit(tile_img, ((size - 1) * (tile_size), (size - 1) * (tile_size)))
            elif i == 8:
                self.screen.blit(tile_img, (0, (size - 1) * (tile_size)))

        offset = tile_size * (1 / 6)
        board_tmp = np.transpose(self._raw_env._go.board)

        # Blit the necessary chips and their positions
        for i in range(0, size):
            for j in range(0, size):
                if board_tmp[i][j] == go_base.BLACK:
                    self.screen.blit(black_stone, ((i * (tile_size) + offset), int(j) * (tile_size) + offset))
                elif board_tmp[i][j] == go_base.WHITE:
                    self.screen.blit(white_stone, ((i * (tile_size) + offset), int(j) * (tile_size) + offset))

        if mode == "render_in_ui":
            pygame.display.update()

        observation = np.array(pygame.surfarray.pixels3d(self.screen))

        return np.transpose(observation, axes=(1, 0, 2)) if mode == "rgb_array" else None

    def capture_frame(self):
        if not hasattr(self, "frames"):
            self.frames = []

        frame = np.array(pygame.surfarray.array3d(self.screen))
        # Transpose the frame to fix the rotation issue
        fixed_frame = np.transpose(frame, axes=(1, 0, 2))
        self.frames.append(fixed_frame)

    def save_gif(self, output_file, duration=20):
        imageio.mimsave(output_file, self.frames, format="GIF", duration=duration)
        print("Gif saved to {}".format(output_file))

    def _encode_player_plane(self, agent):
        if agent == self.possible_agents[0]:
            return np.zeros([self.board_size, self.board_size], dtype=bool)
        else:
            return np.ones([self.board_size, self.board_size], dtype=bool)

    def _convert_to_dict(self, list_of_list):
        return dict(zip(self.possible_agents, list_of_list))

    def _encode_legal_actions(self, actions):
        return np.where(actions == 1)[0]

    def _encode_rewards(self, result):
        return [1, -1] if result == 1 else [-1, 1]

    def clone(self):
        return copy.deepcopy(self)

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

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
        """
        current_player_index = 0, current_player = 1, to_play = 2
        current_player_index = 1, current_player = 2, to_play = 1
        """
        return self.players[0] if self.current_player == self.players[1] else self.players[1]

    @property
    def current_player_to_compute_bot_action(self):
        """
        Overview: to compute expert action easily.
        """
        return -1 if self.current_player == 1 else 1

    @current_player.setter
    def current_player(self, value):
        self._current_player = value

    @property
    def legal_actions(self):
        return to_list(self.legal_moves())

    @property
    def board(self):
        return self._raw_env._go.board

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        return self._action_space

    @property
    def reward_space(self) -> gym.spaces.Space:
        return self._reward_space

    @staticmethod
    def create_collector_env_cfg(cfg: dict) -> List[dict]:
        collector_env_num = cfg.pop('collector_env_num')
        cfg = copy.deepcopy(cfg)
        return [cfg for _ in range(collector_env_num)]

    @staticmethod
    def create_evaluator_env_cfg(cfg: dict) -> List[dict]:
        evaluator_env_num = cfg.pop('evaluator_env_num')
        cfg = copy.deepcopy(cfg)
        # In eval phase, we use ``eval_mode`` to make agent play with the built-in bot to
        # evaluate the performance of the current agent.
        cfg.battle_mode = 'eval_mode'
        return [cfg for _ in range(evaluator_env_num)]

    def __repr__(self) -> str:
        return "LightZero Go Env"

    def close(self) -> None:
        pass

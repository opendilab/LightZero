"""
Overview:
    Adapt the connect4 environment in PettingZoo (https://github.com/Farama-Foundation/PettingZoo) to the BaseEnv interface.
    Connect Four is a 2-player turn based game, where players must connect four of their tokens vertically, horizontally or diagonally. 
    The players drop their respective token in a column of a standing grid, where each token will fall until it reaches the bottom of the column or reaches an existing token. 
    Players cannot place a token in a full column, and the game ends when either a player has made a sequence of 4 tokens, or when all 7 columns have been filled.
Mode:
    - ``self_play_mode``: In ``self_play_mode``, two players take turns to play. This mode is used in AlphaZero for data generating.
    - ``play_with_bot_mode``: In this mode, the environment has a bot inside, which take the role of player 2. So the player may play against the bot.
Bot:
    - MCTSBot: A bot which take action through a Monte Carlo Tree Search, which has a high performance.
    - RuleBot: A bot which take action according to some simple settings, which has a moderate performance. Note: Currently the RuleBot can only exclude actions that would lead to losing the game within three moves. 
        Note: Currently the RuleBot can only exclude actions that would lead to losing the game within three moves. One possible improvement is to further enhance the bot's long-term planning capabilities.
Observation Space:
    The observation in the Connect4 environment is a dictionary with five elements, which contains key information about the current state. 
    - observation (:obj:`array`): An array that represents information about the current state, with a shape of (3, 6, 7). 
        The length of the first dimension is 3, which stores three two-dimensional game boards with shapes (6, 7).
        These boards represent the positions occupied by the current player, the positions occupied by the opponent player, and the identity of the current player, respectively.
    - action_mask (:obj:`array`): A mask for the actions, indicating which actions are executable. It is a one-dimensional array of length 7, corresponding to columns 1 to 7 of the game board. 
        It has a value of 1 for the columns where a move can be made, and a value of 0 for other positions.
    - board (:obj:`array`): A visual representation of the current game board, represented as a 6x7 array, in which the positions where player 1 and player 2 have placed their tokens are marked with values 1 and 2, respectively. 
    - current_player_index (:obj:`int`): The index of the current player, with player 1 having an index of 0 and player 2 having an index of 1. 
    - to_play (:obj:`int`): The player who needs to take an action in the current state, with a value of 1 or 2.
Action Space:
    A set of integers from 0 to 6 (inclusive), where the action represents which column a token should be dropped in.
Reward Space:
    For the ``self_play_mode``, a reward of 1 is returned at the time step when the game terminates, and a reward of 0 is returned at all other time steps.
    For the ``play_with_bot_mode``, at the time step when the game terminates, if the bot wins, the reward is -1; if the agent wins, the reward is 1; and in all other cases, the reward is 0.
"""

import copy
import os
import sys
from typing import List

import imageio
import numpy as np
import pygame
from ding.envs import BaseEnv, BaseEnvTimestep
from ding.utils import ENV_REGISTRY
from ditk import logging
from easydict import EasyDict
from gym import spaces

from zoo.board_games.connect4.envs.rule_bot import Connect4RuleBot
from zoo.board_games.mcts_bot import MCTSBot


def get_image(path):
    from os import path as os_path

    import pygame

    cwd = os_path.dirname(__file__)
    image = pygame.image.load(cwd + "/" + path)
    sfc = pygame.Surface(image.get_size(), flags=pygame.SRCALPHA)
    sfc.blit(image, (0, 0))
    return sfc


@ENV_REGISTRY.register('connect4')
class Connect4Env(BaseEnv):
    config = dict(
        # (str) The name of the environment registered in the environment registry.
        env_name="Connect4",
        # (str) The mode of the environment when take a step.
        battle_mode='self_play_mode',
        # (str) The mode of the environment when doing the MCTS.
        mcts_mode='self_play_mode',
        # (str) The type of the bot of the environment.
        bot_action_type='mcts',
        # (bool) Whether to let human to play with the agent when evaluating. If False, then use the bot to evaluate the agent.
        agent_vs_human=False,
        # (float) The probability that a random agent is used instead of the learning agent.
        prob_random_agent=0,
        # (float) The probability that an expert agent(the bot) is used instead of the learning agent.
        prob_expert_agent=0,
        # (float) The probability that a random action will be taken when calling the bot.
        prob_random_action_in_bot=0.,
        # (float) The scale of the render screen.
        screen_scaling=9,
        # (bool) Whether to save the replay of the game.
        save_replay=False,
        # (bool) Whether to use the 'channel last' format for the observation space. If False, 'channel first' format is used.
        channel_last=False,
        # (bool) Whether to scale the observation.
        scale=False,
        # (float) The stop value when training the agent. If the evalue return reach the stop value, then the training will stop.
        stop_value=2,
    )

    @classmethod
    def default_config(cls: type) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    def __init__(self, cfg=None) -> None:
        # Load the config.
        self.cfg = cfg

        # Set the format of the observation.
        self.channel_last = cfg.channel_last
        self.scale = cfg.scale

        # Set the parameters about replay render.
        self.screen_scaling = cfg.screen_scaling
        self.save_replay = cfg.save_replay
        self.replay_name_suffix = "test"
        self.replay_path = None
        self.replay_format = 'gif'
        self.screen = None
        self.frames = []

        # Set the mode of interaction between the agent and the environment.
        # options = {'self_play_mode', 'play_with_bot_mode', 'eval_mode'}
        self.battle_mode = cfg.battle_mode
        assert self.battle_mode in ['self_play_mode', 'play_with_bot_mode', 'eval_mode']
        # The mode of MCTS is only used in AlphaZero.
        self.mcts_mode = 'self_play_mode'

        # In ``eval_mode``, we can choose to play with the agent.
        self.agent_vs_human = cfg.agent_vs_human

        # Set some randomness for selecting action.
        self.prob_random_agent = cfg.prob_random_agent
        self.prob_expert_agent = cfg.prob_expert_agent
        assert (self.prob_random_agent >= 0 and self.prob_expert_agent == 0) or (
                self.prob_random_agent == 0 and self.prob_expert_agent >= 0), \
            f'self.prob_random_agent:{self.prob_random_agent}, self.prob_expert_agent:{self.prob_expert_agent}'

        # The board state is saved as a one-dimensional array instead of a two-dimensional array for ease of computation in ``step()`` function.
        self.board = [0] * (6 * 7)

        self.players = [1, 2]
        self._current_player = 1
        self._env = self

        # Set the bot type and add some randomness.
        # options = {'rule, 'mcts'}
        self.bot_action_type = cfg.bot_action_type
        self.prob_random_action_in_bot = cfg.prob_random_action_in_bot
        if self.bot_action_type == 'mcts':
            cfg_temp = EasyDict(cfg.copy())
            cfg_temp.save_replay = False
            cfg_temp.bot_action_type = None
            env_mcts = Connect4Env(EasyDict(cfg_temp))
            self.mcts_bot = MCTSBot(env_mcts, 'mcts_player', 50)
        elif self.bot_action_type == 'rule':
            self.rule_bot = Connect4RuleBot(self, self._current_player)

        # Initialize the screen if the replay is to be saved.
        if self.save_replay:
            self.render(mode='rgb_array_render')

    def _player_step(self, action, flag):
        """
        Overview:
            A function that implements the transition of the environment's state. \
            After taking an action in the environment, the function transitions the environment to the next state \
            and returns the relevant information for the next time step.
        Arguments:
            - action (:obj:`int`): A value from 0 to 6 indicating the position to move on the connect4 board.
            - flag (:obj:`str`): A marker indicating the source of an action, for debugging convenience.
        Returns:
            - timestep (:obj:`BaseEnvTimestep`): A namedtuple that records the observation and obtained reward after taking the action, \
            whether the game is terminated, and some other information. 
        """
        if action in self.legal_actions:
            piece = self.players.index(self._current_player) + 1
            for i in list(filter(lambda x: x % 7 == action, list(range(41, -1, -1)))):
                if self.board[i] == 0:
                    self.board[i] = piece
                    break
        else:
            print(np.array(self.board).reshape(6, 7))
            logging.warning(
                f"You input illegal action: {action}, the legal_actions are {self.legal_actions}. "
                f"flag is {flag}."
                f"Now we randomly choice a action from self.legal_actions."
            )
            action = self.random_action()
            print("the random action is", action)
            piece = self.players.index(self._current_player) + 1
            for i in list(filter(lambda x: x % 7 == action, list(range(41, -1, -1)))):
                if self.board[i] == 0:
                    self.board[i] = piece
                    break

        # Check if there is a winner.
        done, winner = self.get_done_winner()
        if not winner == -1:
            reward = np.array(1).astype(np.float32)
        else:
            reward = np.array(0).astype(np.float32)

        info = {}

        self._current_player = self.next_player

        if done:
            info['eval_episode_return'] = reward

        obs = self.observe()

        return BaseEnvTimestep(obs, reward, done, info)

    def step(self, action):
        """
        Overview:
            The step function of the environment. It receives an action from the player and returns the state of the environment after performing that action. \
            In ``self_play_mode``, this function only call ``_player_step()`` once since the agent play with it self and play the role of both two players 1 and 2.\
            In ``play_with_bot_mode``, this function first use the recieved ``action`` to call the ``_player_step()`` and then use the action from bot to call it again.\
            Then return the result of taking these two actions sequentially in the environment.\
            In ``eval_mode``, this function also call ``_player_step()`` twice, and the second action is from human action or from the bot.
        Arguments:
            - action (:obj:`int`): A value from 0 to 6 indicating the position to move on the connect4 board.
        Returns:
            - timestep (:obj:`BaseEnvTimestep`): A namedtuple that records the observation and obtained reward after taking the action, \
            whether the game is terminated, and some other information.        
        """
        if self.battle_mode == 'self_play_mode':

            if self.prob_random_agent > 0:
                if np.random.rand() < self.prob_random_agent:
                    action = self.random_action()
            elif self.prob_expert_agent > 0:
                if np.random.rand() < self.prob_expert_agent:
                    action = self.bot_action()

            flag = "agent"
            timestep = self._player_step(action, flag)

            if self.save_replay:
                self.render(mode='rgb_array_render')

            if timestep.done:
                # The ``eval_episode_return`` is calculated from player 1's perspective.
                timestep.info['eval_episode_return'] = -timestep.reward if timestep.obs[
                                                                               'to_play'] == 1 else timestep.reward
                if self.save_replay:
                    self.save_render_output(replay_name_suffix=self.replay_name_suffix, replay_path=self.replay_path,
                                            format=self.replay_format)

            return timestep

        elif self.battle_mode == 'play_with_bot_mode':
            # Player 1's turn.
            flag = "bot_agent"
            timestep_player1 = self._player_step(action, flag)

            if self.save_replay:
                self.render(mode='rgb_array_render')

            if timestep_player1.done:
                # NOTE: in ``play_with_bot_mode``, we must set to_play as -1, because we don't consider the alternation between players.
                # And the ``to_play`` is used in MCTS.
                timestep_player1.obs['to_play'] = -1

                if self.save_replay:
                    self.save_render_output(replay_name_suffix=self.replay_name_suffix, replay_path=self.replay_path,
                                            format=self.replay_format)

                return timestep_player1

            # Player 2's turn.
            bot_action = self.bot_action()
            flag = "bot_bot"
            timestep_player2 = self._player_step(bot_action, flag)
            # The ``eval_episode_return`` is calculated from player 1's perspective.
            timestep_player2.info['eval_episode_return'] = -timestep_player2.reward
            timestep_player2 = timestep_player2._replace(reward=-timestep_player2.reward)

            if self.save_replay:
                self.render(mode='rgb_array_render')

            timestep = timestep_player2
            # NOTE: in ``play_with_bot_mode``, we must set to_play as -1, because we don't consider the alternation between players.
            # And the ``to_play`` is used in MCTS.
            timestep.obs['to_play'] = -1

            if timestep.done:
                if self.save_replay:
                    self.save_render_output(replay_name_suffix=self.replay_name_suffix, replay_path=self.replay_path,
                                            format=self.replay_format)

            return timestep

        elif self.battle_mode == 'eval_mode':
            # Player 1's turn.
            flag = "eval_agent"
            timestep_player1 = self._player_step(action, flag)

            if self.save_replay:
                self.render(mode='rgb_array_render')

            if timestep_player1.done:
                # NOTE: in ``eval_mode``, we must set to_play as -1, because we don't consider the alternation between players.
                # And the ``to_play`` is used in MCTS.
                timestep_player1.obs['to_play'] = -1

                if self.save_replay:
                    self.save_render_output(replay_name_suffix=self.replay_name_suffix, replay_path=self.replay_path,
                                            format=self.replay_format)

                return timestep_player1

            # Player 2's turn.
            if self.agent_vs_human:
                bot_action = self.human_to_action()
            else:
                bot_action = self.bot_action()

            flag = "eval_bot"
            timestep_player2 = self._player_step(bot_action, flag)

            if self.save_replay:
                self.render(mode='rgb_array_render')

            # The ``eval_episode_return`` is calculated from player 1's perspective.
            timestep_player2.info['eval_episode_return'] = -timestep_player2.reward
            timestep_player2 = timestep_player2._replace(reward=-timestep_player2.reward)

            timestep = timestep_player2
            # NOTE: in ``eval_mode``, we must set to_play as -1, because we don't consider the alternation between players.
            # And the ``to_play`` is used in MCTS.
            timestep.obs['to_play'] = -1

            if timestep.done:
                if self.save_replay:
                    self.save_render_output(replay_name_suffix=self.replay_name_suffix, replay_path=self.replay_path,
                                            format=self.replay_format)

            return timestep

    def reset(self, start_player_index=0, init_state=None, replay_name_suffix=None):
        """
        Overview:
            Env reset and custom state start by init_state.
        Arguments:
            - start_player_index(:obj:`int`): players = [1,2], player_index = [0,1]
            - init_state(:obj:`array`): custom start state.
        """
        if replay_name_suffix is not None:
            self.replay_name_suffix = replay_name_suffix
        if init_state is None:
            self.board = [0] * (6 * 7)
        else:
            self.board = init_state
        self.players = [1, 2]
        self.start_player_index = start_player_index
        self._current_player = self.players[self.start_player_index]

        self._action_space = spaces.Discrete(7)
        self._reward_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self._observation_space = spaces.Dict(
            {
                "observation": spaces.Box(low=0, high=1, shape=(3, 6, 7), dtype=np.int8),
                "action_mask": spaces.Box(low=0, high=1, shape=(7,), dtype=np.int8),
                "board": spaces.Box(low=0, high=2, shape=(6, 7), dtype=np.int8),
                "current_player_index": spaces.Discrete(2),
                "to_play": spaces.Discrete(2),
            }
        )

        obs = self.observe()
        return obs

    def current_state(self):
        """
        Overview:
            Obtain the state from the view of current player.\
            self.board is nd-array, 0 indicates that no stones is placed here,\
            1 indicates that player 1's stone is placed here, 2 indicates player 2's stone is placed here.
        Returns:
            - current_state (:obj:`array`):
                the 0 dim means which positions is occupied by ``self.current_player``,\
                the 1 dim indicates which positions are occupied by ``self.next_player``,\
                the 2 dim indicates which player is the to_play player, 1 means player 1, 2 means player 2.
        """
        board_vals = np.array(self.board).reshape(6, 7)
        board_curr_player = np.where(board_vals == self.current_player, 1, 0)
        board_opponent_player = np.where(board_vals == self.next_player, 1, 0)
        board_to_play = np.full((6, 7), self.current_player)
        raw_obs = np.array([board_curr_player, board_opponent_player, board_to_play], dtype=np.float32)
        if self.scale:
            scale_obs = copy.deepcopy(raw_obs / 2)
        else:
            scale_obs = copy.deepcopy(raw_obs)
        if self.channel_last:
            # move channel dim to last axis
            # (C, W, H) -> (W, H, C)
            return np.transpose(raw_obs, [1, 2, 0]), np.transpose(scale_obs, [1, 2, 0])
        else:
            # (C, W, H)
            return raw_obs, scale_obs

    def observe(self):
        legal_moves = self.legal_actions

        action_mask = np.zeros(7, "int8")
        for i in legal_moves:
            action_mask[i] = 1

        if self.battle_mode == 'play_with_bot_mode' or self.battle_mode == 'eval_mode':
            return {"observation": self.current_state()[1],
                    "action_mask": action_mask,
                    "board": copy.deepcopy(self.board),
                    "current_player_index": self.players.index(self._current_player),
                    "to_play": -1
                    }
        elif self.battle_mode == 'self_play_mode':
            return {"observation": self.current_state()[1],
                    "action_mask": action_mask,
                    "board": copy.deepcopy(self.board),
                    "current_player_index": self.players.index(self._current_player),
                    "to_play": self._current_player
                    }

    @property
    def legal_actions(self):
        return [i for i in range(7) if self.board[i] == 0]

    def render(self, mode='rgb_array_render'):
        """
        Overview:
            Renders the Connect Four game environment.
        Arguments:
            - mode (:obj:`str`): The rendering mode. Options are 'print', 'human', or 'rgb_array_render'.
        """
        # In 'print' mode, print the current game board for rendering.
        if mode == "print":
            print(np.array(self.board).reshape(6, 7))
            return
        # In other two modes, use a screen for rendering. 
        screen_width = 99 * self.screen_scaling
        screen_height = 86 / 99 * screen_width
        if self.screen is None:
            pygame.init()
            if mode == "human":
                pygame.display.set_caption("Connect Four")
                self.screen = pygame.display.set_mode((screen_width, screen_height))
            elif mode == "rgb_array_render":
                self.screen = pygame.Surface((screen_width, screen_height))

        # Load and scale all of the necessary images.
        tile_size = (screen_width * (91 / 99)) / 7

        red_chip = get_image(os.path.join("img", "C4RedPiece.png"))
        red_chip = pygame.transform.scale(
            red_chip, (int(tile_size * (9 / 13)), int(tile_size * (9 / 13)))
        )

        black_chip = get_image(os.path.join("img", "C4BlackPiece.png"))
        black_chip = pygame.transform.scale(
            black_chip, (int(tile_size * (9 / 13)), int(tile_size * (9 / 13)))
        )

        board_img = get_image(os.path.join("img", "Connect4Board.png"))
        board_img = pygame.transform.scale(
            board_img, ((int(screen_width)), int(screen_height))
        )

        self.screen.blit(board_img, (0, 0))

        # Blit the necessary chips and their positions.
        for i in range(0, 42):
            if self.board[i] == 1:
                self.screen.blit(
                    red_chip,
                    (
                        (i % 7) * (tile_size) + (tile_size * (6 / 13)),
                        int(i / 7) * (tile_size) + (tile_size * (6 / 13)),
                    ),
                )
            elif self.board[i] == 2:
                self.screen.blit(
                    black_chip,
                    (
                        (i % 7) * (tile_size) + (tile_size * (6 / 13)),
                        int(i / 7) * (tile_size) + (tile_size * (6 / 13)),
                    ),
                )
        # TODO: complish the human mode.
        if mode == "human":
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])

        # Draw the observation and save to frames.
        observation = np.array(pygame.surfarray.pixels3d(self.screen))
        self.frames.append(np.transpose(observation, axes=(1, 0, 2)))

        return None

    def save_render_output(self, replay_name_suffix: str = '', replay_path=None, format='gif'):
        """
        Overview:
            Save the rendered frames as an output file.
        Arguments:
            - replay_name_suffix (:obj:`str`): The suffix to be added to the replay filename.
            - replay_path (:obj:`str`): The path to save the replay file. If None, the default filename will be used.
            - format (:obj:`str`): The format of the output file. Options are 'gif' or 'mp4'.
        """
        # At the end of the episode, save the frames.
        if replay_path is None:
            filename = f'game_connect4_{replay_name_suffix}.{format}'
        else:
            filename = f'{replay_path}.{format}'

        if format == 'gif':
            # Save frames as a GIF with a duration of 1000 milliseconds per frame.
            imageio.mimsave(filename, self.frames, 'GIF', duration=1000)
        elif format == 'mp4':
            # Save frames as an MP4 video with a frame rate of 30 frames per second.
            imageio.mimsave(filename, self.frames, fps=30, codec='mpeg4')

        else:
            raise ValueError("Unsupported format: {}".format(format))
        logging.info("Saved output to {}".format(filename))
        self.frames = []

    def get_done_winner(self):
        """
        Overview:
            Check if the game is done and find the winner.
        Returns:
            - outputs (:obj:`Tuple`): Tuple containing 'done' and 'winner',
                - if player 1 win,     'done' = True, 'winner' = 1
                - if player 2 win,     'done' = True, 'winner' = 2
                - if draw,             'done' = True, 'winner' = -1
                - if game is not over, 'done' = False,'winner' = -1
        """
        board = copy.deepcopy(np.array(self.board)).reshape(6, 7)
        for piece in [1, 2]:
            # Check horizontal locations for win
            column_count = 7
            row_count = 6

            for c in range(column_count - 3):
                for r in range(row_count):
                    if (
                            board[r][c] == piece
                            and board[r][c + 1] == piece
                            and board[r][c + 2] == piece
                            and board[r][c + 3] == piece
                    ):
                        return True, piece

            # Check vertical locations for win
            for c in range(column_count):
                for r in range(row_count - 3):
                    if (
                            board[r][c] == piece
                            and board[r + 1][c] == piece
                            and board[r + 2][c] == piece
                            and board[r + 3][c] == piece
                    ):
                        return True, piece

            # Check positively sloped diagonals
            for c in range(column_count - 3):
                for r in range(row_count - 3):
                    if (
                            board[r][c] == piece
                            and board[r + 1][c + 1] == piece
                            and board[r + 2][c + 2] == piece
                            and board[r + 3][c + 3] == piece
                    ):
                        return True, piece

            # Check negatively sloped diagonals
            for c in range(column_count - 3):
                for r in range(3, row_count):
                    if (
                            board[r][c] == piece
                            and board[r - 1][c + 1] == piece
                            and board[r - 2][c + 2] == piece
                            and board[r - 3][c + 3] == piece
                    ):
                        return True, piece

        if all(x in [1, 2] for x in self.board):
            return True, -1

        return False, -1

    def get_done_reward(self):
        """
        Overview:
             Check if the game is over and what is the reward in the perspective of player 1.\
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

    def random_action(self):
        action_list = self.legal_actions
        return np.random.choice(action_list)

    def bot_action(self):
        if np.random.rand() < self.prob_random_action_in_bot:
            return self.random_action()
        else:
            if self.bot_action_type == 'rule':
                return self.rule_bot.get_rule_bot_action(self.board, self._current_player)
            elif self.bot_action_type == 'mcts':
                return self.mcts_bot.get_actions(self.board, player_index=self.current_player_index)

    def action_to_string(self, action):
        """
        Overview:
            Convert an action number to a string representing the action.
        Arguments:
            - action: an integer from the action space.
        Returns:
            - String representing the action.
        """
        return f"Play column {action + 1}"

    def human_to_action(self):
        """
        Overview:
            For multiplayer games, ask the user for a legal action \
            and return the corresponding action number.
        Returns:
            An integer from the action space.
        """
        print(np.array(self.board).reshape(6, 7))
        while True:
            try:
                column = int(
                    input(
                        f"Enter the column to play for the player {self.current_player}: "
                    )
                )
                action = column - 1
                if action in self.legal_actions:
                    break
                else:
                    print("Wrong input, try again")
            except KeyboardInterrupt:
                print("exit")
                sys.exit(0)
            except Exception as e:
                print("Wrong input, try again")
        return action

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def __repr__(self) -> str:
        return "LightZero Connect4 Env"

    @property
    def current_player(self):
        return self._current_player

    @property
    def current_player_index(self):
        """
        Overview:
            current_player_index = 0, current_player = 1 \
            current_player_index = 1, current_player = 2
        """
        return 0 if self._current_player == 1 else 1

    @property
    def next_player(self):
        return self.players[0] if self._current_player == self.players[1] else self.players[1]

    @property
    def observation_space(self) -> spaces.Space:
        return self._observation_space

    @property
    def action_space(self) -> spaces.Space:
        return self._action_space

    @property
    def reward_space(self) -> spaces.Space:
        return self._reward_space

    def simulate_action(self, action):
        """
        Overview:
            execute action and get next_simulator_env. used in AlphaZero.
        Arguments:
            - action: an integer from the action space.
        Returns:
            - next_simulator_env: next simulator env after execute action.
        """
        if action not in self.legal_actions:
            raise ValueError("action {0} on board {1} is not legal".format(action, self.board))
        new_board = copy.deepcopy(self.board)
        piece = self.players.index(self._current_player) + 1
        for i in list(filter(lambda x: x % 7 == action, list(range(41, -1, -1)))):
            if new_board[i] == 0:
                new_board[i] = piece
                break
        if self.start_player_index == 0:
            start_player_index = 1  # self.players = [1, 2], start_player = 2, start_player_index = 1
        else:
            start_player_index = 0  # self.players = [1, 2], start_player = 1, start_player_index = 0
        next_simulator_env = copy.deepcopy(self)
        next_simulator_env.reset(start_player_index, init_state=new_board)
        return next_simulator_env

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

    def close(self):
        pass

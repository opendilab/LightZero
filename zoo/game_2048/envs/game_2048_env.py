import copy
import itertools
import logging
import sys
from typing import List

import gym
import imageio
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ding.envs import BaseEnvTimestep
from ding.torch_utils import to_ndarray
from ding.utils import ENV_REGISTRY
from easydict import EasyDict
from gym import spaces
from gym.utils import seeding


@ENV_REGISTRY.register('game_2048')
class Game2048Env(gym.Env):
    config = dict(
        env_name="game_2048",
        save_replay=False,
        replay_format='gif',
        replay_name_suffix='eval',
        replay_path=None,
        render_real_time=False,
        act_scale=True,
        channel_last=True,
        obs_type='raw_observation',  # options=['raw_observation', 'dict_observation', 'array']
        reward_normalize=False,
        reward_norm_scale=100,
        reward_type='raw',  # 'merged_tiles_plus_log_max_tile_num'
        max_tile=int(2 ** 16),  # 2**11=2048, 2**16=65536
        delay_reward_step=0,
        prob_random_agent=0.,
        max_episode_steps=int(1e6),
        is_collect=True,
        ignore_legal_actions=True,
        need_flatten=False,
        num_of_possible_chance_tile=2,
        possible_tiles=np.array([2, 4]),
        tile_probabilities=np.array([0.9, 0.1]),
    )
    metadata = {'render.modes': ['human', 'rgb_array_render']}

    @classmethod
    def default_config(cls: type) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        self._init_flag = False
        self._env_name = cfg.env_name
        self.replay_format = cfg.replay_format
        self.replay_name_suffix = cfg.replay_name_suffix
        self.replay_path = cfg.replay_path
        self.save_replay = cfg.save_replay
        self.render_real_time = cfg.render_real_time

        self._save_replay_count = 0
        self.channel_last = cfg.channel_last
        self.obs_type = cfg.obs_type
        self.reward_type = cfg.reward_type
        self.reward_normalize = cfg.reward_normalize
        self.reward_norm_scale = cfg.reward_norm_scale
        assert self.reward_type in ['raw', 'merged_tiles_plus_log_max_tile_num']
        assert self.reward_type == 'raw' or (
                self.reward_type == 'merged_tiles_plus_log_max_tile_num' and self.reward_normalize == False)
        self.max_tile = cfg.max_tile
        # Define the maximum tile that will end the game (e.g. 2048). None means no limit.
        # This does not affect the state returned.
        assert self.max_tile is None or isinstance(self.max_tile, int)

        self.max_episode_steps = cfg.max_episode_steps
        self.is_collect = cfg.is_collect
        self.ignore_legal_actions = cfg.ignore_legal_actions
        self.need_flatten = cfg.need_flatten
        self.chance = 0
        self.chance_space_size = 16  # 32 for 2 and 4, 16 for 2
        self.max_tile_num = 0
        self.size = 4
        self.w = self.size
        self.h = self.size
        self.squares = self.size * self.size
        self.episode_return = 0
        # Members for gym implementation:
        self._action_space = spaces.Discrete(4)
        self._observation_space = spaces.Box(0, 1, (self.w, self.h, self.squares), dtype=int)
        self._reward_range = (0., self.max_tile)
        self.set_illegal_move_reward(0.)

        # for render
        self.grid_size = 70
        # Initialise the random seed of the gym environment.
        self.seed()
        self.frames = []
        self.num_of_possible_chance_tile = cfg.num_of_possible_chance_tile
        self.possible_tiles = cfg.possible_tiles
        self.tile_probabilities = cfg.tile_probabilities
        if self.num_of_possible_chance_tile > 2:
            self.possible_tiles = np.array([2**(i+1) for i in range(self.num_of_possible_chance_tile)])
            self.tile_probabilities = np.array([1/self.num_of_possible_chance_tile for _ in range(self.num_of_possible_chance_tile)])
            assert self.possible_tiles.shape[0] == self.tile_probabilities.shape[0]
            assert np.sum(self.tile_probabilities) == 1

    def reset(self):
        """Reset the game board-matrix and add 2 tiles."""
        self.episode_length = 0
        self.board = np.zeros((self.h, self.w), np.int32)
        self.episode_return = 0
        self._final_eval_reward = 0.0
        self.should_done = False

        logging.debug("Adding tiles")
        # TODO(pu): why add_tiles twice?
        if self.num_of_possible_chance_tile > 2:
            self.add_random_tile(self.possible_tiles, self.tile_probabilities)
            self.add_random_tile(self.possible_tiles, self.tile_probabilities)
        elif self.num_of_possible_chance_tile == 2:
            self.add_random_2_4_tile()
            self.add_random_2_4_tile()

        action_mask = np.zeros(4, 'int8')
        action_mask[self.legal_actions] = 1

        observation = encode_board(self.board)
        observation = observation.astype(np.float32)
        assert observation.shape == (4, 4, 16)

        if not self.channel_last:
            # move channel dim to fist axis
            # (W, H, C) -> (C, W, H)
            # e.g. (4, 4, 16) -> (16, 4, 4)
            observation = np.transpose(observation, [2, 0, 1])
        if self.need_flatten:
            observation = observation.reshape(-1)

        if self.obs_type == 'dict_observation':
            observation = {'observation': observation, 'action_mask': action_mask, 'to_play': -1, 'chance': self.chance}
        elif self.obs_type == 'array':
            observation = self.board
        else:
            observation = observation
        if self.save_replay:
            self.render(mode='rgb_array_render')
        return observation

    def step(self, action):
        """Perform one step of the game. This involves moving and adding a new tile."""
        self.episode_length += 1

        if action not in self.legal_actions:
            logging.warning(
                f"You input illegal action: {action}, the legal_actions are {self.legal_actions}. "
                f"Now we randomly choice a action from self.legal_actions."
            )
            action = np.random.choice(self.legal_actions)

        if self.reward_type == 'merged_tiles_plus_log_max_tile_num':
            empty_num1 = len(self.get_empty_location())
        raw_reward = float(self.move(action))
        if self.reward_type == 'merged_tiles_plus_log_max_tile_num':
            empty_num2 = len(self.get_empty_location())
            num_of_merged_tiles = float(empty_num2 - empty_num1)
            reward_merged_tiles_plus_log_max_tile_num = num_of_merged_tiles
            max_tile_num = self.highest()
            if max_tile_num > self.max_tile_num:
                reward_merged_tiles_plus_log_max_tile_num += np.log2(max_tile_num) * 0.1
                self.max_tile_num = max_tile_num

        self.episode_return += raw_reward
        assert raw_reward <= 2 ** (self.w * self.h)
        if self.num_of_possible_chance_tile > 2:
            self.add_random_tile(self.possible_tiles, self.tile_probabilities)
        elif self.num_of_possible_chance_tile == 2:
            self.add_random_2_4_tile()
        done = self.is_end()
        if self.reward_type == 'merged_tiles_plus_log_max_tile_num':
            reward_merged_tiles_plus_log_max_tile_num = float(reward_merged_tiles_plus_log_max_tile_num)
        elif self.reward_type == 'raw':
            raw_reward = float(raw_reward)

        if self.episode_length >= self.max_episode_steps:
            # print("episode_length: {}".format(self.episode_length))
            done = True

        observation = encode_board(self.board)
        observation = observation.astype(np.float32)

        assert observation.shape == (4, 4, 16)

        if not self.channel_last:
            # move channel dim to fist axis
            # (W, H, C) -> (C, W, H)
            # e.g. (4, 4, 16) -> (16, 4, 4)
            observation = np.transpose(observation, [2, 0, 1])

        if self.need_flatten:
            observation = observation.reshape(-1)
        action_mask = np.zeros(4, 'int8')
        action_mask[self.legal_actions] = 1

        if self.obs_type == 'dict_observation':
            observation = {'observation': observation, 'action_mask': action_mask, 'to_play': -1, 'chance': self.chance}
        elif self.obs_type == 'array':
            observation = self.board
        else:
            observation = observation

        if self.reward_normalize:
            reward_normalize = raw_reward / self.reward_norm_scale
            reward = reward_normalize
        else:
            reward = raw_reward

        self._final_eval_reward += raw_reward

        if self.reward_type == 'merged_tiles_plus_log_max_tile_num':
            reward = to_ndarray([reward_merged_tiles_plus_log_max_tile_num]).astype(np.float32)
        elif self.reward_type == 'raw':
            reward = to_ndarray([reward]).astype(np.float32)
        info = {"raw_reward": raw_reward, "current_max_tile_num": self.highest()}

        if self.save_replay:
            self.render(mode='rgb_array_render')

        if done:
            info['eval_episode_return'] = self._final_eval_reward
            if self.save_replay:
                self.save_render_output(replay_name_suffix=self.replay_name_suffix, replay_path=self.replay_path, format=self.replay_format)

        return BaseEnvTimestep(observation, reward, done, info)

    def move(self, direction, trial=False):
        """
        Overview:
            Perform one move of the game. Shift things to one side then,
            combine. directions 0, 1, 2, 3 are up, right, down, left.
            Returns the reward that [would have] got.
        Arguments:
            - direction (:obj:`int`): The direction to move.
            - trial (:obj:`bool`): Whether this is a trial move.
        """
        if not trial:
            if direction == 0:
                logging.debug("Up")
            elif direction == 1:
                logging.debug("Right")
            elif direction == 2:
                logging.debug("Down")
            elif direction == 3:
                logging.debug("Left")

        changed = False
        move_reward = 0
        dir_div_two = int(direction / 2)
        dir_mod_two = int(direction % 2)
        # 0 for towards up or left, 1 for towards bottom or right
        shift_direction = dir_mod_two ^ dir_div_two

        # Construct a range for extracting row/column into a list
        rx = list(range(self.w))
        ry = list(range(self.h))

        if dir_mod_two == 0:
            # Up or down, split into columns
            for y in range(self.h):
                old = [self.board[x, y] for x in rx]
                (new, ms) = self.shift(old, shift_direction)
                move_reward += ms
                if old != new:
                    changed = True
                    if not trial:
                        for x in rx:
                            self.board[x, y] = new[x]

        else:
            # Left or right, split into rows
            for x in range(self.w):
                old = [self.board[x, y] for y in ry]
                (new, ms) = self.shift(old, shift_direction)
                move_reward += ms
                if old != new:
                    changed = True
                    if not trial:
                        for y in ry:
                            self.board[x, y] = new[y]

        # TODO(pu): different transition dynamics
        # if not changed:
        #     raise IllegalMove

        return move_reward

    def set_illegal_move_reward(self, reward):
        """
        Define the reward/penalty for performing an illegal move. Also need to update the reward range for this.
        """
        # Guess that the maximum reward is also 2**squares though you'll probably never get that.
        # (assume that illegal move reward is the lowest value that can be returned
        # TODO: check that this is correct
        self.illegal_move_reward = reward
        self.reward_range = (self.illegal_move_reward, float(2 ** self.squares))

    def render(self, mode='human'):
        if mode == 'rgb_array_render':
            grey = (128, 128, 128)
            grid_size = self.grid_size

            # Render with Pillow
            pil_board = Image.new("RGB", (grid_size * 4, grid_size * 4))
            draw = ImageDraw.Draw(pil_board)
            draw.rectangle([0, 0, 4 * grid_size, 4 * grid_size], grey)
            fnt_path = fm.findfont(fm.FontProperties(family='DejaVu Sans'))
            fnt = ImageFont.truetype(fnt_path, 30)

            for y in range(4):
                for x in range(4):
                    o = self.board[y, x]
                    if o:
                        self.draw_tile(draw, x, y, o, fnt)

            # Instead of returning the image, we display it using pyplot
            if self.render_real_time:
                plt.imshow(np.asarray(pil_board))
                plt.draw()
                # plt.pause(0.001)
            # Append the frame to frames for gif
            self.frames.append(np.asarray(pil_board))
        elif mode == 'human':
            s = 'Current Return: {}, '.format(self.episode_return)
            s += 'Current Highest Tile number: {}\n'.format(self.highest())
            npa = np.array(self.board)
            grid = npa.reshape((self.size, self.size))
            s += "{}\n".format(grid)
            sys.stdout.write(s)
            return sys.stdout

    def draw_tile(self, draw, x, y, o, fnt):
        grid_size = self.grid_size
        white = (255, 255, 255)
        tile_colour_map = {
            0: (204, 192, 179),
            2: (238, 228, 218),
            4: (237, 224, 200),
            8: (242, 177, 121),
            16: (245, 149, 99),
            32: (246, 124, 95),
            64: (246, 94, 59),
            128: (237, 207, 114),
            256: (237, 204, 97),
            512: (237, 200, 80),
            1024: (237, 197, 63),
            2048: (237, 194, 46),
            4096: (237, 194, 46),
            8192: (237, 194, 46),
            16384: (237, 194, 46),
        }
        if o:
            draw.rectangle([x * grid_size, y * grid_size, (x + 1) * grid_size, (y + 1) * grid_size],
                           tile_colour_map[o])
            bbox = draw.textbbox((x, y), str(o), font=fnt)
            text_x_size, text_y_size = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.text((x * grid_size + (grid_size - text_x_size) // 2,
                       y * grid_size + (grid_size - text_y_size) // 2), str(o), font=fnt, fill=white)
            # assert text_x_size < grid_size
            # assert text_y_size < grid_size

    def save_render_output(self, replay_name_suffix: str = '', replay_path=None, format='gif'):
        # At the end of the episode, save the frames
        if replay_path is None:
            filename = f'game_2048_{replay_name_suffix}.{format}'
        else:
            filename = f'{replay_path}.{format}'

        if format == 'gif':
            imageio.mimsave(filename, self.frames, 'GIF')
        elif format == 'mp4':
            imageio.mimsave(filename, self.frames, fps=30, codec='mpeg4')

        else:
            raise ValueError("Unsupported format: {}".format(format))

        logging.info("Saved output to {}".format(filename))
        self.frames = []

    # Implementation of game logic for 2048
    def add_random_2_4_tile(self):
        """Add a tile with value 2 or 4 with different probabilities."""
        possible_tiles = np.array([2, 4])
        tile_probabilities = np.array([0.9, 0.1])
        tile_val = self.np_random.choice(possible_tiles, 1, p=tile_probabilities)[0]
        empty_location = self.get_empty_location()
        if empty_location.shape[0] == 0:
            self.should_done = True
            return
        empty_idx = self.np_random.choice(empty_location.shape[0])
        empty = empty_location[empty_idx]
        logging.debug("Adding %s at %s", tile_val, (empty[0], empty[1]))

        # set the chance outcome
        if self.chance_space_size == 16:
            self.chance = 4 * empty[0] + empty[1]
        elif self.chance_space_size == 32:
            if tile_val == 2:
                self.chance = 4 * empty[0] + empty[1]
            elif tile_val == 4:
                self.chance = 16 + 4 * empty[0] + empty[1]

        self.board[empty[0], empty[1]] = tile_val

    def add_random_tile(self, possible_tiles: np.array = np.array([2, 4]), tile_probabilities: np.array = np.array([0.9, 0.1])):
        """Add a tile with a value from possible_tiles array according to given probabilities."""
        if len(possible_tiles) != len(tile_probabilities):
            raise ValueError("Length of possible_tiles and tile_probabilities must be the same")
        if np.sum(tile_probabilities) != 1:
            raise ValueError("Sum of tile_probabilities must be 1")

        tile_val = self.np_random.choice(possible_tiles, 1, p=tile_probabilities)[0]
        tile_idx = np.where(possible_tiles == tile_val)[0][0]  # get the index of the tile value
        empty_location = self.get_empty_location()
        if empty_location.shape[0] == 0:
            self.should_done = True
            return
        empty_idx = self.np_random.choice(empty_location.shape[0])
        empty = empty_location[empty_idx]
        logging.debug("Adding %s at %s", tile_val, (empty[0], empty[1]))

        # set the chance outcome
        self.chance_space_size = len(possible_tiles) * 16  # assuming a 4x4 board
        self.chance = tile_idx * 16 + 4 * empty[0] + empty[1]

        self.board[empty[0], empty[1]] = tile_val
    def get_empty_location(self):
        """Return a 2d numpy array with the location of empty squares."""
        return np.argwhere(self.board == 0)

    def highest(self):
        """Report the highest tile on the board."""
        return np.max(self.board)

    @property
    def legal_actions(self):
        """
        Overview:
            Return the legal actions for the current state.
        Arguments:
            - None
        Returns:
            - legal_actions (:obj:`list`): The legal actions.
        """
        if self.ignore_legal_actions:
            return [0, 1, 2, 3]
        legal_actions = []
        for direction in range(4):
            changed = False
            move_reward = 0
            dir_div_two = int(direction / 2)
            dir_mod_two = int(direction % 2)
            # 0 for towards up or left, 1 for towards bottom or right
            shift_direction = dir_mod_two ^ dir_div_two

            # Construct a range for extracting row/column into a list
            rx = list(range(self.w))
            ry = list(range(self.h))

            if dir_mod_two == 0:
                # Up or down, split into columns
                for y in range(self.h):
                    old = [self.board[x, y] for x in rx]
                    (new, move_reward_tmp) = self.shift(old, shift_direction)
                    move_reward += move_reward_tmp
                    if old != new:
                        changed = True
            else:
                # Left or right, split into rows
                for x in range(self.w):
                    old = [self.board[x, y] for y in ry]
                    (new, move_reward_tmp) = self.shift(old, shift_direction)
                    move_reward += move_reward_tmp
                    if old != new:
                        changed = True

            if changed:
                legal_actions.append(direction)

        return legal_actions

    def combine(self, shifted_row):
        """
        Overview:
            Combine same tiles when moving to one side. This function always
            shifts towards the left. Also count the reward of combined tiles.
        """
        move_reward = 0
        combined_row = [0] * self.size
        skip = False
        output_index = 0
        for p in pairwise(shifted_row):
            if skip:
                skip = False
                continue
            combined_row[output_index] = p[0]
            if p[0] == p[1]:
                combined_row[output_index] += p[1]
                move_reward += p[0] + p[1]
                # Skip the next thing in the list.
                skip = True
            output_index += 1
        if shifted_row and not skip:
            combined_row[output_index] = shifted_row[-1]

        return combined_row, move_reward

    def shift(self, row, direction):
        """Shift one row left (direction == 0) or right (direction == 1), combining if required."""
        length = len(row)
        assert length == self.size
        # assert direction == 0 or direction == 1

        # Shift all non-zero digits up
        shifted_row = [i for i in row if i != 0]

        # Reverse list to handle shifting to the right
        if direction:
            shifted_row.reverse()

        (combined_row, move_reward) = self.combine(shifted_row)

        # Reverse list to handle shifting to the right
        if direction:
            combined_row.reverse()

        assert len(combined_row) == self.size
        return combined_row, move_reward

    def is_end(self):
        """Has the game ended. Game ends if there is a tile equal to the limit
           or there are no legal moves. If there are empty spaces then there
           must be legal moves."""

        if self.max_tile is not None and self.highest() == self.max_tile:
            return True
        elif len(self.legal_actions) == 0:
            # the agent don't have legal_actions to move, so the episode is done
            return True
        elif self.should_done:
            return True
        else:
            return False

    def get_board(self):
        """Get the whole board-matrix, useful for testing."""
        return self.board

    def set_board(self, new_board):
        """Set the whole board-matrix, useful for testing."""
        self.board = new_board

    def seed(self, seed=None, seed1=None):
        """Set the random seed for the gym environment."""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def random_action(self) -> np.ndarray:
        random_action = self.action_space.sample()
        if isinstance(random_action, np.ndarray):
            pass
        elif isinstance(random_action, int):
            random_action = to_ndarray([random_action], dtype=np.int64)
        return random_action

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        return self._action_space

    @property
    def reward_space(self) -> gym.spaces.Space:
        return self._reward_range

    @staticmethod
    def create_collector_env_cfg(cfg: dict) -> List[dict]:
        collector_env_num = cfg.pop('collector_env_num')
        cfg = copy.deepcopy(cfg)
        # when collect data, sometimes we need to normalize the reward
        # reward_normalize is determined by the config.
        cfg.is_collect = True
        return [cfg for _ in range(collector_env_num)]

    @staticmethod
    def create_evaluator_env_cfg(cfg: dict) -> List[dict]:
        evaluator_env_num = cfg.pop('evaluator_env_num')
        cfg = copy.deepcopy(cfg)
        # when evaluate, we don't need to normalize the reward.
        cfg.reward_normalize = False
        cfg.is_collect = False
        return [cfg for _ in range(evaluator_env_num)]

    def __repr__(self) -> str:
        return "LightZero 2048 Env."


def encode_board(flat_board, num_of_template_tiles=16):
    """
    Overview:
        This function converts a [4, 4] raw game board into a [4, 4, num_of_template_tiles] one-hot encoded board.
    Arguments:
        - flat_board (:obj:`np.ndarray`): The raw game board, expected to be a 2D numpy array.
        - num_of_template_tiles (:obj:`int`): The number of unique tiles to consider in the encoding,
                                               default value is 16.
    Returns:
        - one_hot_board (:obj:`np.ndarray`): The one-hot encoded game board.
    """
    # Generate a sequence of powers of 2, corresponding to the unique tile values.
    # In the game, tile values are powers of 2. So, each unique tile is represented by 2 raised to some power.
    # The first tile is considered as 0 (empty tile).
    tile_values = 2 ** np.arange(num_of_template_tiles, dtype=int)
    tile_values[0] = 0  # The first tile represents an empty slot, so set its value to 0.

    # Create a 3D array from the 2D input board by repeating it along a new axis.
    # This creates a 'layered' view of the board, where each layer corresponds to one unique tile value.
    layered_board = np.repeat(flat_board[:, :, np.newaxis], num_of_template_tiles, axis=-1)

    # Perform the one-hot encoding:
    # For each layer of the 'layered_board', mark the positions where the tile value in the 'flat_board'
    # matches the corresponding value in 'tile_values'. If a match is found, mark it as 1 (True), else 0 (False).
    one_hot_board = (layered_board == tile_values).astype(int)

    return one_hot_board


def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


class IllegalMove(Exception):
    pass


class IllegalActionError(Exception):
    pass

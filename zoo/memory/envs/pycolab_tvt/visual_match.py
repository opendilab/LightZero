# pylint: disable=g-bad-file-header
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Visual match task.

The game is split up into three phases:
1. (exploration phase) player is in one room and there's a colour in the other,
2. (distractor phase) player is collecting apples,
3. (reward phase) player sees three doors of different colours and has to select
    the one of the same color as the colour in the first phase.
"""

from pycolab import ascii_art
from pycolab import storytelling

from zoo.memory.envs.pycolab_tvt import common, game, objects


SYMBOLS_TO_SHUFFLE = ["b", "c", "e"]

EXPLORE_GRID = [
    "  ppppppp  ",
    "  p     p  ",
    "  p     p  ",
    "  pp   pp  ",
    "  p+++++p  ",
    "  p+++++p  ",
    "  ppppppp  ",
]
PASSIVE_EXPLORE_GRID = [
    "           ",
    "    ppp    ",
    "    p+p    ",
    "    ppp    ",
    "           ",
    "           ",
    "           ",
]

REWARD_GRID = [
    "###########",
    "# b  c  e #",
    "#         #",
    "#         #",
    "####   ####",
    "   # + #   ",
    "   #####   ",
]

# MAX_FRAMES_PER_PHASE = {"explore": 15, "distractor": 30, "reward": 15}
MAX_FRAMES_PER_PHASE = {"explore": 2, "distractor": 0, "reward": 15}


class Game(game.AbstractGame):
    """Image Match Passive Game."""

    def __init__(
        self,
        rng,
        num_apples=10,
        # apple_reward=(1, 10),
        apple_reward=(0, 0),
        fix_apple_reward_in_episode=False,
        final_reward=10.0,
        respawn_every=common.DEFAULT_APPLE_RESPAWN_TIME,
        crop=True,
        max_frames=MAX_FRAMES_PER_PHASE,
        EXPLORE_GRID=PASSIVE_EXPLORE_GRID,
    ):
        self._rng = rng
        self._num_apples = num_apples
        self._apple_reward = apple_reward
        self._fix_apple_reward_in_episode = fix_apple_reward_in_episode
        self._final_reward = final_reward
        self._max_frames = max_frames
        self._crop = crop
        self._respawn_every = respawn_every
        self._EXPLORE_GRID = EXPLORE_GRID
        self._episode_length = sum(self._max_frames.values())
        self._num_actions = common.NUM_ACTIONS
        self._colours = common.FIXED_COLOURS.copy()
        shuffled_symbol_colour_map = common.get_shuffled_symbol_colour_map(rng, SYMBOLS_TO_SHUFFLE) # TODO：b c e （分别对应左 中 右位置） 的颜色随机
        # shuffled_symbol_colour_map = {'b': (0, 0, 1000), 'c': (1000, 0, 0), 'e': (0, 1000, 0)}   # TODO：phase3-fixed-colormap-bce b c e （分别对应左 中 右位置） 的颜色固定为：蓝色 红色 绿色
        print(f'shuffled_symbol_colour_map: {shuffled_symbol_colour_map}')
        self._colours.update(
            shuffled_symbol_colour_map
        )
        self._extra_observation_fields = ["chapter_reward_as_string"]

    @property
    def extra_observation_fields(self):
        """The field names of extra observations."""
        return self._extra_observation_fields

    @property
    def num_actions(self):
        """Number of possible actions in the game."""
        return self._num_actions

    @property
    def episode_length(self):
        return self._episode_length

    @property
    def colours(self):
        """Symbol to colour map for key to door."""
        return self._colours

    def _make_explore_phase(self, target_char):
        # Keep only one coloured position and one player position.
        grid = common.keep_n_characters_in_grid(
            self._EXPLORE_GRID, "p", 1, common.BORDER
        )  # keeps only 1 p, and replaces the rest with common.BORDER
        grid = common.keep_n_characters_in_grid(
            grid, "p", 0, target_char
        )  # removes p and replaces it with the appropriate color
        grid = common.keep_n_characters_in_grid(grid, common.PLAYER, 1)

        return ascii_art.ascii_art_to_game(
            grid,
            what_lies_beneath=" ",
            sprites={
                common.PLAYER: ascii_art.Partial(
                    common.PlayerSprite, impassable=common.BORDER + target_char
                ),
                target_char: objects.ObjectSprite,
                common.TIMER: ascii_art.Partial(
                    common.TimerSprite, self._max_frames["explore"]
                ),
            },
            update_schedule=[common.PLAYER, target_char, common.TIMER],
            z_order=[target_char, common.PLAYER, common.TIMER],
        )

    def _make_distractor_phase(self):
        return common.distractor_phase(
            player_sprite=common.PlayerSprite,
            num_apples=self._num_apples,
            max_frames=self._max_frames["distractor"],
            apple_reward=self._apple_reward,
            fix_apple_reward_in_episode=self._fix_apple_reward_in_episode,
            respawn_every=self._respawn_every,
        )

    def _make_reward_phase(self, target_char):
        return ascii_art.ascii_art_to_game(
            REWARD_GRID,
            what_lies_beneath=" ",
            sprites={
                common.PLAYER: common.PlayerSprite,
                "b": objects.ObjectSprite,
                "c": objects.ObjectSprite,
                "e": objects.ObjectSprite,
                common.TIMER: ascii_art.Partial(
                    common.TimerSprite,
                    self._max_frames["reward"],
                    track_chapter_reward=True,
                ),
                target_char: ascii_art.Partial(
                    objects.ObjectSprite, reward=self._final_reward
                ),
            },
            update_schedule=[common.PLAYER, "b", "c", "e", common.TIMER],
            z_order=[common.PLAYER, "b", "c", "e", common.TIMER],
        )

    def make_episode(self):
        """Factory method for generating new episodes of the game."""
        if self._crop:
            croppers = common.get_cropper()
        else:
            croppers = None
        target_char = self._rng.choice(SYMBOLS_TO_SHUFFLE)  # TODO：随机目标颜色
        # target_char = 'b'  # TODO：固定目标颜色为左上角位置的颜色
        print(f"self._rng: {self._rng}")
        print(f"symbols_to_shuffle: {SYMBOLS_TO_SHUFFLE}")
        print(f"target_char: {target_char}")
        return storytelling.Story(
            [
                lambda: self._make_explore_phase(target_char),
                self._make_distractor_phase,
                lambda: self._make_reward_phase(target_char),
            ],
            croppers=croppers,
        )

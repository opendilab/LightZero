# Lint as: python2, python3
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
"""Pycolab env."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from pycolab import rendering

from envs.key_to_door import key_to_door, common, visual_match
from tensorflow import nest


class PycolabEnvironment(object):
    """A simple environment adapter for pycolab games."""

    def __init__(
        self,
        game,
        num_apples=10,
        apple_reward=1.0,
        fix_apple_reward_in_episode=False,
        final_reward=10.0,
        respawn_every=20,
        crop=True,
        default_reward=0,
        REWARD_GRID=key_to_door.REWARD_GRID_SR,  # for key to door
        EXPLORE_GRID=key_to_door.EXPLORE_GRID,  # for visual match
        max_frames=key_to_door.MAX_FRAMES_PER_PHASE_SR,
    ):
        """Construct a `environment.Base` adapter that wraps a pycolab game."""
        rng = np.random.RandomState()
        if game == "key_to_door":
            self._game = key_to_door.Game(
                rng=rng,
                num_apples=num_apples,
                apple_reward=apple_reward,
                fix_apple_reward_in_episode=fix_apple_reward_in_episode,
                final_reward=final_reward,
                respawn_every=respawn_every,
                crop=crop,
                REWARD_GRID=REWARD_GRID,
                max_frames=max_frames,
            )
        elif game == "visual_match":
            self._game = visual_match.Game(
                rng=rng,
                num_apples=num_apples,
                apple_reward=apple_reward,
                fix_apple_reward_in_episode=fix_apple_reward_in_episode,
                final_reward=final_reward,
                respawn_every=respawn_every,
                crop=crop,
                EXPLORE_GRID=EXPLORE_GRID,
                max_frames=max_frames,
            )
        else:
            raise ValueError('Unsupported game "%s".' % game)
        self._default_reward = default_reward

        self._num_actions = self._game.num_actions

        # Agents expect HWC uint8 observations, Pycolab uses CHW float observations.
        colours = nest.map_structure(
            lambda c: float(c) * 255 / 1000, self._game.colours
        )
        self._rgb_converter = rendering.ObservationToArray(
            value_mapping=colours, permute=(1, 2, 0), dtype=np.uint8
        )

        episode = self._game.make_episode()
        observation, _, _ = episode.its_showtime()
        self._image_shape = self._rgb_converter(observation).shape

    def _process_outputs(self, observation, reward):
        if reward is None:
            reward = self._default_reward
        image = self._rgb_converter(observation)
        return image, reward

    def reset(self):
        """Start a new episode."""
        self._episode = self._game.make_episode()
        observation, reward, _ = self._episode.its_showtime()
        return self._process_outputs(observation, reward)

    def step(self, action):
        """Take step in episode."""
        observation, reward, _ = self._episode.play(action)
        return self._process_outputs(observation, reward)

    @property
    def num_actions(self):
        return self._num_actions

    @property
    def observation_shape(self):
        return self._image_shape

    @property
    def episode_length(self):
        return self._game.episode_length

    def last_phase_reward(self):
        # In Pycolab games here we only track chapter_reward for final chapter.
        return float(self._episode.the_plot["chapter_reward"])

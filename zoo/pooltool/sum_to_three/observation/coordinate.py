from __future__ import annotations

import numpy as np
from gym import spaces

from numpy.typing import NDArray
import pooltool as pt
from zoo.pooltool.datatypes import State

BALL_DIM = 2

def get_coordinate_obs_space(system: pt.System) -> spaces.Box:
    """
    Overview:
        Generate the observation space based on the ball coordinates.
    Arguments:
        - System (:obj:`pt.System`): A billiards system.
    Returns:
        - space (:obj:`spaces.Box`): The observation space defining the bounds and shape based on the renderer's output.
    """
    table_length = system.table.l
    table_width = system.table.w
    ball_radius = system.balls["cue"].params.R

    xmin, ymin = ball_radius, ball_radius
    xmax, ymax = table_width - ball_radius, table_length - ball_radius

    return spaces.Box(
        low=np.array([xmin, ymin] * len(system.balls), dtype=np.float32),
        high=np.array([xmax, ymax] * len(system.balls), dtype=np.float32),
        shape=(BALL_DIM * len(system.balls),),
        dtype=np.float32,
    )


def _slice(ball_idx: int) -> slice:
    return slice(ball_idx * BALL_DIM, (ball_idx + 1) * BALL_DIM)

def _null_obs(num_balls: int) -> NDArray[np.float32]:
    return np.empty(num_balls * BALL_DIM, dtype=np.float32)

def coordinate_observation_array(state: State) -> NDArray[np.float32]:
    """
    Overview:
        Returns an observation array of the current state.
    Returns:
        - observation (:obj:`NDArray[np.float32]`): A continuous 1D array holding \
            the x- and y- coordinates of the cue ball and the object ball. It has \
            the following entries: ``[x_cue, y_cue, x_obj, y_obj]``. \
            ``x_cue`` and ``y_cue`` are the 2D coordinates of the cue ball, and \
            ``x_obj`` and ``y_obj`` are 2D coordinates of the object ball. The \
            coordinate system is defined by ``self.spaces.observation`` (see \
            :obj:`get_obs_space`).
    """
    obs = _null_obs(len(state.system.balls))
    for ball_idx, ball_id in enumerate(state.system.balls.keys()):
        coords = state.system.balls[ball_id].state.rvw[0, :BALL_DIM]
        obs[_slice(ball_idx)] = coords

    return obs

"""
Overview:
    A module utilities required for the sum to three environment.
"""

from __future__ import annotations
from typing import Dict, Protocol, Tuple
import pooltool as pt
from zoo.pooltool.datatypes import State, Bounds
import numpy as np
from gym import spaces
from numpy.typing import NDArray
from zoo.pooltool.image_representation import PygameRenderer, RenderConfig
from enum import Enum


BALL_DIM = 2


class ObservationType(Enum):
    IMAGE = "image"
    COORDINATE = "coordinate"


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


def get_image_obs_space(render_config: RenderConfig) -> spaces.Box:
    """
    Overview:
        Generate the observation space based on the renderer's configuration.
    Arguments:
        - render_config (:obj:`RenderConfig`): Render config object which contains configuration details including dimensions and planes.
    Returns:
        - space (:obj:`spaces.Box`): The observation space defining the bounds and shape based on the renderer's output.
    """
    return spaces.Box(
        low=0.0,
        high=1.0,
        shape=render_config.observation_shape,
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


def image_observation_array(renderer: PygameRenderer) -> NDArray[np.float32]:
    return renderer.observation()


def binary(state: State) -> float:
    """
    Overview:
        Calculate the reward from the state.
    Returns:
        - reward (:obj:`float`): A reward of 0 or 1. A reward of 1 is returned if the \
            player both **(1)** contacts the object ball with the cue ball and **(2)** \
            the sum of contacted rails by either balls is 3.
    """
    # Count the number of ball-ball collisions
    ball_hits = pt.events.filter_type(state.system.events, pt.EventType.BALL_BALL)

    # Count rails that cue ball hits
    cue_cushion_hits = pt.events.filter_events(
        state.system.events,
        pt.events.by_type(pt.EventType.BALL_LINEAR_CUSHION),
        pt.events.by_ball("cue"),
    )

    # Count rails that object ball hits
    object_cushion_hits = pt.events.filter_events(
        state.system.events,
        pt.events.by_type(pt.EventType.BALL_LINEAR_CUSHION),
        pt.events.by_ball("object"),
    )

    if len(ball_hits) and (len(cue_cushion_hits) + len(object_cushion_hits) == 3):
        return 1.0

    return 0.0


class RewardFunction(Protocol):
    """
    Overview:
        Protocol defining what a reward function call signature should be.
    """

    def __call__(self, state: State) -> float: ...


_reward_functions: Dict[str, Tuple[RewardFunction, Bounds]] = {
    "binary": (binary, Bounds(low=0.0, high=1.0)),
}


def _assert_exists(algorithm: str) -> None:
    if algorithm not in _reward_functions:
        raise AssertionError(
            f"algorithm {algorithm} is unknown. Available algorithms: {_reward_functions.keys()}"
        )


def get_reward_function(algorithm: str) -> RewardFunction:
    """
    Overview:
        Returns a reward function.
    Arguments:
        - algorithm (:obj:`str`): The name of a reward algorithm.
    Returns:
        - reward_fn (:obj:`RewardFunction`): A function that accepts a :obj:`State` and returns a reward.
    """
    _assert_exists(algorithm)
    return _reward_functions[algorithm][0]


def get_reward_bounds(algorithm: str) -> Bounds:
    """
    Overview:
        Returns the bounds of a reward function.
    Arguments:
        - algorithm (:obj:`str`): The name of a reward algorithm.
    Returns:
        - bounds (:obj:`Bounds`): The upper and lower bounds of the reward space.
    """
    _assert_exists(algorithm)
    return _reward_functions[algorithm][1]


def get_reward_space(algorithm: str) -> spaces.Box:
    """
    Overview:
        Determines the reward space based on the reward calculation algorithm.
    Arguments:
        - algorithm (:obj:`str`): The name of the algorithm used to calculate rewards.
    Returns:
        - space (:obj:`spaces.Box`): The reward space for the environment.
    """
    bounds = get_reward_bounds(
        algorithm
    )  # Assumes a function get_reward_bounds exists.
    return spaces.Box(
        low=np.array([bounds.low], dtype=np.float32),
        high=np.array([bounds.high], dtype=np.float32),
        shape=(1,),
        dtype=np.float32,
    )

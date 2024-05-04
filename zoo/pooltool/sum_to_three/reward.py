"""
Overview:
    A module for housing different reward functions for the sum-to-three game.
"""
from typing import Dict, Protocol, Tuple
import pooltool as pt
from zoo.pooltool.datatypes import State, Bounds
import numpy as np
from gym import spaces


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
    bounds = get_reward_bounds(algorithm)  # Assumes a function get_reward_bounds exists.
    return spaces.Box(
        low=np.array([bounds.low], dtype=np.float32),
        high=np.array([bounds.high], dtype=np.float32),
        shape=(1,),
        dtype=np.float32,
    )

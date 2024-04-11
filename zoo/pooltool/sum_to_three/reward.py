from typing import Dict, Protocol, Tuple
import pooltool as pt
from zoo.pooltool.datatypes import State, Bounds


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
    def __call__(self, state: State) -> float: ...


_reward_functions: Dict[str, Tuple[RewardFunction, Bounds]] = {
    "binary": (binary, Bounds(low=0.0, high=1.0)),
}


def get_reward_function(algorithm: str) -> RewardFunction:
    return _reward_functions[algorithm][0]

def get_reward_space(algorithm: str) -> RewardFunction:
    return _reward_functions[algorithm][1]

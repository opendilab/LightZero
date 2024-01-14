import math

import numpy as np
from zoo.pooltool.sum_to_three.envs.sum_to_three_env import (
    ANGLE_BOUNDS,
    BALL_DIM,
    V0_BOUNDS,
    Bounds,
    SumToThreeGym,
    get_action_space,
    get_obs_space,
)

np.random.seed(42)

import pytest


@pytest.fixture
def st3() -> SumToThreeGym:
    return SumToThreeGym.single_player_env()


def test_create_initial_state():
    state = SumToThreeGym.create_initial_state(random_pos=False)

    # Total set of balls equals cue and object
    assert {"cue", "object"} == set(state.system.balls.keys())

    # Cue ball is at starting position
    expected = np.array(
        [
            state.system.table.w / 2,
            state.system.table.l / 4,
            state.system.balls["cue"].params.R,
        ],
        dtype=np.float64,
    )
    assert np.allclose(state.system.balls["cue"].xyz, expected, rtol=1e-3)

    # Object ball is at starting position
    expected = np.array(
        [
            state.system.table.w / 2,
            state.system.table.l / 4 * 3,
            state.system.balls["object"].params.R,
        ],
        dtype=np.float64,
    )
    assert np.allclose(state.system.balls["object"].xyz, expected, rtol=1e-3)


def test_create_initial_state_random():
    state = SumToThreeGym.create_initial_state(random_pos=True)
    length = state.system.table.l
    width = state.system.table.w

    assert state.system.balls["cue"].params.R == state.system.balls["object"].params.R
    R = state.system.balls["cue"].params.R

    # Positions are random, so they don't match the starting positions
    expected = np.array([width / 2, length / 4, R], dtype=np.float64)
    assert not np.allclose(state.system.balls["cue"].xyz, expected, rtol=1e-3)

    # Object ball is at starting position
    expected = np.array([width / 2, length / 4 * 3, R], dtype=np.float64)
    assert not np.allclose(state.system.balls["object"].xyz, expected, rtol=1e-3)

    # Assert balls are within the table bounds
    for _ in range(100):
        state = SumToThreeGym.create_initial_state(random_pos=True)
        cue_ball_pos = state.system.balls["cue"].xyz
        object_ball_pos = state.system.balls["cue"].xyz

        assert cue_ball_pos[0] > R and cue_ball_pos[0] < width - R
        assert cue_ball_pos[1] > R and cue_ball_pos[1] < length - R
        assert object_ball_pos[0] > R and object_ball_pos[0] < width - R
        assert object_ball_pos[1] > R and object_ball_pos[1] < length - R


def test_get_action_space():
    action_space = get_action_space()

    # Default action space
    assert math.isclose(V0_BOUNDS.low, float(action_space.low[0]), rel_tol=1e-3)
    assert math.isclose(V0_BOUNDS.high, float(action_space.high[0]), rel_tol=1e-3)
    assert math.isclose(ANGLE_BOUNDS.low, float(action_space.low[1]), rel_tol=1e-3)
    assert math.isclose(ANGLE_BOUNDS.high, float(action_space.high[1]), rel_tol=1e-3)

    # Custom action space
    custom_V0 = Bounds(0.0, 10.0)
    custom_angle = Bounds(-90.0, 90.0)
    action_space = get_action_space(custom_V0, custom_angle)
    assert math.isclose(custom_V0.low, float(action_space.low[0]), rel_tol=1e-3)
    assert math.isclose(custom_V0.high, float(action_space.high[0]), rel_tol=1e-3)
    assert math.isclose(custom_angle.low, float(action_space.low[1]), rel_tol=1e-3)
    assert math.isclose(custom_angle.high, float(action_space.high[1]), rel_tol=1e-3)


def test_get_obs_space(st3: SumToThreeGym):
    balls = st3.system.balls
    table = st3.system.table
    R = balls["cue"].params.R
    obs = get_obs_space(balls, table)

    # 1D observation space
    assert len(obs.shape) == 1

    # Length 4
    assert obs.shape[0] == BALL_DIM * len(balls) == 4

    for min_val in obs.low:
        # Minimum in x and y is the radius
        assert math.isclose(min_val, R, rel_tol=1e-4)

    for i, max_val in enumerate(obs.high):
        if i % 2 == 0:
            # Maximum in x is the table width minus radius
            assert math.isclose(max_val, table.w - R, rel_tol=1e-4)
        else:
            # Maximum in y is the table width minus radius
            assert math.isclose(max_val, table.l - R, rel_tol=1e-4)

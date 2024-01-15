import math

import numpy as np
from easydict import EasyDict
from zoo.pooltool.sum_to_three.envs.sum_to_three_env import (
    ANGLE_BOUNDS,
    BALL_DIM,
    V0_INIT,
    V0_BOUNDS,
    Bounds,
    SumToThreeGym,
    create_initial_state,
    get_action_space,
    get_obs_space,
)

import pooltool as pt

np.random.seed(42)

import pytest


@pytest.fixture
def st3() -> SumToThreeGym:
    return SumToThreeGym.single_player_env()


def test_create_initial_state():
    state = create_initial_state(random_pos=False)

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
    state = create_initial_state(random_pos=True)
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
        state = create_initial_state(random_pos=True)
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

    # Even numbers index x coordinates, odd numbers index y coordinates
    for i, max_val in enumerate(obs.high):
        if i % 2 == 0:
            # Maximum in x is the table width minus radius
            assert math.isclose(max_val, table.w - R, rel_tol=1e-4)
        else:
            # Maximum in y is the table width minus radius
            assert math.isclose(max_val, table.l - R, rel_tol=1e-4)


def test_observation_array(st3: SumToThreeGym):
    # Get the observation array from the environment
    observation = st3.observation_array()

    # Check if the observation is a 1D numpy array
    assert isinstance(observation, np.ndarray)
    assert observation.ndim == 1

    # Check if the observation length is correct
    assert len(observation) == BALL_DIM * len(st3.system.balls)

    assert math.isclose(observation[0], st3.system.balls["cue"].xyz[0], rel_tol=1e-3)
    assert math.isclose(observation[1], st3.system.balls["cue"].xyz[1], rel_tol=1e-3)
    assert math.isclose(observation[2], st3.system.balls["object"].xyz[0], rel_tol=1e-3)
    assert math.isclose(observation[3], st3.system.balls["object"].xyz[1], rel_tol=1e-3)

    # Now modify system and make sure observation array doesn't change
    st3.system.balls["cue"].state.rvw[0] = [10.0, 10.0, 10.0]  # type: ignore
    st3.system.balls["object"].state.rvw[0] = [10.0, 10.0, 10.0]  # type: ignore
    assert not math.isclose(
        observation[0], st3.system.balls["cue"].xyz[0], rel_tol=1e-3
    )
    assert not math.isclose(
        observation[1], st3.system.balls["cue"].xyz[1], rel_tol=1e-3
    )
    assert not math.isclose(
        observation[2], st3.system.balls["object"].xyz[0], rel_tol=1e-3
    )
    assert not math.isclose(
        observation[3], st3.system.balls["object"].xyz[1], rel_tol=1e-3
    )


def test_set_action(st3: SumToThreeGym):
    # Define a test action
    test_action = np.array([1.5, -30], dtype=np.float32)  # Example values

    # Set the action
    st3.set_action(rescaled_action=test_action)

    # Check if the action was set correctly
    assert math.isclose(st3.system.cue.V0, test_action[0], rel_tol=1e-3)
    assert math.isclose(
        st3.system.cue.phi,
        pt.aim.at_ball(st3.system, "object", cut=test_action[1]),
        rel_tol=1e-3,
    )


def test_reset_single_player_env(st3: SumToThreeGym):
    # Values for later comparison
    R = st3.system.balls["cue"].params.R
    initial_cue_pos = np.array(
        [
            st3.system.table.w / 2,
            st3.system.table.l / 4,
            R,
        ],
        dtype=np.float32,
    )
    initial_object_pos = np.array(
        [
            st3.system.table.w / 2,
            st3.system.table.l * 3 / 4,
            R,
        ],
        dtype=np.float32,
    )

    def matches_initial(env: SumToThreeGym) -> bool:
        if not np.allclose(
            env.system.balls["cue"].state.rvw[0],
            initial_cue_pos,
            atol=1e-3,
        ):
            return False

        if not np.allclose(
            env.system.balls["object"].state.rvw[0],
            initial_object_pos,
            atol=1e-3,
        ):
            return False

        if env.system.cue.V0 != V0_INIT:
            return False

        return True

    assert matches_initial(st3)

    # Simulate a shot
    st3.set_action(rescaled_action=np.array([1.5, -30], dtype=np.float32))
    pt.simulate(st3.system, inplace=True)

    assert not matches_initial(st3)

    # Reset the environment
    st3.reset_single_player_env()

    assert matches_initial(st3)

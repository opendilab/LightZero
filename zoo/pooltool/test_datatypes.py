import numpy as np
import pytest
from zoo.pooltool.datatypes import PoolToolGym


@pytest.fixture
def pooltool_gym() -> PoolToolGym:
    return PoolToolGym.dummy()


def test_scale_action(pooltool_gym: PoolToolGym):
    # Array must be within [-1, 1]
    with pytest.raises(AssertionError):
        pooltool_gym.scale_action(np.array([-1.1, 1], dtype=np.float32))
    with pytest.raises(AssertionError):
        pooltool_gym.scale_action(np.array([1.1, 1], dtype=np.float32))

    # Define an action in the range of [-1, 1]
    test_action = np.array([-1, 1], dtype=np.float32)

    # Scale the action using the method in PoolToolGym
    scaled_action = pooltool_gym.scale_action(test_action)

    # Retrieve the action space from the gym for comparison
    action_space = pooltool_gym.spaces.action

    # Assert that the scaled action is within the bounds of the action space
    assert np.all(scaled_action >= action_space.low)  # type: ignore
    assert np.all(scaled_action <= action_space.high)  # type: ignore

    # Additional checks to ensure scaling is correct
    # Assuming linear scaling, test specific known values
    # For example, if -1 maps to action_space.low and 1 maps to action_space.high
    expected_low_scaled_action = action_space.low  # type: ignore
    expected_high_scaled_action = action_space.high  # type: ignore
    np.testing.assert_allclose(
        scaled_action[0], expected_low_scaled_action[0], atol=1e-6
    )
    np.testing.assert_allclose(
        scaled_action[1], expected_high_scaled_action[1], atol=1e-6
    )


def test_simulate(pooltool_gym: PoolToolGym):
    system_initial = pooltool_gym.system.copy()

    assert not system_initial.simulated
    assert not len(system_initial.events)
    assert system_initial.t == 0.0

    pooltool_gym.simulate()

    system_final = pooltool_gym.system
    assert system_final.simulated
    assert len(system_final.events)
    assert system_final.t != 0.0


def test_seed(pooltool_gym: PoolToolGym):
    pooltool_gym.seed(123)
    value1 = np.random.rand()

    # Reset seed
    np.random.seed(123)
    value2 = np.random.rand()

    # The same seed should produce the same random number
    assert value1 == value2

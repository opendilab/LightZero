from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pytest
from zoo.pooltool.datatypes import PoolToolSimulator, Spaces, State
from gym import spaces


@dataclass
class DummySimulator(PoolToolSimulator):
    def observation_array(self):
        pass

    def set_action(self, rescaled_action):
        pass

    @classmethod
    def instance(cls) -> DummySimulator:
        return cls(
            State.example(),
            Spaces(
                observation=spaces.Box(
                    low=np.array([0.0] * 4, dtype=np.float32),
                    high=np.array([1.0] * 4, dtype=np.float32),
                    shape=(4,),
                    dtype=np.float32,
                ),
                action=spaces.Box(
                    low=np.array([-0.3, 70], dtype=np.float32),
                    high=np.array([-0.3, 70], dtype=np.float32),
                    shape=(2,),
                    dtype=np.float32,
                ),
                reward=spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(1,),
                    dtype=np.float32,
                ),
            ),
        )


@pytest.fixture
def simulator() -> DummySimulator:
    return DummySimulator.instance()


def test_scale_action(simulator: DummySimulator):
    # Array must be within [-1, 1]
    with pytest.raises(AssertionError):
        simulator.scale_action(np.array([-1.1, 1], dtype=np.float32))
    with pytest.raises(AssertionError):
        simulator.scale_action(np.array([1.1, 1], dtype=np.float32))

    # Define an action in the range of [-1, 1]
    test_action = np.array([-1, 1], dtype=np.float32)

    # Scale the action using the method in PoolToolGym
    scaled_action = simulator.scale_action(test_action)

    # Retrieve the action space from the gym for comparison
    action_space = simulator.spaces.action

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


def test_simulate(simulator: DummySimulator):
    system_initial = simulator.state.system.copy()

    assert not system_initial.simulated
    assert not len(system_initial.events)
    assert system_initial.t == 0.0

    simulator.simulate()

    system_final = simulator.state.system
    assert system_final.simulated
    assert len(system_final.events)
    assert system_final.t != 0.0


def test_seed(simulator: DummySimulator):
    simulator.seed(123)
    value1 = np.random.rand()

    # Reset seed
    np.random.seed(123)
    value2 = np.random.rand()

    # The same seed should produce the same random number
    assert value1 == value2

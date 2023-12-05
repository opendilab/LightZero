import pytest
import numpy as np
from zoo.pooltool.sum_to_three.envs.sum_to_three_env import SimulationEnv

def test_spaces():
    obs, action, reward = SimulationEnv.sum_to_three().spaces()

    # 2-ball game, 3 spatial coordinates each
    assert obs.shape[0] == 6

    # We control V0 and phi
    assert action.shape[0] == 2

    # Reward bounded by 1
    assert reward.low == 0
    assert reward.high == 1


def test_env():
    env = SimulationEnv.sum_to_three()

    # V0=2, cut angle = -24 degrees
    action = np.array([2, -24], dtype=np.float32)
    env.set_action(action)
    env.simulate()

    assert len(env.system.events)

import numpy as np

from pooltool.ai.bot.sumtothree_rl.core import single_player_env

def test_env():
    env = single_player_env()

    # V0=2, cut angle = -24 degrees
    action = np.array([2, -24], dtype=np.float32)
    env.set_action(action)
    env.simulate()

    assert len(env.system.events)

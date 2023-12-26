
import datetime

import numpy as np
from numpy.typing import NDArray
from zoo.pooltool.sum_to_three.envs.sum_to_three_env import SumToThreeEnv

from pooltool.utils import get_total_memory_usage
from pooltool.terminal import TimeCode

N = 10_000

def random_action() -> NDArray[np.float32]:
    return 2.0 * np.random.random(size=2) - 1.0

env = SumToThreeEnv(SumToThreeEnv.default_config())
env.reset()
env.step(random_action())

episode = 0

while episode <= N:
    with TimeCode(success_msg=f"[Mem: {get_total_memory_usage()}] - Finished episode {episode} in ") as timer:
        while True:
            env._env.observation()
            output = env.step(random_action())
            if output.done:
                break

    env.reset()
    episode += 1

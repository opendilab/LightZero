"""
Overview:
    A script to run multiple episodes of the SumToThree environment. This script
    initializes the SumToThree environment and runs a specified number of episodes,
    where each episode consists of 10 shots. The observation type can be set to either
    'image' or 'coordinate'.
Usage:
    python run_sum_to_three.py --observation-type [image | coordinate] --num-episodes [int]
Options:
    --observation-type:
        Sets the type of observation for the environment. Use 'image' for visual
        observations or 'coordinate' for numerical observations.
    --num-episodes:
        Sets the number of episodes to run in the environment.
"""

import argparse
import numpy as np
from numpy.typing import NDArray
import pooltool as pt
from zoo.pooltool.sum_to_three.envs.sum_to_three_env import SumToThreeEnv
from zoo.pooltool.sum_to_three.envs.utils import ObservationType

def random_action() -> NDArray[np.float32]:
    return (2.0 * np.random.random(size=2) - 1.0).astype(np.float32)

def main(args):
    # Configure the environment
    config = SumToThreeEnv.default_config()
    config.episode_length = 10
    config.observation_type = ObservationType[args.observation_type.upper()]

    env = SumToThreeEnv(config)
    env.reset()
    assert env._init_flag

    for episode in range(args.num_episodes):
        with pt.terminal.TimeCode(success_msg=f"[Mem: {pt.utils.get_total_memory_usage()}] - Finished episode {episode} ({config.episode_length} shots) in ") as timer:
            for shot_count in range(config.episode_length):
                output = env.step(random_action())

        assert output.done
        env.reset()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SumToThree environment")
    parser.add_argument(
        "--observation-type",
        choices=[t.name.lower() for t in ObservationType],
        required=True,
        help="Type of observation: 'image' or 'coordinate'",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=10000,
        help="Number of episodes to run",
    )
    args = parser.parse_args()
    main(args)

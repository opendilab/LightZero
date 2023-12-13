import os
import tracemalloc

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from zoo.pooltool.sum_to_three.envs.sum_to_three_env import SumToThreeEnv

from pooltool.utils import memory_usage_to_dataframe

N = 2_000

def random_action() -> NDArray[np.float32]:
    return 2.0 * np.random.random(size=2) - 1.0


def track_memory_over_episodes():
    tracemalloc.start()

    env = SumToThreeEnv(SumToThreeEnv.default_config())
    env.reset()

    episode = 0
    memory_records = []

    while episode <= N:
        output = env.step(random_action())
        if output.done:
            print(f"Finished episode {episode}")
            env.reset()

            if episode % (N // 30) == 0:
                snapshot = tracemalloc.take_snapshot()
                frame = memory_usage_to_dataframe(snapshot, limit=200)
                frame["File_Line"] = (
                    frame["File"].astype(str) + ":" + frame["Line Number"].astype(str)
                )
                frame["episode"] = episode
                memory_records.append(frame)

            episode += 1

    # Concatenate all dataframes
    all_data = pd.concat(memory_records, ignore_index=True)
    plot_memory_usage(all_data)

    return all_data


def plot_memory_usage(all_data):
    # Ensure directory exists
    os.makedirs("memory_usage_plots", exist_ok=True)

    for identifier, group in all_data.groupby("File_Line"):
        if not (group["Memory Usage (KiB)"] > 100).any():
            continue

        plt.figure()
        plt.plot(group["episode"], group["Memory Usage (KiB)"])
        plt.scatter(group["episode"], group["Memory Usage (KiB)"])
        plt.title(f"Memory Usage of {identifier}")
        plt.xlabel("Episode")
        plt.ylabel("Memory Usage (KiB)")

        # Filename: Replace problematic characters
        filename = identifier.replace("/", "_").replace(":", "_")
        plt.savefig(f"memory_usage_plots/{filename}_memory_usage.png")
        plt.close()


# Run the tracking function
frames = track_memory_over_episodes()
print("Plots written to current directory (memory_usage_plots)")
import ipdb

ipdb.set_trace()

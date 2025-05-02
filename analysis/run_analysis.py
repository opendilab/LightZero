import wandb
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  # or 'Agg', 'Qt5Agg', depending on your system
import matplotlib.pyplot as plt

from dotenv import load_dotenv
import os

load_dotenv()  # loads variables from .env into environment
api_key = os.getenv("WANDB_API_KEY")
wandb.login(key = api_key)

# Replace with your project name and optionally entity
api = wandb.Api()
runs = api.runs("dani-allegue/LightZero", filters={"state": "finished"})

filtered_runs_vanilla = [
    run for run in runs if {"final", "vanilla_unizero"}.issubset(set(run.tags))
]

filtered_runs_routing = [
    run for run in runs if {"final", "routing-unizero"}.issubset(set(run.tags))
]

def extract_eval_df(runs, label):
    dfs = []
    for run in runs:
        history = run.history(keys=["collector_step/reward_mean", "_step"], pandas=True)
        # evaluator_step/eval_episode_return_mean"
        history["run_name"] = run.name
        history["variant"] = label
        dfs.append(history)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

# Load both sets
df_vanilla = extract_eval_df(filtered_runs_vanilla, "Vanilla UniZero")
df_routing = extract_eval_df(filtered_runs_routing, "Routing UniZero; k = 4")

# Combine all
combined_df = pd.concat([df_vanilla, df_routing], ignore_index=True)

# Group and plot
plt.figure(figsize=(6, 4))
for label, df in combined_df.groupby("variant"):
    grouped = df.groupby("_step")["collector_step/reward_mean"]
    mean = grouped.mean()
    std = grouped.std()
    steps = mean.index
    plt.plot(steps, mean, label=label)
    plt.fill_between(steps, mean - std, mean + std, alpha=0.3)

plt.title("UniZero Variants on Boxing (100k Steps)")
plt.xlabel("Global Step")
plt.ylabel("Eval Episode Return (mean ± std)")
plt.legend()
plt.tight_layout()
plt.show()
# # Plot
# plt.figure(figsize=(4, 3))
# plt.plot(steps, mean_df, color="steelblue", label="Vanilla UniZero")
# plt.fill_between(steps, mean_df - std_df, mean_df + std_df, color="steelblue", alpha=0.3)
# plt.title("Vanilla UniZero in Boxing")
# plt.xlabel("Global Step")
# plt.ylabel("Eval Return")
# plt.legend()
# plt.tight_layout()
# plt.show()
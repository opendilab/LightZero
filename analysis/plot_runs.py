import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # or 'Agg', 'Qt5Agg', depending on your system
import matplotlib.pyplot as plt

# 1. Parameters / placeholders
CSV_PATH = "../results/gaam-4-6-vanilla.csv"  # your WandB-exported CSV
STEP_COL = "collector_step/total_envstep_count"

# 2. Define your four groups of three runs each
# Replace 'runA1','runA2','runA3', etc. with your actual run IDs
df = pd.read_csv(CSV_PATH)
mean_suffix = "evaluator_step/eval_episode_return_mean"
STEP_COL = "collector_step/total_envstep_count"
MEAN_SUFFIX = "evaluator_step/eval_episode_return_mean"

# 2) Read the CSV & discover run IDs
df = pd.read_csv(CSV_PATH)
mean_cols = [c for c in df.columns if c.endswith(MEAN_SUFFIX)]
run_ids   = [c.split(" - ")[0] for c in mean_cols]

# 3) Define your four groups of three runs each
run_groups = {
    #"window_2": ["rosy-dust-253", "vivid-fog-252", "splendid-snowball-251"],
    #"window_4": ["northern-sky-238", "trim-cloud-237",     "ethereal-universe-236"],
    #"window_6": ["fresh-energy-249",  "leafy-waterfall-250","bright-sun-248"],
    #"window_8": ["valiant-pine-262",  "breezy-thunder-260","worthy-dream-261"],
    "gaam_4" : ["distinctive-feather-304", "likely-galaxy-303", "devoted-sunset-302"],
    "gaam_6" : ["fine-waterfall-295", "stoic-firefly-294", "tough-monkey-293"],
    "vanilla_unizero" : ["wild-cosmos-256", "wandering-haze-254", "summer-pond-255"]
}

results = []
for group_label, runs in run_groups.items():
    # union of all steps
    step_sets = []
    for run in runs:
        col = f"{run} - {MEAN_SUFFIX}"
        steps = df[[STEP_COL, col]].dropna(subset=[col])[STEP_COL].unique()
        step_sets.append(set(steps))
    all_steps = np.array(sorted(set().union(*step_sets)))

    # align & interpolate each run
    aligned = pd.DataFrame(index=all_steps, columns=runs, dtype=float)
    for run in runs:
        col = f"{run} - {MEAN_SUFFIX}"
        ser = df[[STEP_COL, col]].dropna(subset=[col]).set_index(STEP_COL)[col]
        filled = (
            ser
            .reindex(all_steps)
            .interpolate(method="index", limit_direction="both")
            .bfill()
            .ffill()
        )
        aligned[run] = filled

    # compute mean and SEM
    group_mean = aligned.mean(axis=1)
    group_sem  = aligned.std(axis=1, ddof=1) / np.sqrt(len(runs))

    results.append((group_label, all_steps, group_mean, group_sem))

# find the minimal “end” across all groups
end_points = [steps[-1] for (_, steps, _, _) in results]
x_max = min(end_points)
print(f"Truncating all series at x = {x_max}")

# === 3. Plot ===
plt.figure(figsize=(10, 6))
colors = plt.get_cmap("tab10")

for idx, (group_label, all_steps, group_mean, group_sem) in enumerate(results):
    # restrict to x ≤ x_max
    #mask = all_steps <= x_max
    # xs   = all_steps[mask]
    # mu   = group_mean[mask]
    # sem  = group_sem[mask]
    xs = all_steps
    mu = group_mean
    sem = group_sem

    color = colors(idx)
    if group_label == "vanilla_unizero":
        plt.plot(xs, mu, label=group_label, color=color, linewidth=2, linestyle='--')
    else:
        plt.plot(xs, mu, label=group_label, color=color, linewidth=2)
    plt.fill_between(xs, mu - sem, mu + sem, color=color, alpha=0.3)

# === 4. Finalize ===
plt.xlabel("Environment Step")
plt.ylabel("Episode Return")
plt.title("Pong Return: Gaussian Attention")
plt.xlim(right=92000)
plt.tight_layout()
plt.show()
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # or 'Agg', 'Qt5Agg', depending on your system
import matplotlib.pyplot as plt

# 1. Parameters / placeholders
CSV_PATH = "../results/local_pong_2.csv"  # your WandB-exported CSV
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
    "window_2": ["rosy-dust-253", "vivid-fog-252", "splendid-snowball-251"],
    "window_4": ["northern-sky-238", "trim-cloud-237",     "ethereal-universe-236"],
    "window_6": ["fresh-energy-249",  "leafy-waterfall-250","bright-sun-248"],
    "window_8": ["valiant-pine-262",  "breezy-thunder-260","worthy-dream-261"],
}

plt.figure(figsize=(10, 6))
colors = plt.get_cmap("tab10")

for idx, (group_label, runs) in enumerate(run_groups.items()):
    # 4a) Find the union of all steps where any run has data
    step_sets = []
    for run in runs:
        col = f"{run} - {MEAN_SUFFIX}"
        run_steps = df[[STEP_COL, col]].dropna(subset=[col])[STEP_COL].unique()
        step_sets.append(set(run_steps))
    all_steps = np.array(sorted(set().union(*step_sets)))

    # 4b) Build aligned DataFrame of shape (len(all_steps), len(runs))
    aligned = pd.DataFrame(index=all_steps, columns=runs, dtype=float)
    for run in runs:
        col = f"{run} - {MEAN_SUFFIX}"
        ser = df[[STEP_COL, col]].dropna(subset=[col]).set_index(STEP_COL)[col]
        # reindex + interpolate + bfill/ffill
        filled = (
            ser
            .reindex(all_steps)
            .interpolate(method="index", limit_direction="both")
            .bfill()
            .ffill()
        )
        aligned[run] = filled

    # 4c) Compute group mean and SEM
    group_mean = aligned.mean(axis=1)
    group_sem  = aligned.std(axis=1, ddof=1) / np.sqrt(len(runs))

    # Debug print for first few steps
    print(f"\n=== Group {group_label} SEM bands ===")
    for step, sem in zip(all_steps[:5], group_sem.values[:5]):
        print(f" Step {step}: SEM = {sem:.4f}")
    print("===============================\n")

    # 4d) Plot mean + SEM ribbon
    color = colors(idx)
    plt.plot(all_steps, group_mean, label=group_label, color=color, linewidth=2)
    plt.fill_between(all_steps,
                     group_mean - group_sem,
                     group_mean + group_sem,
                     color=color, alpha=0.3)
# 5) Finalize
plt.xlabel("Environment Step")
plt.ylabel("Episode Return (mean)")
plt.title("Grouped Return Curves ±1 SEM (interpolated)")
plt.legend(title="Local Window Size")
plt.tight_layout()
plt.show()
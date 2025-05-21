#!/usr/bin/env python3
"""
plot_ci_manual.py

Fetch three WandB runs, compute their mean ± 95% CI,
and plot exactly one line+shaded band per group.
"""

import numpy as np
import pandas as pd
import scipy.stats as st
import wandb
import matplotlib
import matplotlib.pyplot as plt

# In PyCharm on Windows, TkAgg tends to work reliably:
matplotlib.use("TkAgg")


# def fetch_run_history(api, run_id: str, metric: str, x_axis: str) -> pd.DataFrame:
#     """
#     Fetch a single run's full history, auto‐rename the step column,
#     drop duplicates, and return a DataFrame indexed by x_axis with one column.
#     """
#     run = api.run(run_id)
#     df = run.history(pandas=True)
#     # auto‐detect & rename the x_axis if needed
#     if x_axis not in df.columns:
#         for alt in ("step", "_step", "global_step"):
#             if alt in df.columns:
#                 df = df.rename(columns={alt: x_axis})
#                 break
#         else:
#             raise KeyError(f"Could not find any step column in run {run_id}")
#     if metric not in df.columns:
#         raise KeyError(f"Metric '{metric}' not found in run {run_id}")
#
#     # keep only the metric + x_axis, drop duplicate steps, index by x_axis
#     df = df[[x_axis, metric]].drop_duplicates(subset=x_axis).set_index(x_axis)
#     return df
#
#
# def compute_group_stats(dfs: list[pd.DataFrame], confidence: float = 0.95) -> pd.DataFrame:
#     """
#     Given a list of DataFrames all indexed by Step, each with one column,
#     return a DataFrame with columns ['mean','ci_lower','ci_upper'].
#     """
#     alpha = 1.0 - confidence
#     # outer‐join on the index (Step) so we align all runs
#     all_df = pd.concat(dfs, axis=1, join="outer")
#     # Compute the statistics:
#     mean = all_df.mean(axis=1)
#     n = all_df.count(axis=1)                   # how many runs at each step
#     sem = all_df.std(axis=1, ddof=1) / np.sqrt(n)
#     tcrit = st.t.ppf(1 - alpha/2, df=n - 1)
#
#     stats = pd.DataFrame({
#         "mean": mean,
#         "ci_lower": mean - tcrit * sem,
#         "ci_upper": mean + tcrit * sem
#     }, index=all_df.index)
#
#     return stats.dropna(subset=["mean"])
#
#
# def plot_groups(
#     groups: list[list[str]],
#     labels: list[str],
#     metric: str = "eval_episode_return_mean",
#     x_axis: str = "Step",
#     confidence: float = 0.95,
# ):
#     api = wandb.Api()
#     plt.figure(figsize=(10, 6))
#
#     # grab the default color cycle so mean+CI share the same hue
#     prop_cycle = plt.rcParams['axes.prop_cycle']
#     colors = prop_cycle.by_key()['color']
#
#     for idx, run_ids in enumerate(groups):
#         # 1) fetch each run
#         dfs = [ fetch_run_history(api, rid, metric, x_axis) for rid in run_ids ]
#         # 2) compute group mean & CI
#         stats = compute_group_stats(dfs, confidence=confidence)
#
#         x = stats.index.values
#         y = stats["mean"].values
#         lo = stats["ci_lower"].values
#         hi = stats["ci_upper"].values
#
#         color = colors[idx % len(colors)]
#         label = labels[idx] if idx < len(labels) else f"Group {idx+1}"
#
#         # 3) plot mean + CI band
#         plt.plot(x, y, label=label, color=color, linewidth=2)
#         # plt.fill_between(x, lo, hi, color=color, alpha=0.2)
#
#     plt.xlabel(x_axis)
#     plt.ylabel(metric)
#     plt.title(f"{metric} with {int(100*confidence)}% CI")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()
#
#
# def main():
#     # ────── DEFINE YOUR GROUP(S) HERE ──────────────────────────
#     # A single group containing your three runs:
#     groups = [
#         [
#             "dani-allegue/LightZero/yz0vpzug",
#             "dani-allegue/LightZero/bebs4qe5",
#             "dani-allegue/LightZero/05xiaxrz",
#         ]
#     ]
#
#     # A matching label for that group:
#     labels = ["Local = 2"]
#
#     # Which metric & axis name to fetch:
#     metric = "evaluator_step/eval_episode_return_mean"
#     x_axis = "Step"
#     confidence = 0.95
#     # ────────────────────────────────────────────────────────────
#
#     if not groups:
#         raise ValueError("Please define at least one group in main().")
#
#     plot_groups(groups, labels, metric, x_axis, confidence)
#
#
# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
import numpy as np
import pandas as pd
import scipy.stats as st
import wandb
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("TkAgg")

# ─── EDIT THESE ───────────────────────────────────────────────
RUN_IDS = [
    "dani-allegue/LightZero/yz0vpzug",
    "dani-allegue/LightZero/bebs4qe5",
    "dani-allegue/LightZero/05xiaxrz",
]
METRIC = "evaluator_step/eval_episode_return_mean"
X_AXIS = "Step"
CONFIDENCE = 0.95
# ──────────────────────────────────────────────────────────────

#!/usr/bin/env python3
"""
plot_ci_interpolated.py

Fetch multiple WandB runs, interpolate them onto a common Step grid,
compute mean ± 95% CI between runs, and plot the result.
"""

import numpy as np
import pandas as pd
import scipy.stats as st
import wandb
import matplotlib
import matplotlib.pyplot as plt

# Use a friendly interactive backend
matplotlib.use("TkAgg")

# ─── EDIT THESE ───────────────────────────────────────────────
RUN_IDS      = [
    "dani-allegue/LightZero/yz0vpzug",
    "dani-allegue/LightZero/bebs4qe5",
    "dani-allegue/LightZero/05xiaxrz",
]
METRIC       = "evaluator_step/eval_episode_return_mean"
X_AXIS       = "Step"
CONFIDENCE   = 0.95
CI_SMOOTH_WIN = 5    # odd integer for centered rolling average
DENSE_POINTS = 300   # how many points in the final smooth curve
# ──────────────────────────────────────────────────────────────

def fetch_one(run_id: str) -> pd.DataFrame:
    api = wandb.Api()
    run = api.run(run_id)
    df = run.history(pandas=True)

    # rename common step columns
    if X_AXIS not in df.columns:
        for alt in ("step", "_step", "global_step"):
            if alt in df.columns:
                df = df.rename(columns={alt: X_AXIS})
                break

    if METRIC not in df.columns:
        raise KeyError(f"Metric '{METRIC}' not found in run {run_id}")

    return (
        df[[X_AXIS, METRIC]]
        .drop_duplicates(subset=X_AXIS)
        .set_index(X_AXIS)
        .rename(columns={METRIC: run_id.split("/")[-1]})
    )

def compute_stats(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    alpha  = 1.0 - CONFIDENCE
    joined = pd.concat(dfs, axis=1, join="outer").sort_index()

    mean  = joined.mean(axis=1)
    n     = joined.count(axis=1)
    sem   = joined.std(axis=1, ddof=1) / np.sqrt(n)
    tcrit = st.t.ppf(1 - alpha/2, df=n - 1)

    ci_lo = mean - tcrit * sem
    ci_hi = mean + tcrit * sem

    return pd.DataFrame({"mean": mean, "ci_lo": ci_lo, "ci_hi": ci_hi},
                        index=joined.index).dropna(subset=["mean"])

def main():
    # 1) Fetch raw runs
    raw_dfs = [fetch_one(rid) for rid in RUN_IDS]

    # 2) Determine common final step = minimum of each run's max index
    max_steps = [df.index.max() for df in raw_dfs]
    final_step = min(max_steps)

    # 3) Build shared grid up to final_step
    steps_all = sorted(set().union(*[df.index for df in raw_dfs]))
    # restrict to <= final_step
    steps_all = [s for s in steps_all if s <= final_step]

    # 4) Interpolate each run onto that grid
    dfs = []
    for df in raw_dfs:
        df2 = (
            df.reindex(steps_all)
              .interpolate(method="index")
              .fillna(method="bfill")
              .fillna(method="ffill")
        )
        dfs.append(df2)

    # 5) Compute raw mean & CI
    stats = compute_stats(dfs)

    # 6) Smooth CI bounds only
    ci = stats[["ci_lo", "ci_hi"]]
    ci_smooth = ci.rolling(window=CI_SMOOTH_WIN, center=True, min_periods=1).mean()
    mean = stats["mean"]

    # 7) Create dense x grid up to final_step
    x_old = stats.index.values
    x_new = np.linspace(x_old.min(), final_step, DENSE_POINTS)

    lo_new   = np.interp(x_new, x_old, ci_smooth["ci_lo"].values)
    hi_new   = np.interp(x_new, x_old, ci_smooth["ci_hi"].values)
    mean_new = np.interp(x_new, x_old, mean.values)

    # 8) Plot
    plt.figure(figsize=(10, 6))
    color = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]

    # smooth shaded band
    plt.fill_between(x_new, lo_new, hi_new,
                     color=color, alpha=0.2, linewidth=0,
                     label=f"{int(CONFIDENCE*100)}% CI")

    # continuous CI boundary lines
    plt.plot(x_new, lo_new, linestyle="--", linewidth=1,
             color=color, alpha=0.5)
    plt.plot(x_new, hi_new, linestyle="--", linewidth=1,
             color=color, alpha=0.5)

    # raw mean line (up to final_step)
    plt.plot(x_old, mean.values, label="Mean return",
             color=color, linewidth=2)

    plt.xlabel(X_AXIS)
    plt.ylabel(METRIC)
    plt.title(f"{METRIC} (mean + smoothed {int(CONFIDENCE*100)}% CI)")
    plt.legend(loc="upper left")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

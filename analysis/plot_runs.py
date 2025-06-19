import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

CSV_PATH = "../results/ablation_local_pong.csv"  # your WandB-exported CSV
STEP_COL = "collector_step/total_envstep_count" # X-column

df = pd.read_csv(CSV_PATH)
mean_suffix = "evaluator_step/eval_episode_return_mean" # Y-colum
STEP_COL = "collector_step/total_envstep_count"
MEAN_SUFFIX = "evaluator_step/eval_episode_return_mean"

# 2) Read the CSV & discover run IDs
df = pd.read_csv(CSV_PATH)
mean_cols = [c for c in df.columns if c.endswith(MEAN_SUFFIX)]
run_ids   = [c.split(" - ")[0] for c in mean_cols]

# Pong
# run_groups = {
#     "Vanilla" : ["summer-snowball-361", "proud-cherry-362", "peach-hill-363", "spring-paper-364", "major-paper-365"],
#     "Local $w = 2$": ["jumping-glade-330", "winter-music-341", "clean-wood-342", "playful-energy-343", "hopeful-dust-343"],
#     "Gaussian $\sim \mathcal{N}(2, 1)$" : ["wise-plant-351", "sparkling-glitter-317", "ethereal-capybara-351", "wild-terrain-352", "wild-plant-351"],
#     #"gaam_6" : ["fine-waterfall-295", "stoic-firefly-294", "tough-monkey-293"],
# }

# # # Boxing
# run_groups = {
#     # "snowy-dawn-383"
#     "vanilla_unizero": ["wild-salad-265", "floral-shadow-264", "rose-glade-263", "woven-glitter-384", "fearless-pine-66", "hardy-violet-200", "iconic-meadow-201"],
#     "local": ["apricot-dream-370", "robust-paper-368", "absurd-dust-369", "glorious-forest-366", "effortless-disco-366", "wise-music-544", "proud-forest-543"],
#     #"gaam2": ["apricot-universe-356", "sleek-energy-357", "glad-sky-358", "generous-aardvark-359", "neat-elevator-360"],
#     "gaam": ["balmy-sunset-382", "earthy-thunder-373", "soft-donkey-372","olive-hill-371", "treasured-bee-408", "decent-shape-407", "snowy-dawn-383"],
#     #"local" : ["apricot-dream-370", "robust-paper-368", "absurd-dust-369", "glorious-forest-366", "effortless-disco-366"]
# }

#BankHeist
# run_groups = {
#     "Vanilla": ["mild-sunset-442", "youthful-fog-506", "glowing-oath-478", "sweet-silence-507", "usual-dawn-675", "eager-feather-674"],
#     "Local $w = 10$": ["lunar-dust-438", "wobbly-feather-456", "radiant-donkey-441", "balmy-star-481", "stoic-dragon-670", "lyric-butterfly-667"],
#     "Gaussian $\mu = 10, \sigma = 2$" : ["super-terrain-447", "stoic-serenity-448", "lilac-sun-448", "hearty-microwave-508", "apricot-plasma-509", "resilient-durian-680"]
# }

# # Ablation
run_groups = {
    #"vanilla_unizero": ["major-paper-365", "spring-paper-364", "peach-hill-363", "proud-cherry-362", "summer-snowball-361"],
    "$w$ = 2" : ["jumping-glade-330", "winter-music-341", "clean-wood-342", "playful-energy-343", "hopeful-dust-343"],
    "$w$ = 6" : ["likely-wind-512", "wise-darkness-511", "zany-dew-492", "frosty-bush-502", "effortless-sea-490"],
    "$w$ = 12": ["dandy-frog-514", "dainty-firebrand-513", "deft-water-515", "chocolate-field-516", "fluent-shape-517"]
}

# # Mu-Ablation
# run_groups = {
#     "$\mu$ = 2": ["wild-plant-351", "wild-terrain-352", "ethereal-capybara-351", "pious-morning-352", "wise-plant-351"],
#     "$\mu$ = 6": ["swept-moon-545", "stoic-firefly-294", "dry-morning-546", "dark-violet-547", "splendid-flower-549"],
#     "$\mu$ = 12": ["bright-energy-522", "dutiful-monkey-521", "brisk-sky-520", "wise-bush-519", "wobbly-glitter-518"],
# }

# Sigma Ablation
# run_groups = {
#     "$\sigma$ = 1": ["wild-plant-351", "wild-terrain-352", "ethereal-capybara-351", "pious-morning-352", "wise-plant-351"],
#     "$\sigma$ = 2": ["dry-bee-528", "eager-rain-528", "northern-music-528", "dandy-aardvark-531", "silver-oath-537"],
#     "$\sigma$ = 4": ["stellar-brook-541", "ruby-star-540", "hearty-paper-539", "ancient-rain-538"],
# }


# Pacman
# run_groups = {
#     "Vanilla": ["efficient-paper-660", "fragrant-flower-646", "unique-river-644", "classic-fire-589", "dainty-cherry-588", "mild-firefly-586"],
#     "local $w = 10$" : ["ruby-wood-642", "smooth-bee-658", "pleasant-lake-661", "playful-thunder-681", "blooming-thunder-682", "easy-rain-683"],
#     "Gaussian $\mu = 10, \sigma = 2$" : ["comfy-serenity-592", "laced-wood-603", "legendary-waterfall-604", "celestial-feather-624", "comic-dream-625", "tough-donkey-629"]
# }
# # Boxing Mu
# run_groups = {
#     "$w$ = 2" : ["good-dawn-345", "cool-donkey-346", "fragrant-armadillo-347", "denim-wildflower-348", "fragrant-armadillo-349"],
#     "$w$ = 6" : ["effortless-disco-366", "glorious-forest-366", "absurd-dust-369", "robust-paper-368", "apricot-dream-370", "wise-music-544", "proud-forest-543"],
#     "$w$ = 12": ["fine-hill-597", "logical-sun-596", "major-mountain-598", "wild-sea-599", "drawn-wood-608", "astral-energy-610"]
# }

# For statistical analysis
final_scores_at_100k = {}


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

    group_std = aligned.std(axis=1, ddof=1)

    xs = all_steps
    means = group_mean.values
    stds = group_std.values
    m100k = np.interp(100000, xs, means)
    s100k = np.interp(100000, xs, stds) / np.sqrt(len(runs))

    scores_100k_raw = []
    for run_column in aligned.columns:
        # Interpolate the score at 100k for this specific run
        run_specific_scores = aligned[run_column].values
        score_at_100k = np.interp(100000, xs, run_specific_scores)
        scores_100k_raw.append(score_at_100k)

    # Store the list of raw scores in our dictionary
    final_scores_at_100k[group_label] = np.array(scores_100k_raw)

    print(f"Group label {group_label}, mean at 100k: {m100k:.2f} ± {s100k:.2f}")
    results.append((group_label, all_steps, group_mean, group_sem))

import itertools
from scipy.stats import ttest_ind

# final_scores_at_100k is a dict: { group_label: np.array([score1, score2, …]) }
labels = list(final_scores_at_100k.keys())

# print("Pairwise Welch’s t-tests at step 100k\n" + "-"*40)
# for a, b in itertools.combinations(labels, 2):
#     scores_a = final_scores_at_100k[a]
#     scores_b = final_scores_at_100k[b]
#     t_stat, p_val = ttest_ind(scores_a, scores_b, equal_var=False)
#     print(f"{a:>10} vs {b:<10} → t = {t_stat:6.2f}, p = {p_val:.4f}")


plt.rcParams.update({
    "axes.titlesize": 30,     # default for ax.set_title(...)
    "axes.labelsize": 25,     # default for ax.set_xlabel / set_ylabel
    "xtick.labelsize": 19,    # tick labels
    "ytick.labelsize": 19,
    "legend.fontsize": 20,
})

# Poster
# plt.rcParams.update({
#     "axes.titlesize": 35,
#     "axes.labelsize": 35,
#     "xtick.labelsize": 20,
#     "ytick.labelsize": 20,
#     "legend.fontsize": 22,
# })
# Create a figure and a set of subplots. fig is the whole window, ax is the plot.
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(27, 8))
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(45, 10))
fig = plt.figure(figsize=(26, 16))
gs = fig.add_gridspec(2, 2, hspace=0.25, wspace=0.1)

# # Create the axes objects by indexing the grid
ax1 = fig.add_subplot(gs[0, 0])  # Top-left
ax2 = fig.add_subplot(gs[0, 1])  # Top-right
ax3 = fig.add_subplot(gs[1, 0:1])   # Bottom, spanning all columns
# ax4 = fig.add_subplot(gs[1, 1:2])

colors = plt.get_cmap("Set1")

for idx, (group_label, all_steps, group_mean, group_sem) in enumerate(results):
    xs = all_steps
    mu = group_mean
    sem = group_sem

    color = colors(idx)
    linestyle = '-'
    # Set custom colors and styles based on the label
    if group_label == "Vanilla":
        linestyle = '--'

    # Call plotting methods directly on the 'ax' object
    ax1.plot(xs, mu, label=group_label, color=color, linewidth=2, linestyle=linestyle)
    ax1.fill_between(xs, mu - sem, mu + sem, color=color, alpha=0.3)

# Use the 'set_*' methods to customize the axes
#ax1.set_xlabel("Environment Step")
# ax1.set_ylabel("Episode Return")
ax1.set_title("Ablation of Window Size $w$") # Ablation of Window Size $w$

# Set axis limits and ticks directly on 'ax'
ax1.set_xlim(right=100000)
ax1.set_xticks([40000, 60000, 80000, 100000])

ax1.axhline(y=-14.53, color='gray', linestyle='--', linewidth=1, label='UniZero Return at 100k-step')
ax1.tick_params(axis='both', which='major')
ax1.legend()

#ax1.axvline(x=100000, color='gray', linestyle='--', linewidth=1)
# run_groups2 = {
#     # "snowy-dawn-383"
#     "Vanilla": ["wild-salad-265", "floral-shadow-264", "rose-glade-263", "woven-glitter-384", "fearless-pine-66", "hardy-violet-200", "iconic-meadow-201"],
#     "Local $w = 6$": ["apricot-dream-370", "robust-paper-368", "absurd-dust-369", "glorious-forest-366", "effortless-disco-366", "wise-music-544", "proud-forest-543"],
#     #"gaam2": ["apricot-universe-356", "sleek-energy-357", "glad-sky-358", "generous-aardvark-359", "neat-elevator-360"],
#     "Gaussian $\mu = 6, \sigma = 1$": ["balmy-sunset-382", "earthy-thunder-373", "soft-donkey-372","olive-hill-371", "treasured-bee-408", "decent-shape-407", "snowy-dawn-383"],
#     #"local" : ["apricot-dream-370", "robust-paper-368", "absurd-dust-369", "glorious-forest-366", "effortless-disco-366"]
# }

run_groups2 = {
    "$\mu$ = 2": ["wild-plant-351", "wild-terrain-352", "ethereal-capybara-351", "pious-morning-352", "wise-plant-351"],
    "$\mu$ = 6": ["swept-moon-545", "stoic-firefly-294", "dry-morning-546", "dark-violet-547", "splendid-flower-549"],
    "$\mu$ = 12": ["bright-energy-522", "dutiful-monkey-521", "brisk-sky-520", "wise-bush-519", "wobbly-glitter-518"],
}

# run_groups2 = {
#     # "snowy-dawn-383"
#     "Vanilla": ["wild-salad-265", "floral-shadow-264", "rose-glade-263", "woven-glitter-384", "fearless-pine-66", "hardy-violet-200", "iconic-meadow-201"],
#     "Local $w = 6$": ["apricot-dream-370", "robust-paper-368", "absurd-dust-369", "glorious-forest-366", "effortless-disco-366", "wise-music-544", "proud-forest-543"],
#     #"gaam2": ["apricot-universe-356", "sleek-energy-357", "glad-sky-358", "generous-aardvark-359", "neat-elevator-360"],
#     "Gaussian $\sim \mathcal{N}(6, 1)$": ["balmy-sunset-382", "earthy-thunder-373", "soft-donkey-372","olive-hill-371", "treasured-bee-408", "decent-shape-407", "snowy-dawn-383"],
#     #"local" : ["apricot-dream-370", "robust-paper-368", "absurd-dust-369", "glorious-forest-366", "effortless-disco-366"]
# }

final_scores_at_100k = {}


CSV_PATH = "../results/ablation-mu.csv"
df = pd.read_csv(CSV_PATH)
mean_cols = [c for c in df.columns if c.endswith(MEAN_SUFFIX)]
run_ids   = [c.split(" - ")[0] for c in mean_cols]

# Plot second
results2 = []
for group_label, runs in run_groups2.items():
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

    group_std = aligned.std(axis=1, ddof=1)

    xs = all_steps
    means = group_mean.values
    stds = group_std.values
    m100k = np.interp(100000, xs, means)
    s100k = np.interp(100000, xs, stds) / np.sqrt(len(runs))

    scores_100k_raw = []
    for run_column in aligned.columns:
        # Interpolate the score at 100k for this specific run
        run_specific_scores = aligned[run_column].values
        score_at_100k = np.interp(100000, xs, run_specific_scores)
        scores_100k_raw.append(score_at_100k)

    # Store the list of raw scores in our dictionary
    final_scores_at_100k[group_label] = np.array(scores_100k_raw)
    print(f"Group label {group_label}, mean at 100k: {m100k:.2f} ± {s100k:.2f}")

    results2.append((group_label, all_steps, group_mean, group_sem))

labels = list(final_scores_at_100k.keys())

colors = plt.get_cmap("Set1")

for idx, (group_label, all_steps, group_mean, group_sem) in enumerate(results2):
    mask = all_steps >= 20000
    xs   = all_steps[mask]
    mu   = group_mean[mask]
    sem  = group_sem[mask]
    # xs = all_steps
    # mu = group_mean
    # sem = group_sem

    color = colors(idx)
    linestyle = '-'
    # # Set custom colors and styles based on the label
    if group_label == "Vanilla":
        linestyle = '--'

    # Call plotting methods directly on the 'ax' object
    ax2.plot(xs, mu, label=group_label, color=color, linewidth=2, linestyle=linestyle)
    ax2.fill_between(xs, mu - sem, mu + sem, color=color, alpha=0.3)

# Use the 'set_*' methods to customize the axes
# ax2.set_xlabel("Environment Step")
# ax2.set_ylabel("Episode Return", fontsize=14)
ax2.set_title("Ablation of Initial $\mu$") # Ablation of Initial $\mu$

ax2.axhline(y=-14.53, color='gray', linestyle='--', linewidth=1, label='UniZero Return at 100k-step')
# Set axis limits and ticks directly on 'ax'
ax2.set_xlim(right=100000)
ax2.set_xticks([40000, 60000, 80000, 100000])
# ax2.set_yticks([-5, 0, 5, 10, 15])

# Set the font size for the tick labels
ax2.tick_params(axis='both', which='major')
ax2.legend()
# Add a vertical line to the axes


# Third Plot------------------------------------------------------------------------------------------------------------------------------------------
run_groups3 = {
    "$\sigma$ = 1": ["wild-plant-351", "wild-terrain-352", "ethereal-capybara-351", "pious-morning-352", "wise-plant-351"],
    "$\sigma$ = 2": ["dry-bee-528", "eager-rain-528", "northern-music-528", "dandy-aardvark-531", "silver-oath-537"],
    "$\sigma$ = 4": ["stellar-brook-541", "ruby-star-540", "hearty-paper-539", "ancient-rain-538"],
}

final_scores_at_100k = {}


CSV_PATH = "../results/ablation-sigma-prelim.csv"
df = pd.read_csv(CSV_PATH)
mean_cols = [c for c in df.columns if c.endswith(MEAN_SUFFIX)]
run_ids   = [c.split(" - ")[0] for c in mean_cols]

# Plot second
results3 = []
for group_label, runs in run_groups3.items():
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

    group_std = aligned.std(axis=1, ddof=1)

    xs = all_steps
    means = group_mean.values
    stds = group_std.values
    m100k = np.interp(100000, xs, means)
    s100k = np.interp(100000, xs, stds) / np.sqrt(len(runs))

    scores_100k_raw = []
    for run_column in aligned.columns:
        # Interpolate the score at 100k for this specific run
        run_specific_scores = aligned[run_column].values
        score_at_100k = np.interp(100000, xs, run_specific_scores)
        scores_100k_raw.append(score_at_100k)

    # Store the list of raw scores in our dictionary
    final_scores_at_100k[group_label] = np.array(scores_100k_raw)

    print(f"Group label {group_label}, mean at 100k: {m100k:.2f} ± {s100k:.2f}")
    results3.append((group_label, all_steps, group_mean, group_sem))

labels = list(final_scores_at_100k.keys())

colors = plt.get_cmap("Set1")

for idx, (group_label, all_steps, group_mean, group_sem) in enumerate(results3):
    mask = all_steps >= 20000
    xs   = all_steps[mask]
    mu   = group_mean[mask]
    sem  = group_sem[mask]
    # xs = all_steps
    # mu = group_mean
    # sem = group_sem

    color = colors(idx)
    linestyle = '-'
    # # Set custom colors and styles based on the label
    if group_label == "Vanilla":
        linestyle = '--'
    ax3.plot(xs, mu, label=group_label, color=color, linewidth=2, linestyle=linestyle)
    ax3.fill_between(xs, mu - sem, mu + sem, color=color, alpha=0.3)

# ax3.set_xlabel("Environment Step")
ax3.set_title("Ablation of Inital Sigma $\sigma$") # Ablation of Inital Sigma $\sigma$

# ax3.set_ylabel("Episode Return")
ax3.set_xlim(right=100000)
ax3.axhline(y=-14.53, color='gray', linestyle='--', linewidth=1, label='UniZero Return at 100k-step')
ax3.set_xticks([40000, 60000, 80000, 100000])

ax3.tick_params(axis='both', which='major')
ax3.legend()

# Fourth Plot------------------------------------------------------------------------------------------------------------------------------------------
# run_groups4 = {
#     "Vanilla": ["efficient-paper-660", "fragrant-flower-646", "unique-river-644", "classic-fire-589", "dainty-cherry-588", "mild-firefly-586"],
#     "Local $w = 10$" : ["ruby-wood-642", "smooth-bee-658", "pleasant-lake-661", "playful-thunder-681", "blooming-thunder-682", "easy-rain-683"],
#     "Gaussian $\sim \mathcal{N}(10, 2^2)$" : ["comfy-serenity-592", "laced-wood-603", "legendary-waterfall-604", "celestial-feather-624", "comic-dream-625", "tough-donkey-629"]
# }
#
# final_scores_at_100k = {}
#
# CSV_PATH = "../results/pacman.csv"
# df = pd.read_csv(CSV_PATH)
# mean_cols = [c for c in df.columns if c.endswith(MEAN_SUFFIX)]
# run_ids = [c.split(" - ")[0] for c in mean_cols]
#
# # Plot second
# results4 = []
# for group_label, runs in run_groups4.items():
#     # union of all steps
#     step_sets = []
#     for run in runs:
#         col = f"{run} - {MEAN_SUFFIX}"
#         steps = df[[STEP_COL, col]].dropna(subset=[col])[STEP_COL].unique()
#         step_sets.append(set(steps))
#     all_steps = np.array(sorted(set().union(*step_sets)))
#
#     # align & interpolate each run
#     aligned = pd.DataFrame(index=all_steps, columns=runs, dtype=float)
#     for run in runs:
#         col = f"{run} - {MEAN_SUFFIX}"
#         ser = df[[STEP_COL, col]].dropna(subset=[col]).set_index(STEP_COL)[col]
#         filled = (
#             ser
#             .reindex(all_steps)
#             .interpolate(method="index", limit_direction="both")
#             .bfill()
#             .ffill()
#         )
#         aligned[run] = filled
#
#     # compute mean and SEM
#     group_mean = aligned.mean(axis=1)
#     group_sem = aligned.std(axis=1, ddof=1) / np.sqrt(len(runs))
#
#     group_std = aligned.std(axis=1, ddof=1)
#
#     xs = all_steps
#     means = group_mean.values
#     stds = group_std.values
#     m100k = np.interp(100000, xs, means)
#     s100k = np.interp(100000, xs, stds) / np.sqrt(len(runs))
#
#     scores_100k_raw = []
#     for run_column in aligned.columns:
#         # Interpolate the score at 100k for this specific run
#         run_specific_scores = aligned[run_column].values
#         score_at_100k = np.interp(100000, xs, run_specific_scores)
#         scores_100k_raw.append(score_at_100k)
#
#     # Store the list of raw scores in our dictionary
#     final_scores_at_100k[group_label] = np.array(scores_100k_raw)
#     print(f"Group label {group_label}, mean at 100k: {m100k:.2f} ± {s100k:.2f}")
#
#     results4.append((group_label, all_steps, group_mean, group_sem))
#
# labels = list(final_scores_at_100k.keys())
#
# colors = plt.get_cmap("Set1")
#
# for idx, (group_label, all_steps, group_mean, group_sem) in enumerate(results4):
#     xs = all_steps
#     mu = group_mean
#     sem = group_sem
#
#     color = colors(idx)
#     linestyle = '-'
#     # # Set custom colors and styles based on the label
#     if group_label == "Vanilla":
#         linestyle = '--'
#
#     ax4.plot(xs, mu, label=group_label, color=color, linewidth=2, linestyle=linestyle)
#     ax4.fill_between(xs, mu - sem, mu + sem, color=color, alpha=0.3)
#
# # ax4.set_xlabel("Environment Step")
# ax4.set_title("MsPacman")
#
# ax4.set_xlim(right=100000)
# ax4.set_xticks([40000, 60000, 80000, 100000])
#
# ax4.tick_params(axis='both', which='major')
# ax4.legend()

# plt.tight_layout()
fig.subplots_adjust(
    left=0.08,   # move grid right
    right=0.98,  # allow more room on the right
    bottom=0.08, # move grid up
    top=0.95,    # move grid down
    wspace=0.1, # horizontal gap between subplots
    hspace=0.2   # vertical gap between subplots
)

fig.supxlabel("Env Steps", fontsize=30, y=0.02)   # default y≈0.04 → reduce it
fig.supylabel("Episode Return", fontsize=30, x=0.02)


fig.savefig("lr.pdf", bbox_inches='tight')
#plt.show()



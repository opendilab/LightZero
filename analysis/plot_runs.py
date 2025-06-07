import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

CSV_PATH = "../results/paper-pong.csv"  # your WandB-exported CSV
STEP_COL = "collector_step/total_envstep_count" # X-column

df = pd.read_csv(CSV_PATH)
mean_suffix = "evaluator_step/eval_episode_return_mean" # Y-colum
STEP_COL = "collector_step/total_envstep_count"
MEAN_SUFFIX = "evaluator_step/eval_episode_return_mean"

# 2) Read the CSV & discover run IDs
df = pd.read_csv(CSV_PATH)
mean_cols = [c for c in df.columns if c.endswith(MEAN_SUFFIX)]
run_ids   = [c.split(" - ")[0] for c in mean_cols]


run_groups = {
    "local": ["jumping-glade-330", "winter-music-341", "clean-wood-342", "playful-energy-343", "hopeful-dust-343"],
    "gaam_2" : ["wise-plant-351", "sparkling-glitter-317", "ethereal-capybara-351", "wild-terrain-352", "wild-plant-351"],
    #"gaam_6" : ["fine-waterfall-295", "stoic-firefly-294", "tough-monkey-293"],
    "vanilla_unizero" : ["summer-snowball-361", "proud-cherry-362", "peach-hill-363", "spring-paper-364", "major-paper-365"]
}

# # Boxing
# run_groups = {
#     # "snowy-dawn-383"
#     "vanilla_unizero": ["wild-salad-265", "floral-shadow-264", "rose-glade-263", "woven-glitter-384", "fearless-pine-66", "hardy-violet-200", "iconic-meadow-201"],
#     "local": ["apricot-dream-370", "robust-paper-368", "absurd-dust-369", "glorious-forest-366", "effortless-disco-366", "wise-music-544", "proud-forest-543"],
#     #"gaam2": ["apricot-universe-356", "sleek-energy-357", "glad-sky-358", "generous-aardvark-359", "neat-elevator-360"],
#     "gaam": ["balmy-sunset-382", "earthy-thunder-373", "soft-donkey-372","olive-hill-371", "treasured-bee-408", "decent-shape-407", "snowy-dawn-383"],
#     #"local" : ["apricot-dream-370", "robust-paper-368", "absurd-dust-369", "glorious-forest-366", "effortless-disco-366"]
# }

# BankHeist
# run_groups = {
#     "vanilla_unizero": ["mild-sunset-442", "vital-river-445", "glowing-oath-478", "youthful-fog-506", "sweet-silence-507"],
#     "local": ["lunar-dust-438", "wobbly-feather-456", "radiant-donkey-441", "balmy-star-481", "brisk-wildflower-486"],
#     "gaam" : ["super-terrain-447", "stoic-serenity-448", "lilac-sun-448", "hearty-microwave-508", "apricot-plasma-509"]
# }

# Ablation
# run_groups = {
#     "vanilla_unizero": ["major-paper-365", "spring-paper-364", "peach-hill-363", "proud-cherry-362", "summer-snowball-361"],
#     "$w$ = 2" : ["jumping-glade-330", "winter-music-341", "clean-wood-342", "playful-energy-343", "hopeful-dust-343"],
#     "$w$ = 6" : ["likely-wind-512", "wise-darkness-511", "zany-dew-492", "frosty-bush-502", "effortless-sea-490"],
#     "$w$ = 12": ["dandy-frog-514", "dainty-firebrand-513", "deft-water-515", "chocolate-field-516", "fluent-shape-517"]
# }

# Mu-Ablation
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
    # print results at 100k-th step
    print(f"{group_label} @ 100k: mean={m100k:.2f}, std={s100k:.2f}")

    results.append((group_label, all_steps, group_mean, group_sem))



plt.figure(figsize=(10, 6))
colors = plt.get_cmap("tab10")


for idx, (group_label, all_steps, group_mean, group_sem) in enumerate(results):
    # restrict to x > x_min
    # mask = all_steps >= 30000
    # xs   = all_steps[mask]
    # mu   = group_mean[mask]
    # sem  = group_sem[mask]
    xs = all_steps
    mu = group_mean
    sem = group_sem

    linestyle = '-'
    color = colors(idx)
    if group_label == "vanilla_unizero":
        #plt.plot(xs, mu, label=group_label, color='green', linewidth=2, linestyle='--')
        linestyle = '--'
    #     color = 'green'
    # elif group_label == "local":
    #     #plt.plot(xs, mu, label=group_label, color='green', linewidth=2, linestyle='--')
    #     color = 'm'
    # else:
    #     color = 'c'
    plt.plot(xs, mu, label=group_label, color=color, linewidth=2, linestyle = linestyle)
    plt.fill_between(xs, mu - sem, mu + sem, color=color, alpha=0.3)

plt.xlabel("Environment Step",  fontsize=14)
plt.ylabel("Episode Return",  fontsize=14)
plt.title("Pong Return: 5 Seeds",  fontsize=18) #Ablation of Initial $\mu$
plt.xlim(right=110000)
plt.xticks(fontsize=12)  # Make x-axis values bigger
plt.yticks(fontsize=12)
# plt.xlim(20000, 110000)
plt.xticks([40000, 60000, 80000, 100000])
plt.axvline(x=100000, color='gray', linestyle='--', linewidth=1, label='x = 100000')
plt.tight_layout()
# In Pong Mean: -14.53
#plt.axhline(y=-14.53, color='gray', linestyle='--', linewidth=1, label='UniZero at 100k-step')
# plt.legend()
plt.show()

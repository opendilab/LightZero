# pip install wandb matplotlib numpy
import wandb
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # or 'Agg', 'Qt5Agg', depending on your system
import matplotlib.pyplot as plt

# 1. Specify your three runs
run_paths = [
    "dani-allegue/LightZero/mj2w3dhk",
    "dani-allegue/LightZero/zp18283b"
]

api = wandb.Api()
layers = [0, 1]
heads = range(8)

# 2. Collect final span values per head from each run
all_spans = []
for run_path in run_paths:
    run = api.run(run_path)
    spans = []
    for layer in layers:
        for head in heads:
            key = f"learner_step/adaptive_span/layer_{layer}/head_{head}"
            df = run.history(keys=[key], pandas=True)
            # take last non-NaN entry
            val = df[key].dropna().values[-1]
            spans.append(val)
    all_spans.append(spans)

arr = np.array(all_spans)  # shape (n_runs, n_layers*8)

# 3. Compute mean and 95% CI
means = arr.mean(axis=0)
sems  = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
ci95  = 1.96 * sems

# 4. Plot
labels = [f"L{l}H{h}" for l in layers for h in heads]
x = np.arange(len(labels))

plt.figure(figsize=(10,5))
plt.bar(x, means, yerr=ci95, capsize=5)
plt.xticks(x, labels, rotation=45, ha="right")
plt.ylabel("Learned Span")
plt.title("Adaptive Span per Head (mean ±95% CI across runs)")
plt.tight_layout()
plt.show()

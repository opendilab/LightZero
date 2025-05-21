#!/usr/bin/env python
import wandb
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # or 'Agg', 'Qt5Agg', depending on your system
import matplotlib.pyplot as plt

# ─── CONFIGURATION ─────────────────────────────────────────────────────────────
ENTITY       = "dani-allegue"
PROJECT      = "LightZero"
RUN_ID       = "rwqf9q5y"
RUN_IDS = ["dez110gm", "4yjgxtun", "rsrwmmb1"]
# STEPS        = [0, 50_000, 100_000]
STEPS  = list(range(0, 100_001, 10_000))
LAYERS  = [0, 1]
WIDTH_FACTOR = 4  # how many σ’s to span around μ
# ────────────────────────────────────────────────────────────────────────────────
def histogram_stats(hist: dict):
    """
    Given a W&B histogram dict with keys 'bins' (edges length N+1) and
    'values' (counts length N), compute the weighted mean and std.
    """
    edges  = np.array(hist["bins"], dtype=float)
    counts = np.array(hist["values"], dtype=float)
    if len(edges) != len(counts) + 1:
        raise ValueError(f"bins length {len(edges)} vs values length {len(counts)}")
    centers = (edges[:-1] + edges[1:]) / 2.0
    total   = counts.sum()
    if total == 0:
        raise ValueError("Empty histogram (all zero counts)")
    mean = (centers * counts).sum() / total
    var  = (counts * (centers - mean) ** 2).sum() / total
    return mean, np.sqrt(var)

def fetch_distribution_params(entity, project, run_id, steps, layers):
    """
    Fetch and reduce the W&B histograms for mu & sigma into mean+std per step/layer.
    Returns params[step][layer] = {'mu': mean_mu, 'sigma': std_sigma}
    """
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")

    # metric keys
    keys = []
    for L in layers:
        keys += [
            f"learner_step/gaam_mu/layer_{L}",
            f"learner_step/gaam_sigma/layer_{L}",
        ]

    df = run.history(keys=keys, pandas=True)
    # pick whatever step‐column they used
    for idx_col in ("_step", "step", "Step"):
        if idx_col in df.columns:
            df = df.set_index(idx_col)
            break
    else:
        raise KeyError(f"No step column in {df.columns}")

    params = {}
    for step in steps:
        if step in df.index:
            row = df.loc[step]
        else:
            idxs    = np.array(df.index, dtype=float)
            nearest = idxs[np.abs(idxs - step).argmin()]
            print(f"• step {step} not found, using {int(nearest)}")
            row = df.loc[nearest]

        params[step] = {}
        for L in layers:
            mu_hist  = row[f"learner_step/gaam_mu/layer_{L}"]
            sig_hist = row[f"learner_step/gaam_sigma/layer_{L}"]

            if isinstance(mu_hist, dict) and "bins" in mu_hist and "values" in mu_hist:
                mean_mu, _       = histogram_stats(mu_hist)
                _, std_sigma    = histogram_stats(sig_hist)
            else:
                mean_mu  = float(mu_hist)
                std_sigma = float(sig_hist)

            params[step][L] = {"mu": mean_mu, "sigma": std_sigma}
            print(f"[step {step}, layer {L}]  μ={mean_mu:.4f}, σ={std_sigma:.4f}")
    print()
    return params

def aggregate_runs(run_ids: list[str]):
    """
    Fetch each run’s params, stack into arrays:
      mus[r,i,L], sigmas[r,i,L]   where r=run, i=step-index, L=layer.
    """
    all_params = [fetch_distribution_params(ENTITY, PROJECT, rid, STEPS, LAYERS) for rid in run_ids]
    n_runs = len(run_ids)
    n_steps = len(STEPS)
    mus    = np.zeros((n_runs, n_steps, len(LAYERS)))
    sigmas = np.zeros_like(mus)
    for r, params in enumerate(all_params):
        for i, step in enumerate(STEPS):
            for j, L in enumerate(LAYERS):
                mus[r, i, j]    = params[step][L]['mu']
                sigmas[r, i, j] = params[step][L]['sigma']
    return mus, sigmas

def plot_distributions(params, layers, width_factor):
    """
    Plot the Gaussian pdfs at the selected steps for each layer.
    """
    fig, axes = plt.subplots(len(layers), 1,
                             figsize=(8, 4 * len(layers)))
    if len(layers) == 1:
        axes = [axes]

    for ax, L in zip(axes, layers):
        for step in sorted(params):
            mu, sigma = params[step][L]["mu"], params[step][L]["sigma"]
            xs = np.linspace(mu - width_factor*sigma,
                             mu + width_factor*sigma, 500)
            ys = (1 / (sigma * np.sqrt(2*np.pi)) *
                  np.exp(-0.5 * ((xs - mu)/sigma)**2))
            ax.plot(xs, ys, label=f"step {step:,}")

        ax.set_title(f"Layer {L} Gaussian distributions")
        ax.set_xlabel("x")
        ax.set_ylabel("PDF")
        ax.legend()

    plt.tight_layout()
    plt.show()

def plot_parameter_evolution(params, layers):
    """
    Plot the evolution of μ and σ across the selected steps for each layer.
    """
    steps = sorted(params.keys())

    # μ evolution
    plt.figure(figsize=(6,4))
    for L in layers:
        mus = [params[s][L]["mu"] for s in steps]
        plt.plot(steps, mus, label=f"layer {L}")  # no marker
    plt.title("Evolution of μ")
    plt.xlabel("step")
    plt.ylabel("μ")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # σ evolution
    plt.figure(figsize=(6,4))
    for L in layers:
        sigs = [params[s][L]["sigma"] for s in steps]
        plt.plot(steps, sigs, label=f"layer {L}")  # no marker
    plt.title("Evolution of σ")
    plt.xlabel("step")
    plt.ylabel("σ")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_aggregated_evolution(run_ids: list[str]):
    mus, sigmas = aggregate_runs(run_ids)
    mean_mu = mus.mean(axis=0)
    std_mu = mus.std(axis=0, ddof=1)
    mean_sig = sigmas.mean(axis=0)
    std_sig = sigmas.std(axis=0, ddof=1)
    ci_mult = 1.96 / np.sqrt(len(run_ids))  # 95% CI

    # μ evolution with 95% CI
    plt.figure(figsize=(6, 4))
    for j, L in enumerate(LAYERS):
        plt.plot(STEPS, mean_mu[:, j], label=f"layer {L}")
        lower = mean_mu[:, j] - ci_mult * std_mu[:, j]
        upper = mean_mu[:, j] + ci_mult * std_mu[:, j]
        plt.fill_between(STEPS, lower, upper, alpha=0.2)
    plt.title("Aggregated Evolution of μ (95% CI)")
    plt.xlabel("step")
    plt.ylabel("μ")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # σ evolution with 95% CI
    plt.figure(figsize=(6, 4))
    for j, L in enumerate(LAYERS):
        plt.plot(STEPS, mean_sig[:, j], label=f"layer {L}")
        lower = mean_sig[:, j] - ci_mult * std_sig[:, j]
        upper = mean_sig[:, j] + ci_mult * std_sig[:, j]
        plt.fill_between(STEPS, lower, upper, alpha=0.2)
    plt.title("Aggregated Evolution of σ (95% CI)")
    plt.xlabel("step")
    plt.ylabel("σ")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    #print("Fetching GAAM parameters from W&B…")
    params = fetch_distribution_params(ENTITY, PROJECT, RUN_ID, STEPS, LAYERS)

    #print("Plotting distributions…")
    #plot_distributions(params, LAYERS, WIDTH_FACTOR)

    #print("Plotting parameter evolution…")
    #plot_parameter_evolution(params, LAYERS)

    print("Plotting aggregated parameter evolution for runs:", RUN_IDS)
    plot_aggregated_evolution(RUN_IDS)
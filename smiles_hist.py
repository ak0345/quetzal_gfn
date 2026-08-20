"""
Sample molecules from a SMILES file, score them with a reward function from
reward_fn.py, and plot the log-reward distribution.

Run against a corpus such as GEOM-Drugs, this shows how much variation an
objective has over reachable molecules. A component whose log-reward is
near-constant is a dead axis: it offers no gradient, so a guide trained against
it is flat by construction rather than under-trained.

Produces one figure with three panels:
    (1) histogram of log-rewards
    (2) histogram + KDE overlay
    (3) cumulative distribution (empirical CDF) of the histogram

The reward comes from reward_fn.build_reward_smiles(cfg), which returns
log_reward(smiles) -> float. We plot that log-reward directly.

Usage examples
--------------
# QED reward on a random 5000-molecule sample:
python smiles_hist.py --smiles reference/geom_drugs_smiles.txt --n 5000 --reward qed

# One component of the Osimertinib MPO benchmark:
python smiles_hist.py --smiles reference/geom_drugs_smiles.txt --n 2000 \
    --reward guacamol_component --benchmark osimertinib --component logP

# logP Gaussian reward with explicit target/sigma:
python smiles_hist.py --smiles mols.txt --n 1000 --reward logp \
    --target 2.5 --sigma 1.0

# Similarity to a target molecule:
python smiles_hist.py --smiles mols.txt --n 1000 --reward similarity \
    --reward-smiles "CC(=O)Oc1ccccc1C(=O)O"
"""

import os
import sys
import math
import argparse
import random

import numpy as np
import matplotlib
matplotlib.use("Agg")            # headless-safe; writes a file, no display needed
import matplotlib.pyplot as plt

# Compatibility shim: some installed guacamol versions import scipy.histogram,
# removed in scipy >= 1.9. reward_fn imports guacamol, so patch before importing.
import scipy as _scipy
if not hasattr(_scipy, "histogram"):
    _scipy.histogram = np.histogram

import reward_fn as R


# ------------------------------------------------------------------ config shim
class Cfg:
    """Plain attribute bag passed to build_reward_smiles. Only the fields the
    chosen reward kind reads need to be set; the CLI populates them."""
    pass


# ------------------------------------------------------------------ IO
def load_smiles(path):
    """Read non-empty, non-comment lines from a SMILES file. Takes the first
    whitespace-delimited token per line (handles 'SMILES<tab>id' files too)."""
    if not os.path.exists(path):
        sys.exit(f"error: SMILES file not found: {path}")
    smiles = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            smiles.append(line.split()[0])
    if not smiles:
        sys.exit(f"error: no SMILES parsed from {path}")
    return smiles


def sample_smiles(smiles, n, replace, seed):
    """Sample n SMILES. Without replacement, n is capped at the pool size."""
    rng = random.Random(seed)
    if n is None or n >= len(smiles):
        if not replace:
            if n is not None and n > len(smiles):
                print(f"  note: requested n={n} > pool={len(smiles)}; "
                      f"using all {len(smiles)} (no replacement)")
            return list(smiles)
    if replace:
        return [rng.choice(smiles) for _ in range(n)]
    return rng.sample(smiles, n)


# ------------------------------------------------------------------ scoring
def score_all(smiles, log_reward, floor):
    """Score every SMILES. Returns (all_values, finite_mask, n_floored).

    'floored' = values at (or below) the invalid floor -- parse/embed/scoring
    failures. Tracked separately so they can be shown as a spike or dropped,
    without silently distorting the KDE."""
    vals = np.empty(len(smiles), dtype=float)
    for i, smi in enumerate(smiles):
        try:
            vals[i] = log_reward(smi)
        except Exception:
            vals[i] = floor
    floored = vals <= floor + 1e-9
    n_floored = int(floored.sum())
    return vals, ~floored, n_floored


# ------------------------------------------------------------------ KDE
def gaussian_kde_curve(x, grid, bandwidth=None):
    """Simple Gaussian KDE evaluated on `grid`. Uses Scott's rule if no
    bandwidth given. Self-contained so we don't depend on scipy (which has a
    version-skew issue in some of these envs)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return np.zeros_like(grid)
    std = x.std(ddof=1)
    if std == 0:
        std = 1e-6
    if bandwidth is None:
        bandwidth = 1.06 * std * n ** (-1 / 5)   # Scott/Silverman-ish
    if bandwidth <= 0:
        bandwidth = 1e-6
    # (grid x n) kernel matrix, averaged over samples
    u = (grid[:, None] - x[None, :]) / bandwidth
    k = np.exp(-0.5 * u * u) / (bandwidth * math.sqrt(2 * math.pi))
    return k.mean(axis=1)


# ------------------------------------------------------------------ plotting
def make_plots(series, floor, args):
    """series: list of dicts, each {label, vals, finite_mask, n_floored, color}.
    Overlays all series across the three panels with SHARED bins and x-range so
    the distributions are directly comparable."""
    # what actually gets plotted per series (drop-invalid or keep floored)
    for s in series:
        s["finite"] = s["vals"][s["finite_mask"]]
        s["plotted"] = s["finite"] if args.drop_invalid else s["vals"]

    if all(len(s["plotted"]) == 0 for s in series):
        sys.exit("error: nothing to plot (all molecules invalid?)")

    # shared plotting range: use pooled PLOTTED values so both fit the same axes
    pooled_plotted = np.concatenate([s["plotted"] for s in series if len(s["plotted"])])
    pooled_finite = np.concatenate([s["finite"] for s in series if len(s["finite"])]) \
        if any(len(s["finite"]) for s in series) else np.array([floor])
    x_lo = float(pooled_plotted.min())
    x_hi = float(pooled_plotted.max())
    nbins = args.bins
    bins = np.linspace(x_lo, x_hi, nbins + 1)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    title_kind = args.reward
    if args.reward == "guacamol_component":
        title_kind += f" [{args.benchmark}:{args.component}]"
    subtitle = " | ".join(
        f"{s['label']}: n={len(s['vals'])}, {len(s['finite'])} finite, "
        f"{s['n_floored']} floored" for s in series)
    fig.suptitle(f"log-reward distribution -- {title_kind}\n{subtitle}", fontsize=10)

    # --- (1) plain histogram (counts) ---
    ax = axes[0]
    for s in series:
        if len(s["plotted"]):
            ax.hist(s["plotted"], bins=bins, color=s["color"], alpha=0.55,
                    edgecolor="white", linewidth=0.3, label=s["label"])
    ax.set_title("histogram")
    ax.set_xlabel("log reward"); ax.set_ylabel("count")
    if len(series) > 1:
        ax.legend(fontsize=8)

    # --- (2) density histogram + KDE overlay, per series ---
    ax = axes[1]
    grid = np.linspace(x_lo - 0.05 * (x_hi - x_lo + 1e-9),
                       x_hi + 0.05 * (x_hi - x_lo + 1e-9), 512)
    for s in series:
        if len(s["plotted"]):
            ax.hist(s["plotted"], bins=bins, density=True, color=s["color"],
                    alpha=0.35, edgecolor="white", linewidth=0.3)
        # KDE on finite values only (a floor spike is not a smooth density)
        if len(s["finite"]) >= 2:
            dens = gaussian_kde_curve(s["finite"], grid)
            ax.plot(grid, dens, color=s["color"], lw=2, label=f"{s['label']} KDE")
    ax.set_title("histogram + KDE")
    ax.set_xlabel("log reward"); ax.set_ylabel("density")
    ax.legend(fontsize=8)

    # --- (3) cumulative distribution, per series (shared bins) ---
    ax = axes[2]
    for s in series:
        if not len(s["plotted"]):
            continue
        counts, edges = np.histogram(s["plotted"], bins=bins)
        cdf = np.cumsum(counts) / counts.sum()
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.plot(centers, cdf, color=s["color"], lw=2, drawstyle="steps-post",
                label=s["label"])
        ax.fill_between(centers, cdf, step="post", alpha=0.12, color=s["color"])
    ax.set_ylim(0, 1.02)
    ax.set_title("cumulative distribution")
    ax.set_xlabel("log reward"); ax.set_ylabel("cumulative fraction")
    if len(series) > 1:
        ax.legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(args.out, dpi=140)
    print(f"  saved figure -> {args.out}")

    # numeric summary per series
    for s in series:
        f = s["finite"]
        if len(f):
            q = np.percentile(f, [0, 25, 50, 75, 100])
            print(f"  [{s['label']}] finite log-reward: min={q[0]:.3f} "
                  f"q25={q[1]:.3f} median={q[2]:.3f} q75={q[3]:.3f} max={q[4]:.3f} "
                  f"| mean={f.mean():.3f} std={f.std():.3f}")
        else:
            print(f"  [{s['label']}] no finite values")


# ------------------------------------------------------------------ CLI
def build_cfg_from_args(args):
    cfg = Cfg()
    cfg.reward = args.reward
    cfg.invalid_logr = args.invalid_logr
    # populate the fields each kind reads (harmless to set unused ones)
    cfg.reward_target = args.target
    cfg.reward_sigma = args.sigma
    cfg.reward_smiles = args.reward_smiles
    cfg.reward_formula = args.formula
    cfg.reward_benchmark = args.benchmark
    cfg.reward_component = args.component
    cfg.force_method = args.force_method
    return cfg


def parse_component(val):
    """Component selector is int index or a substring label; keep ints as ints."""
    if val is None:
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return val


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--smiles", required=True, help="path to smiles.txt")
    p.add_argument("--smiles2", default=None,
                   help="optional second smiles.txt to overlay for comparison")
    p.add_argument("--label", default=None,
                   help="legend label for --smiles (default: filename)")
    p.add_argument("--label2", default=None,
                   help="legend label for --smiles2 (default: filename)")
    p.add_argument("--n", type=int, default=1000,
                   help="number of molecules to sample from EACH file (default 1000)")
    p.add_argument("--reward", required=True,
                   help="reward kind: qed|logp|tpsa|isomer|similarity|"
                        "guacamol|guacamol_component|force")
    p.add_argument("--out", default="reward_hist.png", help="output image path")
    p.add_argument("--bins", type=int, default=50, help="histogram bins")
    p.add_argument("--seed", type=int, default=0, help="sampling RNG seed")
    p.add_argument("--replace", action="store_true",
                   help="sample with replacement")
    p.add_argument("--drop-invalid", action="store_true",
                   help="drop molecules at the invalid floor instead of showing "
                        "them as a spike")
    p.add_argument("--invalid-logr", type=float, default=-5.0,
                   help="floor value for invalid molecules (default -5.0)")
    # reward-specific params
    p.add_argument("--target", type=float, default=None,
                   help="reward_target (logp/tpsa mean, or similarity threshold)")
    p.add_argument("--sigma", type=float, default=None,
                   help="reward_sigma (logp/tpsa Gaussian width)")
    p.add_argument("--reward-smiles", default=None,
                   help="target SMILES (similarity) or benchmark name (guacamol)")
    p.add_argument("--formula", default=None, help="reward_formula (isomer)")
    p.add_argument("--benchmark", default=None,
                   help="guacamol_component benchmark key (e.g. osimertinib)")
    p.add_argument("--component", default=None,
                   help="guacamol_component index or label (e.g. 3 or logP)")
    p.add_argument("--force-method", default="mmff", choices=["mmff", "xtb"],
                   help="force reward backend (default mmff)")
    args = p.parse_args()
    args.component = parse_component(args.component)

    # build the reward once; both files are scored under the SAME reward
    print(f"1. building reward '{args.reward}'")
    cfg = build_cfg_from_args(args)
    log_reward = R.build_reward_smiles(cfg)

    # one entry per input file: (path, label, seed, color)
    colors = ["#4C72B0", "#DD8452"]
    files = [(args.smiles, args.label or os.path.basename(args.smiles), args.seed, colors[0])]
    if args.smiles2:
        # offset the second seed so the two draws are independent, not identical
        files.append((args.smiles2, args.label2 or os.path.basename(args.smiles2),
                      args.seed + 1, colors[1]))

    series = []
    for path, label, seed, color in files:
        print(f"2. [{label}] loading + sampling n={args.n} "
              f"(replace={args.replace}, seed={seed}) from {path}")
        pool = load_smiles(path)
        sample = sample_smiles(pool, args.n, args.replace, seed)
        vals, finite_mask, n_floored = score_all(sample, log_reward, cfg.invalid_logr)
        print(f"   {len(pool)} in pool -> {len(sample)} sampled; "
              f"{int(finite_mask.sum())} finite, {n_floored} at floor")
        series.append({"label": label, "vals": vals, "finite_mask": finite_mask,
                       "n_floored": n_floored, "color": color})

    print("3. plotting")
    make_plots(series, cfg.invalid_logr, args)
    print("Done.")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
make_fig12_mpo_components.py -- Figure 12: per-component scores of the
Osimertinib objective.

Mean score of each leaf scorer over the top-100 molecules of every fine-tuned
configuration. Three of the four components are near-saturated under all
configurations and one carries almost all of the variation. The profiles are
close to identical across configurations, including one never trained on this
objective, which indicates that the aggregate scores do not arise from any
configuration having moved a particular axis.

INPUTS
  results/oracle_gfn_mols/_results/*.json     extended.components   (stage 8)

USAGE
  python figures/make_fig12_mpo_components.py --out out/fig12.pdf
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt

import figstyle as fs


def short(key):
    """'obj.2:RdkitScoringFunction' -> 'c2 Rdkit'."""
    idx, _, rest = key.partition(":")
    i = idx.split(".")[-1]
    rest = rest.replace("ScoringFunction", "")
    return f"c{i} {rest}".strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="osim", choices=["osim", "peri"])
    ap.add_argument("--stat", default="mean",
                    help="which per-component statistic to read (default mean)")
    fs.add_arg_common(ap, "out/fig12_mpo_components.pdf")
    args = ap.parse_args()
    fs.use_paper_style()

    h = fs.load_harvest(args.bench)
    runs = {n: (v.get("extended") or {}).get("components") or {}
            for n, v in sorted(h.items()) if not n.startswith("_")}
    runs = {n: c for n, c in runs.items() if c}
    if not runs:
        fs.die("no per-component scores in the harvest",
               how="bash scripts/08_analysis.sh harvest   # needs --extended")

    comps = sorted({k for c in runs.values() for k in c})
    x = np.arange(len(comps))

    fig, ax = plt.subplots(figsize=(args.width * 0.75, 3.2))
    for name, c in runs.items():
        ys = []
        for k in comps:
            d = c.get(k) or {}
            ys.append(d.get(args.stat) if isinstance(d, dict) else d)
        fam = fs.ft_family(name)
        style = ":" if "nitrogen" in name else "-"
        ax.plot(x, ys, style, marker="o", ms=4, lw=1.1, alpha=0.8,
                color=fs.FAMILY_COLOURS.get(fam, "0.5"))

    ax.set_xticks(x)
    ax.set_xticklabels([short(k) for k in comps], rotation=20, ha="right",
                       fontsize=7)
    ax.set_ylabel(f"component score ({args.stat}, top-100)")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Per-component scores, {fs.BENCH_TITLE[args.bench]}")

    spreads = []
    for i, k in enumerate(comps):
        vals = [(c.get(k) or {}).get(args.stat) for c in runs.values()]
        vals = [v for v in vals if isinstance(v, (int, float))]
        if vals:
            spreads.append((max(vals) - min(vals), short(k), float(np.mean(vals))))
    for sp, k, mu in sorted(spreads, reverse=True):
        print(f"[fig] {k:24s} mean {mu:.3f}  spread across configurations {sp:.3f}")
    fs.save(fig, args.out, args.dpi)


if __name__ == "__main__":
    main()

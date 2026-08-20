#!/usr/bin/env python3
"""
make_fig09_singles_weights.py -- Figure 9: per-component effect sizes and
composition weights.

Left: effect size of each component guide sampled alone, under a direct policy
rather than a composition operator. The components differ, with one producing a
negative mean shift.

Right: effect size and uniqueness of the composed sampler under different weight
vectors, from equal weights to mass concentrated on a single component.
Concentrating weight changes the effect size but does not lift it out of the
band the sweep reports.

INPUTS
  results/ablations/singles-harmonic/singles_weights_report.json
                                                   (stage 7, `singles`)

USAGE
  python figures/make_fig09_singles_weights.py --out out/fig09.pdf
"""
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

import figstyle as fs

HOW = "bash scripts/07_ablations.sh singles"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default=os.path.join(
        fs.ABL_DIR, "singles-harmonic", "singles_weights_report.json"))
    fs.add_arg_common(ap, "out/fig09_singles_weights.pdf")
    args = ap.parse_args()
    fs.use_paper_style()

    rep = fs.load_json(args.report, how=HOW)
    singles = rep.get("singles") or {}
    sweep = rep.get("weight_sweep") or {}
    if not singles and not sweep:
        fs.die(f"no `singles` / `weight_sweep` block in {args.report}", HOW)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(args.width, 3.2))

    # ---- left: each guide alone ----
    if singles:
        labs = list(singles)
        shifts = [singles[l].get("mean_shift") for l in labs]
        cols = [fs.REF_COLOUR if (s is not None and s < 0)
                else fs.GUIDE_COLOURS["hidden"] for s in shifts]
        axL.bar(np.arange(len(labs)), shifts, color=cols, width=0.6)
        axL.axhline(0, color="0.4", lw=0.9)
        axL.set_xticks(np.arange(len(labs)))
        axL.set_xticklabels(labs, fontsize=7)
        axL.set_ylabel("mean log-reward shift vs prior")
        axL.set_title("Each component guide alone")
        for l, s in zip(labs, shifts):
            print(f"[fig] single {l:6s} mean_shift = {s}")

    # ---- right: composed sampler under different weight vectors ----
    if sweep:
        keys = list(sweep)
        shifts = [sweep[k].get("mean_shift") for k in keys]
        uniq = [sweep[k].get("uniqueness") for k in keys]
        x = np.arange(len(keys))
        axR.bar(x, shifts, color=fs.GUIDE_COLOURS["tempgain"], width=0.6,
                label="mean shift")
        axR.axhline(0, color="0.4", lw=0.9)
        axR.set_xticks(x)
        axR.set_xticklabels(keys, rotation=35, ha="right", fontsize=6)
        axR.set_ylabel("mean log-reward shift vs prior")
        axR.set_title("Composition weights")
        if any(v is not None for v in uniq):
            ax2 = axR.twinx()
            ax2.plot(x, uniq, "^:", ms=4, color="0.45", label="uniqueness")
            ax2.set_ylabel("uniqueness", color="0.45")
            ax2.tick_params(axis="y", colors="0.45")
            ax2.set_ylim(0, 1.05)
            ax2.grid(False)
        fin = [s for s in shifts if s is not None]
        if fin:
            print(f"[fig] weight sweep: shift ranges {min(fin):+.4f} to {max(fin):+.4f}"
                  f" across {len(keys)} weight vectors")

    fs.save(fig, args.out, args.dpi)


if __name__ == "__main__":
    main()

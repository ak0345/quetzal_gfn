#!/usr/bin/env python3
"""
make_fig08_tempgain.py -- Figures 4 and 8: the learned temperature never exceeds
one.

Left: T(h) read from trained checkpoints, binned by the prior's top-1 margin, for
each component guide. The dashed line marks T = 1, above which the mechanism
would soften the prior. The learned values sit between 0.73 and 0.80 everywhere
and are flat in the margin.

Because the forward pass in those runs applied clamp(T, min=1), a learned value
below 1 had no effect and no gradient: the effective temperature was 1 at every
state and the mechanism was inactive throughout. Runs labelled TEMPGAIN report a
gain-scaled residual guide rather than a test of prior softening.

Right: sampled-flip rate by margin for the same checkpoints, showing the decay
above a margin of 8 that all architectures share.

INPUTS
  results/ablations/tempgain/tempgain_probe.json     (stage 7, `tempgain`)

USAGE
  python figures/make_fig08_tempgain.py --out out/fig08.pdf
"""
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

import figstyle as fs

HOW = "bash scripts/07_ablations.sh tempgain"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default=os.path.join(fs.ABL_DIR, "tempgain",
                                                     "tempgain_probe.json"))
    fs.add_arg_common(ap, "out/fig08_tempgain.pdf")
    args = ap.parse_args()
    fs.use_paper_style()

    rep = fs.load_json(args.report, how=HOW)
    learned = rep.get("learned") or {}
    bins = learned.get("gap_bins")
    per = learned.get("per_guide") or {}
    if not bins or not per:
        fs.die(f"no `learned.gap_bins` / `per_guide` block in {args.report}", HOW)

    x = np.arange(len(bins))
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(args.width, 3.2))

    plotted = 0
    for lab, d in per.items():
        if "note" in d:          # a plain residual guide has no temp/gain heads
            print(f"[fig] {lab}: {d['note']}")
            continue
        t = d.get("T_by_gap")
        if t:
            axL.plot(x, t, "o-", ms=4, lw=1.3, label=lab)
            plotted += 1
            fin = [v for v in t if isinstance(v, (int, float))]
            if fin:
                print(f"[fig] {lab:10s} T in [{min(fin):.3f}, {max(fin):.3f}]"
                      f"  {'(never softens)' if max(fin) <= 1.0 else ''}")
        fr = d.get("flip_rate_by_gap")
        if fr:
            axR.plot(x, fr, "o-", ms=4, lw=1.3, label=lab)

    if not plotted:
        fs.die("no guide in this probe has temperature heads", HOW)

    axL.axhline(1.0, color=fs.REF_COLOUR, ls="--", lw=1.3)
    axL.text(x[-1], 1.0, "T = 1 ", fontsize=6.5, color=fs.REF_COLOUR,
             va="bottom", ha="right")
    axL.set_xticks(x); axL.set_xticklabels(bins, rotation=35, ha="right", fontsize=6.5)
    axL.set_xlabel("prior top-1 logit margin")
    axL.set_ylabel("learned temperature T(h)")
    axL.set_title("Learned temperature by margin")
    axL.legend(frameon=False, fontsize=6.5)

    axR.set_xticks(x); axR.set_xticklabels(bins, rotation=35, ha="right", fontsize=6.5)
    axR.set_xlabel("prior top-1 logit margin")
    axR.set_ylabel("sampled-flip rate")
    axR.set_title("Flip rate by margin")
    axR.legend(frameon=False, fontsize=6.5)

    fs.save(fig, args.out, args.dpi)


if __name__ == "__main__":
    main()

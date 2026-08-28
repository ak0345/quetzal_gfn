#!/usr/bin/env python3
"""
make_fig05_ceiling_bins.py -- Figure 5: flip rate against the prior's top-1
logit margin.

Left: coupled sampled-flip rate for each guide, binned by the prior's margin at
the decision, with grey bars giving the fraction of decisions falling in each
bin on the right-hand axis. The flip rate falls to zero above a margin of
roughly 4, while the majority of decisions lie above 8.

Right: standard deviation of the log-reward over prior samples for each leaf
scorer of the objective, with a dashed line marking the threshold below which an
axis carries no gradient. A component with zero variance over reachable
molecules is flat by construction, which is why the benchmark runs train against
the assembled objective rather than its components.

INPUTS
  results/ablations/ceiling/ceiling_report.json      (stage 7, `ceiling`)

USAGE
  python figures/make_fig05_ceiling_bins.py --out out/fig05.pdf
"""
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

import figstyle as fs

HOW = "bash scripts/07_ablations.sh ceiling"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default=os.path.join(fs.ABL_DIR, "ceiling",
                                                     "ceiling_report.json"))
    ap.add_argument("--flat_threshold", type=float, default=0.05,
                    help="std_logr below which an axis carries no gradient")
    fs.add_arg_common(ap, "out/fig05_ceiling_bins.pdf")
    args = ap.parse_args()
    fs.use_paper_style()

    rep = fs.load_json(args.report, how=HOW)
    sat = rep.get("saturation") or {}
    bins = sat.get("gap_bins")
    if not bins:
        fs.die(f"no `saturation.gap_bins` block in {args.report}", HOW)

    # Extra width and a wide gutter: the left panel carries a twin axis on its
    # right-hand side, which collides with the right panel's y-label at the
    # default spacing.
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(args.width * 1.35, 3.4),
                                   gridspec_kw={"wspace": 0.42})

    # ---- left: flip rate by margin bin, with the decision mass behind it ----
    x = np.arange(len(bins))
    dfrac = sat.get("decision_frac_by_gap")
    if dfrac:
        axB = axL.twinx()
        axB.bar(x, dfrac, color="0.85", zorder=1, width=0.75)
        axB.set_ylabel("fraction of decisions", color="0.55")
        axB.tick_params(axis="y", colors="0.55")
        axB.set_ylim(0, max(dfrac) * 1.6)
        axB.grid(False)
        axB.set_zorder(1)
        axL.set_zorder(2)
        axL.patch.set_visible(False)

    per_guide = sat.get("per_guide") or {}
    short = dict(zip(per_guide, fs.distinguishing_labels(list(per_guide))))
    for lab, d in per_guide.items():
        fr = d.get("flip_rate_by_gap")
        if not fr:
            continue
        # colour by guide family so this figure matches the rest of the paper
        guide = next((g for g in ("hidden", "tempgain", "base")
                      if f"-{g}-" in lab), None)
        axL.plot(x, fr, "o-", ms=4, lw=1.3, label=short[lab],
                 color=fs.GUIDE_COLOURS.get(guide), zorder=3)
    axL.set_xticks(x)
    axL.set_xticklabels(bins, rotation=35, ha="right", fontsize=6.5)
    axL.set_xlabel("prior top-1 logit margin")
    axL.set_ylabel("sampled-flip rate")
    axL.legend(frameon=False, fontsize=6.5)
    axL.set_title("Flip rate by prior margin")

    if dfrac:
        hi = sum(v for b, v in zip(bins, dfrac) if _lo(b) >= 8)
        print(f"[fig] {hi*100:.1f}% of decisions sit at a margin above 8")

    # ---- right: per-component reward variance over prior samples ----
    # ablate_ceiling.py nests these under `component_variance`; older reports
    # carried them at the top level, so accept either rather than silently
    # drawing an empty panel.
    src = rep.get("component_variance") or rep
    comp = {k: v for k, v in src.items()
            if isinstance(v, dict) and "std_logr_valid" in v}
    if comp:
        names = list(comp)
        stds = [comp[n].get("std_logr_valid", np.nan) for n in names]
        axR.bar(np.arange(len(names)), stds, color=fs.GUIDE_COLOURS["hidden"],
                width=0.6)
        axR.axhline(args.flat_threshold, color=fs.REF_COLOUR, ls="--", lw=1.2)
        axR.text(0, args.flat_threshold, " no gradient below this", fontsize=6,
                 color=fs.REF_COLOUR, va="bottom")
        axR.set_xticks(np.arange(len(names)))
        axR.set_xticklabels(names, rotation=25, ha="right", fontsize=6.5)
        axR.set_ylabel("std of log-reward (prior samples)")
        axR.set_title("Per-component reward variance")
        for n, s in zip(names, stds):
            flag = "  <- dead axis" if s is not None and s < args.flat_threshold else ""
            print(f"[fig] {n:14s} std_logr_valid = {s:.4f}{flag}")
    else:
        axR.text(0.5, 0.5, "no per-component variance block", ha="center",
                 transform=axR.transAxes, fontsize=8, color="0.5")
        axR.set_axis_off()

    fs.save(fig, args.out, args.dpi)


def _lo(bin_label):
    """Lower edge of a '[a,b)' bin label."""
    try:
        return float(bin_label.strip("[)").split(",")[0])
    except Exception:
        return -1.0


if __name__ == "__main__":
    main()

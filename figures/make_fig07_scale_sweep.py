#!/usr/bin/env python3
"""
make_fig07_scale_sweep.py -- Figure 7: effect size against residual scaling.

Left: the shift in the reward distribution as the trained guide residual is
multiplied by a constant factor at sampling time, together with validity. Both
rise to a maximum near 4x and then fall, with the mean shift turning negative:
scaling the residual past that point moves the sampler off the region where its
outputs remain valid faster than it gains reward.

Right: residual norm relative to the prior's logit norm, and KL(guided||prior),
for each guide. The residual is a small fraction of the prior's logit magnitude
in every case.

INPUTS
  results/ablations/guide-harmonic/ablation_report.json    (stage 7, `guide`)

USAGE
  python figures/make_fig07_scale_sweep.py --out out/fig07.pdf
"""
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

import figstyle as fs

HOW = "bash scripts/07_ablations.sh guide"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default=os.path.join(
        fs.ABL_DIR, "guide-harmonic", "ablation_report.json"))
    fs.add_arg_common(ap, "out/fig07_scale_sweep.pdf")
    args = ap.parse_args()
    fs.use_paper_style()

    rep = fs.load_json(args.report, how=HOW)
    sweep = rep.get("B_scale_sweep") or {}
    resid = rep.get("A_residual") or {}
    if not sweep:
        fs.die(f"no `B_scale_sweep` block in {args.report}", HOW)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(args.width, 3.2))

    # ---- left: effect size and validity against the scale factor ----
    scales, shift, w1, valid = [], [], [], []
    for k in sorted(sweep, key=lambda s: float(s)):
        d = sweep[k]
        scales.append(float(k))
        shift.append(d.get("mean_shift"))
        w1.append(d.get("w1") or d.get("wasserstein") or d.get("w1_distance"))
        valid.append(d.get("validity"))

    axL.axhline(0, color="0.8", lw=0.8)
    axL.plot(scales, shift, "o-", ms=4, color=fs.GUIDE_COLOURS["hidden"],
             label="mean log-reward shift")
    if any(v is not None for v in w1):
        axL.plot(scales, w1, "s-", ms=3.5, color=fs.GUIDE_COLOURS["tempgain"],
                 label="Wasserstein-1 to prior")
    axL.set_xscale("log", base=2)
    axL.set_xlabel("residual scale factor")
    axL.set_ylabel("effect size")
    axL.legend(frameon=False, fontsize=6.5, loc="upper left")

    if any(v is not None for v in valid):
        ax2 = axL.twinx()
        ax2.plot(scales, valid, "^:", ms=4, color="0.5", label="validity")
        ax2.set_ylabel("validity", color="0.5")
        ax2.tick_params(axis="y", colors="0.5")
        ax2.set_ylim(0, 1.05)
        ax2.grid(False)
    axL.set_title("Residual scaling")

    fin = [(s, m) for s, m in zip(scales, shift) if m is not None]
    if fin:
        best = max(fin, key=lambda t: t[1])
        print(f"[fig] peak mean shift {best[1]:+.4f} at scale {best[0]:g}x")
        neg = [s for s, m in fin if m < 0]
        if neg:
            print(f"[fig] mean shift turns negative at scale {min(neg):g}x")

    # ---- right: residual magnitude relative to the prior's logits ----
    labs = list(resid)
    if labs:
        ratio = [resid[l].get("residual_ratio") for l in labs]
        kl = [resid[l].get("kl") or resid[l].get("mean_kl") for l in labs]
        x = np.arange(len(labs))
        axR.bar(x - 0.2, ratio, width=0.4, label="||residual|| / ||prior logits||",
                color=fs.GUIDE_COLOURS["hidden"])
        if any(v is not None for v in kl):
            axR.bar(x + 0.2, kl, width=0.4, label="KL(guided || prior)",
                    color=fs.GUIDE_COLOURS["tempgain"])
        axR.set_xticks(x)
        axR.set_xticklabels(labs, fontsize=7)
        axR.set_yscale("log")
        axR.legend(frameon=False, fontsize=6.5)
        axR.set_title("Residual magnitude")
        for l, r in zip(labs, ratio):
            print(f"[fig] {l:8s} residual/prior-logit norm = {r}")
    else:
        axR.set_axis_off()

    fs.save(fig, args.out, args.dpi)


if __name__ == "__main__":
    main()

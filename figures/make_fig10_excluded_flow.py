#!/usr/bin/env python3
"""
make_fig10_excluded_flow.py -- Figure 10: delivery failure in the excluded
flow-route runs.

Left: flip rate by margin, identically zero in every bin. Right: flip rate by
sequence position, identically zero at every position. The corresponding rollout
diagnostics report a residual norm of exactly 0.000 at every state, against 0.37
to 8.18 for the same checkpoints evaluated on the policy route.

This is the signature of a guide that is computed but never applied, and it is
distinguishable from the bound reported in the main text precisely because
delivery is measured separately from the flip rate. These runs are excluded from
all reported results.

INPUTS
  a flip report from a flow-route run, i.e. one produced with --route flow.
  Pass it with --report; there is no default, because the excluded runs are not
  part of the standard pipeline.

USAGE
  python figures/make_fig10_excluded_flow.py \
      --report results/ablations/flip-flow-t1.0/flip_report.json \
      --out out/fig10.pdf
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt

import figstyle as fs

HOW = ("ROUTE=flow OUT_ROOT=results/ablations/flow bash scripts/07_ablations.sh flip\n"
       "    # the flow route is the excluded configuration; --route policy is the "
       "one used for reported results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True,
                    help="flip report from a --route flow run")
    ap.add_argument("--temp", default="1.0")
    ap.add_argument("--max_pos", type=int, default=16)
    fs.add_arg_common(ap, "out/fig10_excluded_flow.pdf")
    args = ap.parse_args()
    fs.use_paper_style()

    rep = fs.load_json(args.report, how=HOW)
    blk = rep.get(f"flip_temp{args.temp}") or rep
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(args.width, 3.2))

    # ---- left: by margin (the two-bin split every report carries) ----
    hi = blk.get("flip_rate_high_gap")
    lo = blk.get("flip_rate_low_gap")
    if hi is not None and lo is not None:
        axL.bar([0, 1], [lo, hi], color=fs.GUIDE_COLOURS["base"], width=0.55)
        axL.set_xticks([0, 1])
        axL.set_xticklabels(["margin <= 8", "margin > 8"])
        axL.set_ylabel("sampled-flip rate")
        axL.set_ylim(0, max(0.001, (hi or 0), (lo or 0)) * 1.4)
        axL.set_title("Flip rate by margin")
        print(f"[fig] flip rate: low-margin {lo}, high-margin {hi}")

    # ---- right: by position ----
    pr = blk.get("flip_rate_by_position") or []
    n = min(args.max_pos, len(pr))
    x = np.arange(n)
    y = [np.nan if v is None else v for v in pr[:n]]
    axR.plot(x, y, "o-", ms=4, color=fs.GUIDE_COLOURS["base"])
    axR.set_xlabel("sequence position (atom decision)")
    axR.set_ylabel("sampled-flip rate")
    axR.set_title("Flip rate by position")
    axR.set_ylim(bottom=0)

    delivered = blk.get("delivered_frac")
    fin = [v for v in y if isinstance(v, float) and np.isfinite(v)]
    allzero = fin and max(fin) == 0.0
    print(f"[fig] delivered_frac = {delivered} | flip rate identically zero at "
          f"every position: {bool(allzero)}")
    if delivered is not None and delivered > 0.5 and not allzero:
        print("[fig] NOTE: this report does not show the delivery-failure "
              "signature -- it looks like a policy-route run, not a flow-route one.")

    fig.suptitle("Excluded flow-route runs: the residual is computed but never applied",
                 fontsize=9)
    fs.save(fig, args.out, args.dpi)


if __name__ == "__main__":
    main()

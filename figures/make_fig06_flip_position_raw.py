#!/usr/bin/env python3
"""
make_fig06_flip_position_raw.py -- Figure 6: sampled-flip rate by sequence
position, individual guides.

Supports Figure 2, which pools over configurations. Each line here is one guide
evaluated on trajectories rolled by the frozen prior, so the decay can be seen
to be present in every guide rather than an artifact of averaging.

INPUTS
  results/flips-guide/flip_report_*.json          (stage 6)

USAGE
  python figures/make_fig06_flip_position_raw.py --out out/fig06_raw.pdf
  python figures/make_fig06_flip_position_raw.py --rewards osim,peri --max_pos 8
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt

import figstyle as fs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flips_root", default=fs.FLIPS_DIR)
    ap.add_argument("--temps", default="1.0,0.3",
                    help="comma-separated flip temperatures, one panel each "
                         "(default: both recorded temperatures)")
    ap.add_argument("--max_pos", type=int, default=8)
    ap.add_argument("--rewards", default=None, help="e.g. osim,peri")
    ap.add_argument("--exclude", default="",
                    help="comma-separated substrings of run names to drop")
    ap.add_argument("--annotate", action="store_true",
                    help="label the highest and lowest curves")
    fs.add_arg_common(ap, "out/fig06_flip_position_raw.pdf")
    args = ap.parse_args()
    fs.use_paper_style()

    temps = [t.strip() for t in args.temps.split(",") if t.strip()]
    rewards = args.rewards.split(",") if args.rewards else None
    exclude = tuple(x for x in args.exclude.split(",") if x)

    # one panel per temperature, sharing the y-axis: overlaying both on a single
    # axis would put ~500 curves on top of each other and hide the very spread
    # the figure exists to show
    fig, axes = plt.subplots(1, len(temps), figsize=(args.width * 0.62 * len(temps),
                                                     3.2),
                             squeeze=False, sharey=True)
    axes = axes[0]

    for ax, temp in zip(axes, temps):
        reports = fs.load_flip_reports(args.flips_root, temp)
        curves = []
        for label, (doc, blk) in sorted(reports.items()):
            if rewards and not any(f"-{r}-" in label for r in rewards):
                continue
            if any(x in label for x in exclude):
                continue
            raw = blk.get("raw") or {}
            fb, sb = raw.get("flip_by_position"), raw.get("state_by_position")
            if not fb or not sb:
                continue
            n = min(args.max_pos, len(fb), len(sb))
            f_ = np.asarray(fb[:n], float)
            s_ = np.asarray(sb[:n], float)
            # positions no molecule reached stay absent, not zero
            rate = np.where(s_ > 0, f_ / np.maximum(s_, 1), np.nan)
            guide = next((g for g in ("hidden", "tempgain", "base")
                          if f"-{g}-" in label), None)
            ax.plot(np.arange(n), rate, "-", lw=1.0, alpha=0.55,
                    color=fs.GUIDE_COLOURS.get(guide, "0.5"), zorder=2)
            curves.append((rate[0] if np.isfinite(rate[0]) else np.nan,
                           label, rate))

        if not curves:
            fs.die(f"no usable per-position curves at T={temp}")

        ax.set_xlabel("sequence position (atom decision)")
        ax.set_ylim(bottom=0)
        ax.set_title(f"T = {temp}")

        curves.sort(key=lambda t: (-t[0] if np.isfinite(t[0]) else 0))
        hi, lo = curves[0], curves[-1]
        if args.annotate:
            for v, label, rate in (hi, lo):
                ax.annotate(fs.clean_label(label), (0, rate[0]), fontsize=5.5,
                            xytext=(4, 0), textcoords="offset points",
                            va="center")
        finite0 = [c[0] for c in curves if np.isfinite(c[0]) and c[0] > 0]
        print(f"[fig] T={temp}: {len(curves)} guides | position-0 flip rate: "
              f"max {hi[0]:.3f} ({fs.clean_label(hi[1])}), "
              f"min {min(finite0):.3f}, spread {hi[0]/min(finite0):.1f}x")

    axes[0].set_ylabel("coupled sampled-flip rate")
    handles = [plt.Line2D([], [], color=c, lw=1.4)
               for c in fs.GUIDE_COLOURS.values()]
    axes[-1].legend(handles, [f"guide: {g}" for g in fs.GUIDE_COLOURS],
                    frameon=False, loc="upper right")
    fig.suptitle("Flip rate by position, individual guides", fontsize=9.5)
    fs.save(fig, args.out, args.dpi)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
make_fig02_positional.py -- Figure 2: guide influence decays along the
construction path.

Coupled sampled-flip rate against sequence position (left axis) with the prior's
mean top-1 logit margin on the same positions (right axis), pooled over
configurations. Trajectories are rolled by the frozen prior, so every
architecture is compared on an identical state distribution.

The first atom is conditioned on nothing and is genuinely contestable; each
subsequent atom narrows the choice, and the margin growth on the right axis is
what accounts for the decay on the left.

POOLING. Rates are pooled from the raw per-position counts each report carries
-- numerators and denominators summed, divided once at the end. Averaging
per-run rates would weight a run of 400 states the same as one of 40,000, and at
deep positions, where only a few long molecules survive, that difference is
large. Positions no trajectory ever reached stay absent rather than counting as
zero flips.

INPUTS
  results/flips-guide/flip_report_*.json          (stage 6)

The margin axis needs `mean_gap_by_position`, which older reports predate. If it
is absent the script plots the flip curve alone and says so; re-run stage 6 to
record it.

USAGE
  python figures/make_fig02_positional.py --out out/fig02_positional.pdf
  python figures/make_fig02_positional.py --max_pos 16 --temp 1.0
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt

import figstyle as fs


def pooled_curves(reports, max_pos, rewards=None):
    """Pool flip counts and margin sums across runs, position by position."""
    flips = np.zeros(max_pos)
    states = np.zeros(max_pos)
    gapsum = np.zeros(max_pos)
    gapstates = np.zeros(max_pos)
    used = 0
    have_gap = False

    for label, (doc, blk) in reports.items():
        if rewards and not any(f"-{r}-" in label for r in rewards):
            continue
        raw = blk.get("raw") or {}
        fb = raw.get("flip_by_position")
        sb = raw.get("state_by_position")
        if not fb or not sb:
            continue
        used += 1
        n = min(max_pos, len(fb), len(sb))
        flips[:n] += np.asarray(fb[:n], dtype=float)
        states[:n] += np.asarray(sb[:n], dtype=float)

        gs = raw.get("gap_sum_by_position")
        if gs:
            have_gap = True
            m = min(max_pos, len(gs), len(sb))
            gapsum[:m] += np.asarray(gs[:m], dtype=float)
            gapstates[:m] += np.asarray(sb[:m], dtype=float)

    rate = np.where(states > 0, flips / np.maximum(states, 1), np.nan)
    gap = (np.where(gapstates > 0, gapsum / np.maximum(gapstates, 1), np.nan)
           if have_gap else None)
    return rate, gap, states, used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flips_root", default=fs.FLIPS_DIR)
    ap.add_argument("--temp", default="1.0")
    ap.add_argument("--max_pos", type=int, default=16)
    ap.add_argument("--rewards", default=None,
                    help="comma-separated subset, e.g. osim,peri")
    fs.add_arg_common(ap, "out/fig02_positional.pdf")
    args = ap.parse_args()
    fs.use_paper_style()

    reports = fs.load_flip_reports(args.flips_root, args.temp)
    rewards = args.rewards.split(",") if args.rewards else None
    rate, gap, states, used = pooled_curves(reports, args.max_pos, rewards)

    fig, ax = plt.subplots(figsize=(args.width * 0.62, 3.2))
    x = np.arange(len(rate))
    ax.plot(x, rate, "o-", ms=4, color=fs.GUIDE_COLOURS["hidden"],
            label="sampled-flip rate", zorder=3)
    ax.set_xlabel("sequence position (atom decision)")
    ax.set_ylabel("coupled sampled-flip rate", color=fs.GUIDE_COLOURS["hidden"])
    ax.tick_params(axis="y", colors=fs.GUIDE_COLOURS["hidden"])
    ax.set_ylim(bottom=0)

    if gap is not None:
        ax2 = ax.twinx()
        ax2.plot(x, gap, "s--", ms=3.5, color="0.45", lw=1.1,
                 label="prior top-1 margin", zorder=2)
        ax2.set_ylabel("prior mean top-1 logit margin", color="0.45")
        ax2.tick_params(axis="y", colors="0.45")
        ax2.grid(False)
    else:
        print("[fig] no mean_gap_by_position in these reports -- plotting the "
              "flip curve alone. Re-run stage 6 to record the margin axis:\n"
              "    bash scripts/06_flip_diagnostics.sh")

    ax.set_title(f"Guide influence decays along the construction path "
                 f"({used} configurations, T={args.temp})")

    finite = x[np.isfinite(rate)]
    if len(finite):
        p0 = rate[0] if np.isfinite(rate[0]) else np.nan
        below = next((int(i) for i in finite if rate[i] < 0.10), None)
        print(f"[fig] pooled over {used} runs | flip rate at position 0 = {p0:.3f}"
              f" | first position below 0.10 = {below}")
        for i in range(min(6, len(rate))):
            g = f"{gap[i]:.1f}" if gap is not None and np.isfinite(gap[i]) else "n/a"
            print(f"        pos {i}: flip {rate[i]:.4f}  margin {g}  "
                  f"({int(states[i])} states)")

    fs.save(fig, args.out, args.dpi)


if __name__ == "__main__":
    main()
